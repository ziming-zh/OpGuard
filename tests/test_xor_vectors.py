"""Per-vector fingerprints in the lastdim-u32le-v1 encoding (docs/RLAlignment.md).

Pure Python: the format, the merge rule, and both readers (token_match and the
harness) on the v1 and the legacy encoding. With torch installed it also checks
what capture writes, on CPU and on CUDA when a device is visible, including a
small workload captured through the tracer and its background writer.

    python3 tests/test_xor_vectors.py
"""
import base64
import importlib.util
import json
import os
import random
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "harness")]
from opguard_harness.fingerprints import vector_fingerprints

try:
    import torch
except ImportError:
    torch = None


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# Loaded by path: the perfetto package __init__ imports torch. test_token_match
# supplies the synthetic traces and registers token_match as _pf.token_match.
ttm = _load("_ttm", ROOT / "tests" / "test_token_match.py")
tm = ttm.tm
_PF = ROOT / "opguard" / "trace_export" / "perfetto"
sys.modules["_pf"].config = _load("_pf.config", _PF / "config.py")
export = _load("_pf.utils", _PF / "utils.py")
rng = random.Random(21)


def ref_vectors(raw, width):
    """lastdim-u32le-v1 from bytes: the XOR of each vector's little-endian u32 words."""
    out = []
    for k in range(0, len(raw), width):
        acc = 0
        for (word,) in struct.iter_unpack("<I", raw[k:k + width]):
            acc ^= word
        out.append(acc)
    return out


def ref_flat(raw, regions=1):
    """flat-u32-le-v1 from bytes (docs/AgentTools.md)."""
    words = [w for (w,) in struct.iter_unpack("<I", raw + bytes(-len(raw) % 4))]
    base, rem = divmod(len(words), regions)
    out, start = [], 0
    for r in range(regions):
        out.append(xor(words[start:start + base + (r < rem)]))
        start += base + (r < rem)
    return out


def xor(values):
    acc = 0
    for v in values:
        acc ^= v
    return acc


def v1(values, width):
    data = base64.b64encode(struct.pack("<%dI" % len(values), *values)).decode("ascii")
    return {"scheme": "lastdim-u32le-v1", "vector_bytes": width, "count": len(values),
            "encoding": "base64-u32le", "data": data}


def legacy(values):
    """What capture wrote before v1: signed int32 decimals, a float scalar for one vector."""
    signed = [v - (1 << 32) if v >> 31 else v for v in values]
    return float(signed[0]) if len(signed) == 1 else signed


def test_merge_rule_and_two_byte_packing():
    raw = bytes(rng.getrandbits(8) for _ in range(6 * 4 * 8 * 2))            # [6, 4, 8] of a 2-byte dtype
    heads = ref_vectors(raw, 16)
    assert len(heads) == 24 and xor(heads) == ref_flat(raw)[0]               # merges to the whole tensor
    assert [xor(heads[6 * r:6 * r + 6]) for r in range(4)] == ref_flat(raw, 4)   # to regions of 6 vectors
    assert [xor(heads[4 * t:4 * t + 4]) for t in range(6)] == ref_vectors(raw, 64)   # to [6, 32] token rows
    elems = struct.unpack("<8H", raw[:16])                                   # element 2i low half, 2i+1 high
    assert heads[0] == xor(elems[2 * i] | elems[2 * i + 1] << 16 for i in range(4))


def test_harness_reader_accepts_v1_and_legacy():
    values = [0x80000001, 0, 0xFFFFFFFF, 7, 0x7FFFFFFF, 0xDEADBEEF]
    new = vector_fingerprints({"shape": [2, 3, 8], "dtype": "torch.float32", "xor_vectors": v1(values, 32)})
    old = vector_fingerprints({"shape": [2, 3, 8], "xor_rows": legacy(values), "xor_row_bytes": 32})
    assert new["status"] == old["status"] == "available"
    assert list(new["values"]) == list(old["values"]) == values              # unsigned; zero stays a value
    assert (new["vector_bytes"], new["count"]) == (old["vector_bytes"], old["count"]) == (32, 6)
    assert legacy(values)[0] < 0 and isinstance(v1(values, 32)["data"], str)
    one = [0xFFFFFFFB]
    assert legacy(one) == -5.0 and isinstance(legacy(one), float)            # the legacy float scalar
    for tensor in ({"shape": [8], "xor_rows": legacy(one), "xor_row_bytes": 32},
                   {"shape": [8], "xor_vectors": v1(one, 32)}, {"shape": [], "xor_vectors": v1(one, 4)}):
        assert list(vector_fingerprints(tensor)["values"]) == one


def test_harness_reader_rejects_what_is_not_v1():
    good = v1([1, 2, 3], 16)
    for bad in (dict(good, data=[1, 2, 3]), dict(good, data=1.0), dict(good, count=4), dict(good, scheme="rows"),
                dict(good, encoding="decimal"), dict(good, vector_bytes=6), dict(good, data=good["data"][:-2] + "!!")):
        assert vector_fingerprints({"shape": [3, 4], "xor_vectors": bad})["status"] == "unavailable", bad
    assert vector_fingerprints({"shape": [3, 4], "xor_vectors": good})["status"] == "available"
    assert vector_fingerprints({"shape": [4, 4], "xor_vectors": good})["status"] == "unavailable"   # 3 vectors, 4 rows
    assert vector_fingerprints({"shape": [3, 4], "xor_rows": [1, 2.5, 3], "xor_row_bytes": 16})["status"] == "unavailable"
    assert vector_fingerprints({"shape": [3, 6], "xor_vectors_skipped": "vector_bytes_not_multiple_of_4"}) == {
        "status": "skipped", "reason": "vector_bytes_not_multiple_of_4"}
    assert vector_fingerprints({"shape": [3, 4]})["reason"] == "no_vector_fingerprints_recorded"


def recode(node, encode):
    """A copy of trace records with every tensor's vector fingerprints re-encoded."""
    if isinstance(node, dict):
        out = {k: recode(v, encode) for k, v in node.items()}
        if isinstance(node.get("stats"), dict) and "xor_rows" in node["stats"]:
            stats = dict(node["stats"])
            rows = stats.pop("xor_rows")
            out["stats"] = {**stats, **encode(rows, node["shape"][-1] * 2)}
        return out
    if isinstance(node, (list, tuple)):
        return type(node)(recode(v, encode) for v in node)
    return node


def test_token_match_reads_both_encodings():
    toks = [ttm.Tok() for _ in range(6)]
    left = ttm.forward("v", 100, toks, ttm.OPS)
    right = ttm.forward("o", 200, toks, ttm.OPS, split_ops=("q_proj", "rope", "q_norm"), changed=ttm.bad(toks, 3))
    as_v1 = lambda rows, width: {"xor_vectors": v1(rows, width)}
    as_legacy = lambda rows, width: {"xor_rows": legacy(rows), "xor_row_bytes": width}
    reports = [tm.match_tokens(recode(left, enc), recode(right, enc)) for enc in (as_v1, as_legacy)]
    reports.append(tm.match_tokens(recode(left, as_v1), recode(right, as_legacy)))   # an old and a new capture
    assert reports[0] == reports[1] == reports[2] == tm.match_tokens(left, right)
    assert reports[0]["linked_tokens"] == 6 and reports[0]["pivot"]["right"]["record"] == "o/q_norm"
    assert "xor_rows" not in json.dumps(recode(left, as_v1)) and "xor_vectors" not in json.dumps(recode(left, as_legacy))
    row = [((8, "torch.float32", 0xFFFFFFFB), (0xFFFFFFFB,))]
    for stats in ({"xor_rows": -5.0}, {"xor_vectors": v1([0xFFFFFFFB], 32)}):
        assert tm._token_rows({"shape": [1, 8], "stats": dict(stats, dtype="torch.float32")}, 1) == row


def test_export_carries_vectors_only_when_recorded():
    base = {"dtype": "torch.bfloat16", "device": "cuda:0", "numel": 24, "xor_signature": [1, 2, 3, 4],
            "xor_sig_mode": "bytes", "xor_sig_layout": {"scheme": "flat-u32-le-v1"}}
    values = [5, 0, 0x90000000]
    record = {"inputs": {"args": [{"shape": [3, 8], "stats": dict(base, xor_vectors=v1(values, 16))},
                                  {"shape": [3, 8], "stats": dict(base, xor_rows=legacy(values), xor_row_bytes=16)}],
                         "kwargs": {"w": {"shape": [3, 3], "stats": dict(base, xor_vectors_skipped="vector_bytes_not_multiple_of_4")}}},
              "outputs": {"shape": [3, 8], "stats": base}}
    io = export.io_summary_with_xor_values(record)
    new, old, skipped = io["inputs"]
    assert new["xor_vectors"] == v1(values, 16) and "xor_rows" not in new
    assert list(vector_fingerprints(new)["values"]) == list(vector_fingerprints(old)["values"]) == values
    assert vector_fingerprints(skipped)["status"] == "skipped"
    assert set(io["outputs"][0]) == {"shape", "dtype", "device", "numel", "xor_signature", "xor_sig_mode", "xor_sig_layout"}


# ---------------------------------------------------------------- with torch
def _devices():
    return ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _summary(t, on=True):
    from opguard.tracer import tensor_dump as td
    old, td.OPGUARD_XOR_ROWS = td.OPGUARD_XOR_ROWS, int(on)
    try:
        return td.finalize_summary(td.summarize_tensor(t, ignore_cpu=False))
    finally:
        td.OPGUARD_XOR_ROWS = old


def _raw(t):
    return bytes(t.detach().contiguous().reshape(-1).view(torch.uint8).cpu().tolist())


def test_capture_writes_v1_for_every_dtype():
    if torch is None:
        return "skip"
    from opguard.tracer import xor_signature as xs
    for dev in _devices():
        gen = torch.Generator().manual_seed(3)
        rand = lambda *shape: (torch.randn(shape, generator=gen) * 37).to(dev)
        cases = [rand(5, 3, 8).to(torch.bfloat16), rand(4, 6).to(torch.float16), rand(7, 5), rand(3, 3).double(),
                 (rand(2, 4) * 1e6).long(), rand(5, 8).abs().to(torch.uint8), rand(3, 4) > 0,
                 rand(8), rand(), rand(8, 6)[::2]]                       # one vector; 0-dim; strided rows
        for t in cases:
            stats = _summary(t)["stats"]
            width = (t.shape[-1] if t.dim() else 1) * t.element_size()
            got = vector_fingerprints({"shape": list(t.shape), **stats})
            assert got["status"] == "available", (dev, t.dtype, t.shape, got)
            values = list(got["values"])
            assert stats["xor_vectors"] == v1(values, width), (dev, t.dtype)   # the v1 object; data is base64 text
            assert values == ref_vectors(_raw(t), width), (dev, t.dtype)       # a function of the bytes alone
            whole = xs.xor_signature(t, regional=1, by="words").cpu().tolist()[0]
            assert xor(values) == ref_flat(_raw(t))[0] == whole, (dev, t.dtype)    # merge rule
            off = _summary(t, on=False)["stats"]                                # option off: nothing is added
            assert off == {k: v for k, v in stats.items() if k != "xor_vectors"} and "xor_rows" not in stats
            assert set(off) == {"dtype", "device", "numel", "xor_signature", "xor_sig_mode", "xor_sig_layout"}
        skipped = {"vector_bytes_not_multiple_of_4": [rand(5, 6).abs().to(torch.uint8), rand(4, 3).to(torch.bfloat16),
                                                      rand().to(torch.float16)],
                   "empty": [rand(0, 4)], "non_contiguous": [rand(4, 6).t()],
                   "meta": [torch.empty(4, 4, device="meta")]}
        for reason, tensors in skipped.items():
            for t in tensors:
                stats = _summary(t)["stats"]
                assert stats.get("xor_vectors_skipped") == reason and "xor_vectors" not in stats, (dev, reason, stats)
                assert "xor_vectors_skipped" not in _summary(t, on=False)["stats"]


def test_native_kernel_equals_fallback_and_cpu():
    if torch is None or not torch.cuda.is_available():
        return "skip"
    from opguard.tracer import xor_signature as xs
    if not hasattr(xs._xor_reduce_cuda_ext, "xor_reduce_segments_cuda"):
        return "skip"
    torch.manual_seed(5)
    for shape, dtype in [((38, 32, 128), torch.bfloat16), ((64, 4096), torch.bfloat16), ((1000, 2), torch.float32),
                         ((33, 7), torch.float64), ((9, 5), torch.int64), ((3, 1021, 4), torch.uint8),
                         ((), torch.float32)]:
        t = (torch.randn(shape, device="cuda") * 100).abs().to(dtype)
        native, fallback, cpu = (xs.encode_xor_vectors(xs.xor_vector_stats(x, native=n)["xor_vectors"])
                                 for x, n in ((t, True), (t, False), (t.cpu(), True)))
        assert native == fallback == cpu, (shape, dtype)
        assert native["count"] == (t.numel() // t.shape[-1] if t.dim() else 1)


WORKLOAD = '''
import logging, sys, torch
logging.basicConfig(level=logging.ERROR)
from opguard.wrapped_ops import make_wrapper
torch.manual_seed(0)
op = make_wrapper(tag_prefix="/workload/model.py")
x = torch.randn(6, 4, 8, device=sys.argv[1]).to(torch.bfloat16)
y = op(11)(torch.add, x, x)                 # [6, 4, 8] bfloat16: 24 vectors of 16 bytes
z = op(12)(torch.reshape, y, (6, 32))       # the same bytes as 6 rows of 64 bytes
s = op(13)(torch.sum, z.float())            # 0-dim float32: one vector
'''


def _capture(dev, option):
    """Run the workload through the instrumented call path; records by source line."""
    with tempfile.TemporaryDirectory() as tmp:
        env = dict(os.environ, PYTHONPATH=str(ROOT), OPGUARD_XOR_ROWS=option, OPGUARD_EXEC_OUTPUT_DIR=tmp,
                   OPGUARD_AUTOSTART="0", OPGUARD_RUNTIME_CALLSTACK="0", OPGUARD_PY_AGG="0")
        subprocess.run([sys.executable, "-c", WORKLOAD, dev], env=env, cwd=tmp, check=True, timeout=600)
        records = [json.loads(line) for path in sorted(Path(tmp).rglob("*.jsonl"))
                   for line in path.read_text().splitlines()]
    return {record["line_num"]: record for record in records}


def _strip(node):
    if isinstance(node, dict):
        return {k: _strip(v) for k, v in node.items() if k not in ("xor_vectors", "xor_vectors_skipped")}
    return [_strip(v) for v in node] if isinstance(node, list) else node


def test_workload_captured_through_the_tracer():
    if torch is None:
        return "skip"
    for dev in _devices():
        on, off = _capture(dev, "1"), _capture(dev, "0")
        assert sorted(on) == sorted(off) == [11, 12, 13], (dev, sorted(on), sorted(off))
        out = {n: on[n]["outputs"] for n in on}
        heads, rows, one = (list(vector_fingerprints({"shape": out[n]["shape"], **out[n]["stats"]})["values"])
                            for n in (11, 12, 13))
        assert out[11]["stats"]["xor_vectors"] == v1(heads, 16) and len(heads) == 24, dev
        assert out[12]["stats"]["xor_vectors"] == v1(rows, 64) and out[13]["stats"]["xor_vectors"] == v1(one, 4)
        assert on[12]["inputs"]["args"][1]["stats"]["xor_vectors"] == v1(heads, 16)      # inputs carry them too
        assert [xor(heads[4 * t:4 * t + 4]) for t in range(6)] == rows                   # vectors merge to token rows
        assert [xor(heads[6 * r:6 * r + 6]) for r in range(4)] == out[11]["stats"]["xor_signature"]   # and to regions
        assert one == [xor(out[13]["stats"]["xor_signature"])]
        for n in on:                                                                     # option off adds nothing
            assert off[n]["inputs"] == _strip(on[n]["inputs"]) and off[n]["outputs"] == _strip(on[n]["outputs"])
            assert "xor_vectors" not in json.dumps(off[n]) and "xor_rows" not in json.dumps(on[n])


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            print("SKIP (needs torch or CUDA)" if fn() == "skip" else "PASS", name)
