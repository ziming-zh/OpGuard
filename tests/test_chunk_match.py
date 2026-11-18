"""Chunk-join alignment on synthetic traces built with a pure-Python chunk XOR."""
import importlib.util
import random
import sys
import types
from pathlib import Path

# Load the two modules by path: the perfetto package __init__ imports torch.
_DIR = Path(__file__).resolve().parents[1] / "opguard" / "trace_export" / "perfetto"
_pkg = types.ModuleType("_pf"); _pkg.__path__ = [str(_DIR)]; sys.modules["_pf"] = _pkg
for _name in ("value_match", "chunk_match"):
    _spec = importlib.util.spec_from_file_location(f"_pf.{_name}", _DIR / f"{_name}.py")
    _mod = importlib.util.module_from_spec(_spec); sys.modules[f"_pf.{_name}"] = _mod
    _spec.loader.exec_module(_mod)
cm = sys.modules["_pf.chunk_match"]

CB = 64                      # chunk bytes = 16 words
ROW_WORDS = 64               # one row = 128 bf16 = 256 bytes = 4 chunks
rng = random.Random(7)


def rows(n):
    return [[rng.getrandbits(32) for _ in range(ROW_WORDS)] for _ in range(n)]


def flip(rs, r, w=5):
    out = [list(x) for x in rs]
    out[r][w] ^= 1 << 16
    return out


def t(rs, dtype="torch.bfloat16"):
    words = [w for row in rs for w in row]
    chunks = []
    for k in range(0, len(words), CB // 4):
        acc = 0
        for w in words[k:k + CB // 4]:
            acc ^= w
        chunks.append(acc)
    return {"shape": [len(rs), ROW_WORDS * 2], "stats": {"dtype": dtype, "xor_chunks": chunks,
                                                         "xor_chunk_bytes": CB}}


def rec(counter, fn, ins, outs):
    return {"counter": counter, "function": fn, "fname": f"/{fn}.py", "line_num": 1, "gpu_rank": 0,
            "inputs": {"args": ins}, "outputs": outs}


X, Q = rows(38), rows(38)
QN = rows(38)
QN_OLD = flip(QN, 12)
WQ, WK = rows(16), rows(16)


def prefill_case(with_weights=False):
    vllm = [("v/embed", rec(10, "embed", [], t(X))),
            ("v/qkv", rec(11, "qkv_proj", [t(X)], t(Q))),
            ("v/rms", rec(12, "rms_norm_batch_invariant", [t(Q)], t(QN)))]
    old = [("o/embed", rec(20, "embedding", [], t(X))),
           ("o/linear", rec(21, "linear", [t(X)], t(Q))),
           # q_norm itself not captured; only the trailing .float()
           ("o/float", rec(22, "float", [t(QN_OLD)], t(rows(76), "torch.float32")))]
    if with_weights:
        vllm = [("v/narrow", rec(1, "narrow", [t(WQ + WK)], t(WQ)))] + vllm
        old = [("o/cat", rec(2, "cat", [t(WQ), t(WK)], t(WQ + WK)))] + old
    return vllm, old


def test_prefill_q_norm_pivots_on_the_row():
    rep = cm.match_chunks(*prefill_case())
    left, right = rep["pivot"]["left"], rep["pivot"]["right"]
    assert left["record"] == "v/rms" and left["kind"] == "operator"
    assert left["diverged_output"]["rows"] == [12]
    assert left["diverged_output"]["counterpart"]["record"] == "o/float"
    assert right["record"] == "o/float" and right["kind"] == "uncaptured_upstream"
    assert right["diverged_input"]["rows"] == [12]
    assert right["diverged_input"]["counterpart"]["record"] == "v/rms"


def test_weight_split_versus_cat_raises_nothing():
    rep = cm.match_chunks(*prefill_case(with_weights=True))
    assert rep["pivot"]["left"]["record"] == "v/rms"
    assert rep["pivot"]["right"]["record"] == "o/float"
    assert all(c["record"] not in ("v/narrow", "o/cat")
               for side in ("left", "right") for c in rep["candidates"][side])


def test_rollout_decode_batched_against_trainer_prefill():
    T = 8
    xa, xb, qa, qb = rows(T), rows(T), rows(T), rows(T)
    rollout = []
    for s in range(T):   # each decode step: sequence A and an unrelated sequence B
        rollout.append((f"r/embed{s}", rec(100 + 2 * s, "embed", [], t([xa[s], xb[s]]))))
        rollout.append((f"r/qnorm{s}", rec(101 + 2 * s, "q_norm", [t([xa[s], xb[s]])], t([qa[s], qb[s]]))))
    trainer = [("t/embed", rec(1, "embed", [], t(xa))),
               ("t/qnorm", rec(2, "q_norm", [t(xa)], t(flip(qa, 5))))]
    rep = cm.match_chunks(rollout, trainer)
    tr = rep["pivot"]["right"]
    assert tr["record"] == "t/qnorm" and tr["diverged_output"]["rows"] == [5]
    assert tr["diverged_output"]["counterpart"]["record"] == "r/qnorm5"
    ro = rep["pivot"]["left"]
    assert ro["record"] == "r/qnorm5" and ro["diverged_output"]["rows"] == [0]   # B's row is a gap


def test_identical_runs_have_no_pivot():
    vllm, _ = prefill_case()
    rep = cm.match_chunks(vllm, vllm)
    assert rep["pivot"] == {"left": None, "right": None}


def test_nothing_shared_is_not_applicable():
    rep = cm.match_chunks([("a", rec(1, "f", [t(rows(2))], t(rows(2))))],
                          [("b", rec(1, "g", [t(rows(2))], t(rows(2))))])
    assert not rep["applicable"]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
