"""Value-match fallback: pairing records that call-site pairing cannot place."""
import importlib.util
import json
from pathlib import Path

# Load by path: the perfetto package __init__ pulls in torch through build.py.
_PATH = Path(__file__).resolve().parents[1] / "opguard" / "trace_export" / "perfetto" / "value_match.py"
_spec = importlib.util.spec_from_file_location("opguard_value_match", _PATH)
vm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vm)


def t(shape, sig, dtype="torch.bfloat16"):
    return {"shape": shape, "stats": {"dtype": dtype, "xor_signature": sig}}


def rec(counter, fn, fname, line, inputs, outputs):
    return {"counter": counter, "function": fn, "fname": fname, "line_num": line, "gpu_rank": 0,
            "inputs": inputs, "outputs": outputs}


# Signatures from the layer 0 q_norm case (vLLM batch-invariant vs old training stack).
Q_IN = [181307420, 62324766, 2160823604, 2224394210]
W = [20152268, 2128444819, 2106262384, 4456692]
VLLM_OUT = [110858097, 2146923170, 2265745208, 2165307055]
OLD_OUT = [110858097, 2146857634, 2265745208, 2165307055]
OLD_FLOAT = [2313814016, 223608832, 3407872, 2006974464]
HIDDEN = [11, 22, 33, 44]
Q_W = [55, 66, 77, 88]


def vllm_side():
    return [
        ("vllm/qkv:0", rec(3190, "<vllm.qkv_proj>", "/work/runtime/vllm/qwen.py", 200,
                           {"args": [t([38, 4096], HIDDEN)]}, t([38, 4096], Q_IN))),
        ("vllm/rms:35", rec(3199, "<vllm.model_executor.determinism.batch_invariant.rms_norm_batch_invariant>",
                            "/work/runtime/vllm/model_executor/layers/layernorm.py", 108,
                            {"self": None, "args": ["<fn>", t([38, 32, 128], Q_IN), t([128], W), 1e-06],
                             "kwargs": {"residual": None}},
                            t([38, 32, 128], VLLM_OUT))),
    ]


def old_side():
    return [
        ("old/linear:0", rec(4300, "<torch.nn.functional.linear>", "/work/training/p3/model.py", 50,
                             {"args": [t([38, 4096], HIDDEN), t([4096, 4096], Q_W)]}, t([38, 4096], Q_IN))),
        # TE q_norm itself was not captured; only the trailing .float() at model.py:53.
        ("old/float:0", rec(4310, "<None.float>", "/work/training/p3/model.py", 53,
                            {"self": t([38, 32, 128], OLD_OUT), "args": ["<None.float>"], "kwargs": {}},
                            t([38, 32, 128], OLD_FLOAT, "torch.float32"))),
    ]


def test_q_norm_case_pivots_on_both_sides():
    report = vm.match_values(vllm_side(), old_side())
    assert report["applicable"]
    # The q input matches across different call sites and trace files.
    assert [(l["left"], l["right"]) for l in report["links"]] == [("vllm/qkv:0", "old/linear:0")]
    # vLLM: rms_norm has matched input (weight not captured on old side), unmatched output.
    left = report["pivot"]["left"]
    assert left["record"] == "vllm/rms:35" and left["kind"] == "operator"
    assert left["strength"] == "near_counterpart"
    assert left["counterpart"][0]["differing_regions"] == [1]
    assert left["counterpart"][0]["first_seen"]["record"] == "old/float:0"
    # old: .float consumes the diverged q_norm output that no captured record produced.
    right = report["pivot"]["right"]
    assert right["record"] == "old/float:0" and right["kind"] == "uncaptured_upstream"
    assert right["counterpart"][0]["producer"]["record"] == "vllm/rms:35"
    assert len(report["near_links"]) == 1


def test_identical_runs_have_no_pivot():
    report = vm.match_values(vllm_side(), vllm_side())
    assert report["pivot"] == {"left": None, "right": None}
    assert len(report["links"]) == 2


def test_no_shared_values_is_not_applicable():
    left = [("a", rec(1, "f", "a.py", 1, {"args": [t([128], [1, 2, 3, 4])]}, t([128], [5, 6, 7, 8])))]
    right = [("b", rec(1, "g", "b.py", 1, {"args": [t([128], [9, 10, 11, 12])]}, t([128], [13, 14, 15, 16])))]
    report = vm.match_values(left, right)
    assert not report["applicable"] and report["pivot"] is None


def test_trivial_tensors_are_not_evidence():
    zeros = t([4096], [0, 0, 0, 0])
    tiny = t([8], [1, 2, 3, 4])
    left = [("a", rec(1, "f", "a.py", 1, {"args": [zeros, tiny]}, t([128], [5, 6, 7, 8])))]
    right = [("b", rec(1, "g", "b.py", 1, {"args": [zeros, tiny]}, t([128], [5, 6, 7, 9])))]
    report = vm.match_values(left, right)
    assert report["matched_values"] == 0


def test_views_are_one_value_and_crossing_links_are_dropped():
    a, b = [1, 2, 3, 4], [5, 6, 7, 8]
    left = [("l0", rec(1, "f", "a.py", 1, {}, t([64, 2], a))),
            ("l1", rec(2, "view", "a.py", 2, {"self": t([64, 2], a)}, t([128], a))),
            ("l2", rec(3, "g", "a.py", 3, {}, t([128], b)))]
    right = [("r0", rec(1, "g", "b.py", 9, {}, t([128], b))),
             ("r1", rec(2, "f", "b.py", 8, {}, t([2, 64], a)))]
    report = vm.match_values(left, right)
    assert report["matched_values"] == 2
    assert len(report["links"]) == 1 and report["crossing_links_dropped"] == 1


def test_cli_reads_trace_dirs_across_files(tmp_path):
    for name, side in (("vllm", vllm_side()), ("old", old_side())):
        for rid, record in side:
            path = tmp_path / name / "gpu_000" / "ops" / (rid.split(":")[0].replace("/", "_") + ".jsonl")
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a") as f:
                f.write(json.dumps(record) + "\n")
    report = vm.main([str(tmp_path / "vllm"), str(tmp_path / "old"), "--out", str(tmp_path / "r.json")])
    assert report["0"]["pivot"]["left"]["function"].endswith("rms_norm_batch_invariant>")
    assert report["0"]["pivot"]["right"]["function"] == "<None.float>"
