"""Token-chain alignment on synthetic traces (vector XORs built in pure Python)."""
import importlib.util
import random
import sys
import types
from pathlib import Path

# Load by path: the perfetto package __init__ imports torch.
_DIR = Path(__file__).resolve().parents[1] / "opguard" / "trace_export" / "perfetto"
_pkg = types.ModuleType("_pf"); _pkg.__path__ = [str(_DIR)]; sys.modules["_pf"] = _pkg
for _name in ("value_match", "token_match"):
    _spec = importlib.util.spec_from_file_location(f"_pf.{_name}", _DIR / f"{_name}.py")
    _mod = importlib.util.module_from_spec(_spec); sys.modules[f"_pf.{_name}"] = _mod
    _spec.loader.exec_module(_mod)
tm = sys.modules["_pf.token_match"]

rng = random.Random(11)
H = 64            # hidden size (last dim), bf16
HEADS = 4         # a head-split layout [T, HEADS, H // HEADS] has 4 vectors per token


def sig():
    return rng.getrandbits(32)


class Tok:
    """One token's row value at successive ops: a fingerprint per op, split into head vectors."""
    def __init__(self):
        self.heads = {}

    def row(self, op):
        if op not in self.heads:
            self.heads[op] = [sig() for _ in range(HEADS)]
        return self.heads[op]


def merged(heads):
    acc = 0
    for h in heads:
        acc ^= h
    return acc


def act(rows, split=False, dtype="torch.bfloat16"):
    """A [T, H] tensor (one vector per token) or a [T, HEADS, H/HEADS] tensor."""
    if split:
        return {"shape": [len(rows), HEADS, H // HEADS],
                "stats": {"dtype": dtype, "xor_rows": [h for r in rows for h in r]}}
    return {"shape": [len(rows), H], "stats": {"dtype": dtype, "xor_rows": [merged(r) for r in rows]}}


def ids(n):
    return {"shape": [n], "stats": {"dtype": "torch.int64"}}


def rec(c, fn, ins, out):
    return {"counter": c, "function": fn, "fname": f"/{fn}.py", "line_num": 1, "gpu_rank": 0,
            "inputs": {"args": ins}, "outputs": out}


def forward(prefix, c0, toks, ops, split_ops=(), changed=None, extra=None, tail=None):
    """Records of one forward over `toks` through `ops` (each op maps row -> next row).

    changed: {(op, token index): Tok-like override} for rows that differ on this side.
    extra: {after_op: name} a one-sided op whose output is new rows.
    """
    changed = changed or {}
    out, c = [], c0
    rows = lambda op: [changed.get((op, i), t).row(op) for i, t in enumerate(toks)]
    out.append((f"{prefix}/embed", rec(c, "embed", [ids(len(toks))], act(rows("embed")))))
    prev = "embed"
    for op in ops:
        c += 1
        out.append((f"{prefix}/{op}", rec(c, op, [act(rows(prev), prev in split_ops), ids(len(toks))]
                                          if op == "rope" else [act(rows(prev), prev in split_ops)],
                                          act(rows(op), op in split_ops))))
        if extra and op in extra:
            c += 1
            out.append((f"{prefix}/{extra[op]}", rec(c, extra[op], [act(rows(op))], act([[sig() for _ in range(HEADS)] for _ in toks]))))
        prev = op
    if tail:
        c += 1
        out.append((f"{prefix}/{tail}", rec(c, tail, [act(rows(prev))], act([[sig() for _ in range(HEADS)] for _ in toks]))))
    return out


OPS = ["q_proj", "rope", "q_norm", "o_proj", "mlp"]


def bad(i, from_op="q_norm"):
    """Token i's rows differ from `from_op` on (whole rows changed, propagated downstream)."""
    return {(op, i): Tok() for op in OPS[OPS.index(from_op):]}


def test_prefill_head_split_vs_merged_pivots_at_q_norm():
    toks = [Tok() for _ in range(6)]
    diff = bad(3)
    left = forward("v", 100, toks, OPS)
    right = forward("o", 200, toks, OPS, split_ops=("q_proj", "rope", "q_norm"), changed=diff)
    rep = tm.match_tokens(left, right)
    assert rep["linked_tokens"] == 6
    assert rep["pivot"]["left"]["record"] == "v/q_norm" and rep["pivot"]["left"]["kind"] == "operator"
    assert rep["pivot"]["right"]["record"] == "o/q_norm"
    assert rep["pivot"]["left"]["token"]["row"] == 3


def test_uncaptured_operator_points_at_its_consumer():
    toks = [Tok() for _ in range(6)]
    diff = bad(2)
    left = forward("v", 100, toks, OPS)
    right = [r for r in forward("o", 200, toks, OPS, changed=diff) if r[0] != "o/q_norm"]
    rep = tm.match_tokens(left, right)
    assert rep["pivot"]["left"]["record"] == "v/q_norm" and rep["pivot"]["left"]["kind"] == "operator"
    assert rep["pivot"]["right"]["record"] == "o/o_proj"
    assert rep["pivot"]["right"]["kind"] == "uncaptured_upstream"


def test_rollout_prefill_decode_batched_vs_trainer_prefill():
    A, B = [Tok() for _ in range(8)], [Tok() for _ in range(8)]
    rollout = forward("r/pre", 1, [B[0], A[0], B[1], A[1], B[2], A[2], B[3], A[3]], OPS)
    for s in range(4, 8):   # decode: one token of B and of A per step; A's token 5 differs
        ch = bad(1) if s == 5 else None
        rollout += forward(f"r/dec{s}", 10 * s, [B[s], A[s]], OPS, changed=ch)
    trainer = forward("t", 500, A, OPS)
    rep = tm.match_tokens(rollout, trainer)
    assert rep["linked_tokens"] == 8                       # B's tokens stay unlinked (gaps)
    assert rep["pivot"]["left"]["record"] == "r/dec5/q_norm"
    assert rep["pivot"]["right"]["record"] == "t/q_norm" and rep["pivot"]["right"]["token"]["row"] == 5


def test_weight_load_before_any_forward_is_ignored():
    toks = [Tok() for _ in range(4)]
    w = [[sig() for _ in range(HEADS)] for _ in range(8)]
    load_l = [("v/narrow", rec(1, "narrow", [act(w)], act(w[:4])))]
    load_r = [("o/cat", rec(1, "cat", [act(w[:4]), act(w[4:])], act(w)))]
    rep = tm.match_tokens(load_l + forward("v", 100, toks, OPS), load_r + forward("o", 200, toks, OPS))
    assert rep["pivot"] == {"left": None, "right": None}


def test_one_sided_middle_op_and_tail_raise_nothing():
    toks = [Tok() for _ in range(5)]
    left = forward("v", 100, toks, OPS, extra={"q_proj": "float_cast"}, tail="sample")
    right = forward("o", 200, toks, OPS)
    rep = tm.match_tokens(left, right)
    assert rep["linked_tokens"] == 5
    assert rep["pivot"] == {"left": None, "right": None}


def test_repeated_token_ids_link_by_position_dependent_rows():
    toks = [Tok() for _ in range(6)]
    toks[4].heads["embed"] = toks[1].row("embed")          # same token id at positions 1 and 4
    diff = bad(4)
    rep = tm.match_tokens(forward("v", 100, toks, OPS), forward("o", 200, toks, OPS, changed=diff))
    assert rep["linked_tokens"] == 6
    assert rep["pivot"]["left"]["token"]["row"] == 4
    assert rep["pivot"]["right"]["record"] == "o/q_norm"


def test_identical_runs_have_no_pivot():
    toks = [Tok() for _ in range(5)]
    rep = tm.match_tokens(forward("v", 1, toks, OPS), forward("o", 1, toks, OPS))
    assert rep["applicable"] and rep["pivot"] == {"left": None, "right": None}


def test_nothing_shared_is_not_applicable():
    rep = tm.match_tokens(forward("v", 1, [Tok() for _ in range(3)], OPS),
                          forward("o", 1, [Tok() for _ in range(3)], OPS))
    assert not rep["applicable"] and rep["pivot"] is None


def marker(c, n):
    return {"counter": c, "function": "<opguard.forward>", "tokens": n, "gpu_rank": 0}


def test_branch_that_parts_while_another_agrees_later():
    """Q parts at q_norm; K is computed after it in trace order and still agrees."""
    toks, wrong = [Tok() for _ in range(5)], Tok()
    def side(p, q_bad):
        rows = lambda op, i: (wrong.row(op) if (q_bad and i == 2 and op in ("q_norm", "attn", "o_proj"))
                              else toks[i].row(op))
        R = lambda op: [rows(op, i) for i in range(5)]
        return [(f"{p}/embed", rec(1, "embed", [ids(5)], act(R("embed")))),
                (f"{p}/q_proj", rec(2, "q_proj", [act(R("embed"))], act(R("q_proj")))),
                (f"{p}/q_norm", rec(3, "q_norm", [act(R("q_proj"))], act(R("q_norm")))),
                (f"{p}/k_proj", rec(4, "k_proj", [act(R("embed"))], act(R("k_proj")))),
                (f"{p}/attn", rec(5, "attn", [act(R("q_norm")), act(R("k_proj"))], act(R("attn")))),
                (f"{p}/o_proj", rec(6, "o_proj", [act(R("attn"))], act(R("o_proj"))))]
    rep = tm.match_tokens(side("v", False), side("o", True))
    assert rep["pivot"]["left"]["record"] == "v/q_norm"
    assert rep["pivot"]["right"]["record"] == "o/q_norm" and rep["pivot"]["right"]["token"]["row"] == 2


def test_moe_dispatch_inside_a_marked_forward_is_not_a_new_forward():
    toks = [Tok() for _ in range(4)]
    def side(p, diff):
        recs = forward(p, 10, toks, OPS, changed=diff)
        embed = recs[0][1]; embed["inputs"] = {"args": []}           # embedding's int input not captured
        routed = [[sig() for _ in range(HEADS)] for _ in range(8)]                  # T * topk rows
        dispatch = (f"{p}/dispatch", rec(12, "moe_dispatch",
                                         [{"shape": [8], "stats": {"dtype": "torch.int64"}}, act(routed)], act(routed)))
        return [(f"{p}/fwd", marker(9, 4))] + recs[:2] + [dispatch] + recs[2:]
    rep = tm.match_tokens(side("v", None), side("o", bad(1)))
    assert rep["linked_tokens"] == 4
    assert rep["right"]["forward_source"] == "markers"
    assert rep["pivot"]["right"]["record"] == "o/q_norm"
    assert rep["pivot"]["right"]["token"] == {"forward": 0, "row": 1}


def test_one_sided_cast_whose_consumer_agrees_again_is_set_aside():
    toks = [Tok() for _ in range(4)]
    def side(p, with_cast):
        R = lambda op: [t.row(op) for t in toks]
        cast = [[sig() for _ in range(HEADS)] for _ in toks]
        recs = [(f"{p}/embed", rec(1, "embed", [ids(4)], act(R("embed")))),
                (f"{p}/q_proj", rec(2, "q_proj", [act(R("embed"))], act(R("q_proj"))))]
        src = R("q_proj")
        if with_cast:
            recs.append((f"{p}/cast", rec(3, "float", [act(src)], act(cast, dtype="torch.float32"))))
            src = cast
        recs += [(f"{p}/rope", rec(4, "rope", [act(src)], act(R("rope")))),
                 (f"{p}/o_proj", rec(5, "o_proj", [act(R("rope"))], act(R("o_proj"))))]
        return recs
    rep = tm.match_tokens(side("v", True), side("o", False))
    assert rep["pivot"] == {"left": None, "right": None}
    assert rep["left"]["set_aside_total"] == 4
    assert rep["left"]["set_aside"][0]["agreement_returned_at"]["record"] == "v/rope"


def test_forwards_list_from_the_harness():
    toks = [Tok() for _ in range(4)]
    def side(p, diff):
        recs = forward(p, 10, toks, OPS, changed=diff)
        recs[0][1]["inputs"] = {"args": []}                           # nothing to infer a forward from
        return recs
    left, right = side("v", None), side("o", bad(2))
    assert not tm.match_tokens(left, right)["applicable"]
    rep = tm.match_tokens(left, right, [{"counter": 10, "tokens": 4}], [{"counter": 10, "tokens": 4}])
    assert rep["left"]["forward_source"] == "list"
    assert rep["pivot"]["right"]["record"] == "o/q_norm"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
