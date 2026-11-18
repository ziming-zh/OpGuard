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
            out.append((f"{prefix}/{extra[op]}", rec(c, extra[op], [act(rows(op))], act([[sig()] * HEADS for _ in toks]))))
        prev = op
    if tail:
        c += 1
        out.append((f"{prefix}/{tail}", rec(c, tail, [act(rows(prev))], act([[sig()] * HEADS for _ in toks]))))
    return out


OPS = ["q_proj", "rope", "q_norm", "o_proj"]


def bad(tok_list, op, i):
    """A different value for token i at op (whole row changed)."""
    return {(op, i): Tok()}


def test_prefill_head_split_vs_merged_pivots_at_q_norm():
    toks = [Tok() for _ in range(6)]
    diff = bad(toks, "q_norm", 3)
    for o in ("o_proj",):
        diff[(o, 3)] = Tok()
    left = forward("v", 100, toks, OPS)
    right = forward("o", 200, toks, OPS, split_ops=("q_proj", "rope", "q_norm"), changed=diff)
    rep = tm.match_tokens(left, right)
    assert rep["linked_tokens"] == 6
    assert rep["pivot"]["left"]["record"] == "v/q_norm" and rep["pivot"]["left"]["kind"] == "operator"
    assert rep["pivot"]["right"]["record"] == "o/q_norm"
    assert rep["pivot"]["left"]["token"]["row"] == 3


def test_uncaptured_operator_points_at_its_consumer():
    toks = [Tok() for _ in range(6)]
    diff = {("q_norm", 2): Tok(), ("o_proj", 2): Tok()}
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
        ch = {("q_norm", 1): Tok(), ("o_proj", 1): Tok()} if s == 5 else None
        rollout += forward(f"r/dec{s}", 10 * s, [B[s], A[s]], OPS, changed=ch)
    trainer = forward("t", 500, A, OPS)
    rep = tm.match_tokens(rollout, trainer)
    assert rep["linked_tokens"] == 8                       # B's tokens stay unlinked (gaps)
    assert rep["pivot"]["left"]["record"] == "r/dec5/q_norm"
    assert rep["pivot"]["right"]["record"] == "t/q_norm" and rep["pivot"]["right"]["token"]["row"] == 5


def test_weight_load_before_any_forward_is_ignored():
    toks = [Tok() for _ in range(4)]
    w = [[sig()] * HEADS for _ in range(8)]
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
    diff = {("q_norm", 4): Tok(), ("o_proj", 4): Tok()}
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


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
