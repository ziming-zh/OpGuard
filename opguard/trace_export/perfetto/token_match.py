"""Cross-stack alignment by token chains (prototype).

Needs captures taken with OPGUARD_XOR_ROWS=1 on both sides: each tensor then
carries ``stats.xor_rows``, one XOR per vector of its last dimension. Capture
assumes nothing about what a tensor means.

At compare time:
1. Each side's trace is cut into forwards. A forward starts at an explicit
   ``<opguard.forward>`` marker record (``tokens``: T), or at a record that
   turns an integer tensor of N elements into a float tensor whose leading
   dimensions multiply to N while no float input has that leading size (an
   embedding). That N is the forward's token count T. Records before the first
   forward (weight loading, setup) are not compared.
2. In a forward, a tensor whose leading dimensions multiply to T has one row
   per token. Its vector XORs merge (XOR is order-free) into one fingerprint per
   token, so ``[T, heads, dim]`` and ``[T, hidden]`` give the same row.
3. Within a side a token keeps its identity through a forward: an output row
   inherits the identity of the same row of the record's token-shaped input,
   and an input row is recognised by the fingerprint of the earlier row it is.
4. Tokens are linked across sides by row fingerprints that belong to one token
   on each side; a link is kept only when each token is the other's best match.
   Order, batching, prefill/decode and call sites play no part.
5. Each linked token has a chain: its output rows in trace order, each found or
   not among the other side's rows of the same token. It breaks after its last
   found row only if both sides go on computing it (one-sided extra ops or a
   one-sided tail are not breaks). The pivot is the earliest break. If the
   breaking record's own input row already disagrees, the divergence is in
   uncaptured code just before it.

Exact fingerprint equality only; an unlinked token or unmatched row is a gap.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .value_match import _tensors, load_dir

Ident = Tuple[int, int]          # (forward id, row index at the forward's start)


def _shape(t: dict) -> Optional[List[int]]:
    s = t.get("shape")
    return s if isinstance(s, list) and all(isinstance(x, int) for x in s) else None


def _prod(xs) -> int:
    p = 1
    for x in xs:
        p *= x
    return p


def _dtype(t: dict) -> str:
    return str(t["stats"].get("dtype", ""))


def _is_int(t: dict) -> bool:
    d = _dtype(t)
    return "int" in d or d.endswith("long")


def _leading(shape: List[int], n: int) -> Optional[int]:
    """Number k of leading dims (k < ndim) whose product is n, else None."""
    p = 1
    for k in range(1, len(shape)):
        p *= shape[k - 1]
        if p == n:
            return k
        if p > n:
            return None
    return None


def _token_rows(t: dict, T: int) -> Optional[List[int]]:
    """One fingerprint per token, merged from the tensor's vector XORs."""
    st, shape = t["stats"], _shape(t)
    raw = st.get("xor_rows")
    if raw is None or shape is None:
        return None
    if not isinstance(raw, list):
        raw = [raw]
    k = _leading(shape, T)
    if k is None or len(raw) != _prod(shape[:-1]):
        return None
    per = _prod(shape[k:-1])
    rows = []
    for i in range(T):
        acc = 0
        for v in raw[i * per:(i + 1) * per]:
            acc ^= int(v) & 0xFFFFFFFF
        rows.append(acc)
    return rows


def _forward_start(rec: dict) -> Optional[int]:
    if rec.get("function") == "<opguard.forward>" and isinstance(rec.get("tokens"), int):
        return rec["tokens"]
    ins = list(_tensors(rec.get("inputs")))
    outs = list(_tensors(rec.get("outputs")))
    for it in ins:
        s = _shape(it)
        if not _is_int(it) or s is None or not s:
            continue
        n = _prod(s)
        if n <= 0:
            continue
        if any(not _is_int(x) and _shape(x) and _leading(_shape(x), n) for x in ins):
            continue
        if any(not _is_int(o) and _shape(o) and _leading(_shape(o), n) for o in outs):
            return n
    return None


class _Side:
    def __init__(self, records: Sequence[Tuple[Any, dict]]):
        self.ids = [rid for rid, _ in records]
        self.records = [rec for _, rec in records]
        # per record: list of (identity, sig) for inputs and outputs
        self.ins: List[List[Tuple[Ident, int]]] = [[] for _ in self.records]
        self.outs: List[List[Tuple[Ident, int]]] = [[] for _ in self.records]
        f, T, seen = -1, None, {}
        for r, rec in enumerate(self.records):
            n = _forward_start(rec)
            if n is not None:
                f, T, seen = f + 1, n, {}
            if T is None:
                continue
            in_ids: Optional[List[Ident]] = None
            for t in _tensors(rec.get("inputs")):
                rows = _token_rows(t, T)
                if rows is None:
                    continue
                ids = []
                for i, sig in enumerate(rows):
                    prev = seen.get(sig)
                    ident = prev if prev not in (None, "ambiguous") else (f, i)
                    ids.append(ident)
                    if sig:
                        self.ins[r].append((ident, sig))
                if in_ids is None:
                    in_ids = ids
            for t in _tensors(rec.get("outputs")):
                rows = _token_rows(t, T)
                if rows is None:
                    continue
                for i, sig in enumerate(rows):
                    ident = in_ids[i] if in_ids is not None else (f, i)
                    if sig:
                        self.outs[r].append((ident, sig))
                        seen[sig] = ident if seen.get(sig, ident) == ident else "ambiguous"
        self.sigs: Dict[Ident, set] = defaultdict(set)
        self.owners: Dict[int, set] = defaultdict(set)
        for obs in self.ins + self.outs:
            for ident, sig in obs:
                self.sigs[ident].add(sig)
                self.owners[sig].add(ident)

    def chain(self, ident: Ident) -> List[Tuple[int, int]]:
        return [(r, sig) for r, obs in enumerate(self.outs) for i, sig in obs if i == ident]

    def describe(self, r: Optional[int]) -> Optional[dict]:
        if r is None:
            return None
        rec = self.records[r]
        return {"record": self.ids[r], "function": rec.get("function"), "file": rec.get("fname"),
                "line": rec.get("line_num") or (rec.get("meta") or {}).get("lineno"),
                "counter": rec.get("counter")}


def _links(L: _Side, R: _Side) -> List[Tuple[Ident, Ident]]:
    votes: Counter = Counter()
    for sig, owners in L.owners.items():
        if len(owners) == 1 and len(R.owners.get(sig, ())) == 1:
            votes[(next(iter(owners)), next(iter(R.owners[sig])))] += 1
    best_l: Dict[Ident, Tuple[int, Ident]] = {}
    best_r: Dict[Ident, Tuple[int, Ident]] = {}
    for (a, b), n in votes.items():
        if n > best_l.get(a, (0, None))[0]:
            best_l[a] = (n, b)
        if n > best_r.get(b, (0, None))[0]:
            best_r[b] = (n, a)
    return sorted((a, b) for a, (n, b) in best_l.items() if best_r.get(b, (0, None))[1] == a)


def _break(side: _Side, ident: Ident, other: set) -> Tuple[Optional[int], Optional[int]]:
    """(last agreeing record, first record of the disagreeing tail) of a chain."""
    ch = side.chain(ident)
    last = max((k for k, (_, sig) in enumerate(ch) if sig in other), default=None)
    if last is None:
        return None, None
    tail = ch[last + 1:]
    return ch[last][0], (tail[0][0] if tail else None)


def match_tokens(left: Sequence[Tuple[Any, dict]], right: Sequence[Tuple[Any, dict]]) -> dict:
    """Token-chain alignment of two ordered record lists of one rank."""
    L, R = _Side(left), _Side(right)
    links = _links(L, R)
    breaks = []
    for a, b in links:
        l_last, l_brk = _break(L, a, R.sigs[b])
        r_last, r_brk = _break(R, b, L.sigs[a])
        if l_brk is None or r_brk is None:
            continue                 # agrees to the end, or only one side goes on
        breaks.append({"left_token": list(a), "right_token": list(b),
                       "left": l_brk, "right": r_brk, "left_last": l_last, "right_last": r_last})

    def kind(side: _Side, r: int, ident: Ident, other: set) -> str:
        own = [sig for i, sig in side.ins[r] if i == ident]
        return "operator" if own and all(s in other for s in own) else "uncaptured_upstream"

    def pivot(which: str, side: _Side, other_side: _Side):
        if not breaks:
            return None
        b = min(breaks, key=lambda x: (x[which], x["left"] + x["right"]))
        ident = tuple(b[f"{which}_token"])
        other_ident = tuple(b["right_token" if which == "left" else "left_token"])
        o = "right" if which == "left" else "left"
        return {"kind": kind(side, b[which], ident, other_side.sigs[other_ident]),
                **side.describe(b[which]),
                "token": {"forward": ident[0], "row": ident[1]},
                "last_agreeing": side.describe(b[f"{which}_last"]),
                "other_side": {"token": {"forward": other_ident[0], "row": other_ident[1]},
                               "break": other_side.describe(b[o]),
                               "last_agreeing": other_side.describe(b[f"{o}_last"])},
                "tokens_broken": len(breaks)}

    return {"method": "xor_token_chain_v1", "linked_tokens": len(links),
            "applicable": bool(links),
            "pivot": {"left": pivot("left", L, R), "right": pivot("right", R, L)} if links else None}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Cross-stack token-chain alignment (prototype)")
    ap.add_argument("run1"); ap.add_argument("run2")
    ap.add_argument("--rank", type=int, action="append")
    ap.add_argument("--out")
    args = ap.parse_args(argv)
    a, b = load_dir(args.run1), load_dir(args.run2)
    ranks = sorted(set(a) & set(b)) if not args.rank else args.rank
    report = {str(r): match_tokens(a.get(r, []), b.get(r, [])) for r in ranks}
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=1))
    for r, rep in report.items():
        print(f"rank {r}: linked tokens={rep['linked_tokens']}")
        if not rep["applicable"]:
            print("  no token linked: no row fingerprints, no forward found, or runs not bitwise-aligned")
            continue
        for side in ("left", "right"):
            p = rep["pivot"][side]
            if p:
                print(f"  pivot[{'run1' if side == 'left' else 'run2'}]: {p['kind']} {p['function']} "
                      f"{p['file']}:{p['line']} counter={p['counter']} token={p['token']} "
                      f"(tokens broken: {p['tokens_broken']})")
            else:
                print(f"  pivot[{'run1' if side == 'left' else 'run2'}]: none")
    return report


if __name__ == "__main__":
    main()
