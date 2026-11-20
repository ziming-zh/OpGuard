"""Cross-stack alignment by token chains (prototype).

Needs captures taken with OPGUARD_XOR_ROWS=1 on both sides: each tensor then
carries ``stats.xor_vectors`` (lastdim-u32le-v1; older captures: ``stats.xor_rows``),
one XOR per vector of its last dimension. Capture assumes nothing about what a
tensor means.

At compare time:
1. Each side's trace is cut into forwards at the boundaries the OpGuard harness
   gives: ``<opguard.forward>`` marker records (``tokens``: T) in the trace, or a
   list of {"counter", "tokens"} passed in (``--run1-forwards`` / ``--run2-forwards``).
   Only when it gives none, a forward starts at a record that turns an integer
   tensor of N elements into a float tensor whose leading dims multiply to N.
   Records before the first forward (weight loading, setup) are not compared.
2. In a forward, a tensor whose leading dimensions multiply to T has one row
   per token. Its vector XORs merge (XOR is order-free) into one fingerprint per
   token, so ``[T, heads, dim]`` and ``[T, hidden]`` give the same row. Tensors
   of another leading size (e.g. MoE dispatch, T*topk rows) are skipped.
3. Within a side a token keeps its identity through a forward: an output row
   inherits the identity of the same row of the record's token-shaped input,
   and an input row is recognised by the fingerprint of the earlier row it is.
   Equal fingerprints also give the data dependencies: a record whose input row
   equals an earlier output row consumed that row.
4. Tokens are linked across sides by row fingerprints that belong to one token
   on each side; a link is kept only when each token is the other's best match.
5. Data flow, not trace order (a token's work branches: Q, K, V, residual). A
   record is a candidate for a linked token when its input rows of that token
   agree with the other side and its output row does not ("operator"), or when
   a disagreeing input row has no captured producer ("uncaptured_upstream").
   It is confirmed when the rows that consume its output downstream also
   disagree, set aside (and listed) when they agree again, and listed as
   unconfirmed when nothing captured consumes it. The pivot is the earliest
   confirmed candidate; each token's earliest is listed too.

Exact fingerprint equality only; an unlinked token or unmatched row is a gap.
"""
from __future__ import annotations

import argparse
import base64
import json
import struct
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .value_match import _tensors, load_dir

Ident = Tuple[int, int]          # (forward id, row index at the forward's start)
Key = Tuple[int, str, int]       # (elements per token, dtype, merged row XOR)
MARKER = "<opguard.forward>"
CONFIRM_HOPS = 4                 # how far downstream a candidate is followed


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


def _vectors(st: dict) -> Optional[List[int]]:
    """Per-vector XORs as unsigned 32-bit values: ``xor_vectors`` (lastdim-u32le-v1,
    base64) or the legacy ``xor_rows`` (signed decimals; a scalar for one vector)."""
    v1 = st.get("xor_vectors")
    if isinstance(v1, dict) and v1.get("scheme") == "lastdim-u32le-v1" and isinstance(v1.get("data"), str):
        raw = base64.b64decode(v1["data"])
        return list(struct.unpack("<%dI" % (len(raw) // 4), raw[:len(raw) // 4 * 4]))
    raw = st.get("xor_rows")
    if raw is None:
        return None
    return [int(v) & 0xFFFFFFFF for v in (raw if isinstance(raw, list) else [raw])]


def _token_rows(t: dict, T: int):
    """Per token: (key, vectors). key = (elements per token, dtype, merged XOR), so rows
    of different widths never alias; vectors are the per-vector XORs (e.g. heads)."""
    st, shape = t["stats"], _shape(t)
    raw = _vectors(st)
    if raw is None or shape is None:
        return None
    k = _leading(shape, T)
    if k is None or len(raw) != _prod(shape[:-1]):
        return None
    per = _prod(shape[k:-1])
    width, dtype = _prod(shape[k:]), _dtype(t)
    rows = []
    for i in range(T):
        vecs = tuple(raw[i * per:(i + 1) * per])
        acc = 0
        for v in vecs:
            acc ^= v
        rows.append(((width, dtype, acc), vecs))
    return rows


def _forward_start(rec: dict) -> Optional[int]:
    if rec.get("function") == MARKER and isinstance(rec.get("tokens"), int):
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
    def __init__(self, records: Sequence[Tuple[Any, dict]], forwards: Optional[List[dict]] = None):
        self.ids = [rid for rid, _ in records]
        self.records = [rec for _, rec in records]
        # per record: list of (identity, sig) for inputs and outputs
        self.ins: List[List[Tuple[Ident, int]]] = [[] for _ in self.records]
        self.outs: List[List[Tuple[Ident, int]]] = [[] for _ in self.records]
        markers = any(rec.get("function") == MARKER for rec in self.records)
        plan = sorted(forwards or [], key=lambda x: x["counter"])
        self.forward_source = "list" if plan else "markers" if markers else "inferred"
        self.vecs: Dict[Key, Tuple[int, ...]] = {}
        f, T, seen, k = -1, None, {}, 0
        for r, rec in enumerate(self.records):
            n = None
            if plan:
                c = rec.get("counter")
                while isinstance(c, int) and k < len(plan) and c >= plan[k]["counter"]:
                    n, k = plan[k]["tokens"], k + 1
            elif markers:
                n = rec.get("tokens") if rec.get("function") == MARKER else None
            else:
                n = _forward_start(rec)
            if n is not None:
                f, T, seen = f + 1, n, {}
            if T is None or rec.get("function") == MARKER:
                continue
            in_ids: Optional[List[Ident]] = None
            for t in _tensors(rec.get("inputs")):
                rows = _token_rows(t, T)
                if rows is None:
                    continue
                ids = []
                for i, (key, vecs) in enumerate(rows):
                    prev = seen.get(key)
                    ident = prev if prev not in (None, "ambiguous") else (f, i)
                    ids.append(ident)
                    if key[2]:
                        self.ins[r].append((ident, key))
                        self.vecs.setdefault(key, vecs)
                if in_ids is None:
                    in_ids = ids
            for t in _tensors(rec.get("outputs")):
                rows = _token_rows(t, T)
                if rows is None:
                    continue
                for i, (key, vecs) in enumerate(rows):
                    ident = in_ids[i] if in_ids is not None else (f, i)
                    if key[2]:
                        self.outs[r].append((ident, key))
                        self.vecs.setdefault(key, vecs)
                        seen[key] = ident if seen.get(key, ident) == ident else "ambiguous"
        self.sigs: Dict[Ident, set] = defaultdict(set)
        self.owners: Dict[int, set] = defaultdict(set)
        self.producer: Dict[Tuple[Ident, int], int] = {}
        self.consumers: Dict[Tuple[Ident, int], List[int]] = defaultdict(list)
        for r in range(len(self.records)):
            for ident, sig in self.ins[r]:
                self.consumers[(ident, sig)].append(r)
            for ident, sig in self.outs[r]:
                self.producer.setdefault((ident, sig), r)
        for obs in self.ins + self.outs:
            for ident, sig in obs:
                self.sigs[ident].add(sig)
                self.owners[sig].add(ident)

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


def _confirm(side: _Side, r: int, ident: Ident, out_key: Key, P: set):
    """Follow the token's row downstream on this side. Value-preserving consumers
    (output == input key, e.g. views) are passed through without counting; the
    candidate is confirmed only if a value-changing consumer still disagrees."""
    frontier, seen, changed, steps = [(r, out_key, 0)], {(r, out_key)}, False, 0
    while frontier and steps < 4096:
        r0, key, hops = frontier.pop(0)
        steps += 1
        for r2 in side.consumers.get((ident, key), []):
            if r2 <= r0:
                continue
            for i2, o2 in side.outs[r2]:
                if i2 != ident:
                    continue
                if o2 in P:
                    return "set_aside", r2          # agreement returned: one-sided or fused-away step
                h2 = hops if o2 == key else hops + 1
                changed = changed or o2 != key
                if h2 <= CONFIRM_HOPS and (r2, o2) not in seen:
                    seen.add((r2, o2))
                    frontier.append((r2, o2, h2))
    return ("confirmed" if changed else "unconfirmed"), None


def _near(vecs: Optional[Tuple[int, ...]], key: Key, other: _Side, pid: Ident):
    """A row of the partner token with the same width and dtype that shares most of
    its vectors with this one and differs in a few: the same tensor, diverged."""
    if not vecs or len(vecs) < 2:
        return None
    best = None
    for k2 in other.sigs[pid]:
        if k2[:2] != key[:2] or k2 == key:
            continue
        v2 = other.vecs.get(k2)
        if not v2 or len(v2) != len(vecs):
            continue
        diff = [i for i, (a, b) in enumerate(zip(vecs, v2)) if a != b]
        if 0 < len(diff) * 2 <= len(vecs) and (best is None or len(diff) < len(best[1])):
            best = (k2, diff)
    return best


def _pair(side: _Side, other: _Side, ident: Ident, pid: Ident, in_keys, out_key: Key, bad_in) -> Optional[dict]:
    """Positive evidence from the other side that this is a divergence, not a gap."""
    mine = side.sigs[ident]
    if bad_in is None:
        # (a) the other side ran the same input row and got a different output of the same width
        for k in in_keys:
            for r2 in other.consumers.get((pid, k), []):
                outs = [o for i, o in other.outs[r2] if i == pid and o[:2] == out_key[:2]]
                if outs and all(o not in mine for o in outs):
                    item = {"by": "same_input", "other_record": other.describe(r2)}
                    nm = _near(side.vecs.get(out_key), out_key, other, pid)
                    if nm:
                        item.update(differing_vectors=nm[1][:16], vectors=len(side.vecs[out_key]))
                    return item
    # (b) the diverged row itself is held by the other side, with a few vectors differing
    probe = bad_in if bad_in is not None else out_key
    nm = _near(side.vecs.get(probe), probe, other, pid)
    if nm:
        r2 = other.producer.get((pid, nm[0]))
        if r2 is None:
            r2 = min(other.consumers.get((pid, nm[0]), [None]) or [None], key=lambda x: x if x is not None else 1 << 60)
        return {"by": "near_match", "other_record": other.describe(r2),
                "differing_vectors": nm[1][:16], "vectors": len(side.vecs[probe])}
    return None


def _analyze(side: _Side, other: _Side, partner: Dict[Ident, Ident]) -> dict:
    buckets: Dict[str, list] = {"confirmed": [], "set_aside": [], "unconfirmed": [], "unpaired": []}
    for r in range(len(side.records)):
        ins_by: Dict[Ident, List[Key]] = defaultdict(list)
        for ident, key in side.ins[r]:
            ins_by[ident].append(key)
        done = set()
        for ident, out_key in side.outs[r]:
            if ident not in partner or (ident, out_key) in done:
                continue
            done.add((ident, out_key))
            pid = partner[ident]
            P = other.sigs[pid]
            ins = ins_by.get(ident)
            if out_key in P or not ins:
                continue
            bad = [k for k in ins if k not in P]
            if not bad:
                kind, bad_in = "operator", None
            elif all(side.producer.get((ident, k), r) >= r for k in bad):
                kind, bad_in = "uncaptured_upstream", bad[0]
            else:
                continue                             # an earlier captured record parted first
            item = {"kind": kind, **side.describe(r),
                    "token": {"forward": ident[0], "row": ident[1]},
                    "other_token": {"forward": pid[0], "row": pid[1]}}
            pair = _pair(side, other, ident, pid, [k for k in ins if k in P], out_key, bad_in)
            if pair is None:
                item["status"] = "unpaired"
            else:
                item["pair"] = pair
                item["status"], where = _confirm(side, r, ident, out_key, P)
                if where is not None:
                    item["agreement_returned_at"] = side.describe(where)
            buckets[item["status"]].append((r, item))
    confirmed = buckets["confirmed"]
    per_token, firsts = [], set()
    for r, item in confirmed:
        key = (item["token"]["forward"], item["token"]["row"])
        if key not in firsts:
            firsts.add(key)
            per_token.append(item)
    out = {"pivot": confirmed[0][1] if confirmed else None,
           "tokens_diverged": len(firsts), "per_token": per_token[:64],
           "forward_source": side.forward_source}
    for name in ("set_aside", "unconfirmed", "unpaired"):
        out[name] = [i for _, i in buckets[name][:10]]
        out[f"{name}_total"] = len(buckets[name])
    return out


def match_tokens(left: Sequence[Tuple[Any, dict]], right: Sequence[Tuple[Any, dict]],
                 left_forwards: Optional[List[dict]] = None,
                 right_forwards: Optional[List[dict]] = None) -> dict:
    """Token alignment of two ordered record lists of one rank, with a data-flow pivot."""
    L, R = _Side(left, left_forwards), _Side(right, right_forwards)
    links = _links(L, R)
    lp = {a: b for a, b in links}
    rp = {b: a for a, b in links}
    rep = {"method": "xor_token_dataflow_v2", "linked_tokens": len(links), "applicable": bool(links)}
    if links:
        rep["left"], rep["right"] = _analyze(L, R, lp), _analyze(R, L, rp)
        rep["pivot"] = {"left": rep["left"]["pivot"], "right": rep["right"]["pivot"]}
    else:
        rep["pivot"] = None
    return rep


def _forwards(path: Optional[str], rank: int) -> Optional[List[dict]]:
    if not path:
        return None
    data = json.loads(Path(path).read_text())
    return data.get(str(rank), []) if isinstance(data, dict) else data


def main(argv=None):
    ap = argparse.ArgumentParser(description="Cross-stack token alignment with a data-flow pivot (prototype)")
    ap.add_argument("run1"); ap.add_argument("run2")
    ap.add_argument("--rank", type=int, action="append")
    ap.add_argument("--run1-forwards", help='JSON list of {"counter", "tokens"} (or {rank: list}) from the harness')
    ap.add_argument("--run2-forwards", help="same, for run2")
    ap.add_argument("--out")
    args = ap.parse_args(argv)
    a, b = load_dir(args.run1), load_dir(args.run2)
    ranks = sorted(set(a) & set(b)) if not args.rank else args.rank
    report = {str(r): match_tokens(a.get(r, []), b.get(r, []),
                                   _forwards(args.run1_forwards, r), _forwards(args.run2_forwards, r))
              for r in ranks}
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=1))
    for r, rep in report.items():
        print(f"rank {r}: linked tokens={rep['linked_tokens']}")
        if not rep["applicable"]:
            print("  no token linked: no row fingerprints, no forward found, or runs not bitwise-aligned")
            continue
        for side, name in (("left", "run1"), ("right", "run2")):
            s, p = rep[side], rep[side]["pivot"]
            head = (f"{p['kind']} {p['function']} {p['file']}:{p['line']} counter={p['counter']} token={p['token']}"
                    if p else "none")
            if p and p.get("pair", {}).get("differing_vectors") is not None:
                head += f" differing vectors={p['pair']['differing_vectors']}/{p['pair']['vectors']}"
            print(f"  pivot[{name}]: {head}  (forwards: {s['forward_source']}; tokens diverged: "
                  f"{s['tokens_diverged']}; set aside: {s['set_aside_total']}; unpaired: {s['unpaired_total']}; "
                  f"unconfirmed: {s['unconfirmed_total']})")
    return report


if __name__ == "__main__":
    main()
