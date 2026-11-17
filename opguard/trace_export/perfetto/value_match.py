"""Value-match fallback for records that call-site pairing leaves unplaced.

Two implementations of the same model (e.g. vLLM and a training stack) often
record the same tensor at different call sites, in different trace files, at
different capture granularity. Call-site pairing then places nothing
(``no_corresponding_trace_file``). This pass ignores call sites and works on the
XOR fingerprints already in the trace:

* A non-trivial tensor whose fingerprint appears on both sides is taken to be
  the same value. Its producer on each side is the earliest record where it is
  an output; producers are linked in an order-preserving way.
* A value that diverged usually keeps most of its regional fingerprints. Two
  unmatched values with the same dtype, element count and at least half of
  their regions equal are reported as a *near* link: the same tensor, after the
  runs parted, with the differing regions named.
* The pivot on each side is the earliest record that either
  (operator) has matched tensor inputs (any other inputs have no captured
  producer, e.g. weights) and an unmatched output, or
  (uncaptured_upstream) consumes an unmatched value that no captured record on
  its side produced and that has a near link: the divergence happened in code
  that was not captured, just before this record.

Equal fingerprints are evidence of the same value; unequal ones are not
evidence of non-correspondence. Nothing here changes call-site pairs or the
diff status of any record. No torch import: usable directly on trace dirs.
"""
from __future__ import annotations

import argparse
import bisect
import gzip
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

MIN_NUMEL = 64
REMATCH_HOPS = 4  # how far along data flow a diverged value is followed

ValueKey = Tuple[str, int, Tuple[int, ...]]


# ----------------------------------------------------------------- tensors
def _is_tensor(obj: Any) -> bool:
    return isinstance(obj, dict) and "shape" in obj and isinstance(obj.get("stats"), dict)


def _tensors(obj: Any) -> Iterable[dict]:
    if _is_tensor(obj):
        yield obj
    elif isinstance(obj, dict):
        for value in obj.values():
            yield from _tensors(value)
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            yield from _tensors(value)


def _numel(tensor: dict) -> Optional[int]:
    numel = tensor["stats"].get("numel")
    if isinstance(numel, int):
        return numel
    shape = tensor.get("shape")
    if not isinstance(shape, list) or not all(isinstance(s, int) for s in shape):
        return None
    total = 1
    for size in shape:
        total *= size
    return total


def value_key(tensor: dict) -> Optional[ValueKey]:
    """(dtype, numel, regional signature), or None for trivial tensors.

    Shape is left out on purpose: views and reshapes are the same value.
    """
    stats = tensor["stats"]
    sig = stats.get("xor_signature")
    if isinstance(sig, int):
        sig = [sig]
    if not isinstance(sig, list) or not sig or not all(isinstance(s, int) for s in sig):
        return None
    numel = _numel(tensor)
    if numel is None or numel < MIN_NUMEL or not any(sig):
        return None
    return (str(stats.get("dtype")), numel, tuple(sig))


def _keys(obj: Any) -> List[ValueKey]:
    seen, out = set(), []
    for tensor in _tensors(obj):
        key = value_key(tensor)
        if key is not None and key not in seen:
            seen.add(key)
            out.append(key)
    return out


# ----------------------------------------------------------------- matching
class _Side:
    def __init__(self, records: Sequence[Tuple[Any, dict]]):
        self.ids = [rid for rid, _ in records]
        self.records = [rec for _, rec in records]
        self.ins = [_keys(rec.get("inputs")) for rec in self.records]
        self.outs = [_keys(rec.get("outputs")) for rec in self.records]
        self.producer: Dict[ValueKey, int] = {}
        self.first_seen: Dict[ValueKey, int] = {}
        self.consumers: Dict[ValueKey, List[int]] = defaultdict(list)
        for i, (ins, outs) in enumerate(zip(self.ins, self.outs)):
            for key in ins:
                self.consumers[key].append(i)
            for key in outs:
                self.producer.setdefault(key, i)
            for key in ins + outs:
                self.first_seen.setdefault(key, i)

    def values(self):
        return self.first_seen.keys()


def _monotone(pairs: List[Tuple[int, int, ValueKey]]):
    """Longest order-preserving subset of (left, right) pairs; the rest cross."""
    pairs = sorted(pairs, key=lambda p: (p[0], p[1]))
    tails, tail_idx, prev = [], [], [-1] * len(pairs)
    for n, (_, j, _) in enumerate(pairs):
        pos = bisect.bisect_left(tails, j)
        if pos == len(tails):
            tails.append(j); tail_idx.append(n)
        else:
            tails[pos] = j; tail_idx[pos] = n
        prev[n] = tail_idx[pos - 1] if pos else -1
    keep, n = set(), tail_idx[-1] if tail_idx else -1
    while n >= 0:
        keep.add(n); n = prev[n]
    return ([p for n, p in enumerate(pairs) if n in keep],
            [p for n, p in enumerate(pairs) if n not in keep])


def _near_links(left: _Side, right: _Side, matched: set) -> Dict[str, Dict[ValueKey, Tuple[ValueKey, List[int]]]]:
    """Unmatched values with same dtype/numel and >= half of regions equal, unique both ways."""
    def index(side):
        idx = defaultdict(list)
        for key in side.values():
            if key not in matched and len(key[2]) > 1:
                idx[(key[0], key[1], len(key[2]))].append(key)
        return idx

    def best(key, candidates):
        regions = len(key[2])
        scored = [(sum(a == b for a, b in zip(key[2], c[2])), c) for c in candidates]
        scored = [(s, c) for s, c in scored if 2 * s >= regions and s < regions]
        if not scored:
            return None
        top = max(s for s, _ in scored)
        winners = [c for s, c in scored if s == top]
        return winners[0] if len(winners) == 1 else None

    li, ri = index(left), index(right)
    l2r, r2l = {}, {}
    for group, lkeys in li.items():
        rkeys = ri.get(group, [])
        for lk in lkeys:
            rk = best(lk, rkeys)
            if rk is not None and best(rk, lkeys) == lk:
                diff = [n for n, (a, b) in enumerate(zip(lk[2], rk[2])) if a != b]
                l2r[lk] = (rk, diff)
                r2l[rk] = (lk, diff)
    return {"left": l2r, "right": r2l}


def _describe(side: _Side, i: Optional[int]) -> Optional[dict]:
    if i is None:
        return None
    rec = side.records[i]
    return {"record": side.ids[i], "function": rec.get("function"), "file": rec.get("fname"),
            "line": rec.get("line_num") or (rec.get("meta") or {}).get("lineno"),
            "counter": rec.get("counter")}


def _key_json(key: ValueKey) -> dict:
    return {"dtype": key[0], "numel": key[1], "xor_signature": list(key[2])}


def _rematch(side: _Side, other: _Side, keys: List[ValueKey], near, matched: set) -> Optional[dict]:
    """A consumer of the diverged value (on either side) whose output matches both runs.

    A real divergence propagates; a layout or packing difference (weights split
    with narrow on one side, packed with cat on the other) is consumed into values
    that agree again. Such a candidate is set aside, with this record as evidence.
    """
    for this, frontier in ((side, list(keys)), (other, [near[k][0] for k in keys if k in near])):
        seen = set(frontier)
        for hops in range(1, REMATCH_HOPS + 1):
            nxt = []
            for key in frontier:
                for i in this.consumers.get(key, []):
                    hit = [k for k in this.outs[i] if k in matched]
                    if hit:
                        return {"consumer": _describe(this, i), "rematched_value": _key_json(hit[0]),
                                "side": "same" if this is side else "other", "hops": hops}
                    nxt += [k for k in this.outs[i] if k not in seen]
                    seen.update(this.outs[i])
            frontier = nxt
    return None


def _pivot(side: _Side, other: _Side, matched: set, near: Dict[ValueKey, Tuple[ValueKey, List[int]]]):
    candidates, set_aside = [], []
    seen_matched = False
    for i, (ins, outs) in enumerate(zip(side.ins, side.outs)):
        m = [k for k in ins if k in matched]
        u = [k for k in ins if k not in matched]
        u_uncaptured = [k for k in u if side.producer.get(k, i) >= i]
        out_unmatched = [k for k in outs if k not in matched]
        found, diverged = None, []
        if m and len(u_uncaptured) == len(u) and out_unmatched:
            strong = [k for k in out_unmatched if k in near]
            diverged = strong or out_unmatched
            found = {"kind": "operator", "strength": "near_counterpart" if strong else "unmatched_output",
                     "matched_inputs": [_key_json(k) for k in m],
                     "inputs_without_captured_producer": [_key_json(k) for k in u],
                     "diverged": [_key_json(k) for k in (strong or out_unmatched)]}
            if strong:
                found["counterpart"] = _counterparts(other, strong, near)
        elif seen_matched:
            upstream = [k for k in u_uncaptured if k in near]
            if upstream:
                diverged = upstream
                found = {"kind": "uncaptured_upstream", "strength": "near_counterpart",
                         "diverged": [_key_json(k) for k in upstream],
                         "counterpart": _counterparts(other, upstream, near)}
        if found:
            found.update(_describe(side, i))
            evidence = _rematch(side, other, diverged, near, matched)
            if evidence:
                found["set_aside"] = {"reason": "diverged_value_rematches_downstream", **evidence}
                set_aside.append(found)
            else:
                candidates.append(found)
        seen_matched = seen_matched or bool(m) or any(k in matched for k in outs)
    strong = [c for c in candidates if c["strength"] == "near_counterpart"]
    pivot = strong[0] if strong else (candidates[0] if candidates else None)
    return pivot, candidates[:5], set_aside[:5]


def _counterparts(other: _Side, keys: List[ValueKey], near) -> List[dict]:
    out = []
    for key in keys:
        okey, diff = near[key]
        out.append({"value": _key_json(okey), "differing_regions": diff,
                    "producer": _describe(other, other.producer.get(okey)),
                    "first_seen": _describe(other, other.first_seen.get(okey))})
    return out


def match_values(left: Sequence[Tuple[Any, dict]], right: Sequence[Tuple[Any, dict]]) -> dict:
    """Value-match two ordered record lists of one rank. Records are (id, record)."""
    L, R = _Side(left), _Side(right)
    matched = set(L.values()) & set(R.values())
    producer_pairs = [(L.producer[k], R.producer[k], k) for k in matched
                      if k in L.producer and k in R.producer]
    kept, crossing = _monotone(producer_pairs)
    near = _near_links(L, R, matched)
    left_pivot, left_candidates, left_aside = _pivot(L, R, matched, near["left"])
    right_pivot, right_candidates, right_aside = _pivot(R, L, matched, near["right"])
    return {
        "method": "xor_value_match_v1",
        "evidence": "equal fingerprints of non-trivial tensors; unequal fingerprints are not evidence",
        "matched_values": len(matched),
        "links": [{"left": L.ids[i], "right": R.ids[j], "basis": "same_value_producer",
                   "value": _key_json(k)} for i, j, k in kept],
        "crossing_links_dropped": len(crossing),
        "near_links": [{"left_value": _key_json(lk), "right_value": _key_json(rk), "differing_regions": diff,
                        "left_producer": _describe(L, L.producer.get(lk)),
                        "left_first_seen": _describe(L, L.first_seen.get(lk)),
                        "right_producer": _describe(R, R.producer.get(rk)),
                        "right_first_seen": _describe(R, R.first_seen.get(rk))}
                       for lk, (rk, diff) in near["left"].items()],
        "pivot": {"left": left_pivot, "right": right_pivot} if matched else None,
        "candidates": {"left": left_candidates, "right": right_candidates} if matched else None,
        "set_aside": {"left": left_aside, "right": right_aside} if matched else None,
        "applicable": bool(matched),
    }


# ----------------------------------------------------------------- CLI
def _read_jsonl(path: Path) -> List[dict]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _rank_from_path(rel: str) -> Optional[int]:
    m = re.search(r"gpu_(\d+)", rel)
    return int(m.group(1)) if m else None


def load_dir(base: str) -> Dict[int, List[Tuple[str, dict]]]:
    """All operator records of a trace dir, grouped by rank, ordered by counter."""
    root = Path(base)
    by_rank: Dict[int, List[Tuple[str, dict]]] = defaultdict(list)
    for path in sorted(list(root.rglob("*.jsonl")) + list(root.rglob("*.jsonl.gz"))):
        rel = str(path.relative_to(root))
        for n, rec in enumerate(_read_jsonl(path)):
            if not isinstance(rec, dict) or not isinstance(rec.get("function"), str):
                continue
            rank = rec.get("gpu_rank", _rank_from_path(rel))
            by_rank[rank if isinstance(rank, int) else -1].append((f"{rel}:{n}", rec))
    for records in by_rank.values():
        records.sort(key=lambda x: (x[1].get("counter") if isinstance(x[1].get("counter"), int) else 10**15, x[0]))
    return by_rank


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run1"); ap.add_argument("run2")
    ap.add_argument("--rank", type=int, action="append", help="limit to these ranks")
    ap.add_argument("--out", help="write the full JSON report here")
    for run in ("run1", "run2"):
        ap.add_argument(f"--{run}-from", type=int, help=f"only {run} records with counter >= this")
        ap.add_argument(f"--{run}-to", type=int, help=f"only {run} records with counter <= this")
    args = ap.parse_args(argv)

    def window(by_rank, lo, hi):
        def keep(rec):
            c = rec.get("counter")
            return not isinstance(c, int) or ((lo is None or c >= lo) and (hi is None or c <= hi))
        return {r: [x for x in recs if keep(x[1])] for r, recs in by_rank.items()}

    a = window(load_dir(args.run1), args.run1_from, args.run1_to)
    b = window(load_dir(args.run2), args.run2_from, args.run2_to)
    ranks = sorted(set(a) & set(b)) if not args.rank else args.rank
    report = {str(r): match_values(a.get(r, []), b.get(r, [])) for r in ranks}
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=1))
    for r, rep in report.items():
        print(f"rank {r}: matched values={rep['matched_values']} links={len(rep['links'])} "
              f"near={len(rep['near_links'])} crossing_dropped={rep['crossing_links_dropped']}")
        if not rep["applicable"]:
            print("  no shared values: runs not bitwise-aligned here; value fallback does not apply")
            continue
        for side in ("left", "right"):
            for s in rep["set_aside"][side]:
                ev = s["set_aside"]["consumer"]
                print(f"  set aside[{'run1' if side == 'left' else 'run2'}]: {s['function']} counter={s['counter']} "
                      f"(re-matches downstream at {ev['function']} counter={ev['counter']}, "
                      f"{s['set_aside']['side']} side, {s['set_aside']['hops']} hop(s))")
            p = rep["pivot"][side]
            if p:
                print(f"  pivot[{'run1' if side == 'left' else 'run2'}]: {p['kind']} ({p['strength']}) "
                      f"{p['function']} {p['file']}:{p['line']} counter={p['counter']} record={p['record']}")
                for c in p.get("counterpart", []):
                    q = c["producer"] or c["first_seen"]
                    if q:
                        print(f"    counterpart: {q['function']} {q['file']}:{q['line']} counter={q['counter']} "
                              f"differing regions={c['differing_regions']}")
    return report


if __name__ == "__main__":
    main()
