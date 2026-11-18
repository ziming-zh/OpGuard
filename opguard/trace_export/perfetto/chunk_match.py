"""Cross-stack alignment by chunk content, with a row-level first divergence.

Prototype. Needs captures taken with OPGUARD_XOR_CHUNK_BYTES set (the same value
on both sides), which records ``stats.xor_chunks``: one XOR per fixed-size chunk
of a tensor's flat bytes.

Why chunks: a rollout engine batches sequences and decodes token by token, a
trainer runs the whole sequence at once, and weights are split or concatenated
along rows. The same rows then sit in different tensors, so whole-tensor
fingerprints never meet. Chunk fingerprints do: a chunk seen on both sides is
the same data, whatever its order, batch, tensor or call site (a join on
content, not an alignment of sequences).

Positive evidence only. A row (bytes of one index along dim 0) spans several
chunks. If some of its chunks are found on the other side and some are not, that
row exists on both sides and differs: it *diverged*. A row none of whose chunks
is found is a gap (another sequence in the batch, a padding row, a different
packing), never a divergence. Splitting, concatenating and batching move whole
rows, so they leave every row matched and raise nothing.

Pivot per side, in trace order:
* operator: no input has a diverged row, some input row is matched, and an
  output has a diverged row.
* uncaptured_upstream: an input has a diverged row and no captured record on
  this side produced it: the divergence is in uncaptured code just before.
No tolerance anywhere: chunk equality is exact.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .value_match import _tensors, load_dir

CHUNK_FREQ_MAX = 64   # a chunk in more distinct values than this on a side is not evidence
_ELEM = {"bfloat16": 2, "float16": 2, "half": 2, "float32": 4, "float": 4, "float64": 8,
         "double": 8, "int64": 8, "long": 8, "int32": 4, "int": 4, "int16": 2, "int8": 1,
         "uint8": 1, "bool": 1, "float8_e4m3fn": 1, "float8_e5m2": 1}


class _Tensor:
    __slots__ = ("chunks", "chunk_bytes", "rows")

    def __init__(self, chunks: Tuple[int, ...], chunk_bytes: int, rows: List[Tuple[int, int]]):
        self.chunks, self.chunk_bytes, self.rows = chunks, chunk_bytes, rows


def _as_tensor(t: dict) -> Optional[_Tensor]:
    st = t["stats"]
    raw = st.get("xor_chunks")
    cb = st.get("xor_chunk_bytes")
    if raw is None or not isinstance(cb, int):
        return None
    if not isinstance(raw, list):
        raw = [raw]                      # a single chunk is serialized as a scalar
    chunks = tuple(int(v) & 0xFFFFFFFF for v in raw)
    shape = t.get("shape") or []
    elem = _ELEM.get(str(st.get("dtype", "")).replace("torch.", ""))
    rows = [(0, len(chunks))]            # default: the whole tensor is one row
    if elem and len(shape) >= 2 and all(isinstance(s, int) for s in shape):
        row_bytes = elem
        for s in shape[1:]:
            row_bytes *= s
        if row_bytes % cb == 0 and row_bytes // cb >= 2:
            per = row_bytes // cb
            rows = [(r * per, (r + 1) * per) for r in range(shape[0])]
    return _Tensor(chunks, cb, rows)


def _io(obj: Any) -> List[_Tensor]:
    out, seen = [], set()
    for t in _tensors(obj):
        x = _as_tensor(t)
        if x is not None and x.chunks not in seen:
            seen.add(x.chunks)
            out.append(x)
    return out


class _Side:
    def __init__(self, records: Sequence[Tuple[Any, dict]]):
        self.ids = [rid for rid, _ in records]
        self.records = [rec for _, rec in records]
        self.ins = [_io(r.get("inputs")) for r in self.records]
        self.outs = [_io(r.get("outputs")) for r in self.records]
        self.producer: Dict[Tuple[int, ...], int] = {}
        self.holders: Dict[int, List[int]] = defaultdict(list)   # chunk -> records holding it
        freq: Counter = Counter()
        seen_values = set()
        for i, (ins, outs) in enumerate(zip(self.ins, self.outs)):
            for t in outs:
                self.producer.setdefault(t.chunks, i)
            for t in ins + outs:
                for c in set(t.chunks):
                    self.holders[c].append(i)
                if t.chunks not in seen_values:
                    seen_values.add(t.chunks)
                    freq.update(set(t.chunks))
        self.freq = freq
        self.chunk_bytes = {t.chunk_bytes for ts in self.ins + self.outs for t in ts}


def _row_states(t: _Tensor, side: _Side, other: _Side):
    """Per row: 'matched', 'diverged' (with chunk indices) or None (gap/trivial)."""
    states = []
    for lo, hi in t.rows:
        ev = [k for k in range(lo, min(hi, len(t.chunks)))
              if t.chunks[k] and side.freq[t.chunks[k]] <= CHUNK_FREQ_MAX]
        if not ev:
            states.append(None)
            continue
        found = [k for k in ev if t.chunks[k] in other.freq]
        missing = [k for k in ev if t.chunks[k] not in other.freq]
        if not missing:
            states.append(("matched", []))
        elif found:
            states.append(("diverged", missing))
        else:
            states.append(None)
    return states


def _describe(side: _Side, i: Optional[int]) -> Optional[dict]:
    if i is None:
        return None
    rec = side.records[i]
    return {"record": side.ids[i], "function": rec.get("function"), "file": rec.get("fname"),
            "line": rec.get("line_num") or (rec.get("meta") or {}).get("lineno"),
            "counter": rec.get("counter")}


def _diverged_rows(t: _Tensor, states, other: _Side) -> dict:
    rows = [r for r, s in enumerate(states) if s and s[0] == "diverged"]
    # Where the other side holds this row: any record holding one of its matched chunks.
    r0 = rows[0]
    lo, hi = t.rows[r0]
    anchor = next((t.chunks[k] for k in range(lo, hi) if t.chunks[k] in other.freq), None)
    held = other.holders.get(anchor, []) if anchor is not None else []
    return {"rows": rows[:16], "rows_total": len(rows),
            "chunks": [k for r in rows[:4] for k in states[r][1]][:16],
            "chunk_bytes": t.chunk_bytes,
            "counterpart": _describe(other, held[0]) if held else None}


def _pivot(side: _Side, other: _Side):
    candidates = []
    seen_matched = False
    for i, (ins, outs) in enumerate(zip(side.ins, side.outs)):
        in_states = [(t, _row_states(t, side, other)) for t in ins]
        out_states = [(t, _row_states(t, side, other)) for t in outs]
        in_div = [(t, s) for t, s in in_states if any(x and x[0] == "diverged" for x in s)]
        in_match = any(any(x and x[0] == "matched" for x in s) for _, s in in_states)
        out_div = [(t, s) for t, s in out_states if any(x and x[0] == "diverged" for x in s)]
        found = None
        if not in_div and in_match and out_div:
            t, s = out_div[0]
            found = {"kind": "operator", "diverged_output": _diverged_rows(t, s, other)}
        elif in_div and seen_matched:
            uncaptured = [(t, s) for t, s in in_div if side.producer.get(t.chunks, i) >= i]
            if uncaptured:
                t, s = uncaptured[0]
                found = {"kind": "uncaptured_upstream", "diverged_input": _diverged_rows(t, s, other)}
        if found:
            found.update(_describe(side, i))
            candidates.append(found)
        seen_matched = seen_matched or in_match or any(
            any(x and x[0] == "matched" for x in s) for _, s in out_states)
    return (candidates[0] if candidates else None), candidates[:5]


def match_chunks(left: Sequence[Tuple[Any, dict]], right: Sequence[Tuple[Any, dict]]) -> dict:
    """Chunk-join two ordered record lists of one rank. Records are (id, record)."""
    L, R = _Side(left), _Side(right)
    if L.chunk_bytes and R.chunk_bytes and L.chunk_bytes != R.chunk_bytes:
        raise ValueError(f"chunk sizes differ between runs: {L.chunk_bytes} vs {R.chunk_bytes}")
    shared = sum(1 for c in L.freq if c and c in R.freq)
    left_pivot, left_c = _pivot(L, R)
    right_pivot, right_c = _pivot(R, L)
    return {"method": "xor_chunk_join_v1", "shared_chunks": shared,
            "applicable": shared > 0,
            "pivot": {"left": left_pivot, "right": right_pivot} if shared else None,
            "candidates": {"left": left_c, "right": right_c} if shared else None}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Cross-stack chunk-join alignment (prototype)")
    ap.add_argument("run1"); ap.add_argument("run2")
    ap.add_argument("--rank", type=int, action="append")
    ap.add_argument("--out")
    args = ap.parse_args(argv)
    a, b = load_dir(args.run1), load_dir(args.run2)
    ranks = sorted(set(a) & set(b)) if not args.rank else args.rank
    report = {str(r): match_chunks(a.get(r, []), b.get(r, [])) for r in ranks}
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=1))
    for r, rep in report.items():
        print(f"rank {r}: shared chunks={rep['shared_chunks']}")
        if not rep["applicable"]:
            print("  nothing shared: no chunk fingerprints, or runs not bitwise-aligned")
            continue
        for side in ("left", "right"):
            p = rep["pivot"][side]
            if p:
                d = p.get("diverged_output") or p.get("diverged_input")
                cp = d["counterpart"]
                print(f"  pivot[{'run1' if side == 'left' else 'run2'}]: {p['kind']} {p['function']} "
                      f"{p['file']}:{p['line']} counter={p['counter']} rows={d['rows']}"
                      + (f" | other side: {cp['function']} counter={cp['counter']}" if cp else ""))
    return report


if __name__ == "__main__":
    main()
