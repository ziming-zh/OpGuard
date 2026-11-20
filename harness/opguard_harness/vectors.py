"""Exact facts over per-vector fingerprints in an OpGuard Perfetto export.

A vector is addressed as event_ref + slot (["inputs" | "outputs", i] of the
exported io, as in alignment maps) + vector index k. Values are read from
xor_vectors (lastdim-u32le-v1) or the legacy xor_rows. Equality is exact 32-bit
XOR equality. Nothing here decides correspondence, token layout or divergence:
token dimensions are declared by the caller and echoed back.
"""
import collections

from .fingerprints import vector_fingerprints
from .trace_reader import Trace

MAX_LISTED = 200
NOTE = ('Exact 32-bit XOR equality; equal fingerprints can collide. A zero fingerprint is a valid value '
        'but never evidence, and a match between different vector widths or dtypes is never evidence.')


def _slot(trace, index, slot):
    if (not isinstance(slot, list) or len(slot) != 2 or slot[0] not in ('inputs', 'outputs')
            or type(slot[1]) is not int or slot[1] < 0):
        raise ValueError('slot must be ["inputs" | "outputs", nonnegative index] of the exported io')
    values = (trace.events[index]['args'].get('io') or {}).get(slot[0]) or []
    if slot[1] >= len(values) or not isinstance(values[slot[1]], dict):
        raise ValueError('slot is absent from the exported io')
    return values[slot[1]]


def _vectors(trace, index, slot):
    found = vector_fingerprints(_slot(trace, index, slot))
    if found['status'] != 'available':
        raise ValueError(f'no vector fingerprints at {slot}: {found["status"]}, {found["reason"]}')
    return found


def _where(trace, index):
    args = trace.events[index]['args']
    return {'event_ref': trace.ref(index), 'run': args.get('run'), 'rank': args.get('rank'),
            'counter': args.get('counter'), 'fn': args.get('fn')}


def _scope_filter(scope):
    if scope is None:
        scope = {}
    if not isinstance(scope, dict) or set(scope) - {'runs', 'ranks', 'counter_min', 'counter_max'}:
        raise ValueError('scope is null or {runs, ranks, counter_min, counter_max}, each null or set')
    runs, ranks = scope.get('runs'), scope.get('ranks')
    lo, hi = scope.get('counter_min'), scope.get('counter_max')

    def keep(args):
        c = args.get('counter')
        return ((runs is None or args.get('run') in runs) and (ranks is None or args.get('rank') in ranks)
                and (lo is None or (type(c) is int and c >= lo)) and (hi is None or (type(c) is int and c <= hi)))
    return keep, {'runs': runs, 'ranks': ranks, 'counter_min': lo, 'counter_max': hi}


def _index(trace, keep):
    """fingerprint -> [(event index, slot, vector index, vector_bytes, dtype)] over the scope."""
    found = collections.defaultdict(list)
    for i in trace.ops:
        if not keep(trace.events[i]['args']):
            continue
        for side in ('inputs', 'outputs'):
            for j, tensor in enumerate((trace.events[i]['args'].get('io') or {}).get(side) or []):
                if not isinstance(tensor, dict):
                    continue
                vf = vector_fingerprints(tensor)
                if vf['status'] == 'available':
                    for k, value in enumerate(vf['values']):
                        found[value].append((i, [side, j], k, vf['vector_bytes'], tensor.get('dtype')))
    return found


def vectors_describe(artifact, event_ref, first_n):
    if type(first_n) is not int or not 0 <= first_n <= 256:
        raise ValueError('first_n must be an integer from 0 to 256')
    trace = Trace(artifact)
    index = trace.index(event_ref)
    slots = []
    for side in ('inputs', 'outputs'):
        for j, tensor in enumerate((trace.events[index]['args'].get('io') or {}).get(side) or []):
            if not isinstance(tensor, dict):
                continue
            vf = vector_fingerprints(tensor)
            item = {'slot': [side, j], 'shape': tensor.get('shape'), 'dtype': tensor.get('dtype'), 'status': vf['status']}
            if vf['status'] == 'available':
                values = vf['values']
                item.update(encoding=vf['encoding'], vector_bytes=vf['vector_bytes'], count=vf['count'],
                            zero_vectors=sum(v == 0 for v in values),
                            first=[{'vector': k, 'fingerprint': v, 'zero': v == 0} for k, v in enumerate(values[:first_n])])
            else:
                item['reason'] = vf['reason']
            slots.append(item)
    return {'artifact': trace.identity(), 'event': trace.compact(index), 'slots': slots,
            'verdict': 'not_evaluated', 'note': NOTE}


def vectors_lookup(artifact, fingerprint, event_ref, slot, vector, scope):
    trace = Trace(artifact)
    query = {'fingerprint': fingerprint, 'vector_bytes': None, 'dtype': None, 'source': None}
    if fingerprint is not None:
        if event_ref is not None or slot is not None or vector is not None:
            raise ValueError('give a fingerprint, or event_ref + slot + vector, not both')
        if type(fingerprint) is not int or not 0 <= fingerprint <= 0xFFFFFFFF:
            raise ValueError('fingerprint must be an unsigned 32-bit integer')
    else:
        if event_ref is None or slot is None or type(vector) is not int:
            raise ValueError('give a fingerprint, or event_ref + slot + vector')
        index = trace.index(event_ref)
        vf = _vectors(trace, index, slot)
        if not 0 <= vector < vf['count']:
            raise ValueError(f'vector index outside 0..{vf["count"] - 1}')
        query.update(fingerprint=vf['values'][vector], vector_bytes=vf['vector_bytes'],
                     dtype=_slot(trace, index, slot).get('dtype'),
                     source={**_where(trace, index), 'slot': slot, 'vector': vector})
    keep, scope = _scope_filter(scope)
    hits = _index(trace, keep).get(query['fingerprint'], [])
    listed = []
    for i, s, k, width, dtype in hits[:MAX_LISTED]:
        item = {**_where(trace, i), 'slot': s, 'vector': k, 'vector_bytes': width, 'dtype': dtype}
        if query['vector_bytes'] is not None:
            item.update(same_vector_bytes=width == query['vector_bytes'], same_dtype=dtype == query['dtype'],
                        is_query=query['source'] is not None and item['event_ref'] == query['source']['event_ref']
                        and s == slot and k == vector)
        listed.append(item)
    zero = query['fingerprint'] == 0
    return {'artifact': trace.identity(), 'query': {**query, 'zero': zero}, 'scope': scope,
            'occurrence_count': 0 if zero else len(hits), 'occurrences': [] if zero else listed,
            'zero_fingerprint_matches': listed if zero else [], 'zero_fingerprint_match_count': len(hits) if zero else 0,
            'truncated': len(hits) > MAX_LISTED, 'verdict': 'not_evaluated', 'note': NOTE}


def vectors_coverage(artifact, event_ref, slot, scope):
    trace = Trace(artifact)
    index = trace.index(event_ref)
    vf = _vectors(trace, index, slot)
    dtype = _slot(trace, index, slot).get('dtype')
    if scope is None:      # default: the other run(s) of this export
        runs = sorted({trace.events[i]['args'].get('run') for i in trace.ops} - {trace.events[index]['args'].get('run')},
                      key=str)
        scope = {'runs': runs}
    keep, scope = _scope_filter(scope)
    found = _index(trace, keep)
    rows, counts = [], collections.Counter()
    for k, value in enumerate(vf['values']):
        hits = [h for h in found.get(value, []) if not (h[0] == index and h[1] == slot and h[2] == k)]
        same = [h for h in hits if h[3] == vf['vector_bytes'] and h[4] == dtype]
        state = 'zero' if value == 0 else 'occurs' if same else 'absent'
        counts[state] += 1
        rows.append({'vector': k, 'fingerprint': value, 'zero': value == 0, 'state': state,
                     'same_width_occurrences': len(same), 'other_width_or_dtype_occurrences': len(hits) - len(same),
                     'first': [{**_where(trace, i), 'slot': s, 'vector': j} for i, s, j, _, _ in same[:3]]})
    return {'artifact': trace.identity(), 'event': trace.compact(index), 'slot': slot, 'dtype': dtype,
            'vector_bytes': vf['vector_bytes'], 'count': vf['count'], 'scope': scope,
            'summary': {'occurs': counts['occurs'], 'absent': counts['absent'], 'zero': counts['zero']},
            'vectors': rows[:4096], 'truncated': len(rows) > 4096, 'verdict': 'not_evaluated',
            'note': NOTE + ' "occurs" counts only nonzero matches of the same width and dtype.'}


def _token_rows(trace, ref, slot, dims, ids):
    index = trace.index(ref)
    tensor = _slot(trace, index, slot)
    vf = _vectors(trace, index, slot)
    shape = tensor.get('shape')
    if not isinstance(shape, list) or any(type(n) is not int for n in shape):
        raise ValueError(f'{slot} has no recorded shape')
    if type(dims) is not int or not 0 <= dims < max(1, len(shape)):
        raise ValueError(f'token_dims must be a number of leading dims from 0 to {max(0, len(shape) - 1)}')
    tokens = 1
    for n in shape[:dims]:
        tokens *= n
    per = vf['count'] // tokens if tokens else 0
    if ids is None:
        ids = list(range(tokens))
    elif not isinstance(ids, list) or len(ids) != tokens:
        raise ValueError(f'token_ids must give one distinct id for each of the {tokens} token rows')
    else:
        ids = [tuple(t) if isinstance(t, list) else t for t in ids]      # e.g. [seq_id, position] from forward rows
        real = [t for t in ids if t is not None]                         # null: a padding row, left out
        if len(set(real)) != len(real):
            raise ValueError(f'token_ids must give one distinct id (or null) for each of the {tokens} token rows')
    rows = {t: vf['values'][r * per:(r + 1) * per] for r, t in enumerate(ids) if t is not None}
    ids = [t for t in ids if t is not None]
    return {'event': _where(trace, index), 'slot': slot, 'shape': shape, 'dtype': tensor.get('dtype'),
            'declared_token_dims': dims, 'tokens': tokens, 'vectors_per_token': per,
            'vector_bytes': vf['vector_bytes'], 'row_bytes': per * vf['vector_bytes']}, ids, rows


def _xor(values):
    acc = 0
    for v in values:
        acc ^= v
    return acc


def vectors_compare(artifact, left_ref, left_slot, left_token_dims, left_token_ids,
                    right_ref, right_slot, right_token_dims, right_token_ids):
    trace = Trace(artifact)
    left, lids, lrows = _token_rows(trace, left_ref, left_slot, left_token_dims, left_token_ids)
    right, rids, rrows = _token_rows(trace, right_ref, right_slot, right_token_dims, right_token_ids)
    result = {'artifact': trace.identity(), 'left': left, 'right': right,
              'token_layout': 'declared_by_caller', 'verdict': 'not_evaluated', 'note': NOTE}
    if left['dtype'] != right['dtype'] or left['row_bytes'] != right['row_bytes']:
        return {**result, 'status': 'incompatible',
                'reason': 'different dtype or bytes per token row: no row of one can equal a row of the other'}
    by_vector = left['vector_bytes'] == right['vector_bytes']
    rows, counts = [], collections.Counter()
    for token in lids + [t for t in rids if t not in lrows]:
        a, b = lrows.get(token), rrows.get(token)
        if a is None or b is None:
            state, item = ('missing_left' if a is None else 'missing_right'), {}
        else:
            fa, fb = _xor(a), _xor(b)
            differing = [k for k, (x, y) in enumerate(zip(a, b)) if x != y] if by_vector else None
            equal = (not differing) if by_vector else fa == fb
            # Equality carried only by zero fingerprints is no evidence: a distinct state.
            zero_only = equal and (not any(a) if by_vector else fa == 0)
            state = 'equal_zero_only' if zero_only else 'equal' if equal else 'differ'
            item = {'left_row_fingerprint': fa, 'right_row_fingerprint': fb, 'zero_row': fa == 0 or fb == 0,
                    'differing_vectors': differing}
            counts['zero_row'] += item['zero_row']
        counts[state] += 1
        rows.append({'token': list(token) if isinstance(token, tuple) else token, 'state': state, **item})
    return {**result, 'status': 'compared', 'unit': 'vectors' if by_vector else 'merged_token_rows',
            'summary': {k: counts[k] for k in ('equal', 'equal_zero_only', 'differ', 'missing_left', 'missing_right',
                                                  'zero_row')},
            'rows': rows[:4096], 'truncated': len(rows) > 4096}


VECTOR_TOOLS = {fn.__name__: fn for fn in (vectors_describe, vectors_lookup, vectors_coverage, vectors_compare)}
