"""Interpret recorded regional and per-vector fingerprints, without inferring value equality."""
import base64
import struct

VECTOR_SCHEME = 'lastdim-u32le-v1'


def vector_fingerprints(tensor):
    """Per-vector fingerprints of one recorded tensor, as unsigned 32-bit values.

    Reads `xor_vectors` (lastdim-u32le-v1: base64 of little-endian uint32) and the
    legacy `xor_rows` (signed int32 decimals, or one scalar for a single vector),
    which is masked with 0xFFFFFFFF. Returns status `available` with
    `vector_bytes`, `count` and `values`, `skipped` with the capture's reason, or
    `unavailable`. A zero value is valid but never evidence, and values of
    different `vector_bytes` are never comparable: the caller applies both.
    """
    shape = tensor.get('shape')
    expected = last = None
    if isinstance(shape, list) and all(type(n) is int and n >= 0 for n in shape):
        expected, last = 1, (shape[-1] if shape else 1)
        for n in shape[:-1]:
            expected *= n
    def result(encoding, width, values):
        if (type(width) is not int or width < 4 or width % 4 or not values
                or expected not in (None, len(values)) or (last and width % last)):
            return {'status': 'unavailable', 'reason': 'vector_fingerprints_inconsistent_with_shape'}
        return {'status': 'available', 'encoding': encoding, 'vector_bytes': width,
                'count': len(values), 'values': values}
    v1 = tensor.get('xor_vectors')
    if v1 is not None:
        if (not isinstance(v1, dict) or v1.get('scheme') != VECTOR_SCHEME
                or v1.get('encoding') != 'base64-u32le' or not isinstance(v1.get('data'), str)):
            return {'status': 'unavailable', 'reason': 'malformed_xor_vectors'}
        try:
            raw = base64.b64decode(v1['data'], validate=True)
        except ValueError:
            return {'status': 'unavailable', 'reason': 'malformed_xor_vectors'}
        if len(raw) % 4 or v1.get('count') != len(raw) // 4:
            return {'status': 'unavailable', 'reason': 'malformed_xor_vectors'}
        return result(VECTOR_SCHEME, v1.get('vector_bytes'), struct.unpack('<%dI' % (len(raw) // 4), raw))
    rows = tensor.get('xor_rows')
    if rows is not None:
        rows = rows if isinstance(rows, list) else [rows]
        if any(type(v) not in (int, float) or not -2**31 <= v < 2**32 or v != int(v) for v in rows):
            return {'status': 'unavailable', 'reason': 'malformed_legacy_xor_rows'}
        return result('legacy-xor_rows', tensor.get('xor_row_bytes'), tuple(int(v) & 0xFFFFFFFF for v in rows))
    if isinstance(tensor.get('xor_vectors_skipped'), str):
        return {'status': 'skipped', 'reason': tensor['xor_vectors_skipped']}
    return {'status': 'unavailable', 'reason': 'no_vector_fingerprints_recorded'}


def _layout(tensor):
    layout = tensor.get('xor_sig_layout')
    values = tensor.get('xor_signature')
    if tensor.get('xor_sig_mode') != 'bytes' or not isinstance(layout, dict):
        return None
    count, size = layout.get('regions'), layout.get('logical_nbytes')
    if (layout.get('scheme') != 'flat-u32-le-v1' or type(count) is not int
            or not 1 <= count <= 64 or type(size) is not int or size < 0):
        return None
    if count == 1 and type(values) in (int, float):
        values = [values]
    if not isinstance(values, list) or len(values) != count or any(
            type(x) not in (int, float) or not 0 <= x <= 0xffffffff or int(x) != x for x in values):
        return None
    base, rem = divmod((size + 3) // 4, count)
    ranges, start = [], 0
    for i in range(count):
        end = start + base + (i < rem)
        ranges.append([min(start * 4, size), min(end * 4, size)])
        start = end
    if layout.get('byte_ranges') != ranges:
        return None
    return layout, list(map(int, values))


def compare_regions(left, right):
    a, b = _layout(left), _layout(right)
    if a is None or b is None:
        return {'status': 'unavailable', 'reason': 'missing_or_unsupported_content_fingerprint_layout'}
    if a[0] != b[0] or left.get('dtype') != right.get('dtype'):
        return {'status': 'incompatible', 'reason': 'different_fingerprint_layout_or_dtype'}
    regions = [{'region': i, 'byte_range': bounds, 'left': av, 'right': bv,
                'fingerprint_xor': av ^ bv, 'fingerprints_agree': av == bv}
               for i, (bounds, av, bv) in enumerate(zip(a[0]['byte_ranges'], a[1], b[1]))]
    return {'status': 'available', 'scheme': a[0]['scheme'], 'regions': regions,
            'differing_regions': [r['region'] for r in regions if not r['fingerprints_agree']],
            'tensor_equality': 'not_established', 'different_element_count': 'not_available',
            'padding_assessment': 'not_determined'}


def regional_comparisons(left, right):
    """Same-slot proposals, or the explicit selected slots of a reviewed pair."""
    a, b = left.get('io') or {}, right.get('io') or {}
    claim = left.get('correspondence') or {}
    selected = (claim.get('basis') == 'explicit_alignment_map'
                and claim.get('peer_record_id') == right.get('record_id')
                and claim.get('tensor_slot') is not None)
    if selected:
        slots = [(claim['tensor_slot'], claim['peer_tensor_slot'])]
    else:
        slots = [([side, i], [side, i]) for side in ('inputs', 'outputs')
                 for i in range(min(len(a.get(side, [])), len(b.get(side, []))))]
    tensors = []
    for x, y in slots[:64]:
        av, bv = a[x[0]][x[1]], b[y[0]][y[1]]
        tensors.append({'left_tensor': x, 'right_tensor': y, **compare_regions(av, bv)})
    return {'selection_basis': 'reviewed_tensor_slots' if selected else 'same_io_slot_candidates',
            'tensors': tensors, 'truncated': len(slots) > 64,
            'note': 'Region differences localize recorded byte changes, not element counts or padding noise. '
                    'Matching regions may still contain XOR collisions. Validate semantic correspondence first.'}
