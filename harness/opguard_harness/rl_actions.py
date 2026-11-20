"""Fact actions over RL harness records (docs/RLAlignment.md). No tolerance, no verdict.

Every value says which records it came from ("path:line"). Exactness: a
log-probability keeps its bits; differences are computed as exact rationals,
reported as the nearest float64 with `exact: false` when that rounds, and a
zero is decided on bits (same dtype, same bits), never on a rounded value.
"""
import collections
from fractions import Fraction
import json
import math
from pathlib import Path

from .fingerprints import vector_fingerprints
from .rl_records import decode_logprob, read_records, weights_fingerprint

NOTE = 'Facts from recorded values only; verdict not_evaluated. Zero is decided on bits.'


def _load(records):
    if isinstance(records, str):
        records = [records]
    if not isinstance(records, list) or not records or any(not isinstance(p, str) for p in records):
        raise ValueError('records must be a nonempty list of record file paths')
    return read_records(records)


def _one(rows, kind, test, what):
    found = [(ref, rec) for ref, rec in rows if rec['type'] == kind and test(rec)]
    if not found:
        raise ValueError(f'no {kind} record for {what}')
    if len(found) > 1:
        raise ValueError(f'{len(found)} {kind} records for {what}: {[ref for ref, _ in found]}')
    return found[0]


def rl_records_summary(records):
    rows = _load(records)
    kinds = collections.Counter(rec['type'] for _, rec in rows)
    batches = collections.defaultdict(lambda: {'rollout_batch': [], 'trainer_logprob_versions': []})
    forwards = collections.Counter()
    for ref, rec in rows:
        if rec['type'] == 'rollout_batch':
            versions = sorted({v for s in rec['sequences'] for v in s['policy_version']})
            batches[str(rec['batch_id'])]['rollout_batch'].append(
                {'ref': ref, 'sequences': len(rec['sequences']), 'policy_versions': versions,
                 'composition_recorded': rec.get('composition') is not None, 'engine_flags': rec.get('engine_flags')})
        elif rec['type'] == 'trainer_logprob':
            batches[str(rec['batch_id'])]['trainer_logprob_versions'].append({'ref': ref, 'version': rec['policy_version']})
        elif rec['type'] == 'forward':
            forwards[rec['rank']] += 1
    pick = lambda kind: sorted(rec['version'] for _, rec in rows if rec['type'] == kind)
    return {'records': len(rows), 'by_type': dict(kinds), 'batches': dict(batches),
            'forwards_per_rank': {str(r): n for r, n in sorted(forwards.items())},
            'weight_sync_versions': pick('weight_sync'), 'snapshot_versions': pick('snapshot'),
            'basis': 'observed', 'verdict': 'not_evaluated'}


def _diff(a, b):
    """a - b for (dtype, bits, value) triples: exact, with a zero decided on bits."""
    out = {'zero': a[0] == b[0] and a[1] == b[1], 'dtype_mismatch': a[0] != b[0]}
    if not (math.isfinite(a[2]) and math.isfinite(b[2])):
        return {**out, 'value': None, 'exact': None, 'non_finite': True}
    exact = Fraction(a[2]) - Fraction(b[2])
    return {**out, 'value': float(exact), 'exact': Fraction(float(exact)) == exact}


def _weights_check(rows, roll):
    """Which recorded weight syncs hold the weights the batch says it ran (observed fingerprint)."""
    wf = roll.get('weights_fingerprint')
    if wf is None:
        return {'status': 'not_recorded'}
    syncs = {}
    for ref, rec in rows:
        if rec['type'] == 'weight_sync':
            for side in ('rollout', 'trainer'):
                if rec[side]:
                    fp = weights_fingerprint(rec[side])
                    syncs.setdefault(rec['version'], {})[side] = (fp['value'], fp['rows'], ref)
    labelled = sorted({v for s in roll['sequences'] for v in s['policy_version']})
    match = lambda side: sorted(v for v, d in syncs.items() if side in d and d[side][:2] == (wf['value'], wf['rows']))
    rollout_match = match('rollout')
    return {'status': 'recorded', 'batch_weights': wf, 'labelled_versions': labelled,
            'rollout_sync_versions_matching': rollout_match, 'trainer_sync_versions_matching': match('trainer'),
            'labelled_versions_without_sync': [v for v in labelled if v not in syncs],
            'first_labelled_version_matches': labelled[0] in rollout_match if labelled else None,
            'zero_fingerprint': wf['value'] == 0,
            'note': 'XOR of all weight rows (32 bits, order-free): a match can collide and zero is never evidence. '
                    'A mixed batch changes weights mid-batch; the fingerprint is of the weights when it started.'}


def rl_decompose(records, batch_id, trainer_version, policy_version_override):
    if type(trainer_version) is not int:
        raise ValueError('trainer_version (the trainer\'s current policy version) must be an integer')
    if policy_version_override is not None and type(policy_version_override) is not int:
        raise ValueError('policy_version_override is null or an integer version supplied by the caller')
    rows = _load(records)
    rref, roll = _one(rows, 'rollout_batch', lambda r: r['batch_id'] == batch_id, f'batch {batch_id!r}')
    weights = _weights_check(rows, roll)
    if policy_version_override is not None:
        roll = {**roll, 'sequences': [{**s, 'recorded_policy_version': s['policy_version'],
                                       'policy_version': [policy_version_override] * len(s['positions'])}
                                      for s in roll['sequences']]}
    trainer = collections.defaultdict(list)        # (version, seq_id, position) -> [(ref, value)]
    for ref, rec in rows:
        if rec['type'] == 'trainer_logprob' and rec['batch_id'] == batch_id:
            for s in rec['sequences']:
                for pos, value in zip(s['positions'], decode_logprob(s['logprob'])):
                    trainer[(rec['policy_version'], s['seq_id'], pos)].append((ref, value))

    def at(version, seq, pos):
        got = trainer.get((version, seq, pos), [])
        if not got:
            return None, f'no trainer_logprob at version {version}'
        if len({v[:2] for _, v in got}) > 1:
            return None, f'trainer_logprob records at version {version} disagree: {[r for r, _ in got]}'
        return got[0], None

    tokens, counts, replay = [], collections.Counter(), set()
    for s in roll['sequences']:
        mixed = len(set(s['policy_version'])) > 1
        counts['mixed_sequences'] += mixed
        for i, (pos, tok, v, r) in enumerate(zip(s['positions'], s['token_ids'], s['policy_version'],
                                                  decode_logprob(s['logprob']))):
            item = {'seq_id': s['seq_id'], 'index': i, 'position': pos, 'token_id': tok, 'policy_version': v,
                    **({'recorded_policy_version': s['recorded_policy_version'][i]}
                       if 'recorded_policy_version' in s else {}),
                    'mixed_version': mixed, 'rollout_logprob': r[2], 'rollout_dtype': r[0],
                    'staleness_zero_point': v == trainer_version, 'sources': {'rollout': rref}}
            counts['tokens'] += 1
            if mixed:
                counts['mixed_version_tokens'] += 1
                item.update(numeric=None, staleness=None, not_split='mixed_version_sequence')
                tokens.append(item)
                continue
            counts['staleness_zero_points'] += v == trainer_version
            tv, why = at(v, s['seq_id'], pos)
            if tv is None:
                item.update(numeric=None, numeric_missing=why)
                counts['numeric_missing'] += 1
                replay.add(v)
            else:
                item['sources']['trainer_at_policy_version'] = tv[0]
                item['numeric'] = _diff(r, tv[1])
                counts['numeric_zero' if item['numeric']['zero'] else 'numeric_nonzero'] += 1
            if v == trainer_version:
                item['staleness'] = {'zero': True, 'value': 0.0, 'exact': True, 'dtype_mismatch': False,
                                     'basis': 'policy_version equals trainer_version'}
                counts['staleness_zero'] += 1
            else:
                tc, why_c = at(trainer_version, s['seq_id'], pos)
                if tv is None or tc is None:
                    item.update(staleness=None, staleness_missing=why if tv is None else why_c)
                    counts['staleness_missing'] += 1
                else:
                    item['sources']['trainer_at_current_version'] = tc[0]
                    item['staleness'] = _diff(tv[1], tc[1])
                    counts['staleness_zero' if item['staleness']['zero'] else 'staleness_nonzero'] += 1
            tokens.append(item)
    keys = ('tokens', 'mixed_sequences', 'mixed_version_tokens', 'staleness_zero_points', 'numeric_zero',
            'numeric_nonzero', 'numeric_missing', 'staleness_zero', 'staleness_nonzero', 'staleness_missing')
    return {'batch_id': batch_id, 'trainer_version': trainer_version, 'rollout_record': rref,
            'policy_version_source': 'recorded' if policy_version_override is None else 'caller_supplied',
            'policy_version_override': policy_version_override, 'weights_check': weights,
            'engine_flags': roll.get('engine_flags'), 'summary': {k: counts[k] for k in keys},
            'versions_without_trainer_logprob': sorted(replay), 'tokens': tokens,
            **({'override_note': 'Every token was split as if generated at the caller-supplied version: a hypothesis, '
                                 'not an observation. recorded_policy_version keeps what was recorded.'}
               if policy_version_override is not None else {}),
            'definitions': {'numeric': 'rollout_logprob(v) - trainer_logprob(v)',
                            'staleness': 'trainer_logprob(v) - trainer_logprob(trainer_version)'},
            'verdict': 'not_evaluated', 'note': NOTE + ' Mixed-version sequences are never split.'}


def _rows(params):
    """[(param, row values, key per row)] where key = (vector_bytes, dtype, fingerprint)."""
    out = []
    for p in params:
        vf = vector_fingerprints({'shape': p['shape'], 'xor_vectors': p['xor_vectors']})
        values = list(vf['values']) if vf['status'] == 'available' else None
        out.append((p, vf, values and [(vf['vector_bytes'], p['dtype'], v) for v in values]))
    return out


def rl_weight_sync_check(records, version):
    rows = _load(records)
    ref, rec = _one(rows, 'weight_sync', lambda r: r['version'] == version, f'version {version}')
    sides = {name: _rows(rec[name]) for name in ('trainer', 'rollout')}
    pools = {name: collections.Counter(k for _, _, keys in sides[name] if keys for k in keys) for name in sides}
    report, totals = {}, collections.Counter()
    for name, other in (('trainer', 'rollout'), ('rollout', 'trainer')):
        pool, params = pools[other].copy(), []
        for p, vf, keys in sides[name]:
            item = {'name': p['name'], 'shape': p['shape'], 'dtype': p['dtype']}
            if keys is None:
                params.append({**item, 'status': 'unavailable', 'reason': vf['reason']})
                totals[name + '_unavailable'] += 1
                continue
            unmatched, zero = [], 0
            for i, k in enumerate(keys):
                zero += k[2] == 0
                if pool[k]:
                    pool[k] -= 1
                else:
                    unmatched.append(i)
            totals[name + '_rows'] += len(keys)
            totals[name + '_unmatched_rows'] += len(unmatched)
            params.append({**item, 'vector_bytes': vf['vector_bytes'], 'rows': len(keys),
                           'status': 'all_rows_matched' if not unmatched else 'rows_unmatched',
                           'unmatched_rows': unmatched[:256], 'unmatched_count': len(unmatched),
                           'zero_rows': zero, 'truncated': len(unmatched) > 256})
        report[name] = params
    return {'version': version, 'record': ref, 'summary': dict(totals), 'trainer': report['trainer'],
            'rollout': report['rollout'], 'verdict': 'not_evaluated',
            'note': ('Rows (last-dimension vectors) are matched by exact (width, dtype, fingerprint) across all '
                     'parameters of a side, as multisets: permuted, split or concatenated rows are not differences. '
                     'An unmatched row is listed by parameter and row index on its own side. Zero rows match as '
                     'values but are never evidence; packing that changes the row width cannot match.')}


BATCH_INVARIANT_NOTE = ('A batch-invariant replay does not reproduce the original batch composition. It localizes '
                        'implementation differences between the two sides, not the specific error of the original run.')


def rl_replay_request(records, batch_id, version, side, composition, capture, output):
    if side not in ('trainer', 'rollout') or composition not in ('original', 'batch_invariant'):
        raise ValueError('side is trainer|rollout and composition is original|batch_invariant')
    if type(version) is not int:
        raise ValueError('version must be an integer policy version')
    if capture is not None and (not isinstance(capture, dict) or set(capture) - {'vector_fingerprints', 'env'}):
        raise ValueError('capture is null or {vector_fingerprints: bool, env: {NAME: value}}')
    capture = capture or {}
    rows = _load(records)
    rref, roll = _one(rows, 'rollout_batch', lambda r: r['batch_id'] == batch_id, f'batch {batch_id!r}')
    if composition == 'original' and roll.get('composition') is None:
        raise ValueError('the batch composition was not recorded; only a batch_invariant replay can be requested')
    snaps = [(ref, rec) for ref, rec in rows if rec['type'] == 'snapshot' and rec['version'] == version]
    env = {str(k): str(v) for k, v in (capture.get('env') or {}).items()}
    if capture.get('vector_fingerprints'):
        env['OPGUARD_XOR_ROWS'] = '1'
    flags = dict(roll.get('engine_flags') or {})
    if composition == 'batch_invariant':
        flags['batch_invariant'] = True
    request = {'schema': 'opguard.rl.replay.v1', 'batch_id': batch_id, 'version': version, 'side': side,
               'composition': composition,
               'weights': {'location': snaps[0][1]['location'], 'record': snaps[0][0]} if snaps else None,
               'sequences': [{'seq_id': s['seq_id'], 'prompt_ids': s.get('prompt_ids'), 'token_ids': s['token_ids'],
                              'positions': s['positions']} for s in roll['sequences']],
               'forward_composition': roll['composition'] if composition == 'original' else None,
               'engine_flags': flags, 'capture_env': env,
               'capture_path': {'assumes': 'opguard_tracer',
                                'note': 'capture_env reaches OpGuard\'s tracer (instrumented call sites, summarize_tensor). '
                                        'A replay that writes its own boundary records must compute vector fingerprints '
                                        'itself, e.g. with opguard.tracer.xor_signature.xor_vector_stats.'},
               'source_records': [rref] + [r for r, _ in snaps]}
    missing_prompts = [s['seq_id'] for s in roll['sequences'] if s.get('prompt_ids') is None]
    if output is not None:
        path = Path(output).expanduser()
        if not path.is_absolute() or path.exists():
            raise ValueError('output must be a fresh absolute path')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(request, indent=2) + '\n')
    return {'request': request, 'written_to': str(output) if output else None, 'executed': False,
            'weights_located': bool(snaps), 'sequences_without_prompt_ids': missing_prompts[:64],
            'runnable_alone': bool(snaps) and not missing_prompts,
            'reproduces': 'original_batch_composition' if composition == 'original' else 'implementation_differences_only',
            'note': (BATCH_INVARIANT_NOTE if composition == 'batch_invariant' else
                     'Replays the recorded co-batching. Nothing was run: pass this request to your replay workload '
                     'and launch it offline through create_experiment / advance_experiment.')
                    + ('' if snaps else ' No snapshot record names the weights for this version.'),
            'verdict': 'not_evaluated'}


def rl_forwards(records, rank, counter_min, counter_max):
    """The forward records of a rank in counter order, with their observed row layout when recorded."""
    out = []
    for ref, rec in _load(records):
        c = rec.get('counter')
        if (rec['type'] == 'forward' and (rank is None or rec['rank'] == rank)
                and (counter_min is None or c >= counter_min) and (counter_max is None or c <= counter_max)):
            out.append({'ref': ref, 'counter': c, 'rank': rec['rank'], 'tokens': rec['tokens'],
                        'token_ranks': rec.get('token_ranks'), 'rows': rec.get('rows'),
                        'row_layout': 'observed' if rec.get('rows') is not None else 'not_recorded'})
    out.sort(key=lambda f: (f['rank'], f['counter'], f['ref']))
    return {'forwards': out[:2048], 'count': len(out), 'truncated': len(out) > 2048, 'basis': 'observed',
            'verdict': 'not_evaluated',
            'note': 'rows can be passed as token_ids to vectors.compare (rows then pair by [seq_id, position]).'}


RL_TOOLS = {fn.__name__: fn for fn in (rl_records_summary, rl_decompose, rl_weight_sync_check, rl_replay_request,
                                       rl_forwards)}
