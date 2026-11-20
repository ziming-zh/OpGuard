"""RL harness records and the vectors.* / rl.* fact actions, on synthetic records and traces.

    python3 harness/tests/test_rl_actions.py
"""
import base64
import json
from pathlib import Path
import random
import struct
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from opguard_harness.rl_records import (RecordWriter, decode_logprob, encode_logprob, read_records,
                                        validate_record, vectors_from_bytes, weights_fingerprint)
from opguard_harness.tools import TOOLS, dispatch

rng = random.Random(7)


def f32(values):
    raw = struct.pack('<%df' % len(values), *values)
    return {'dtype': 'float32', 'encoding': 'base64-le', 'count': len(values), 'data': base64.b64encode(raw).decode()}


def xor(values):
    acc = 0
    for v in values:
        acc ^= v
    return acc


def v1(values, width):
    return {'scheme': 'lastdim-u32le-v1', 'vector_bytes': width, 'count': len(values), 'encoding': 'base64-u32le',
            'data': base64.b64encode(struct.pack('<%dI' % len(values), *values)).decode()}


def values_of(obj):
    raw = base64.b64decode(obj['data'])
    return list(struct.unpack('<%dI' % (len(raw) // 4), raw))


class Base(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.dir = Path(self.temp.name)

    def call(self, name, **arguments):
        result = dispatch(self.dir / 'store', name, arguments)
        if name not in ('rl_records_summary', 'rl.records_summary'):
            self.assertEqual(result['verdict'], 'not_evaluated')
        return result


class RecordTests(Base):
    def test_round_trip_and_exact_logprob_bits(self):
        path = self.dir / 'rank0.jsonl'
        with RecordWriter(path) as w:
            w.forward(counter=10, rank=0, tokens=3, token_ranks=[4, 5, 6])
            w.rollout_batch('b', [{'seq_id': 'A', 'token_ids': [5, 6], 'positions': [0, 1], 'policy_version': 2,
                                   'logprob': f32([-0.1, -2.5])}], composition=[['A']], engine_flags={'cuda_graph': True})
            w.trainer_logprob('b', 2, [{'seq_id': 'A', 'positions': [0, 1], 'logprob': [-0.1, 1e-300]}])
            w.snapshot(2, '/ckpt/step-2')
        rows = read_records(path)
        self.assertEqual([r['type'] for _, r in rows], ['forward', 'rollout_batch', 'trainer_logprob', 'snapshot'])
        self.assertTrue(all(r['schema'] == 'opguard.rl.v1' for _, r in rows) and rows[0][0].endswith(':1'))
        seq = rows[1][1]['sequences'][0]
        self.assertEqual(seq['policy_version'], [2, 2])                       # one int is expanded per token
        got = decode_logprob(seq['logprob'])
        self.assertEqual(got[0][1], struct.unpack('<I', struct.pack('<f', -0.1))[0])   # float32 bits kept
        self.assertEqual(got[1][2], -2.5)
        exact = decode_logprob(rows[2][1]['sequences'][0]['logprob'])        # list -> float64, exact
        self.assertEqual([v for _, _, v in exact], [-0.1, 1e-300])
        self.assertEqual(decode_logprob(encode_logprob([0.5, -0.0]))[1][1], 1 << 63)   # -0.0 keeps its sign bit

    def test_validation_errors(self):
        good = {'schema': 'opguard.rl.v1', 'type': 'rollout_batch', 'batch_id': 'b', 'composition': None,
                'engine_flags': None, 'sequences': [{'seq_id': 'A', 'token_ids': [1, 2], 'positions': [0, 1],
                                                     'policy_version': [1, 1], 'logprob': f32([-1.0, -2.0])}]}
        validate_record(good)
        bad_seq = lambda **kw: {**good, 'sequences': [{**good['sequences'][0], **kw}]}
        for bad in ({**good, 'schema': 'other'}, {**good, 'type': 'loss'}, bad_seq(policy_version=[1]),
                    bad_seq(logprob=f32([-1.0])), bad_seq(positions=[0, 0]), bad_seq(token_ids=[1, 'x']),
                    {**good, 'composition': [['B']]}, {**good, 'sequences': good['sequences'] * 2},
                    {'schema': 'opguard.rl.v1', 'type': 'forward', 'counter': 1, 'rank': 0, 'tokens': 2, 'token_ranks': [1]},
                    {'schema': 'opguard.rl.v1', 'type': 'weight_sync', 'version': 1, 'trainer': [{'name': 'w'}], 'rollout': []},
                    {'schema': 'opguard.rl.v1', 'type': 'snapshot', 'version': 1, 'location': ''}):
            with self.assertRaises(ValueError, msg=bad):
                validate_record(bad)
        path = self.dir / 'bad.jsonl'
        path.write_text(json.dumps(good) + '\n' + json.dumps({**good, 'type': 'x'}) + '\n')
        with self.assertRaisesRegex(ValueError, 'bad.jsonl:2'):
            read_records(path)
        self.assertEqual(len(read_records(path, strict=False)[1]), 3)
        with self.assertRaises(ValueError), RecordWriter(self.dir / 'w.jsonl') as w:
            w.forward(counter=1, rank=0, tokens=0)
        out = subprocess.run([sys.executable, '-m', 'opguard_harness.rl_records', str(path)], capture_output=True, text=True,
                             cwd=Path(__file__).resolve().parents[1])
        self.assertEqual((out.returncode, json.loads(out.stdout)['records']), (1, 1))


class DecomposeTests(Base):
    """Trainer is at version 4. A: version 4 (staleness-zero points). B: version 3, trainer at 3 recorded for
    position 0 only. C: tokens of versions 3 and 4 (mixed). D: rollout float64, trainer float32."""

    def records(self):
        path = self.dir / 'rl.jsonl'
        with RecordWriter(path) as w:
            w.rollout_batch('b1', [
                {'seq_id': 'A', 'token_ids': [1, 2, 3], 'positions': [0, 1, 2], 'policy_version': 4,
                 'logprob': f32([-1.5, -0.25, -2.0])},
                {'seq_id': 'B', 'token_ids': [4, 5], 'positions': [0, 1], 'policy_version': 3, 'logprob': f32([-1.0, -3.0])},
                {'seq_id': 'C', 'token_ids': [6, 7], 'positions': [0, 1], 'policy_version': [3, 4], 'logprob': f32([-1.0, -1.0])},
                {'seq_id': 'D', 'token_ids': [8], 'positions': [0], 'policy_version': 4, 'logprob': [-0.5]}])
            w.trainer_logprob('b1', 4, [{'seq_id': 'A', 'positions': [0, 1, 2], 'logprob': f32([-1.5, -0.25, -2.0009765625])},
                                        {'seq_id': 'B', 'positions': [0, 1], 'logprob': f32([-1.125, -2.5])},
                                        {'seq_id': 'C', 'positions': [0, 1], 'logprob': f32([-1.0, -1.0])},
                                        {'seq_id': 'D', 'positions': [0], 'logprob': f32([-0.5])}])
            w.trainer_logprob('b1', 3, [{'seq_id': 'B', 'positions': [0], 'logprob': f32([-1.0])}])
        return [str(path)]

    def test_exact_split_with_sources(self):
        rec = self.call('rl.decompose', records=self.records(), batch_id='b1', trainer_version=4, policy_version_override=None)
        tok = {(t['seq_id'], t['index']): t for t in rec['tokens']}
        a0, a2 = tok[('A', 0)], tok[('A', 2)]
        self.assertTrue(a0['numeric']['zero'] and a0['staleness']['zero'] and a0['staleness_zero_point'])
        self.assertEqual(a2['numeric']['value'], 0.0009765625)               # numeric needs no replay at v == c
        self.assertFalse(a2['numeric']['zero'])
        self.assertIn(':2', a2['sources']['trainer_at_policy_version'])
        self.assertTrue(a2['sources']['rollout'].endswith(':1'))
        b0 = tok[('B', 0)]
        self.assertTrue(b0['numeric']['zero'])
        self.assertEqual(b0['staleness']['value'], 0.125)                    # -1.0 - (-1.125)
        self.assertEqual(set(b0['sources']), {'rollout', 'trainer_at_policy_version', 'trainer_at_current_version'})
        b1 = tok[('B', 1)]                                                   # trainer at version 3 missing here
        self.assertIsNone(b1['numeric'])
        self.assertIn('no trainer_logprob at version 3', b1['numeric_missing'])
        self.assertTrue(all(tok[('C', i)]['mixed_version'] and tok[('C', i)]['numeric'] is None
                            and tok[('C', i)]['not_split'] == 'mixed_version_sequence' for i in (0, 1)))
        d = tok[('D', 0)]['numeric']                                          # equal values, different dtypes
        self.assertEqual((d['value'], d['zero'], d['dtype_mismatch']), (0.0, False, True))
        s = rec['summary']
        self.assertEqual((s['tokens'], s['mixed_version_tokens'], s['numeric_zero'], s['numeric_nonzero'],
                          s['numeric_missing'], s['staleness_zero_points']), (8, 2, 3, 2, 1, 4))
        self.assertEqual(rec['versions_without_trainer_logprob'], [3])

    def test_exact_arithmetic_and_bits(self):
        path = self.dir / 'x.jsonl'
        with RecordWriter(path) as w:
            w.rollout_batch(7, [{'seq_id': 1, 'token_ids': [1, 2], 'positions': [0, 1], 'policy_version': 1,
                                 'logprob': [1e30, 0.0]}])
            w.trainer_logprob(7, 1, [{'seq_id': 1, 'positions': [0, 1], 'logprob': [-1e-30, -0.0]}])
        rec = self.call('rl_decompose', records=[str(path)], batch_id=7, trainer_version=1, policy_version_override=None)
        big, signed = rec['tokens'][0]['numeric'], rec['tokens'][1]['numeric']
        self.assertEqual((big['value'], big['exact'], big['zero']), (1e30, False, False))    # rounded, said so
        self.assertEqual((signed['value'], signed['zero']), (0.0, False))                    # 0.0 vs -0.0: bits differ
        with self.assertRaisesRegex(ValueError, 'no rollout_batch'):
            self.call('rl_decompose', records=[str(path)], batch_id=8, trainer_version=1, policy_version_override=None)

    def test_summary_lists_records(self):
        summary = self.call('rl_records_summary', records=self.records())
        self.assertEqual(summary['by_type'], {'rollout_batch': 1, 'trainer_logprob': 2})
        self.assertEqual(summary['batches']['b1']['rollout_batch'][0]['policy_versions'], [3, 4])
        self.assertEqual(sorted(v['version'] for v in summary['batches']['b1']['trainer_logprob_versions']), [3, 4])


def rows(n, width):
    return [bytes(rng.getrandbits(8) for _ in range(width)) for _ in range(n)]


def entry(name, raw_rows, dtype='float32'):
    width = len(raw_rows[0])
    return {'name': name, 'shape': [len(raw_rows), width // 4], 'dtype': dtype,
            'xor_vectors': vectors_from_bytes(b''.join(raw_rows), width)}


class WeightSyncTests(Base):
    def write(self, trainer, rollout, version=5):
        path = self.dir / f'sync{version}.jsonl'
        with RecordWriter(path) as w:
            w.weight_sync(version, trainer, rollout)
        return [str(path)]

    def test_repacked_rows_are_not_differences(self):
        q, k, v = rows(4, 32), rows(2, 32), rows(2, 32)
        qkv = [k[1], q[2], v[0], q[0], k[0], q[3], v[1], q[1]]              # concatenated and permuted
        rec = self.call('rl.weight_sync_check', records=self.write(
            [entry('q', q), entry('k', k), entry('v', v)], [entry('qkv', qkv)]), version=5)
        self.assertEqual(rec['summary'], {'trainer_rows': 8, 'trainer_unmatched_rows': 0,
                                          'rollout_rows': 8, 'rollout_unmatched_rows': 0})
        self.assertTrue(all(p['status'] == 'all_rows_matched' for p in rec['trainer'] + rec['rollout']))

    def test_one_flipped_bit_names_the_row_on_each_side(self):
        q, k = rows(4, 32), rows(2, 32)
        flipped = bytearray(q[2]); flipped[13] ^= 0x04
        packed = [k[0], q[0], q[1], bytes(flipped), k[1], q[3]]
        rec = self.call('rl_weight_sync_check', records=self.write([entry('q', q), entry('k', k)], [entry('qk', packed)]),
                        version=5)
        self.assertEqual([(p['name'], p['unmatched_rows']) for p in rec['trainer']], [('q', [2]), ('k', [])])
        self.assertEqual([(p['name'], p['unmatched_rows']) for p in rec['rollout']], [('qk', [3])])

    def test_zero_rows_dtype_and_width_are_reported(self):
        zero = [bytes(32)]
        rec = self.call('rl_weight_sync_check', records=self.write(
            [entry('a', zero + rows(1, 32)), entry('b', rows(2, 32))],
            [entry('a', zero + rows(1, 32)), entry('b', rows(1, 64), dtype='bfloat16')]), version=5)
        self.assertEqual(rec['trainer'][0]['zero_rows'], 1)
        self.assertEqual(rec['trainer'][0]['unmatched_rows'], [1])          # the zero row matched, the other did not
        self.assertEqual(rec['rollout'][1]['unmatched_rows'], [0])          # other width and dtype: never a match
        with self.assertRaisesRegex(ValueError, 'no weight_sync'):
            self.call('rl_weight_sync_check', records=self.write([], []), version=6)


class ReplayTests(DecomposeTests):
    def test_request_is_built_never_run(self):
        records = self.records()
        with RecordWriter(self.dir / 'snap.jsonl') as w:
            w.snapshot(3, '/ckpt/v3')
        records.append(str(self.dir / 'snap.jsonl'))
        out = self.dir / 'req' / 'replay.json'
        rec = self.call('rl.replay_request', records=records, batch_id='b1', version=3, side='trainer',
                        composition='batch_invariant', capture={'vector_fingerprints': True, 'env': {'X': 1}},
                        output=str(out))
        self.assertFalse(rec['executed'])
        self.assertIn('localizes implementation differences between the two sides, not the specific error', rec['note'])
        self.assertEqual(rec['reproduces'], 'implementation_differences_only')
        req = json.loads(out.read_text())
        self.assertEqual((req['weights']['location'], req['engine_flags']['batch_invariant']), ('/ckpt/v3', True))
        self.assertEqual(req['capture_env'], {'X': '1', 'OPGUARD_XOR_ROWS': '1'})
        self.assertIsNone(req['forward_composition'])
        with self.assertRaisesRegex(ValueError, 'fresh'):
            self.call('rl_replay_request', records=records, batch_id='b1', version=3, side='trainer',
                      composition='batch_invariant', capture=None, output=str(out))
        with self.assertRaisesRegex(ValueError, 'composition was not recorded'):
            self.call('rl_replay_request', records=records, batch_id='b1', version=3, side='rollout',
                      composition='original', capture=None, output=None)
        rec = self.call('rl_replay_request', records=records, batch_id='b1', version=4, side='rollout',
                        composition='batch_invariant', capture=None, output=None)
        self.assertFalse(rec['weights_located'])
        self.assertIn('No snapshot record', rec['note'])


def op(run, counter, outputs, rank=0):
    return {'ph': 'X', 'cat': 'opguard', 'name': 'op', 'ts': counter, 'dur': 1, 'pid': 1 if run == 'run1' else 2,
            'tid': 1, 'args': {'run': run, 'rank': rank, 'status': 'ok', 'counter': counter, 'fn': f'<f{counter}>',
                               'io': {'inputs': [], 'outputs': outputs}}}


class VectorTests(Base):
    """run1 writes [4, 2, 8] bf16 head vectors in v1; run2 writes the same tokens as [4, 16] rows in the
    legacy encoding, with token 2 changed. Token 0 and 3 share their head-1 fingerprint; token 1 head 0 is zero."""

    def artifact(self):
        heads = [[rng.getrandbits(32) for _ in range(2)] for _ in range(4)]
        heads[3][1] = heads[0][1]
        heads[1][0] = 0
        flat = [h for t in heads for h in t]
        legacy_rows = [xor(t) for t in heads]
        legacy_rows[2] ^= 0x100
        signed = [v - (1 << 32) if v >> 31 else v for v in legacy_rows]
        changed = [list(t) for t in heads]
        changed[2][1] ^= 0x8
        events = [op('run1', 1, [{'shape': [4, 2, 8], 'dtype': 'torch.bfloat16', 'xor_vectors': v1(flat, 16)}]),
                  op('run2', 1, [{'shape': [4, 16], 'dtype': 'torch.bfloat16', 'xor_rows': signed, 'xor_row_bytes': 32}]),
                  op('run2', 2, [{'shape': [4, 2, 8], 'dtype': 'torch.bfloat16', 'xor_rows': [h - (1 << 32) if h >> 31 else h
                                 for t in changed for h in t], 'xor_row_bytes': 16},
                                 {'shape': [4, 3], 'dtype': 'torch.bfloat16', 'xor_vectors_skipped': 'vector_bytes_not_multiple_of_4'}])]
        path = self.dir / 'trace.json'
        path.write_text(json.dumps({'traceEvents': events}))
        from opguard_harness.trace_reader import Trace
        trace = Trace(str(path))
        return str(path), [trace.ref(i) for i in range(3)], heads, legacy_rows

    def test_describe_both_encodings(self):
        path, refs, heads, legacy_rows = self.artifact()
        a = self.call('vectors.describe', artifact=path, event_ref=refs[0], first_n=3)['slots'][0]
        self.assertEqual((a['encoding'], a['vector_bytes'], a['count'], a['zero_vectors']), ('lastdim-u32le-v1', 16, 8, 1))
        self.assertEqual([x['fingerprint'] for x in a['first']], [heads[0][0], heads[0][1], 0])
        self.assertTrue(a['first'][2]['zero'])
        b = self.call('vectors_describe', artifact=path, event_ref=refs[1], first_n=4)['slots'][0]
        self.assertEqual((b['encoding'], b['vector_bytes'], [x['fingerprint'] for x in b['first']]),
                         ('legacy-xor_rows', 32, legacy_rows))                # unsigned after masking
        skipped = self.call('vectors_describe', artifact=path, event_ref=refs[2], first_n=0)['slots'][1]
        self.assertEqual((skipped['status'], skipped['reason']), ('skipped', 'vector_bytes_not_multiple_of_4'))

    def test_lookup_shared_and_zero_fingerprints(self):
        path, refs, heads, _ = self.artifact()
        shared = self.call('vectors_lookup', artifact=path, fingerprint=None, event_ref=refs[0], slot=['outputs', 0],
                           vector=1, scope={'runs': ['run1']})
        self.assertEqual(sorted(o['vector'] for o in shared['occurrences']), [1, 7])   # tokens 0 and 3 share it
        self.assertEqual(sum(o['is_query'] for o in shared['occurrences']), 1)
        everywhere = self.call('vectors_lookup', artifact=path, fingerprint=heads[0][1], event_ref=None, slot=None,
                               vector=None, scope=None)
        self.assertEqual(everywhere['occurrence_count'], 4)                 # run2 counter 2 holds it twice too
        zero = self.call('vectors_lookup', artifact=path, fingerprint=0, event_ref=None, slot=None, vector=None, scope=None)
        self.assertEqual((zero['occurrences'], zero['zero_fingerprint_match_count'], zero['query']['zero']), ([], 2, True))
        with self.assertRaises(ValueError):
            self.call('vectors_lookup', artifact=path, fingerprint=1, event_ref=refs[0], slot=None, vector=None, scope=None)

    def test_coverage_against_the_other_run(self):
        path, refs, _, _ = self.artifact()
        cov = self.call('vectors.coverage', artifact=path, event_ref=refs[0], slot=['outputs', 0], scope=None)
        self.assertEqual(cov['scope']['runs'], ['run2'])
        self.assertEqual([v['state'] for v in cov['vectors']],
                         ['occurs', 'occurs', 'zero', 'occurs', 'occurs', 'absent', 'occurs', 'occurs'])
        self.assertEqual(cov['vectors'][5]['first'], [])                     # token 2 head 1 changed on run2

    def test_compare_by_declared_token_dims(self):
        path, refs, _, _ = self.artifact()
        merged = self.call('vectors_compare', artifact=path, left_ref=refs[0], left_slot=['outputs', 0],
                           left_token_dims=1, left_token_ids=None, right_ref=refs[1], right_slot=['outputs', 0],
                           right_token_dims=1, right_token_ids=None)
        self.assertEqual((merged['unit'], [r['state'] for r in merged['rows']]),
                         ('merged_token_rows', ['equal', 'equal', 'differ', 'equal']))
        self.assertIsNone(merged['rows'][2]['differing_vectors'])           # widths differ: no vector indices
        heads = self.call('vectors_compare', artifact=path, left_ref=refs[0], left_slot=['outputs', 0],
                          left_token_dims=1, left_token_ids=[10, 11, 12, 13], right_ref=refs[2], right_slot=['outputs', 0],
                          right_token_dims=1, right_token_ids=[13, 12, 11, 10])
        by_token = {r['token']: r for r in heads['rows']}                   # rows pair by declared token id
        self.assertEqual(by_token[12]['state'], 'differ')
        self.assertEqual(heads['summary']['equal'], 0)                      # right ids reverse the rows
        same = self.call('vectors_compare', artifact=path, left_ref=refs[0], left_slot=['outputs', 0], left_token_dims=1,
                         left_token_ids=None, right_ref=refs[2], right_slot=['outputs', 0], right_token_dims=1,
                         right_token_ids=None)
        self.assertEqual([r['differing_vectors'] for r in same['rows']], [[], [], [1], []])
        self.assertEqual(same['summary']['equal'], 3)
        wrong = self.call('vectors_compare', artifact=path, left_ref=refs[0], left_slot=['outputs', 0], left_token_dims=2,
                          left_token_ids=None, right_ref=refs[1], right_slot=['outputs', 0], right_token_dims=1,
                          right_token_ids=None)
        self.assertEqual(wrong['status'], 'incompatible')

    def test_merge_rule(self):
        raw = b''.join(rows(6, 16))
        vecs = values_of(vectors_from_bytes(raw, 16))
        words = [w for (w,) in struct.iter_unpack('<I', raw)]
        self.assertEqual(xor(vecs), xor(words))                             # flat-u32-le-v1, one region
        self.assertEqual([xor(vecs[2 * t:2 * t + 2]) for t in range(3)], values_of(vectors_from_bytes(raw, 32)))


class GapTests(Base):
    """What a real RL loop needed: prompts in replays, observed row layouts, weights per batch,
    caller-supplied versions, zero-only equality, event search and call-count facts."""

    def test_prompts_rows_and_runnable_replay(self):
        path = self.dir / 'r.jsonl'
        with RecordWriter(path) as w:
            w.forward(counter=3, rank=0, tokens=3, rows=[['A', 0], ['A', 1], ['B', 0]])
            w.forward(counter=4, rank=0, tokens=2, rows=[['A', 2], None])                 # a padding row
            w.rollout_batch('b', [{'seq_id': 'A', 'prompt_ids': [9, 8], 'token_ids': [5], 'positions': [2],
                                   'policy_version': 1, 'logprob': f32([-1.0])}], composition=[['A']])
            w.snapshot(1, '/ckpt/1')
        with self.assertRaises(ValueError), RecordWriter(self.dir / 'x.jsonl') as w:
            w.forward(counter=1, rank=0, tokens=2, rows=[['A', 0], ['A', 0]])
        fw = self.call('rl.forwards', records=[str(path)], rank=0, counter_min=None, counter_max=None)['forwards']
        self.assertEqual((fw[0]['rows'], fw[0]['row_layout']), ([['A', 0], ['A', 1], ['B', 0]], 'observed'))
        self.assertEqual(fw[1]['rows'], [['A', 2], None])
        rq = self.call('rl_replay_request', records=[str(path)], batch_id='b', version=1, side='rollout',
                       composition='original', capture={'vector_fingerprints': True}, output=None)
        self.assertEqual(rq['request']['sequences'][0]['prompt_ids'], [9, 8])
        self.assertTrue(rq['runnable_alone'])
        self.assertEqual(rq['request']['capture_path']['assumes'], 'opguard_tracer')

    def test_weights_fingerprint_names_the_sync_it_ran(self):
        q, k = rows(4, 32), rows(2, 32)
        v1w, v2w = [entry('q', q), entry('k', k)], [entry('q', q[:3] + rows(1, 32)), entry('k', k)]
        packed = [entry('qk', [k[1], q[0], q[3], q[1], k[0], q[2]])]
        fp = weights_fingerprint(packed)
        self.assertEqual(fp, weights_fingerprint(v1w))                      # packing does not change it
        path = self.dir / 'w.jsonl'
        with RecordWriter(path) as w:
            w.weight_sync(1, v1w, packed)
            w.weight_sync(2, v2w, v2w)
            w.rollout_batch(5, [{'seq_id': 0, 'token_ids': [1], 'positions': [0], 'policy_version': 2,
                                 'logprob': f32([-1.0])}], weights_fingerprint=fp)     # labelled 2, ran on 1
            w.trainer_logprob(5, 1, [{'seq_id': 0, 'positions': [0], 'logprob': f32([-1.0])}])
            w.trainer_logprob(5, 2, [{'seq_id': 0, 'positions': [0], 'logprob': f32([-1.5])}])
        d = self.call('rl.decompose', records=[str(path)], batch_id=5, trainer_version=2, policy_version_override=None)
        wc = d['weights_check']
        self.assertEqual((wc['labelled_versions'], wc['rollout_sync_versions_matching'], wc['first_labelled_version_matches']),
                         ([2], [1], False))
        self.assertEqual((d['policy_version_source'], d['summary']['numeric_nonzero']), ('recorded', 1))
        h = self.call('rl_decompose', records=[str(path)], batch_id=5, trainer_version=2, policy_version_override=1)
        t = h['tokens'][0]
        self.assertEqual((h['policy_version_source'], t['policy_version'], t['recorded_policy_version']),
                         ('caller_supplied', 1, 2))
        self.assertTrue(t['numeric']['zero'])
        self.assertEqual(t['staleness']['value'], 0.5)
        self.assertIn('hypothesis', h['override_note'])

    def test_zero_only_equality_and_event_search(self):
        a, b = [[7, 7], [0, 0]], [[7, 7], [0, 0]]
        io = lambda heads, w: [{'shape': [2, 2, 4], 'dtype': 'torch.float32', 'xor_vectors': v1([x for t in heads for x in t], w)}]
        events = [op('run1', 1, io(a, 16)), op('run2', 1, io(b, 16)),
                  op('run2', 2, [{'shape': [2, 8], 'dtype': 'torch.float32', 'xor_vectors': v1([0, 0], 32)}]),
                  op('run1', 2, [{'shape': [2, 8], 'dtype': 'torch.float32', 'xor_vectors': v1([0, 0], 32)}])]
        events[0]['args']['fn'] = '<model.layers.0.self_attn.o_proj>'
        events[0]['name'] = '0.o_proj#1'
        path = self.dir / 't.json'
        path.write_text(json.dumps({'traceEvents': events}))
        from opguard_harness.trace_reader import Trace
        refs = [Trace(str(path)).ref(i) for i in range(4)]
        args = dict(artifact=str(path), left_slot=['outputs', 0], left_token_dims=1, left_token_ids=None,
                    right_slot=['outputs', 0], right_token_dims=1, right_token_ids=None)
        heads = self.call('vectors_compare', left_ref=refs[0], right_ref=refs[1], **args)
        self.assertEqual([r['state'] for r in heads['rows']], ['equal', 'equal_zero_only'])   # [7, 7] XORs to 0 but its vectors are not zero
        merged = self.call('vectors_compare', left_ref=refs[0], right_ref=refs[2], **args)
        self.assertEqual([r['state'] for r in merged['rows']], ['equal_zero_only', 'equal_zero_only'])
        self.assertEqual(merged['summary']['equal'], 0)
        padded = self.call('vectors_compare', left_ref=refs[0], right_ref=refs[1], **{**args, 'left_token_ids': [['s', 0], None],
                                                                                    'right_token_ids': [['s', 0], ['s', 1]]})
        self.assertEqual([(r['token'], r['state']) for r in padded['rows']], [(['s', 0], 'equal'), (['s', 1], 'missing_left')])
        found = dispatch(self.dir, 'find_events', {'artifact': str(path), 'query': 'self_attn.o_proj', 'run': None,
                                                     'rank': None, 'counter_min': None, 'counter_max': None,
                                                     'cursor': 0, 'limit': 10})
        self.assertEqual((found['match_count'], found['events'][0]['event_ref'], found['events'][0]['slots'][0]['vectors']),
                         (1, refs[0], True))
        page = dispatch(self.dir, 'find_events', {'artifact': str(path), 'query': None, 'run': 'run2', 'rank': 0,
                                                    'counter_min': 2, 'counter_max': None, 'cursor': 0, 'limit': 1})
        self.assertEqual((page['match_count'], page['next_cursor']), (1, None))
        events.append(op('run2', 3, io(b, 16)))                             # run2 now calls f1-like ops more often
        path.write_text(json.dumps({'traceEvents': events}))
        fd = dispatch(self.dir, 'first_diff', {'artifact': str(path)})
        seq = fd['call_sequences'][0]
        self.assertEqual((seq['operations'], seq['functions_with_different_call_counts']), ({'run1': 2, 'run2': 3}, 3))
        self.assertIn('not evidence of agreement', fd['no_observed_diff_note'])


class ContractTests(Base):
    def test_schemas_and_aliases(self):
        names = {t['name'] for t in TOOLS}
        self.assertTrue({'vectors_describe', 'vectors_lookup', 'vectors_coverage', 'vectors_compare', 'rl_decompose',
                         'rl_weight_sync_check', 'rl_replay_request', 'rl_records_summary', 'rl_forwards',
                         'find_events'} <= names)
        self.assertTrue(all(set(t['parameters']['required']) == set(t['parameters']['properties']) for t in TOOLS))
        with self.assertRaises(ValueError):
            dispatch(self.dir, 'rl.decompose', {'records': []})
        out = subprocess.run([sys.executable, '-m', 'opguard_harness', 'call', 'rl.records_summary'],
                             input=json.dumps({'records': [str(self.dir / 'missing.jsonl')]}), capture_output=True,
                             text=True, cwd=Path(__file__).resolve().parents[1])
        self.assertEqual((out.returncode, json.loads(out.stdout)['ok']), (1, False))


if __name__ == '__main__':
    unittest.main()
