"""RL harness records: a small writer for training and rollout processes, a reader and validation.

One JSONL file of records per process (one per rank is simplest; files are read
together). Every line is one record with ``"schema": "opguard.rl.v1"`` and a
``type``: forward, rollout_batch, trainer_logprob, weight_sync or snapshot
(docs/RLAlignment.md). Records are facts observed at known code points; nothing
here infers them.

Standard library only. ``torch`` and ``numpy`` are used when the values passed in
are tensors or arrays (a training process already has them); OpGuard's native
kernel is used for weight fingerprints when ``opguard`` is importable.

    from opguard_harness.rl_records import RecordWriter, param_entries
    with RecordWriter("/runs/rl/records-rank0.jsonl") as rec:
        rec.forward(counter=c, rank=0, tokens=T)
        rec.rollout_batch(batch_id, sequences, composition, engine_flags)
        rec.trainer_logprob(batch_id, policy_version, sequences, engine_flags)
        rec.weight_sync(version, trainer=model.named_parameters(), rollout=entries_from_engine)
        rec.snapshot(version, location)

Log-probabilities are stored as their exact bits: ``{"dtype": "float32" | "float64",
"encoding": "base64-le", "count", "data"}``. float16 and bfloat16 values are
widened to float32, which is exact (``source_dtype`` keeps the original). A plain
JSON list of numbers is also accepted by the reader, as float64 values.
"""
import base64
import json
import os
from pathlib import Path
import struct
import sys
import threading

SCHEMA = 'opguard.rl.v1'
TYPES = ('forward', 'rollout_batch', 'trainer_logprob', 'weight_sync', 'snapshot')
VECTOR_SCHEME = 'lastdim-u32le-v1'
WEIGHTS_SCHEME = 'xor-of-rows-v1'
_FLOAT = {'float32': ('<f', '<I', 4), 'float64': ('<d', '<Q', 8)}


# ------------------------------------------------------------ log-probabilities
def encode_logprob(values):
    """Exact log-probabilities of one sequence: a 1-D torch tensor, numpy array or list of floats."""
    source = None
    if hasattr(values, 'detach'):                                    # torch.Tensor
        source = str(values.dtype).replace('torch.', '')
        t = values.detach().reshape(-1).cpu()
        if source in ('float16', 'bfloat16'):
            t = t.float()
        if source not in ('float16', 'bfloat16', 'float32', 'float64'):
            raise ValueError(f'log-probabilities must be floating point, not {source}')
        values = t.numpy()
    if hasattr(values, 'dtype') and hasattr(values, 'tobytes'):     # numpy.ndarray
        name = str(values.dtype)
        source = source or name
        if name == 'float16':
            values, name = values.astype('float32'), 'float32'
        if name not in _FLOAT:
            raise ValueError(f'log-probabilities must be float16/32/64, not {name}')
        raw = values.reshape(-1).astype(_FLOAT[name][0], copy=False).tobytes()
    else:
        values = list(values)
        if any(type(v) not in (int, float) for v in values):
            raise ValueError('log-probabilities must be numbers')
        name, raw = 'float64', struct.pack('<%dd' % len(values), *values)
    item = {'dtype': name, 'encoding': 'base64-le', 'count': len(raw) // _FLOAT[name][2],
            'data': base64.b64encode(raw).decode('ascii')}
    if source and source != name:
        item['source_dtype'] = source
    return item


def decode_logprob(item):
    """[(dtype, bits, value)] per token, from the encoded object or a JSON list of numbers."""
    if isinstance(item, list):
        if any(type(v) not in (int, float) for v in item):
            raise ValueError('logprob list must hold numbers')
        return [('float64', struct.unpack('<Q', struct.pack('<d', float(v)))[0], float(v)) for v in item]
    if (not isinstance(item, dict) or item.get('encoding') != 'base64-le' or item.get('dtype') not in _FLOAT
            or not isinstance(item.get('data'), str)):
        raise ValueError('logprob must be {"dtype": float32|float64, "encoding": "base64-le", "data"} or a list')
    value_fmt, bits_fmt, size = _FLOAT[item['dtype']]
    raw = base64.b64decode(item['data'], validate=True)
    if len(raw) % size or item.get('count') != len(raw) // size:
        raise ValueError('logprob count does not match its data')
    return [(item['dtype'], struct.unpack(bits_fmt, raw[i:i + size])[0], struct.unpack(value_fmt, raw[i:i + size])[0])
            for i in range(0, len(raw), size)]


# ------------------------------------------------------------ weight fingerprints
def vectors_from_bytes(raw, vector_bytes):
    """lastdim-u32le-v1 object from a tensor's bytes in logical contiguous order."""
    if type(vector_bytes) is not int or vector_bytes < 4 or vector_bytes % 4 or len(raw) % vector_bytes:
        raise ValueError('vector_bytes must be a multiple of 4 that divides the byte count')
    try:
        import numpy as np
        words = np.frombuffer(raw, dtype='<u4').reshape(-1, vector_bytes // 4)
        out = np.bitwise_xor.reduce(words, axis=1).astype('<u4').tobytes() if len(words) else b''
    except ImportError:
        values = []
        for k in range(0, len(raw), vector_bytes):
            acc = 0
            for (w,) in struct.iter_unpack('<I', raw[k:k + vector_bytes]):
                acc ^= w
            values.append(acc)
        out = struct.pack('<%dI' % len(values), *values)
    return {'scheme': VECTOR_SCHEME, 'vector_bytes': vector_bytes, 'count': len(out) // 4,
            'encoding': 'base64-u32le', 'data': base64.b64encode(out).decode('ascii')}


def tensor_xor_vectors(tensor):
    """Per-row (last-dimension vector) fingerprints of a torch tensor, as an xor_vectors object."""
    try:
        from opguard.tracer.xor_signature import encode_xor_vectors, xor_vector_stats
    except Exception:
        t = tensor.detach()
        width = (t.shape[-1] if t.dim() else 1) * t.element_size()
        if t.numel() == 0 or width % 4:
            raise ValueError('no vector fingerprints: empty tensor or vector width not a multiple of 4 bytes')
        import torch
        raw = t.contiguous().reshape(-1).view(torch.uint8).cpu().numpy().tobytes()
        return vectors_from_bytes(raw, width)
    stats = xor_vector_stats(tensor)
    if 'xor_vectors' not in stats:
        raise ValueError(f'no vector fingerprints: {stats.get("xor_vectors_skipped")}')
    return encode_xor_vectors(stats['xor_vectors'])


def weights_fingerprint(params):
    """XOR of every row fingerprint of a set of weights: {scheme, value, rows}. params are entries from
    param_entries() or (name, tensor) pairs / a dict. XOR merges, so any packing of the same weight bytes
    (rows permuted, split or concatenated) gives the same value. Record it on rollout_batch, computed from
    the weights the engine actually holds when the batch runs."""
    if not (isinstance(params, list) and all(isinstance(p, dict) for p in params)):
        params = param_entries(params)
    value = rows = 0
    for p in params:
        raw = base64.b64decode(p['xor_vectors']['data'])
        for (v,) in struct.iter_unpack('<I', raw):
            value ^= v
        rows += len(raw) // 4
    return {'scheme': WEIGHTS_SCHEME, 'value': value, 'rows': rows}


def param_entries(named_tensors):
    """[{name, shape, dtype, xor_vectors}] for one side of a weight sync, from (name, tensor) pairs or a dict.
    Run it in the process that holds those tensors and pass the list to RecordWriter.weight_sync."""
    items = named_tensors.items() if hasattr(named_tensors, 'items') else named_tensors
    return [{'name': name, 'shape': list(t.shape), 'dtype': str(t.dtype).replace('torch.', ''),
             'xor_vectors': tensor_xor_vectors(t)} for name, t in items]


# ------------------------------------------------------------ validation
def _int(v, name, minimum=0):
    if type(v) is not int or v < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')


def _ints(v, name):
    if not isinstance(v, list) or any(type(x) is not int for x in v):
        raise ValueError(f'{name} must be a list of integers')


def _key(v, name):
    if type(v) not in (int, str) or v == '':
        raise ValueError(f'{name} must be a nonempty string or an integer')


def _flags(v, name='engine_flags'):
    if v is not None and not isinstance(v, dict):
        raise ValueError(f'{name} must be an object or null')


def _sequences(seqs, per_token_version):
    if not isinstance(seqs, list) or not seqs:
        raise ValueError('sequences must be a nonempty list')
    seen = set()
    for s in seqs:
        if not isinstance(s, dict):
            raise ValueError('each sequence must be an object')
        _key(s.get('seq_id'), 'seq_id')
        if s['seq_id'] in seen:
            raise ValueError(f'duplicate seq_id {s["seq_id"]!r}')
        seen.add(s['seq_id'])
        _ints(s.get('positions'), 'positions')
        n = len(s['positions'])
        if len(set(s['positions'])) != n:
            raise ValueError(f'sequence {s["seq_id"]!r}: positions repeat')
        if len(decode_logprob(s.get('logprob'))) != n:
            raise ValueError(f'sequence {s["seq_id"]!r}: one logprob per position is required')
        if s.get('prompt_ids') is not None:
            _ints(s['prompt_ids'], 'prompt_ids')
        if per_token_version:
            for name in ('token_ids', 'policy_version'):
                _ints(s.get(name), name)
                if len(s[name]) != n:
                    raise ValueError(f'sequence {s["seq_id"]!r}: {name} needs one entry per position')
    return seen


def validate_record(rec):
    """Raise ValueError if a record does not follow schema opguard.rl.v1."""
    if not isinstance(rec, dict) or rec.get('schema') != SCHEMA or rec.get('type') not in TYPES:
        raise ValueError(f'record needs "schema": "{SCHEMA}" and a type in {TYPES}')
    kind = rec['type']
    if kind == 'forward':
        _int(rec.get('counter'), 'counter'); _int(rec.get('rank'), 'rank'); _int(rec.get('tokens'), 'tokens', 1)
        if rec.get('token_ranks') is not None:
            _ints(rec['token_ranks'], 'token_ranks')
            if len(rec['token_ranks']) != rec['tokens']:
                raise ValueError('token_ranks needs one global token index per token')
        rows = rec.get('rows')
        if rows is not None:
            if (not isinstance(rows, list) or len(rows) != rec['tokens'] or any(
                    r is not None and (not isinstance(r, list) or len(r) != 2 or type(r[0]) not in (int, str)
                                       or type(r[1]) is not int) for r in rows)):
                raise ValueError('rows needs one [seq_id, position] (or null for padding) per token')
            real = [(r[0], r[1]) for r in rows if r is not None]
            if len(set(real)) != len(real):
                raise ValueError('rows repeat a (seq_id, position)')
    elif kind == 'rollout_batch':
        _key(rec.get('batch_id'), 'batch_id')
        ids = _sequences(rec.get('sequences'), True)
        comp = rec.get('composition')
        if comp is not None:
            if not isinstance(comp, list) or any(not isinstance(g, list) for g in comp):
                raise ValueError('composition must be a list of co-batched seq_id lists, or null')
            if any(x not in ids for g in comp for x in g):
                raise ValueError('composition names a seq_id that is not in the batch')
        _flags(rec.get('engine_flags'))
        wf = rec.get('weights_fingerprint')
        if wf is not None and (not isinstance(wf, dict) or wf.get('scheme') != WEIGHTS_SCHEME
                               or type(wf.get('value')) is not int or not 0 <= wf['value'] <= 0xFFFFFFFF
                               or type(wf.get('rows')) is not int):
            raise ValueError(f'weights_fingerprint must be {{"scheme": "{WEIGHTS_SCHEME}", "value", "rows"}} or null')
    elif kind == 'trainer_logprob':
        _key(rec.get('batch_id'), 'batch_id'); _int(rec.get('policy_version'), 'policy_version')
        _sequences(rec.get('sequences'), False)
        _flags(rec.get('engine_flags'))
    elif kind == 'weight_sync':
        _int(rec.get('version'), 'version')
        for side in ('trainer', 'rollout'):
            params = rec.get(side)
            if not isinstance(params, list):
                raise ValueError(f'weight_sync.{side} must be a list of parameters')
            for p in params:
                if (not isinstance(p, dict) or not isinstance(p.get('name'), str) or not isinstance(p.get('dtype'), str)
                        or not isinstance(p.get('xor_vectors'), dict) or p['xor_vectors'].get('scheme') != VECTOR_SCHEME):
                    raise ValueError(f'weight_sync.{side}: each parameter needs name, shape, dtype and xor_vectors')
                _ints(p.get('shape'), 'shape')
    else:
        _int(rec.get('version'), 'version')
        if not isinstance(rec.get('location'), str) or not rec['location']:
            raise ValueError('snapshot.location must name the saved weights or checkpoint step')
    return rec


# ------------------------------------------------------------ writer and reader
class RecordWriter:
    """Append validated records to one JSONL file. Thread-safe; one file per process."""

    def __init__(self, path, fsync=False):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = self.path.open('a', encoding='utf-8')
        self._lock = threading.Lock()
        self._fsync = fsync

    def write(self, record):
        record = validate_record({'schema': SCHEMA, **record})
        line = json.dumps(record, ensure_ascii=False, separators=(',', ':'), allow_nan=False) + '\n'
        with self._lock:
            self._stream.write(line)
            self._stream.flush()
            if self._fsync:
                os.fsync(self._stream.fileno())
        return record

    def forward(self, counter, rank, tokens, token_ranks=None, rows=None):
        """At each model forward entry: the OpGuard counter it precedes, the rank and its token count T.
        rows (optional, observed): the [seq_id, position] each of the T rows holds, in row order;
        null for a padding row that holds no token."""
        rec = {'type': 'forward', 'counter': counter, 'rank': rank, 'tokens': tokens}
        if token_ranks is not None:
            rec['token_ranks'] = [int(x) for x in token_ranks]
        if rows is not None:
            rec['rows'] = [None if r is None else [r[0], int(r[1])] for r in rows]
        return self.write(rec)

    def rollout_batch(self, batch_id, sequences, composition=None, engine_flags=None, weights_fingerprint=None):
        """sequences: [{seq_id, token_ids, positions, policy_version (per token, or one int), logprob,
        prompt_ids (optional: the context tokens before token_ids)}]. weights_fingerprint (optional,
        observed): weights_fingerprint() of the weights the engine held when the batch started."""
        out = []
        for s in sequences:
            s = dict(s)
            n = len(s['positions'])
            if type(s.get('policy_version')) is int:
                s['policy_version'] = [s['policy_version']] * n
            for name in ('token_ids', 'positions', 'policy_version', 'prompt_ids'):
                if s.get(name) is not None:
                    s[name] = [int(x) for x in s[name]]
            s['logprob'] = s['logprob'] if isinstance(s['logprob'], dict) else encode_logprob(s['logprob'])
            out.append(s)
        return self.write({'type': 'rollout_batch', 'batch_id': batch_id, 'sequences': out,
                           'composition': composition, 'engine_flags': engine_flags,
                           **({'weights_fingerprint': weights_fingerprint} if weights_fingerprint else {})})

    def trainer_logprob(self, batch_id, policy_version, sequences, engine_flags=None):
        """sequences: [{seq_id, positions, logprob}] computed by the trainer at policy_version."""
        out = [{'seq_id': s['seq_id'], 'positions': [int(x) for x in s['positions']],
                'logprob': s['logprob'] if isinstance(s['logprob'], dict) else encode_logprob(s['logprob'])}
               for s in sequences]
        return self.write({'type': 'trainer_logprob', 'batch_id': batch_id, 'policy_version': policy_version,
                           'sequences': out, 'engine_flags': engine_flags})

    def weight_sync(self, version, trainer, rollout):
        """trainer / rollout: (name, tensor) pairs or a dict, or lists already built by param_entries."""
        side = lambda v: v if isinstance(v, list) and all(isinstance(p, dict) for p in v) else param_entries(v)
        return self.write({'type': 'weight_sync', 'version': version, 'trainer': side(trainer), 'rollout': side(rollout)})

    def snapshot(self, version, location):
        return self.write({'type': 'snapshot', 'version': version, 'location': str(location)})

    def close(self):
        with self._lock:
            self._stream.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def read_records(paths, strict=True):
    """[(ref, record)] from one or more JSONL files; ref is "path:line". With strict=False,
    invalid lines are returned as (ref, None, error) triples instead of raising."""
    if isinstance(paths, (str, Path)):
        paths = [paths]
    out = []
    for path in paths:
        path = Path(path).expanduser().resolve()
        with path.open(encoding='utf-8') as stream:
            for n, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                ref = f'{path}:{n}'
                try:
                    out.append((ref, validate_record(json.loads(line))))
                except ValueError as exc:
                    if strict:
                        raise ValueError(f'{ref}: {exc}') from None
                    out.append((ref, None, str(exc)))
    return out


if __name__ == '__main__':                     # python -m opguard_harness.rl_records FILE...: validate
    rows = read_records(sys.argv[1:], strict=False)
    bad = [r for r in rows if len(r) == 3]
    print(json.dumps({'records': len(rows) - len(bad), 'invalid': [{'ref': r[0], 'error': r[2]} for r in bad]}))
    sys.exit(1 if bad else 0)
