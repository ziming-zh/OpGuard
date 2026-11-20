"""Provider-neutral function schemas and the matching dispatch boundary."""
from .engine import Harness


def tool(name, description, properties, required):
    return {'name': name, 'description': description,
            'parameters': {'type': 'object', 'properties': properties,
                           'required': required, 'additionalProperties': False}}


ID = {'type': 'string', 'description': 'Experiment ID returned by create_experiment'}
ARGV = {'type': 'array', 'items': {'type': 'string'}, 'minItems': 1}
WORKLOAD = {
    'type': 'object', 'additionalProperties': False,
    'properties': {
        'cwd': {'type': 'string', 'description': 'Dedicated source checkout, visible to all workers'},
        'script': {'type': 'string', 'description': 'Python script within cwd'},
        'launcher': ARGV,
        'args': {'type': 'array', 'items': {'type': 'string'}},
        'env': {'type': 'object', 'additionalProperties': {'type': 'string'}},
        'ranks': {'type': 'array', 'items': {'type': 'integer', 'minimum': 0}, 'minItems': 1, 'uniqueItems': True},
    },
    'required': ['cwd', 'script', 'launcher'],
}
SPEC = {
    'type': 'object', 'additionalProperties': False,
    'properties': {
        'baseline': WORKLOAD, 'candidate': WORKLOAD,
        'backend': {**ARGV, 'description': 'Backend argv prefix, e.g. /env/bin/python -m opguard.tools'},
        'local_root': {'type': 'string', 'description': 'Absolute node-local staging directory'},
        'timeout_seconds': {'type': 'number', 'exclusiveMinimum': 0, 'default': 3600},
        'max_stage_attempts': {'type': 'integer', 'minimum': 1, 'maximum': 10, 'default': 2},
        'label': {'type': 'string'}, 'input_id': {'type': 'string'},
    },
    'required': ['baseline', 'candidate', 'backend', 'local_root'],
}
TOOLS = [
    tool('create_experiment', 'Validate a spec and create a persistent experiment. Does not run workloads.', {'spec': SPEC}, ['spec']),
    tool('advance_experiment', 'Execute exactly one pending stage. May launch an expensive workload. Inspect status after return; failures do not imply a comparison verdict.', {'experiment_id': ID}, ['experiment_id']),
    tool('experiment_status', 'Inspect stage progress, attempts, errors and log paths.', {'experiment_id': ID}, ['experiment_id']),
    tool('experiment_results', 'Get comparison artifacts. not_evaluated must never be interpreted as equivalent.', {'experiment_id': ID}, ['experiment_id']),
    tool('refine_experiment', 'Create a new experiment with a revised full spec, preserving the parent evidence. No automatic diagnosis or cache reuse.', {'experiment_id': ID, 'spec': SPEC}, ['experiment_id', 'spec']),
    tool('recover_experiment', 'Reconcile an interrupted controller only when the backend has exited and source restoration is confirmed.', {'experiment_id': ID}, ['experiment_id']),
]


ARTIFACT = {'type': 'string', 'description': 'Local path to an OpGuard Perfetto JSON or JSON.gz export'}
EVENT = {'type': 'string', 'description': 'Artifact-bound event_ref returned by a trace query'}
TOOLS += [
    tool('code_context', 'Read the enclosing function around a source location, with signature and bounded body from the captured snapshot.', {'artifact': ARTIFACT, 'source': {'type': 'string'}, 'line': {'type': 'integer', 'minimum': 1}}, ['artifact', 'source', 'line']),
    tool('compare_events', 'Read selected candidate events and regional fingerprint differences with byte ranges; does not certify alignment, tensor equality or padding noise.', {'artifact': ARTIFACT, 'left_ref': EVENT, 'right_ref': EVENT}, ['artifact', 'left_ref', 'right_ref']),
    tool('trace_summary', 'Inspect recorded runs/ranks, statuses, embedded sources and evidence limitations.', {'artifact': ARTIFACT}, ['artifact']),
    tool('first_diff', 'Find exporter-annotated first differences per run/rank; no root-cause or equality inference.', {'artifact': ARTIFACT}, ['artifact']),
    tool('inspect_event', 'Read IO fingerprints, regional byte ranges/differences, exporter-aligned peer and source context. Equal XOR is not proof of equal tensors.', {'artifact': ARTIFACT, 'event_ref': EVENT}, ['artifact', 'event_ref']),
    tool('trace_neighbors', 'Read bounded same-run/rank presentation neighbors, NOT data dependencies.', {'artifact': ARTIFACT, 'event_ref': EVENT, 'radius': {'type': 'integer', 'minimum': 0, 'maximum': 20}}, ['artifact', 'event_ref', 'radius']),
    tool('source_excerpt', 'Read up to 100 lines from an embedded source snapshot, without accessing the original filesystem path.', {'artifact': ARTIFACT, 'source': {'type': 'string'}, 'start_line': {'type': 'integer', 'minimum': 1}, 'end_line': {'type': 'integer', 'minimum': 1}}, ['artifact', 'source', 'start_line', 'end_line']),
    tool('find_events', 'Page operation events whose fn (full name) or event name contains a literal, filtered by run, rank and counter range, with their io slots. Order is run/rank/counter, not causality.',
         {'artifact': ARTIFACT, 'query': {'type': ['string', 'null']}, 'run': {'type': ['string', 'null']},
          'rank': {'type': ['integer', 'null']}, 'counter_min': {'type': ['integer', 'null']},
          'counter_max': {'type': ['integer', 'null']}, 'cursor': {'type': 'integer', 'minimum': 0},
          'limit': {'type': 'integer', 'minimum': 1, 'maximum': 200}},
         ['artifact', 'query', 'run', 'rank', 'counter_min', 'counter_max', 'cursor', 'limit']),
    tool('search_source', 'Literal search across embedded source snapshots, capped at 50 hits.', {'artifact': ARTIFACT, 'query': {'type': 'string'}}, ['artifact', 'query']),
]

TENSOR_SLOT = {'type': ['array', 'null'], 'description': 'Exported io slot ["inputs" or "outputs", integer index]; null for reviewing an existing whole-event relation.',
               'minItems': 2, 'maxItems': 2, 'items': {'type': ['string', 'integer']}}
REVIEW = {'type': 'object', 'additionalProperties': False, 'properties': {
    'left_ref': EVENT, 'right_ref': EVENT, 'left_tensor': TENSOR_SLOT, 'right_tensor': TENSOR_SLOT,
    'decision': {'type': 'string', 'enum': ['comparable', 'not_comparable', 'undetermined']},
    'reason': {'type': 'string', 'minLength': 1},
    'evidence': {'type': 'array', 'minItems': 1, 'items': {'type': 'object', 'additionalProperties': False,
        'properties': {'basis': {'type': 'string', 'enum': ['observed', 'source_inference', 'intervention_tested', 'externally_reported']},
                       'reference': {'type': 'string', 'minLength': 1}}, 'required': ['basis', 'reference']}},
}, 'required': ['left_ref', 'right_ref', 'left_tensor', 'right_tensor', 'decision', 'reason', 'evidence']}
TOOLS += [
    tool('review_pair', 'Append an artifact-scoped comparability judgment with evidence. It is an agent assertion, not a numerical verdict. Use null slots for not_comparable/undetermined; new comparable pairs require both tensor slots.',
         {'artifact': ARTIFACT, 'review': REVIEW}, ['artifact', 'review']),
    tool('difference_candidates', 'Page exporter-labelled differences, deduplicating reciprocal peers. Skip reviewed candidates or revisit them. Empty results and undetermined reviews never establish equivalence.',
         {'artifact': ARTIFACT, 'cursor': {'type': 'integer', 'minimum': 0},
          'limit': {'type': 'integer', 'minimum': 1, 'maximum': 50}, 'include_reviewed': {'type': 'boolean'}},
         ['artifact', 'cursor', 'limit', 'include_reviewed']),
    tool('render_alignment', 'Freeze the review map and ask OpGuard to write a new Perfetto view. No workload launch, implicit transforms or tensor-equality certification. Never overwrites an existing output.',
         {'artifact': ARTIFACT, 'output': {'type': 'string', 'description': 'Fresh absolute .json or .json.gz trace path'},
          'backend': ARGV}, ['artifact', 'output', 'backend']),
]


NUM = {'type': 'integer', 'minimum': 0}
NULLABLE_SLOT = {'type': ['array', 'null'], 'description': 'Exported io slot ["inputs" or "outputs", integer index]',
                 'minItems': 2, 'maxItems': 2, 'items': {'type': ['string', 'integer']}}
SLOT = {**NULLABLE_SLOT, 'type': 'array'}
SCOPE = {'type': ['object', 'null'], 'additionalProperties': False,
         'description': 'Null for all events, or limits: runs, ranks (lists) and counter_min/counter_max',
         'properties': {'runs': {'type': ['array', 'null']}, 'ranks': {'type': ['array', 'null'], 'items': NUM},
                        'counter_min': {'type': ['integer', 'null']}, 'counter_max': {'type': ['integer', 'null']}}}
TOKEN_DIMS = {'type': 'integer', 'minimum': 0, 'description': 'Declared by the agent: leading dims whose product is the token count T'}
TOKEN_IDS = {'type': ['array', 'null'], 'description': 'Null (rows numbered 0..T-1) or one global token id per row, e.g. forward token_ranks; rows pair by id'}
RECORDS = {'type': 'array', 'items': {'type': 'string'}, 'minItems': 1,
           'description': 'RL harness record files (JSONL, schema opguard.rl.v1), read together'}
BATCH = {'type': ['string', 'integer'], 'description': 'batch_id of a rollout_batch record'}
NOTE_FACTS = ' Facts only: exact fingerprint equality, verdict not_evaluated; a zero fingerprint or a match across widths is never evidence.'
TOOLS += [
    tool('vectors_describe', 'Per tensor slot of an event: shape, dtype, vector_bytes, count, zero vectors and the first N per-vector fingerprints (lastdim-u32le-v1 or legacy xor_rows).' + NOTE_FACTS,
         {'artifact': ARTIFACT, 'event_ref': EVENT, 'first_n': {'type': 'integer', 'minimum': 0, 'maximum': 256}},
         ['artifact', 'event_ref', 'first_n']),
    tool('vectors_lookup', 'Every exact occurrence of a fingerprint, given directly or as event_ref + slot + vector, within a scope. Zero-fingerprint matches are returned apart, never as occurrences.' + NOTE_FACTS,
         {'artifact': ARTIFACT, 'fingerprint': {'type': ['integer', 'null'], 'minimum': 0, 'maximum': 4294967295},
          'event_ref': {**EVENT, 'type': ['string', 'null']}, 'slot': NULLABLE_SLOT,
          'vector': {'type': ['integer', 'null'], 'minimum': 0}, 'scope': SCOPE},
         ['artifact', 'fingerprint', 'event_ref', 'slot', 'vector', 'scope']),
    tool('vectors_coverage', 'For each vector of a slot, whether and where it occurs (same width and dtype, nonzero) in the scope; null scope means the other run(s).' + NOTE_FACTS,
         {'artifact': ARTIFACT, 'event_ref': EVENT, 'slot': SLOT, 'scope': SCOPE},
         ['artifact', 'event_ref', 'slot', 'scope']),
    tool('vectors_compare', 'Compare two slots token row by token row under token dims the agent declares for each: equal, or the indices of the differing vectors.' + NOTE_FACTS,
         {'artifact': ARTIFACT, 'left_ref': EVENT, 'left_slot': SLOT, 'left_token_dims': TOKEN_DIMS, 'left_token_ids': TOKEN_IDS,
          'right_ref': EVENT, 'right_slot': SLOT, 'right_token_dims': TOKEN_DIMS, 'right_token_ids': TOKEN_IDS},
         ['artifact', 'left_ref', 'left_slot', 'left_token_dims', 'left_token_ids',
          'right_ref', 'right_slot', 'right_token_dims', 'right_token_ids']),
    tool('rl_records_summary', 'List the RL harness records present: counts by type, batches with their policy versions and trainer_logprob versions, forwards per rank, weight-sync and snapshot versions.',
         {'records': RECORDS}, ['records']),
    tool('rl_forwards', 'Forward records of a rank in counter order: token count, token_ranks and the observed [seq_id, position] of each row when recorded.',
         {'records': RECORDS, 'rank': {'type': ['integer', 'null']}, 'counter_min': {'type': ['integer', 'null']},
          'counter_max': {'type': ['integer', 'null']}}, ['records', 'rank', 'counter_min', 'counter_max']),
    tool('rl_decompose', 'Split each token\'s rollout/trainer log-probability gap exactly into numeric (rollout - trainer at the token\'s version) and staleness (trainer at that version - trainer at trainer_version), with zero flags decided on bits and the source records of each value. Mixed-version sequences are flagged and never split.',
         {'records': RECORDS, 'batch_id': BATCH, 'trainer_version': {'type': 'integer'},
          'policy_version_override': {'type': ['integer', 'null'], 'description': 'Null, or a version to assume for every token (a hypothesis; the result labels it caller_supplied)'}},
         ['records', 'batch_id', 'trainer_version', 'policy_version_override']),
    tool('rl_weight_sync_check', 'Per parameter of a weight_sync record: rows matched by exact fingerprint across the other side (repacking along rows is not a difference), or the unmatched row indices on each side.',
         {'records': RECORDS, 'version': {'type': 'integer'}}, ['records', 'version']),
    tool('rl_replay_request', 'Build (never run) a replay request that recomputes a rollout batch at a policy version on one side, with original or batch-invariant composition and capture options. A batch-invariant replay localizes implementation differences, not the original error.',
         {'records': RECORDS, 'batch_id': BATCH, 'version': {'type': 'integer'},
          'side': {'type': 'string', 'enum': ['trainer', 'rollout']},
          'composition': {'type': 'string', 'enum': ['original', 'batch_invariant']},
          'capture': {'type': ['object', 'null'], 'description': 'Null or {vector_fingerprints: bool, env: {NAME: value}}'},
          'output': {'type': ['string', 'null'], 'description': 'Null, or a fresh absolute path to write the request JSON'}},
         ['records', 'batch_id', 'version', 'side', 'composition', 'capture', 'output']),
]
# The protocol names (docs/RLAlignment.md) use dots; function-calling names cannot.
ALIASES = {item['name'].replace('_', '.', 1): item['name'] for item in TOOLS
           if item['name'].startswith(('vectors_', 'rl_'))}


def dispatch(root, name, arguments):
    name = ALIASES.get(name, name)
    schema = next((item for item in TOOLS if item['name'] == name), None)
    if schema is None:
        raise ValueError(f'Unknown tool: {name}')
    if not isinstance(arguments, dict) or set(arguments) != set(schema['parameters']['required']):
        raise ValueError('Tool argument names do not match its schema')
    from .trace_reader import READERS
    if name in READERS:
        return READERS[name](**arguments)
    from .vectors import VECTOR_TOOLS
    from .rl_actions import RL_TOOLS
    if name in VECTOR_TOOLS:
        return VECTOR_TOOLS[name](**arguments)
    if name in RL_TOOLS:
        return RL_TOOLS[name](**arguments)
    from .reviews import REVIEW_TOOLS
    if name in REVIEW_TOOLS:
        return REVIEW_TOOLS[name](root, **arguments)
    harness = Harness(root)
    functions = {'create_experiment': harness.create, 'advance_experiment': harness.advance,
                 'experiment_status': harness.status, 'experiment_results': harness.results,
                 'refine_experiment': harness.refine, 'recover_experiment': harness.recover}
    return functions[name](**arguments)
