---
name: rl-alignment
description: Check whether an RL rollout engine and trainer compute the same policy, separate policy staleness from numeric error, and localize numeric error to an operator and token using OpGuard per-vector fingerprints. Use for train-inference mismatch in synchronous or asynchronous RL.
---

# RL alignment

Own one question: for sampled RL batches, how much of the rollout/trainer
log-probability gap is policy staleness and how much is numeric error, and if
numeric error is nonzero where parity was intended, where it starts and why.
Return the split, a localization (operator and token on each side) or a bounded
inconclusive result, and a classified cause with its evidence basis.

The protocol (records, fingerprint format, actions) is in
[docs/RLAlignment.md](../../../docs/RLAlignment.md). Backend actions return facts
only. Every correspondence, gap or cause in your report is your judgment and
carries evidence. Keep the shared record from
[investigate-first-diff/references/evidence-contract.md](../investigate-first-diff/references/evidence-contract.md).
Equality is exact throughout. Never call a numeric error "small enough".

## 0. Establish the setup

From records, configs and source, not assumption, collect the following:
- the sync mode and staleness window;
- the policy versions involved;
- each engine's flags: batch-invariant, CUDA graph, attention backend, split-K;
- each side's parallelism: TP, EP, SP;
- model features that change numerics, such as linear attention or MoE.

List which harness records exist: `forward`, `rollout_batch`, `trainer_logprob`,
`weight_sync`, `snapshot`. If one you need is missing, say which and ask for it.
Label anything you infer instead (forward starts, token layout, routing) as
inferred.

## 1. Check the weights first

Run `rl.weight_sync_check` for every version you will compare. Rows are matched
by fingerprint, so a different packing on the two sides is not a difference. If
any parameter row differs, stop and report the parameter and rows: every
log-probability comparison at that version is confounded until the sync is
fixed.

## 2. Split staleness from numeric error

Run `rl.decompose`. Prefer staleness-zero points (the batch's version equals the
trainer's), which need no replay. Otherwise request a trainer replay at the
recorded version with `rl.replay_request` (side `trainer`). Run it offline, and
count it against the launch budget.

- Exclude tokens flagged `mixed_version` from the split, and report how many
  there were.
- `numeric` is exactly zero on every sampled token: the infrastructure is
  exonerated for these batches under these flags. Any remaining gap is staleness,
  which IS/TIS settings must handle. Report that and stop.
- `numeric` is nonzero and parity was not intended (for example batch-invariant
  off): report its distribution over tokens and positions. Localize only if asked.
- `numeric` is nonzero and parity was intended: continue to step 3.

## 3. Localize offline

1. Pick a few tokens with nonzero `numeric`. Earliest positions first, never
   mixed-version tokens.
2. Request per-vector recapture (`OPGUARD_XOR_ROWS=1`) of that batch on both
   sides. Use `composition: original` when the batch composition was recorded.
   Otherwise use `batch_invariant`, and state that the result then shows
   implementation differences, not this run's specific error.
3. Take forward boundaries and token counts from `forward` records. Declare token
   dims per tensor: leading dims whose product is T, or `token_ranks` when
   sequence-sharded. Declare anything else as inferred.
4. Anchor: find an early slot whose vectors agree on both sides, using
   `vectors.compare` or `vectors.coverage` (the embedding output usually does).
   Link tokens across sides only by fingerprints that belong to one token on each
   side.
5. Walk the data flow, not trace order. A record is a candidate for a token when
   its input rows agree and its output row does not. Accept it only with a pair
   on the other side:
   - the other side ran the same input row and got a different output of the
     same width; or
   - a row of that token, with the same width, shares most vectors and differs
     in a few (this names the differing heads).

   An unmatched row on its own is a gap: a step the other side fuses away or
   never captures. It is not a divergence. Follow the candidate's consumers
   through value-preserving ops (views) to a value-changing one. If agreement
   returns, set the candidate aside and record where agreement returned.
6. Tokens that change rank (all-to-all, send/recv) are followed by content,
   because those moves keep bytes. If bytes change in transit (quantized
   dispatch), ask for a routing record instead of guessing.
7. When correspondence or a producer is unclear, read the source with
   `trace-upstream`. Write each supported correspondence as an alignment-map
   review with its evidence.

Stop at the first confirmed, paired candidate for each token studied. If the walk
cannot continue, report the narrowed boundary and the smallest missing
observation: a missing slot, record or rank.

`perfetto/token_match.py` runs a reference version of this walk. Use its output
as a lead to check, never as the result.

## 4. Classify the cause

Name the mechanism and the evidence for it. Common classes:

- **Rounding point or fusion order.** The same math is cast or fused at a
  different place: a norm rounding before or after its weight, a residual add in
  fp32 or bf16, a RoPE formula variant, router weights cast to bf16.
- **Batch dependence.** The result changes with the batch composition. Compare an
  original-composition replay with a batch-invariant one.
- **Algorithmic choice.** Two forms of the same math with different reduction
  orders: chunked against recurrent linear attention, split-K, decode against
  prefill attention. This is a choice, not a bug. Name it and its cost.
- **Parallel layout.** TP/EP/SP splits a reduction differently, or expert outputs
  combine in a different order.
- **Log-probability formula.** Different log_softmax/gather paths. Check the final
  op before anything deeper.
- **Bookkeeping.** A wrong policy version, a stale snapshot, or a weight-sync
  fault (step 1).

Test a class with `validate-hypothesis` using the smallest discriminating
change, such as making one side use the other's order for the one op and
checking that `numeric` becomes zero at that op.

## 5. Report

Report the following:
- the setup and records used;
- the weight-sync result;
- the split: counts of zero and nonzero `numeric`, and mixed-version tokens
  excluded;
- for each localized token, the event references on both sides, the token, the
  differing vectors or heads, and the evidence for the pair;
- the cause class with its evidence basis;
- remaining unknowns;
- launches used.

Add a short handbook entry: scenario, symptom, where it diverged, fix options and
their throughput cost.

## Guardrails

- Never change the training job's flags, weights or schedule. Replays and
  recaptures run offline.
- Batch-invariant modes slow rollouts. Lengthen replay timeouts accordingly, so a
  killed rollout is not mistaken for a numeric result.
- A matching fingerprint can collide. Matching across differing widths, or on a
  zero fingerprint, is never evidence.
- Externally reported causes, such as an issue thread or a blog post, are
  `externally_reported` until tested here.
