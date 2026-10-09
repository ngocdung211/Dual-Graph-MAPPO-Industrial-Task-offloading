# Experiment Plan

## Current paper setting (2026-10-09)

- The effective failed-offload penalty is `-1.5`
  (`lambda5=1.0`, `p_out_value=-1.5`); request admission does not change it.
- The previous 2026-09-24 decision used `-1`. Preserve the actual settings in
  historical run metadata rather than reinterpreting old runs as current ones.
- Do not use `--hyperparameters-dir` or the old `unmasked_penalty_0_5`
  profile as a requirement for new paper experiments.
- Keep older `penalty_0_5` results and command records labeled with their
  original settings; they are historical evidence, not `-1` results.
- Before ablations, record the actual reward weights, seed, topology, model
  settings, and code revision for each new run.

## Separate library-GAT experiment (2026-10-04)

- **Implemented:** the separate library-GAT ablation described in the
  [architecture](dual_gat_mappo_architecture.md#separate-pyg-gat-mappo-experiment-2026-10-04).
- **Verified on CPU:** independent versus batched local outputs; independent
  versus batched rollout policy/value outputs and gradients; GAE/minibatch PPO
  updates to encoder, actor, and critic; checkpoint prediction round-trip; and
  a paired one-episode real-data smoke. Both runner checkpoints were reloaded
  with finite parameters. This is integration evidence only, not a performance
  comparison.
- The [smoke manifest](../experiments/pyg_gat/smoke_manifest.json) owns the
  effective run settings, artifact paths, dataset metadata, parameter counts,
  library versions, and validation-file hashes. Its provenance note identifies
  the uncommitted implementation and subsequent non-behavioral edits.
- **Planned, not run:** matched full training over multiple seeds; compare mean
  and standard deviation of reward, delay, energy, rejection rate, convergence,
  and runtime. Choose the full episode budget, seeds, and topology scope before
  long training. GPU behavior and performance superiority remain unverified.

For the masked encoder ablation described in the
[architecture](dual_gat_mappo_architecture.md#separate-pyg-gat-mappo-experiment-2026-10-04),
use the command below with
`--algorithms "Graph-GAT Mask MAPPO" "PyG-GAT Mask MAPPO"` and a separate note
such as `pyg_gat_mask_real_data_smoke`. Keep masking enabled for both models;
comparing masked PyG against unmasked custom GAT would change two factors.
Masked registration, disconnected-link sampling, all-disconnected local
fallback, PPO updates, and checkpoint restoration are covered by the
[focused behavioral tests](../tests/test_pyg_topology_gat.py#L1).
**Verified on CPU:** both masked variants completed the paired integration smoke
and reloaded separate checkpoints with finite parameters and masking enabled.
The [masked smoke manifest](../experiments/pyg_gat/masked_smoke_manifest.json)
owns the effective settings, dataset metadata, artifact paths, and check counts.
This verifies integration, not a performance advantage.

**Prepared, not run:** [note.txt](../note.txt#L1) owns the three full-training
commands and their run settings. The requested `penalty1p5` names are labels
only; the [current paper setting](#current-paper-setting-2026-09-24) still
applies. No reward configuration was changed for these commands.

Install the optional dependency as described in the [README](../../README.md#installation).

Run a short comparison (real dataset required; output directory is separate):

```bash
python run_comparision.py \
  --algorithms "Graph-GAT MAPPO" "PyG-GAT MAPPO" \
  --topology-scenario paper_10d_3s --episodes 1 --experiment-seed 75 \
  --use-gae --num-minibatches 4 --graph-gat-device cpu \
  --wandb-mode disabled --local-output-root results/pyg_gat_comparison \
  --note pyg_gat_real_data_smoke
```

GAE and minibatching must be enabled identically for both versions. Task-GAT,
graph edges (including disconnected pairs), actor/critic heads, topology, data,
reward, action masking, and warmup are held fixed. Encoder semantics and actor
pair features are described in the
[architecture](dual_gat_mappo_architecture.md#separate-pyg-gat-mappo-experiment-2026-10-04).

## Subtask offloading count report (2026-10-09)

- [x] Add [plot_offloading_counts.py](../utils/reporting/plot_offloading_counts.py)
  to export full training counts from W&B and plot CSV exports offline.
- [x] Define successful edge executions and rejected requests at subtask level,
  require count conservation and complete unique episodes, and keep GATMA v3
  separate from historical v2. Counts use final-window seed means; rates use
  summed outcomes/requests per seed, then across-seed mean/sample SD.
- [x] Verify weighted denominators, final-window selection, undefined rates,
  malformed counts, missing/duplicate seeds, v2 exclusion, and rejection of
  performance-only CSVs in the focused reporting checks.
- [x] Validate all 12,000 histories against the saved comparison performance
  curves, verify 15,000 subtask actions per episode, and inspect the figures.
  Results: [summary](../results/analysis/20261009_offloading_counts/summary.md).
- [x] Extend reporting with the three matched Mask MAPPO runs and validate
  their 3,000 performance rows against the saved ablation CSVs. Inspect all
  five-model figures; the original four model summaries remain unchanged.
  Results: [Mask MAPPO comparison](../results/analysis/20261009_offloading_counts_with_mask_mappo/summary.md).
- [ ] Add count/rate results and metric definitions to the manuscript.

The [architecture](dual_gat_mappo_architecture.md#subtask-offloading-count-reporting-2026-10-09)
documents sources, outputs, and metric semantics. Computational cost remains a
separate task; this report changes no training or energy-accounting behavior.

## Optional manuscript figure: local actor and global critic (2026-10-09)

**Deferred, not added to the manuscript:** explain local actor graphs and the
global critic readout in text first. The architecture's
[actor/critic tensor flow](dual_gat_mappo_architecture.md#4-actor-and-critic-tensor-paths)
is the reference for a later publication figure.

- [ ] Prepare a vector figure with local graph extraction, separate local/global
  encoder outputs, and a shared encoder-parameter annotation. Show that the
  critic flattens only global device embeddings, not all nodes or raw topology.
- [ ] Include both edge directions at the encoder input, the actor's pair
  features, and connectivity masking/renormalization. Fix clipped labels,
  verify dimensions against code, and export PDF/PNG before manuscript insertion.

## Offload request admission implementation (2026-10-09)

**Implemented and opt-in; no full retraining performed.** The approved minimum
model adds a 1 ms device request, 1 ms ACK/rejection wait, 100 ms no-response
timeout, and 0.05 W device listening power. These are declared assumptions;
the timeout is an adaptation of a connection-establishment example, not a
measured DITEN parameter. Existing reward weights/penalty are unchanged.

- [x] Implement accepted, actively rejected, and no-response timeout outcomes
  before task upload. Reject/timeout still fall back to local with penalty.
  Direct local actions pay no control costs. Account for signaling/listening
  once, shift data/CPU readiness, and reserve server work only after admission.
- [x] Add `--request-overhead` (default off) and `--request-timeout-s`, effective
  checkpoint/W&B/JSONL/CSV settings, episode outcome counts, and control-energy
  diagnostics. Restore saved protocol settings in evaluation; absent metadata
  retains legacy behavior, with an explicit evaluation override available.
- [x] Pass 67 focused tests covering environment, artifact publication,
  tracking, reporting, digital twin, and topology-state behavior. Real-data
  backward-compatibility trajectories match the HEAD environment exactly over
  1,500 joint steps (100 slots each on small, medium, and modular scenarios).
- [x] Run one 100-slot real-data episode for e-ATN-MADDPG, Shared MAPPO,
  GATMA v3, Shared Mask MAPPO, and Graph-GAT Mask MAPPO, using 352 images,
  seed 190, modular topology, CPU, GAE, four minibatches, and tracking disabled.
  Check all 15,000 subtask decisions per model, finite checkpoint weights,
  outcome conservation, and protocol metadata. All five checkpoints pass a
  separate two-slot greedy evaluation and legacy-default/override equivalence.
- [ ] Before matched full training, agree the budget/seeds and any timeout
  sensitivity runs. Keep new results separate from previous training histories.
- [ ] Update manuscript results only after an appropriate new comparison;
  the integration smoke provides no trained-policy superiority evidence.

The unmasked smoke policies exhibit queue backlog and approximately 11 s mean
task delay, versus approximately 1.5 s for masked policies. These are initial
one-episode smoke observations, not final performance. Request duration totals
must not be added to existing delay curves: changed readiness/queue feedback
requires a new rollout. Partial-upload failures, retries, server ACK energy,
and radio contention remain outside this model.

Settings, flows, and code references are in the
[architecture](dual_gat_mappo_architecture.md#opt-in-offload-request-admission-2026-10-09).
Smoke outputs are under `results/request_overhead_smoke`; their verification
manifest records effective settings, source hashes, and validation scope.

## Historical `-0.5` plan (not the current paper setting)

### Locked configuration

- Environment: effective failed-offload penalty `-0.5`
  (`lambda5=0.5`, `p_out_value=-1.0`).
- Policy input: unmasked.
- Compute workload: MobileNetV3-Large-based five-stage task DAG.
- Hyperparameter-tuning budget: 500 episodes.
- Topology-comparison budget: 1000 episodes.
- Hyperparameter profile:
  `configs/hyperparameters/unmasked_penalty_0_5`.
- Selected Optuna trials: e-ATN-MADDPG trial 11, MAPPO trial 23,
  Graph-GAT Warmup MAPPO trial 20.

### Completed topology comparison

1. [x] Train e-ATN-MADDPG, MAPPO, Graph-GAT MAPPO, and Graph-GAT Warmup
   MAPPO on `paper_10d_3s` for 1000 episodes.
2. [x] Train the same models on `medium_20d_6s` for 1000 episodes.
3. [x] Train the same models on `large_30d_9s` for 1000 episodes.
4. [x] Preserve final JSONL, plots, checkpoints, profile name, topology
   metrics, and actual CPU statistics for every run.

### Immediate next work

1. [x] Preserve complete per-episode histories locally, not only plots and the
   final episode, so paper-style averages and final-window statistics can be
   reproduced without relying on W&B.
2. [x] Build one three-topology summary using training metrics, including
   reward, delay, energy, requested/resolved edge ratios, rejection rate, and
   topology connectivity metrics. Seed-75 outputs are stored under
   `results/analysis/penalty_0_5_best_1000ep_seed75`.
3. Select the smallest follow-up experiment after the summary:
   - transfer the locked unmasked hyperparameters to masked variants; or
   - confirm only MAPPO and Graph-GAT Warmup MAPPO across additional seeds.

### Seed provenance note

- The comparison group named `penalty_0_5_best_1000ep_seed175` created on
  2026-08-08 actually used seed 75 because the runner previously had no seed
  CLI override. Its 12,000 episode metrics exactly reproduce the original
  seed-75 histories and are retained only as a deterministic repeat at
  `results/analysis/penalty_0_5_best_1000ep_repeat_seed75`.
- `run_comparision.py` now accepts `--experiment-seed`; the corrected commands
  in `note.txt` explicitly pass `--experiment-seed 175`.

### Deferred work

1. Confirm the selected configuration with independent training seeds;
   rerank only if performance is unstable.
2. Train masked variants using the unmasked hyperparameters as the initial
   configuration.
3. Run Graph-GAT ablations: device-only output versus pairwise
   server-conditioned output, and warmup versus no warmup.
4. Evaluate fixed baselines with the same topology and test seeds.
5. Aggregate mean, standard deviation, reward, delay, energy, requested and
   resolved edge ratios, penalty count, and convergence plots for reporting.
6. If the timing breakdown is used in the paper, separate dependency wait,
   transfer time, local queue time, and server queue time; the current
   `queue_or_wait_time` diagnostic combines more than pure server queuing.

### Optional validation

1. Evaluate saved checkpoints with deterministic greedy actions on independent
   validation seeds and report mean plus standard deviation separately from
   the paper-style training curves. The baseline paper plots metrics collected
   during 1000 training episodes and does not describe a separate greedy test
   phase, so this validation strengthens the evidence but does not block the
   reproduction workflow.
