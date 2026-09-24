# Experiment Plan

## Current paper setting (2026-09-24)

- Keep the current effective failed-offload penalty at `-1`
  (`lambda5=1.0`, `p_out_value=-1.0`).
- Do not use `--hyperparameters-dir` or the old `unmasked_penalty_0_5`
  profile as a requirement for new paper experiments.
- Keep older `penalty_0_5` results and command records labeled with their
  original settings; they are historical evidence, not `-1` results.
- Before ablations, record the actual reward weights, seed, topology, model
  settings, and code revision for each new run.

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
