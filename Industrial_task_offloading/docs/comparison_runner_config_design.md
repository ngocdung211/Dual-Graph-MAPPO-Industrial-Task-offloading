# Comparison Runner CLI and Model Configuration Design

**Date:** 2026-09-24
**Status:** Awaiting design review

## Goal

Keep the supplied `run_comparision.py` command valid while reducing its CLI to run choices and moving model hyperparameters into the existing `utils/paper_config.py`. Preserve the effective settings of that command and make new paper runs reproducible.

## Scope and architecture

The current three model-facing responsibilities remain: environment and data supply workloads, geometry, and connection windows; agent models choose and learn actions; experiment and output code records results and artifacts. `run_comparision.py` remains the entry point and orchestrator. `utils/paper_config.py` is the single home for default model settings. This change does not add a model role or alter network architecture, reward, route, state, or action semantics.

The model roles remain: frozen Task-GAT for task priority; flat and shared MAPPO policies; Graph-GAT MAPPO with the trainable topology encoder; GATMA-Adapted; e-ATN-MADDPG; and fixed baselines. Moving a setting changes where it is selected, not which agent consumes it.

## CLI boundary

Keep every option in the user's modular-cell command:

- Run identity and scope: `--algorithms`, `--topology-scenario`, `--server-profile`, `--episodes`, `--experiment-seed`, `--note`.
- Experiment switches: `--task-priority`, `--use-gae`, `--num-minibatches`, `--maddpg-updates-per-episode`, `--maddpg-actor-replay-actions`.
- Execution and data: `--gatma-device`, `--graph-gat-device`, `--dataset-path`, `--local-output-root`, `--drive-artifact-root`.
- Tracking: `--wandb-mode`, `--wandb-project`, `--wandb-group`.

Also keep `--topology-seed`, because it controls the sampled physical topology independently of `--experiment-seed`, and `--allow-dummy-data`, because synthetic data should require an explicit request. Restore the currently missing `--server-profile` parser declaration; the main runner still reads it.

Move `--baseline-episodes` and `--wandb-entity` defaults into `paper_config.py`. Remove the following model override flags from the CLI and read their effective values from `paper_config.py`: all `--graph-gat-lr`, `--graph-gat-encoder-lr`, `--graph-gat-hidden-dim`, `--graph-gat-embedding-dim`, `--graph-gat-clip-param`, `--graph-gat-ppo-epochs`, `--graph-gat-entropy-coef`, `--graph-gat-value-loss-coef`, `--graph-gat-max-grad-norm`, `--graph-gat-warmup-episodes`, `--graph-gat-warmup-updates-per-step`, `--graph-gat-warmup-lr`, `--mappo-entropy-coef`, `--mappo-max-grad-norm`, and `--maddpg-epsilon-schedule`.

Remove `--hyperparameters-dir` from the active runner and the now-unused tuning-profile loading and application path. Preserve the old profile files and results as historical evidence. Commands with removed options will fail with argparse's clear unrecognized-argument error; update active command documentation and label old commands as historical.

## Configuration and behavior

Reuse `PAPER_PARAMS` in `utils/paper_config.py`, grouping model settings by existing model prefixes rather than adding a second config system. Add the current GATMA-Adapted settings that are hard-coded in `build_algorithm_configs()` and the MADDPG exploration-schedule default. Retain the present effective default for each moved setting, including the paper failed-offload penalty of `-1`. Keep the supplied command's explicit GAE, minibatch, replay-update, and actor-replay choices as CLI values.

`build_algorithm_configs()` reads model defaults from `paper_config.py` and accepts only the retained device and run-level choices. Remove its optional per-model hyperparameter override arguments and the tuned-profile branch. `tune_hyperparameters.py` does not call this builder; tests that currently pass model override arguments will instead verify the configured defaults and explicit run switches. Algorithm names and action-mask distinctions remain unchanged.

Continue recording resolved agent constructor settings in W&B metadata and local run artifacts. Record the selected seed, topology seed and profile, reward weights, task-priority mode, and run switches with each comparison. The source code revision identifies the versioned config used for the run.

## Verification and compatibility

1. Parse the supplied modular-cell command and confirm all arguments are accepted, especially `--server-profile scenario`.
2. Compare effective agent settings before and after the refactor for all six named algorithms, including GATMA's formerly hard-coded values. Confirm `lambda5=1.0` and `p_out_value=-1.0`.
3. Confirm each removed CLI option is rejected, while the corresponding value is read from `paper_config.py` where still relevant.
4. Check that selected models receive the intended devices, GAE/minibatch settings, and MADDPG update and replay-action behavior.
5. Run focused parser/config tests and the relevant runner test suite; distinguish existing failures from regressions.
6. Keep `docs/dual_gat_mappo_architecture.md` synchronized with the final CLI and config flow.

No full 1000-episode training is needed to validate this configuration refactor. A short smoke run should precede the next full paper comparison.
