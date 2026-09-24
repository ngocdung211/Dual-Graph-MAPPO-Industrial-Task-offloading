# Comparison Runner Configuration Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Keep the approved modular-cell comparison command valid while removing per-model tuning flags and centralizing model defaults in `utils/paper_config.py`.

**Architecture:** `run_comparision.py` retains CLI parsing and experiment orchestration. `utils/paper_config.py` holds model defaults, including GATMA-Adapted settings. The environment, model roles, reward, route, state, and action interfaces remain unchanged.

**Tech Stack:** Python, argparse, PyTorch, pytest. Run commands from `Industrial_task_offloading/` with `/home/daniel_window/miniconda3/envs/task_offloading/bin/python`; system Python has no torch or pytest.

**Spec:** `docs/comparison_runner_config_design.md`.

## Global Constraints

- Preserve the user-supplied modular-cell command and keep `--topology-seed` and `--allow-dummy-data`.
- Keep failed-offload penalty `lambda5=1.0` and `p_out_value=-1.0`.
- Preserve current effective model settings and algorithm names; move their source to config without retuning.
- Preserve historical profile files and result artifacts; remove their unused runtime path.
- Preserve unrelated uncommitted changes, including the episode-route fix.
- Update `docs/dual_gat_mappo_architecture.md` with any runtime configuration change.
- Compare the six requested models to `/tmp/comparison_runner_pre_cleanup_configs.json`, captured before implementation with Graph-GAT on CUDA, GATMA on CPU, GAE enabled, four minibatches, and replay actions enabled.
- Commit only task-owned hunks. Several files already contain user edits, so an exact file list alone is insufficient; inspect each diff before staging and leave mixed files uncommitted when hunk separation is unsafe.

## Review Focus

1. The exact six-model command includes `--server-profile scenario`; parser must accept and use it.
2. Model settings read from config must preserve masked/unmasked and warmup/non-warmup distinctions.
3. Graph-GAT device and GATMA device must remain scoped to their respective agents.
4. GAE and minibatch switches must reach all MAPPO-family agents, while MADDPG switches reach e-ATN-MADDPG.
5. W&B disabled mode and strict local dataset defaults must remain usable without optional services.

---

### Task 1: Narrow the CLI to run choices

**Files:**
- Modify: `run_comparision.py`, `utils/paper_config.py`.
- Test: `tests/test_topology_scenarios_config.py`, `tests/test_local_artifact_workflow.py`.

**Interfaces:**
- Consumes: `PAPER_PARAMS["provisional_table2_needed"]`.
- Produces: `parse_args()` with the approved CLI options, including `server_profile`, and no model-tuning or profile-dir flags.

**Test and implementation anchors:**

```python
# tests/test_topology_scenarios_config.py
monkeypatch.setattr(sys, "argv", [
    "run_comparision.py", "--algorithms", "Shared MAPPO", "GATMA-Adapted",
    "Graph-GAT MAPPO", "Graph-GAT Warmup MAPPO",
    "Graph-GAT Warmup Mask MAPPO", "e-ATN-MADDPG",
    "--topology-scenario", "modular_cells_30d_9s", "--server-profile", "scenario",
    "--episodes", "1", "--experiment-seed", "75", "--task-priority", "on",
    "--use-gae", "--num-minibatches", "4", "--maddpg-updates-per-episode", "16",
    "--maddpg-actor-replay-actions", "--gatma-device", "cpu",
    "--graph-gat-device", "cuda", "--dataset-path", "dataset/KolektorSDD",
    "--local-output-root", "plots", "--drive-artifact-root", "G:\\My Drive\\Dual-Graph-MAPPO-Artifacts",
    "--wandb-mode", "online", "--wandb-project", "industrial-task-offloading",
    "--wandb-group", "modular_cells_30d_9s-seed75", "--note", "modular_cells_30d_9s-seed75",
])
args = parse_args()
assert args.server_profile == "scenario"
assert args.num_minibatches == 4
assert args.maddpg_actor_replay_actions is True
```

```python
# run_comparision.py: parser and main
parser.add_argument(
    "--server-profile",
    choices=("scenario", "uniform", "heterogeneous", "stress"),
    default="scenario",
)
BASELINE_EVALUATION_EPISODES = int(provisional["baseline_evaluation_episodes"])
# initialize_experiment_tracker(..., entity=str(provisional["wandb_entity"]), ...)
```

- [ ] **Step 1: Add a parser regression test.** In `tests/test_topology_scenarios_config.py`, set `sys.argv` to the user's command arguments (use `--episodes 1` in the test) and assert `args.server_profile == "scenario"`, the six algorithm names are preserved, `args.use_gae is True`, `args.num_minibatches == 4`, `args.maddpg_updates_per_episode == 16`, and `args.maddpg_actor_replay_actions is True`. Add a separate parametrized check that `--hyperparameters-dir`, `--graph-gat-lr`, `--mappo-entropy-coef`, and `--maddpg-epsilon-schedule` are rejected with `SystemExit(2)`.
- [ ] **Step 2: Run the new parser tests.** Run `/home/daniel_window/miniconda3/envs/task_offloading/bin/python -m pytest tests/test_topology_scenarios_config.py -q`; confirm the supplied command test fails because the current parser lacks `--server-profile`.
- [ ] **Step 3: Restore the `--server-profile` parser option.** Use choices `scenario`, `uniform`, `heterogeneous`, `stress` and default `scenario`. Retain every user-supplied flag, `--topology-seed`, and `--allow-dummy-data`. Remove `--baseline-episodes`, `--wandb-entity`, `--hyperparameters-dir`, all twelve `--graph-gat-*` tuning flags, both `--mappo-*` tuning flags, and `--maddpg-epsilon-schedule`.
- [ ] **Step 4: Replace removed run-option reads.** Resolve baseline episode count from `provisional["baseline_evaluation_episodes"]` and W&B entity from `provisional["wandb_entity"]`. Remove active `args.hyperparameters_dir` profile loading. Leave existing `--episodes`, seed, devices, dataset, output, and tracking behavior intact.
- [ ] **Step 5: Re-run parser and local-artifact tests.** Run `/home/daniel_window/miniconda3/envs/task_offloading/bin/python -m pytest tests/test_topology_scenarios_config.py tests/test_local_artifact_workflow.py -q`; confirm retained defaults and the new parser contract pass. Commit only separable task hunks after reviewing the staged diff; leave mixed files uncommitted if staging would include existing user edits.

### Task 2: Read all model defaults from config

**Files:**
- Modify: `utils/paper_config.py`, `run_comparision.py`.
- Test: `tests/test_comparison_diagnostics.py`, `tests/test_graph_gat_mappo.py`, `tests/test_gatma.py`.

**Interfaces:**
- Consumes: existing `PAPER_PARAMS` groups, `graph_gat_device`, `gatma_device`, `use_gae`, `num_minibatches`, and `maddpg_actor_replay_actions`.
- Produces: `build_algorithm_configs()` with the same algorithm map and effective constructor values, without per-model override arguments or tuned-profile input.

**Test and implementation anchors:**

```python
# tests/test_graph_gat_mappo.py
provisional = PAPER_PARAMS["provisional_table2_needed"]
monkeypatch.setitem(provisional, "graph_gat_entropy_coef", 0.005)
configs = build_algorithm_configs(graph_gat_device="cpu")
assert configs["Graph-GAT MAPPO"]["kwargs"]["entropy_coef"] == 0.005
assert configs["Shared MAPPO"]["kwargs"]["entropy_coef"] != 0.005
```

```python
# utils/paper_config.py: existing provisional settings gain these values.
"maddpg_epsilon_schedule": "linear_progress",
"gatma_batch_size": 128,
"gatma_replay_buffer_capacity": 100000,
"gatma_replay_updates_per_episode": 16,
"gatma_actor_lr": 1e-4,
"gatma_critic_lr": 1e-5,
"gatma_gamma": 0.95,
"gatma_tau": 0.01,
"gatma_hidden_dim": 64,
"gatma_embedding_dim": 64,
"gatma_num_heads": 4,
"gatma_epsilon_init": 0.99,
"gatma_epsilon_min": 0.01,
"gatma_exploration_fraction": 1.0,
```

```python
# run_comparision.py: representative builder substitutions.
"batch_size": int(provisional["gatma_batch_size"]),
"replay_buffer_capacity": int(provisional["gatma_replay_buffer_capacity"]),
"replay_updates_per_episode": int(provisional["gatma_replay_updates_per_episode"]),
"actor_lr": float(provisional["gatma_actor_lr"]),
"epsilon_schedule": str(provisional["maddpg_epsilon_schedule"]),
```

- [ ] **Step 1: Write config-driven tests.** Replace the two locked-profile transfer tests in `tests/test_comparison_diagnostics.py` with assertions that `build_algorithm_configs()` uses `PAPER_PARAMS` for MAPPO, Graph-GAT, e-ATN-MADDPG, and GATMA. Replace the old Graph-GAT CLI-override test in `tests/test_graph_gat_mappo.py` with `monkeypatch.setitem` on the config dictionary, then assert the changed Graph-GAT values reach only Graph-GAT variants. In `tests/test_gatma.py`, assert GATMA's batch size, replay capacity, update count, actor and critic learning rates, gamma, and device match config and the retained device switch.
- [ ] **Step 2: Run the changed tests red.** Run `/home/daniel_window/miniconda3/envs/task_offloading/bin/python -m pytest tests/test_comparison_diagnostics.py tests/test_graph_gat_mappo.py tests/test_gatma.py -q` and note failures caused by still-hard-coded GATMA values and removed-API expectations.
- [ ] **Step 3: Add config values.** Add `maddpg_epsilon_schedule="linear_progress"`; add the existing GATMA values under `gatma_` keys: batch size `128`, replay capacity `100000`, replay updates `16`, actor LR `1e-4`, critic LR `1e-5`, gamma `0.95`, tau `0.01`, hidden and embedding dimensions `64`, heads `4`, epsilon initial `0.99`, epsilon minimum `0.01`, exploration fraction `1.0`. Keep the current Graph-GAT, MAPPO, MADDPG, reward, and GAE defaults unchanged. Remove the obsolete `comparison_hyperparameters_dir` config entry.
- [ ] **Step 4: Simplify `build_algorithm_configs()`.** Remove `tuned_hyperparameters` and per-model override parameters, the tuned-profile branch, and the model override dictionaries. Read GATMA settings and MADDPG epsilon schedule from config. Keep device arguments and shared GAE/minibatch/replay-action switches. Make the main runner pass only those retained arguments. Remove `TUNED_MODEL_FILES` and `load_tuned_hyperparameters()` if no remaining imports use them.
- [ ] **Step 5: Verify settings and tests.** Run the three test modules above. Compare constructor kwargs, batch sizes, replay capacities, and update counts for the six requested agents against `/tmp/comparison_runner_pre_cleanup_configs.json`; values should match exactly because changing their source must not change behavior. Confirm masks, warmup settings, GATMA update count, and both device choices. Commit only separable task hunks after reviewing the staged diff; preserve existing user edits in the runner.

### Task 3: Verify provenance and update active documentation

**Files:**
- Modify: `run_comparision.py`, `utils/comparison/outputs.py`.
- Test: `tests/test_comparison_diagnostics.py`, `tests/test_experiment_tracking.py`, `tests/test_local_artifact_workflow.py`.
- Modify: `docs/dual_gat_mappo_architecture.md`, `docs/wandb_tracking.md`, `note.txt`.

**Interfaces:**
- Consumes: resolved `args`, `PAPER_PARAMS`, selected algorithm configurations.
- Produces: W&B and local run records containing effective settings, and docs whose active commands use only retained flags.

**Test and implementation anchors:**

```python
# tests/test_comparison_diagnostics.py
row = build_last_training_state_line(
    "MAPPO", {"reward": [1.0]}, 1,
    agent_kwargs={"use_gae": True, "num_minibatches": 4},
    reward_weights={"lambda5": 1.0, "p_out_value": -1.0},
    maddpg_updates_per_episode=16,
)
assert row["agent_kwargs"]["num_minibatches"] == 4
assert row["reward_weights"]["lambda5"] * row["reward_weights"]["p_out_value"] == -1.0
assert row["maddpg_updates_per_episode"] == 16
```

```python
# utils/comparison/outputs.py: new optional parameters at the end of
# build_last_training_state_line(...).
agent_kwargs: Optional[Dict[str, object]] = None,
reward_weights: Optional[Dict[str, float]] = None,
maddpg_updates_per_episode: Optional[int] = None,
# Before return row:
if agent_kwargs is not None:
    row["agent_kwargs"] = dict(agent_kwargs)
if reward_weights is not None:
    row["reward_weights"] = dict(reward_weights)
if maddpg_updates_per_episode is not None:
    row["maddpg_updates_per_episode"] = int(maddpg_updates_per_episode)
```

- [ ] **Step 1: Add a provenance assertion.** Test that run metadata includes selected agent kwargs, experiment and topology seeds, server profile, reward weights including `lambda5=1.0` and `p_out_value=-1.0`, task-priority mode, GAE/minibatch settings, and MADDPG replay switches. Use a small recording tracker or output builder fixture; do not launch 1000 episodes.
- [ ] **Step 2: Run the provenance test red, then add only missing fields.** W&B already records `agent_kwargs`, so preserve it. Pass resolved agent kwargs, reward weights, and replay-update count from the runner to `build_last_training_state_line()` for the local JSONL row. Existing checkpoints already save `agent_kwargs`; preserve that field. Add reward weights to the checkpoint metadata using the same resolved config values.
- [ ] **Step 3: Update documentation.** In the architecture document, list retained CLI flags and explain that model settings come from `paper_config.py`; remove claims that `--hyperparameters-dir` remains active. In W&B tracking docs, replace active tuning-flag examples with config-edit instructions. In `note.txt`, comment out the three old executable `--hyperparameters-dir` commands as historical and provide the current modular-cell command using only retained options.
- [ ] **Step 4: Verify the public command and tests.** Run `/home/daniel_window/miniconda3/envs/task_offloading/bin/python run_comparision.py --help` and check retained flags appear and removed tuning/profile flags do not. Run focused tracking/artifact tests and `/home/daniel_window/miniconda3/envs/task_offloading/bin/python -m pytest tests/test_topology_scenarios_config.py tests/test_local_artifact_workflow.py tests/test_comparison_diagnostics.py tests/test_graph_gat_mappo.py tests/test_gatma.py tests/test_experiment_tracking.py -q`. Record pre-existing unrelated failures separately. Run `git diff --check` on task files.
- [ ] **Step 5: Review and stage only task hunks.** Inspect `git diff --cached` before any commit. Leave mixed files uncommitted if the existing user edits cannot be separated safely. Report the effective model settings and any remaining test limitations.
