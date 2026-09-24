# W&B Training Monitoring

W&B tracking is optional and disabled by default. Each algorithm is recorded as
one W&B run, while all algorithms launched by one comparison command share a
group so their curves can be compared directly.

The online mode transmits experiment configuration, episode metrics, and W&B
system telemetry. This integration does not upload the dataset, images,
checkpoints, or source code (`save_code=False`).

## Install on Windows

From the project environment:

```powershell
pip install -r requirements-wandb.txt
wandb login
```

Alternatively, set `WANDB_API_KEY` in the Windows environment instead of
placing a key in source code. Never commit the API key.

## Live online monitoring

```powershell
python run_comparision.py `
  --topology-scenario medium_20d_6s `
  --episodes 3 `
  --graph-gat-device cuda `
  --wandb-mode online `
  --wandb-project industrial-task-offloading `
  --wandb-group medium-gpu-smoke-seed75 `
  --note medium-gpu-smoke
```

Set `wandb_entity` in `utils/paper_config.py` when runs belong to a W&B
team rather than the account configured by `wandb login`.

## Offline monitoring

Use offline mode when the training machine cannot reach W&B reliably:

```powershell
python run_comparision.py `
  --topology-scenario large_30d_9s `
  --episodes 3 `
  --graph-gat-device cuda `
  --wandb-mode offline `
  --wandb-project industrial-task-offloading `
  --note large-gpu-smoke
```

Preserve the generated `wandb/` directory. It can be synchronized later from a
machine with Internet access using the `wandb sync` command printed by W&B.

## Logged metrics

One record is logged after every completed episode:

- reward, delay, and energy;
- local/edge action ratios and requested/resolved action counts;
- penalty count and simulated penalty time;
- simulated local, server, transfer, and queue/wait time under `simulation/*`;
- Graph-GAT graph-build, warmup, action-selection, PPO-update time, and rollout
  transition count;
- wall-clock DAG generation, priority inference, slot initialization, action
  collection, environment step, metric summary, rollout storage, model update,
  connection-window, joint-state, accounted, and unaccounted time;
- connection-window requests, actual updates, and sampled-point count;
- episode duration, elapsed time, progress percentage, and ETA;
- explicit CUDA allocated/reserved/peak memory for Graph-GAT, in addition to
  W&B system telemetry.

The project calls `run.log()` once per episode with an explicit episode step, so
metrics from the same episode stay aligned.

The legacy `execution/*` metrics remain for compatibility. They are simulated
system delays, not CPU profiler timings. Use `runtime/*` when investigating
training speed.

## Post-optimization CPU/CUDA validation

After synchronizing the environment connection-window update, launch these as
separate commands so the only intended difference is `--graph-gat-device`.
The `--algorithms` filter prevents the rest of the ablation matrix from running.

CPU:

```powershell
python run_comparision.py `
  --topology-scenario large_30d_9s `
  --episodes 20 `
  --algorithms "Graph-GAT MAPPO" `
  --graph-gat-device cpu `
  --wandb-mode online `
  --wandb-project industrial-task-offloading `
  --wandb-group env-repair-large-seed75 `
  --note env-repair-large-cpu-20ep
```

CUDA:

```powershell
python run_comparision.py `
  --topology-scenario large_30d_9s `
  --episodes 20 `
  --algorithms "Graph-GAT MAPPO" `
  --graph-gat-device cuda `
  --wandb-mode online `
  --wandb-project industrial-task-offloading `
  --wandb-group env-repair-large-seed75 `
  --note env-repair-large-cuda-20ep
```

Compare medians over episodes 2-20 for `training/episode_seconds`,
`runtime/env_step_seconds`, `runtime/connection_window_seconds`,
`runtime/action_collection_seconds`, `runtime/rollout_storage_seconds`, and
`runtime/model_update_seconds`.

## Model hyperparameters for new comparisons

Edit and version the model settings in `utils/paper_config.py` before launching
an experiment. The runner no longer accepts per-model tuning flags or the old
`--hyperparameters-dir` profile. Record the code revision and effective agent
settings with each run; W&B and local outputs save the latter. To compare a
new candidate, change the config in a separate code revision and run the same
model list, topology, and seed budget. Historical tuning commands belong to
older results and should not be reused for the current `-1` penalty setting.

## Disable tracking

No W&B installation or login is required in the default mode:

```powershell
python run_comparision.py --wandb-mode disabled
```

Official references:

- [W&B Python `init` reference](https://docs.wandb.ai/models/ref/python/functions/init)
- [W&B environment variables](https://docs.wandb.ai/models/track/environment-variables)
- [W&B run initialization](https://docs.wandb.ai/models/runs/initialize-run)
