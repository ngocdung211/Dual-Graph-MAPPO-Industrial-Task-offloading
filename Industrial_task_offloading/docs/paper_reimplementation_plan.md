# Paper Reimplementation Plan and Checklist

**Goal:** Reimplement the paper environment and MADDPG path with minimum code changes, then compare trained DRL models against the currently enabled comparison baselines.

**Success criteria:**
- The environment follows the paper's task, mobility, communication, computation, MDP state, action, reward, and constraint definitions closely enough that training results are meaningful.
- MADDPG is verified first in isolation before comparing against baselines.
- Baseline comparisons are only trusted after environment invariants pass.
- If MADDPG cannot beat simple baselines after the model update path is verified, treat that as evidence to re-audit the environment against the paper before tuning the model.

**Working rule:** Every small task updates this checklist. Code changes should be surgical and limited to functions needed for the current verified goal.

---

## Active Research Plan - Updated 2026-07-14

**Status:** This section is the active plan. Older open queues below are historical context unless a task is explicitly moved here.

**Current research questions:**
- Does `Graph-GAT Mask MAPPO` fail because the topology encoder is trained from scratch inside PPO?
- Does `Graph-GAT Mask MAPPO` help more when the topology is larger, denser, and harder than the current 10-device/3-server map?
- Why does CPU runtime grow from about 1.5 hours on the 10-device map to 8 hours on the 20-device map and 16 hours on the 30-device map for Graph-GAT MAPPO?
- After batching Graph-GAT, which environment operations account for the remaining wall-clock time, and can they be optimized without changing the simulated delay, reward, mobility, or connection-window semantics?
- How should the physical system, digital-twin belief state, task-level GAT, and topology-level GAT be connected so the implementation matches the proposed dual-GAT DITEN architecture?
- How can we prove `GAT Scheduling` is better than `GCN Scheduling` for task priority, instead of only showing one lucky training curve?

**Success criteria:**
- Pretrained Topology-GAT experiments compare at least three variants under the same seeds and episode budget: scratch, pretrained-frozen, and pretrained-finetuned.
- Larger-topology experiments report topology complexity metrics, not only reward/delay/energy.
- The Graph-GAT CPU path is profiled before optimization, preserves actor locality and numerical behavior, and becomes at least 5x faster on a fixed medium/large benchmark.
- Environment wall-clock profiling accounts for at least 90% of an episode, clearly separates simulated seconds from runtime seconds, and preserves fixed-seed connection windows and reward trajectories after optimization.
- The final proposal explicitly separates physical truth from digital-twin observations and fuses the current-subtask GAT embedding, topology-GAT embedding, and DT scalar state before action selection.
- GAT-vs-GCN task-priority evidence is separated into supervised priority quality, fixed-policy scheduling quality, and end-to-end DRL impact.
- Every result includes the exact config, checkpoint path, seed list, final JSONL row, and plot path.

### Phase 0: Close the Previous Plan and Fix Drift

Goal: make the plan match the code before adding new experiments.

- [x] Treat the old `Next Task Queue`, old `Ablation Study Queue`, and old `Implementation Checklist` as historical context.
- [x] Move the remaining active work into this 2026-07-08 section.
- [x] Select a working local CPU PyTorch runtime for tests on this Mac.
  - The base Conda environment remains broken because `libtorch_cpu.dylib` is missing.
  - Use `/Users/admin/miniconda3/envs/task_offloading/bin/python`; it imports PyTorch `2.13.0` successfully.
  - Verify: `/Users/admin/miniconda3/envs/task_offloading/bin/python -c "import torch; print(torch.__version__)"` works.
- [x] Align model names across plan, config, tests, plots, and JSONL.
  - Standard names: `Graph-GAT MAPPO` for the unmasked ablation and `Graph-GAT Mask MAPPO` for the masked proposal.
  - Reason: these names make clear that the GAT is the topology encoder, not the task-priority GAT.
  - Verify: `tests/test_graph_gat_mappo.py` expects the same names returned by `run_comparision.build_algorithm_configs`.
- [x] Defer GPU-device work until the CPU path is profiled and refactored.
  - Decision: fix the avoidable Python-loop and repeated-encoding cost first.
  - Impact: optional CUDA support may be reconsidered only after CPU numerical-equivalence and scaling tests pass.

### Phase 1: Pretrained Topology-GAT Experiment

Goal: test whether the topology encoder helps more when it starts from useful topology knowledge instead of random weights.

Assumption: pretraining should learn connection feasibility and topology quality from graph states before PPO begins. PPO then either freezes that encoder or fine-tunes it.

1. [ ] **Define the Topology-GAT pretraining target**
   - Input: `TopologyGraphState` from `utils/topology_graph_state.py`.
   - Encoder: `models/topology_gat.py::TopologyGATEncoder`.
   - Minimal supervised target:
     - per-device feasible server mask from edge feature `is_connected`;
     - optional per-device best-server target using lowest edge wait and longest valid window.
   - Verify:
     - target can be computed from graph tensors only;
     - no environment physics or reward function changes.

2. [ ] **Add topology pretraining dataset generation**
   - Goal: collect graph states from random or scripted rollouts without training MAPPO.
   - Candidate file: `utils/topology_gat_pretraining.py`.
   - Output:
     - train/validation graph samples;
     - feasible action mask targets;
     - optional best-server labels.
   - Verify:
     - dataset contains connected and disconnected links;
     - report average feasible servers per device.

3. [ ] **Train and save a pretrained Topology-GAT checkpoint**
   - Candidate checkpoint: `models/checkpoints/topology_gat_pretrained.pt`.
   - Metrics:
     - feasible-mask binary accuracy or F1;
     - best-server accuracy if the best-server head is used;
     - validation loss.
   - Verify:
     - checkpoint saves encoder weights separately from the pretraining head;
     - loading the encoder into `GraphGATMAPPOAgent` changes initial encoder weights.

4. [ ] **Add three Graph-GAT MAPPO topology modes**
   - `scratch`: current behavior, encoder trains from random initialization during PPO.
   - `pretrained_frozen`: load pretrained encoder, freeze encoder, train actor and critic only.
   - `pretrained_finetune`: load pretrained encoder and continue PPO updates through encoder.
   - Verify:
     - frozen mode excludes encoder parameters from the optimizer;
     - finetune mode includes encoder parameters in the optimizer;
     - all three modes produce one final JSONL row with mode name and checkpoint path.

5. [ ] **Run smoke tests for all topology modes**
   - Run: 1-3 episodes each.
   - Compare:
     - `Graph-GAT Mask MAPPO scratch`;
     - `Graph-GAT Mask MAPPO pretrained_frozen`;
     - `Graph-GAT Mask MAPPO pretrained_finetune`.
   - Verify:
     - no crash;
     - reward, delay, and energy are finite;
     - requested edge actions do not target disconnected servers when mask is enabled;
     - graph build/action/update cost is printed.

6. [ ] **Run short learning comparison**
   - Run: 10-30 episodes first, same seeds and same episode budget.
   - Expected interpretation:
     - frozen better than scratch means topology features are useful but PPO damages the encoder;
     - finetune better than frozen means pretraining is useful and PPO can improve it;
     - scratch matching both means current topology task may be too easy or features are already sufficient.

### Phase 2: Larger and Denser Topology Stress Test

Goal: test whether Graph-GAT Mask MAPPO only shows value when topology is complex enough.

1. [x] **Add named topology scenario configs**
   - Keep current paper-like scenario as `paper_10d_3s`.
   - Add `medium_20d_6s`.
   - Add `large_30d_9s` (the current layout contains 9 servers).
   - Config controls:
     - map size;
     - number of devices;
     - number of servers;
     - server locations;
     - coverage radius;
     - device route templates.
   - Verify:
     - old `paper_10d_3s` results are reproducible with the new config path.
     - Scenario source: `utils/topology_scenarios.py`.
     - Preview source: `utils/topology_scenario_preview.py`.
     - Runner selection: `run_comparision.py --topology-scenario <name>`.

2. [x] **Add topology complexity diagnostics**
   - Metrics:
     - average feasible servers per device;
     - percent of devices with zero feasible servers;
     - graph edge count;
     - graph density;
     - edge request success rate;
     - mask rejection count or masked-action count;
     - graph build/action/update time.
   - Verify:
     - static scenario diagnostics are saved in final JSONL/checkpoints and printed before training.
     - Static scenario diagnostics are saved in final JSONL/checkpoints:
       `topology_avg_feasible_servers`, `topology_density`,
       `topology_zero_link_ratio`, `topology_multi_link_ratio`, and
       `topology_device_server_ratio`.

3. [ ] **Run topology scale smoke tests**
   - Run 1-3 episodes for:
     - `Mask-MAPPO`;
     - `Graph-GAT Mask MAPPO scratch`;
     - best pretrained mode from Phase 1, if available.
   - Scenarios:
     - `paper_10d_3s`;
     - `medium_20d_6s`;
     - `large_30d_9s`.
   - Verify:
     - no scenario has impossible all-disconnected or all-connected topology unless intentionally configured;
     - reward/delay/energy are finite.

4. [ ] **Run controlled scale comparison**
   - Run: same seeds, same episode count, same priority model, same reward config.
   - Compare:
     - `Mask-MAPPO`;
     - `Graph-GAT Mask MAPPO scratch`;
     - `Graph-GAT Mask MAPPO pretrained_frozen`;
     - `Graph-GAT Mask MAPPO pretrained_finetune`.
   - Success signal:
     - GAT advantage should increase when average feasible server count and graph density increase.
   - If GAT still equals Mask-MAPPO:
     - inspect attention weights or edge-feature sensitivity;
     - check whether flat state already gives MAPPO all topology information in an easy-to-learn order;
     - do not tune reward before checking those explanations.

### Phase 2.5: Current Ablation Study Matrix

Goal: prove which implemented component improves MAPPO before adding pretrained Topology-GAT.

**Status - 2026-07-13:** The remote screen reported `All training complete` and displayed saved plots, checkpoints, and the final JSONL path. Do not restart it. The outputs still need to be copied/imported and audited locally before marking the ablation rows complete.

**Monitoring checkpoint - 2026-07-13:** Optional W&B monitoring is implemented with disabled, online, and offline modes. Each algorithm uses a separate run in one comparison group and logs one aligned record per episode. Online mode does not upload datasets, checkpoints, or source code. Windows setup and commands are documented in `docs/wandb_tracking.md`.

**Observed wall-clock cost reported during the run:**
- 10 devices: Graph-GAT MAPPO takes about 1.5 hours.
- 20 devices: MAPPO takes about 1.5 hours, while Graph-GAT MAPPO takes about 8 hours.
- 30 devices: Graph-GAT MAPPO takes about 16 hours.

1. [ ] **Run core component ablation on the medium map**
   - Scenario: `medium_20d_6s`.
   - Same seed from `utils/paper_config.py::PAPER_PARAMS["provisional_table2_needed"]["experiment_seed"]`.
   - Same episode budget from `comparison_full_episodes` or the command `--episodes`.
   - Compare:
     - `MAPPO`;
     - `Mask-MAPPO`;
     - `Graph-GAT MAPPO`;
     - `Graph-GAT Mask MAPPO`;
     - `Graph-GAT Warmup MAPPO`;
     - `Graph-GAT Warmup Mask MAPPO`.
   - Verify:
     - final JSONL has one row per model;
     - note includes map, purpose, episode count, and seed;
     - reward, delay, energy, penalty count, local/edge ratio, and graph warmup metrics are saved.

2. [ ] **Run core component ablation on the large map**
   - Scenario: `large_30d_9s`.
   - Use the same model list and seed as the medium run.
   - Purpose: show whether Graph-GAT helps more when there are more agents, servers, and overlapping coverage choices.
   - Success signal:
     - `Graph-GAT Mask MAPPO` or `Graph-GAT Warmup Mask MAPPO` improves more over `Mask-MAPPO` on the large map than on the medium map.

3. [ ] **Run paper-map control ablation if medium/large results are promising**
   - Scenario: `paper_10d_3s`.
   - Purpose: show the small topology is easier, so the GAT advantage may be smaller.
   - Use this as a control, not the main evidence if the small map is too simple.

4. [ ] **Summarize ablation evidence**
   - Required tables:
     - final reward/delay/energy by model and map;
     - penalty count and edge success behavior;
     - topology metrics from JSONL.
   - Required plots:
     - reward curve;
     - delay curve;
     - energy curve.
   - Interpretation:
     - `Mask-MAPPO - MAPPO` estimates the value of action masking;
     - `Graph-GAT MAPPO - MAPPO` estimates the value of topology embedding;
     - `Graph-GAT Mask MAPPO - Mask-MAPPO` estimates the added value of GAT after masking;
     - `Graph-GAT Warmup Mask MAPPO - Graph-GAT Mask MAPPO` estimates the value of online topology warmup.

---

### Phase 2.6: Graph-GAT CPU Runtime and Scaling Repair

Goal: explain and remove avoidable Graph-GAT cost before rerunning medium/large ablations or adding more model variants.

**Source-audit findings to verify with a profiler:**
- `GraphGATMAPPOAgent._encode_local_actor_embeddings` performs one encoder call per device.
- `_local_subgraph_for_device` rebuilds tensors and scans the complete topology edge list for every device.
- `_policy_and_values_for_graphs` loops over every rollout transition for every PPO epoch, then separately encodes local actor graphs and the global critic graph.
- With `P` PPO epochs and `N` devices, each transition causes approximately `(P + 1)N + P + 2` encoder forwards across action collection and update; with `P=4`, this is `5N+6` forwards.
- `TopologyGraphAttentionLayer.forward` loops over every node and builds a boolean mask over all edges in each of its two layers.
- The graph currently stores both connected and disconnected device-server pairs, so it carries `2NS` directed edges even though the builder documentation calls them valid links.
- Repeated `.tolist()`, Python dictionaries, small tensor construction, and unbatched CPU kernels amplify overhead as devices and servers increase.

**Implementation checkpoint - 2026-07-13:**
- The local Actor topology path now uses one batched dense bipartite attention call for all devices instead of `N` sequential local-graph encoder calls.
- Warmup targets and action masks now use the ordered `(N, S, 2, E)` edge tensor directly instead of scanning edge tuples.
- The centralized critic now uses dense bipartite global attention instead of the per-node incoming-edge loop.
- PPO evaluation now stacks the complete rollout as `(T, N, S, ...)` tensors. With four PPO epochs, encoder calls are reduced from approximately `(5N+6)T` in the original path to `T+10`: `T` online local-Actor calls during environment collection, two batched global value calls, and eight batched Actor/Critic calls across the four PPO epochs.
- Sequential-reference verification passes for embeddings and encoder gradients; the maximum observed embedding difference is `1.79e-7`, and the actor-locality test remains exact.
- A single-thread local-Actor microbenchmark measured approximately `6.1x`, `13.3x`, and `27.0x` speedup for 10/3, 20/6, and 30/10 device/server sizes. These are local-Actor measurements, not end-to-end training claims.
- A single-thread global-encoder microbenchmark measured approximately `2.3x`, `3.9x`, and `5.5x` speedup for 10/3, 20/6, and 30/10 device/server sizes, with maximum embedding difference `1.79e-7`.
- A synthetic PPO forward/backward benchmark with `T=20`, one CPU thread, and 64-dimensional embeddings measured `5.98x`, `2.78x`, and `1.52x` speedup for 10/3, 20/6, and 30/10 device/server sizes. This isolates model evaluation and backpropagation; it is not an end-to-end training claim.
- Batched rollout log-probabilities, values, and gradients match the sequential reference within `1e-6`; relevant Graph-GAT/topology tests: 25/25 pass in the `task_offloading` Conda environment.

1. [ ] **Preserve and annotate the running ablation**
   - Record the exact command, scenario, seed, episode count, priority model, PPO epochs, and warmup settings.
   - Save elapsed time separately for graph build, warmup, action selection, rollout update, and total model time.
   - Do not use this long run as the optimization benchmark because its models have different workloads.
   - Verify: every completed model has a checkpoint/final JSONL row, and incomplete models are clearly labeled rather than silently rerun.

2. [ ] **Build a deterministic scaling benchmark before changing code**
   - Run the same fixed number of transitions and PPO updates for `paper_10d_3s`, `medium_20d_6s`, and `large_30d_9s`.
   - Add counters for local-subgraph builds, local encoder forwards, global encoder forwards, actor time, critic time, and backward time.
   - Use `cProfile` and a PyTorch CPU profiler on a 1-3 episode run after the local PyTorch runtime is repaired.
   - Verify: the measured encoder-call count matches the expected formula and the top functions explain most of the Graph-GAT/MAPPO gap.

3. [x] **Remove repeated local-subgraph construction**
   - Reshape the builder's ordered edge features directly into `(N, S, 2, E)` tensors.
   - Represent all device-local bipartite graphs as one batched tensor operation instead of creating `N` `TopologyGraphState` objects per forward.
   - Eliminate `.tolist()`, Python membership checks, and per-device full-edge scans from the hot path.
   - Preserve decentralized actor semantics: changing another device's private state must not change the selected device's actor probabilities.
   - Verify: fixed-seed local embeddings and encoder gradients match the sequential implementation within `1e-6`; actor locality remains unchanged.

4. [x] **Vectorize topology attention and action masks**
   - Completed for device-local Actor attention, centralized global critic attention, topology warmup targets, and action masks.
   - Replace the per-node incoming-edge loop with batched dense bipartite attention or indexed/scatter aggregation.
   - Build the feasibility matrix directly as shape `(num_devices, num_servers)` and derive action masks without scanning edge tuples.
   - Keep disconnected pairs as present edges with an explicit disconnected feature so the optimized path remains equivalent to existing experiments.
   - Verify: complete-graph embeddings, centralized critic values, and encoder gradients match the sequential graph implementation within `1e-6`.

5. [x] **Batch rollout evaluation across transitions**
   - Stack fixed-shape graph tensors across the rollout time dimension.
   - Compute policy log-probabilities, entropy, and critic values in batches for each PPO epoch.
   - Avoid recomputing embeddings that are identical within the same loss evaluation, while retaining gradients where the encoder is trainable.
   - Verify: policy outputs, critic values, and encoder/Actor/Critic gradients match the sequential implementation within `1e-6`; PPO update uses `2 + P` global rollout batches and `P` local rollout batches.

6. [ ] **Benchmark each optimization separately**
   - Report wall time, encoder-call count, peak memory, and speedup for 10/20/30 devices.
   - Primary target: at least 5x faster than the current Graph-GAT path on a fixed medium/large CPU benchmark.
   - Scaling target: runtime growth should be explained by tensor sizes, not nested Python loops.
   - Only after CPU equivalence passes, evaluate optional CUDA/device support and `torch.compile`; do not use GPU migration to hide an inefficient algorithm.

7. [ ] **Rerun only the interrupted or invalidated ablation rows**
   - Existing completed results remain valid if the optimized implementation is numerically equivalent.
   - If behavior changes beyond tolerance, label old/new implementations separately and rerun the full controlled matrix.

### Phase 2.6.1: Environment Wall-Clock Profiling and Connection-Window Repair

Goal: remove the new dominant CPU cost after Graph-GAT batching, while preserving the exact environment trajectory and scientific meaning of the experiment.

**Metric-semantics correction - 2026-07-14:**
- W&B metrics `execution/wait_seconds` and `penalty/seconds` are simulated scheduling delays. They are outputs of the physical model, not CPU wall-clock measurements and must not be used to locate runtime bottlenecks.
- The controlled five-episode `large_30d_10s` comparison completed in 136.43 seconds on CPU and 133.58 seconds on RTX A4000 CUDA, only about `1.02x` end-to-end speedup.
- On the final episode, Graph-GAT action plus PPO update took 1.66 seconds on CPU and 0.98 seconds on CUDA. CUDA accelerated the PPO update by about `3.27x`, but this model-only saving is a small part of the episode.

**Local source audit and microbenchmark evidence - 2026-07-14:**
- `_update_connection_windows` loops in Python over every device, server, and `subslot_count + 1` sample and calls `np.linalg.norm` on each two-element location separately.
- With `subslot_count=200`, an episode invokes the function 101 times: once in `reset_episode`, 50 times in `start_time_slot`, and 50 times after each completed slot in `step`. Consecutive calls often recompute the same slot, and the final call computes a terminal window that may not be consumed.
- Approximate sampled coverage checks per episode are 609,030 for `paper_10d_3s`, 2,436,120 for `medium_20d_6s`, and 6,090,300 for `large_30d_10s`.
- Measured local cost for 101 window updates is about 1.42, 5.61, and 14.06 seconds for the 10/3, 20/6, and 30/10 scenarios.
- A controlled large all-local environment episode took 14.04 seconds normally and 0.24 seconds when repeated window recomputation was disabled after creating an initial valid window. The 13.80-second difference is about 98% of this environment-only microbenchmark; it is not yet an end-to-end remote training result.
- Generating 50 slots of DAGs from 798 local images plus 1,500 unbatched Task-GAT priority forwards took about 0.39 seconds locally. Keep these as secondary candidates unless the full profiler ranks them higher on Windows or the synced drive.

**Implementation and verification checkpoint - 2026-07-14:**
- `DITENEnv` now records connection-window requests, actual updates, sampled points, connection-window wall time, joint-state wall time, and joint-state calls. The comparison runner separately records DAG generation, priority inference, slot initialization, action collection, environment steps, metric summarization, rollout storage, model update, accounted time, unaccounted time, and tracker-log time.
- W&B retains the legacy `execution/*` keys for compatibility and adds explicit `simulation/*` keys so modeled queue/transfer/penalty seconds are not confused with `runtime/*` wall-clock seconds.
- An unchanged slot/position/direction/server signature now reuses the cached windows. A normal 50-slot episode still makes 101 requests but performs only 51 actual updates.
- Sampled positions and device-server distances are evaluated with NumPy broadcasting. The first connected sample and first subsequent disconnected sample preserve the previous sequential semantics.
- Vectorized windows match the sequential reference within `1e-12` on all 10/3, 20/6, and 30/10 scenario tests. Cache invalidation and slot-transition tests pass.
- Three-repeat deterministic benchmark medians after repair are 0.034/0.086/0.179 seconds total environment time for 10/3, 20/6, and 30/10. Connection-window time is 0.008/0.025/0.065 seconds.
- The matched large all-local environment protocol fell from 14.04 seconds before repair to 0.324 seconds after repair, about `43x` faster. The deterministic fixed-task benchmark is 0.182 seconds, about `77x` faster than the earlier environment-only observation, but the fixed-task protocol excludes per-slot image/DAG generation.
- A one-episode large Graph-GAT MAPPO CPU smoke run completed in 3.62 seconds. Accounted phases totaled 3.61 seconds with 0.010 seconds unaccounted; connection windows used 0.073 seconds, environment steps 0.206 seconds, action collection 0.739 seconds, rollout storage 0.428 seconds, and PPO update 1.993 seconds.
- Post-repair joint-state construction is about 0.073 seconds in the large deterministic benchmark and is below 5% of the 3.62-second Graph-GAT smoke episode. No secondary environment optimization is justified yet.

1. [x] **Add true wall-clock phase instrumentation**
   - Measure DAG generation, task-priority inference, `start_time_slot`, `env.step`, connection-window updates, joint-state construction, Graph-GAT rollout-buffer push, optimizer update, W&B logging, and unaccounted episode time with `time.perf_counter`.
   - Log connection-window call count and sampled pair count per episode.
   - Keep simulated delay metrics, but expose them under an explicitly simulated namespace or label so they cannot be confused with profiler timings.
   - Verify: timed phases account for at least 90% of episode wall time and their sum does not materially increase runtime.

2. [x] **Add a deterministic environment scaling benchmark**
   - Benchmark exactly 50 slots and five subtasks per slot on 10/3, 20/6, and 30/10 with fixed DAGs, actions, seed, and no W&B/network upload.
   - Capture `cProfile` output and per-function timers for the normal path.
   - Verify: results report total environment time, connection-window time/calls, state-build time, scheduling time, and checks per second for every topology.

3. [x] **Remove duplicate connection-window recomputation**
   - Give one lifecycle point ownership of each slot's connection-window update, or cache by current slot plus device position/direction so repeated calls are no-ops.
   - Do not remove a terminal update until tests prove the terminal `next_state` and rollout semantics do not consume it.
   - Verify: fixed-seed windows, states, masks, actions, rewards, penalties, and done flags match the current implementation exactly; measured calls fall from 101 to at most 51 per episode.

4. [x] **Vectorize sampled connection-window evaluation**
   - Precompute the `(subslot_count + 1)` relative time offsets.
   - Use NumPy broadcasting to evaluate device positions and distances for all device-server-time samples in batches instead of millions of tiny Python/NumPy calls.
   - Preserve the current sampled first-entry/first-exit behavior before considering an analytical line-circle intersection implementation.
   - Verify: every `(window_start, window_end)` matches the sequential reference within `1e-12` across all scenarios, route segments, boundary contacts, always-disconnected links, and links that remain connected through the horizon.

5. [x] **Evaluate secondary environment costs and defer unjustified changes**
   - Candidates: batch the 1,500 Task-GAT priority forwards, cache image metadata instead of reopening image files from the synced drive, reuse static priority adjacency/features, and reduce temporary Python dictionaries in `_get_joint_state` and `step`.
   - Do not add these changes based only on asymptotic suspicion; require a measured contribution of at least 5% after connection-window repair.
   - Verify each accepted optimization independently against fixed-seed outputs and wall time.

6. [ ] **Run end-to-end CPU/CUDA validation after environment repair**
   - Run matched 20-30 episode CPU and CUDA Graph-GAT MAPPO jobs on `large_30d_9s`, same seed and configuration, excluding episode one from median timing.
   - Report median episode time, environment time, connection-window time, graph build/action/update time, unaccounted time, and GPU memory.
   - Success targets: at least `5x` faster connection-window evaluation, at least `3x` faster environment-only large benchmark, and no fixed-seed trajectory difference. Treat end-to-end speedup as a measured result rather than a guaranteed target.

### Phase 2.6.2: Graph-GAT 1000-Episode Stability Tuning

Goal: determine whether stable PPO/encoder optimization lets masked Graph-GAT
outperform Mask-MAPPO, without changing the environment, reward, or action
mask.

**500-episode evidence - 2026-07-15:**
- All 24 runs completed for the 10/3, 20/6, and 30/10 maps with seed 75.
- Action masking is mandatory: final unmasked Graph-GAT penalty counts were
  1,688, 4,118, and 6,718, while masked Graph-GAT counts were 9, 36, and 108.
- Mask-MAPPO won the paper and large maps. Graph-GAT Warmup Mask MAPPO only
  won the medium map, so the topology encoder has not shown a consistent
  scaling benefit.
- Unmasked Graph-GAT reward collapsed late in training. Do not spend the new
  budget tuning unmasked variants.

1. [x] **Expose Graph-GAT optimization hyperparameters through the CLI**
   - Overrides cover actor/critic LR, encoder LR, hidden/embedding width, PPO
     clip/epochs, entropy coefficient, value-loss coefficient, gradient norm,
     and topology-warmup duration/update rate/LR.
   - Defaults preserve the 500-episode implementation.
   - Verify: overrides apply only to Graph-GAT; warmup overrides apply only to
     variants whose names include `Warmup`.

2. [ ] **Run the unchanged 1000-episode medium control**
   - Models: `Mask-MAPPO` and `Graph-GAT Warmup Mask MAPPO`.
   - Keep seed 75 and all current hyperparameters.
   - Purpose: separate the effect of a longer run from the effect of tuning.

3. [ ] **Run stability candidate v1 on the medium map**
   - Actor/critic LR `8e-5`; encoder LR `3e-5`; PPO clip `0.15`;
     entropy coefficient `0.005`; value-loss coefficient `0.5`; gradient norm
     `0.5`.
   - Warmup: 20 episodes, two updates per step, LR `3e-4`.
   - Keep hidden and embedding dimensions at 64 so this experiment isolates
     optimization stability rather than capacity.
   - Select using final-100-episode mean and standard deviation for reward,
     delay, energy, penalty count, and edge ratio.

4. [ ] **Transfer the selected configuration without per-map retuning**
   - Reuse the exact medium-selected configuration on `paper_10d_3s` and
     `large_30d_9s` for 1000 episodes.
   - Compare against Mask-MAPPO with the same seed and episode budget.
   - Do not claim improvement unless Graph-GAT is better across multiple
     metrics and remains stable through episodes 900-1000.

5. [ ] **Confirm the two finalists across multiple seeds**
   - Candidates: Mask-MAPPO and the selected Graph-GAT Warmup Mask MAPPO.
   - Add seed control before this step, then run at least three seeds.
   - Report mean and standard deviation rather than the best seed.

---

### Phase 2.7: Digital-Twin and Dual-GAT Architecture Alignment

Goal: implement the proposed model as a real DT-assisted dual-graph policy rather than a topology-GAT policy attached to a partially DT-like flat state.

**Minimal DT foundation - 2026-07-22:**
- [x] Add a `DigitalTwin` logic class and per-slot `DigitalTwinSnapshot` with device/server CPU estimates.
- [x] Preserve a perfect-synchronization mode when estimation error is zero and bounded noisy CPU estimates otherwise.
- [x] Keep physical device/server CPU values unchanged and expose the synchronized snapshot through `DITENEnv`.
- Current scope is deliberately small: topology, queues, tasks, and connection windows still use the existing exact environment state.
- Deferred below: task twin state, synchronization delay/staleness, confidence/deviation fields, noisy connectivity, serialization metadata, and policy observation isolation.

**Baseline environment calibration - 2026-07-28:**
- [x] Seed server and device CPU sampling before physical entity creation and save actual CPU min/mean/max/std plus seeds in final JSONL.
- [x] Use actual CPU for physical compute delay while retaining estimated CPU in the DT observation.
- [x] Make Edge Only choose local when no estimated edge execution fits a connection window.
- [x] Treat failed offload as rejected before upload; fallback-local transfer accounting follows the resolved physical path.
- [x] Rename the current nine-server large scenario to `large_30d_9s`.
- [x] Normalize execution timing diagnostics per device-task.
- [x] Remove the one-step lag from accumulated reward inputs.
- [x] Set `task_cpu_cycle_scale=1.35`; one-episode calibration puts Local Only at 1.15-1.46 seconds across the three named topologies.
- [ ] Improve the Paper topology Feature-vs-Local separation only after deciding whether to increase coverage or change reward weights; sparse coverage currently permits 93 of 500 feature tasks (3.7% of all tasks) to execute at edge in the one-episode calibration.

**Current implementation gap:**
- `DigitalTwin` now samples estimated device/server compute power once per slot, while `DITENEnv` exposes the snapshot and continues to own physical entities and execution.
- Physical compute delay now uses only actual CPU power; DT CPU estimates affect observations but no longer alter execution truth.
- The minimal snapshot only models CPU-estimation error. There is no complete physical-state versus twin-observation boundary, synchronization timestamp, observation delay/staleness, or recorded estimation deviation.
- Environment execution and observation construction share the same objects, making it difficult to test whether the policy receives oracle physical state.
- `TaskPriorityGAT` currently produces a priority order before environment execution; its per-subtask embedding is not passed to Graph-GAT MAPPO.
- `GraphGATMAPPOAgent` consumes topology embeddings only. It does not implement the proposed fused state `[h_task || h_topo || z_DT]` for the current subtask.

1. [ ] **Freeze the DT contract from the baseline paper and current manuscript**
   - Physical truth includes actual device/server compute capability, queues, locations, channel/connectivity, tasks, and execution results.
   - Twin belief includes estimated counterparts, last synchronization time, age/staleness, and deviation or confidence fields.
   - Decide the synchronization cadence, observation delay, and noise model from cited equations/configuration before coding.
   - Correct the manuscript notation typo `\hat{f}i^{loc}` and state explicitly which original deviation terms are retained or intentionally omitted.
   - Verify: a table maps every paper state symbol to one owner, code field, normalization, and update time.

2. [ ] **Add explicit digital-twin snapshot types**
   - Candidate types: `DeviceTwinState`, `ServerTwinState`, `TaskTwinState`, and `DigitalTwinSnapshot`.
   - Keep these as state containers first; do not change reward or environment physics in this task.
   - Verify: snapshots can be serialized in checkpoints/JSONL metadata and reconstructed deterministically from a seeded environment.

3. [ ] **Separate physical evolution from twin synchronization**
   - Physical execution uses actual compute power, queues, mobility, and connectivity.
   - Policy observations are built only from the latest twin snapshot.
   - Add configurable perfect, noisy, and stale synchronization modes.
   - Verify: perfect zero-error synchronization reproduces the legacy state and reward trajectory; noisy/stale modes change observations without directly changing physical truth.
   - [x] Minimal CPU boundary: local and edge compute delay use `f_actual`; `f_est` remains an observation field.
   - [ ] Remaining boundary: queues, mobility, and connectivity still need explicit twin-owned observation state.

4. [ ] **Define DT-aware feasibility and masking semantics**
   - The policy mask must be derived from twin-predicted connectivity, not hidden physical truth.
   - Track mask false positives, false negatives, stale-link age, and actual offload rejection/fallback outcomes.
   - Keep local execution valid for every device.
   - Verify: tests cover twin-predicted connected/actual disconnected and twin-predicted disconnected/actual connected cases.

5. [ ] **Expose task-GAT embeddings for the current subtask**
   - Change the task GAT interface so it can return per-node embeddings as well as priority scores.
   - Select the embedding of the current dependency-ready subtask for each device.
   - Keep the existing priority-order output as a separate ablation feature rather than treating it as the task embedding.
   - Verify: embeddings follow the DAG node IDs and do not violate predecessor constraints.

6. [ ] **Implement the fused dual-GAT policy state**
   - For each device/current subtask, build `[h_task || h_topo || z_DT]`.
   - Actor input uses only the device's permitted local/twin information.
   - Centralized critic may use the joint fused state during training.
   - Verify: perturbing another device's private observation does not change the selected device's actor output, while the centralized critic may change.

7. [ ] **Add DT and dual-GAT ablations**
   - Compare flat DT state, task-GAT only, topology-GAT only, dual GAT, and dual GAT plus action mask.
   - Compare perfect twin, noisy twin, and stale twin under the same seeds and topology.
   - Report DT estimation error, synchronization age, mask error rates, reward, delay, energy, and runtime.
   - Claim DT robustness only if the dual-GAT model remains better under controlled observation error/staleness.

8. [ ] **Align the manuscript after verified implementation**
   - Update equations, architecture diagram, algorithm pseudocode, complexity analysis, and experiment table from the implemented interfaces.
   - Do not state that the two GAT embeddings are fused or that DT synchronization is modeled until the corresponding tests and ablations pass.

---

### Phase 3: Prove GAT Task Priority Beats GCN Task Priority

Goal: separate task-priority quality from offloading-policy quality.

1. [ ] **Freeze the comparison protocol**
   - Use the same DAG generator, train/validation split, model hidden size policy, optimizer, epochs, and seeds for GCN and GAT.
   - Compare checkpoints:
     - `models/checkpoints/gcn_priority.pt`;
     - `models/checkpoints/gat_priority.pt`.
   - Verify:
     - both models train from scratch when running the priority-only comparison;
     - old checkpoints do not leak into the comparison unless explicitly requested.

2. [ ] **Run supervised priority-quality comparison**
   - Metrics:
     - validation loss against the current priority targets;
     - topological-order validity rate;
     - Kendall or Spearman rank agreement with target order if added.
   - Success signal:
     - GAT has lower validation loss or better ranking quality across multiple seeds.
   - Caveat:
     - if targets are simple hand-built labels, GAT beating GCN here only proves better fit to the current target, not better scheduling.

3. [ ] **Run fixed-policy scheduling comparison**
   - Goal: isolate priority order from RL learning.
   - Keep the offloading policy fixed, for example:
     - local-only;
     - edge-only first-valid;
     - a simple mask-aware heuristic.
   - Compare priority modes:
     - random;
     - greedy;
     - GCN;
     - GAT.
   - Metrics:
     - reward;
     - delay;
     - energy;
     - deadline/energy-constraint violation count if available.
   - Success signal:
     - GAT priority improves scheduling metrics under at least one fixed offloading policy without changing the DRL agent.

4. [ ] **Run end-to-end DRL priority comparison**
   - Keep the DRL algorithm fixed.
   - Recommended first algorithm: `Graph-GAT Mask MAPPO pretrained_frozen` or current best stable MAPPO variant.
   - Compare only the priority extractor:
     - `priority_model = "gcn"`;
     - `priority_model = "gat"`.
   - Verify:
     - same seeds;
     - same topology scenario;
     - same episode budget;
     - same pretrained topology checkpoint mode if used.

5. [ ] **Report statistical evidence**
   - Use at least 3 seeds for smoke-level evidence and 5 seeds for final evidence.
   - Report mean and standard deviation.
   - Prefer paired comparison by seed.
   - Claim “GAT priority is better” only if it wins in at least two evidence layers:
     - supervised priority validation;
     - fixed-policy scheduling;
     - end-to-end DRL.

### Recommended Execution Order

**Current status - 2026-07-14:** Steps 1-5 are complete locally. Step 6 is next and requires the synchronized Windows project to run the matched W&B CPU/CUDA commands from `docs/wandb_tracking.md`.

1. Add wall-clock phase instrumentation that separates CPU runtime from simulated delay metrics.
2. Run the deterministic 10/20/30-device environment scaling benchmark and archive profiler evidence.
3. Remove duplicate connection-window recomputation and verify exact fixed-seed trajectory equivalence.
4. Vectorize the sampled connection-window calculation and repeat the equivalence/scaling benchmarks.
5. Optimize Task-GAT priority, dataset I/O, or state construction only if the post-repair profiler shows a contribution of at least 5%.
6. Run matched 20-30 episode CPU/CUDA W&B validation on `large_30d_9s` and compare medians excluding episode one.
7. Import and audit the completed remote ablation outputs; rerun only interrupted or behavior-invalidated rows.
8. Freeze the physical-state/digital-twin-state contract from the baseline paper and current manuscript.
9. Add explicit twin snapshots and synchronization while keeping environment physics unchanged.
10. Expose current-subtask GAT embeddings, implement `[h_task || h_topo || z_DT]`, and run the controlled dual-GAT/DT ablations.

### Optional Waiting List

- [ ] **CTDE-clean shared baseline models**
  - Goal: make baseline learning models fairer when comparing against `Graph-GAT MAPPO`.
  - Current status:
    - `Graph-GAT MAPPO` is CTDE-style: actor uses local subgraph, critic uses global graph.
    - Current `MAPPO` / `Mask-MAPPO` are CTDE-style only partially: actor uses local state and critic sees joint state, but each device owns a separate actor/critic object.
    - Fixed baselines such as `Local Only`, `Edge Only`, and random/heuristic baselines are not CTDE models and should remain labeled as fixed baselines.
  - Future implementation:
    - add `Shared MAPPO`: one shared actor, one centralized critic, actor receives each device local state, critic receives joint state;
    - add `Shared Mask-MAPPO`: same shared CTDE baseline with action masking;
    - keep existing independent MAPPO names or clearly rename them as independent-agent baselines.
  - Purpose:
    - make ablation cleaner: `Shared MAPPO -> Shared Mask-MAPPO -> Graph-GAT MAPPO -> Graph-GAT Mask MAPPO`;
    - support stronger paper claims about CTDE fairness across learned baselines.
  - Status: optional future work. Do not block current Graph-GAT ablation runs.

- [ ] **Graph-GAT MADDPG / GATMA-style CTDE**
  - Actor: local graph/subgraph embedding to local action.
  - Critic: global graph plus joint action to cooperative Q-value.
  - Buffer/update: replay buffer, TD loss, target networks, soft update.
  - Status: optional later ablation. Do not mix into current Graph-GAT MAPPO path.

- [ ] **C-4 Graph-GAT topology warmup decay with hybrid link-quality loss**
  - Current code stays on the simpler warmup design for now.
  - Future schedule:
    - episodes 0-4: train Topology-GAT 10 times per environment step before action selection;
    - episodes 5-20: reduce to 1-2 topology updates per environment step;
    - after episode 20: stop auxiliary topology warmup and let PPO continue normally.
  - Future loss: `BCE(feasible_link) + MSE(edge_delay_estimate) + ranking_loss(server_quality)`.
  - Purpose: avoid training Topology-GAT from scratch only through PPO, then test whether smoother warmup decay and richer link-quality supervision improve convergence.
  - Status: optional future ablation, not part of the current implemented experiment.

---

## Current Checkpoint - 2026-06-11

**Status:** Environment is frozen for the first e-ATN-MADDPG sanity phase. The user manually fixed the environment and reported it is good enough for the model to run.

What is now implemented:
- [x] Local-only baseline.
- [x] Edge-only baseline.
- [x] Random offloading baseline implemented, but not required/enabled for the current long-run comparison.
- [x] Feature-extraction-only edge baseline implemented, but not required/enabled for the current long-run comparison.
- [x] Timing diagnostics for local compute time, successful server compute time, attempted server compute time, transfer time, wait time, penalty count, and penalty time.
- [x] Action diagnostics for requested local/edge counts and actual resolved local/edge counts.
- [x] Readable comparison output for short runs, printed only on the final episode.
- [x] Environment logic was manually adjusted by the user and is reported as good enough for the model to run.
- [x] Environment freeze decision completed for first model sanity phase.
- [x] Paper-style delay fix: `environment/diten_env.py` now accumulates DAG makespan delay instead of summing overlapping branch subtask latencies.
- [x] Efficient 100-episode comparison plotting: train learning models for 100 episodes, run fixed baselines for a short evaluation, and plot fixed baselines as mean-flat reference lines.
- [x] Last training state JSONL: save one flat line per model beside the generated plot images for easy comparison.
- [x] Experiment parameters are centralized in `utils/paper_config.py` for the active comparison runner, including reward weights, penalties, estimation errors, GCN pretraining size, compute-power ranges, episode budgets, MAPPO settings, and Graph-GAT settings.
- [x] `t_max` and `e_max` from `PAPER_PARAMS` are passed into GCN DAG sampling and per-episode task DAG generation in `run_comparision.py`.
- [x] MAPPO and Graph-GAT MAPPO action masks can be turned on/off from `utils/paper_config.py`.
- [x] Trainable model checkpoints are saved into the same plot folder as images and last-state JSONL for later evaluation without retraining.
- [x] GPU readiness stage added without changing training logic: `utils/gpu_readiness.py` reports PyTorch/CUDA/GPU status, and `docs/windows_gpu_readiness.md` documents the Windows RTX 4080 setup check.
- [x] GAT task-priority extractor added as an option beside GCN: `models/task_priority_gat.py` implements `TaskPriorityGAT`, `priority_model` in `utils/paper_config.py` selects `gcn` or `gat`, and GAT uses its own `models/checkpoints/gat_priority.pt`.

Current output should now look like:

```text
[Random Offloading] Episode 10 diagnostics
  Requested actions: local=2329 edge=171
  Actual execution:  local=2480 edge=20
  Penalties:         count=151 time=0.856s
  Timing/device-task: local=2.750s server=0.081s transfer=0.210s wait=0.440s
```

Interpretation:
- `Requested actions` means what the policy asked for.
- `Actual execution` means what really ran after fallback.
- `Penalties` means edge requests that violated the connection window.
- `Timing/device-task` helps identify whether high delay comes from local compute, edge compute, transfer, or queue/wait.

---

## Historical Task Queue - Superseded by Active Research Plan

Historical note: this queue recorded the earlier paper-reimplementation path. Use the 2026-07-08 active research plan above for current work.

1. [x] **Verify e-ATN-MADDPG agent mechanics**
   - Goal: Confirm current `EpsilonATNMADDPGAgent` can act and update before running longer training.
   - Files to inspect/test only:
     - `models/maddpg.py`
     - `models/replay_buffer.py`
     - `run_comparision.py::_update_agents_from_buffer`
   - Verify:
     - `select_action` with epsilon `1.0` samples valid random actions. `DONE`
     - `select_action` with epsilon disabled uses actor argmax. `DONE`
     - replay buffer samples shape correctly for multi-agent training. `DONE`
     - actor parameters change after one update. `DONE`
     - critic parameters change after one update. `DONE`
     - target networks move after soft update. `DONE`
   - Result: mechanics tests pass without production model changes.
   - Verification: `PYTHONDONTWRITEBYTECODE=1 /Users/admin/miniconda3/bin/pytest tests/test_maddpg_update.py -q -p no:cacheprovider`

2. [x] **Run e-ATN-MADDPG smoke training on frozen env**
   - Goal: Verify model execution path runs on the fixed environment.
   - Run: very small episode count first, for example 1-3 episodes.
   - Verify:
     - no crash. `DONE by user`
     - replay buffer receives samples. `DONE by user`
     - actor/critic update path runs. `DONE by user`
     - reward, delay, and energy are finite. `DONE by user`
     - epsilon decays. `DONE by user`
     - final diagnostics show meaningful requested/actual local-edge counts. `DONE by user`

3. [x] **Verify MAPPO baseline mechanics**
   - Goal: Confirm current `MAPPOAgent` can act and update before trusting it as a comparison baseline.
   - Files to inspect/test only:
     - `baselines/mappo.py`
     - `run_comparision.py::_update_agents_from_buffer`
   - Verify:
     - stochastic actor outputs a valid probability distribution. `DONE`
     - `select_action` samples valid discrete actions. `DONE`
     - centralized value critic accepts flattened joint state and returns one value per sample. `DONE`
     - actor parameters change after one MAPPO update. `DONE`
     - critic parameters change after one MAPPO update. `DONE`
     - comparison update loop routes MAPPO agents through `update_agent`. `DONE`
   - Result: MAPPO mechanics tests pass without production code changes.
   - Verification: `PYTHONDONTWRITEBYTECODE=1 /Users/admin/miniconda3/bin/pytest tests/test_mappo_update.py -q -p no:cacheprovider`

4. [x] **Fix MAPPO rollout buffer correctness**
   - Goal: Make MAPPO use on-policy rollout data instead of replay memory samples.
   - Files touched:
     - `baselines/mappo.py`
     - `run_comparision.py`
     - `tests/test_mappo_update.py`
   - Verify:
     - MAPPO stores action log-probability when sampling action. `DONE`
     - MAPPO rollout buffer stores state, action, reward, next state, old log-probability, and done. `DONE`
     - PPO update uses rollout old log-probabilities instead of recomputing them after collection. `DONE`
     - PPO target value masks terminal transitions with `done`. `DONE`
     - comparison loop routes MAPPO through rollout update and clears the rollout after update. `DONE`
   - Verification: `PYTHONDONTWRITEBYTECODE=1 /Users/admin/miniconda3/bin/pytest tests/test_mappo_update.py -q -p no:cacheprovider`

5. [ ] **Add e-ATN-MADDPG training diagnostics if smoke output is insufficient** `HISTORICAL`
   - Goal: Make model learning observable without changing env.
   - Candidate diagnostics:
     - actor loss.
     - critic loss.
     - epsilon.
     - replay buffer size.
     - action distribution.
   - Rule: add diagnostics only if the smoke run does not already show enough information to judge model health.

6. [ ] **Run short e-ATN-MADDPG learning check**
   - Goal: Check whether reward trend and action behavior move in a plausible direction.
   - Run: 10-30 episodes.
   - Verify:
     - reward does not become NaN or explode.
     - delay/energy are finite.
     - action distribution does not collapse immediately without explanation.
     - if reward is flat, inspect actor/critic losses before changing reward/env.

7. [ ] **Compare e-ATN-MADDPG against enabled baselines only after smoke passes**
   - Goal: Check trend direction, not exact paper numbers.
   - Currently enabled comparison set:
     - Local Only
     - Edge Only
     - MAPPO
     - Graph-GAT MAPPO
     - e-ATN-MADDPG
     - MADDPG
     - MAAC
   - Implemented but currently disabled/not required:
     - Feature Extraction Edge
     - Random Offloading
   - Verify:
     - Learning algorithms run for the requested full episode count. `DONE`
     - Fixed baselines run for a shorter episode count and are extended only for plotting. `DONE`
     - e-ATN-MADDPG should improve over episodes.
     - If e-ATN-MADDPG cannot beat simple baselines, first inspect model training/update path before changing the frozen environment.

8. [ ] **Only then move toward paper-style long training**
   - Goal: Produce a training trend similar to the paper where the MADDPG-based method is best.
   - Verify:
     - Do not tune reward weights until baseline and smoke model runs are understood.
     - Do not claim paper reproduction from short runs.

9. [x] **Add GPU readiness check without training rewiring**
   - Goal: Prepare for Windows RTX 4080 training while keeping the Mac demo stable.
   - Files touched:
     - `utils/gpu_readiness.py`
     - `docs/windows_gpu_readiness.md`
     - `tests/test_gpu_readiness.py`
   - Current status:
     - Windows setup guide exists. `DONE`
     - readiness helper reports human-readable and JSON CUDA status and fails clearly when required CUDA is unavailable. `DONE`
     - CPU-only readiness and device-resolution tests pass. `DONE`

10. [ ] **Only move one training path to CUDA after Windows readiness passes**
    - Goal: Avoid messy GPU bugs by proving the machine setup first.
    - First target selected by user:
      - `Graph-GAT MAPPO`
    - Verify before/after code changes:
      - `python -m utils.gpu_readiness --preferred-device cuda` reports `CUDA available: True`.
      - GPU name shows the server GPU.
      - Mac CPU demo still remains the control path.
    - Status:
      - [x] `GraphGATMAPPOAgent` accepts a `device` setting and moves encoder/actor/critic/warmup modules to that torch device.
      - [x] Graph-GAT rollout update tensors are created on the agent device.
      - [x] Graph tensors remain built from CPU environment state, then move to the agent device at encode/mask boundaries.
      - [x] `run_comparision.py` passes `graph_gat_device` from `utils/paper_config.py` or the CLI override only to Graph-GAT MAPPO variants.
      - [x] CUDA timing synchronizes before/after Graph-GAT warmup, action selection, and PPO update.
      - [ ] Run 1-3 Graph-GAT MAPPO smoke episodes on the Windows GPU server.

11. [x] **Add GAT as task-priority option beside GCN**
    - Goal: Compare GCN priority extraction against a GAT priority extractor without removing the original GCN baseline.
    - Files touched:
      - `models/task_priority_gat.py`
      - `utils/experiment_setup.py`
      - `utils/priority_model_training.py`
      - `run_comparision.py`
      - `main.py`
      - `tests/test_task_priority_gat.py`
    - Verify:
      - GAT returns one priority score per subtask. `DONE`
      - GAT can train against the current supervised priority targets. `DONE`
      - `build_priorities` accepts a GAT model through the existing `(features, adjacency)` interface. `DONE`
      - `priority_model = "gat"` uses a separate `gat_priority.pt` checkpoint. `DONE`
      - GCN remains the default unless `priority_model` is changed. `DONE`

---

## Historical Ablation Study Queue - Graph-GAT State Encoder

Historical purpose: test whether converting the flat environment state into a topology graph and encoding it with GAT gives DRL models a better state representation for task offloading. The current active version of this work is the pretrained Topology-GAT and topology stress-test plan above.

Working assumption:
- "Typology" means topology: device nodes, edge-server nodes, task/subtask features, connection-window features, queue/wait features, and feasible device-server links.
- This is a new ablation/proposal path, not a paper-faithful baseline.
- GAT is a reusable state encoder idea, not tied to only one DRL algorithm.
- First integration target is MAPPO because its rollout/PPO update path is now corrected and easier to extend with graph-state rollouts.
- Later integration targets can include e-ATN-MADDPG and MAAC after the MAPPO ablation is verified.
- Implementation choice: use Option A first, a separate `GraphGATMAPPOAgent`, while keeping reusable graph/GAT utilities. Do not refactor all DRL models or put trainable GAT only in the training loop yet.
- Rationale: a separate agent is safer for the first ablation because it does not risk breaking flat-state MAPPO, and it keeps GAT inside the model/update path so gradients can update GAT correctly.
- The first ablation comparison is:
  - flat-state MAPPO.
  - Graph-GAT MAPPO.

Historical note: this queue is preserved for traceability. Use the 2026-07-08 active research plan above for current work.

1. [x] **Define graph-state representation**
   - Goal: Decide exactly how to convert current flat state into graph data.
   - Candidate node types:
     - device node: local compute power, local wait, current subtask CPU/data/result, accumulated local context.
     - server node: edge compute power, edge wait.
   - Candidate edge types:
     - device-to-server edge when the connection window is valid.
     - edge features: normalized `l_start`, `l_end`, window length, optional transfer-rate estimate.
   - Verify:
     - [x] graph can be built from information already available in `DITENEnv._get_joint_state`.
     - [x] no environment physics changes.
     - [x] graph shape is stable from `joint_state`, `num_devices`, and `num_servers`.

2. [x] **Add graph builder tests before implementation**
   - Goal: Lock the graph API before adding model code.
   - Candidate file:
     - `tests/test_topology_graph_state.py`
   - Verify:
     - [x] node feature tensor shape is `(num_devices + num_servers, node_feature_dim)`.
     - [x] edge index shape is `(2, num_edges)`.
     - [x] edge feature tensor length matches `num_edges`.
     - [x] disconnected server windows are represented with explicit edge flags.

3. [x] **Implement minimal topology graph builder**
   - Goal: Convert existing flat joint state into graph tensors without touching env internals.
   - Candidate file:
     - `utils/topology_graph_state.py`
   - Rule:
     - read only the current `joint_state`, `num_devices`, and `num_servers`.
     - do not change `DITENEnv._get_joint_state`.
   - Verify:
     - [x] graph builder tests pass with `PYTHONDONTWRITEBYTECODE=1 /Users/admin/miniconda3/bin/pytest tests/test_topology_graph_state.py -q -p no:cacheprovider`.

4. [x] **Add a small GAT encoder unit test**
   - Goal: Verify the encoder can consume graph tensors and produce one embedding per device.
   - Candidate files:
     - `models/topology_gat.py`
     - `tests/test_topology_gat.py`
   - Verify:
     - [x] output shape is `(num_devices, embedding_dim)`.
     - [x] gradients flow through the encoder.
     - [x] encoder works without `torch_geometric` unless the repo already depends on it.

5. [x] **Implement minimal GAT state encoder**
   - Goal: Create a small PyTorch-only GAT encoder to avoid new dependencies.
   - Candidate architecture:
     - one or two graph-attention layers.
     - masked attention over graph edges.
     - output only device-node embeddings for actor input.
   - Verify:
     - [x] unit tests pass with `PYTHONDONTWRITEBYTECODE=1 /Users/admin/miniconda3/bin/pytest tests/test_topology_gat.py -q -p no:cacheprovider`.
     - [x] no training script changes yet.

6. [x] **Create Graph-GAT MAPPO ablation agent tests**
   - Goal: Define the first graph-state DRL integration around MAPPO before implementation.
   - Candidate files:
     - `tests/test_graph_gat_mappo.py`
     - `baselines/graph_gat_mappo.py`
   - Verify:
     - [x] action dimension remains `0=local, 1..S=edge server`.
     - [x] actor samples valid actions from graph-derived device embeddings.
     - [x] rollout buffer can store graph states, next graph states, actions, rewards, old log-probs, and done flags.
     - [x] GAT parameters change after one PPO rollout update.
     - [x] flat-state `MAPPOAgent` tests still pass unchanged.

7. [x] **Implement Graph-GAT MAPPO agent**
   - Goal: Add a separate MAPPO variant that consumes graph states through a GAT encoder.
   - Candidate name:
     - `GraphGATMAPPOAgent`
   - Candidate file:
     - `baselines/graph_gat_mappo.py`
   - Verify:
     - [x] uses `TopologyGATEncoder` to create one embedding per device.
     - [x] MAPPO actor uses each device embedding to sample an action.
     - [x] centralized critic uses graph/device embeddings.
     - [x] PPO update backpropagates through MAPPO and GAT.
     - [x] no changes are required in `baselines/mappo.py` for flat MAPPO behavior.
     - [x] related tests pass with `PYTHONDONTWRITEBYTECODE=1 /Users/admin/miniconda3/bin/pytest tests/test_topology_graph_state.py tests/test_topology_gat.py tests/test_graph_gat_mappo.py tests/test_mappo_update.py -q -p no:cacheprovider`.

8. [x] **Wire Graph-GAT MAPPO into comparison config behind a separate name**
   - Goal: Add the model as an optional comparison line, not as a replacement for flat MAPPO.
   - Candidate display name:
     - `Graph-GAT MAPPO`
   - Candidate file:
     - `run_comparision.py`
   - Verify:
     - [x] original flat-state `MAPPO` tests still pass.
     - [x] `Graph-GAT MAPPO` is registered as a separate comparison model.
     - [x] comparison loop can collect graph-state actions and update graph rollout.
     - [x] Graph-GAT timing diagnostics report graph build, action selection, update time, and transition count.
     - [x] Graph-GAT PPO update reuses each epoch embedding for both actor log-probabilities and critic value.
     - [x] Graph state includes task priority vector in device nodes and explicit connected/disconnected edge flags.
     - [x] Graph-GAT MAPPO masks disconnected server actions during sampling and PPO log-prob recomputation.
     - [ ] `Graph-GAT MAPPO` can run 1-3 smoke episodes.
     - [ ] JSONL saves a separate one-line final state for the ablation model.

9. [ ] **Run Graph-GAT MAPPO smoke test** (next)
   - Goal: Check integration before long training.
   - Run:
     - 1-3 episodes for `Graph-GAT MAPPO`.
   - Verify:
     - no crash.
     - reward/delay/energy finite.
     - graph rollout buffer fills.
     - PPO actor/critic/GAT update path runs.
     - Graph-GAT cost line shows whether graph build, action, or update is the bottleneck.
     - requested edge actions should no longer target disconnected servers.

10. [ ] **Run short MAPPO ablation comparison**
   - Goal: Compare trend, not final paper-quality numbers.
   - Compare:
     - flat-state MAPPO.
     - Graph-GAT MAPPO.
   - Run:
     - 10-30 episodes first.
   - Verify:
     - Graph-GAT MAPPO does not collapse to all-local/all-edge immediately.
     - Graph-GAT MAPPO has plausible action distribution.
     - if Graph-GAT MAPPO is worse, inspect graph features and attention before tuning reward.

11. [ ] **Attach reusable GAT encoder to other models only after MAPPO ablation works**
   - Goal: Extend the same graph-state encoder idea after the MAPPO path is stable.
    - Candidate next targets:
      - Graph-GAT e-ATN-MADDPG.
      - Graph-GAT MAAC.
    - Verify:
      - each new model is added as a separate comparison line.
      - the flat-state version remains as the control.
      - only consider a shared encoder abstraction after at least one graph model shows useful behavior.

12. [ ] **Run final ablation only after short comparison is stable**
    - Goal: Produce a clean ablation result.
    - Compare:
      - flat-state MAPPO control.
      - Graph-GAT MAPPO proposal.
      - optional Graph-GAT e-ATN-MADDPG only after separate smoke tests.
    - Verify:
      - plots, checkpoints, and JSONL include each enabled model as separate outputs.
      - result summary clearly states this is a proposed graph-state encoder ablation, not the original paper model.

---

## Current Status

- [x] Read the `AGENTS.md` rules supplied by the user; skip `CLAUDE.md` per user request.
- [x] Inspect repo structure and identify main code paths.
- [x] Read current environment code in `environment/diten_env.py`.
- [x] Read system model code in `environment/system_model.py`.
- [x] Read network model code in `environment/network_env.py`.
- [x] Read MADDPG code in `models/maddpg.py` and training update loops in `main.py` / `run_comparision.py`.
- [x] Read baseline scheduler code in `baselines/scheduling_baselines.py`.
- [x] Extract and inspect `baseline_paper.pdf`.
- [x] Confirm Table II experiment parameters from the rendered paper page.
- [x] Draft this plan/checklist.
- [x] Confirm baseline interpretation with user before writing code.
- [x] Verify and fix MADDPG actor update so actor parameters change after replay-buffer training.
- [x] Implement and test local-only, edge-only, and random offloading baseline policies.
- [x] Implement and test feature-extraction-only edge baseline.
- [x] Add timing diagnostics for local time, server time, transfer time, wait time, action counts, and penalty count/time.
- [x] Make comparison diagnostics readable with short progress postfix plus multi-line episode summaries.

---

## Code Understanding

### Main Training Path

- `main.py`
  - Function `train_maddpg` trains `EpsilonATNMADDPGAgent` over episodes, time slots, and subtask steps.
  - It generates one `TaskDAG` per device at each time slot, builds priorities using GCN, calls `env.start_time_slot`, collects joint actions, calls `env.step`, stores replay samples, and updates agents from replay.
  - State dimension is already dynamic through `env.get_state_dim()`.

- `run_comparision.py`
  - Function `train_algorithm` trains multiple algorithms on the same `DITENEnv` style.
  - Current comparisons include `e-ATN-MADDPG`, `MAAC`, and `MAPPO`.
  - Random and greedy currently mean priority-generation modes, not offloading-location baselines.

### Environment Path

- `environment/diten_env.py`
  - Class `DITENEnv` owns slot state, connection windows, queues, accumulated delay/energy, and reward.
  - `_get_joint_state` already attempts Eq. 23: local compute/wait, current subtask `(C, D, R)`, priority sequence, edge compute/wait, connection start/end windows.
  - `_calculate_reward` already follows Eq. 24 form.
  - `_update_connection_windows` uses subslots to estimate `l_start` and `l_end`.
  - `_schedule_edge_items` currently falls back to local execution when an offload violates the connection window. This may conflict with paper constraints C1/C2, which define feasibility of offloaded execution.
  - Decision: different edge servers are assumed to have a direct inter-server link with negligible transfer delay and energy. `_calculate_result_transfer` intentionally returns zero for edge-to-different-edge transfer. Keep this out of the current P0 work; revisit only if modeling backhaul cost or validating this assumption with system measurements.

### MADDPG Path

- `models/maddpg.py`
  - `ActorNetwork` outputs a discrete action probability distribution.
  - `CriticNetwork` consumes joint states and one-hot joint actions and can use self-attention.
  - `EpsilonATNMADDPGAgent.select_action` uses epsilon-greedy exploration.

- `main.py::_update_agents_from_buffer` and `run_comparision.py::_update_agents_from_buffer`
  - Current critic update is structurally MADDPG-like.
  - High-risk issue: actor loss uses `argmax(agent.actor(...))` and one-hot conversion. `argmax` is non-differentiable, so the actor may not receive a useful gradient. This must be verified before trusting any training trend.

### Existing Tests

- `tests/` currently contains only `__pycache__` bytecode files, not source `.py` tests.
- `python3 -m pytest --version` failed because pytest is not installed in the active system Python.
- We should add small source tests before touching model/environment behavior.

---

## Paper Understanding

### Environment and MDP

- The scenario has `D` mobile industrial devices and `S` fixed edge servers.
- Each device generates an image-recognition task at the beginning of each time slot.
- Each task is a DAG with subtasks. Each subtask has CPU cycles `C`, data size `D`, and result size `R`.
- Offloading action is discrete:
  - `0`: process locally.
  - `j`: offload to edge server `s_j`.
- Mobility matters. Each device moves on a predetermined path. A device can offload to an edge server only when execution starts after `l_start` and finishes before `l_end`.
- State Eq. 23 contains local compute/wait, subtask features, execution priorities, edge compute/wait, and connection windows.
- Reward Eq. 24 rewards lower immediate and accumulated delay/energy, plus a penalty for offloading outside the link window.
- Algorithm 1 loops over episode, time slot, and subtask index, then stores joint transition samples `(S, A, R, S')`.

### Confirmed Table II Parameters

- `training_episodes`: 1000
- `batch_size`: 64
- `t`: `{0, 1, ..., 49}`
- `E_max`: `1 J`
- `r`: `12 m`
- `B`: `10 MHz`
- `p_i`: `0.5 W`
- `P_j`: `1.2 W`
- `N_0`: `-43 dBm`
- `T_max`: `1 s`
- `f_i^loc`: `[0.8, 1.2] GHz`
- `f_j^edge`: `[2.3, 2.5] GHz`
- `tau_i^local`: `10^-28`
- `tau_j^edge`: `10^-27`

### Paper Baselines and Expected Trend

- Priority comparison: GCN scheduling should beat greedy scheduling, which should beat random scheduling.
- DRL comparison: epsilon + attention MADDPG should have better reward, delay, and energy than MAAC, MAPPO, MADDPG, GR-MADDPG, and ATN-MADDPG.
- For our requested minimum scope, first compare MADDPG/e-ATN-MADDPG against simple offloading baselines:
  - local-only
  - edge-only
  - random offloading

---

## Assumptions and Tradeoffs

- The paper does not fully specify reward weights `lambda1` through `lambda5` or `p_out` magnitude. We should keep these configurable and avoid tuning them until invariants pass.
- The paper does not specify the exact channel gain model. Current code uses distance to the power of `-2`; keep it unless it blocks trend reproduction.
- The paper does not fully specify how KolektorSDD image properties map to subtask `C`, `D`, and `R`. Current code uses proportional heuristics; keep them unless environment checks show impossible scales.
- The paper says random scheduling ignores dependencies, but the environment still should not silently execute a subtask before an unfinished predecessor without recording that as invalid or blocked.
- Edge-only baseline needs clarification because there are three edge servers and a device may not be connected long enough to execute on every server.

---

## Historical Implementation Checklist

Historical note: this checklist documents the original paper-reimplementation audit. Open boxes here are not the current execution queue unless they are moved into the active research plan above.

### Phase 1: Verification Scaffolding Before Behavior Changes

- [ ] Add source tests under `tests/` for environment invariants.
  - Verify state dimension and state contents from `DITENEnv.get_state_dim` and `_get_joint_state`.
  - Verify Eq. 24 reward arithmetic in `_calculate_reward`.
  - Verify connection-window violation is observable in `last_step_metrics`.
  - Verify predecessor dependencies do not silently disappear when priority order is invalid.
- [x] Add a small gradient test for MADDPG actor update.
  - Verify actor parameters change after one replay-buffer update with a deterministic batch.
  - Fixed the actor-loss path in `main.py::_update_agents_from_buffer` and `run_comparision.py::_update_agents_from_buffer` by passing differentiable actor probabilities into the critic instead of non-differentiable `argmax` one-hot actions.
- [ ] Add a smoke script or test fixture that runs one tiny episode without dataset files.
  - Verify no crash.
  - Verify finite reward, delay, energy.
  - Verify `last_step_metrics` has one row per device per subtask step.

### Phase 2: Baseline Policies

- [x] Implement local-only baseline.
  - Scope: action policy only, not a new environment.
  - Expected behavior: every device always returns action `0`.
  - Verification: all `last_step_metrics[*]["action"]` values are `0`.

- [x] Implement edge-only baseline after confirming server-selection rule.
  - Option A: always use nearest edge server.
  - Option B: always use first valid edge server by connection window.
  - Option C: always use fixed server `1`.
  - Implemented default: first server whose DT-estimated compute time fits its connection window; otherwise local action `0`.
  - Verification: disconnected devices select local, while residual rejection records DT-estimate or transfer/queue mismatch against physical feasibility.

- [x] Implement random offloading baseline after confirming interpretation.
  - Implemented default: random execution location from `{local, edge1, edge2, edge3}` with the same priority source as MADDPG, because this isolates offloading-location quality.
  - Verification: observed action histogram includes both local and edge actions over a short run.

- [x] Implement feature-extraction-only edge baseline.
  - Scope: comparison-path baseline only, not a DRL state/model change.
  - Implemented default: subtask `4` uses the same first-connected-edge rule as `EdgeOnlyAgent`; all other subtasks use local action `0`.
  - Function impact: `FeatureExtractionEdgeAgent.select_action_for_subtask` chooses edge only for the feature extraction subtask; `run_comparision._collect_joint_actions` passes current subtask id from `env.current_step` and `env.priorities` only to agents that expose `select_action_for_subtask`.
  - Diagnostic meaning: if this baseline loses to all-local, inspect reward weights, communication rate/energy, and transfer accounting before tuning DRL.

### Phase 3: MADDPG Correctness First

- [ ] Verify `EpsilonATNMADDPGAgent.select_action`.
  - Epsilon `1.0` should sample random actions.
  - Epsilon `0.0` should choose actor argmax.

- [ ] Verify critic update.
  - Critic loss should be finite.
  - Critic parameters should change after one update.

- [x] Verify actor update.
  - Actor parameters should change after one update.
  - Replaced the non-differentiable argmax actor-loss path with a minimal differentiable action-probability path.

- [ ] Verify replay buffer shapes.
  - State shape: `(batch, num_agents, state_dim)`.
  - Action shape after one-hot: `(batch, num_agents, action_dim)`.
  - Reward shape: `(batch, num_agents)`.

### Phase 4: Environment Audit and Minimal Fixes

- [ ] Verify mobility and connection windows.
  - `l_start` and `l_end` should be within the current slot.
  - Offloaded execution should start after `l_start` and finish before `l_end`.

- [ ] Verify waiting delay Eq. 7 and Eq. 12 behavior.
  - Local queue waits for previous local subtask on the same device.
  - Edge queue waits for previous subtask assigned to the same edge server.

- [ ] Verify predecessor-result transfer Eq. 8, Eq. 13, Eq. 15, Eq. 16, Eq. 17.
  - Local-to-edge upload is counted.
  - Edge-to-local download is counted.
  - Edge-to-different-edge transfer is either counted or explicitly justified as ignored according to paper assumptions.

- [ ] Decide whether invalid offload should reject, fallback to local, or execute with penalty.
  - Paper constraints C1/C2 say offloaded execution must satisfy the connection window.
  - Current code falls back to local with penalty.
  - Minimum-change path: keep fallback only if we document it as an environment repair action; otherwise change to explicit invalid action handling.

- [ ] Verify task constraints C3 and C6.
  - `T_max` and `E_max` should be both reward terms and feasibility checks.
  - Current code uses them in reward but not as explicit constraints.

### Phase 5: Comparison Run

- [x] Add diagnostic counters for baseline comparison runs.
  - `DITENEnv.last_step_metrics` now records requested action, resolved action, start/finish/current slot, execution time, local compute time, successful server compute time, attempted server compute time, transfer time, wait time, penalty flag, and penalty time.
  - `run_comparision._summarize_step_metrics` aggregates local time, server time, attempted server time, transfer time, wait time, requested local/edge counts, resolved local/edge counts, penalty count, and penalty time.
  - Training progress now shows compact reward/delay/energy, actual local/edge counts, and penalty count in the progress bar postfix.
  - Short runs now print one readable diagnostic block on the final episode, with requested actions, actual execution counts, penalties, and timing averages.

- [ ] Run a small deterministic comparison first.
  - Episodes: 3 to 10.
  - Purpose: catch crashes and impossible metric scales.
  - Expected: all methods produce finite reward, delay, and energy.

- [ ] Run a medium comparison.
  - Episodes: enough to see trend direction, not final paper reproduction.
  - Expected: MADDPG reward should improve over time if actor update is correct.

- [ ] Compare against local-only, edge-only, and random.
  - If MADDPG is worse than all simple baselines, stop tuning and re-audit environment.
  - If local-only or edge-only dominates unrealistically, inspect delay/energy scale, connection-window handling, and transfer-energy accounting.

- [ ] Only after correctness passes, run longer training toward paper-like trend.
  - Target trend: MADDPG/e-ATN-MADDPG best or improving toward best.
  - Exact numeric match is not required.

---

## Historical Immediate Next Step

Historical note: this was the immediate next step for the older paper-reimplementation phase, not the current active research plan.

Run the first frozen-env DRL task: verify the current `EpsilonATNMADDPGAgent` mechanics before any longer training run.

Scope:
- inspect and test `models/maddpg.py`, `models/replay_buffer.py`, and `_update_agents_from_buffer`;
- do not change `environment/diten_env.py`, `environment/network_env.py`, or `dataset/data_loader.py`;
- stop after the mechanics check so the user can verify before continuing.
