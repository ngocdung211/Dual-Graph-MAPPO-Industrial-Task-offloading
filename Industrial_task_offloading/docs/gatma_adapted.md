# GATMA-Adapted v3 comparison baseline

## Scope and application structure

Implement the learning design of *Topology-Cognitive Task Offloading and
Resource Allocation: A GAT-Enhanced MADRL Approach*
(DOI: 10.1109/TCCN.2026.3683874) on the existing DITEN workload. This is an
adaptation, not a reproduction of the paper's cloud-edge-end experiments.

Three application layers reuse the existing project structure:

1. Environment/data: `environment/` and `dataset/` provide devices, servers,
   five-subtask DAGs, observations, shared rewards and execution constraints.
2. Algorithm: `baselines/gatma.py` builds independent local actors and global
   Q critics; `utils/training/gatma_training.py` performs synchronized replay updates.
3. Experiment/output: `run_comparision.py` selects, trains and checkpoints the
   baseline; `inference_priority_comparison.py` evaluates saved policies.

Each device has two learned roles, an Actor and a Q-Critic, each with online
and target copies. These roles are separate from the three application layers
and from neural-network layers. Frozen Task-GAT, when enabled, supplies the
same priority order to all algorithms; it is not part of GATMA's encoder.

**Current default, 2026-10-08:** new runs use `adaptation_version=3`.
Version 2 remains available to load historical checkpoints, whose constructor
settings did not include a version. The [v2 audit](gatma_implementation_audit_2026-10-08.md)
records the motivating observations; it is historical evidence, not a v3 result.

## Paper mapping and deliberate differences

| Aspect | Paper | Adaptation |
|---|---|---|
| Agent / node | Edge-server agents, edge/cloud nodes (18–20) | Device agents; device/server nodes |
| Actor neighborhood | First-hop connected servers (25–28) | Own device + connected edge servers + self edges |
| Node features | Associated terminal tasks and server resources | Base width 14; actor appends its own pair's window start/end to each server node |
| Edge representation | Neighbor membership in attention | Boolean attention adjacency plus retained pair windows `[B,D,S,2]` |
| Encoder | Input MLP, multi-head GAT, output MLP | Same pattern, separate Actor/Critic weights per device |
| Actions (21) | Location, subchannel, transmit power | Local execution or one edge server; no cloud/channel/power decisions |
| Actor output | MLP decisions from local GAT features | Local head plus shared per-server scorer; score j maps to action j+1 |
| Critic (29) | Attention aggregation toward central cloud node | One GAT attention readout into a training-only collector with focal-device context; attends device and resource nodes |
| Objective (22) | Deadline minus task completion latency | Existing DITEN reward shared with comparison methods |
| Actor gradient (30) | Deterministic policy gradient | Straight-through argmax with softmax gradient; other actions from replay |
| Targets (31–36) | Target Actor/Critic, TD loss, Polyak updates | Retained; terminal bootstrap masked; targets updated after each replay round |

The existing defaults are retained: widths 64/64, four concatenated heads,
actor/critic learning rates 1e-4/1e-5, gamma 0.95, tau 0.01, batch 128,
replay capacity 100,000 and episode-progress epsilon 0.99 toward 0.01.
The audit verified these learning constants against the graphical Table IV.
The paper's full experiment is not reproduced: its 3000 episodes and 200 steps
differ from the DITEN comparison budget and decision granularity. The linear
epsilon schedule and 16 replay-update rounds after each episode are explicit
adaptation choices. For a finite run, the last episode starts just before the
schedule's endpoint because progress uses the number of already completed
episodes.

## V3 representation and checkpoint contract

For 30 devices, nine servers, five subtasks, and batch size B:

| Tensor or head | Shape / input width |
|---|---|
| Flat joint state | `[B,30,46]` |
| Base node features | `[B,39,14]` |
| Device-server windows | `[B,30,9,2]`, slot-relative start/end from the environment |
| Actor local node features | `[B,10,16]` |
| Actor local logit | `64 -> 64 -> 1` |
| Shared server scorer | `[device_embedding, server_embedding, start, end, connected]`, `131 -> 64 -> 1` |
| Actor probabilities | `[B,10]`, local action first, then server order |
| Critic physical-node inputs | `[B,39,52]`: base 14 + action 10 + windows 18 + server identity 9 + focal marker 1 |
| Critic GAT input / readout | `[B,40,64] -> collector embedding [B,64] -> Q_i [B,1]` |

The [actor](../baselines/gatma.py#L230) concatenates only its own device's
windows to local server features. A shared scorer preserves server/action
correspondence under a joint permutation of all server observation blocks.
Disconnected server embeddings are zeroed before scoring so unseen resource
values cannot influence decisions. Every categorical action remains available;
neither the score nor exploration applies an action mask.

The [critic](../baselines/gatma.py#L327) appends all pair windows, server-slot
identities, and a marker for the device whose reward it predicts. Its additional
collector copies that device's projected context and attends all physical
device/resource nodes, excluding itself. This restores a central attention
readout pattern; the collector is not a physical cloud server and adds no
execution action. Attending device and resource nodes, focal context, pair
features, and the categorical scorer remain explicit DITEN adaptations.

The shared GAT width stays 64 with four heads. Each v3 actor has 18,050
parameters and each critic 11,841, versus v2's 9,994/10,049; therefore an equal
sample budget is not an equal parameter or runtime budget.

[New configurations](../utils/comparison/algorithm_config.py#L126) record
version 3 in constructor kwargs. [Checkpoint metadata](../run_comparision.py#L1246)
records the readout and window contracts. [Inference configuration](../inference_priority_comparison.py#L260)
and [evaluation](../utils/comparison/evaluation.py#L81) resolve the saved version,
defaulting missing historical version fields to v2. V2 preserves the old
parameter shapes and mean readout; its weights are not converted into v3.

## Training and runtime behavior

- Networks move to the selected device before optimizer construction. Targets
  are frozen for autograd. Replay storage remains on CPU; each sampled joint
  batch is transferred once and shared across the agents' updates.
- The node features, connection mask, per-device local adjacency and global
  adjacency derived from a joint-state tensor are immutable during one action
  collection or replay update. The runner therefore builds one shared topology
  batch for all Actors. A replay update builds exactly two batches (current and
  next state), which every online/target Actor and Critic reuses. The cache does
  not retain tensors across environment steps or optimizer updates.
- Executed, replayed and target actions are one-hot for Critic evaluation.
  For the optimized actor, `one_hot(argmax(p)) + (p - stop_gradient(p))`
  provides a discrete forward action and a softmax surrogate gradient.
  This estimator is biased and is documented as part of the adaptation.
- Adjacency masking restricts attention. It does not mask output actions:
  exploration or argmax may request an unavailable server, handled through the
  same DITEN fallback and penalty mechanism as other unmasked baselines.
- A nonterminal last-subtask transition is held until the next slot has
  generated its tasks. Its replay next state then equals the next decision's
  observation. It becomes sampleable in the following slot. Terminal
  transitions are stored immediately. No transitions cross episode resets.
- The default episode has 100 slots and 500 joint transitions. After collection,
  GATMA performs 16 replay-update rounds with batch 128. Each network therefore
  sees 2,048 sampled transitions per episode, closely matching the 2,000
  transition exposures of a 500-transition rollout trained for four PPO epochs.
  The generic MADDPG updates-per-episode flag does not change this schedule.
- Checkpoints record adaptation version, representation, discrete gradient,
  replay settings, resolved device, horizon and actual priority order.
  Evaluation can load CUDA-trained weights on CPU and uses greedy actions.
  These are evaluation checkpoints; exact interrupted-training resume is not
  supported because replay contents and RNG state are not saved.

## Comparison protocol and commands

Use the same dataset snapshot, topology, physical-system seed, task seed,
priority order, horizon and episode budget for both methods. Report delay,
energy and reward over matched held-out evaluation seeds, along with update
counts. Equal environment interactions do not imply equal optimizer budgets.
Compare against `Graph-GAT Warmup MAPPO` for an unmasked baseline comparison;
if a masked variant is included, label its action-feasibility advantage.
This comparison measures the complete methods, not the encoder alone.

From a working directory containing the project's dataset/checkpoint/output
paths (Drive symlinks are supported), invoke the runner by absolute path:

```bash
python /path/to/Industrial_task_offloading/run_comparision.py \
  --algorithms GATMA-Adapted 'Graph-GAT Warmup MAPPO' \
  --gatma-device cuda --graph-gat-device cuda \
  --topology-scenario paper_10d_3s --episodes 500 \
  --experiment-seed 75 --wandb-mode disabled --note gatma-comparison
```

`--algorithms GATMA` remains a CLI alias, but outputs use `GATMA-Adapted`.
Repeat training with seeds 75, 175 and 190 for a multi-seed study. That full
study is separate from the short implementation verification run.

```bash
python /path/to/Industrial_task_offloading/inference_priority_comparison.py \
  --checkpoint-dir /path/to/comparison-run \
  --algorithms GATMA-Adapted 'Dual-GAT MAPPO' --device cpu \
  --dataset-path '/path/to/KolektorSDD' --evaluation-seeds 275 375 475 \
  --episodes-per-seed 5
```

The evaluation script compares default and inferred priority orders separately
and reports the current evaluation horizon. Use one shared inferred order for
the primary method comparison. The historical three-policy default selection
remains available when `--algorithms` is omitted.

## Verification scope

`tests/test_gatma.py` checks server/action permutation correspondence, window
response and gradients, unmasked disconnected actions, collector reach and
focal-device context, local attention isolation, cached/uncached output
and gradient equivalence, topology build counts, discrete Critic inputs,
actor/critic encoder gradients, frozen targets, Polyak updates, terminal
masking, slot-boundary replay continuity, checkpoint round trips and CPU/CUDA
operation. Historical CUDA verification below applies to v2. Neither basic
mechanics nor a short smoke establishes convergence or superiority.

### V3 verification, 2026-10-08

- Focused GATMA tests: 14 passed, one CUDA test skipped on this CPU host.
- Broader runner/tracking/artifact checks: 51 passed, one CUDA skip, two existing
  Optuna failures. Both concern default Graph-GAT warmup 15 outside the search
  choices `[5,10,20]`; the same failure was reproduced from the HEAD tuner and
  unchanged `paper_config.py`. Those unrelated defaults were not changed.
- Real-data CPU smoke: modular 30-device/9-server topology, training seed 192,
  three 100-slot episodes, 500 joint transitions and 16 replay rounds each.
  All 30 actors and critics have 48 Adam steps and finite online/target weights.
- Saved v3 checkpoint was reloaded and completed greedy held-out inference on
  task seed 1192, with finite reward/delay/energy. Training and evaluation used
  the Drive snapshot of 352 input images, not synthetic fallback.
- A historical seed192 v2 checkpoint reproduced every audit metric at task seed
  1192 within absolute tolerance `1e-7` using the new version-aware evaluator.
- The [smoke manifest](../experiments/gatma_v3/smoke_manifest.json) owns settings,
  source hashes, checkpoint hash/path, training and inference metrics, and the
  uncommitted-code provenance. Three episodes are integration evidence only.
- Planned: matched full training over multiple seeds and separate readout/input
  ablations. V3 convergence and performance superiority remain unverified.

### Verified run before the fixed-budget revision, 2026-09-23

- All nine GATMA tests passed with CUDA accessible, including CPU/CUDA gradient
  updates and checkpoint portability. The related GPU-readiness, digital-twin
  and experiment-tracking checks also passed in the sandbox; its CUDA-specific
  GATMA test was skipped there and subsequently passed outside the sandbox.
- RTX 3060, `paper_10d_3s`, seed 75, one episode with 100 slots: training
  completed in approximately 35.72 seconds (runner's episode timer).
- The Drive loader found 352 input images; this is not the full expected
  399-image dataset. The run used real images, not synthetic fallback.
- Saved checkpoint: ten device agents, finite online/target weights, 75 Adam
  steps for each online actor/critic, and recorded CUDA/horizon provenance.
- The saved CUDA checkpoint completed CPU greedy evaluation on seed 275,
  one 100-slot episode per priority order, with finite reward/delay/energy.
- Artifacts under `G:\My Drive\Dual-Graph-MAPPO-Artifacts\runs\`:
  `gatma-adapted-validation.JcUoFc/` contains training/evaluation logs,
  `cpu-checkpoint-evaluation.json/.csv`, and the timestamped training outputs.

This validates execution and serialization only. A matched multi-seed training
comparison with Graph-GAT MAPPO has not been run.

### Topology-cache and device benchmark, 2026-09-23

The cache was checked against the uncached forward path for Actor outputs,
Critic values and every Actor/Critic parameter gradient. The numerical results
matched. On the 10-device/3-server GATMA workload (batch 128), median timings
from alternating same-process measurements were:

| Operation | CPU uncached → cached | RTX 3060 uncached → cached | Speedup |
|---|---:|---:|---:|
| joint action selection | 3.826 ms → 2.734 ms | 14.369 ms → 9.697 ms | 1.40× / 1.48× |
| one replay update | 183.919 ms → 162.864 ms | 283.396 ms → 243.144 ms | 1.13× / 1.17× |

GATMA remains faster on CPU for this small, sequential collection of independent
networks; caching removes redundant graph preparation but does not batch those
networks into a single large GPU operation.

For Graph-GAT MAPPO, a matched synthetic 30-device/9-server episode used 100
slots, 500 transitions, GAE, four PPO epochs and four minibatches. End-to-end
training took 9.94 s on CPU and 8.58 s on the RTX 3060 (1.16× overall). The
model update was 5.06 s versus 1.39 s (3.64× GPU speedup), while graph action
time was 0.93 s versus 2.97 s (GPU 3.20× slower). Graph construction and the
environment remain CPU work, and per-step CPU↔GPU transfer/synchronization
limits the end-to-end gain. These are runtime measurements, not convergence or
reward-quality results.

### Fixed-budget verification, 2026-09-23

The current adaptation collects all 500 episode transitions before running 16
replay rounds. A 100-slot `paper_10d_3s` CPU smoke run completed with 500 stored
transitions, exactly 16 rounds, 2.44 seconds of model-update time and adaptation
checkpoint version 2. The synthetic workload run took 3.83 seconds overall.
All nine GATMA tests passed with CUDA available. This smoke run checks scheduling
and execution only; it does not replace real-data convergence experiments.
