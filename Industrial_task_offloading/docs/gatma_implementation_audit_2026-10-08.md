# GATMA implementation and performance audit - 2026-10-08

**Historical v2 audit:** this report precedes the approved v3 implementation.
For current defaults and verification, see [GATMA-Adapted v3](gatma_adapted.md).
The findings below describe the inspected v2 checkpoints and recorded revision.

The current baseline retains GATMA's actor/critic learning mechanics, but changes
its central graph readout and loses information needed by the DITEN action
interface. The saved policies exhibit action saturation and poor reward relative
to uniform random and Local Only in this diagnostic evaluation. These results
describe **GATMA-Adapted v2**, not the original paper's performance.

No algorithm, training default, reward, topology, checkpoint, or old result was
changed. This audit adds documentation and diagnostic evidence only.

## Scope, sources, and provenance

- Source paper: *Topology-Cognitive Task Offloading and Resource Allocation:
  A GAT-Enhanced MADRL Approach*, DOI `10.1109/TCCN.2026.3683874`.
  Inspected sections V-B/V-C, equations 18-36, Algorithm 1, and graphical Table IV.
  Source PDF is in `Dual-Graph-MAPPO-Artifacts/Others/` on the connected Drive;
  SHA-256: `fedc9235dbffb8568a3faaed09018ae54979e7dcc513b7390e062bc01705f8b4`.
- Reviewed code revision: `b0d39527b9b7334b0b582b7b18938a2dfcf968ae`.
  Inspected historical W&B configs/checkpoints do not record their source revision;
  exact historical code identity cannot be certified from these artifacts.
- Training runs: [seed190](https://wandb.ai/ObjectPromptDA/industrial-task-offloading/runs/4a88o3ap),
  [seed191](https://wandb.ai/ObjectPromptDA/industrial-task-offloading/runs/dayfopgh),
  [seed192](https://wandb.ai/ObjectPromptDA/industrial-task-offloading/runs/g1y3iz3s).
  Seed190 also has a duplicate W&B entry with the same inspected metric histories;
  it is not counted as another independent training seed.
- Checkpoints: October 7 runs for seeds 190/191/192 in the Drive artifact root.
- Real-data evaluation uses the Drive dataset snapshot with **352 input images**.
  Its count, minimum/maximum pixels, and mean pixels match historical metadata.
  The repository dataset has 399 images; preliminary evaluations using it were
  superseded and are excluded from the reported numbers.
- [Machine-readable evidence](../experiments/gatma_audit/2026-10-08/evidence.json)
  contains inspected settings, window summaries, 27 evaluation episodes,
  checkpoint diagnostics, representation probes, and source-code hashes.

## Verified performance

Training metrics below are means over episodes 1-100 and 901-1000, not values
read from a smoothed plot.

| Training seed | Initial reward | Final reward | Reward change | Initial delay (s) | Final delay (s) |
|---|---:|---:|---:|---:|---:|
| 190 | 3.0002 | 2.1577 | -28.1% | 1.6337 | 1.7706 |
| 191 | 5.4970 | 4.9154 | -10.6% | 1.1962 | 1.2855 |
| 192 | 5.6016 | 5.5489 | -0.94% | 1.1768 | 1.1873 |

Energy decreases in all three runs. Thus the decline is in reward and delay;
the policy is not worse in every measured dimension.

Each checkpoint was evaluated without updates on task seeds 1192/2192/3192,
one 100-slot episode per policy per seed. Each episode has 500 joint decisions
and 15,000 device actions. The original physical compute seed and recorded
priority order `[1,3,2,4,5]` were retained. Reward weights are
`lambda1..4=5`, `lambda5=1`, `p_out_value=-1.5` in both inspected runs and
evaluation. Uniform random selects from all ten action indices; Local Only
always selects action zero. All three policies share the same task sampler,
physical system, priority order, and reward within each training-seed comparison.

| Training seed | GATMA greedy reward | Uniform random reward | Local Only reward |
|---|---:|---:|---:|
| 190 | 2.1459 | 2.9689 | 3.2603 |
| 191 | 4.8215 | 5.5976 | 6.0680 |
| 192 | 5.5577 | 5.6163 | 6.5568 |

These are three-task-seed means, not a significance claim. The seed192 gap to
random is much smaller than the gaps for seeds190/191. GATMA greedy consumes
less energy than these references; it also has lower delay than Local Only.
Reward differences include the delay/energy tradeoff and rejection penalty.

## Verified policy behavior

In the first held-out episode, **29/30, 26/30, and 29/30** actors for training
seeds190/191/192 respectively chose one unchanged action for all 500 decisions.
Mean maximum softmax probabilities across actors were approximately
`0.999989`, `0.999456`, and `0.999964`. This establishes nearly deterministic,
mostly state-insensitive behavior on the inspected trajectories, not that every
actor is constant for every possible input.

Across the three evaluation seeds, approximately **94.9%, 95.0%, and 95.5%** of
requested edge executions were rejected for these respective checkpoints.
Rejected requests fall back to local execution and incur the configured penalty:
[edge scheduling](../environment/diten_env.py#L445).

The training epsilon falls linearly from 0.99 toward 0.01:
[schedule](../baselines/gatma.py#L402),
[runner invocation](../run_comparision.py#L739).
As epsilon decreases, collection increasingly uses the poor saved policy instead
of uniform exploration. This is consistent with the observed declining training
reward and the held-out comparison. Intermediate checkpoints were not available,
so this audit does not reconstruct how each actor evolved through training.

All 30 online actors and critics in each inspected checkpoint have **16,000 Adam
steps**. W&B records 16 replay rounds each episode. Missing optimizer updates
are therefore not the explanation.

## Fidelity and representation findings

### 1. The critic readout is a material deviation from GATMA

Paper equations 29 and 32 aggregate agent state/action features through attention
toward the central cloud node. The adaptation performs one bipartite GAT layer
and an unweighted mean of all device/server embeddings:
[GATMACritic.forward](../baselines/gatma.py#L303).

This still evaluates a global graph, but does not preserve the paper's central
attention readout. Calling the result an exact GATMA reproduction would be
unsupported. The distinction was already disclosed in
[the adaptation mapping](gatma_adapted.md#paper-mapping-and-deliberate-differences);
its performance effect has not been isolated by an ablation.

### 2. Actor inputs discard connection-window timing

The flat environment observation includes each server's window start and end:
[state construction](../environment/diten_env.py#L975).
GATMA reduces them to `window_end > window_start`; neither timestamp nor duration
is passed to its node/edge features:
[topology inputs](../baselines/gatma.py#L62).

Reducing positive durations to 5% of their original values while retaining the
same connection mask left the inspected actor outputs exactly unchanged for
all three checkpoints. DITEN's feasibility condition checks that execution
finishes inside the actual window, so these observations can require different
actions even though this actor receives identical topology inputs.

This is a verified information loss in the adaptation. It is not a claim that
the source paper requires DITEN-specific connection-window features.

### 3. The actor loses explicit server-to-action alignment

The actor aggregates server features into a single device embedding, then
produces logits for fixed action IDs. Server node features do not contain a
server identifier or a position, and the output does not score corresponding
server nodes separately:
[server features](../baselines/gatma.py#L53),
[actor readout](../baselines/gatma.py#L230).

Reversing all server-indexed compute/wait/window blocks together left output
probabilities unchanged for the three checkpoints. The same probe on an
untrained actor using a real observed state left its **pre-head embedding and
probabilities unchanged**, ruling out softmax saturation as the sole explanation.

The fixed output action IDs do not permute with the observed server slots.
The model may memorize distinct resource values as implicit server identities
on one fixed topology, but explicit slot alignment and topology generalization
are not guaranteed by this representation. A connectivity mask on attention
also does not prohibit selecting a disconnected output action.

### 4. Global pooling does not explicitly identify the focal device

Each critic is trained against its own device reward:
[TD target](../utils/training/gatma_training.py#L68).
Its graph inputs contain node types, task/resources, and joint actions, but no
marker for the focal device. Jointly permuting device rows and their actions
leaves global pooled Q unchanged, up to floating-point error, while the reward
target is indexed by device.

Separate critic weights may exploit fixed device resource signatures. Therefore
this is an identity/credit-assignment limitation, not proof that Q can never
learn on the fixed topology, nor an established isolated cause of collapse.

## What matches the paper or passed checks

- Graph attention exists in both actor and critic; actor inference uses its own
  task features and neighbor server features rather than other devices' private
  task observations. Independent online/target actor and critic networks exist.
- Table IV's actor LR `1e-4`, critic LR `1e-5`, gamma `0.95`, epsilon endpoints
  `0.99/0.01`, tau `0.01`, four heads, replay capacity `100000`, and batch `128`
  match the inspected implementation/configurations.
- TD MSE, terminal bootstrap masking, target initialization/freezing, Polyak
  updates, replay action representation, and gradient flow passed existing tests.
  The actor uses a straight-through categorical estimator; it is a documented
  adaptation to discrete DITEN actions, not a source-paper-certified estimator.
- `python -m pytest tests/test_gatma.py -q`: **9 passed, 1 skipped**.
  CUDA was unavailable for the skipped parametrized test; this audit's checkpoint
  evaluation ran on CPU. These tests verify mechanics, not convergence.
- Source paper uses server agents, location/channel/power decisions, deadline
  rewards, and a cloud-edge-end network. Device agents, location-only actions,
  DITEN reward, 1000 episodes, and 16 replay rounds are adaptation choices.
  Table IV specifies 3000 episodes and 200 steps per episode. Equal numerical
  hyperparameters do not imply equal temporal granularity or experiments.

## Limits and proposed next implementation step

The audit establishes policy saturation, frequent rejection, concrete input
limitations, and the readout deviation. It does **not** establish the fraction
of performance loss caused by each issue, prove Q overestimation, or prove that a
particular repair will outperform other methods. Increased Q or reduced actor
loss during training alone cannot establish better real policy performance.

The Drive snapshot matches recorded dataset statistics; historical image byte
hashes and enumeration order were not recorded. Evaluation reconstructs settings
not stored in the old checkpoint from current shared defaults, including
`task_cpu_cycle_scale=1.35`. This is diagnostic held-out evaluation, not exact
historical trajectory replay.

Before implementation, agree on a separate v3 baseline contract:

1. Preserve server/action correspondence and DITEN connection-window information.
2. Restore a global attention readout consistent with equations 29/32; decide
   explicitly how a training-only collector and focal-agent context map to DITEN.
3. Retain the unmasked action protocol for the primary comparison. Adding action
   masking is a separately labeled experiment, since it changes feasibility behavior.
4. Add focused checks for these information/identity contracts and policy diagnostics;
   keep the old basic gradient tests. Do not tune multiple hyperparameters at once.
5. Verify a real-data smoke, then matched multi-seed training and held-out evaluation
   with frozen reward/topology/workload, recorded code revision, and preserved v2
   artifacts. Isolate representation/readout changes before attributing improvement.

For the manuscript, report the current method as **GATMA-Adapted v2** with explicit
differences. Do not infer that the original GATMA algorithm degrades with training
from these adapted runs.
