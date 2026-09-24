# Dual-GAT MAPPO Architecture

This document describes the **current implementation**, using the large topology
as the running example. It is the primary architecture reference and should be
updated whenever the model or training procedure changes.
Select this 30-device/9-server scenario with
`--topology-scenario large_30d_9s` when running `run_comparision.py`.

### Industrial topology variants

The legacy `large_30d_9s` scenario remains unchanged so earlier checkpoints
and experiment results stay reproducible. New experiments that need a more
realistic factory layout can select `large_industrial_30d_9s`. It uses a
180 m × 120 m floor, 15 partially overlapping rectangular aisle loops, 30
mobile devices, and nine servers placed near aisle intersections and production
zones. The arrangement is structured but deliberately asymmetric.

Each scenario resolves one coverage radius per server. Unless explicit radii
are configured as described below, coverage is sampled once when the scenario
is built, then remains fixed for the entire comparison
so every algorithm sees the same physical system. `--topology-seed` reproduces
the sampled radii and `--server-profile` selects:

- `scenario`: the intended default (`heterogeneous` for the industrial map and
  `uniform` for legacy maps);
- `uniform`: the scenario's nominal radius for every server;
- `heterogeneous`: deterministic radii from 83.3% to 111.1% of nominal; and
- `stress`: deterministic radii from 66.7% to 83.3% of nominal.

For the industrial scenario the nominal radius is 18 m, so the heterogeneous
range is 15–20 m and the stress range is 12–15 m. With seed 2026, the default
heterogeneous layout has route-sampled average feasible servers 0.58,
zero-link ratio 0.44, and multi-link ratio 0.02. These metrics describe the
configured geometry; they are not training results.

```bash
python run_comparision.py \
  --topology-scenario large_industrial_30d_9s \
  --topology-seed 2026 \
  --server-profile scenario
```

The approved `modular_cells_30d_9s` variant uses the same 180 m × 120 m floor,
with three distinct modules, 15 routes, 30 devices, and nine servers. Module A
has grouped cells, B has staggered horizontal cells, and C has mixed vertical
cells. Each route carries two devices starting at opposite corners. Every
training episode resets both devices to those same corners and replays the
route from its beginning. In `utils/topology_scenarios_config.py`, edit
`modular_cell_routes()` to change
`rectangle(left, bottom, right, top)` bounds, and edit the named scenario's
`server_locations` and `coverage_radii` to change server positions and radii.
Coordinates and radii are in metres; server entries follow S1 through S9.

This variant's default `heterogeneous` profile uses explicit radii
`(12, 15, 12, 12, 15, 12, 12, 15, 12)` rather than sampling them again.
S2, S5, and S8 therefore remain at 15 m, regardless of topology seed. Both
`--server-profile scenario` and `--server-profile heterogeneous` preserve these
values. Explicit `uniform` or `stress` overrides instead use the nominal 12 m
radius and the profile rules above. Plot the editable configuration from the
`Industrial_task_offloading` directory with:

```bash
python -m utils.topology_scenario_preview \
  --scenarios modular_cells_30d_9s \
  --server-profile scenario \
  --output-dir results/topology_preview/modular_cells
```

Select it for a future experiment with `--topology-scenario modular_cells_30d_9s`.
Adding the scenario does not change the experiment's default topology.

The training application still has three model-facing layers: environment/data
resolves workloads, geometry, and connection windows; algorithms consume the
resulting state/graph; and experiment/output records metadata and artifacts.
The local-first storage workflow has four narrower responsibilities: CLI path
selection, strict local dataset loading, local run publication, and post-run
Drive synchronization. The learned model roles are unchanged: the frozen
Task-GAT produces task priority and the trainable topology policy/critic
consumes connectivity. Storage and topology changes do not add a neural-network
role.

### Local-first dataset and artifact flow

`run_comparision.py` reads only the project-local KolektorSDD replica selected
by `--dataset-path` (default `dataset/KolektorSDD`). Missing or empty data is a
startup error. Synthetic data is available only when an intentional smoke run
passes `--allow-dummy-data`. Dataset mode, resolved path, input-image count, and
mean pixel count are recorded with topology metadata in W&B, JSONL rows, and
model checkpoints.

Plots, CSV/JSONL histories, and model checkpoints are first written below
`--local-output-root` (default `plots`). When `--drive-artifact-root` or
`TASK_OFFLOADING_DRIVE_ROOT` is configured, the runner copies the completed run
directory to `<Drive root>/runs` only after all algorithms finish and all local
files close. A temporary Drive-side directory is renamed only after the copy
succeeds; the local run remains available if synchronization fails.

```mermaid
flowchart LR
    GD["Google Drive dataset"] -->|explicit replica refresh| LD["Project-local KolektorSDD"]
    LD --> TRAIN["Comparison training"]
    TRAIN --> LOCAL["Completed local run<br/>plots + histories + checkpoints"]
    LOCAL -->|post-run copy| TEMP["Drive temporary run"]
    TEMP -->|rename after successful copy| RUNS["Drive runs directory"]
```

## 1. Configuration and end-to-end flow

| Symbol | Meaning | Value |
|---|---|---:|
| `D` | device agents | 30 |
| `S` | edge servers | 9 |
| `M` | subtasks per DAG | 5 |
| `A=S+1` | actions per device: local + servers | 10 |
| `T` | time slots per episode | 100 |
| `L=T×M` | joint transitions per episode | 500 |
| `H`, `Z` | topology-GAT hidden/embedding widths | 64, 64 |
| `K` | PPO passes over one rollout | 4 |

```mermaid
flowchart LR
    TEMPLATE["Representative task DAG<br/>X [5,6], adjacency [5,5]"]
    TGAT["Frozen Task-GAT<br/>6→32→32→1<br/>one inference before RL"]
    ORDER["One shared priority list[5]<br/>reused for all devices and slots"]
    DAG["Per-slot task DAGs"]
    STATE["Joint state<br/>[30,46]"]
    GRAPH["Topology graph<br/>nodes [39,14]<br/>edges [540,7]"]

    subgraph POLICY["Shared topology policy"]
        LGAT["Local Topology-GAT<br/>Actor context"]
        ACTOR["Shared actor<br/>probabilities [30,10]"]
        GGAT["Global Topology-GAT<br/>Critic context"]
        CRITIC["Centralized critic<br/>V(s) [1,1]"]
    end

    ACTION["Joint action [30]"]
    ENV["Environment step"]
    NEXT["Next state [30,46]"]
    BUFFER["Rollout buffer<br/>500 transitions"]
    PPO["4 PPO passes<br/>encoder + actor + critic"]

    TEMPLATE --> TGAT --> ORDER --> STATE
    DAG --> STATE --> GRAPH
    GRAPH --> LGAT --> ACTOR --> ACTION --> ENV --> NEXT
    GRAPH --> GGAT --> CRITIC
    GRAPH --> BUFFER
    ACTION --> BUFFER
    ENV --> BUFFER
    NEXT --> BUFFER
    BUFFER --> PPO
    PPO -.gradient.-> LGAT
    PPO -.gradient.-> GGAT
    PPO -.gradient.-> ACTOR
    PPO -.gradient.-> CRITIC
```

The two GATs have separate roles:

- **Task-GAT** scores the subtasks used to form a priority order. It is
  pretrained and frozen. One representative DAG is inferred before the RL
  episode loop, and the resulting list is reused for every device and slot.
- **Topology-GAT** represents device–server connectivity. Its weights are shared
  by the actor and critic paths and are jointly optimized by MAPPO.

## 2. Task priority and environment state

Task-GAT/GCN receives five nodes from one representative DAG:

```text
Min-max normalized task node features [5,6]
  columns = [hierarchy level, out-degree, cumulative successor CPU,
             own CPU, input size, result size]

Adjacency [5,5]
  current DAG: 1→2, 1→3, 2→4, 3→4, 4→5

[5,6] → GAT(6→32) → [5,32]
      → GAT(32→32) → [5,32]
      → Linear(32→1) → scores [5,1]
      → highest-scoring ready-node topological sort
      → one shared priority list[5]
```

Supervised targets use the normalized upward CPU rank
`rank(v)=CPU(v)+max rank(successor)`. Tasks 2 and 3 remain parallel, but their
workloads are now asymmetric: task 2 uses 50 cycles/pixel and task 3 uses 250
cycles/pixel before the configured CPU scale is applied. Their relative order is
inferred from the model scores and representative DAG; `1→3→2→4→5` is a
possible result, not a hard-coded guarantee. The offloading actor still decides
local versus edge.

Saved-policy inference is handled by `inference_priority_comparison.py`. It
compares the fixed inferred order against `1→2→3→4→5` using deterministic
greedy actions, identical task seeds, and no optimizer or warmup updates.
For training ablations, `run_comparision.py --task-priority off` bypasses the
task graph model and uses the default DAG-safe order `1→2→3→4→5`.

The environment then builds one 46-dimensional state per device:

| Slice | Shape | Content |
|---|---:|---|
| `0:5` | `[5]` | local power/wait and current subtask CPU/input/result size |
| `5:10` | `[5]` | normalized priority ranks for the complete DAG |
| `10:19` | `[9]` | server compute powers |
| `19:28` | `[9]` | server waiting times |
| `28:37` | `[9]` | connection-window starts |
| `37:46` | `[9]` | connection-window ends |

The state exposes DT estimates `f_hat`. Physical execution reconstructs
`delta_f = f_hat - f` and `f = f_hat - delta_f`; estimated delay plus its
deviation therefore equals `CPU/f`. Local computation energy uses
`tau × CPU × f²`; edge-server computation energy is temporarily excluded, so
offloaded execution contributes transmission energy only. Consecutive
subtasks placed on different edge servers are assumed to use a direct
inter-server link with negligible transfer delay and energy; the simulator
currently records both as zero. This is a modeling assumption, not a measured
backhaul property. Current compute ranges are 0.8–1.2 GHz for devices and
2.3–2.5 GHz for servers. Both device and server DT estimates use independent
uniform relative errors in `[-5%, 5%]` at each time slot. Reward weights are
`(lambda1,...,lambda5)=(5,5,5,5,1)`, the failed-offload penalty is `-1`,
and sampled task CPU demand is scaled by `1.4`.

Thus the joint state is `[D,46] = [30,46]`. One joint action completes the
current subtask for all 30 devices, increments their subtask indices, and returns
a new `[30,46]` state. A slot therefore contains five state transitions, not one:

```text
S₁ᵗ → A₁ᵗ → S₂ᵗ → ... → A₅ᵗ → S₆ᵗ
```

## 3. State-to-graph transformation

```mermaid
flowchart TD
    STATE["Joint state [30,46]"]
    DN["Device nodes [30,14]<br/>type + local/subtask + priority"]
    SN["Server nodes [9,14]<br/>type + power + wait"]
    NODES["Node features [39,14]"]
    PAIRS["30×9 device–server pairs<br/>two directions"]
    EI["edge_index [2,540]"]
    EF["edge features [540,7]"]
    ER["reshape [30,9,2,7]<br/>forward/backward: [30,9,7]"]

    STATE --> DN --> NODES
    STATE --> SN --> NODES
    STATE --> PAIRS --> EI
    PAIRS --> EF --> ER
```

Each directed edge has seven features:

```text
[direction_1, direction_2, connected, disconnected,
 window_start, window_end, window_length]
```

The graph contains all 270 device–server pairs, including disconnected pairs.
Unmasked Graph-GAT variants set `use_action_mask=False`, so disconnected server
actions remain in the policy distribution and are handled by the environment.
Mask variants set `use_action_mask=True`; this choice is independent of the
30-device/9-server topology.

Each Topology-GAT layer applies:

```text
node projection:      F_in → F_out
edge projection:      7 → F_out
attention projection: 3F_out → 1

attention(u→v) = LeakyReLU(Wa [Wn xu || Wn xv || We euv])
message(u→v)   = Wn xu + We euv
output(v)      = Σu softmax(attention(u→v)) × message(u→v)
```

The encoder has two layers: `14→64`, ELU, then `64→64`.

### Lightweight topology ablation

The optional lightweight representation is implemented in the graph builder and
agent, but `Lightweight Graph-GAT Warmup MAPPO` is not currently registered in
`build_algorithm_configs()`. When instantiated with the same training settings,
it keeps the node features, actor/critic structure, reward, and rollout length
while compressing the topology representation:

| Component | Standard | Lightweight |
|---|---:|---:|
| topology edges | `[2,540]` | `[2,270]` |
| edge features | `[540,7]` | `[270,3]` |
| direction | device↔server | server→device |
| edge columns | direction flags, connection flags, start/end/length | connected, start, length |
| Topology-GAT | `14→64→64` | `14→64` |

Server→device is retained because each device embedding needs messages from the
nine candidate servers. The reverse edge is omitted, and the one-layer encoder
returns projected server embeddings directly to the actor. For `D=30, S=9`,
stored edge scalars fall from `540×7=3,780` to `270×3=810` per graph.

## 4. Actor and critic tensor paths

```mermaid
flowchart TD
    G["Graph tensors<br/>devices [30,14], servers [9,14]<br/>edges [30,9,7]"]

    subgraph LOCAL["Actor: 30 local graphs in one vectorized call"]
        L1["Topology-GAT 14→64→64"]
        LD["Device embeddings [30,64]"]
        LS["Device-specific server embeddings [30,9,64]"]
        LB["Local branch<br/>64→64→64→1<br/>logits [30,1]"]
        PAIR["Pair concat<br/>device 64 + server 64 + edge 7<br/>[30,9,135]"]
        SB["Server branch<br/>135→64→64→1<br/>logits [30,9]"]
        PROB["Concat + softmax<br/>[30,10]"]
    end

    subgraph GLOBAL["Critic: one global graph"]
        G1["Topology-GAT 14→64→64"]
        GD["Global device embeddings [30,64]"]
        FLAT["Flatten [1,1920]"]
        VALUE["MLP 1920→64→64→1<br/>V(s) [1,1]"]
    end

    G --> L1
    L1 --> LD --> LB --> PROB
    L1 --> LS --> PAIR
    LD --> PAIR
    G --> PAIR --> SB --> PROB
    G --> G1 --> GD --> FLAT --> VALUE
```

The actor sees one local graph per device: that device, all nine servers, and
their edges. The critic sees one global graph in which every server aggregates
messages from all 30 devices. Both paths reuse the same Topology-GAT parameters.

In the lightweight variant, `L1/G1` above becomes `14→64`, and the actor pair
width becomes `64+64+3=131` instead of 135.

Action collection produces:

```text
probabilities [30,10]
  → 30 Categorical distributions
  → joint actions [30] + old log-probabilities [30]
```

The deployment policy requires only the topology encoder and actor: **27,586
parameters**. The centralized critic is training-only.

## 5. Warmup, rollout, and MAPPO update

For Warmup variants, ten auxiliary updates are performed before each joint
action during the first five episodes. Warmup and MAPPO therefore coexist;
MAPPO is active from episode 1. Non-Warmup variants perform no auxiliary updates.

```text
Local embeddings:
  device [30,64], server [30,9,64]
  → pair embeddings [30,9,128]
  → MLP 128→64→64
  → feasibility logits [30,9]
  → window estimates [30,9]

L_aux = BCEWithLogits(feasibility) + 0.5 × MSE(window length)
```

Warmup updates `Topology-GAT + warmup head`; it does not update the actor or
critic. After episode 5, only the auxiliary updates stop.

```mermaid
sequenceDiagram
    participant Q as Priority preprocessing
    participant E as Episode
    participant T as 100 time slots
    participant M as 5 joint decisions/slot
    participant B as Rollout buffer
    participant P as MAPPO update

    Q->>Q: infer one shared priority list
    Q->>E: reuse fixed list for all episodes
    loop each time slot
        E->>T: sample per-slot task DAGs
        T->>M: initialize state [30,46]
        loop each subtask index
            M->>M: build graph
            M->>M: optional 10-step warmup
            M->>M: sample joint action [30]
            M->>B: store transition
        end
    end
    B->>P: 500 transitions
    loop K = 4 full-rollout passes
        P->>P: actor loss + critic loss
        P->>P: update encoder, actor, critic
    end
    P->>B: clear buffer
```

After a complete episode, stacked rollout shapes are:

| Tensor | Shape |
|---|---:|
| actions, rewards, old log-probabilities | `[500,30]` |
| dones, team rewards, values, TD targets, advantages | `[500,1]` |
| device node batch | `[500,30,14]` |
| server node batch | `[500,9,14]` |
| forward/backward edge batches | `[500,30,9,7]` |
| actor probabilities | `[500,30,10]` |
| global critic embeddings | `[500,30,64]` |

The default advantage is a normalized **one-step TD advantage**:

```text
team_reward_l = mean_i reward_l,i                         [500,1]
target_l = team_reward_l + γ V(next_l)(1-done_l)          [500,1]
advantage_l = normalize(target_l - V(current_l))          [500,1]

ratio = exp(new_log_prob - old_log_prob)                  [500,30]
actor loss = PPO clipped objective - entropy bonus
critic loss = MSE(V(current), target)
```

`utils/rl_advantages.py` also provides GAE(λ), selected by `--use-gae` and
applied identically to every MAPPO-family agent:

```text
δ_l = team_reward_l + γ V(s_{l+1})(1-done_l) - V(s_l)
A_l = δ_l + γλ(1-done_l) A_{l+1}
target_l = A_l + V(s_l)                                   λ-return
advantage_l = normalize(A_l)                              full-batch
```

`λ` is fixed at `0.95` in `paper_config.py` and is not tuned, so enabling it
changes every variant the same way. GAE reads `V(s_{l+1})` from the next stored
transition rather than from the stored next state, so the slot-boundary caveat
below only affects the single terminal bootstrap. It also halves the critic
encoder passes per update, from `2×500` to `501`.

By default each PPO epoch processes the full 500-transition rollout.
`--num-minibatches M` shuffles the rollout once per epoch and takes `M`
optimizer steps instead of one; advantages are still normalized over the
complete rollout before the split. `M=1` reproduces the full-batch behaviour
exactly, including the unshuffled ordering.

For the lightweight variant, one compact edge batch `[500,30,9,3]` is reused by
the optimized encoder API; no second direction is stored in the graph state.

### Shared MAPPO comparison baseline

`Shared MAPPO` and `Shared Mask MAPPO` are the canonical MAPPO recipe for
homogeneous agents, added so the Graph-GAT policy can be compared against an
MLP policy under the *same* centralized-training structure:

| Component | `MAPPO` baseline | `Shared MAPPO` | `Graph-GAT MAPPO` |
|---|---|---|---|
| actors | 30 separate networks | **1 shared** | 1 shared |
| critics | 30 copies of `V_i(s)` | **1 `V(s)`** | 1 `V(s)` |
| advantage | per-agent reward `r_i` | **team mean** | team mean |
| observation encoder | MLP over the flat state | MLP over the flat state | topology GAT |

The existing `MAPPO` entry keeps per-agent actors, per-agent critics, and
per-agent rewards; it is closer to independent PPO with a centralized critic
than to MAPPO with parameter sharing. `Shared MAPPO` and `Graph-GAT MAPPO`
both use a shared actor, one team-value critic, and mean team reward. The
comparison also includes different action-scoring heads: the flat-state MLP
outputs all action logits, while the graph actor scores local execution and
each device–server pair separately.

```text
Actor:  device state [46] → MLP 46→64→64→10 → softmax   (shared by 30 devices)
Critic: joint state [1380] → MLP 1380→64→64→1           (one team value)
```

### GATMA-Adapted comparison baseline

`GATMA-Adapted` is adapted to the same DITEN task while preserving its paper-level
learning design. Devices remain the 30 agents, and the action remains one of 10
offloading locations; channel, transmit-power, and cloud actions are omitted.

```text
Actor per device:
  local device-server graph → MLP 14→64
  → one 4-head GAT (concatenated width 64)
  → MLP 64→64→10 → epsilon-greedy action

Critic per device:
  global nodes + joint one-hot actions
  → MLP 24→64 → one 4-head GAT → global mean pool
  → MLP 64→64→1 → Q_i(s,A)
```

GATMA uses a replay capacity of 100,000, batch size 128, actor/critic learning
rates `1e-4/1e-5`, `gamma=0.95`, `tau=0.01`, and epsilon decay `0.99→0.01`.
After collecting an episode, it performs 16 synchronized replay-update rounds.
This exposes each network to 2,048 replay samples, approximately matching the
2,000 samples seen across four PPO passes over a 500-transition rollout. Each
device owns separate online and target networks.

Topology preprocessing is shared within each decision/update without changing
the network equations. One joint action collection builds the node features,
connection mask and all local/global adjacency tensors once for every Actor.
One replay update builds exactly two immutable topology batches, for `state`
and `next_state`, and reuses them across all online and target Actor/Critic
forwards. No cache survives a new observation or an optimizer update.

The runner accepts `--gatma-device auto|cpu|cuda|cuda:<index>`; `auto` selects
CUDA when available. Replay remains on CPU and sampled batches move to the
network device. `--algorithms GATMA` is a legacy alias for `GATMA-Adapted`.
The actor update uses straight-through one-hot actions for its own device and
replayed one-hot actions for the others. The graph adjacency masks messages,
not the categorical action outputs. Nonterminal slot-end transitions wait for
the next slot's real task observation before entering replay.

Unlike the paper's cloud-node attention readout, this adaptation uses global
mean pooling; it also omits channel/power control and continuous link-window
features. These distinctions and the shared evaluation protocol are documented
in [GATMA-Adapted](gatma_adapted.md). The inference comparison script accepts
`--algorithms GATMA-Adapted` and `--device cpu` for CUDA-to-CPU evaluation.

## 6. Parameter and gradient ownership

| Module | Parameters | Updated by |
|---|---:|---|
| frozen Task-GAT | 1,377 | separate pretraining only |
| Topology-GAT encoder | 6,272 | auxiliary warmup and MAPPO |
| shared actor | 21,314 | MAPPO actor loss |
| centralized critic | 127,169 | MAPPO critic loss |
| warmup head | 12,546 | auxiliary warmup only |
| lightweight Topology-GAT encoder | 1,280 | auxiliary warmup and MAPPO |
| lightweight shared actor | 21,058 | MAPPO actor loss |
| GATMA actor ×30 | 299,820 | deterministic policy gradient |
| GATMA critic ×30 | 301,470 | TD Q-loss |

```text
MAPPO-optimized parameters       = 154,755
Warmup variant active parameters = 167,301
Deployment parameters            = 27,586  (encoder + actor)
Lightweight deployment parameters = 22,338 (encoder + actor)
```

During PPO, actor-loss gradients reach the shared encoder through the local
path, while critic-loss gradients reach it through the global path. The two
gradients are combined in the same backward pass.

## 7. Implementation caveats

1. **Priority order:** the highest-scoring ready-node sort respects DAG
   dependencies. The environment also validates the resulting order.
2. **Warmup behavior policy:** during the first five episodes, auxiliary steps
   modify the encoder while the rollout is being collected; the behavior policy
   is therefore not fixed throughout those episodes.
3. **Slot boundary:** the stored next graph after subtask 5 is generated before
   the next slot's new DAG is initialized, so it is not the actual graph used by
   the following decision. One-step TD reads this stored graph at every slot
   boundary; GAE reads it only for the terminal bootstrap.
4. **PPO variant:** the default update uses one-step TD targets, no full
   returns, and a single full-batch step per epoch. GAE(λ=0.95) and
   `M`-way minibatching are available through `--use-gae` and
   `--num-minibatches`, and both are off by default so earlier runs reproduce.
5. **Warmup sampling:** auxiliary warmup uses all current device–server pairs;
   there is no `same_class=True` sampling step in the current implementation.
6. **Model settings:** `utils/paper_config.py` supplies MAPPO, Graph-GAT,
   GATMA-Adapted, and e-ATN-MADDPG hyperparameters. The comparison CLI keeps
   run choices such as topology, seed, devices, GAE, minibatches, and replay
   updates. The old `--hyperparameters-dir` and per-model tuning flags are no
   longer accepted; saved `penalty_0_5` profiles remain historical artifacts.
   W&B config and local JSONL/checkpoints record resolved agent settings and
   the current reward weights, including failed-offload penalty `-1`.

## 8. Where each shape is defined

```text
run_comparision.py
  algorithm name/config and edge_feature_dim=7 or 3
        ↓
utils/topology_graph_state.py :: build_topology_graph_state()
  [D,state_dim] → nodes [D+S,14] and edges [2DS,7] or [DS,3]
        ↓
baselines/graph_gat_mappo.py :: GraphGATMAPPOAgent
  reshapes edges to device×server pairs; coordinates warmup, action selection, PPO
        ↓
models/topology_gat.py :: TopologyGATEncoder
  standard 14→64→64 or lightweight 14→64 message passing

models/graph_gat_heads.py
  shared actor, centralized critic, optional topology warmup head
baselines/graph_gat_rollout.py
  on-policy graph transition and rollout buffer

baselines/gatma.py + utils/gatma_training.py
  shared immutable topology batch + separate 4-head Actor-GAT/Q-Critic per device
  + replay/target updates

utils/rl_advantages.py
  one-step TD, GAE(lambda), full-batch normalization, minibatch indices
        ↓
baselines/mappo.py, baselines/shared_mappo.py, baselines/graph_gat_mappo.py
  every MAPPO-family agent shares the same estimator and minibatch loop
```

`540` and `270` are runtime edge counts (`2×D×S` and `D×S`), so searching only
for the literal number will not find them. Search for `build_topology_graph_state`,
`edge_feature_dim`, `lightweight_topology`, or `class TopologyGATEncoder`.

The runner uses Python with NumPy, PyTorch, and Pillow. `requirements.txt`
lists Pillow and plotting packages; PyTorch and NumPy must also be available in
the runtime environment. The comparison runner fails fast when its local
KolektorSDD replica is unavailable; `--allow-dummy-data` explicitly enables
synthetic tasks for smoke runs. W&B and Optuna have separate optional
requirements files.
