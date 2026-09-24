"""Run training comparisons across e-ATN-MADDPG and baselines.

This script trains multiple algorithms on the DITEN environment and
generates plots and JSON summaries for reward, delay, and energy.
"""

import argparse
import os
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import random
from tqdm import trange
from baselines.gatma import GATMAAgent, build_gatma_topology_batch
from baselines.graph_gat_mappo import GraphGATMAPPOAgent, GraphGATRolloutBuffer
from baselines.maac import MAACAgent
from baselines.mappo import MAPPOAgent, MultiAgentRolloutBuffer
from baselines.shared_mappo import SharedMAPPOAgent, select_shared_joint_actions
from baselines.offloading_baselines import (
    EdgeOnlyAgent,
    FeatureExtractionEdgeAgent,
    LocalOnlyAgent,
    RandomOffloadingAgent,
)
from baselines.scheduling_baselines import BaselineSchedulers
# Environment & Models
from environment.network_env import NetworkEnvironment
from environment.system_model import EdgeServer, IndustrialDevice, TaskDAG
from environment.diten_env import DITENEnv
from dataset.data_loader import KolektorSDDLoader
from models.replay_buffer import MultiAgentReplayBuffer
from models.maddpg import EpsilonATNMADDPGAgent
from utils.artifact_sync import sync_completed_run
from utils.comparison_outputs import (
    build_last_training_state_line,
    build_model_checkpoint,
    flatten_topology_metrics,
    save_comparison_outputs,
)
from utils.priority_model_training import load_or_train_priority_model
from utils.experiment_setup import (
    TASK_PRIORITY_FEATURE_DIM,
    broadcast_priority_order,
    build_priorities,
    build_task_priority_model,
    generate_task_dags_for_episode,
    get_priority_checkpoint_path,
    make_priority_dag_sampler,
)
from utils.experiment_tracking import (
    ExperimentTracker,
    initialize_experiment_tracker,
)
from utils.paper_config import PAPER_PARAMS
from utils.gatma_training import update_gatma_agents_from_buffer
from utils.maddpg_training import update_maddpg_agents_from_buffer
from utils.topology_graph_state import (
    LIGHTWEIGHT_TOPOLOGY_EDGE_FEATURE_DIM,
    STANDARD_TOPOLOGY_EDGE_FEATURE_DIM,
    TopologyGraphState,
    build_topology_graph_state,
)
from utils.topology_scenarios_config import (
    TopologyScenario,
    available_topology_scenario_names,
    compute_topology_metrics,
    device_start_points,
    get_topology_scenario,
)
import time

# On-policy agents that share the PPO advantage estimator and minibatch loop.
PPO_FAMILY_AGENT_CLASSES = (
    MAPPOAgent,
    SharedMAPPOAgent,
    GraphGATMAPPOAgent,
)
FIXED_BASELINE_ALGORITHMS = frozenset(
    {
        "Local Only",
        "Edge Only",
        "Feature Extraction Edge",
        "Random Offloading",
    }
)

def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducible runs.

    Args:
        seed: Random seed value.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)


def parse_args() -> argparse.Namespace:
    """Parse comparison-runner CLI arguments."""
    provisional = PAPER_PARAMS["provisional_table2_needed"]
    parser = argparse.ArgumentParser(description="Run DITEN comparison experiments.")
    parser.add_argument(
        "--topology-scenario",
        default=str(provisional["topology_scenario"]),
        choices=available_topology_scenario_names(),
        help="Named topology scenario to run.",
    )
    parser.add_argument(
        "--topology-seed",
        type=int,
        default=2026,
        help="Seed for reproducible per-server coverage sampling.",
    )
    parser.add_argument(
        "--server-profile",
        choices=("scenario", "uniform", "heterogeneous", "stress"),
        default="scenario",
        help=(
            "Coverage profile. 'scenario' uses each topology's intended "
            "default; the other values override it."
        ),
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Override comparison_full_episodes for smoke or short runs.",
    )
    parser.add_argument(
        "--note",
        default="",
        help="Short experiment note stored in outputs and appended to the output folder.",
    )
    parser.add_argument(
        "--dataset-path",
        default="dataset/KolektorSDD",
        help="Project-local KolektorSDD replica used during training.",
    )
    parser.add_argument(
        "--allow-dummy-data",
        action="store_true",
        help="Allow synthetic tasks when the local dataset is unavailable.",
    )
    parser.add_argument(
        "--local-output-root",
        default="plots",
        help="Local directory that receives completed run outputs and checkpoints.",
    )
    parser.add_argument(
        "--drive-artifact-root",
        default=os.environ.get("TASK_OFFLOADING_DRIVE_ROOT", ""),
        help=(
            "Optional Google Drive artifact root. The completed local run is "
            "copied to its runs directory only after training finishes."
        ),
    )
    parser.add_argument(
        "--graph-gat-device",
        default=None,
        help="Override Graph-GAT device: auto, cpu, cuda, or cuda:<index>.",
    )
    parser.add_argument(
        "--gatma-device",
        default="auto",
        help="GATMA-Adapted device: auto, cpu, cuda, or cuda:<index>.",
    )
    parser.add_argument(
        "--experiment-seed",
        type=int,
        default=None,
        help="Override the configured experiment seed.",
    )
    parser.add_argument(
        "--wandb-mode",
        default=str(provisional["wandb_mode"]),
        choices=("disabled", "online", "offline"),
        help="W&B tracking mode. Disabled by default.",
    )
    parser.add_argument(
        "--wandb-project",
        default=str(provisional["wandb_project"]),
        help="W&B project name.",
    )
    parser.add_argument(
        "--wandb-group",
        default="",
        help="Optional group shared by all algorithm runs in this comparison.",
    )
    parser.add_argument(
        "--algorithms",
        nargs="+",
        default=None,
        help=(
            "Optional exact algorithm names to run, for example "
            "--algorithms \"Graph-GAT MAPPO\". Default: run all configured algorithms."
        ),
    )
    parser.add_argument(
        "--maddpg-updates-per-episode",
        type=int,
        default=1,
        help=(
            "Replay gradient rounds per episode for e-ATN-MADDPG and other "
            "off-policy agents. The default of 1 reproduces earlier runs; set "
            "it to the number of environment steps per episode "
            "(time_slots * subtasks) to match the on-policy PPO update budget."
        ),
    )
    parser.add_argument(
        "--maddpg-actor-replay-actions",
        action="store_true",
        help=(
            "Evaluate the e-ATN-MADDPG policy gradient with the other agents' "
            "replayed actions, as in the published update rule, instead of "
            "their current policy outputs."
        ),
    )
    parser.add_argument(
        "--use-gae",
        action="store_true",
        help=(
            "Use GAE(lambda) advantages instead of one-step TD for every "
            f"MAPPO-family agent. Lambda is fixed at "
            f"{provisional['ppo_gae_lambda']}."
        ),
    )
    parser.add_argument(
        "--num-minibatches",
        type=int,
        default=1,
        help=(
            "PPO minibatches per epoch for every MAPPO-family agent. One "
            "keeps the historical single full-batch update per epoch; the "
            f"configured study value is {provisional['ppo_num_minibatches']}."
        ),
    )
    parser.add_argument(
        "--task-priority",
        choices=("on", "off"),
        default="on",
        help=(
            "Use the configured task graph model ('on') or bypass it with the "
            "default DAG order 1,2,3,4,5 ('off')."
        ),
    )
    return parser.parse_args()


def build_servers_for_scenario(
    scenario: TopologyScenario,
    confirmed: Dict[str, float],
    provisional: Dict[str, float],
) -> List[EdgeServer]:
    """Build edge servers from a named topology scenario."""
    servers = []
    for server_index, location in enumerate(scenario.server_locations, start=1):
        servers.append(
            EdgeServer(
                server_index,
                np.asarray(location, dtype=float),
                np.random.uniform(
                    provisional["server_compute_power_min_ghz"],
                    provisional["server_compute_power_max_ghz"],
                )
                * 1e9,
                confirmed["server_tx_power_w"],
                confirmed["server_energy_coeff"],
                coverage_radius=scenario.coverage_radius_for_server(server_index - 1),
            )
        )
    return servers


def build_devices_for_scenario(
    scenario: TopologyScenario,
    confirmed: Dict[str, float],
    provisional: Dict[str, float],
) -> List[IndustrialDevice]:
    """Build mobile devices from a named topology scenario."""
    starts = device_start_points(scenario)
    devices = []
    for device_index, start_location in enumerate(starts, start=1):
        devices.append(
            IndustrialDevice(
                device_index,
                start_location.copy(),
                np.random.uniform(
                    provisional["device_compute_power_min_ghz"],
                    provisional["device_compute_power_max_ghz"],
                )
                * 1e9,
                confirmed["device_tx_power_w"],
                confirmed["device_energy_coeff"],
                speed_mps=confirmed["device_speed_mps"],
            )
        )
    return devices


def summarize_physical_compute(
    devices: Sequence[IndustrialDevice],
    servers: Sequence[EdgeServer],
    device_seed: int,
    server_seed: int,
) -> Dict[str, float]:
    """Return actual CPU statistics for reproducibility metadata."""
    device_ghz = np.asarray([device.compute_power for device in devices]) / 1e9
    server_ghz = np.asarray([server.compute_power for server in servers]) / 1e9
    return {
        "actual_device_compute_seed": int(device_seed),
        "actual_server_compute_seed": int(server_seed),
        "actual_device_compute_power_min_ghz": float(np.min(device_ghz)),
        "actual_device_compute_power_mean_ghz": float(np.mean(device_ghz)),
        "actual_device_compute_power_max_ghz": float(np.max(device_ghz)),
        "actual_device_compute_power_std_ghz": float(np.std(device_ghz)),
        "actual_server_compute_power_min_ghz": float(np.min(server_ghz)),
        "actual_server_compute_power_mean_ghz": float(np.mean(server_ghz)),
        "actual_server_compute_power_max_ghz": float(np.max(server_ghz)),
        "actual_server_compute_power_std_ghz": float(np.std(server_ghz)),
    }


def build_algorithm_configs(
    graph_gat_device: Optional[str] = None,
    *,
    gatma_device: str = "auto",
    use_gae: bool = False,
    num_minibatches: int = 1,
    maddpg_actor_replay_actions: bool = False,
) -> Dict[str, Dict[str, object]]:
    """Build agent configurations from model defaults and run switches.

    Args:
        graph_gat_device: Device for Graph-GAT MAPPO variants.
        gatma_device: Device for GATMA-Adapted agents.
        use_gae: Enable GAE for every MAPPO-family agent.
        num_minibatches: PPO minibatches per epoch for MAPPO-family agents.
        maddpg_actor_replay_actions: Use replayed peer actions in MADDPG.

    Returns:
        Algorithm names mapped to agent classes and constructor settings.
    """
    confirmed = PAPER_PARAMS["confirmed"]
    provisional = PAPER_PARAMS["provisional_table2_needed"]
    selected_graph_gat_device = graph_gat_device or str(
        provisional["graph_gat_device"]
    )
    configs = {
        "Local Only": {"class": LocalOnlyAgent, "kwargs": {}},
        "Edge Only": {"class": EdgeOnlyAgent, "kwargs": {}},
        "Feature Extraction Edge": {"class": FeatureExtractionEdgeAgent, "kwargs": {}},
        "Random Offloading": {"class": RandomOffloadingAgent, "kwargs": {}},
        "Mask MAPPO": {
                    "class": MAPPOAgent,
                    "kwargs": {
                        "actor_lr": confirmed["rl_lr"],
                        "critic_lr": confirmed["rl_lr"],
                        "gamma": provisional["gamma"],
                        "clip_param": provisional["mappo_clip_param"],
                        "ppo_epochs": int(provisional["mappo_ppo_epochs"]),
                        "entropy_coef": provisional["mappo_entropy_coef"],
                        "value_loss_coef": provisional["mappo_value_loss_coef"],
                        "max_grad_norm": provisional["mappo_max_grad_norm"],
                        "hidden_dim": int(provisional["mappo_hidden_dim"]),
                        "use_action_mask": True,
                    },
        },
        "Shared MAPPO": {
            "class": SharedMAPPOAgent,
            "kwargs": {
                "actor_lr": confirmed["rl_lr"],
                "critic_lr": confirmed["rl_lr"],
                "gamma": provisional["gamma"],
                "clip_param": provisional["mappo_clip_param"],
                "ppo_epochs": int(provisional["mappo_ppo_epochs"]),
                "entropy_coef": provisional["mappo_entropy_coef"],
                "value_loss_coef": provisional["mappo_value_loss_coef"],
                "max_grad_norm": provisional["mappo_max_grad_norm"],
                "hidden_dim": int(provisional["mappo_hidden_dim"]),
                "use_action_mask": False,
            },
        },
        "Shared Mask MAPPO": {
            "class": SharedMAPPOAgent,
            "kwargs": {
                "actor_lr": confirmed["rl_lr"],
                "critic_lr": confirmed["rl_lr"],
                "gamma": provisional["gamma"],
                "clip_param": provisional["mappo_clip_param"],
                "ppo_epochs": int(provisional["mappo_ppo_epochs"]),
                "entropy_coef": provisional["mappo_entropy_coef"],
                "value_loss_coef": provisional["mappo_value_loss_coef"],
                "max_grad_norm": provisional["mappo_max_grad_norm"],
                "hidden_dim": int(provisional["mappo_hidden_dim"]),
                "use_action_mask": True,
            },
        },
        "e-ATN-MADDPG": {
            "class": EpsilonATNMADDPGAgent,
            "batch_size": int(provisional["batch_size"]),
            "kwargs": {
                "use_attention": True,
                "use_epsilon_greedy": True,
                "actor_lr": provisional["maddpg_actor_lr"],
                "critic_lr": provisional["maddpg_critic_lr"],
                "gamma": provisional["gamma"],
                "tau": provisional["tau_soft_update"],
                "hidden_dim": int(provisional["maddpg_hidden_dim"]),
                "epsilon_init": provisional["epsilon_init"],
                "epsilon_min": provisional["epsilon_min"],
                "epsilon_final": provisional["maddpg_epsilon_final"],
                "exploration_fraction": provisional[
                    "maddpg_exploration_fraction"
                ],
            },
        },
        "GATMA-Adapted": {
            "class": GATMAAgent,
            "batch_size": int(provisional["gatma_batch_size"]),
            "replay_buffer_capacity": int(provisional["gatma_replay_buffer_capacity"]),
            "replay_updates_per_episode": int(
                provisional["gatma_replay_updates_per_episode"]
            ),
            "kwargs": {
                "actor_lr": provisional["gatma_actor_lr"],
                "critic_lr": provisional["gatma_critic_lr"],
                "device": gatma_device,
                "gamma": provisional["gatma_gamma"],
                "tau": provisional["gatma_tau"],
                "hidden_dim": int(provisional["gatma_hidden_dim"]),
                "embedding_dim": int(provisional["gatma_embedding_dim"]),
                "num_heads": int(provisional["gatma_num_heads"]),
                "epsilon_init": provisional["gatma_epsilon_init"],
                "epsilon_min": provisional["gatma_epsilon_min"],
                "exploration_fraction": provisional["gatma_exploration_fraction"],
            },
        },
        "MAPPO": {
            "class": MAPPOAgent,
            "kwargs": {
                "actor_lr": confirmed["rl_lr"],
                "critic_lr": confirmed["rl_lr"],
                "gamma": provisional["gamma"],
                "clip_param": provisional["mappo_clip_param"],
                "ppo_epochs": int(provisional["mappo_ppo_epochs"]),
                "entropy_coef": provisional["mappo_entropy_coef"],
                "value_loss_coef": provisional["mappo_value_loss_coef"],
                "max_grad_norm": provisional["mappo_max_grad_norm"],
                "hidden_dim": int(provisional["mappo_hidden_dim"]),
                "use_action_mask": False,
            },
        },

        "Graph-GAT MAPPO": {
            "class": GraphGATMAPPOAgent,
            "kwargs": {
                "lr": confirmed["rl_lr"],
                "encoder_lr": confirmed["rl_lr"],
                "gamma": provisional["gamma"],
                "hidden_dim": int(provisional["graph_gat_hidden_dim"]),
                "embedding_dim": int(provisional["graph_gat_embedding_dim"]),
                "clip_param": provisional["graph_gat_clip_param"],
                "ppo_epochs": int(provisional["graph_gat_ppo_epochs"]),
                "entropy_coef": provisional["graph_gat_entropy_coef"],
                "value_loss_coef": provisional["graph_gat_value_loss_coef"],
                "max_grad_norm": provisional["graph_gat_max_grad_norm"],
                "use_action_mask": False,
                "topology_warmup_episodes": 0,
                "topology_warmup_updates_per_step": 0,
                "device": selected_graph_gat_device,
            },
        },

        "Graph-GAT Mask MAPPO": {
                    "class": GraphGATMAPPOAgent,
                    "kwargs": {
                        "lr": confirmed["rl_lr"],
                        "encoder_lr": confirmed["rl_lr"],
                        "gamma": provisional["gamma"],
                        "hidden_dim": int(provisional["graph_gat_hidden_dim"]),
                        "embedding_dim": int(provisional["graph_gat_embedding_dim"]),
                        "clip_param": provisional["graph_gat_clip_param"],
                        "ppo_epochs": int(provisional["graph_gat_ppo_epochs"]),
                        "entropy_coef": provisional["graph_gat_entropy_coef"],
                        "value_loss_coef": provisional["graph_gat_value_loss_coef"],
                        "max_grad_norm": provisional["graph_gat_max_grad_norm"],
                        "use_action_mask": True,
                        "topology_warmup_episodes": 0,
                        "topology_warmup_updates_per_step": 0,
                        "device": selected_graph_gat_device,
                    },
        },

        "Graph-GAT Warmup MAPPO": {
            "class": GraphGATMAPPOAgent,
            "kwargs": {
                "lr": confirmed["rl_lr"],
                "encoder_lr": confirmed["rl_lr"],
                "gamma": provisional["gamma"],
                "hidden_dim": int(provisional["graph_gat_hidden_dim"]),
                "embedding_dim": int(provisional["graph_gat_embedding_dim"]),
                "clip_param": provisional["graph_gat_clip_param"],
                "ppo_epochs": int(provisional["graph_gat_ppo_epochs"]),
                "entropy_coef": provisional["graph_gat_entropy_coef"],
                "value_loss_coef": provisional["graph_gat_value_loss_coef"],
                "max_grad_norm": provisional["graph_gat_max_grad_norm"],
                "use_action_mask": False,
                "topology_warmup_episodes": int(
                    provisional["graph_gat_topology_warmup_episodes"]
                ),
                "topology_warmup_updates_per_step": int(
                    provisional["graph_gat_topology_warmup_updates_per_step"]
                ),
                "topology_warmup_lr": provisional["graph_gat_topology_warmup_lr"],
                "device": selected_graph_gat_device,
            },
        },

        "Graph-GAT Warmup Mask MAPPO": {
                    "class": GraphGATMAPPOAgent,
                    "kwargs": {
                        "lr": confirmed["rl_lr"],
                        "encoder_lr": confirmed["rl_lr"],
                        "gamma": provisional["gamma"],
                        "hidden_dim": int(provisional["graph_gat_hidden_dim"]),
                        "embedding_dim": int(provisional["graph_gat_embedding_dim"]),
                        "clip_param": provisional["graph_gat_clip_param"],
                        "ppo_epochs": int(provisional["graph_gat_ppo_epochs"]),
                        "entropy_coef": provisional["graph_gat_entropy_coef"],
                        "value_loss_coef": provisional["graph_gat_value_loss_coef"],
                        "max_grad_norm": provisional["graph_gat_max_grad_norm"],
                        "use_action_mask": True,
                        "topology_warmup_episodes": int(
                            provisional["graph_gat_topology_warmup_episodes"]
                        ),
                        "topology_warmup_updates_per_step": int(
                            provisional["graph_gat_topology_warmup_updates_per_step"]
                        ),
                        "topology_warmup_lr": provisional["graph_gat_topology_warmup_lr"],
                        "device": selected_graph_gat_device,
                    },
        },
    }
    ppo_estimator_kwargs = {
        "use_gae": bool(use_gae),
        "gae_lambda": float(provisional["ppo_gae_lambda"]),
        "num_minibatches": int(num_minibatches),
    }
    maddpg_fidelity_kwargs = {
        "epsilon_schedule": str(provisional["maddpg_epsilon_schedule"]),
        "actor_uses_replay_actions": bool(maddpg_actor_replay_actions),
    }
    for config in configs.values():
        if config["class"] in PPO_FAMILY_AGENT_CLASSES:
            config["kwargs"].update(ppo_estimator_kwargs)
        if config["class"] is EpsilonATNMADDPGAgent:
            config["kwargs"].update(maddpg_fidelity_kwargs)
    return configs


def select_algorithm_configs(
    algorithm_configs: Dict[str, Dict[str, object]],
    requested_algorithms: Optional[Sequence[str]],
) -> Dict[str, Dict[str, object]]:
    """Return requested algorithm configs while preserving requested order.

    Args:
        algorithm_configs: All configured algorithms keyed by display name.
        requested_algorithms: Optional exact names requested by the CLI.

    Returns:
        Selected algorithm configuration mapping.

    Raises:
        ValueError: If any requested algorithm name is not configured.
    """
    if not requested_algorithms:
        return algorithm_configs
    # Accept the historical CLI name without running the same baseline twice.
    requested_algorithms = [
        "GATMA-Adapted" if name == "GATMA" else name
        for name in requested_algorithms
    ]
    unknown_algorithms = [
        name for name in requested_algorithms if name not in algorithm_configs
    ]
    if unknown_algorithms:
        valid_names = ", ".join(algorithm_configs)
        unknown_names = ", ".join(unknown_algorithms)
        raise ValueError(
            f"unknown algorithm(s): {unknown_names}. Valid algorithms: {valid_names}"
        )
    return {name: algorithm_configs[name] for name in requested_algorithms}


def _episodes_for_algorithm(algo_name: str, full_episodes: int, baseline_episodes: int) -> int:
    """Return the episode budget for a learning algorithm or fixed baseline."""
    if algo_name in FIXED_BASELINE_ALGORITHMS:
        return min(full_episodes, baseline_episodes)
    return full_episodes


def build_priorities_by_mode(
    task_dags: Dict[int, TaskDAG],
    priority_model: torch.nn.Module,
    mode: str = "gcn",
) -> Dict[int, List[int]]:
    """Build per-device subtask priorities based on the selected mode.

    Args:
        task_dags: Mapping of device IDs to TaskDAGs.
        priority_model: GCN/GAT priority model used to infer priorities.
        mode: Scheduling mode ("gcn", "gat", "random", "greedy").

    Returns:
        Mapping of device IDs to ordered subtask IDs.
    """
    priorities = {}
    for device_id, task_dag in task_dags.items():
        if mode == "random":
            priorities[device_id] = BaselineSchedulers.random_scheduling(task_dag)
        elif mode == "greedy":
            priorities[device_id] = BaselineSchedulers.greedy_scheduling(task_dag)
        elif mode in {"gcn", "gat"}:
            priorities[device_id] = build_priorities(
                {device_id: task_dag}, priority_model
            )[device_id]
        else:
            raise ValueError(
                "priority mode must be one of: gcn, gat, random, greedy"
            )
    return priorities


def build_fixed_priorities_by_mode(
    task_dag: TaskDAG,
    device_ids: List[int],
    priority_model: torch.nn.Module,
    mode: str,
) -> Dict[int, List[int]]:
    """Infer one priority order and reuse it for every device and slot."""
    template_priorities = build_priorities_by_mode(
        {task_dag.id: task_dag}, priority_model, mode
    )
    priority_order = template_priorities[task_dag.id]
    return broadcast_priority_order(device_ids, priority_order)


def _collect_joint_actions(
    agents: Sequence[object], joint_state: np.ndarray, env: DITENEnv | None = None
) -> Tuple[List[int], int, int]:
    """Select joint actions and count local/edge choices.

    Args:
        agents: Agents selecting actions.
        joint_state: Joint state array shaped (num_agents, state_dim).
        env: Optional environment for rule-based baselines that need subtask id.

    Returns:
        Tuple of (joint_actions, local_count, edge_count).
    """
    joint_actions: List[int] = []
    local_count = 0
    edge_count = 0
    full_joint_state = torch.as_tensor(joint_state, dtype=torch.float32)
    gatma_topology = None
    if agents and isinstance(agents[0], GATMAAgent):
        full_joint_state = full_joint_state.to(agents[0].device)
        gatma_topology = build_gatma_topology_batch(
            full_joint_state, agents[0].num_servers
        )
    for agent_index, agent in enumerate(agents):
        agent_state = torch.FloatTensor(joint_state[agent_index])
        if isinstance(agent, GATMAAgent):
            action = agent.select_action(gatma_topology)
        elif env is not None and hasattr(agent, "select_action_for_subtask"):
            device = env.devices[agent_index]
            step_index = env.current_step[device.id]
            priority_order = env.priorities.get(device.id, [])
            subtask_id = priority_order[step_index] if step_index < len(priority_order) else -1
            action = agent.select_action_for_subtask(agent_state, subtask_id)
        elif hasattr(agent, "select_action_with_log_prob"):
            action, log_prob = agent.select_action_with_log_prob(agent_state)
            agent.last_action_log_prob = log_prob
        else:
            action = agent.select_action(agent_state)
        joint_actions.append(action)
        if action == 0:
            local_count += 1
        else:
            edge_count += 1
    return joint_actions, local_count, edge_count


def _collect_graph_gat_actions(
    agent: GraphGATMAPPOAgent,
    joint_state: np.ndarray,
    num_devices: int,
    num_servers: int,
    episode_index: int,
) -> Tuple[List[int], int, int, List[float], TopologyGraphState, float, float, float, float]:
    """Build graph state and select Graph-GAT MAPPO joint actions.

    Args:
        agent: Shared Graph-GAT MAPPO controller.
        joint_state: Joint flat state array shaped `(num_devices, state_dim)`.
        num_devices: Number of device agents.
        num_servers: Number of edge servers.

    Returns:
        Joint actions, local count, edge count, old log-probs, graph state, graph
        build time, topology warmup time/loss, and action selection time.
    """
    graph_build_start = time.perf_counter()
    graph_state = build_topology_graph_state(
        torch.as_tensor(joint_state, dtype=torch.float32),
        num_devices=num_devices,
        num_servers=num_servers,
        lightweight=agent.lightweight_topology,
    )
    graph_build_time = time.perf_counter() - graph_build_start

    graph_warmup_time = 0.0
    graph_warmup_loss = 0.0
    if agent.should_warmup_topology(episode_index):
        agent.synchronize_device()
        graph_warmup_start = time.perf_counter()
        graph_warmup_loss = agent.warmup_topology_encoder(
            graph_state, agent.topology_warmup_updates_per_step
        )
        agent.synchronize_device()
        graph_warmup_time = time.perf_counter() - graph_warmup_start

    agent.synchronize_device()
    graph_action_start = time.perf_counter()
    joint_actions, old_log_probs = agent.select_actions_with_log_probs(graph_state)
    agent.synchronize_device()
    graph_action_time = time.perf_counter() - graph_action_start
    local_count = sum(1 for action in joint_actions if action == 0)
    edge_count = len(joint_actions) - local_count
    return (
        joint_actions,
        local_count,
        edge_count,
        old_log_probs,
        graph_state,
        graph_build_time,
        graph_warmup_time,
        graph_warmup_loss,
        graph_action_time,
    )


def _summarize_step_metrics(step_metrics: Sequence[Dict[str, float]]) -> Dict[str, float]:
    """Summarize timing diagnostics from environment step metrics.

    Args:
        step_metrics: Per-device metrics from `DITENEnv.last_step_metrics`.

    Returns:
        Aggregated timing diagnostics for the step.
    """
    summary = {
        "local_time": 0.0,
        "server_time": 0.0,
        "attempted_server_time": 0.0,
        "transfer_time": 0.0,
        "queue_or_wait_time": 0.0,
        "penalty_time": 0.0,
        "penalty_count": 0.0,
        "requested_local_count": 0.0,
        "requested_edge_count": 0.0,
        "resolved_local_count": 0.0,
        "resolved_edge_count": 0.0,
    }
    for metric in step_metrics:
        requested_action = int(metric.get("requested_action", -1.0))
        resolved_action = int(metric.get("action", -1.0))
        summary["local_time"] += float(metric.get("local_time", 0.0))
        summary["server_time"] += float(metric.get("server_time", 0.0))
        summary["attempted_server_time"] += float(metric.get("attempted_server_time", 0.0))
        summary["transfer_time"] += float(metric.get("transfer_time", 0.0))
        summary["queue_or_wait_time"] += float(metric.get("queue_or_wait_time", 0.0))
        summary["penalty_time"] += float(metric.get("penalty_time", 0.0))
        summary["penalty_count"] += float(metric.get("penalty_applied", 0.0))
        if requested_action == 0:
            summary["requested_local_count"] += 1.0
        elif requested_action > 0:
            summary["requested_edge_count"] += 1.0
        if resolved_action == 0:
            summary["resolved_local_count"] += 1.0
        elif resolved_action > 0:
            summary["resolved_edge_count"] += 1.0
    return summary


def _format_diagnostic_summary(algo_name: str, episode_number: int, history: Dict[str, List[float]]) -> str:
    """Format timing diagnostics as a readable multi-line block.

    Args:
        algo_name: Algorithm display name.
        episode_number: One-based episode number.
        history: Metric history populated by `train_algorithm`.

    Returns:
        Human-readable diagnostic summary.
    """
    requested_local = history["requested_local_count"][-1]
    requested_edge = history["requested_edge_count"][-1]
    resolved_local = history["resolved_local_count"][-1]
    resolved_edge = history["resolved_edge_count"][-1]
    penalty_count = history["penalty_count"][-1]
    penalty_time = history["penalty_time"][-1]
    local_time = history["local_time"][-1]
    server_time = history["server_time"][-1]
    transfer_time = history["transfer_time"][-1]
    wait_time = history["queue_or_wait_time"][-1]

    summary = (
        f"[{algo_name}] Episode {episode_number} diagnostics\n"
        f"  Requested actions: local={requested_local:.0f} edge={requested_edge:.0f}\n"
        f"  Actual execution:  local={resolved_local:.0f} edge={resolved_edge:.0f}\n"
        f"  Penalties:         count={penalty_count:.0f} time={penalty_time:.3f}s\n"
        f"  Timing/device-task: local={local_time:.3f}s server={server_time:.3f}s "
        f"transfer={transfer_time:.3f}s wait={wait_time:.3f}s"
    )
    graph_transition_count = history.get("graph_transition_count", [0.0])[-1]
    if graph_transition_count > 0:
        graph_build_time = history["graph_build_time"][-1]
        graph_warmup_time = history.get("graph_warmup_time", [0.0])[-1]
        graph_warmup_loss = history.get("graph_warmup_loss", [0.0])[-1]
        graph_warmup_count = history.get("graph_warmup_count", [0.0])[-1]
        graph_action_time = history["graph_action_time"][-1]
        graph_update_time = history["graph_update_time"][-1]
        summary += (
            f"\n  Graph-GAT cost:    build={graph_build_time:.3f}s "
            f"warmup={graph_warmup_time:.3f}s/{graph_warmup_count:.0f} "
            f"warmup_loss={graph_warmup_loss:.4f} "
            f"action={graph_action_time:.3f}s update={graph_update_time:.3f}s "
            f"transitions={graph_transition_count:.0f}"
        )
    runtime_env_step = history.get("runtime_env_step_time", [0.0])[-1]
    runtime_windows = history.get("runtime_connection_window_time", [0.0])[-1]
    runtime_unaccounted = history.get("runtime_unaccounted_time", [0.0])[-1]
    if runtime_env_step > 0.0 or runtime_windows > 0.0:
        window_requests = history.get(
            "runtime_connection_window_requests", [0.0]
        )[-1]
        window_updates = history.get(
            "runtime_connection_window_updates", [0.0]
        )[-1]
        summary += (
            f"\n  Wall-clock cost:   env_step={runtime_env_step:.3f}s "
            f"windows={runtime_windows:.3f}s "
            f"requests/updates={window_requests:.0f}/{window_updates:.0f} "
            f"unaccounted={runtime_unaccounted:.3f}s"
        )
    return summary


def _should_print_diagnostics(episode_number: int, num_episodes: int) -> bool:
    """Return whether to print readable diagnostics for this episode."""
    provisional = PAPER_PARAMS["provisional_table2_needed"]
    interval = int(provisional["diagnostic_interval_episodes"])
    return episode_number == num_episodes or (
        interval > 0 and episode_number % interval == 0
    )


def _build_episode_tracking_metrics(
    history: Dict[str, List[float]],
    episode_number: int,
    num_episodes: int,
    episode_elapsed_seconds: float,
    total_elapsed_seconds: float,
    graph_gat_agent: Optional[GraphGATMAPPOAgent] = None,
) -> Dict[str, float]:
    """Build one complete W&B record from the latest episode metrics."""
    def latest(key: str) -> float:
        """Return the latest history value, or zero for optional metrics."""
        return float(history.get(key, [0.0])[-1])

    progress_fraction = episode_number / max(num_episodes, 1)
    average_episode_seconds = total_elapsed_seconds / max(episode_number, 1)
    eta_seconds = average_episode_seconds * max(num_episodes - episode_number, 0)
    metrics = {
        "episode": float(episode_number),
        "training/progress_percent": progress_fraction * 100.0,
        "training/episode_seconds": episode_elapsed_seconds,
        "training/elapsed_seconds": total_elapsed_seconds,
        "training/eta_seconds": eta_seconds,
        "performance/reward": history["reward"][-1],
        "performance/delay_seconds": history["delay"][-1],
        "performance/energy_joules": history["energy"][-1],
        "actions/local_ratio_percent": history["local_ratio"][-1],
        "actions/edge_ratio_percent": history["edge_ratio"][-1],
        "actions/requested_local_count": history["requested_local_count"][-1],
        "actions/requested_edge_count": history["requested_edge_count"][-1],
        "actions/resolved_local_count": history["resolved_local_count"][-1],
        "actions/resolved_edge_count": history["resolved_edge_count"][-1],
        "execution/local_seconds": history["local_time"][-1],
        "execution/server_seconds": history["server_time"][-1],
        "execution/transfer_seconds": history["transfer_time"][-1],
        "execution/wait_seconds": history["queue_or_wait_time"][-1],
        "penalty/count": history["penalty_count"][-1],
        "penalty/seconds": history["penalty_time"][-1],
        "maddpg/replay_buffer_size": latest("replay_buffer_size"),
        "maddpg/update_rounds": latest("maddpg_update_rounds"),
        "maddpg/actor_loss": latest("maddpg_actor_loss"),
        "maddpg/critic_loss": latest("maddpg_critic_loss"),
        "maddpg/mean_q": latest("maddpg_mean_q"),
        "maddpg/epsilon": latest("epsilon"),
        "simulation/local_compute_seconds": history["local_time"][-1],
        "simulation/edge_compute_seconds": history["server_time"][-1],
        "simulation/transfer_seconds": history["transfer_time"][-1],
        "simulation/queue_wait_seconds": history["queue_or_wait_time"][-1],
        "simulation/penalty_seconds": history["penalty_time"][-1],
        "graph/build_seconds": history["graph_build_time"][-1],
        "graph/warmup_seconds": history["graph_warmup_time"][-1],
        "graph/warmup_loss": history["graph_warmup_loss"][-1],
        "graph/action_seconds": history["graph_action_time"][-1],
        "graph/update_seconds": history["graph_update_time"][-1],
        "graph/transition_count": history["graph_transition_count"][-1],
        "runtime/dag_generation_seconds": latest("runtime_dag_generation_time"),
        "runtime/priority_inference_seconds": latest("runtime_priority_inference_time"),
        "runtime/start_slot_seconds": latest("runtime_start_slot_time"),
        "runtime/action_collection_seconds": latest("runtime_action_collection_time"),
        "runtime/env_step_seconds": latest("runtime_env_step_time"),
        "runtime/metric_summary_seconds": latest("runtime_metric_summary_time"),
        "runtime/rollout_storage_seconds": latest("runtime_rollout_storage_time"),
        "runtime/model_update_seconds": latest("runtime_model_update_time"),
        "runtime/connection_window_seconds": latest(
            "runtime_connection_window_time"
        ),
        "runtime/connection_window_requests": latest(
            "runtime_connection_window_requests"
        ),
        "runtime/connection_window_updates": latest(
            "runtime_connection_window_updates"
        ),
        "runtime/connection_window_samples": latest(
            "runtime_connection_window_samples"
        ),
        "runtime/joint_state_seconds": latest("runtime_joint_state_time"),
        "runtime/accounted_seconds": latest("runtime_accounted_time"),
        "runtime/unaccounted_seconds": latest("runtime_unaccounted_time"),
    }
    if graph_gat_agent is not None and graph_gat_agent.device.type == "cuda":
        megabyte = 1024.0**2
        metrics.update(
            {
                "gpu/memory_allocated_mb": torch.cuda.memory_allocated(
                    graph_gat_agent.device
                )
                / megabyte,
                "gpu/memory_reserved_mb": torch.cuda.memory_reserved(
                    graph_gat_agent.device
                )
                / megabyte,
                "gpu/max_memory_allocated_mb": torch.cuda.max_memory_allocated(
                    graph_gat_agent.device
                )
                / megabyte,
            }
        )
    return metrics


def _update_agents_from_buffer(
    agents: Sequence[object],
    replay_buffer: MultiAgentReplayBuffer,
    batch_size: int,
    gamma: float,
) -> Dict[str, float]:
    """Update agents from replay buffer samples when available.

    Args:
        agents: Agents to update.
        replay_buffer: Shared replay buffer.
        batch_size: Batch size for sampling.
        gamma: Discount factor.
    """
    if agents and all(isinstance(agent, GATMAAgent) for agent in agents):
        return update_gatma_agents_from_buffer(
            agents,
            replay_buffer,
            batch_size,
            gamma,
        )

    if agents and all(
        isinstance(agent, EpsilonATNMADDPGAgent) for agent in agents
    ):
        return update_maddpg_agents_from_buffer(
            agents,
            replay_buffer,
            batch_size,
            gamma,
        )

    if len(replay_buffer) < batch_size:
        return {
            "update_rounds": 0.0,
            "actor_loss": 0.0,
            "critic_loss": 0.0,
            "mean_q": 0.0,
        }

    state_b, action_b, reward_b, next_state_b, _ = replay_buffer.sample(
        batch_size
    )
    for agent_index, agent in enumerate(agents):
        action_dim = agent.action_dim
        action_b_onehot = F.one_hot(action_b.long(), num_classes=action_dim).float()
        if hasattr(agent, "update_agent"):
            agent.update_agent(state_b, action_b, reward_b, next_state_b, agent_index=agent_index)
            if hasattr(agent, "update_epsilon"):
                agent.update_epsilon()
            continue

        if hasattr(agent, "target_actor"):
            agent_rewards = reward_b[:, agent_index].unsqueeze(1)
            with torch.no_grad():
                target_joint_action_idx = torch.stack(
                    [
                        torch.argmax(a.target_actor(next_state_b[:, j, :]), dim=1)
                        for j, a in enumerate(agents)
                    ],
                    dim=1,
                )
                target_joint_actions = F.one_hot(
                    target_joint_action_idx.long(), num_classes=action_dim
                ).float()
                target_q = agent.target_critic(next_state_b, target_joint_actions)
                y_i = agent_rewards + gamma * target_q

            current_q = agent.critic(state_b, action_b_onehot)
            critic_loss = F.mse_loss(current_q, y_i)
            agent.critic_optimizer.zero_grad()
            critic_loss.backward()
            agent.critic_optimizer.step()

            predicted_actions = agent.actor(state_b[:, agent_index, :])
            predicted_joint_actions = action_b_onehot.clone()
            predicted_joint_actions[:, agent_index] = predicted_actions

            actor_loss = -agent.critic(state_b, predicted_joint_actions).mean()
            agent.actor_optimizer.zero_grad()
            actor_loss.backward()
            agent.actor_optimizer.step()

            agent.soft_update(agent.target_actor, agent.actor)
            agent.soft_update(agent.target_critic, agent.critic)

        if hasattr(agent, "update_epsilon"):
            agent.update_epsilon()
    return {
        "update_rounds": 1.0,
        "actor_loss": 0.0,
        "critic_loss": 0.0,
        "mean_q": 0.0,
    }


def _update_agents_from_rollout(
    agents: Sequence[object],
    rollout_buffer: MultiAgentRolloutBuffer,
    gamma: float,
) -> None:
    """Update on-policy MAPPO agents from a collected rollout and clear it."""
    if len(rollout_buffer) == 0:
        return

    state_b, action_b, reward_b, next_state_b, old_log_prob_b, done_b = (
        rollout_buffer.as_tensors()
    )
    for agent_index, agent in enumerate(agents):
        agent.update_agent(
            state_b,
            action_b,
            reward_b,
            next_state_b,
            agent_index=agent_index,
            old_log_prob_b=old_log_prob_b,
            done_b=done_b,
        )
    rollout_buffer.clear()


def _update_graph_gat_mappo_from_rollout(
    agent: GraphGATMAPPOAgent,
    rollout_buffer: GraphGATRolloutBuffer,
    gamma: float,
) -> float:
    """Update Graph-GAT MAPPO from graph rollouts and return update time."""
    if len(rollout_buffer) == 0:
        return 0.0

    agent.gamma = gamma
    agent.synchronize_device()
    update_start = time.perf_counter()
    agent.update_from_rollout(rollout_buffer)
    agent.synchronize_device()
    return time.perf_counter() - update_start


def train_algorithm(
    algo_name: str,
    agent_config: Dict[str, object],
    devices: Sequence[IndustrialDevice],
    servers: Sequence[EdgeServer],
    network_env: NetworkEnvironment,
    data_loader: KolektorSDDLoader,
    priority_model: torch.nn.Module,
    num_episodes: int,
    priority_mode: str = "gcn",
    topology_scenario: Optional[TopologyScenario] = None,
    topology_metrics: Optional[Dict[str, object]] = None,
    experiment_note: str = "",
    experiment_tracker: Optional[ExperimentTracker] = None,
    experiment_seed: Optional[int] = None,
    fixed_priority_order: Optional[List[int]] = None,
    episode_callback: Optional[
        Callable[[int, Dict[str, List[float]]], None]
    ] = None,
    show_progress: bool = True,
    replay_updates_per_episode: int = 1,
) -> Tuple[Dict[str, List[float]], Optional[Dict[str, Any]]]:
    """Train one algorithm configuration and return metrics and checkpoint.

    Args:
        algo_name: Display name for logging.
        agent_config: Dict with "class" and optional "kwargs" for agent init.
        devices: List of IndustrialDevice instances.
        servers: List of EdgeServer instances.
        network_env: NetworkEnvironment instance.
        data_loader: Dataset loader for DAG generation.
        priority_model: GCN/GAT model used for priority extraction.
        num_episodes: Number of episodes to train.
        priority_mode: Scheduling mode ("gcn", "gat", "random", "greedy").
        topology_scenario: Optional named topology scenario for mobility routes.
        topology_metrics: Optional topology metrics stored in checkpoints.
        experiment_note: Optional note stored in checkpoint metadata.
        experiment_tracker: Optional per-episode external metric tracker.
        experiment_seed: Optional training seed override.
        fixed_priority_order: Optional shared order inferred before this call.
        episode_callback: Optional callback invoked after each completed update.
        show_progress: Whether to print progress and diagnostic summaries.
        replay_updates_per_episode: Gradient rounds per episode for off-policy
            replay agents such as e-ATN-MADDPG.

    Returns:
        Tuple of metric history and optional trainable model checkpoint payload.
    """
    if show_progress:
        print(f"\n{'='*50}\nStarting Training for: {algo_name}\n{'='*50}")
    confirmed = PAPER_PARAMS["confirmed"]
    provisional = PAPER_PARAMS["provisional_table2_needed"]
    if experiment_seed is None:
        experiment_seed = int(provisional["experiment_seed"])
    set_seed(experiment_seed)  # Reset seed for fair comparison.
    # The loader keeps a private generator so every algorithm replays the same
    # task-workload sequence, even though agents consume the global `random`
    # stream at different rates (e-ATN-MADDPG draws per epsilon-greedy action).
    data_loader.reseed(experiment_seed)

    time_slots = int(confirmed["time_slots"])
    env = DITENEnv(
        devices,
        servers,
        network_env,
        slot_duration=confirmed["slot_duration_s"],
        subslot_count=int(provisional["subslot_count"]),
        time_slots=time_slots,
        lambda1=provisional["lambda1"],
        lambda2=provisional["lambda2"],
        lambda3=provisional["lambda3"],
        lambda4=provisional["lambda4"],
        lambda5=provisional["lambda5"],
        p_out_value=provisional["p_out_value"],
        local_estimation_error=provisional["local_estimation_error"],
        edge_estimation_error=provisional["edge_estimation_error"],
        route_rectangles=(
            topology_scenario.route_rectangles if topology_scenario is not None else None
        ),
        world_size=(
            topology_scenario.world_size
            if topology_scenario is not None
            else (100.0, 100.0)
        ),
    )
    # State/Action dimensions
    STATE_DIM = env.get_state_dim()
    ACTION_DIM = 1 + len(servers)
    graph_priority_width = STATE_DIM - (5 + 4 * len(servers))
    graph_node_feature_dim = 9 + graph_priority_width
    lightweight_topology = bool(
        agent_config.get("kwargs", {}).get("lightweight_topology", False)
    )
    graph_edge_feature_dim = (
        LIGHTWEIGHT_TOPOLOGY_EDGE_FEATURE_DIM
        if lightweight_topology
        else STANDARD_TOPOLOGY_EDGE_FEATURE_DIM
    )
    # JOINT_STATE_DIM = STATE_DIM * len(devices)
    # JOINT_ACTION_DIM = len(devices)
    
    # Initialize Agents dynamically based on the config
    agent_class = agent_config["class"]
    uses_graph_gat_mappo = agent_class is GraphGATMAPPOAgent
    uses_gatma = agent_class is GATMAAgent
    uses_shared_mappo = agent_class is SharedMAPPOAgent
    agents: List[object] = []
    if uses_graph_gat_mappo:
        agents.append(
            agent_class(
                num_devices=len(devices),
                num_servers=len(servers),
                node_feature_dim=graph_node_feature_dim,
                edge_feature_dim=graph_edge_feature_dim,
                **agent_config.get("kwargs", {}),
            )
        )
        if show_progress:
            print(f"[{algo_name}] Graph-GAT device: {agents[0].device}")
    elif uses_shared_mappo:
        agents.append(
            agent_class(
                state_dim=STATE_DIM,
                action_dim=ACTION_DIM,
                num_agents=len(devices),
                **agent_config.get("kwargs", {}),
            )
        )
    else:
        for agent_index in range(len(devices)):
            extra_agent_kwargs = {}
            if uses_gatma:
                extra_agent_kwargs = {
                    "num_servers": len(servers),
                    "agent_index": agent_index,
                }
            agent = agent_class(
                state_dim=STATE_DIM, action_dim=ACTION_DIM,
                num_agents=len(devices),
                **extra_agent_kwargs,
                **agent_config.get("kwargs", {})
            )
            agents.append(agent)

    if uses_gatma and show_progress:
        print(f"[{algo_name}] GATMA device: {agents[0].device}")

    replay_buffer = MultiAgentReplayBuffer(
        capacity=int(
            agent_config.get(
                "replay_buffer_capacity",
                provisional["replay_buffer_capacity"],
            )
        )
    )
    rollout_buffer = MultiAgentRolloutBuffer()
    graph_rollout_buffer = GraphGATRolloutBuffer()
    uses_rollout_buffer = uses_shared_mappo or (
        not uses_graph_gat_mappo
        and all(hasattr(agent, "select_action_with_log_prob") for agent in agents)
    )
    batch_size = int(agent_config.get("batch_size", provisional["batch_size"]))
    gatma_replay_updates_per_episode = max(
        1, int(agent_config.get("replay_updates_per_episode", 16))
    )
    gamma = float(
        agent_config.get("kwargs", {}).get("gamma", provisional["gamma"])
    )
    shared_priority_inference_time = 0.0
    if fixed_priority_order is None:
        template_task_dags = generate_task_dags_for_episode(
            [devices[0]],
            data_loader,
            t_max=provisional["t_max"],
            e_max=provisional["e_max"],
            cpu_cycle_scale=provisional["task_cpu_cycle_scale"],
        )
        template_task_dag = template_task_dags[devices[0].id]
        shared_priority_start = time.perf_counter()
        fixed_priorities = build_fixed_priorities_by_mode(
            template_task_dag,
            [device.id for device in devices],
            priority_model,
            priority_mode,
        )
        shared_priority_inference_time = (
            time.perf_counter() - shared_priority_start
        )
    else:
        fixed_priorities = broadcast_priority_order(
            [device.id for device in devices], fixed_priority_order
        )
    if show_progress:
        print(
            f"[{algo_name}] Fixed task-priority order: "
            f"{fixed_priorities[devices[0].id]}"
        )
    
    # Track metrics
    history = {
        "reward": [],
        "delay": [],
        "energy": [],
        "local_ratio": [],
        "edge_ratio": [],
        "local_time": [],
        "server_time": [],
        "attempted_server_time": [],
        "transfer_time": [],
        "queue_or_wait_time": [],
        "penalty_time": [],
        "penalty_count": [],
        "requested_local_count": [],
        "requested_edge_count": [],
        "resolved_local_count": [],
        "resolved_edge_count": [],
        "replay_buffer_size": [],
        "maddpg_update_rounds": [],
        "maddpg_actor_loss": [],
        "maddpg_critic_loss": [],
        "maddpg_mean_q": [],
        "epsilon": [],
        "graph_build_time": [],
        "graph_warmup_time": [],
        "graph_warmup_loss": [],
        "graph_warmup_count": [],
        "graph_action_time": [],
        "graph_update_time": [],
        "graph_transition_count": [],
        "runtime_dag_generation_time": [],
        "runtime_priority_inference_time": [],
        "runtime_start_slot_time": [],
        "runtime_action_collection_time": [],
        "runtime_env_step_time": [],
        "runtime_metric_summary_time": [],
        "runtime_rollout_storage_time": [],
        "runtime_model_update_time": [],
        "runtime_connection_window_time": [],
        "runtime_connection_window_requests": [],
        "runtime_connection_window_updates": [],
        "runtime_connection_window_samples": [],
        "runtime_joint_state_time": [],
        "runtime_accounted_time": [],
        "runtime_unaccounted_time": [],
        "runtime_tracking_log_time": [],
    }

    training_start_time = time.perf_counter()
    episode_iterator = trange(
        num_episodes,
        desc=f"Training {algo_name}",
        leave=True,
        disable=not show_progress,
    )
    for episode in episode_iterator:
        episode_start_time = time.perf_counter()
        for agent in agents:
            set_training_progress = getattr(
                agent, "set_training_progress", None
            )
            if set_training_progress is not None:
                set_training_progress(episode, num_episodes)
        env.reset_episode()
        count_local = 0
        count_edge = 0
        slot_rewards = []
        slot_delays = []
        slot_energies = []
        slot_local_times = []
        slot_server_times = []
        slot_attempted_server_times = []
        slot_transfer_times = []
        slot_queue_or_wait_times = []
        slot_penalty_times = []
        slot_penalty_counts = []
        slot_requested_local_counts = []
        slot_requested_edge_counts = []
        slot_resolved_local_counts = []
        slot_resolved_edge_counts = []
        episode_graph_build_time = 0.0
        episode_graph_warmup_time = 0.0
        episode_graph_warmup_loss_total = 0.0
        episode_graph_warmup_count = 0.0
        episode_graph_action_time = 0.0
        episode_graph_update_time = 0.0
        episode_graph_transition_count = 0.0
        episode_dag_generation_time = 0.0
        episode_priority_inference_time = (
            shared_priority_inference_time if episode == 0 else 0.0
        )
        episode_start_slot_time = 0.0
        episode_action_collection_time = 0.0
        episode_env_step_time = 0.0
        episode_metric_summary_time = 0.0
        episode_rollout_storage_time = 0.0
        episode_model_update_time = 0.0
        pending_gatma_transition = None
        episode_done = False
        for _ in range(time_slots):
            if episode_done:
                break
            slot_reward = 0.0
            slot_local = 0
            slot_edge = 0
            slot_steps = 0
            prev_delay_mean = np.mean(list(env.device_accumulated_delay.values()))
            prev_energy_mean = np.mean(list(env.device_accumulated_energy.values()))
            dag_generation_start = time.perf_counter()
            task_dags = generate_task_dags_for_episode(
                devices,
                data_loader,
                t_max=provisional["t_max"],
                e_max=provisional["e_max"],
                cpu_cycle_scale=provisional["task_cpu_cycle_scale"],
            )
            episode_dag_generation_time += (
                time.perf_counter() - dag_generation_start
            )
            start_slot_start = time.perf_counter()
            current_joint_state = env.start_time_slot(task_dags, fixed_priorities)
            episode_start_slot_time += time.perf_counter() - start_slot_start
            if pending_gatma_transition is not None:
                # Bootstrap from the new task actually seen by the next action,
                # not the exhausted DAG returned by the preceding slot's step.
                replay_buffer.push(
                    *pending_gatma_transition, current_joint_state, done=False
                )
                pending_gatma_transition = None
            slot_done = False
            while not slot_done and not episode_done:
                action_collection_start = time.perf_counter()
                if uses_graph_gat_mappo:
                    (
                        joint_actions,
                        local_count,
                        edge_count,
                        old_log_probs,
                        graph_state,
                        graph_build_time,
                        graph_warmup_time,
                        graph_warmup_loss,
                        graph_action_time,
                    ) = _collect_graph_gat_actions(
                        agent=agents[0],
                        joint_state=current_joint_state,
                        num_devices=len(devices),
                        num_servers=len(servers),
                        episode_index=episode,
                    )
                    episode_graph_build_time += graph_build_time
                    episode_graph_warmup_time += graph_warmup_time
                    if graph_warmup_time > 0.0:
                        episode_graph_warmup_loss_total += graph_warmup_loss
                        episode_graph_warmup_count += 1.0
                    episode_graph_action_time += graph_action_time
                elif uses_shared_mappo:
                    joint_actions, local_count, edge_count = (
                        select_shared_joint_actions(
                            agents[0], current_joint_state
                        )
                    )
                else:
                    joint_actions, local_count, edge_count = _collect_joint_actions(
                        agents, current_joint_state, env
                    )
                episode_action_collection_time += (
                    time.perf_counter() - action_collection_start
                )
                count_local += local_count
                count_edge += edge_count
                slot_local += local_count
                slot_edge += edge_count

                env_step_start = time.perf_counter()
                next_joint_state, joint_rewards, step_episode_done, info = env.step(
                    joint_actions
                )
                episode_env_step_time += time.perf_counter() - env_step_start
                metric_summary_start = time.perf_counter()
                metric_summary = _summarize_step_metrics(env.last_step_metrics)
                episode_metric_summary_time += (
                    time.perf_counter() - metric_summary_start
                )
                slot_done = info.get("slot_done", False)
                episode_done = step_episode_done
                # Aggregate team reward as average per-agent immediate reward.
                step_reward = sum(joint_rewards) / len(devices)
                slot_reward += step_reward
                slot_steps += 1
                slot_local_times.append(metric_summary["local_time"])
                slot_server_times.append(metric_summary["server_time"])
                slot_attempted_server_times.append(metric_summary["attempted_server_time"])
                slot_transfer_times.append(metric_summary["transfer_time"])
                slot_queue_or_wait_times.append(metric_summary["queue_or_wait_time"])
                slot_penalty_times.append(metric_summary["penalty_time"])
                slot_penalty_counts.append(metric_summary["penalty_count"])
                slot_requested_local_counts.append(metric_summary["requested_local_count"])
                slot_requested_edge_counts.append(metric_summary["requested_edge_count"])
                slot_resolved_local_counts.append(metric_summary["resolved_local_count"])
                slot_resolved_edge_counts.append(metric_summary["resolved_edge_count"])
                rollout_storage_start = time.perf_counter()
                if uses_graph_gat_mappo:
                    next_graph_build_start = time.perf_counter()
                    next_graph_state = build_topology_graph_state(
                        torch.as_tensor(next_joint_state, dtype=torch.float32),
                        num_devices=len(devices),
                        num_servers=len(servers),
                        lightweight=agents[0].lightweight_topology,
                    )
                    episode_graph_build_time += (
                        time.perf_counter() - next_graph_build_start
                    )
                    graph_rollout_buffer.push(
                        graph_state=graph_state,
                        actions=joint_actions,
                        rewards=joint_rewards,
                        next_graph_state=next_graph_state,
                        old_log_probs=old_log_probs,
                        done=step_episode_done,
                    )
                    episode_graph_transition_count += 1.0
                elif uses_rollout_buffer:
                    if uses_shared_mappo:
                        old_log_probs = list(agents[0].last_joint_log_probs)
                    else:
                        old_log_probs = [
                            float(getattr(agent, "last_action_log_prob", 0.0))
                            for agent in agents
                        ]
                    rollout_buffer.push(
                        current_joint_state,
                        joint_actions,
                        joint_rewards,
                        next_joint_state,
                        old_log_probs,
                        done=step_episode_done,
                    )
                elif uses_gatma and slot_done and not step_episode_done:
                    pending_gatma_transition = (
                        current_joint_state, joint_actions, joint_rewards
                    )
                else:
                    replay_buffer.push(
                        current_joint_state,
                        joint_actions,
                        joint_rewards,
                        next_joint_state,
                        done=step_episode_done,
                    )
                episode_rollout_storage_time += (
                    time.perf_counter() - rollout_storage_start
                )
                current_joint_state = next_joint_state

            total_slot_actions = slot_local + slot_edge
            if total_slot_actions == 0:
                history["local_ratio"].append(0.0)
                history["edge_ratio"].append(0.0)
            else:
                history["local_ratio"].append(slot_local / total_slot_actions * 100)
                history["edge_ratio"].append(slot_edge / total_slot_actions * 100)

            current_delay_mean = np.mean(list(env.device_accumulated_delay.values()))
            current_energy_mean = np.mean(list(env.device_accumulated_energy.values()))
            slot_rewards.append(slot_reward / max(slot_steps, 1))
            slot_delays.append(max(0.0, current_delay_mean - prev_delay_mean))
            slot_energies.append(max(0.0, current_energy_mean - prev_energy_mean))

        episode_avg_reward = float(np.mean(slot_rewards)) if slot_rewards else 0.0
        episode_avg_delay = float(np.mean(slot_delays)) if slot_delays else 0.0
        episode_avg_energy = float(np.mean(slot_energies)) if slot_energies else 0.0
        device_count = max(len(devices), 1)
        episode_local_time = (
            float(np.mean(slot_local_times)) / device_count
            if slot_local_times
            else 0.0
        )
        episode_server_time = (
            float(np.mean(slot_server_times)) / device_count
            if slot_server_times
            else 0.0
        )
        episode_attempted_server_time = (
            float(np.mean(slot_attempted_server_times)) / device_count
            if slot_attempted_server_times
            else 0.0
        )
        episode_transfer_time = (
            float(np.mean(slot_transfer_times)) / device_count
            if slot_transfer_times
            else 0.0
        )
        episode_queue_or_wait_time = (
            float(np.mean(slot_queue_or_wait_times)) / device_count
            if slot_queue_or_wait_times
            else 0.0
        )
        episode_penalty_time = (
            float(np.mean(slot_penalty_times)) / device_count
            if slot_penalty_times
            else 0.0
        )
        episode_penalty_count = float(np.sum(slot_penalty_counts)) if slot_penalty_counts else 0.0
        episode_requested_local_count = (
            float(np.sum(slot_requested_local_counts)) if slot_requested_local_counts else 0.0
        )
        episode_requested_edge_count = (
            float(np.sum(slot_requested_edge_counts)) if slot_requested_edge_counts else 0.0
        )
        episode_resolved_local_count = (
            float(np.sum(slot_resolved_local_counts)) if slot_resolved_local_counts else 0.0
        )
        episode_resolved_edge_count = (
            float(np.sum(slot_resolved_edge_counts)) if slot_resolved_edge_counts else 0.0
        )
        total_actions = count_local + count_edge
        local_ratio = (count_local / total_actions * 100.0) if total_actions > 0 else 0.0
        edge_ratio = (count_edge / total_actions * 100.0) if total_actions > 0 else 0.0

        history["reward"].append(episode_avg_reward)
        history["delay"].append(episode_avg_delay)
        history["energy"].append(episode_avg_energy)
        history["local_ratio"].append(local_ratio)
        history["edge_ratio"].append(edge_ratio)
        history["local_time"].append(episode_local_time)
        history["server_time"].append(episode_server_time)
        history["attempted_server_time"].append(episode_attempted_server_time)
        history["transfer_time"].append(episode_transfer_time)
        history["queue_or_wait_time"].append(episode_queue_or_wait_time)
        history["penalty_time"].append(episode_penalty_time)
        history["penalty_count"].append(episode_penalty_count)
        history["requested_local_count"].append(episode_requested_local_count)
        history["requested_edge_count"].append(episode_requested_edge_count)
        history["resolved_local_count"].append(episode_resolved_local_count)
        history["resolved_edge_count"].append(episode_resolved_edge_count)

        if uses_graph_gat_mappo:
            episode_graph_update_time = _update_graph_gat_mappo_from_rollout(
                agents[0], graph_rollout_buffer, gamma
            )
            episode_model_update_time = episode_graph_update_time
        history["graph_build_time"].append(episode_graph_build_time)
        history["graph_warmup_time"].append(episode_graph_warmup_time)
        history["graph_warmup_loss"].append(
            episode_graph_warmup_loss_total / max(episode_graph_warmup_count, 1.0)
        )
        history["graph_warmup_count"].append(episode_graph_warmup_count)
        history["graph_action_time"].append(episode_graph_action_time)
        history["graph_update_time"].append(episode_graph_update_time)
        history["graph_transition_count"].append(episode_graph_transition_count)

        if show_progress:
            episode_iterator.set_postfix(
                reward=f"{episode_avg_reward:.3f}",
                delay=f"{episode_avg_delay:.3f}s",
                energy=f"{episode_avg_energy:.3f}J",
                actual=(
                    f"L{episode_resolved_local_count:.0f}/"
                    f"E{episode_resolved_edge_count:.0f}"
                ),
                penalty=f"{episode_penalty_count:.0f}",
            )

        # 4. Network Updates
        model_update_start = time.perf_counter()
        maddpg_update_metrics = {
            "update_rounds": 0.0,
            "actor_loss": 0.0,
            "critic_loss": 0.0,
            "mean_q": 0.0,
        }
        if uses_gatma:
            for _ in range(gatma_replay_updates_per_episode):
                round_metrics = update_gatma_agents_from_buffer(
                    agents, replay_buffer, batch_size, gamma
                )
                update_rounds = round_metrics["update_rounds"]
                if update_rounds == 0.0:
                    break
                maddpg_update_metrics["update_rounds"] += update_rounds
                for metric_name in ("actor_loss", "critic_loss", "mean_q"):
                    maddpg_update_metrics[metric_name] += (
                        update_rounds * round_metrics[metric_name]
                    )
            completed_updates = maddpg_update_metrics["update_rounds"]
            if completed_updates > 0.0:
                for metric_name in ("actor_loss", "critic_loss", "mean_q"):
                    maddpg_update_metrics[metric_name] /= completed_updates
            episode_model_update_time = time.perf_counter() - model_update_start
        elif uses_shared_mappo:
            agents[0].update_from_rollout(rollout_buffer)
        elif uses_rollout_buffer:
            _update_agents_from_rollout(agents, rollout_buffer, gamma)
        elif not uses_graph_gat_mappo:
            # Off-policy agents run `replay_updates_per_episode` gradient
            # rounds so their update budget can be matched to the on-policy
            # PPO family instead of being fixed at one round per episode.
            for _ in range(max(1, int(replay_updates_per_episode))):
                round_metrics = _update_agents_from_buffer(
                    agents, replay_buffer, batch_size, gamma
                )
                if round_metrics["update_rounds"] == 0.0:
                    break
                maddpg_update_metrics = {
                    "update_rounds": (
                        maddpg_update_metrics["update_rounds"]
                        + round_metrics["update_rounds"]
                    ),
                    "actor_loss": round_metrics["actor_loss"],
                    "critic_loss": round_metrics["critic_loss"],
                    "mean_q": round_metrics["mean_q"],
                }
        if not uses_graph_gat_mappo and not uses_gatma:
            episode_model_update_time = time.perf_counter() - model_update_start
        history["replay_buffer_size"].append(float(len(replay_buffer)))
        history["maddpg_update_rounds"].append(
            maddpg_update_metrics["update_rounds"]
        )
        history["maddpg_actor_loss"].append(
            maddpg_update_metrics["actor_loss"]
        )
        history["maddpg_critic_loss"].append(
            maddpg_update_metrics["critic_loss"]
        )
        history["maddpg_mean_q"].append(maddpg_update_metrics["mean_q"])
        history["epsilon"].append(
            float(getattr(agents[0], "epsilon", 0.0))
        )

        environment_runtime = env.get_runtime_metrics()
        episode_accounted_time = sum(
            [
                episode_dag_generation_time,
                episode_priority_inference_time,
                episode_start_slot_time,
                episode_action_collection_time,
                episode_env_step_time,
                episode_metric_summary_time,
                episode_rollout_storage_time,
                episode_model_update_time,
            ]
        )
        episode_elapsed_before_tracking = time.perf_counter() - episode_start_time
        episode_unaccounted_time = max(
            0.0, episode_elapsed_before_tracking - episode_accounted_time
        )
        history["runtime_dag_generation_time"].append(episode_dag_generation_time)
        history["runtime_priority_inference_time"].append(
            episode_priority_inference_time
        )
        history["runtime_start_slot_time"].append(episode_start_slot_time)
        history["runtime_action_collection_time"].append(
            episode_action_collection_time
        )
        history["runtime_env_step_time"].append(episode_env_step_time)
        history["runtime_metric_summary_time"].append(episode_metric_summary_time)
        history["runtime_rollout_storage_time"].append(
            episode_rollout_storage_time
        )
        history["runtime_model_update_time"].append(episode_model_update_time)
        history["runtime_connection_window_time"].append(
            environment_runtime["connection_window_seconds"]
        )
        history["runtime_connection_window_requests"].append(
            environment_runtime["connection_window_requests"]
        )
        history["runtime_connection_window_updates"].append(
            environment_runtime["connection_window_updates"]
        )
        history["runtime_connection_window_samples"].append(
            environment_runtime["connection_window_samples"]
        )
        history["runtime_joint_state_time"].append(
            environment_runtime["joint_state_seconds"]
        )
        history["runtime_accounted_time"].append(episode_accounted_time)
        history["runtime_unaccounted_time"].append(episode_unaccounted_time)
        history["runtime_tracking_log_time"].append(0.0)

        episode_number = episode + 1
        if show_progress and _should_print_diagnostics(
            episode_number, num_episodes
        ):
            episode_iterator.write(
                _format_diagnostic_summary(algo_name, episode_number, history)
            )

        if experiment_tracker is not None:
            total_elapsed_seconds = time.perf_counter() - training_start_time
            tracking_log_start = time.perf_counter()
            experiment_tracker.log(
                _build_episode_tracking_metrics(
                    history=history,
                    episode_number=episode_number,
                    num_episodes=num_episodes,
                    episode_elapsed_seconds=time.perf_counter() - episode_start_time,
                    total_elapsed_seconds=total_elapsed_seconds,
                    graph_gat_agent=agents[0] if uses_graph_gat_mappo else None,
                ),
                step=episode_number,
            )
            history["runtime_tracking_log_time"][-1] = (
                time.perf_counter() - tracking_log_start
            )
    
        if show_progress and (
            episode_number == 1
            or _should_print_diagnostics(episode_number, num_episodes)
        ):
            print(
                f"[{algo_name}] Ep {episode_number}/{num_episodes} ----|--- Avg R/Slot: {history['reward'][-1]:.3f} "
                f"---|--- D: {history['delay'][-1]:.3f}s ---|--- E: {history['energy'][-1]:.3f}J"
            )
            print(
                f"[{algo_name}] Local ({count_local} / {history['local_ratio'][-1]:.1f}%) - "
                f"Edge ({count_edge} / {history['edge_ratio'][-1]:.1f}%)"
            )
            print(
                f"[{algo_name}] Timing | Local: {history['local_time'][-1]:.3f}s "
                f"| Server: {history['server_time'][-1]:.3f}s "
                f"| Transfer: {history['transfer_time'][-1]:.3f}s "
                f"| Wait: {history['queue_or_wait_time'][-1]:.3f}s "
                f"| Requested L/E: {history['requested_local_count'][-1]:.0f}/"
                f"{history['requested_edge_count'][-1]:.0f} "
                f"| Resolved L/E: {history['resolved_local_count'][-1]:.0f}/"
                f"{history['resolved_edge_count'][-1]:.0f} "
                f"| Penalty: {history['penalty_count'][-1]:.0f} / "
                f"{history['penalty_time'][-1]:.3f}s"
            )
            if uses_graph_gat_mappo:
                print(
                    f"[{algo_name}] Graph-GAT Cost | Build: "
                    f"{history['graph_build_time'][-1]:.3f}s "
                    f"| Warmup: {history['graph_warmup_time'][-1]:.3f}s "
                    f"/ {history['graph_warmup_count'][-1]:.0f} "
                    f"| Warmup Loss: {history['graph_warmup_loss'][-1]:.4f} "
                    f"| Action: {history['graph_action_time'][-1]:.3f}s "
                    f"| Update: {history['graph_update_time'][-1]:.3f}s "
                    f"| Transitions: {history['graph_transition_count'][-1]:.0f}"
                )

        if episode_callback is not None:
            episode_callback(episode_number, history)
            
    checkpoint = build_model_checkpoint(
        algo_name=algo_name,
        agents=agents,
        agent_config=agent_config,
        history=history,
        episode_count=num_episodes,
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        num_devices=len(devices),
        num_servers=len(servers),
        graph_node_feature_dim=graph_node_feature_dim if uses_graph_gat_mappo else None,
        graph_edge_feature_dim=graph_edge_feature_dim if uses_graph_gat_mappo else None,
        topology_metrics=topology_metrics,
        experiment_note=experiment_note,
        experiment_seed=experiment_seed,
        reward_weights={
            key: float(provisional[key])
            for key in ("lambda1", "lambda2", "lambda3", "lambda4", "lambda5", "p_out_value")
        },
    )
    if uses_gatma and checkpoint is not None:
        checkpoint["adaptation"] = {
            "name": "GATMA-Adapted",
            "version": 2,
            "paper_doi": "10.1109/TCCN.2026.3683874",
            "agent_role": "device",
            "graph": "connected_device_server_with_self_edges",
            "critic_readout": "mean_pool",
            "actor_gradient": "straight_through_argmax_other_replay_actions",
            "action_mask": False,
            "replay_updates_per_episode": gatma_replay_updates_per_episode,
            "batch_size": batch_size,
            "replay_buffer_capacity": replay_buffer.buffer.maxlen,
            "device": str(agents[0].device),
            "time_slots": time_slots,
            "priority_order": list(fixed_priorities[devices[0].id]),
        }
    return history, checkpoint


def evaluate_algorithm_checkpoint(
    agent_config: Dict[str, object],
    checkpoint: Dict[str, Any],
    devices: Sequence[IndustrialDevice],
    servers: Sequence[EdgeServer],
    network_env: NetworkEnvironment,
    data_loader: KolektorSDDLoader,
    priority_model: torch.nn.Module,
    num_episodes: int,
    experiment_seed: int,
    priority_mode: str = "gat",
    topology_scenario: Optional[TopologyScenario] = None,
    fixed_priority_order: Optional[List[int]] = None,
) -> Dict[str, List[float]]:
    """Evaluate a trained policy with deterministic actions.

    Evaluation never performs optimizer updates, topology warmup, epsilon
    exploration, or stochastic policy sampling.
    """
    confirmed = PAPER_PARAMS["confirmed"]
    provisional = PAPER_PARAMS["provisional_table2_needed"]
    set_seed(experiment_seed)
    data_loader.reseed(experiment_seed)
    env = DITENEnv(
        devices,
        servers,
        network_env,
        slot_duration=confirmed["slot_duration_s"],
        subslot_count=int(provisional["subslot_count"]),
        time_slots=int(confirmed["time_slots"]),
        lambda1=provisional["lambda1"],
        lambda2=provisional["lambda2"],
        lambda3=provisional["lambda3"],
        lambda4=provisional["lambda4"],
        lambda5=provisional["lambda5"],
        p_out_value=provisional["p_out_value"],
        local_estimation_error=provisional["local_estimation_error"],
        edge_estimation_error=provisional["edge_estimation_error"],
        route_rectangles=(
            topology_scenario.route_rectangles
            if topology_scenario is not None
            else None
        ),
        world_size=(
            topology_scenario.world_size
            if topology_scenario is not None
            else (100.0, 100.0)
        ),
    )
    state_dim = env.get_state_dim()
    action_dim = 1 + len(servers)
    agent_class = agent_config["class"]
    uses_graph_gat_mappo = agent_class is GraphGATMAPPOAgent
    uses_gatma = agent_class is GATMAAgent
    uses_shared_mappo = agent_class is SharedMAPPOAgent
    agents: List[object] = []
    if uses_graph_gat_mappo:
        graph_dims = checkpoint.get("graph_dims", {})
        agents.append(
            agent_class(
                num_devices=len(devices),
                num_servers=len(servers),
                node_feature_dim=int(graph_dims["node_feature_dim"]),
                edge_feature_dim=int(graph_dims["edge_feature_dim"]),
                **agent_config.get("kwargs", {}),
            )
        )
    elif uses_shared_mappo:
        agents.append(
            agent_class(
                state_dim=state_dim,
                action_dim=action_dim,
                num_agents=len(devices),
                **agent_config.get("kwargs", {}),
            )
        )
    else:
        for agent_index, _ in enumerate(devices):
            extra_agent_kwargs = {}
            if uses_gatma:
                extra_agent_kwargs = {
                    "num_servers": len(servers),
                    "agent_index": agent_index,
                }
            agents.append(
                agent_class(
                    state_dim=state_dim,
                    action_dim=action_dim,
                    num_agents=len(devices),
                    **extra_agent_kwargs,
                    **agent_config.get("kwargs", {}),
                )
            )

    checkpoint_agents = checkpoint.get("agents", [])
    if len(checkpoint_agents) != len(agents):
        raise ValueError("checkpoint agent count does not match evaluation system")
    for agent, agent_state in zip(agents, checkpoint_agents):
        for module_name in (
            "encoder",
            "actor",
            "critic",
            "target_actor",
            "target_critic",
        ):
            module_state = agent_state.get(module_name)
            module = getattr(agent, module_name, None)
            if module_state is not None and module is not None:
                module.load_state_dict(module_state)
                module.eval()

    history = {
        "reward": [],
        "delay": [],
        "energy": [],
        "edge_ratio": [],
        "resolved_edge_ratio": [],
        "penalty_count": [],
    }
    time_slots = int(confirmed["time_slots"])
    if fixed_priority_order is None:
        template_task_dags = generate_task_dags_for_episode(
            [devices[0]],
            data_loader,
            t_max=provisional["t_max"],
            e_max=provisional["e_max"],
            cpu_cycle_scale=provisional["task_cpu_cycle_scale"],
        )
        fixed_priorities = build_fixed_priorities_by_mode(
            template_task_dags[devices[0].id],
            [device.id for device in devices],
            priority_model,
            priority_mode,
        )
    else:
        fixed_priorities = broadcast_priority_order(
            [device.id for device in devices], fixed_priority_order
        )
    for _ in range(num_episodes):
        env.reset_episode()
        episode_done = False
        slot_rewards = []
        slot_delays = []
        slot_energies = []
        requested_edge_count = 0
        resolved_edge_count = 0.0
        total_action_count = 0
        penalty_count = 0.0
        for _ in range(time_slots):
            if episode_done:
                break
            previous_delay = np.mean(
                list(env.device_accumulated_delay.values())
            )
            previous_energy = np.mean(
                list(env.device_accumulated_energy.values())
            )
            task_dags = generate_task_dags_for_episode(
                devices,
                data_loader,
                t_max=provisional["t_max"],
                e_max=provisional["e_max"],
                cpu_cycle_scale=provisional["task_cpu_cycle_scale"],
            )
            current_joint_state = env.start_time_slot(task_dags, fixed_priorities)
            slot_done = False
            slot_reward = 0.0
            slot_step_count = 0
            while not slot_done and not episode_done:
                if uses_graph_gat_mappo:
                    graph_state = build_topology_graph_state(
                        torch.as_tensor(
                            current_joint_state, dtype=torch.float32
                        ),
                        num_devices=len(devices),
                        num_servers=len(servers),
                        lightweight=agents[0].lightweight_topology,
                    )
                    joint_actions = agents[0].select_greedy_actions(graph_state)
                elif uses_shared_mappo:
                    joint_actions = agents[0].select_greedy_joint_actions(
                        torch.as_tensor(
                            current_joint_state, dtype=torch.float32
                        )
                    )
                elif uses_gatma:
                    full_joint_state = torch.as_tensor(
                        current_joint_state, dtype=torch.float32,
                        device=agents[0].device,
                    )
                    joint_actions = [
                        agent.select_greedy_action(full_joint_state)
                        for agent in agents
                    ]
                else:
                    joint_actions = [
                        agent.select_greedy_action(
                            torch.as_tensor(
                                current_joint_state[agent_index],
                                dtype=torch.float32,
                            )
                        )
                        for agent_index, agent in enumerate(agents)
                    ]
                requested_edge_count += sum(
                    int(action > 0) for action in joint_actions
                )
                total_action_count += len(joint_actions)
                next_joint_state, rewards, episode_done, info = env.step(
                    joint_actions
                )
                slot_done = bool(info.get("slot_done", False))
                summary = _summarize_step_metrics(env.last_step_metrics)
                resolved_edge_count += summary["resolved_edge_count"]
                penalty_count += summary["penalty_count"]
                slot_reward += sum(rewards) / len(devices)
                slot_step_count += 1
                current_joint_state = next_joint_state

            current_delay = np.mean(
                list(env.device_accumulated_delay.values())
            )
            current_energy = np.mean(
                list(env.device_accumulated_energy.values())
            )
            slot_rewards.append(slot_reward / max(slot_step_count, 1))
            slot_delays.append(max(0.0, current_delay - previous_delay))
            slot_energies.append(max(0.0, current_energy - previous_energy))

        history["reward"].append(float(np.mean(slot_rewards)))
        history["delay"].append(float(np.mean(slot_delays)))
        history["energy"].append(float(np.mean(slot_energies)))
        history["edge_ratio"].append(
            100.0 * requested_edge_count / max(total_action_count, 1)
        )
        history["resolved_edge_ratio"].append(
            100.0 * resolved_edge_count / max(total_action_count, 1)
        )
        history["penalty_count"].append(float(penalty_count))
    return history

if __name__ == "__main__":
    # 1. Base Environment Setup
    args = parse_args()
    confirmed = PAPER_PARAMS["confirmed"]
    provisional = PAPER_PARAMS["provisional_table2_needed"]
    reward_weights = {
        key: float(provisional[key])
        for key in ("lambda1", "lambda2", "lambda3", "lambda4", "lambda5", "p_out_value")
    }
    # experiment_seed = int(provisional["experiment_seed"])
    experiment_seed = (
        int(args.experiment_seed)
        if args.experiment_seed is not None
        else int(provisional["experiment_seed"])
    )
    # set_seed(experiment_seed)
    BANDWIDTH, NOISE_POWER = confirmed["bandwidth_hz"], confirmed["noise_power_dbm"]
    topology_scenario = get_topology_scenario(
        args.topology_scenario,
        topology_seed=args.topology_seed,
        server_profile=(
            None if args.server_profile == "scenario" else args.server_profile
        ),
    )
    topology_metrics = compute_topology_metrics(topology_scenario)
    flat_topology_metrics = flatten_topology_metrics(topology_metrics)

    print(
        "Topology scenario: "
        f"{topology_scenario.name} "
        f"({topology_scenario.device_count} devices / "
        f"{len(topology_scenario.server_locations)} servers, "
        f"world={topology_scenario.world_size[0]:g}x"
        f"{topology_scenario.world_size[1]:g}m, "
        f"profile={topology_scenario.server_profile}, "
        f"radius={min(topology_scenario.coverage_radii):.2f}-"
        f"{max(topology_scenario.coverage_radii):.2f}m)"
    )
    print(
        "Topology metrics: "
        f"avg_links={flat_topology_metrics['topology_avg_feasible_servers']:.2f}, "
        f"density={flat_topology_metrics['topology_density']:.2f}, "
        f"zero={flat_topology_metrics['topology_zero_link_ratio']:.2f}, "
        f"multi={flat_topology_metrics['topology_multi_link_ratio']:.2f}"
    )
    
    network_env = NetworkEnvironment(bandwidth=BANDWIDTH, noise_power_dbm=NOISE_POWER)
    data_loader = KolektorSDDLoader(
        dataset_path=args.dataset_path,
        allow_dummy_data=args.allow_dummy_data,
    )
    dataset_stats = data_loader.get_dataset_statistics()
    topology_metrics["dataset"] = dataset_stats
    print(
        f"Dataset mode: {dataset_stats['mode']} | "
        f"images={dataset_stats['total_images']} | "
        f"mean_pixels={dataset_stats['mean_pixels']} | "
        f"path={dataset_stats['dataset_path']}"
    )

    priority_model_name = str(provisional["priority_model"]).lower()
    priority_model = build_task_priority_model(
        priority_model_name,
        num_features=TASK_PRIORITY_FEATURE_DIM,
        hidden_dim=int(confirmed["gcn_hidden_dim"]),
    )
    priority_ckpt_path = get_priority_checkpoint_path(priority_model_name)
    sample_training_dag = make_priority_dag_sampler(
        data_loader,
        t_max=provisional["t_max"],
        e_max=provisional["e_max"],
        cpu_cycle_scale=provisional["task_cpu_cycle_scale"],
    )

    priority_template = sample_training_dag()
    if args.task_priority == "on":
        priority_model = load_or_train_priority_model(
            priority_model=priority_model,
            dag_sampler=sample_training_dag,
            checkpoint_path=priority_ckpt_path,
            epochs=int(provisional["gcn_pretrain_epochs"]),
            samples_per_epoch=int(provisional["gcn_samples_per_epoch"]),
            lr=confirmed["gcn_lr"],
            model_label=priority_model_name.upper(),
        )
        priority_inference_start = time.perf_counter()
        fixed_priority_order = build_priorities(
            {priority_template.id: priority_template}, priority_model
        )[priority_template.id]
        priority_inference_seconds = time.perf_counter() - priority_inference_start
        print(
            f"Task priority ON ({priority_model_name.upper()}): "
            f"{fixed_priority_order} "
            f"(one-time inference: {priority_inference_seconds:.6f}s)"
        )
    else:
        fixed_priority_order = sorted(priority_template.subtasks)
        print(f"Task priority OFF (default DAG order): {fixed_priority_order}")
    
    server_compute_seed = experiment_seed
    device_compute_seed = experiment_seed + 1
    np.random.seed(server_compute_seed)
    servers = build_servers_for_scenario(topology_scenario, confirmed, provisional)
    np.random.seed(device_compute_seed)
    devices = build_devices_for_scenario(topology_scenario, confirmed, provisional)
    topology_metrics["physical_compute"] = summarize_physical_compute(
        devices,
        servers,
        device_seed=device_compute_seed,
        server_seed=server_compute_seed,
    )
    compute_metrics = topology_metrics["physical_compute"]
    print(
        "Actual CPU: "
        f"devices={compute_metrics['actual_device_compute_power_mean_ghz']:.3f}GHz mean, "
        f"servers={compute_metrics['actual_server_compute_power_mean_ghz']:.3f}GHz mean"
    )

    # 2. Define Algorithms to Compare
    algorithms = select_algorithm_configs(
        build_algorithm_configs(
            args.graph_gat_device,
            gatma_device=args.gatma_device,
            use_gae=args.use_gae,
            num_minibatches=args.num_minibatches,
            maddpg_actor_replay_actions=args.maddpg_actor_replay_actions,
        ),
        args.algorithms,
    )

    # 3. Run Comparisons
    results = {"reward": {}, "delay": {}, "energy": {}}
    priority_results = {"reward": {}, "delay": {}, "energy": {}}
    FULL_EPISODES = (
        int(args.episodes)
        if args.episodes is not None
        else int(provisional["comparison_full_episodes"])
    )
    BASELINE_EVALUATION_EPISODES = int(provisional["baseline_evaluation_episodes"])
    algorithm_episode_counts = {}
    last_training_state_rows = []
    model_checkpoints = []
    experiment_note = args.note.strip()
    tracking_group = args.wandb_group.strip() or (
        f"{topology_scenario.name}-"
        f"{experiment_note or 'comparison'}-seed{experiment_seed}"
    )
    if experiment_note:
        print(f"Experiment note: {experiment_note}")
    print(f"Experiment seed: {experiment_seed}")

    # Fig.6-style priority extraction comparison under fixed e-ATN-MADDPG.
    priority_modes = {"Random Scheduling": "random", "Greedy Scheduling": "greedy", "GCN Scheduling": "gcn", "GAT Scheduling": "gat"}
    # priority_modes = {"GCN Scheduling": "gcn"}
    # e_atn_cfg = {
    #     "class": EpsilonATNMADDPGAgent,
    #     "kwargs": {
    #         "use_attention": True,
    #         "use_epsilon_greedy": True,
    #         "lr": confirmed["rl_lr"],
    #         "epsilon_init": provisional["epsilon_init"],
    #         "epsilon_min": provisional["epsilon_min"],
    #         "decay": provisional["epsilon_decay"],
    #     },
    # }
    # for label, mode in priority_modes.items():
    #     history = train_algorithm(
    #         f"e-ATN-MADDPG ({label})",
    #         e_atn_cfg,
    #         devices,
    #         servers,
    #         network_env,
    #         data_loader,
    #         priority_model=priority_model,
    #         num_episodes=FULL_EPISODES,
    #         priority_mode=mode,
    #     )
    #     priority_results["reward"][label] = history["reward"]
    #     priority_results["delay"][label] = history["delay"]
    #     priority_results["energy"][label] = history["energy"]

    for algo_name, config in algorithms.items():
        run_episodes = _episodes_for_algorithm(
            algo_name,
            full_episodes=FULL_EPISODES,
            baseline_episodes=BASELINE_EVALUATION_EPISODES,
        )
        algorithm_episode_counts[algo_name] = run_episodes
        experiment_tracker = initialize_experiment_tracker(
            mode=args.wandb_mode,
            project=args.wandb_project,
            entity=str(provisional["wandb_entity"]),
            run_name=(
                f"{algo_name} - {topology_scenario.name} - "
                f"priority-{args.task_priority}"
            ),
            group=tracking_group,
            notes=experiment_note,
            config={
                "algorithm": algo_name,
                "agent_class": config["class"].__name__,
                "agent_kwargs": dict(config.get("kwargs", {})),
                "reward_weights": reward_weights,
                "use_gae": args.use_gae,
                "num_minibatches": args.num_minibatches,
                "maddpg_updates_per_episode": args.maddpg_updates_per_episode,
                "maddpg_actor_replay_actions": args.maddpg_actor_replay_actions,
                "topology_scenario": topology_scenario.name,
                "topology_metrics": topology_metrics,
                "num_devices": len(devices),
                "num_servers": len(servers),
                "episodes": run_episodes,
                "seed": experiment_seed,
                "task_priority": args.task_priority,
                "priority_model": (
                    priority_model_name if args.task_priority == "on" else "none"
                ),
                "priority_order": fixed_priority_order,
            },
        )
        if experiment_tracker.url:
            print(f"[{algo_name}] W&B: {experiment_tracker.url}")
        try:
            history, checkpoint = train_algorithm(
                algo_name,
                config,
                devices,
                servers,
                network_env,
                data_loader,
                num_episodes=run_episodes,
                priority_model=priority_model,
                priority_mode=priority_model_name,
                topology_scenario=topology_scenario,
                topology_metrics=topology_metrics,
                experiment_note=experiment_note,
                experiment_tracker=experiment_tracker,
                fixed_priority_order=fixed_priority_order,
                experiment_seed=experiment_seed,
                replay_updates_per_episode=args.maddpg_updates_per_episode,
            )
        finally:
            experiment_tracker.finish()
        results["reward"][algo_name] = history["reward"]
        results["delay"][algo_name] = history["delay"]
        results["energy"][algo_name] = history["energy"]
        last_training_state_rows.append(
            build_last_training_state_line(
                algo_name,
                history,
                episode_count=run_episodes,
                topology_metrics=topology_metrics,
                experiment_note=experiment_note,
                experiment_seed=experiment_seed,
                agent_kwargs=config.get("kwargs", {}),
                reward_weights=reward_weights,
                maddpg_updates_per_episode=args.maddpg_updates_per_episode,
                task_priority=args.task_priority,
            )
        )
        if checkpoint is not None:
            checkpoint["task_priority"] = args.task_priority
            checkpoint["maddpg_updates_per_episode"] = args.maddpg_updates_per_episode
            model_checkpoints.append(checkpoint)

    # 4. Plotting Results
    print("\nAll training complete! Generating comparison plots...")
    output_paths = save_comparison_outputs(
        raw_results=results,
        full_episodes=FULL_EPISODES,
        last_training_state_rows=last_training_state_rows,
        model_checkpoints=model_checkpoints,
        fixed_baseline_algorithms=FIXED_BASELINE_ALGORITHMS,
        experiment_note=experiment_note,
        topology_scenario=topology_scenario.name,
        output_root=args.local_output_root,
    )
    last_state_path = output_paths["last_state_path"]
    checkpoint_paths = output_paths["checkpoint_paths"]
    
    print(f"Plots and last training states saved successfully! JSONL: {last_state_path}")
    print(f"Per-episode history CSV: {output_paths['episode_history_path']}")
    if checkpoint_paths:
        print("Model checkpoints saved:")
        for checkpoint_path in checkpoint_paths:
            print(f"  {checkpoint_path}")
    if args.drive_artifact_root.strip():
        synced_run = sync_completed_run(
            output_paths["output_dir"],
            args.drive_artifact_root,
        )
        print(f"Completed local run synced to Drive: {synced_run}")
    else:
        print(
            "Drive sync skipped: pass --drive-artifact-root or set "
            "TASK_OFFLOADING_DRIVE_ROOT."
        )
