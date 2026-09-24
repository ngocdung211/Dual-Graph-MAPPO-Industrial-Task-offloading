"""Agent configuration and comparison selection."""

from typing import Dict, Optional, Sequence

from baselines.gatma import GATMAAgent
from baselines.graph_gat_mappo import GraphGATMAPPOAgent
from baselines.mappo import MAPPOAgent
from baselines.offloading_baselines import (
    EdgeOnlyAgent,
    FeatureExtractionEdgeAgent,
    LocalOnlyAgent,
    RandomOffloadingAgent,
)
from baselines.shared_mappo import SharedMAPPOAgent
from models.maddpg import EpsilonATNMADDPGAgent
from utils.paper_config import PAPER_PARAMS

PPO_FAMILY_AGENT_CLASSES = (MAPPOAgent, SharedMAPPOAgent, GraphGATMAPPOAgent)
FIXED_BASELINE_ALGORITHMS = frozenset(
    {"Local Only", "Edge Only", "Feature Extraction Edge", "Random Offloading"}
)


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
