"""Deterministic evaluation of saved comparison-agent checkpoints."""

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

from baselines.gatma import GATMAAgent
from baselines.graph_gat_mappo import GraphGATMAPPOAgent
from baselines.shared_mappo import SharedMAPPOAgent
from dataset.data_loader import KolektorSDDLoader
from environment.diten_env import DITENEnv
from environment.network_env import NetworkEnvironment
from environment.system_model import EdgeServer, IndustrialDevice
from utils.comparison_diagnostics import _summarize_step_metrics
from utils.comparison_setup import build_fixed_priorities_by_mode, set_seed
from utils.experiment_setup import (
    broadcast_priority_order,
    generate_task_dags_for_episode,
)
from utils.paper_config import PAPER_PARAMS
from utils.topology_graph_state import build_topology_graph_state
from utils.topology_scenarios_config import TopologyScenario


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
