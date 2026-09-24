"""Seed and construct comparison environments and task priorities."""

import random
from typing import Dict, List

import numpy as np
import torch

from baselines.scheduling_baselines import BaselineSchedulers
from environment.system_model import EdgeServer, IndustrialDevice, TaskDAG
from utils.task_priority.experiment_setup import broadcast_priority_order, build_priorities
from utils.topology.scenarios import TopologyScenario, device_start_points


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducible runs.

    Args:
        seed: Random seed value.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)


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
