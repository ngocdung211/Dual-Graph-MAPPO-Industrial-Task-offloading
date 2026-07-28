"""Helper functions to build DAGs and task-priority graph models."""

from __future__ import annotations

from typing import Callable, Dict, List

import torch

from dataset.data_loader import KolektorSDDLoader
from environment.system_model import IndustrialDevice, Subtask, TaskDAG
from models.gcn import TaskPriorityGCN
from models.task_priority_gat import TaskPriorityGAT
from utils.graph_utils import extract_task_graph_inputs


DEFAULT_DAG_EDGES = [(1, 2), (1, 3), (2, 4), (3, 4), (4, 5)]


def build_task_priority_model(
    model_name: str, num_features: int, hidden_dim: int
) -> torch.nn.Module:
    """Build a task-priority model by name.

    Args:
        model_name: Priority model name, either "gcn" or "gat".
        num_features: Input node feature dimension.
        hidden_dim: Hidden layer width.

    Returns:
        Task priority model.

    Raises:
        ValueError: If model_name is not supported.
    """
    normalized_name = model_name.lower()
    if normalized_name == "gcn":
        return TaskPriorityGCN(num_features=num_features, hidden_dim=hidden_dim)
    if normalized_name == "gat":
        return TaskPriorityGAT(num_features=num_features, hidden_dim=hidden_dim)
    raise ValueError("priority model must be one of: gcn, gat")


def get_priority_checkpoint_path(model_name: str) -> str:
    """Return the checkpoint path for a priority model name."""
    normalized_name = model_name.lower()
    if normalized_name not in {"gcn", "gat"}:
        raise ValueError("priority model must be one of: gcn, gat")
    return f"models/checkpoints/{normalized_name}_priority.pt"


def build_task_dag(
    task_id: int,
    task_params: Dict[str, Dict[str, float]],
    t_max: float = 1.0,
    e_max: float = 1.0,
    cpu_cycle_scale: float = 1.0,
) -> TaskDAG:
    """Build a TaskDAG from task parameters.

    Args:
        task_id: Unique task identifier.
        task_params: Subtask parameter mapping.
        t_max: Maximum tolerable delay.
        e_max: Maximum tolerable energy.
        cpu_cycle_scale: Multiplier applied to every subtask's CPU demand.

    Returns:
        Constructed TaskDAG instance.
    """
    task_dag = TaskDAG(task_id=task_id, t_max=t_max, e_max=e_max)
    for subtask_id in range(1, 6):
        params = task_params[f"subtask_{subtask_id}"]
        task_dag.add_subtask(
            Subtask(
                subtask_id,
                params["cpu_cycles"] * cpu_cycle_scale,
                params["data_size"],
                params["result_size"],
            )
        )
    for pred, succ in DEFAULT_DAG_EDGES:
        task_dag.add_dependency(pred, succ)
    return task_dag


def generate_task_dags_for_episode(
    devices: List[IndustrialDevice],
    data_loader: KolektorSDDLoader,
    t_max: float = 1.0,
    e_max: float = 1.0,
    cpu_cycle_scale: float = 1.0,
) -> Dict[int, TaskDAG]:
    """Generate a TaskDAG per device for a single episode.

    Args:
        devices: Devices participating in the episode.
        data_loader: Dataset loader for random task parameters.
        t_max: Maximum tolerable delay.
        e_max: Maximum tolerable energy.
        cpu_cycle_scale: Multiplier applied to every subtask's CPU demand.

    Returns:
        Mapping of device IDs to TaskDAGs.
    """
    task_dags: Dict[int, TaskDAG] = {}
    for device in devices:
        task_params = data_loader.get_random_task_parameters()
        task_dags[device.id] = build_task_dag(
            device.id,
            task_params,
            t_max=t_max,
            e_max=e_max,
            cpu_cycle_scale=cpu_cycle_scale,
        )
    return task_dags


def build_priorities(
    task_dags: Dict[int, TaskDAG], priority_model: torch.nn.Module
) -> Dict[int, List[int]]:
    """Build execution priorities per device using a priority graph model.

    Args:
        task_dags: Task DAGs keyed by device ID.
        priority_model: Trained GCN or GAT priority model.

    Returns:
        Mapping of device IDs to priority-ordered subtask IDs.
    """
    priorities: Dict[int, List[int]] = {}
    for device_id, task_dag in task_dags.items():
        features, adjacency = extract_task_graph_inputs(task_dag)
        with torch.no_grad():
            scores = priority_model(features, adjacency)
        sorted_indices = torch.argsort(scores.squeeze(), descending=True).tolist()
        priorities[device_id] = [idx + 1 for idx in sorted_indices]
    return priorities


def make_priority_dag_sampler(
    data_loader: KolektorSDDLoader,
    t_max: float = 1.0,
    e_max: float = 1.0,
    cpu_cycle_scale: float = 1.0,
) -> Callable[[], TaskDAG]:
    """Create a callable that samples TaskDAGs for priority-model training.

    Args:
        data_loader: Dataset loader for random task parameters.
        t_max: Maximum tolerable delay.
        e_max: Maximum tolerable energy.
        cpu_cycle_scale: Multiplier applied to every subtask's CPU demand.

    Returns:
        Callable that returns a TaskDAG.
    """
    def _sampler() -> TaskDAG:
        task_params = data_loader.get_random_task_parameters()
        return build_task_dag(
            task_id=0,
            task_params=task_params,
            t_max=t_max,
            e_max=e_max,
            cpu_cycle_scale=cpu_cycle_scale,
        )

    return _sampler
