"""Train or load graph models for task-priority extraction."""

from __future__ import annotations

import os
from typing import Callable, Dict, List

import torch
import torch.nn.functional as F

from environment.system_model import TaskDAG


def _minmax_normalize(values: List[float]) -> List[float]:
    """Normalize a list of values into [0, 1] with min-max scaling.

    Args:
        values: Input values.

    Returns:
        Normalized values in the range [0, 1].
    """
    v_min = min(values)
    v_max = max(values)
    if v_max - v_min < 1e-9:
        return [0.0 for _ in values]
    return [(v - v_min) / (v_max - v_min) for v in values]


def _compute_upward_cpu_rank(task_dag: TaskDAG) -> Dict[int, float]:
    """Compute a critical-path CPU rank for each subtask.

    Args:
        task_dag: Task DAG definition.

    Returns:
        Mapping of subtask IDs to upward CPU ranks.
    """
    succs: Dict[int, List[int]] = {sid: [] for sid in task_dag.subtasks.keys()}
    for pred, succ in task_dag.edges:
        succs[pred].append(succ)

    memo: Dict[int, float] = {}

    def dfs(node_id: int) -> float:
        if node_id in memo:
            return memo[node_id]
        successor_rank = max(
            (dfs(next_id) for next_id in succs[node_id]),
            default=0.0,
        )
        rank = float(task_dag.subtasks[node_id].cpu_cycles) + successor_rank
        memo[node_id] = rank
        return rank

    for sid in task_dag.subtasks.keys():
        dfs(sid)
    return memo


def build_task_priority_targets(task_dag: TaskDAG) -> torch.Tensor:
    """Build supervised targets for task-priority model pretraining.

    Args:
        task_dag: Task DAG definition.

    Returns:
        Tensor of priority targets shaped (num_nodes, 1).
    """
    subtask_ids = sorted(task_dag.subtasks.keys())
    upward_rank = _compute_upward_cpu_rank(task_dag)
    targets = _minmax_normalize(
        [float(upward_rank[subtask_id]) for subtask_id in subtask_ids]
    )
    return torch.tensor(targets, dtype=torch.float32).unsqueeze(1)


def load_or_train_priority_model(
    priority_model: torch.nn.Module,
    dag_sampler: Callable[[], TaskDAG],
    checkpoint_path: str,
    epochs: int = 200,
    samples_per_epoch: int = 32,
    lr: float = 1e-3,
    model_label: str = "priority",
) -> torch.nn.Module:
    """Load a pretrained priority model or train one if missing.

    Args:
        priority_model: Priority model instance.
        dag_sampler: Callable that returns TaskDAG samples.
        checkpoint_path: Path to the model checkpoint.
        epochs: Number of training epochs.
        samples_per_epoch: DAG samples per epoch.
        lr: Learning rate.
        model_label: Human-readable model label for logs.

    Returns:
        Trained or loaded priority model.
    """
    if os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        priority_model.load_state_dict(state_dict)
        priority_model.eval()
        print(f"Loaded pretrained {model_label} weights from: {checkpoint_path}")
        return priority_model

    print(f"No pretrained {model_label} weights found. Training priority model...")
    optimizer = torch.optim.Adam(priority_model.parameters(), lr=lr)
    priority_model.train()

    for epoch in range(epochs):
        epoch_loss = 0.0
        for _ in range(samples_per_epoch):
            task_dag = dag_sampler()
            from utils.graph_utils import extract_task_graph_inputs

            features, adjacency = extract_task_graph_inputs(task_dag)
            y = build_task_priority_targets(task_dag)
            pred = priority_model(features, adjacency)
            loss = F.mse_loss(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())

        if (epoch + 1) % 50 == 0:
            avg_loss = epoch_loss / float(samples_per_epoch)
            print(
                f"{model_label} pretrain epoch {epoch + 1}/{epochs} "
                f"- loss: {avg_loss:.6f}"
            )

    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    torch.save(priority_model.state_dict(), checkpoint_path)
    print(f"Saved {model_label} weights to: {checkpoint_path}")
    priority_model.eval()
    return priority_model
