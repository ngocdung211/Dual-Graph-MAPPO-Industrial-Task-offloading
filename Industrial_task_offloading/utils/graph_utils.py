"""Graph utilities for TaskDAG feature extraction."""

from collections import deque
from typing import Dict, List, Tuple

import numpy as np
import torch

from environment.system_model import TaskDAG


def _build_successors(task_dag: TaskDAG) -> Dict[int, List[int]]:
    """Build successor adjacency lists for each subtask.

    Args:
        task_dag: Task DAG definition.

    Returns:
        Mapping of subtask IDs to successor ID lists.
    """
    successors: Dict[int, List[int]] = {sid: [] for sid in task_dag.subtasks.keys()}
    for pred, succ in task_dag.edges:
        successors[pred].append(succ)
    return successors


def _compute_hierarchy_levels(task_dag: TaskDAG) -> Dict[int, int]:
    """Compute hierarchy levels for each subtask in the DAG.

    Args:
        task_dag: Task DAG definition.

    Returns:
        Mapping of subtask IDs to hierarchy levels.
    """
    indegree = {sid: 0 for sid in task_dag.subtasks.keys()}
    successors = _build_successors(task_dag)
    for _, succ in task_dag.edges:
        indegree[succ] += 1

    queue = deque([sid for sid, deg in indegree.items() if deg == 0])
    levels: Dict[int, int] = {sid: 1 for sid in queue}
    while queue:
        cur = queue.popleft()
        for nxt in successors[cur]:
            levels[nxt] = max(levels.get(nxt, 1), levels[cur] + 1)
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                queue.append(nxt)
    for sid in task_dag.subtasks.keys():
        levels.setdefault(sid, 1)
    return levels


def _compute_cumulative_successor_cpu(task_dag: TaskDAG) -> Dict[int, float]:
    """Compute cumulative successor CPU cycles for each subtask.

    Args:
        task_dag: Task DAG definition.

    Returns:
        Mapping of subtask IDs to cumulative successor CPU cycles.
    """
    successors = _build_successors(task_dag)
    memo: Dict[int, float] = {}

    def dfs(subtask_id: int) -> float:
        if subtask_id in memo:
            return memo[subtask_id]
        total = 0.0
        for succ in successors[subtask_id]:
            total += float(task_dag.subtasks[succ].cpu_cycles)
            total += dfs(succ)
        memo[subtask_id] = total
        return total

    for sid in task_dag.subtasks.keys():
        dfs(sid)
    return memo


def _minmax_normalize_column(values: np.ndarray) -> np.ndarray:
    """Normalize one task-feature column into the range [0, 1]."""
    value_min = float(np.min(values))
    value_max = float(np.max(values))
    if value_max - value_min < 1e-9:
        return np.zeros_like(values)
    return (values - value_min) / (value_max - value_min)


def extract_task_graph_inputs(task_dag: TaskDAG) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract adjacency and feature matrices for task-priority graph models.

    Args:
        task_dag: Task DAG definition.

    Returns:
        Tuple of (features, adjacency) tensors.
    """
    num_nodes = len(task_dag.subtasks)
    adjacency = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    features = np.zeros((num_nodes, 6), dtype=np.float32)

    # 1. Xây dựng ma trận kề A dựa trên dependencies (edges)
    for pred, succ in task_dag.edges:
        adjacency[pred - 1, succ - 1] = 1.0  # Giả sử id bắt đầu từ 1

    hierarchy_levels = _compute_hierarchy_levels(task_dag)
    cumulative_successor_cpu = _compute_cumulative_successor_cpu(task_dag)

    # 2. Xây dựng ma trận đặc trưng X
    for sub_id, subtask in task_dag.subtasks.items():
        idx = sub_id - 1
        # Out-degree: Số lượng subtask con phụ thuộc trực tiếp vào nó
        out_degree = sum(1 for p, s in task_dag.edges if p == sub_id)

        # Structural features followed by the current subtask workload features.
        features[idx, 0] = float(hierarchy_levels[sub_id])
        features[idx, 1] = float(out_degree)
        features[idx, 2] = cumulative_successor_cpu[sub_id]
        features[idx, 3] = float(subtask.cpu_cycles)
        features[idx, 4] = float(subtask.data_size)
        features[idx, 5] = float(subtask.result_size)

    for feature_index in range(features.shape[1]):
        features[:, feature_index] = _minmax_normalize_column(
            features[:, feature_index]
        )

    return torch.FloatTensor(features), torch.FloatTensor(adjacency)
