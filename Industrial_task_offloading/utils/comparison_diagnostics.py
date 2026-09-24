"""Compute and format comparison diagnostics."""

from typing import Dict, List, Sequence

import numpy as np

from environment.system_model import EdgeServer, IndustrialDevice
from utils.paper_config import PAPER_PARAMS


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
