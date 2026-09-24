"""Output helpers for comparison training runs."""

from __future__ import annotations

import csv
import json
import os
import time
from typing import Any, Dict, List, Optional, Sequence

import torch

from utils.plotter2 import DITENPlotter2


def flatten_topology_metrics(topology_metrics: Dict[str, object]) -> Dict[str, object]:
    """Flatten topology metrics for JSONL/checkpoint metadata."""
    route_samples = topology_metrics["route_samples"]
    flattened = {
        "topology_scenario": topology_metrics["name"],
        "topology_num_devices": topology_metrics["num_devices"],
        "topology_num_servers": topology_metrics["num_servers"],
        "topology_coverage_radius_m": topology_metrics["coverage_radius_m"],
        "topology_coverage_radius_min_m": topology_metrics[
            "coverage_radius_min_m"
        ],
        "topology_coverage_radius_max_m": topology_metrics[
            "coverage_radius_max_m"
        ],
        "topology_world_width_m": topology_metrics["world_size_m"][0],
        "topology_world_height_m": topology_metrics["world_size_m"][1],
        "topology_seed": topology_metrics["topology_seed"],
        "topology_server_profile": topology_metrics["server_profile"],
        "topology_avg_feasible_servers": route_samples["avg_feasible_servers"],
        "topology_density": route_samples["density"],
        "topology_zero_link_ratio": route_samples["zero_link_ratio"],
        "topology_multi_link_ratio": route_samples["multi_link_ratio"],
        "topology_device_server_ratio": route_samples["device_server_ratio"],
    }
    physical_compute = topology_metrics.get("physical_compute")
    if physical_compute is not None:
        flattened.update(physical_compute)
    hyperparameter_profile = topology_metrics.get("hyperparameter_profile")
    if hyperparameter_profile is not None:
        flattened.update(
            {
                "hyperparameter_profile": hyperparameter_profile["name"],
                "hyperparameter_profile_path": hyperparameter_profile["path"],
                "hyperparameter_effective_failed_offload_penalty": (
                    hyperparameter_profile["effective_failed_offload_penalty"]
                ),
            }
        )
    dataset = topology_metrics.get("dataset")
    if dataset is not None:
        flattened.update(
            {
                "dataset_path": dataset["dataset_path"],
                "dataset_mode": dataset["mode"],
                "dataset_total_images": dataset["total_images"],
                "dataset_mean_pixels": dataset["mean_pixels"],
            }
        )
    return flattened


def build_plot_results(
    raw_results: Dict[str, Dict[str, List[float]]],
    target_episodes: int,
    fixed_baseline_algorithms: frozenset[str],
) -> Dict[str, Dict[str, List[float]]]:
    """Build plot histories, extending fixed baselines as mean-flat lines."""
    plot_results: Dict[str, Dict[str, List[float]]] = {}
    for metric_name, algorithm_histories in raw_results.items():
        plot_results[metric_name] = {}
        for algo_name, history in algorithm_histories.items():
            if algo_name in fixed_baseline_algorithms:
                plot_results[metric_name][algo_name] = _mean_flat_history(
                    history, target_episodes
                )
            else:
                plot_results[metric_name][algo_name] = list(history)
    return plot_results


def build_plot_diagnostics(
    last_training_state_rows: Sequence[Dict[str, object]],
) -> tuple[List[str], List[List[str]]]:
    """Build a compact final-episode diagnostics table for output plots.

    Edge requested/executed percentages use all actions as the denominator.
    Rejection percentage uses requested edge actions as the denominator.
    """
    headers = [
        "Model",
        "Edge requested",
        "Edge executed",
        "Rejected edge",
        "Time breakdown",
    ]
    diagnostic_rows = []
    for row in last_training_state_rows:
        requested_local = int(row.get("requested_local_count", 0))
        requested_edge = int(row.get("requested_edge_count", 0))
        resolved_local = int(row.get("resolved_local_count", 0))
        resolved_edge = int(row.get("resolved_edge_count", 0))
        penalty_count = int(row.get("penalty_count", 0))
        requested_total = requested_local + requested_edge
        resolved_total = resolved_local + resolved_edge
        requested_edge_percent = 100.0 * requested_edge / max(requested_total, 1)
        resolved_edge_percent = 100.0 * resolved_edge / max(resolved_total, 1)
        rejection_percent = 100.0 * penalty_count / max(requested_edge, 1)
        diagnostic_rows.append(
            [
                str(row.get("model", "Unknown")),
                f"{requested_edge} ({requested_edge_percent:.1f}%)",
                f"{resolved_edge} ({resolved_edge_percent:.1f}%)",
                f"{penalty_count} ({rejection_percent:.1f}% req.)",
                (
                    f"{float(row.get('local_time_s', 0.0)):.3f} / "
                    f"{float(row.get('server_time_s', 0.0)):.3f} / "
                    f"{float(row.get('transfer_time_s', 0.0)):.3f} / "
                    f"{float(row.get('wait_time_s', 0.0)):.3f}"
                ),
            ]
        )
    return headers, diagnostic_rows


def build_last_training_state_line(
    algo_name: str,
    history: Dict[str, List[float]],
    episode_count: int,
    topology_metrics: Optional[Dict[str, object]] = None,
    experiment_note: str = "",
    experiment_seed: Optional[int] = None,
) -> Dict[str, object]:
    """Build one flat JSONL row for the final training state of one model."""

    def last_value(metric_name: str) -> float:
        values = history.get(metric_name, [])
        return float(values[-1]) if values else 0.0

    requested_local = int(round(last_value("requested_local_count")))
    requested_edge = int(round(last_value("requested_edge_count")))
    resolved_local = int(round(last_value("resolved_local_count")))
    resolved_edge = int(round(last_value("resolved_edge_count")))

    row = {
        "model": algo_name,
        "episode": int(episode_count),
        "reward": last_value("reward"),
        "delay_s": last_value("delay"),
        "energy_j": last_value("energy"),
        "local_count": requested_local,
        "edge_count": requested_edge,
        "local_ratio_percent": last_value("local_ratio"),
        "edge_ratio_percent": last_value("edge_ratio"),
        "requested_local_count": requested_local,
        "requested_edge_count": requested_edge,
        "resolved_local_count": resolved_local,
        "resolved_edge_count": resolved_edge,
        "local_time_s": last_value("local_time"),
        "server_time_s": last_value("server_time"),
        "transfer_time_s": last_value("transfer_time"),
        "wait_time_s": last_value("queue_or_wait_time"),
        "penalty_count": int(round(last_value("penalty_count"))),
        "penalty_time_s": last_value("penalty_time"),
        "graph_warmup_time_s": last_value("graph_warmup_time"),
        "graph_warmup_loss": last_value("graph_warmup_loss"),
        "graph_warmup_count": int(round(last_value("graph_warmup_count"))),
    }
    if experiment_note:
        row["experiment_note"] = experiment_note
    if experiment_seed is not None:
        row["experiment_seed"] = int(experiment_seed)
    if topology_metrics is not None:
        row.update(flatten_topology_metrics(topology_metrics))
    return row


def build_model_checkpoint(
    algo_name: str,
    agents: Sequence[object],
    agent_config: Dict[str, object],
    history: Dict[str, List[float]],
    episode_count: int,
    state_dim: int,
    action_dim: int,
    num_devices: int,
    num_servers: int,
    graph_node_feature_dim: Optional[int] = None,
    graph_edge_feature_dim: Optional[int] = None,
    topology_metrics: Optional[Dict[str, object]] = None,
    experiment_note: str = "",
    experiment_seed: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    """Build a serializable checkpoint payload for a trainable algorithm."""
    agent_states = [
        _agent_checkpoint_state(agent, agent_index)
        for agent_index, agent in enumerate(agents)
    ]
    trainable_agent_states = [
        agent_state
        for agent_state in agent_states
        if any(key in agent_state for key in ("encoder", "actor", "critic"))
    ]
    if not trainable_agent_states:
        return None

    agent_class = agent_config.get("class")
    checkpoint: Dict[str, Any] = {
        "model": algo_name,
        "episode_count": int(episode_count),
        "agent_class": getattr(agent_class, "__name__", str(agent_class)),
        "agent_kwargs": dict(agent_config.get("kwargs", {})),
        "state_dim": int(state_dim),
        "action_dim": int(action_dim),
        "num_devices": int(num_devices),
        "num_servers": int(num_servers),
        "agents": trainable_agent_states,
        "final_metrics": build_last_training_state_line(
            algo_name,
            history,
            episode_count,
            topology_metrics=topology_metrics,
            experiment_note=experiment_note,
            experiment_seed=experiment_seed,
        ),
    }
    if experiment_note:
        checkpoint["experiment_note"] = experiment_note
    if experiment_seed is not None:
        checkpoint["experiment_seed"] = int(experiment_seed)
    if topology_metrics is not None:
        checkpoint["topology_metrics"] = topology_metrics
    if graph_node_feature_dim is not None or graph_edge_feature_dim is not None:
        checkpoint["graph_dims"] = {
            "node_feature_dim": graph_node_feature_dim,
            "edge_feature_dim": graph_edge_feature_dim,
        }
    return checkpoint


def save_comparison_outputs(
    raw_results: Dict[str, Dict[str, List[float]]],
    full_episodes: int,
    last_training_state_rows: Sequence[Dict[str, object]],
    model_checkpoints: Sequence[Dict[str, Any]],
    fixed_baseline_algorithms: frozenset[str],
    experiment_note: str = "",
    topology_scenario: Optional[str] = None,
    output_root: str = "plots",
) -> Dict[str, object]:
    """Save plots, final JSONL rows, and model checkpoints."""
    date_string = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    safe_note = _safe_experiment_note(experiment_note)
    output_folder = f"{date_string}-{safe_note}" if safe_note else date_string
    plotter = DITENPlotter2(save_dir=os.path.join(output_root, output_folder))
    plot_results = build_plot_results(
        raw_results,
        target_episodes=full_episodes,
        fixed_baseline_algorithms=fixed_baseline_algorithms,
    )
    diagnostic_headers, diagnostic_rows = build_plot_diagnostics(
        last_training_state_rows
    )
    plot_paths = _save_training_plots(
        plotter,
        plot_results,
        date_string,
        topology_scenario,
        diagnostic_headers=diagnostic_headers,
        diagnostic_rows=diagnostic_rows,
    )
    last_state_path = os.path.join(
        plotter.save_dir, f"{date_string}_last_training_state.jsonl"
    )
    _write_last_training_state_jsonl(last_state_path, last_training_state_rows)
    episode_history_path = os.path.join(
        plotter.save_dir, f"{date_string}_episode_history.csv"
    )
    _write_episode_history_csv(
        episode_history_path, raw_results, last_training_state_rows
    )
    checkpoint_paths = [
        _save_model_checkpoint(plotter.save_dir, checkpoint)
        for checkpoint in model_checkpoints
    ]
    return {
        "output_dir": plotter.save_dir,
        "plot_paths": plot_paths,
        "last_state_path": last_state_path,
        "episode_history_path": episode_history_path,
        "checkpoint_paths": checkpoint_paths,
    }


def _write_episode_history_csv(
    output_path: str,
    raw_results: Dict[str, Dict[str, List[float]]],
    last_training_state_rows: Sequence[Dict[str, object]],
) -> None:
    """Write one row per model and episode with reward, delay, and energy.

    The file is the local equivalent of a W&B history export so that runs
    logged without tracking can still be aggregated across seeds.

    Args:
        output_path: Destination CSV path.
        raw_results: Metric name to model name to per-episode values.
        last_training_state_rows: Final-state rows used for run metadata.
    """
    metadata_by_model = {
        str(row["model"]): row for row in last_training_state_rows
    }
    reward_by_model = raw_results.get("reward", {})
    delay_by_model = raw_results.get("delay", {})
    energy_by_model = raw_results.get("energy", {})
    with open(output_path, "w", encoding="utf-8", newline="") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(
            [
                "model",
                "seed",
                "topology_scenario",
                "experiment_note",
                "episode",
                "reward",
                "delay_s",
                "energy_j",
            ]
        )
        for model_name, rewards in reward_by_model.items():
            metadata = metadata_by_model.get(model_name, {})
            delays = delay_by_model.get(model_name, [])
            energies = energy_by_model.get(model_name, [])
            for episode_index, reward in enumerate(rewards):
                writer.writerow(
                    [
                        model_name,
                        metadata.get("experiment_seed", ""),
                        metadata.get("topology_scenario", ""),
                        metadata.get("experiment_note", ""),
                        episode_index + 1,
                        reward,
                        delays[episode_index] if episode_index < len(delays) else "",
                        energies[episode_index]
                        if episode_index < len(energies)
                        else "",
                    ]
                )


def _mean_flat_history(history: Sequence[float], target_episodes: int) -> List[float]:
    """Convert a short fixed-baseline history into a mean-flat plot line."""
    if not history or target_episodes <= 0:
        return []
    mean_value = float(sum(history) / len(history))
    return [mean_value for _ in range(target_episodes)]


def _safe_experiment_note(note: str) -> str:
    """Return a filesystem-safe suffix for an experiment note."""
    safe_chars = []
    for char in note.strip().lower():
        if char.isascii() and char.isalnum():
            safe_chars.append(char)
        elif char in {"-", "_"}:
            safe_chars.append(char)
        elif char.isspace():
            safe_chars.append("_")
        else:
            safe_chars.append("_")
    safe_note = "".join(safe_chars)
    return "_".join(part for part in safe_note.split("_") if part).strip("-_")


def _write_last_training_state_jsonl(
    output_path: str, rows: Sequence[Dict[str, object]]
) -> None:
    """Write one final-state JSON object per model line."""
    with open(output_path, "w", encoding="utf-8") as output_file:
        for row in rows:
            output_file.write(json.dumps(row, ensure_ascii=True) + "\n")


def _agent_checkpoint_state(agent: object, agent_index: int) -> Dict[str, object]:
    """Build checkpoint state for one trainable agent."""
    agent_state: Dict[str, object] = {
        "agent_index": int(agent_index),
        "agent_class": agent.__class__.__name__,
    }
    for module_name in ("encoder", "actor", "critic", "target_actor", "target_critic"):
        module = getattr(agent, module_name, None)
        if module is not None and hasattr(module, "state_dict"):
            agent_state[module_name] = module.state_dict()

    optimizer_states = {}
    for optimizer_name in ("optimizer", "actor_optimizer", "critic_optimizer"):
        optimizer = getattr(agent, optimizer_name, None)
        if optimizer is not None and hasattr(optimizer, "state_dict"):
            optimizer_states[optimizer_name] = optimizer.state_dict()
    if optimizer_states:
        agent_state["optimizers"] = optimizer_states

    return agent_state


def _safe_checkpoint_name(algo_name: str) -> str:
    """Return a filesystem-safe checkpoint name for an algorithm."""
    safe_name = "".join(
        char if char.isalnum() or char in {"-", "_", "."} else "_"
        for char in algo_name
    ).strip("_")
    return safe_name or "model"


def _save_model_checkpoint(save_dir: str, checkpoint: Dict[str, Any]) -> str:
    """Save one model checkpoint into a plot output directory."""
    os.makedirs(save_dir, exist_ok=True)
    safe_name = _safe_checkpoint_name(str(checkpoint["model"]))
    checkpoint_path = os.path.join(save_dir, f"{safe_name}_checkpoint.pt")
    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path


def _save_training_plots(
    plotter: DITENPlotter2,
    plot_results: Dict[str, Dict[str, List[float]]],
    date_string: str,
    topology_scenario: Optional[str] = None,
    diagnostic_headers: Optional[Sequence[str]] = None,
    diagnostic_rows: Optional[Sequence[Sequence[str]]] = None,
) -> List[str]:
    """Save reward, delay, and energy training plots."""
    plot_specs = [
        (
            "reward",
            f"Performance Comparison in Reward ({topology_scenario})",
            "Reward",
        ),
        (
            "delay",
            f"Performance Comparison in Task Processing Delay ({topology_scenario})",
            "Task Processing Delay (s)",
        ),
        (
            "energy",
            f"Performance Comparison in Energy Consumption ({topology_scenario})",
            "Energy Consumption (J)",
        ),
    ]
    plot_paths = []
    for metric_name, title, ylabel in plot_specs:
        filename = (
            f"{date_string}_comparison_{metric_name}_{topology_scenario}.png"
        )
        plotter.plot_training_curve(
            data_dict=plot_results[metric_name],
            title=title,
            ylabel=ylabel,
            filename=filename,
            diagnostic_headers=diagnostic_headers,
            diagnostic_rows=diagnostic_rows,
        )
        plot_paths.append(os.path.join(plotter.save_dir, filename))
    return plot_paths
