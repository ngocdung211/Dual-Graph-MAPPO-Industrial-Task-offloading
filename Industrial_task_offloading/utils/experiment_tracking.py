"""Optional experiment tracking for comparison training runs."""

from typing import Dict, List, Optional

import torch

from baselines.graph_gat_mappo import GraphGATMAPPOAgent


class ExperimentTracker:
    """Small wrapper around an optional W&B run."""

    def __init__(self, run: Optional[object] = None):
        """Initialize the tracker with an optional active run."""
        self._run = run

    @property
    def enabled(self) -> bool:
        """Return whether an external tracking run is active."""
        return self._run is not None

    @property
    def url(self) -> Optional[str]:
        """Return the run URL when the backend provides one."""
        if self._run is None:
            return None
        return getattr(self._run, "url", None)

    def log(self, metrics: Dict[str, float], step: int) -> None:
        """Log one complete episode metric record."""
        if self._run is not None:
            self._run.log(metrics, step=step)

    def finish(self) -> None:
        """Flush and close the active tracking run."""
        if self._run is not None:
            self._run.finish()
            self._run = None


def initialize_experiment_tracker(
    mode: str,
    project: str,
    run_name: str,
    group: str,
    config: Dict[str, object],
    entity: str = "",
    notes: str = "",
) -> ExperimentTracker:
    """Initialize optional W&B tracking without importing it when disabled.

    Args:
        mode: W&B mode: ``disabled``, ``online``, or ``offline``.
        project: W&B project name.
        run_name: Human-readable algorithm run name.
        group: Comparison group shared by related algorithm runs.
        config: Serializable hyperparameters and experiment metadata.
        entity: Optional W&B user or team entity.
        notes: Optional experiment note.

    Returns:
        Experiment tracker containing an active W&B run, or a disabled tracker.

    Raises:
        RuntimeError: If tracking is enabled but W&B is unavailable or cannot
            initialize.
        ValueError: If the requested mode is unsupported.
    """
    normalized_mode = mode.strip().lower()
    if normalized_mode == "disabled":
        return ExperimentTracker()
    if normalized_mode not in {"online", "offline"}:
        raise ValueError("tracking mode must be disabled, online, or offline")

    try:
        import wandb
    except ImportError as error:
        raise RuntimeError(
            "W&B tracking is enabled but the 'wandb' package is not installed. "
            "Install requirements-wandb.txt or use --wandb-mode disabled."
        ) from error

    try:
        run = wandb.init(
            project=project,
            entity=entity or None,
            name=run_name,
            group=group or None,
            notes=notes or None,
            config=config,
            mode=normalized_mode,
            force=normalized_mode == "online",
            save_code=False,
        )
    except Exception as error:
        raise RuntimeError(f"Failed to initialize W&B tracking: {error}") from error
    return ExperimentTracker(run)


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
