"""Tests for optional W&B experiment tracking."""

import pathlib
import sys
import types

import pytest

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from run_comparision import _build_episode_tracking_metrics
from utils.comparison.tracking import initialize_experiment_tracker


class _FakeRun:
    """Record W&B wrapper calls without network access."""

    url = "https://wandb.example/run/test"

    def __init__(self):
        self.logged = []
        self.finished = False

    def log(self, metrics, step):
        self.logged.append((metrics, step))

    def finish(self):
        self.finished = True


def _make_history():
    """Create the latest episode metric fixture."""
    return {
        "reward": [3.5],
        "delay": [0.4],
        "energy": [0.2],
        "local_ratio": [60.0],
        "edge_ratio": [40.0],
        "requested_local_count": [6.0],
        "requested_edge_count": [4.0],
        "resolved_local_count": [7.0],
        "resolved_edge_count": [3.0],
        "local_time": [0.1],
        "server_time": [0.2],
        "transfer_time": [0.03],
        "queue_or_wait_time": [0.04],
        "penalty_count": [1.0],
        "penalty_time": [0.05],
        "graph_build_time": [0.01],
        "graph_warmup_time": [0.0],
        "graph_warmup_loss": [0.0],
        "graph_action_time": [0.02],
        "graph_update_time": [0.06],
        "graph_transition_count": [10.0],
        "runtime_env_step_time": [0.3],
        "runtime_connection_window_time": [0.2],
        "runtime_connection_window_requests": [5.0],
        "runtime_connection_window_updates": [3.0],
        "runtime_connection_window_samples": [300.0],
        "runtime_accounted_time": [3.8],
        "runtime_unaccounted_time": [0.2],
    }


def test_disabled_tracker_does_not_require_wandb() -> None:
    """Disabled mode should be a dependency-free no-op."""
    tracker = initialize_experiment_tracker(
        mode="disabled",
        project="test-project",
        run_name="test-run",
        group="test-group",
        config={},
    )

    tracker.log({"performance/reward": 1.0}, step=1)
    tracker.finish()

    assert not tracker.enabled
    assert tracker.url is None


def test_offline_tracker_initializes_logs_and_finishes(monkeypatch) -> None:
    """Enabled tracking should pass metadata and one explicit episode step."""
    fake_run = _FakeRun()
    init_kwargs = {}

    def fake_init(**kwargs):
        init_kwargs.update(kwargs)
        return fake_run

    monkeypatch.setitem(sys.modules, "wandb", types.SimpleNamespace(init=fake_init))
    tracker = initialize_experiment_tracker(
        mode="offline",
        project="test-project",
        entity="test-team",
        run_name="Graph-GAT MAPPO",
        group="medium-seed75",
        notes="smoke",
        config={"seed": 75},
    )

    tracker.log({"performance/reward": 3.0}, step=2)
    tracker.finish()

    assert init_kwargs["mode"] == "offline"
    assert init_kwargs["force"] is False
    assert init_kwargs["project"] == "test-project"
    assert init_kwargs["entity"] == "test-team"
    assert init_kwargs["group"] == "medium-seed75"
    assert fake_run.logged == [({"performance/reward": 3.0}, 2)]
    assert fake_run.finished


def test_episode_tracking_metrics_include_progress_and_graph_costs() -> None:
    """One episode record should contain outcomes, ETA, and Graph-GAT cost."""
    metrics = _build_episode_tracking_metrics(
        history=_make_history(),
        episode_number=2,
        num_episodes=10,
        episode_elapsed_seconds=4.0,
        total_elapsed_seconds=10.0,
    )

    assert metrics["performance/reward"] == pytest.approx(3.5)
    assert metrics["training/progress_percent"] == pytest.approx(20.0)
    assert metrics["training/eta_seconds"] == pytest.approx(40.0)
    assert metrics["graph/action_seconds"] == pytest.approx(0.02)
    assert metrics["graph/update_seconds"] == pytest.approx(0.06)
    assert metrics["simulation/queue_wait_seconds"] == pytest.approx(0.04)
    assert metrics["runtime/env_step_seconds"] == pytest.approx(0.3)
    assert metrics["runtime/connection_window_seconds"] == pytest.approx(0.2)
    assert metrics["runtime/connection_window_updates"] == pytest.approx(3.0)
    assert metrics["runtime/unaccounted_seconds"] == pytest.approx(0.2)
