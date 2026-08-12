"""Tests for comparison metric diagnostics."""

import pathlib
import sys

import pytest
import torch

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from baselines.mappo import MAPPOAgent
from baselines.offloading_baselines import LocalOnlyAgent
from run_comparision import (
    _episodes_for_algorithm,
    _format_diagnostic_summary,
    _should_print_diagnostics,
    _summarize_step_metrics,
    build_algorithm_configs,
    load_tuned_hyperparameters,
)
from utils.comparison_outputs import (
    _save_model_checkpoint,
    build_last_training_state_line,
    build_model_checkpoint,
    build_plot_results,
)
from utils.paper_config import PAPER_PARAMS
from tune_hyperparameters import (
    MODEL_E_ATN_MADDPG,
    MODEL_GRAPH_GAT_WARMUP_MAPPO,
    MODEL_MAPPO,
    build_agent_config_from_params,
    build_default_trial_params,
    phase_episode_budget,
    suggest_agent_config,
)


class _FixedSuggestionTrial:
    """Return predefined values through the Optuna trial suggestion API."""

    def __init__(self, values):
        self.values = values

    def suggest_float(self, name, *args, **kwargs):
        del args, kwargs
        return self.values[name]

    def suggest_categorical(self, name, choices):
        assert self.values[name] in choices
        return self.values[name]


def test_summarize_step_metrics_counts_penalties_and_times() -> None:
    """Step summary should expose penalty count and local/server time."""
    metrics = [
        {
            "requested_action": 1.0,
            "action": 0.0,
            "local_time": 0.4,
            "server_time": 0.0,
            "attempted_server_time": 0.2,
            "transfer_time": 0.1,
            "queue_or_wait_time": 0.3,
            "penalty_time": 0.05,
            "penalty_applied": 1.0,
        },
        {
            "requested_action": 1.0,
            "action": 1.0,
            "local_time": 0.0,
            "server_time": 0.7,
            "attempted_server_time": 0.7,
            "transfer_time": 0.2,
            "queue_or_wait_time": 0.1,
            "penalty_time": 0.0,
            "penalty_applied": 0.0,
        },
    ]

    summary = _summarize_step_metrics(metrics)

    assert summary["penalty_count"] == pytest.approx(1.0)
    assert summary["requested_local_count"] == pytest.approx(0.0)
    assert summary["requested_edge_count"] == pytest.approx(2.0)
    assert summary["resolved_local_count"] == pytest.approx(1.0)
    assert summary["resolved_edge_count"] == pytest.approx(1.0)
    assert summary["local_time"] == pytest.approx(0.4)
    assert summary["server_time"] == pytest.approx(0.7)
    assert summary["attempted_server_time"] == pytest.approx(0.9)
    assert summary["transfer_time"] == pytest.approx(0.3)
    assert summary["queue_or_wait_time"] == pytest.approx(0.4)
    assert summary["penalty_time"] == pytest.approx(0.05)


def test_format_diagnostic_summary_is_readable() -> None:
    """Diagnostic formatter should make counts and timings easy to scan."""
    history = {
        "requested_local_count": [2329.0],
        "requested_edge_count": [171.0],
        "resolved_local_count": [2480.0],
        "resolved_edge_count": [20.0],
        "penalty_count": [151.0],
        "local_time": [2.75],
        "server_time": [0.081],
        "transfer_time": [0.21],
        "queue_or_wait_time": [0.44],
        "penalty_time": [0.856],
    }

    summary = _format_diagnostic_summary("Random Offloading", 10, history)

    assert "[Random Offloading] Episode 10 diagnostics" in summary
    assert "Requested actions: local=2329 edge=171" in summary
    assert "Actual execution:  local=2480 edge=20" in summary
    assert "Penalties:         count=151 time=0.856s" in summary
    assert (
        "Timing/device-task: local=2.750s server=0.081s "
        "transfer=0.210s wait=0.440s"
    ) in summary


def test_short_run_diagnostics_print_only_on_final_episode() -> None:
    """Short comparison runs should avoid printing diagnostics every epoch."""
    assert not _should_print_diagnostics(1, 10)
    assert not _should_print_diagnostics(9, 10)
    assert _should_print_diagnostics(10, 10)
    assert not _should_print_diagnostics(50, 1000)
    assert _should_print_diagnostics(500, 1000)


def test_fixed_baselines_use_short_episode_budget() -> None:
    """Fixed policies should not run for the full learning episode budget."""
    assert _episodes_for_algorithm("Local Only", full_episodes=100, baseline_episodes=10) == 10
    assert _episodes_for_algorithm("Edge Only", full_episodes=100, baseline_episodes=10) == 10
    assert _episodes_for_algorithm("Feature Extraction Edge", full_episodes=100, baseline_episodes=10) == 10
    assert _episodes_for_algorithm("Random Offloading", full_episodes=100, baseline_episodes=10) == 10
    assert _episodes_for_algorithm("e-ATN-MADDPG", full_episodes=100, baseline_episodes=10) == 100
    assert _episodes_for_algorithm("MAPPO", full_episodes=100, baseline_episodes=10) == 100


def test_learning_algorithm_kwargs_use_config_values() -> None:
    """Comparison model kwargs should come from centralized paper config."""
    provisional = PAPER_PARAMS["provisional_table2_needed"]
    configs = build_algorithm_configs()

    assert configs["MAPPO"]["kwargs"]["gamma"] == provisional["gamma"]
    assert configs["MAPPO"]["kwargs"]["clip_param"] == provisional["mappo_clip_param"]
    assert configs["MAPPO"]["kwargs"]["ppo_epochs"] == int(
        provisional["mappo_ppo_epochs"]
    )
    assert configs["MAPPO"]["kwargs"]["use_action_mask"] is False
    assert configs["Graph-GAT MAPPO"]["kwargs"]["gamma"] == provisional["gamma"]
    assert configs["Graph-GAT MAPPO"]["kwargs"]["hidden_dim"] == int(
        provisional["graph_gat_hidden_dim"]
    )
    assert configs["Graph-GAT MAPPO"]["kwargs"]["embedding_dim"] == int(
        provisional["graph_gat_embedding_dim"]
    )
    assert configs["Graph-GAT MAPPO"]["kwargs"]["use_action_mask"] is False


def test_locked_profile_transfers_best_params_to_comparison_models() -> None:
    """The locked Optuna profile should configure all four unmasked models."""
    profile_dir = PAPER_PARAMS["provisional_table2_needed"][
        "comparison_hyperparameters_dir"
    ]
    tuned_params, metadata = load_tuned_hyperparameters(profile_dir)

    configs = build_algorithm_configs(
        graph_gat_device="cpu",
        tuned_hyperparameters=tuned_params,
    )

    assert metadata["effective_failed_offload_penalty"] == pytest.approx(-0.5)
    assert configs["e-ATN-MADDPG"]["batch_size"] == 256
    assert configs["MAPPO"]["kwargs"]["actor_lr"] == pytest.approx(
        0.0004647005894999619
    )
    assert configs["Graph-GAT MAPPO"]["kwargs"][
        "topology_warmup_episodes"
    ] == 0
    assert configs["Graph-GAT Warmup MAPPO"]["kwargs"][
        "topology_warmup_episodes"
    ] == 20
    assert configs["Graph-GAT Warmup MAPPO"]["kwargs"][
        "topology_warmup_updates_per_step"
    ] == 4


def test_optuna_output_directory_transfers_without_manual_profile() -> None:
    """A normal Optuna output directory should be directly reusable."""
    tuned_params, metadata = load_tuned_hyperparameters(
        "results/optuna/unmasked_pairwise_penalty_0_25_5trial"
    )

    assert metadata["profile_generated_at_runtime"] is True
    assert tuned_params["e-ATN-MADDPG"]["batch_size"] == 256
    assert tuned_params["Graph-GAT Warmup MAPPO"]["warmup_episodes"] == 20


def test_optuna_default_trials_build_requested_unmasked_models() -> None:
    """Default enqueued trials should map to the three unmasked agents."""
    for model_name in (
        MODEL_E_ATN_MADDPG,
        MODEL_MAPPO,
        MODEL_GRAPH_GAT_WARMUP_MAPPO,
    ):
        trial = _FixedSuggestionTrial(build_default_trial_params(model_name))
        config = suggest_agent_config(model_name, trial, "cpu")

        assert config["kwargs"].get("use_action_mask", False) is False
    e_atn_config = suggest_agent_config(
        MODEL_E_ATN_MADDPG,
        _FixedSuggestionTrial(
            build_default_trial_params(MODEL_E_ATN_MADDPG)
        ),
        "cpu",
    )
    assert e_atn_config["kwargs"]["use_attention"] is True
    assert e_atn_config["kwargs"]["use_epsilon_greedy"] is True


def test_stored_optuna_params_rebuild_exact_unmasked_configs() -> None:
    """Rerank/final stages should reconstruct search configs without masks."""
    for model_name in (
        MODEL_E_ATN_MADDPG,
        MODEL_MAPPO,
        MODEL_GRAPH_GAT_WARMUP_MAPPO,
    ):
        params = build_default_trial_params(model_name)
        direct_config = build_agent_config_from_params(
            model_name, params, "cpu"
        )
        trial_config = suggest_agent_config(
            model_name, _FixedSuggestionTrial(params), "cpu"
        )

        assert direct_config == trial_config
        assert direct_config["kwargs"].get("use_action_mask", False) is False


def test_tuning_phase_budgets_match_balanced_plan() -> None:
    """Search, rerank, and final should use the agreed episode budgets."""
    assert phase_episode_budget("search", None) == 150
    assert phase_episode_budget("rerank", None) == 500
    assert phase_episode_budget("final", None) == 1000
    assert phase_episode_budget("final", 2) == 2
    with pytest.raises(ValueError, match="episodes must be positive"):
        phase_episode_budget("search", 0)


def test_fixed_baseline_plot_results_are_mean_flat_lines() -> None:
    """Short fixed-baseline runs should plot as mean-flat reference lines."""
    raw_results = {
        "reward": {
            "Local Only": [1.0, 3.0],
            "e-ATN-MADDPG": [0.5, 0.7, 0.9],
        },
        "delay": {
            "Local Only": [2.0, 4.0],
            "e-ATN-MADDPG": [4.0, 3.0, 2.0],
        },
    }

    plot_results = build_plot_results(
        raw_results,
        target_episodes=5,
        fixed_baseline_algorithms=frozenset({"Local Only"}),
    )

    assert plot_results["reward"]["Local Only"] == pytest.approx([2.0] * 5)
    assert plot_results["delay"]["Local Only"] == pytest.approx([3.0] * 5)
    assert plot_results["reward"]["e-ATN-MADDPG"] == pytest.approx([0.5, 0.7, 0.9])
    assert plot_results["delay"]["e-ATN-MADDPG"] == pytest.approx([4.0, 3.0, 2.0])


def test_last_training_state_line_is_flat_for_easy_comparison() -> None:
    """Last-state JSONL rows should keep one model on one flat line."""
    history = {
        "reward": [2.1, 2.866],
        "delay": [1.5, 1.24],
        "energy": [0.2, 0.116],
        "local_ratio": [90.0, 97.5],
        "edge_ratio": [10.0, 2.5],
        "local_time": [1.2, 1.109],
        "server_time": [0.1, 0.007],
        "transfer_time": [0.2, 0.008],
        "queue_or_wait_time": [1.4, 1.357],
        "penalty_time": [0.1, 0.0],
        "penalty_count": [3.0, 0.0],
        "requested_local_count": [2300.0, 2437.0],
        "requested_edge_count": [200.0, 63.0],
        "resolved_local_count": [2310.0, 2437.0],
        "resolved_edge_count": [190.0, 63.0],
    }

    line = build_last_training_state_line("MAPPO", history, episode_count=1000)

    assert line == {
        "model": "MAPPO",
        "episode": 1000,
        "reward": pytest.approx(2.866),
        "delay_s": pytest.approx(1.24),
        "energy_j": pytest.approx(0.116),
        "local_count": 2437,
        "edge_count": 63,
        "local_ratio_percent": pytest.approx(97.5),
        "edge_ratio_percent": pytest.approx(2.5),
        "requested_local_count": 2437,
        "requested_edge_count": 63,
        "resolved_local_count": 2437,
        "resolved_edge_count": 63,
        "local_time_s": pytest.approx(1.109),
        "server_time_s": pytest.approx(0.007),
        "transfer_time_s": pytest.approx(0.008),
        "wait_time_s": pytest.approx(1.357),
        "penalty_count": 0,
        "penalty_time_s": pytest.approx(0.0),
        "graph_warmup_time_s": pytest.approx(0.0),
        "graph_warmup_loss": pytest.approx(0.0),
        "graph_warmup_count": 0,
    }


def test_fixed_baseline_does_not_create_model_checkpoint() -> None:
    """Rule-based baselines should not save empty model checkpoints."""
    agent = LocalOnlyAgent(state_dim=4, action_dim=3, num_agents=1)

    checkpoint = build_model_checkpoint(
        algo_name="Local Only",
        agents=[agent],
        agent_config={"class": LocalOnlyAgent, "kwargs": {}},
        history={"reward": [1.0]},
        episode_count=5,
        state_dim=4,
        action_dim=3,
        num_devices=1,
        num_servers=2,
    )

    assert checkpoint is None


def test_mappo_checkpoint_contains_agent_weights_and_metadata() -> None:
    """Trainable checkpoints should include weights, config, and dimensions."""
    agent = MAPPOAgent(state_dim=4, action_dim=3, num_agents=1)

    checkpoint = build_model_checkpoint(
        algo_name="MAPPO",
        agents=[agent],
        agent_config={"class": MAPPOAgent, "kwargs": {"lr": 0.0001}},
        history={"reward": [1.0], "delay": [0.5]},
        episode_count=5,
        state_dim=4,
        action_dim=3,
        num_devices=1,
        num_servers=2,
    )

    assert checkpoint is not None
    assert checkpoint["model"] == "MAPPO"
    assert checkpoint["episode_count"] == 5
    assert checkpoint["state_dim"] == 4
    assert checkpoint["action_dim"] == 3
    assert checkpoint["num_devices"] == 1
    assert checkpoint["num_servers"] == 2
    assert checkpoint["agent_class"] == "MAPPOAgent"
    assert checkpoint["agent_kwargs"] == {"lr": 0.0001}
    assert len(checkpoint["agents"]) == 1
    assert "actor" in checkpoint["agents"][0]
    assert "critic" in checkpoint["agents"][0]
    assert checkpoint["final_metrics"]["reward"] == pytest.approx(1.0)


def test_save_model_checkpoint_writes_safe_filename(tmp_path) -> None:
    """Checkpoint files should be saved with readable safe names."""
    checkpoint = {"model": "Graph-GAT MAPPO", "episode_count": 5}

    checkpoint_path = _save_model_checkpoint(str(tmp_path), checkpoint)

    assert checkpoint_path.endswith("Graph-GAT_MAPPO_checkpoint.pt")
    loaded = torch.load(checkpoint_path, map_location="cpu")
    assert loaded["model"] == "Graph-GAT MAPPO"
