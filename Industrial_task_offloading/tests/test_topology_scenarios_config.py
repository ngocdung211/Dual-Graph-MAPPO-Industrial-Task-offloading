"""Tests for reproducible industrial topology scenario configuration."""

import pathlib
import sys

import numpy as np
import pytest

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from environment.diten_env import DITENEnv
from environment.network_env import NetworkEnvironment
from environment.system_model import EdgeServer, IndustrialDevice
from inference_priority_comparison import _checkpoint_scenario
from run_comparision import build_servers_for_scenario, parse_args
from utils.comparison.outputs import flatten_topology_metrics
from utils.paper_config import PAPER_PARAMS
from utils.topology.scenarios import (
    compute_topology_metrics,
    get_topology_scenario,
)
from utils.topology.preview import write_previews


def test_large_industrial_topology_uses_rectangular_180_by_120_world() -> None:
    """The expanded scenario must not be clipped to the legacy 100 m square."""
    scenario = get_topology_scenario("large_industrial_30d_9s")

    assert scenario.world_size == (180.0, 120.0)
    assert scenario.device_count == 30
    assert len(scenario.server_locations) == 9
    all_route_points = np.asarray(scenario.route_rectangles, dtype=float).reshape(-1, 2)
    all_server_points = np.asarray(scenario.server_locations, dtype=float)
    assert np.max(all_route_points[:, 0]) > 100.0
    assert np.max(all_server_points[:, 0]) > 100.0
    assert np.all((all_route_points >= [0.0, 0.0]))
    assert np.all((all_route_points <= [180.0, 120.0]))


def test_heterogeneous_coverage_is_seeded_and_varies_by_server() -> None:
    """A changed seed should alter radii, while repeated seeds reproduce them."""
    first = get_topology_scenario(
        "large_industrial_30d_9s",
        topology_seed=41,
        server_profile="heterogeneous",
    )
    repeated = get_topology_scenario(
        "large_industrial_30d_9s",
        topology_seed=41,
        server_profile="heterogeneous",
    )
    changed = get_topology_scenario(
        "large_industrial_30d_9s",
        topology_seed=42,
        server_profile="heterogeneous",
    )

    assert first.coverage_radii == repeated.coverage_radii
    assert first.coverage_radii != changed.coverage_radii
    assert len(set(first.coverage_radii)) > 1
    assert min(first.coverage_radii) >= 15.0
    assert max(first.coverage_radii) <= 20.0
    assert first.server_locations == changed.server_locations
    assert first.route_rectangles == changed.route_rectangles


def test_server_profiles_control_coverage_difficulty() -> None:
    """Uniform and stress profiles must provide distinct reproducible regimes."""
    uniform = get_topology_scenario(
        "large_industrial_30d_9s", topology_seed=7, server_profile="uniform"
    )
    stress = get_topology_scenario(
        "large_industrial_30d_9s", topology_seed=7, server_profile="stress"
    )

    assert uniform.coverage_radii == (18.0,) * 9
    assert min(stress.coverage_radii) >= 12.0
    assert max(stress.coverage_radii) <= 15.0
    assert max(stress.coverage_radii) < min(uniform.coverage_radii)


def test_topology_metrics_report_world_and_heterogeneous_coverage() -> None:
    """Experiment metadata must describe the exact sampled physical topology."""
    scenario = get_topology_scenario(
        "large_industrial_30d_9s",
        topology_seed=73,
        server_profile="heterogeneous",
    )

    metrics = compute_topology_metrics(scenario)

    assert metrics["world_size_m"] == [180.0, 120.0]
    assert metrics["topology_seed"] == 73
    assert metrics["server_profile"] == "heterogeneous"
    assert metrics["coverage_radii_m"] == list(scenario.coverage_radii)
    assert metrics["coverage_radius_min_m"] == min(scenario.coverage_radii)
    assert metrics["coverage_radius_max_m"] == max(scenario.coverage_radii)
    assert 0.35 <= metrics["route_samples"]["zero_link_ratio"] <= 0.45
    assert 0.02 <= metrics["route_samples"]["multi_link_ratio"] <= 0.10
    flattened = flatten_topology_metrics(metrics)
    assert flattened["topology_world_width_m"] == 180.0
    assert flattened["topology_world_height_m"] == 120.0
    assert flattened["topology_seed"] == 73
    assert flattened["topology_server_profile"] == "heterogeneous"
    assert flattened["topology_coverage_radius_min_m"] == min(
        scenario.coverage_radii
    )


def test_environment_preserves_routes_beyond_legacy_world_boundary() -> None:
    """A configured 180 m world must not clip route waypoints at 100 m."""
    device = IndustrialDevice(
        device_id=1,
        location=np.array([150.0, 10.0]),
        compute_power=1e9,
        transmit_power=0.5,
        energy_coeff=1e-28,
    )
    server = EdgeServer(
        server_id=1,
        location=np.array([155.0, 20.0]),
        compute_power=2e9,
        transmit_power=1.2,
        energy_coeff=1e-27,
        coverage_radius=20.0,
    )

    environment = DITENEnv(
        [device],
        [server],
        NetworkEnvironment(bandwidth=10e6, noise_power_dbm=-43),
        route_rectangles=[[[150.0, 10.0], [150.0, 30.0], [170.0, 30.0], [170.0, 10.0]]],
        world_size=(180.0, 120.0),
    )

    assert environment.world_max.tolist() == [180.0, 120.0]
    assert max(point[0] for point in environment.device_waypoints[1]) == 170.0


def test_server_builder_uses_each_resolved_coverage_radius() -> None:
    """Physical servers must receive their own scenario coverage radii."""
    scenario = get_topology_scenario(
        "large_industrial_30d_9s",
        topology_seed=91,
        server_profile="heterogeneous",
    )

    servers = build_servers_for_scenario(
        scenario,
        PAPER_PARAMS["confirmed"],
        PAPER_PARAMS["provisional_table2_needed"],
    )

    assert tuple(server.coverage_radius for server in servers) == scenario.coverage_radii


def test_comparison_cli_accepts_topology_seed_and_server_profile(monkeypatch) -> None:
    """Experiment runs must expose reproducible physical-topology controls."""
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_comparision.py",
            "--topology-scenario",
            "large_industrial_30d_9s",
            "--topology-seed",
            "314",
            "--server-profile",
            "stress",
        ],
    )

    args = parse_args()

    assert args.topology_seed == 314
    assert args.server_profile == "stress"


def test_comparison_cli_accepts_modular_paper_run(monkeypatch) -> None:
    """The planned six-model comparison command must remain runnable."""
    algorithms = [
        "Shared MAPPO",
        "GATMA-Adapted",
        "Graph-GAT MAPPO",
        "Graph-GAT Warmup MAPPO",
        "Graph-GAT Warmup Mask MAPPO",
        "e-ATN-MADDPG",
    ]
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_comparision.py",
            "--algorithms", *algorithms,
            "--topology-scenario", "modular_cells_30d_9s",
            "--server-profile", "scenario",
            "--episodes", "1",
            "--experiment-seed", "75",
            "--task-priority", "on",
            "--use-gae",
            "--num-minibatches", "4",
            "--maddpg-updates-per-episode", "16",
            "--maddpg-actor-replay-actions",
            "--gatma-device", "cpu",
            "--graph-gat-device", "cuda",
            "--dataset-path", "dataset/KolektorSDD",
            "--local-output-root", "plots",
            "--drive-artifact-root", r"G:\My Drive\Dual-Graph-MAPPO-Artifacts",
            "--wandb-mode", "online",
            "--wandb-project", "industrial-task-offloading",
            "--wandb-group", "modular_cells_30d_9s-seed75",
            "--note", "modular_cells_30d_9s-seed75",
        ],
    )

    args = parse_args()

    assert args.algorithms == algorithms
    assert args.server_profile == "scenario"
    assert args.use_gae is True
    assert args.num_minibatches == 4
    assert args.maddpg_updates_per_episode == 16
    assert args.maddpg_actor_replay_actions is True
    assert args.graph_gat_device == "cuda"
    assert args.gatma_device == "cpu"


@pytest.mark.parametrize(
    "removed_flag, value",
    [
        ("--hyperparameters-dir", "old-profile"),
        ("--graph-gat-lr", "0.001"),
        ("--mappo-entropy-coef", "0.01"),
        ("--maddpg-epsilon-schedule", "paper_decay"),
    ],
)
def test_comparison_cli_rejects_removed_model_flags(
    monkeypatch, removed_flag: str, value: str
) -> None:
    """Model tuning moves to config, so old CLI overrides must fail clearly."""
    monkeypatch.setattr(sys, "argv", ["run_comparision.py", removed_flag, value])

    with pytest.raises(SystemExit) as error:
        parse_args()

    assert error.value.code == 2


def test_preview_writes_selected_seeded_industrial_scenario(tmp_path) -> None:
    """Preview output must use the same scenario controls as training."""
    result = write_previews(
        str(tmp_path),
        topology_seed=271,
        server_profile="heterogeneous",
        scenario_names=["large_industrial_30d_9s"],
    )

    assert (tmp_path / "large_industrial_30d_9s.png").stat().st_size > 0
    assert list(result["metrics"]) == ["large_industrial_30d_9s"]
    metrics = result["metrics"]["large_industrial_30d_9s"]
    assert metrics["topology_seed"] == 271
    assert metrics["world_size_m"] == [180.0, 120.0]


def test_checkpoint_reconstructs_exact_seeded_server_coverage() -> None:
    """Inference must recreate the physical coverage stored by training."""
    topology_metrics = {
        "name": "large_industrial_30d_9s",
        "topology_seed": 811,
        "server_profile": "stress",
    }
    checkpoints = {
        "MAPPO": {"topology_metrics": dict(topology_metrics)},
        "Graph-GAT MAPPO": {"topology_metrics": dict(topology_metrics)},
    }

    scenario = _checkpoint_scenario(checkpoints)

    expected = get_topology_scenario(
        "large_industrial_30d_9s", topology_seed=811, server_profile="stress"
    )
    assert scenario.coverage_radii == expected.coverage_radii
    assert scenario.topology_seed == 811
    assert scenario.server_profile == "stress"


def test_modular_preview_and_servers_preserve_explicit_coverage(tmp_path) -> None:
    """Scenario resolution must not resample the approved per-server radii."""
    expected_radii = (12.0, 15.0, 12.0, 12.0, 15.0, 12.0, 12.0, 15.0, 12.0)
    scenario = get_topology_scenario("modular_cells_30d_9s", topology_seed=73)
    servers = build_servers_for_scenario(
        scenario,
        PAPER_PARAMS["confirmed"],
        PAPER_PARAMS["provisional_table2_needed"],
    )
    result = write_previews(
        str(tmp_path), topology_seed=73, scenario_names=[scenario.name]
    )
    metadata = result["metrics"][scenario.name]
    reconstructed = _checkpoint_scenario({"MAPPO": {"topology_metrics": metadata}})

    assert tuple(server.coverage_radius for server in servers) == expected_radii
    assert metadata["coverage_radii_m"] == list(expected_radii)
    assert reconstructed.coverage_radii == expected_radii
    assert (tmp_path / f"{scenario.name}.png").stat().st_size > 0


def test_modular_explicit_profile_override_still_works() -> None:
    """An explicit uniform override should replace the heterogeneous radii."""
    scenario = get_topology_scenario(
        "modular_cells_30d_9s", server_profile="uniform"
    )

    assert scenario.server_profile == "uniform"
    assert scenario.coverage_radii == (scenario.coverage_radius,) * 9
