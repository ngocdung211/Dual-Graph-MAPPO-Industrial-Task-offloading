"""Tests for DITEN environment invariants from the paper MDP."""

import pathlib
import sys
from typing import Dict, Tuple

import numpy as np
import pytest

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from environment.diten_env import DITENEnv
from environment.network_env import NetworkEnvironment
from environment.system_model import EdgeServer, IndustrialDevice, Subtask, TaskDAG
from utils.topology_scenarios_config import device_start_points, get_topology_scenario


def _build_task_dag() -> TaskDAG:
    """Build a small dependent task DAG."""
    task_dag = TaskDAG(task_id=1, t_max=1.0, e_max=1.0)
    task_dag.add_subtask(Subtask(1, cpu_cycles=1e6, data_size=1e5, result_size=1e4))
    task_dag.add_subtask(Subtask(2, cpu_cycles=2e6, data_size=2e5, result_size=2e4))
    task_dag.add_dependency(1, 2)
    return task_dag


def _build_diamond_task_dag() -> TaskDAG:
    """Build a diamond DAG where branch subtasks can overlap."""
    task_dag = TaskDAG(task_id=1, t_max=1.0, e_max=1.0)
    task_dag.add_subtask(Subtask(1, cpu_cycles=1e8, data_size=0.0, result_size=0.0))
    task_dag.add_subtask(Subtask(2, cpu_cycles=8e8, data_size=0.0, result_size=0.0))
    task_dag.add_subtask(Subtask(3, cpu_cycles=1e6, data_size=0.0, result_size=0.0))
    task_dag.add_subtask(Subtask(4, cpu_cycles=1e6, data_size=0.0, result_size=0.0))
    task_dag.add_dependency(1, 2)
    task_dag.add_dependency(1, 3)
    task_dag.add_dependency(2, 4)
    task_dag.add_dependency(3, 4)
    return task_dag


def _build_env(server_location: np.ndarray) -> DITENEnv:
    """Build a one-device, one-server environment."""
    device = IndustrialDevice(
        device_id=1,
        location=np.array([0.0, 0.0]),
        compute_power=1e9,
        transmit_power=0.5,
        energy_coeff=1e-28,
        speed_mps=1.0,
    )
    server = EdgeServer(
        server_id=1,
        location=server_location,
        compute_power=2.4e9,
        transmit_power=1.2,
        energy_coeff=1e-27,
        coverage_radius=12.0,
    )
    network_env = NetworkEnvironment(bandwidth=10e6, noise_power_dbm=-43)
    return DITENEnv(
        [device],
        [server],
        network_env,
        slot_duration=1.0,
        subslot_count=10,
        time_slots=1,
        lambda1=1.0,
        lambda2=2.0,
        lambda3=3.0,
        lambda4=4.0,
        lambda5=5.0,
        p_out_value=-0.5,
        local_estimation_error=0.0,
        edge_estimation_error=0.0,
    )


def _build_scenario_env(scenario_name: str, subslot_count: int = 20) -> DITENEnv:
    """Build a deterministic environment for a named topology."""
    scenario = get_topology_scenario(scenario_name)
    devices = [
        IndustrialDevice(
            device_id=device_index + 1,
            location=start.copy(),
            compute_power=1e9,
            transmit_power=0.5,
            energy_coeff=1e-28,
            speed_mps=1.0,
        )
        for device_index, start in enumerate(device_start_points(scenario))
    ]
    servers = [
        EdgeServer(
            server_id=server_index + 1,
            location=np.asarray(location, dtype=float),
            compute_power=2e9,
            transmit_power=1.2,
            energy_coeff=1e-27,
            coverage_radius=scenario.coverage_radius_for_server(server_index),
        )
        for server_index, location in enumerate(scenario.server_locations)
    ]
    return DITENEnv(
        devices,
        servers,
        NetworkEnvironment(bandwidth=10e6, noise_power_dbm=-43),
        slot_duration=1.0,
        subslot_count=subslot_count,
        time_slots=2,
        local_estimation_error=0.0,
        edge_estimation_error=0.0,
        route_rectangles=scenario.route_rectangles,
        world_size=scenario.world_size,
    )


def _sequential_connection_windows(
    env: DITENEnv,
) -> Dict[Tuple[int, int], Tuple[float, float]]:
    """Compute the pre-optimization sampled windows as a test reference."""
    windows = {}
    horizon = env.slot_duration
    delta_time = env.slot_duration / float(env.subslot_count)
    for device in env.devices:
        for server in env.servers:
            window_start = None
            window_end = None
            for sample_index in range(env.subslot_count + 1):
                relative_time = sample_index * delta_time
                location = (
                    device.location
                    + device.direction * device.speed_mps * relative_time
                )
                inside = (
                    np.linalg.norm(location - server.location)
                    <= server.coverage_radius
                )
                if inside and window_start is None:
                    window_start = env.current_slot + relative_time
                if not inside and window_start is not None:
                    window_end = env.current_slot + relative_time
                    break

            if window_start is None:
                window_start = env.current_slot + horizon
                window_end = env.current_slot + horizon
            elif window_end is None:
                window_end = env.current_slot + horizon
            windows[(device.id, server.id)] = (window_start, window_end)
    return windows


def test_joint_state_matches_declared_dimension() -> None:
    """Eq. 23 state should match the declared state dimension."""
    env = _build_env(server_location=np.array([0.0, 1.0]))
    task_dag = _build_task_dag()

    joint_state = env.start_time_slot({1: task_dag}, {1: [1, 2]})

    assert joint_state.shape == (1, env.get_state_dim())


def test_reward_matches_equation_24_terms() -> None:
    """Reward should follow Eq. 24 arithmetic exactly."""
    env = _build_env(server_location=np.array([0.0, 1.0]))
    task_dag = _build_task_dag()

    reward = env._calculate_reward(
        t_im=0.1,
        e_im=0.2,
        t_accm=0.3,
        e_accm=0.4,
        p_out=-0.5,
        task_dag=task_dag,
    )

    expected = (
        1.0 * ((1.0 / 2) - 0.1)
        + 2.0 * (1.0 - 0.3)
        + 3.0 * ((1.0 / 2) - 0.2)
        + 4.0 * (1.0 - 0.4)
        + 5.0 * (-0.5)
    )
    assert reward == pytest.approx(expected)


def test_physical_compute_delay_uses_actual_cpu_only() -> None:
    """Changing the DT estimate must not alter physical execution delay."""
    network_env = NetworkEnvironment(bandwidth=10e6, noise_power_dbm=-43)

    local_delay_a, _ = network_env.calculate_local_computation(
        cpu_cycles=2e9,
        energy_coeff=1e-28,
        f_est=0.5e9,
        f_actual=1e9,
    )
    local_delay_b, _ = network_env.calculate_local_computation(
        cpu_cycles=2e9,
        energy_coeff=1e-28,
        f_est=1.5e9,
        f_actual=1e9,
    )
    edge_delay, edge_energy = network_env.calculate_edge_computation(
        cpu_cycles=2e9,
        energy_coeff=1e-27,
        f_est=2e9,
        f_actual=4e9,
    )

    assert local_delay_a == pytest.approx(2.0)
    assert local_delay_b == pytest.approx(2.0)
    assert edge_delay == pytest.approx(0.5)
    assert edge_energy == pytest.approx(0.0)


def test_connection_window_violation_is_recorded() -> None:
    """Invalid edge offload should be visible in step metrics."""
    env = _build_env(server_location=np.array([100.0, 100.0]))
    task_dag = TaskDAG(task_id=1, t_max=1.0, e_max=1.0)
    task_dag.add_subtask(Subtask(1, cpu_cycles=1e6, data_size=1e5, result_size=1e4))
    env.reset({1: task_dag}, {1: [1]})

    env.step([1])

    assert env.last_step_metrics[0]["p_out"] == pytest.approx(-0.5)
    assert env.last_step_metrics[0]["fallback_local"] == pytest.approx(1.0)
    assert env.last_step_metrics[0]["requested_action"] == pytest.approx(1.0)
    assert env.last_step_metrics[0]["action"] == pytest.approx(0.0)
    assert env.last_step_metrics[0]["penalty_applied"] == pytest.approx(1.0)
    assert env.last_step_metrics[0]["penalty_time"] > 0.0
    assert env.last_step_metrics[0]["local_time"] > 0.0
    assert env.last_step_metrics[0]["server_time"] == pytest.approx(0.0)
    assert env.last_step_metrics[0]["attempted_server_time"] > 0.0
    assert env.last_step_metrics[0]["queue_or_wait_time"] >= 0.0
    assert env.last_step_metrics[0]["transfer_time"] >= 0.0


def test_rejected_initial_offload_does_not_charge_upload_energy() -> None:
    """Feasibility rejection should occur before an initial input upload."""
    env = _build_env(server_location=np.array([100.0, 100.0]))
    task_dag = TaskDAG(task_id=1, t_max=1.0, e_max=1.0)
    task_dag.add_subtask(
        Subtask(1, cpu_cycles=1e6, data_size=1e8, result_size=1e4)
    )
    env.reset({1: task_dag}, {1: [1]})

    env.step([1])

    assert env.last_step_metrics[0]["transfer_time"] == pytest.approx(0.0)
    assert env.last_step_metrics[0]["tx_energy"] == pytest.approx(0.0)


def test_step_reward_uses_updated_accumulated_costs() -> None:
    """Eq. 24 accumulated terms should include the current subtask."""
    env = _build_env(server_location=np.array([0.0, 1.0]))
    task_dag = TaskDAG(task_id=1, t_max=1.0, e_max=1.0)
    task_dag.add_subtask(
        Subtask(1, cpu_cycles=1e6, data_size=0.0, result_size=0.0)
    )
    env.reset({1: task_dag}, {1: [1]})

    _, rewards, _, _ = env.step([0])
    metric = env.last_step_metrics[0]
    expected_reward = env._calculate_reward(
        metric["delay"],
        metric["energy"],
        env.slot_accumulated_delay[1],
        env.slot_accumulated_energy[1],
        0.0,
        task_dag,
    )

    assert rewards[0] == pytest.approx(expected_reward)


def test_invalid_priority_order_is_rejected() -> None:
    """A successor cannot be scheduled before its predecessor."""
    env = _build_env(server_location=np.array([0.0, 1.0]))
    task_dag = _build_task_dag()

    with pytest.raises(ValueError, match="predecessor"):
        env.start_time_slot({1: task_dag}, {1: [2, 1]})


def test_parallel_branch_delay_uses_dag_makespan() -> None:
    """A faster parallel branch should not add to accumulated DAG delay."""
    env = _build_env(server_location=np.array([10.0, 10.0]))
    task_dag = _build_diamond_task_dag()
    env.reset({1: task_dag}, {1: [1, 2, 3, 4]})

    env.step([0])
    env.step([0])
    delay_after_local_branch = env.device_accumulated_delay[1]

    env.step([1])

    edge_branch_finish_elapsed = env.last_step_metrics[0]["finish_time"] - env.current_slot
    expected_makespan = max(delay_after_local_branch, edge_branch_finish_elapsed)
    assert edge_branch_finish_elapsed < delay_after_local_branch
    assert env.device_accumulated_delay[1] == pytest.approx(expected_makespan)


@pytest.mark.parametrize(
    "scenario_name",
    [
        "paper_10d_3s",
        "medium_20d_6s",
        "large_30d_9s",
        "large_industrial_30d_9s",
    ],
)
def test_vectorized_connection_windows_match_sequential_reference(
    scenario_name: str,
) -> None:
    """Vectorized sampled windows must preserve the original semantics."""
    env = _build_scenario_env(scenario_name)
    env.reset_episode()

    expected = _sequential_connection_windows(env)

    assert env.connection_windows.keys() == expected.keys()
    for pair, expected_window in expected.items():
        assert env.connection_windows[pair] == pytest.approx(
            expected_window, abs=1e-12
        )


@pytest.mark.parametrize(
    ("device_location", "device_direction", "server_location", "radius"),
    [
        ([-0.5, 0.0], [1.0, 0.0], [0.0, 0.0], 0.25),
        ([0.0, 0.0], [1.0, 0.0], [5.0, 0.0], 0.1),
        ([0.0, 0.0], [1.0, 0.0], [0.0, 0.0], 10.0),
    ],
)
def test_vectorized_connection_window_boundary_cases_match_reference(
    device_location: list,
    device_direction: list,
    server_location: list,
    radius: float,
) -> None:
    """Boundary, disconnected, and full-horizon links must remain equivalent."""
    env = _build_env(server_location=np.asarray(server_location, dtype=float))
    env.devices[0].location = np.asarray(device_location, dtype=float)
    env.devices[0].direction = np.asarray(device_direction, dtype=float)
    env.servers[0].coverage_radius = radius
    env._connection_window_cache_key = None

    expected = _sequential_connection_windows(env)
    env._update_connection_windows()

    for pair, expected_window in expected.items():
        assert env.connection_windows[pair] == pytest.approx(
            expected_window, abs=1e-12
        )


def test_duplicate_connection_window_request_uses_cached_result() -> None:
    """An unchanged slot and mobility state should not rescan all samples."""
    env = _build_scenario_env("paper_10d_3s")
    env.reset_episode()
    first_metrics = env.get_runtime_metrics()
    first_windows = dict(env.connection_windows)

    env._update_connection_windows()
    second_metrics = env.get_runtime_metrics()

    assert env.connection_windows == first_windows
    assert second_metrics["connection_window_requests"] == pytest.approx(
        first_metrics["connection_window_requests"] + 1.0
    )
    assert second_metrics["connection_window_updates"] == pytest.approx(
        first_metrics["connection_window_updates"]
    )
    assert second_metrics["connection_window_samples"] == pytest.approx(
        first_metrics["connection_window_samples"]
    )


def test_slot_transition_invalidates_connection_window_cache() -> None:
    """Moving devices to a new slot must compute fresh connection windows."""
    env = _build_scenario_env("paper_10d_3s")
    task_dags = {}
    priorities = {}
    for device in env.devices:
        task_dag = TaskDAG(task_id=device.id, t_max=1.0, e_max=1.0)
        task_dag.add_subtask(
            Subtask(1, cpu_cycles=1e6, data_size=1e5, result_size=1e4)
        )
        task_dags[device.id] = task_dag
        priorities[device.id] = [1]

    env.reset_episode()
    env.start_time_slot(task_dags, priorities)
    metrics_before_step = env.get_runtime_metrics()
    env.step([0] * len(env.devices))
    metrics_after_step = env.get_runtime_metrics()

    assert metrics_before_step["connection_window_requests"] == pytest.approx(2.0)
    assert metrics_before_step["connection_window_updates"] == pytest.approx(1.0)
    assert metrics_after_step["connection_window_requests"] == pytest.approx(3.0)
    assert metrics_after_step["connection_window_updates"] == pytest.approx(2.0)
    assert metrics_after_step["joint_state_calls"] == pytest.approx(2.0)
