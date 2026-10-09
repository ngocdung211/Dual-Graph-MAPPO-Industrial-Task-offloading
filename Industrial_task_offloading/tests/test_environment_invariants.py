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
from utils.topology.scenarios import device_start_points, get_topology_scenario


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


def test_episode_reset_replays_route_from_initial_corner() -> None:
    """Every device must replay the same route after an episode reset."""
    env = _build_scenario_env("paper_10d_3s", subslot_count=10)

    env.reset_episode()
    initial_locations = [device.location.copy() for device in env.devices]
    initial_directions = [device.direction.copy() for device in env.devices]
    for device in env.devices:
        env._move_device_on_path(device, elapsed=25.0)
    first_positions_after_movement = [device.location.copy() for device in env.devices]
    assert any(
        not np.array_equal(start, moved)
        for start, moved in zip(initial_locations, first_positions_after_movement)
    )

    env.reset_episode()

    for device, location, direction in zip(
        env.devices, initial_locations, initial_directions
    ):
        np.testing.assert_array_equal(device.location, location)
        np.testing.assert_array_equal(device.direction, direction)
        env._move_device_on_path(device, elapsed=25.0)
    for device, position in zip(env.devices, first_positions_after_movement):
        np.testing.assert_array_equal(device.location, position)


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


def _request_case(cpu_cycles=1e6, data_size=0.0, window=(0.0, 1.0)):
    """Prepare a deterministic one-subtask request admission scenario."""
    env = _build_env(server_location=np.array([0.0, 1.0]))
    env.enable_request_overhead = True
    env.lambda5 = 1.0
    env.p_out_value = -1.5
    task = TaskDAG(1, t_max=1.0, e_max=1.0)
    task.add_subtask(Subtask(1, cpu_cycles, data_size, 0.0))
    env.reset({1: task}, {1: [1]})
    env.connection_windows[(1, 1)] = window
    return env


def test_request_accepted_upload_starts_after_ack():
    env = _request_case(data_size=1e6)
    upload_time, upload_energy = env._calculate_input_upload(
        env.devices[0], 1, 1e6
    )
    env.step([1])
    metric = env.last_step_metrics[0]
    assert metric["request_accepted_count"] == 1
    assert metric["request_rejected_count"] == 0
    assert metric["request_timeout_count"] == 0
    assert metric["request_finish_time"] == pytest.approx(0.002)
    assert metric["start_time"] == pytest.approx(0.002 + upload_time)
    assert metric["energy"] == pytest.approx(upload_energy + 0.00055)
    assert metric["comp_energy"] == 0
    assert metric["p_out"] == 0


def test_request_rejected_before_upload_pays_response_and_penalty_once():
    env = _request_case(cpu_cycles=1e8, data_size=1e8, window=(0, 0.05))
    env.step([1])
    metric = env.last_step_metrics[0]
    assert metric["request_rejected_count"] == 1
    assert metric["request_timeout_count"] == 0
    assert metric["start_time"] == pytest.approx(0.002)
    assert metric["finish_time"] == pytest.approx(0.102)
    assert metric["energy"] == pytest.approx(0.01055)
    assert metric["tx_energy"] == 0
    assert metric["p_out"] == -1.5
    assert metric["penalty_applied"] == 1
    assert metric["server_time"] == 0
    assert env.server_finish_time[1] == 0


def test_request_timeout_waits_from_end_of_transmission():
    env = _request_case(data_size=1e8, window=(1, 1))
    env.step([1])
    metric = env.last_step_metrics[0]
    assert metric["request_timeout_count"] == 1
    assert metric["request_rejected_count"] == 0
    assert metric["request_finish_time"] == pytest.approx(0.101)
    assert metric["start_time"] == pytest.approx(0.101)
    assert metric["finish_time"] == pytest.approx(0.102)
    assert metric["request_wait_energy"] == pytest.approx(0.005)
    assert metric["energy"] == pytest.approx(0.0056)
    assert metric["tx_energy"] == 0
    assert metric["p_out"] == -1.5
    assert env.server_finish_time[1] == 0


@pytest.mark.parametrize("window", [(0, 0.0015), (0.2, 1)])
def test_missing_link_or_expired_response_causes_timeout(window):
    env = _request_case(window=window)
    env.step([1])
    metric = env.last_step_metrics[0]
    assert metric["request_timeout_count"] == 1
    assert metric["request_finish_time"] == pytest.approx(0.101)


def test_handshake_changes_execution_admission():
    env = _request_case(cpu_cycles=1e8, data_size=1e5)
    upload_time, _ = env._calculate_input_upload(env.devices[0], 1, 1e5)
    window = (0, upload_time + 1e8 / 2.4e9 + 0.001)
    env.connection_windows[(1, 1)] = window
    legacy = _request_case(cpu_cycles=1e8, data_size=1e5, window=window)
    legacy.enable_request_overhead = False
    legacy.step([1])
    env.step([1])
    assert legacy.last_step_metrics[0]["action"] == 1
    assert env.last_step_metrics[0]["request_rejected_count"] == 1
    assert env.last_step_metrics[0]["action"] == 0
    assert env.last_step_metrics[0]["tx_energy"] == 0


def test_local_actions_do_not_pay_request_overhead():
    env = _request_case()
    env.step([0])
    metric = env.last_step_metrics[0]
    assert metric["energy"] == pytest.approx(0.0001)
    assert metric["finish_time"] == pytest.approx(0.001)
    assert metric["request_time"] == 0
    assert metric["request_energy"] == 0
    assert metric["request_wait_energy"] == 0


def test_timeout_can_overlap_local_queue_but_energy_is_retained():
    env = _request_case(window=(1, 1))
    env.local_finish_time[1] = 0.5
    env.step([1])
    metric = env.last_step_metrics[0]
    assert metric["start_time"] == pytest.approx(0.5)
    assert metric["finish_time"] == pytest.approx(0.501)
    assert metric["energy"] == pytest.approx(0.0056)


def test_request_waits_for_predecessors_without_serializing_dag_branches():
    env = _request_case()
    task = TaskDAG(1, 1, 1)
    task.add_subtask(Subtask(1, 1e8, 0, 0))
    task.add_subtask(Subtask(2, 1e6, 0, 0))
    task.add_subtask(Subtask(3, 1e6, 0, 0))
    task.add_dependency(1, 2)
    task.add_dependency(1, 3)
    env.reset({1: task}, {1: [1, 2, 3]})
    env.connection_windows[(1, 1)] = (1, 1)
    env.step([0])
    env.step([1])
    assert env.last_step_metrics[0]["request_start_time"] == pytest.approx(0.1)
    assert env.last_step_metrics[0]["finish_time"] == pytest.approx(0.202)
    env.step([1])
    assert env.last_step_metrics[0]["request_start_time"] == pytest.approx(0.1)


def test_disabled_request_parameters_preserve_legacy_trajectory():
    first = _build_env(np.array([0.0, 1.0]))
    second = _build_env(np.array([0.0, 1.0]))
    second.request_duration_s = 0.2
    second.response_duration_s = 0.1
    second.request_timeout_s = 0.3
    second.request_listen_power_w = 10
    states = [env.reset({1: _build_task_dag()}, {1: [1, 2]})
              for env in (first, second)]
    np.testing.assert_array_equal(*states)
    for action in [1, 0]:
        result_a = first.step([action])
        result_b = second.step([action])
        np.testing.assert_array_equal(result_a[0], result_b[0])
        assert result_a[1:] == result_b[1:]
        assert first.last_step_metrics == second.last_step_metrics


def test_queue_rejection_does_not_reserve_server_resources():
    env = _request_case(window=(0, 0.1))
    env.server_finish_time[1] = 0.2
    env.step([1])
    metric = env.last_step_metrics[0]
    assert metric["request_rejected_count"] == 1
    assert metric["finish_time"] == pytest.approx(0.003)
    assert env.server_finish_time[1] == 0.2


def test_request_admission_reserves_only_successful_edge_work():
    first = _build_env(np.array([0.0, 1.0]))
    second_device = IndustrialDevice(
        2, np.array([0.0, 0.0]), 1e9, 0.5, 1e-28, 1.0
    )
    env = DITENEnv(
        [first.devices[0], second_device], first.servers, first.network_env,
        enable_request_overhead=True, time_slots=1,
        local_estimation_error=0, edge_estimation_error=0,
        lambda5=1, p_out_value=-1.5,
    )
    tasks = {}
    for device in env.devices:
        task = TaskDAG(device.id, 1, 1)
        task.add_subtask(Subtask(1, 2.4e8, 0, 0))
        tasks[device.id] = task
    env.reset(tasks, {1: [1], 2: [1]})
    env.connection_windows = {(1, 1): (0, 0.15), (2, 1): (0, 0.15)}
    env.step([1, 1])
    from utils.comparison.diagnostics import _summarize_step_metrics

    summary = _summarize_step_metrics(env.last_step_metrics)
    assert summary["requested_edge_count"] == 2
    assert summary["request_accepted_count"] == 1
    assert summary["request_rejected_count"] == 1
    assert summary["request_timeout_count"] == 0
    assert summary["resolved_edge_count"] == 1
    assert summary["penalty_count"] == 1
    assert env.server_finish_time[1] == pytest.approx(0.102)


def test_request_checkpoint_evaluation_restores_settings_and_legacy_default(
    monkeypatch,
):
    from utils.comparison.evaluation import evaluate_algorithm_checkpoint
    from utils.paper_config import PAPER_PARAMS

    class FixedEdgeAgent:
        def __init__(self, **kwargs):
            del kwargs

        def select_greedy_action(self, state):
            del state
            return 1

    class FixedWorkload:
        def reseed(self, seed):
            del seed

        def get_random_task_parameters(self):
            return {
                f"subtask_{index}": {
                    "cpu_cycles": 1e6, "data_size": 0, "result_size": 0
                } for index in range(1, 6)
            }

    monkeypatch.setitem(PAPER_PARAMS["confirmed"], "time_slots", 1)
    physical = _build_env(np.array([100.0, 100.0]))
    checkpoint = {"agents": [{}]}
    settings = _request_case().get_request_overhead_config()
    common = {
        "agent_config": {"class": FixedEdgeAgent},
        "devices": physical.devices, "servers": physical.servers,
        "network_env": physical.network_env, "data_loader": FixedWorkload(),
        "priority_model": None, "num_episodes": 1, "experiment_seed": 190,
        "fixed_priority_order": [1, 2, 3, 4, 5],
    }
    legacy = evaluate_algorithm_checkpoint(checkpoint=checkpoint, **common)
    restored = evaluate_algorithm_checkpoint(
        checkpoint={**checkpoint, "request_overhead_config": settings},
        **common,
    )
    overridden = evaluate_algorithm_checkpoint(
        checkpoint=checkpoint, request_overhead_config=settings, **common
    )
    assert legacy["request_timeout_count"] == [0]
    assert restored["request_timeout_count"] == [5]
    assert restored["energy"][0] > legacy["energy"][0]
    assert restored == overridden


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
