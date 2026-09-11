"""Tests for the minimal physical-network digital twin."""

import pathlib
import sys

import numpy as np
import pytest

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from environment.digital_twin import DigitalTwin
from environment.diten_env import DITENEnv
from environment.network_env import NetworkEnvironment
from environment.system_model import (
    EdgeServer,
    IndustrialDevice,
    Subtask,
    TaskDAG,
)


def _build_physical_entities() -> tuple[list[IndustrialDevice], list[EdgeServer]]:
    """Build one physical device and server for twin synchronization."""
    devices = [
        IndustrialDevice(
            device_id=1,
            location=np.array([0.0, 0.0]),
            compute_power=1e9,
            transmit_power=0.5,
            energy_coeff=1e-28,
        )
    ]
    servers = [
        EdgeServer(
            server_id=1,
            location=np.array([1.0, 0.0]),
            compute_power=5e9,
            transmit_power=1.2,
            energy_coeff=1e-27,
            coverage_radius=12.0,
        )
    ]
    return devices, servers


def test_zero_error_snapshot_matches_physical_compute_power() -> None:
    """Perfect synchronization should reproduce physical CPU values."""
    devices, servers = _build_physical_entities()
    digital_twin = DigitalTwin()

    snapshot = digital_twin.synchronize(devices, servers, time_slot_index=3)

    assert snapshot.time_slot_index == 3
    assert snapshot.device_states[1].estimated_compute_power == pytest.approx(1e9)
    assert snapshot.server_states[1].estimated_compute_power == pytest.approx(5e9)
    assert digital_twin.latest_snapshot is snapshot


def test_noisy_snapshot_stays_in_range_without_changing_physical_state() -> None:
    """Twin CPU noise should be bounded and leave physical entities unchanged."""
    devices, servers = _build_physical_entities()
    digital_twin = DigitalTwin(
        local_estimation_error=0.2,
        edge_estimation_error=0.1,
        random_generator=np.random.default_rng(75),
    )

    snapshot = digital_twin.synchronize(devices, servers, time_slot_index=0)

    device_estimate = snapshot.device_states[1].estimated_compute_power
    server_estimate = snapshot.server_states[1].estimated_compute_power
    assert 0.8e9 <= device_estimate <= 1.2e9
    assert 4.5e9 <= server_estimate <= 5.5e9
    assert devices[0].compute_power == pytest.approx(1e9)
    assert servers[0].compute_power == pytest.approx(5e9)


def test_reset_discards_latest_snapshot() -> None:
    """Episode reset should remove the previous synchronized snapshot."""
    devices, servers = _build_physical_entities()
    digital_twin = DigitalTwin()
    digital_twin.synchronize(devices, servers, time_slot_index=0)

    digital_twin.reset()

    assert digital_twin.latest_snapshot is None


def test_environment_uses_synchronized_twin_estimates() -> None:
    """DITENEnv observations should use the current twin snapshot."""
    devices, servers = _build_physical_entities()
    env = DITENEnv(
        devices,
        servers,
        NetworkEnvironment(bandwidth=10e6, noise_power_dbm=-43),
        time_slots=1,
        local_estimation_error=0.2,
        edge_estimation_error=0.1,
    )
    task_dag = TaskDAG(task_id=1, t_max=1.0, e_max=1.0)
    task_dag.add_subtask(
        Subtask(1, cpu_cycles=1e6, data_size=1e5, result_size=1e4)
    )

    env.start_time_slot({1: task_dag}, {1: [1]})

    snapshot = env.digital_twin_snapshot
    assert snapshot is not None
    assert snapshot.time_slot_index == 0
    assert env.device_estimated_power[1] == pytest.approx(
        snapshot.device_states[1].estimated_compute_power
    )
    assert env.server_estimated_power[1] == pytest.approx(
        snapshot.server_states[1].estimated_compute_power
    )


def test_realized_compute_uses_estimate_minus_deviation() -> None:
    """Physical delay and energy should use f_actual = f_est - delta_f."""
    network = NetworkEnvironment(bandwidth=10e6, noise_power_dbm=-43)
    cpu_cycles = 2e9
    estimated_power = 1.2e9
    actual_power = 1.0e9

    delay, energy = network.calculate_local_computation(
        cpu_cycles=cpu_cycles,
        energy_coeff=1e-28,
        f_est=estimated_power,
        f_actual=actual_power,
    )

    compute_deviation = estimated_power - actual_power
    reconstructed_power = estimated_power - compute_deviation
    assert reconstructed_power == pytest.approx(actual_power)
    assert delay == pytest.approx(cpu_cycles / actual_power)
    assert energy == pytest.approx(
        1e-28 * cpu_cycles * reconstructed_power ** 2
    )
