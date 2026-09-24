"""Deterministic wall-clock benchmark for DITEN environment scaling."""

import argparse
import cProfile
import io
import pathlib
import pstats
import statistics
import sys
import time
from typing import Dict, List

import numpy as np

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from environment.diten_env import DITENEnv
from environment.network_env import NetworkEnvironment
from environment.system_model import EdgeServer, IndustrialDevice, Subtask, TaskDAG
from utils.paper_config import PAPER_PARAMS
from utils.topology.scenarios import (
    available_topology_scenario_names,
    device_start_points,
    get_topology_scenario,
)


TASK_EDGES = [(1, 2), (1, 3), (2, 4), (3, 4), (4, 5)]


def _build_environment(scenario_name: str) -> DITENEnv:
    """Build a deterministic environment without model or dataset overhead."""
    confirmed = PAPER_PARAMS["confirmed"]
    provisional = PAPER_PARAMS["provisional_table2_needed"]
    scenario = get_topology_scenario(scenario_name)
    devices = [
        IndustrialDevice(
            device_id=device_index + 1,
            location=start.copy(),
            compute_power=1e9,
            transmit_power=confirmed["device_tx_power_w"],
            energy_coeff=confirmed["device_energy_coeff"],
            speed_mps=confirmed["device_speed_mps"],
        )
        for device_index, start in enumerate(device_start_points(scenario))
    ]
    servers = [
        EdgeServer(
            server_id=server_index + 1,
            location=np.asarray(location, dtype=float),
            compute_power=2e9,
            transmit_power=confirmed["server_tx_power_w"],
            energy_coeff=confirmed["server_energy_coeff"],
            coverage_radius=scenario.coverage_radius_for_server(server_index),
        )
        for server_index, location in enumerate(scenario.server_locations)
    ]
    network = NetworkEnvironment(
        bandwidth=confirmed["bandwidth_hz"],
        noise_power_dbm=confirmed["noise_power_dbm"],
    )
    return DITENEnv(
        devices,
        servers,
        network,
        slot_duration=confirmed["slot_duration_s"],
        subslot_count=int(provisional["subslot_count"]),
        time_slots=int(confirmed["time_slots"]),
        local_estimation_error=0.0,
        edge_estimation_error=0.0,
        route_rectangles=scenario.route_rectangles,
        world_size=scenario.world_size,
    )


def _build_fixed_tasks(devices: List[IndustrialDevice]) -> Dict[int, TaskDAG]:
    """Build identical five-subtask DAGs for each device."""
    task_dags = {}
    for device in devices:
        task_dag = TaskDAG(task_id=device.id, t_max=1.0, e_max=1.0)
        for subtask_id in range(1, 6):
            task_dag.add_subtask(
                Subtask(
                    subtask_id,
                    cpu_cycles=float(subtask_id) * 1e7,
                    data_size=float(subtask_id) * 1e5,
                    result_size=float(subtask_id) * 1e4,
                )
            )
        for predecessor, successor in TASK_EDGES:
            task_dag.add_dependency(predecessor, successor)
        task_dags[device.id] = task_dag
    return task_dags


def run_environment_episode(env: DITENEnv) -> Dict[str, float]:
    """Run one fixed all-local episode and return wall-clock metrics."""
    task_dags = _build_fixed_tasks(env.devices)
    priorities = {device.id: [1, 2, 3, 4, 5] for device in env.devices}
    start = time.perf_counter()
    env.reset_episode()
    for _ in range(env.time_slots):
        env.start_time_slot(task_dags, priorities)
        slot_done = False
        while not slot_done:
            _, _, _, info = env.step([0] * len(env.devices))
            slot_done = info["slot_done"]
    metrics = env.get_runtime_metrics()
    metrics["total_seconds"] = time.perf_counter() - start
    return metrics


def _print_profile(scenario_name: str) -> None:
    """Print cumulative cProfile output for one deterministic episode."""
    profiler = cProfile.Profile()
    profiler.enable()
    run_environment_episode(_build_environment(scenario_name))
    profiler.disable()
    output = io.StringIO()
    pstats.Stats(profiler, stream=output).sort_stats("cumulative").print_stats(20)
    print(output.getvalue())


def main() -> None:
    """Run the requested topology scaling benchmarks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenarios",
        nargs="+",
        choices=available_topology_scenario_names(),
        default=available_topology_scenario_names(),
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Print cProfile output for the final requested scenario.",
    )
    args = parser.parse_args()

    print(
        "scenario,total_s,window_s,requests,updates,samples,state_s,window_share_pct"
    )
    for scenario_name in args.scenarios:
        runs = [
            run_environment_episode(_build_environment(scenario_name))
            for _ in range(max(args.repeats, 1))
        ]
        median = {
            key: statistics.median(run[key] for run in runs) for key in runs[0]
        }
        window_share = (
            100.0 * median["connection_window_seconds"] / median["total_seconds"]
        )
        print(
            f"{scenario_name},{median['total_seconds']:.6f},"
            f"{median['connection_window_seconds']:.6f},"
            f"{median['connection_window_requests']:.0f},"
            f"{median['connection_window_updates']:.0f},"
            f"{median['connection_window_samples']:.0f},"
            f"{median['joint_state_seconds']:.6f},{window_share:.2f}"
        )

    if args.profile:
        _print_profile(args.scenarios[-1])


if __name__ == "__main__":
    main()
