"""Named topology scenarios shared by preview plots and experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np


@dataclass(frozen=True)
class TopologyScenario:
    """Topology layout for one experiment scenario."""

    name: str
    device_count: int
    server_locations: Sequence[Sequence[float]]
    coverage_radius: float
    route_rectangles: Sequence[Sequence[Sequence[float]]]


def rectangle(left: float, bottom: float, right: float, top: float) -> List[List[float]]:
    """Return rectangle route corners in clockwise order."""
    return [[left, bottom], [left, top], [right, top], [right, bottom]]


def paper_routes() -> List[List[List[float]]]:
    """Return the current 10-device route layout used by DITENEnv."""
    return [
        rectangle(10.0, 10.0, 30.0, 30.0),
        [[70.0, 10.0], [40.0, 10.0], [40.0, 20.0], [70.0, 20.0]],
        rectangle(40.0, 20.0, 60.0, 50.0),
        rectangle(70.0, 10.0, 90.0, 40.0),
        rectangle(65.0, 45.0, 90.0, 55.0),
    ]


def medium_routes() -> List[List[List[float]]]:
    """Return 10 route rectangles for 20 devices."""
    return [
        rectangle(8.0, 8.0, 30.0, 30.0),
        rectangle(24.0, 10.0, 48.0, 34.0),
        rectangle(40.0, 18.0, 62.0, 50.0),
        rectangle(58.0, 10.0, 84.0, 36.0),
        rectangle(66.0, 34.0, 92.0, 62.0),
        rectangle(10.0, 40.0, 36.0, 64.0),
        rectangle(30.0, 48.0, 58.0, 76.0),
        rectangle(54.0, 52.0, 84.0, 84.0),
        rectangle(18.0, 22.0, 46.0, 48.0),
        rectangle(48.0, 28.0, 78.0, 58.0),
    ]


def large_routes() -> List[List[List[float]]]:
    """Return 15 route rectangles for 30 devices."""
    return [
        rectangle(6.0, 8.0, 26.0, 28.0),
        rectangle(18.0, 14.0, 42.0, 38.0),
        rectangle(34.0, 12.0, 58.0, 36.0),
        rectangle(50.0, 10.0, 76.0, 34.0),
        rectangle(68.0, 12.0, 94.0, 38.0),
        rectangle(8.0, 34.0, 32.0, 58.0),
        rectangle(24.0, 38.0, 50.0, 64.0),
        rectangle(42.0, 38.0, 68.0, 66.0),
        rectangle(60.0, 36.0, 88.0, 64.0),
        rectangle(12.0, 58.0, 38.0, 86.0),
        rectangle(32.0, 60.0, 60.0, 90.0),
        rectangle(54.0, 62.0, 84.0, 92.0),
        rectangle(18.0, 24.0, 54.0, 54.0),
        rectangle(46.0, 24.0, 84.0, 54.0),
        rectangle(36.0, 44.0, 76.0, 78.0),
    ]


def build_topology_scenarios() -> List[TopologyScenario]:
    """Build supported topology scenarios."""
    return [
        TopologyScenario(
            name="paper_10d_3s",
            device_count=10,
            server_locations=[[20.0, 30.0], [50.0, 50.0], [70.0, 20.0]],
            coverage_radius=12.0,
            route_rectangles=paper_routes(),
        ),
        TopologyScenario(
            name="medium_20d_6s",
            device_count=20,
            server_locations=[
                [22.0, 26.0],
                [30.0, 50.0],
                [50.0, 42.0],
                [62.0, 42.0],
                [78.0, 32.0],
                [62.0, 66.0],
            ],
            coverage_radius=12.0,
            route_rectangles=medium_routes(),
        ),
        TopologyScenario(
            name="large_30d_9s",
            device_count=30,
            server_locations=[
                [18.0, 24.0],
                [22.0, 50.0],
                [43.0, 20.0],
                [55.0, 38.0],
                [68.0, 28.0],
                [80.0, 30.0],
                [70.0, 83.0],
                [82.0, 54.0],
                [56.0, 64.0],
            ],
            coverage_radius=12.0,
            route_rectangles=large_routes(),
        ),
    ]


def available_topology_scenario_names() -> List[str]:
    """Return supported topology scenario names."""
    return [scenario.name for scenario in build_topology_scenarios()]


def get_topology_scenario(name: str) -> TopologyScenario:
    """Return a topology scenario by name."""
    for scenario in build_topology_scenarios():
        if scenario.name == name:
            return scenario
    valid_names = ", ".join(available_topology_scenario_names())
    raise ValueError(f"unknown topology scenario '{name}'. Valid: {valid_names}")


def route_to_array(route: Sequence[Sequence[float]]) -> np.ndarray:
    """Return a closed route array."""
    route_array = np.asarray(route, dtype=float)
    return np.vstack([route_array, route_array[0]])


def device_start_points(scenario: TopologyScenario) -> np.ndarray:
    """Return one initial point for each device in a scenario."""
    starts = []
    for device_index in range(scenario.device_count):
        route = np.asarray(scenario.route_rectangles[device_index // 2], dtype=float)
        corner_index = 0 if device_index % 2 == 0 else 2
        starts.append(route[corner_index])
    return np.asarray(starts, dtype=float)


def route_waypoints_for_device(
    scenario: TopologyScenario, device_index: int
) -> List[np.ndarray]:
    """Return closed route waypoints for one zero-based device index."""
    route = [
        np.asarray(point, dtype=float).copy()
        for point in scenario.route_rectangles[device_index // 2]
    ]
    route.append(route[0].copy())
    return route


def sample_route_points(
    routes: Sequence[Sequence[Sequence[float]]], samples_per_segment: int = 12
) -> np.ndarray:
    """Sample points along every route edge."""
    points = []
    for route in routes:
        route_array = route_to_array(route)
        for start, end in zip(route_array[:-1], route_array[1:]):
            for sample_index in range(samples_per_segment):
                ratio = sample_index / float(samples_per_segment)
                points.append(start + ratio * (end - start))
    return np.asarray(points, dtype=float)


def connection_counts(
    points: np.ndarray, server_locations: np.ndarray, coverage_radius: float
) -> np.ndarray:
    """Count how many servers cover each point."""
    deltas = points[:, np.newaxis, :] - server_locations[np.newaxis, :, :]
    distances = np.linalg.norm(deltas, axis=2)
    return np.sum(distances <= coverage_radius, axis=1)


def summarize_counts(
    counts: np.ndarray, device_count: int, server_count: int
) -> Dict[str, float]:
    """Build topology metrics from per-point feasible-server counts."""
    return {
        "avg_feasible_servers": float(np.mean(counts)),
        "density": float(np.mean(counts) / max(server_count, 1)),
        "zero_link_ratio": float(np.mean(counts == 0)),
        "multi_link_ratio": float(np.mean(counts >= 2)),
        "device_server_ratio": float(device_count / max(server_count, 1)),
    }


def compute_topology_metrics(scenario: TopologyScenario) -> Dict[str, object]:
    """Compute start-point and route-sampled connectivity metrics."""
    server_locations = np.asarray(scenario.server_locations, dtype=float)
    start_points = device_start_points(scenario)
    route_points = sample_route_points(scenario.route_rectangles)
    start_counts = connection_counts(
        start_points, server_locations, scenario.coverage_radius
    )
    route_counts = connection_counts(
        route_points, server_locations, scenario.coverage_radius
    )
    return {
        "name": scenario.name,
        "num_devices": scenario.device_count,
        "num_servers": len(scenario.server_locations),
        "coverage_radius_m": scenario.coverage_radius,
        "start_points": summarize_counts(
            start_counts, scenario.device_count, len(scenario.server_locations)
        ),
        "route_samples": summarize_counts(
            route_counts, scenario.device_count, len(scenario.server_locations)
        ),
    }
