"""Named topology scenarios shared by preview plots and experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class TopologyScenario:
    """Topology layout for one experiment scenario."""

    name: str
    device_count: int
    server_locations: Sequence[Sequence[float]]
    coverage_radius: float
    route_rectangles: Sequence[Sequence[Sequence[float]]]
    world_size: Tuple[float, float] = (100.0, 100.0)
    coverage_radii: Tuple[float, ...] = ()
    topology_seed: int = 2026
    server_profile: str = "uniform"

    def coverage_radius_for_server(self, server_index: int) -> float:
        """Return the resolved radius for one zero-based server index."""
        if self.coverage_radii:
            return float(self.coverage_radii[server_index])
        return float(self.coverage_radius)


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


def large_industrial_routes() -> List[List[List[float]]]:
    """Return structured but non-uniform routes for a 180 m by 120 m plant."""
    return [
        rectangle(8.0, 10.0, 38.0, 29.0),
        rectangle(36.0, 13.0, 68.0, 34.0),
        rectangle(70.0, 9.0, 101.0, 31.0),
        rectangle(99.0, 14.0, 133.0, 36.0),
        rectangle(136.0, 11.0, 172.0, 33.0),
        rectangle(10.0, 44.0, 43.0, 66.0),
        rectangle(40.0, 41.0, 73.0, 69.0),
        rectangle(72.0, 45.0, 106.0, 67.0),
        rectangle(104.0, 42.0, 138.0, 71.0),
        rectangle(136.0, 46.0, 170.0, 68.0),
        rectangle(15.0, 80.0, 49.0, 105.0),
        rectangle(47.0, 76.0, 80.0, 109.0),
        rectangle(79.0, 82.0, 114.0, 106.0),
        rectangle(112.0, 77.0, 147.0, 110.0),
        rectangle(143.0, 81.0, 172.0, 104.0),
    ]


def modular_cell_routes() -> List[List[List[float]]]:
    """Return the approved modular layout; edit rectangle bounds in metres.

    Each rectangle(left, bottom, right, top) carries two devices starting at
    opposite corners. Route R1 carries D1/D2, R2 carries D3/D4, and so on.
    """
    return [
        # Module A: grouped cells, R1-R5 / D1-D10.
        rectangle(8.0, 16.0, 26.0, 36.0),
        rectangle(34.0, 18.0, 52.0, 40.0),
        rectangle(14.0, 50.0, 46.0, 70.0),
        rectangle(8.0, 82.0, 26.0, 104.0),
        rectangle(34.0, 84.0, 52.0, 102.0),
        # Module B: staggered horizontal cells, R6-R10 / D11-D20.
        rectangle(70.0, 14.0, 106.0, 26.0),
        rectangle(76.0, 32.0, 112.0, 44.0),
        rectangle(72.0, 50.0, 112.0, 66.0),
        rectangle(70.0, 72.0, 106.0, 84.0),
        rectangle(76.0, 90.0, 112.0, 104.0),
        # Module C: mixed vertical cells, R11-R15 / D21-D30.
        rectangle(128.0, 16.0, 146.0, 54.0),
        rectangle(154.0, 16.0, 172.0, 36.0),
        rectangle(154.0, 44.0, 172.0, 72.0),
        rectangle(128.0, 64.0, 146.0, 104.0),
        rectangle(154.0, 82.0, 172.0, 104.0),
    ]


def _coverage_radii(
    server_count: int,
    nominal_radius: float,
    profile: str,
    topology_seed: int,
) -> Tuple[float, ...]:
    """Resolve reproducible per-server radii for a coverage profile."""
    if profile == "uniform":
        return (float(nominal_radius),) * server_count

    rng = np.random.default_rng(topology_seed)
    if profile == "heterogeneous":
        lower, upper = (5.0 / 6.0) * nominal_radius, (10.0 / 9.0) * nominal_radius
    elif profile == "stress":
        lower, upper = (2.0 / 3.0) * nominal_radius, (5.0 / 6.0) * nominal_radius
    else:
        raise ValueError(
            "unknown server profile "
            f"'{profile}'. Valid: uniform, heterogeneous, stress"
        )
    sampled_radii = rng.uniform(lower, upper, server_count)
    return tuple(float(value) for value in np.round(sampled_radii, 2))


def build_topology_scenarios(
    topology_seed: int = 2026,
    server_profile: Optional[str] = None,
) -> List[TopologyScenario]:
    """Build supported topology scenarios."""
    definitions = [
        TopologyScenario(
            name="paper_10d_3s",
            device_count=10,
            server_locations=[[20.0, 30.0], [50.0, 50.0], [70.0, 20.0]],
            coverage_radius=12.0,
            route_rectangles=paper_routes(),
            topology_seed=topology_seed,
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
            topology_seed=topology_seed,
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
            topology_seed=topology_seed,
        ),
        TopologyScenario(
            name="large_industrial_30d_9s",
            device_count=30,
            server_locations=[
                [22.0, 16.0],
                [100.0, 36.0],
                [142.0, 76.0],
                [136.0, 40.0],
                [44.0, 72.0],
                [110.0, 72.0],
                [72.0, 28.0],
                [42.0, 34.0],
                [78.0, 68.0],
            ],
            coverage_radius=18.0,
            route_rectangles=large_industrial_routes(),
            world_size=(180.0, 120.0),
            topology_seed=topology_seed,
            server_profile="heterogeneous",
        ),
        TopologyScenario(
            name="modular_cells_30d_9s",
            device_count=30,
            # Edit [x, y] positions in metres; order determines server IDs.
            server_locations=[
                [30.0, 39.0],  # S1, module A.
                [30.0, 59.0],  # S2, module A centre.
                [30.0, 81.0],  # S3, module A.
                [85.0, 30.0],  # S4, module B.
                [85.0, 59.0],  # S5, module B centre.
                [85.0, 87.0],  # S6, module B.
                [150.0, 34.0],  # S7, module C.
                [150.0, 59.0],  # S8, module C centre.
                [150.0, 84.0],  # S9, module C.
            ],
            coverage_radius=12.0,  # Nominal radius for optional profile overrides.
            # Explicit radii take precedence for this scenario's default profile.
            # S1-S3, S4-S6, S7-S9; the middle server in each module is 15 m.
            coverage_radii=(
                12.0, 15.0, 12.0,
                12.0, 15.0, 12.0,
                12.0, 15.0, 12.0,
            ),
            route_rectangles=modular_cell_routes(),
            world_size=(180.0, 120.0),
            topology_seed=topology_seed,
            server_profile="heterogeneous",
        ),
    ]
    scenarios = []
    for definition in definitions:
        resolved_profile = server_profile or definition.server_profile
        # Keep manually configured radii when loading the default profile,
        # including checkpoint reconstruction with an explicit profile name.
        if (
            definition.coverage_radii
            and resolved_profile == definition.server_profile
        ):
            resolved_radii = definition.coverage_radii
        else:
            resolved_radii = _coverage_radii(
                len(definition.server_locations),
                definition.coverage_radius,
                resolved_profile,
                topology_seed,
            )
        scenarios.append(
            TopologyScenario(
                name=definition.name,
                device_count=definition.device_count,
                server_locations=definition.server_locations,
                coverage_radius=definition.coverage_radius,
                route_rectangles=definition.route_rectangles,
                world_size=definition.world_size,
                coverage_radii=resolved_radii,
                topology_seed=topology_seed,
                server_profile=resolved_profile,
            )
        )
    return scenarios


def available_topology_scenario_names() -> List[str]:
    """Return supported topology scenario names."""
    return [scenario.name for scenario in build_topology_scenarios()]


def get_topology_scenario(
    name: str,
    topology_seed: int = 2026,
    server_profile: Optional[str] = None,
) -> TopologyScenario:
    """Return a topology scenario by name."""
    for scenario in build_topology_scenarios(topology_seed, server_profile):
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
    points: np.ndarray,
    server_locations: np.ndarray,
    coverage_radii: Sequence[float],
) -> np.ndarray:
    """Count how many servers cover each point."""
    deltas = points[:, np.newaxis, :] - server_locations[np.newaxis, :, :]
    distances = np.linalg.norm(deltas, axis=2)
    radii = np.asarray(coverage_radii, dtype=float)
    return np.sum(distances <= radii[np.newaxis, :], axis=1)


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
        start_points, server_locations, scenario.coverage_radii
    )
    route_counts = connection_counts(
        route_points, server_locations, scenario.coverage_radii
    )
    coverage_radii = np.asarray(scenario.coverage_radii, dtype=float)
    return {
        "name": scenario.name,
        "num_devices": scenario.device_count,
        "num_servers": len(scenario.server_locations),
        "coverage_radius_m": float(np.mean(coverage_radii)),
        "coverage_radii_m": coverage_radii.tolist(),
        "coverage_radius_min_m": float(np.min(coverage_radii)),
        "coverage_radius_max_m": float(np.max(coverage_radii)),
        "world_size_m": list(scenario.world_size),
        "topology_seed": scenario.topology_seed,
        "server_profile": scenario.server_profile,
        "start_points": summarize_counts(
            start_counts, scenario.device_count, len(scenario.server_locations)
        ),
        "route_samples": summarize_counts(
            route_counts, scenario.device_count, len(scenario.server_locations)
        ),
    }
