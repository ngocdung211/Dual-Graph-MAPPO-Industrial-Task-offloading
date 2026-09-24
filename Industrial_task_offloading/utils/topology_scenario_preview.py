"""Preview candidate topology scenarios as PNG files and JSON metrics."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
from typing import Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from utils.topology_scenarios_config import (
    TopologyScenario,
    build_topology_scenarios,
    compute_topology_metrics,
    device_start_points,
    route_to_array,
)


def build_preview_scenarios(
    topology_seed: int = 2026,
    server_profile: Optional[str] = None,
) -> List[TopologyScenario]:
    """Build topology candidates requested for review."""
    return build_topology_scenarios(topology_seed, server_profile)


def _route_to_array(route: Sequence[Sequence[float]]) -> np.ndarray:
    """Return a closed route array."""
    return route_to_array(route)


def _device_start_points(scenario: TopologyScenario) -> np.ndarray:
    """Return one initial point for each device."""
    return device_start_points(scenario)


def _draw_direction_arrow(
    ax,
    start: np.ndarray,
    end: np.ndarray,
    label: str,
) -> None:
    """Draw one initial movement direction arrow."""
    direction = end - start
    distance = np.linalg.norm(direction)
    if distance <= 1e-9:
        return
    unit_direction = direction / distance
    arrow_start = start + unit_direction * min(2.2, distance * 0.18)
    arrow_end = start + unit_direction * min(8.0, distance * 0.45)
    ax.annotate(
        "",
        xy=(arrow_end[0], arrow_end[1]),
        xytext=(arrow_start[0], arrow_start[1]),
        arrowprops={
            "arrowstyle": "-|>",
            "color": "#d35400",
            "linewidth": 1.4,
            "mutation_scale": 10,
        },
        zorder=6,
    )
    label_point = start + unit_direction * min(9.0, distance * 0.52)
    ax.text(
        label_point[0],
        label_point[1],
        label,
        fontsize=6,
        color="#a04000",
        fontweight="bold",
        ha="center",
        va="center",
        zorder=7,
    )


def compute_scenario_metrics(scenario: TopologyScenario) -> Dict[str, object]:
    """Compute start-point and route-sampled connectivity metrics."""
    return compute_topology_metrics(scenario)


def plot_scenario(scenario: TopologyScenario, output_path: str) -> None:
    """Plot one paper-ready topology scenario to a PNG file."""
    server_locations = np.asarray(scenario.server_locations, dtype=float)
    start_points = _device_start_points(scenario)

    world_width, world_height = scenario.world_size
    fig_width = 8.5
    fig_height = max(5.5, fig_width * world_height / world_width)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=300)
    ax.set_xlim(0.0, world_width)
    ax.set_ylim(0.0, world_height)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks(np.arange(0.0, world_width + 1.0, 20.0))
    ax.set_yticks(np.arange(0.0, world_height + 1.0, 20.0))
    ax.set_xlabel("Factory floor x (m)")
    ax.set_ylabel("Factory floor y (m)")
    ax.set_title(
        f"{scenario.name} · {scenario.server_profile} coverage · "
        f"seed {scenario.topology_seed}"
    )
    ax.tick_params(axis="both", direction="out", labelsize=9, length=3.5, width=0.8)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)

    zone_colors = ("#566573", "#7d6608", "#6c5b7b")
    for route_index, route in enumerate(scenario.route_rectangles):
        route_array = _route_to_array(route)
        ax.plot(
            route_array[:, 0],
            route_array[:, 1],
            color=zone_colors[min(route_index // 5, len(zone_colors) - 1)],
            linewidth=1.15,
            alpha=0.62,
        )
        route_points = np.asarray(route, dtype=float)
        first_device_label = f"D{route_index * 2 + 1}"
        second_device_label = f"D{route_index * 2 + 2}"
        _draw_direction_arrow(
            ax, route_points[0], route_points[1], first_device_label
        )
        _draw_direction_arrow(
            ax, route_points[2], route_points[3], second_device_label
        )

    server_color = "#2878B5"
    for server_index, server_location in enumerate(server_locations, start=1):
        coverage_radius = scenario.coverage_radius_for_server(server_index - 1)
        circle = plt.Circle(
            server_location,
            coverage_radius,
            color=server_color,
            alpha=0.10,
            linewidth=1.0,
            fill=True,
        )
        outline = plt.Circle(
            server_location,
            coverage_radius,
            color=server_color,
            alpha=0.85,
            linewidth=1.1,
            fill=False,
        )
        ax.add_patch(circle)
        ax.add_patch(outline)
        ax.scatter(
            server_location[0],
            server_location[1],
            marker="s",
            s=70,
            color=server_color,
            edgecolors="black",
            linewidths=0.7,
            zorder=4,
        )
        ax.text(
            server_location[0],
            server_location[1] + 2.2,
            f"S{server_index} ({coverage_radius:.1f}m)",
            fontsize=8,
            fontweight="bold",
            ha="center",
            va="bottom",
            zorder=5,
        )

    ax.scatter(
        start_points[:, 0],
        start_points[:, 1],
        marker="o",
        s=34,
        color="#f39c12",
        edgecolors="black",
        linewidths=0.5,
        zorder=4,
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.tight_layout(pad=0.35)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def write_previews(
    output_dir: str,
    topology_seed: int = 2026,
    server_profile: Optional[str] = None,
    scenario_names: Optional[Sequence[str]] = None,
) -> Dict[str, object]:
    """Write PNG previews and one JSON metrics file."""
    os.makedirs(output_dir, exist_ok=True)
    metrics = {}
    scenarios = build_preview_scenarios(topology_seed, server_profile)
    if scenario_names is not None:
        selected_names = set(scenario_names)
        scenarios = [
            scenario for scenario in scenarios if scenario.name in selected_names
        ]
        missing_names = selected_names - {scenario.name for scenario in scenarios}
        if missing_names:
            raise ValueError(f"unknown preview scenarios: {sorted(missing_names)}")
    for scenario in scenarios:
        png_path = os.path.join(output_dir, f"{scenario.name}.png")
        plot_scenario(scenario, png_path)
        metrics[scenario.name] = compute_scenario_metrics(scenario)

    metrics_path = os.path.join(output_dir, "topology_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as metrics_file:
        json.dump(metrics, metrics_file, indent=2, ensure_ascii=True)
        metrics_file.write("\n")
    return {"output_dir": output_dir, "metrics_path": metrics_path, "metrics": metrics}


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Plot candidate topology scenarios for review."
    )
    parser.add_argument(
        "--output-dir",
        # default=os.path.join("results", "topology_preview"),
        default="archive/topology_preview",
        help="Directory for generated PNG and JSON files.",
    )
    parser.add_argument(
        "--topology-seed",
        type=int,
        default=2026,
        help="Seed for reproducible per-server coverage sampling.",
    )
    parser.add_argument(
        "--server-profile",
        choices=("scenario", "uniform", "heterogeneous", "stress"),
        default="scenario",
        help="Coverage profile used by generated previews.",
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=None,
        help="Optional subset of named topology scenarios to render.",
    )
    args = parser.parse_args()
    result = write_previews(
        args.output_dir,
        topology_seed=args.topology_seed,
        server_profile=(
            None if args.server_profile == "scenario" else args.server_profile
        ),
        scenario_names=args.scenarios,
    )
    print(json.dumps(result, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
