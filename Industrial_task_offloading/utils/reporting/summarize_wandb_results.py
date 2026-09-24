"""Export and summarize paper-style training histories from W&B."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence


HISTORY_KEYS = {
    "episode": "episode",
    "performance/reward": "reward",
    "performance/delay_seconds": "delay_s",
    "performance/energy_joules": "energy_j",
    "actions/requested_local_count": "requested_local_count",
    "actions/requested_edge_count": "requested_edge_count",
    "actions/resolved_local_count": "resolved_local_count",
    "actions/resolved_edge_count": "resolved_edge_count",
    "actions/edge_ratio_percent": "requested_edge_ratio_percent",
    "penalty/count": "penalty_count",
}

DEFAULT_MODELS = (
    "e-ATN-MADDPG",
    "MAPPO",
    "Graph-GAT MAPPO",
    "Graph-GAT Warmup MAPPO",
)
DEFAULT_TOPOLOGIES = (
    "paper_10d_3s",
    "medium_20d_6s",
    "large_30d_9s",
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Export complete W&B histories and summarize all/final training "
            "episodes for one comparison group."
        )
    )
    parser.add_argument(
        "--project-path",
        default="ObjectPromptDA/industrial-task-offloading",
        help="W&B entity/project path.",
    )
    parser.add_argument(
        "--source",
        choices=("local", "api"),
        default="local",
        help="Read local .wandb files by default, or request histories via API.",
    )
    parser.add_argument(
        "--wandb-dir",
        type=Path,
        default=Path("wandb"),
        help="Directory containing local W&B run folders.",
    )
    parser.add_argument("--group", required=True, help="W&B comparison group.")
    parser.add_argument("--seed", type=int, required=True, help="Training seed.")
    parser.add_argument(
        "--expected-episodes",
        type=int,
        default=1000,
        help="Required number of logged episodes per run.",
    )
    parser.add_argument(
        "--final-window",
        type=int,
        default=100,
        help="Number of final episodes in the convergence summary.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for exported histories and summaries.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(DEFAULT_MODELS),
        help="Exact model names expected in the group, in report order.",
    )
    parser.add_argument(
        "--topologies",
        nargs="+",
        default=list(DEFAULT_TOPOLOGIES),
        help="Exact topology scenario names expected, in report order.",
    )
    return parser.parse_args()


def _topology_metadata(config: Mapping[str, object]) -> Dict[str, float]:
    """Return flattened topology diagnostics from a W&B run config."""
    topology_metrics = config.get("topology_metrics", {})
    if not isinstance(topology_metrics, Mapping):
        return {}
    route_samples = topology_metrics.get("route_samples", {})
    physical_compute = topology_metrics.get("physical_compute", {})
    if not isinstance(route_samples, Mapping):
        route_samples = {}
    if not isinstance(physical_compute, Mapping):
        physical_compute = {}
    return {
        "topology_avg_feasible_servers": float(
            route_samples.get("avg_feasible_servers", 0.0)
        ),
        "topology_zero_link_ratio": float(
            route_samples.get("zero_link_ratio", 0.0)
        ),
        "actual_device_compute_power_mean_ghz": float(
            physical_compute.get("actual_device_compute_power_mean_ghz", 0.0)
        ),
        "actual_server_compute_power_mean_ghz": float(
            physical_compute.get("actual_server_compute_power_mean_ghz", 0.0)
        ),
    }


def normalize_history_row(
    raw_row: Mapping[str, object],
    *,
    run_id: str,
    run_url: str,
    model: str,
    topology: str,
    seed: int,
    topology_metadata: Mapping[str, float],
) -> Dict[str, object]:
    """Normalize one W&B history row into stable local field names."""
    row: Dict[str, object] = {
        "run_id": run_id,
        "run_url": run_url,
        "model": model,
        "topology": topology,
        "seed": seed,
        **topology_metadata,
    }
    for wandb_key, output_key in HISTORY_KEYS.items():
        value = raw_row.get(wandb_key)
        if value is None:
            raise ValueError(f"run {run_id} is missing history key {wandb_key}")
        row[output_key] = int(value) if output_key == "episode" else float(value)

    resolved_total = float(row["resolved_local_count"]) + float(
        row["resolved_edge_count"]
    )
    row["resolved_edge_ratio_percent"] = (
        100.0 * float(row["resolved_edge_count"]) / max(resolved_total, 1.0)
    )
    requested_edge = float(row["requested_edge_count"])
    row["rejected_edge_ratio_percent"] = (
        100.0 * float(row["penalty_count"]) / max(requested_edge, 1.0)
    )
    return row


def run_episodes_are_complete(
    rows: Sequence[Mapping[str, object]], expected_episodes: int
) -> bool:
    """Return whether rows hold one complete, unique 1..N episode sequence."""
    episodes = sorted(int(row["episode"]) for row in rows)
    return episodes == list(range(1, expected_episodes + 1))


def select_complete_run(
    candidate_runs: Sequence[Sequence[Mapping[str, object]]],
    combination: tuple,
    expected_episodes: int,
) -> List[Dict[str, object]]:
    """Return the single complete run for one topology/model combination.

    Interrupted or resumed W&B runs leave partial histories next to the
    finished one. Those are reported and skipped; two complete runs remain an
    error because the intended run would then be ambiguous.

    Args:
        candidate_runs: Row lists for every run matching the combination.
        combination: The `(topology, model)` pair being resolved.
        expected_episodes: Required number of logged episodes.

    Returns:
        Rows of the one complete run.

    Raises:
        ValueError: If no run or more than one run is complete.
    """
    complete_runs = [
        rows
        for rows in candidate_runs
        if run_episodes_are_complete(rows, expected_episodes)
    ]
    skipped = len(candidate_runs) - len(complete_runs)
    if skipped:
        print(
            f"skipped {skipped} incomplete run(s) for {combination}: "
            f"{[len(rows) for rows in candidate_runs if rows not in complete_runs]}"
            f" of {expected_episodes} episodes"
        )
    if not complete_runs:
        validate_run_episodes(candidate_runs[0], expected_episodes)
    if len(complete_runs) > 1:
        raise ValueError(f"duplicate complete W&B run for {combination}")
    return list(complete_runs[0])


def validate_run_episodes(
    rows: Sequence[Mapping[str, object]], expected_episodes: int
) -> None:
    """Require one complete, unique 1..N episode sequence."""
    episodes = [int(row["episode"]) for row in rows]
    expected = list(range(1, expected_episodes + 1))
    if sorted(episodes) != expected:
        missing = sorted(set(expected) - set(episodes))
        duplicates = sorted(
            episode for episode in set(episodes) if episodes.count(episode) > 1
        )
        raise ValueError(
            "incomplete history: "
            f"expected {expected_episodes} episodes, got {len(episodes)}, "
            f"missing={missing[:10]}, duplicates={duplicates[:10]}"
        )


def fetch_api_histories(
    project_path: str,
    group: str,
    seed: int,
    expected_episodes: int,
    models: Sequence[str] = DEFAULT_MODELS,
    topologies: Sequence[str] = DEFAULT_TOPOLOGIES,
) -> List[Dict[str, object]]:
    """Fetch and validate all runs in one W&B comparison group."""
    import wandb

    api = wandb.Api(timeout=60)
    runs = api.runs(project_path, filters={"group": group})
    rows: List[Dict[str, object]] = []
    seen_combinations = set()
    history_keys = list(HISTORY_KEYS)

    for run in runs:
        config = dict(run.config)
        if int(config.get("seed", -1)) != seed:
            continue
        model = str(config.get("algorithm", ""))
        topology = str(config.get("topology_scenario", ""))
        if model not in models or topology not in topologies:
            continue
        combination = (topology, model)
        if combination in seen_combinations:
            raise ValueError(f"duplicate W&B run for {topology} / {model}")

        raw_history = list(
            run.scan_history(keys=history_keys, page_size=expected_episodes)
        )
        metadata = _topology_metadata(config)
        run_rows = [
            normalize_history_row(
                raw_row,
                run_id=run.id,
                run_url=run.url,
                model=model,
                topology=topology,
                seed=seed,
                topology_metadata=metadata,
            )
            for raw_row in raw_history
        ]
        validate_run_episodes(run_rows, expected_episodes)
        rows.extend(run_rows)
        seen_combinations.add(combination)

    expected_combinations = {
        (topology, model) for topology in topologies for model in models
    }
    missing_combinations = sorted(expected_combinations - seen_combinations)
    if missing_combinations:
        raise ValueError(f"missing W&B runs: {missing_combinations}")
    return sorted(
        rows,
        key=lambda row: (
            list(topologies).index(str(row["topology"])),
            list(models).index(str(row["model"])),
            int(row["episode"]),
        ),
    )


def _decode_run_config(run_record: object) -> Dict[str, object]:
    """Decode the JSON-valued config stored in a W&B run record."""
    return {
        update.key: json.loads(update.value_json)
        for update in run_record.config.update
    }


def _decode_history_record(history_record: object) -> Dict[str, object]:
    """Decode one W&B protobuf history record."""
    row = {}
    for item in history_record.item:
        if item.key:
            key = item.key
        else:
            key = "/".join(item.nested_key)
        row[key] = json.loads(item.value_json)
    return row


def _read_local_run(
    wandb_path: Path,
    *,
    project_path: str,
    group: str,
    seed: int,
    models: Sequence[str],
    topologies: Sequence[str],
) -> tuple[tuple[str, str] | None, List[Dict[str, object]]]:
    """Read one matching local W&B run, returning its key and history rows."""
    from wandb.proto import wandb_internal_pb2
    from wandb.sdk.internal.datastore import DataStore

    datastore = DataStore()
    datastore.open_for_scan(str(wandb_path))
    combination = None
    normalized_rows: List[Dict[str, object]] = []
    run_id = ""
    run_url = ""
    model = ""
    topology = ""
    topology_metadata: Dict[str, float] = {}

    while True:
        record_bytes = datastore.scan_data()
        if record_bytes is None:
            break
        record = wandb_internal_pb2.Record()
        record.ParseFromString(record_bytes)
        if record.HasField("run"):
            run_config = _decode_run_config(record.run)
            run_seed = int(run_config.get("seed", -1))
            if record.run.run_group != group or run_seed != seed:
                return None, []
            model = str(run_config.get("algorithm", ""))
            topology = str(run_config.get("topology_scenario", ""))
            if model not in models or topology not in topologies:
                return None, []
            run_id = record.run.run_id
            run_url = f"https://wandb.ai/{project_path}/runs/{run_id}"
            topology_metadata = _topology_metadata(run_config)
            combination = (topology, model)
        elif record.HasField("history") and combination is not None:
            raw_row = _decode_history_record(record.history)
            if all(key in raw_row for key in HISTORY_KEYS):
                normalized_rows.append(
                    normalize_history_row(
                        raw_row,
                        run_id=run_id,
                        run_url=run_url,
                        model=model,
                        topology=topology,
                        seed=seed,
                        topology_metadata=topology_metadata,
                    )
                )
    return combination, normalized_rows


def fetch_local_histories(
    wandb_dir: Path,
    project_path: str,
    group: str,
    seed: int,
    expected_episodes: int,
    models: Sequence[str] = DEFAULT_MODELS,
    topologies: Sequence[str] = DEFAULT_TOPOLOGIES,
) -> List[Dict[str, object]]:
    """Read and validate histories from local W&B binary run files."""
    candidates: Dict[tuple, List[List[Dict[str, object]]]] = defaultdict(list)
    for wandb_path in sorted(wandb_dir.glob("run-*/run-*.wandb")):
        combination, run_rows = _read_local_run(
            wandb_path,
            project_path=project_path,
            group=group,
            seed=seed,
            models=models,
            topologies=topologies,
        )
        if combination is None:
            continue
        candidates[combination].append(run_rows)

    rows: List[Dict[str, object]] = []
    for combination, candidate_runs in candidates.items():
        rows.extend(
            select_complete_run(candidate_runs, combination, expected_episodes)
        )

    expected_combinations = {
        (topology, model) for topology in topologies for model in models
    }
    missing_combinations = sorted(expected_combinations - set(candidates))
    if missing_combinations:
        raise ValueError(f"missing local W&B runs: {missing_combinations}")
    return sorted(
        rows,
        key=lambda row: (
            list(topologies).index(str(row["topology"])),
            list(models).index(str(row["model"])),
            int(row["episode"]),
        ),
    )


def _mean(values: Iterable[float]) -> float:
    """Return the arithmetic mean of a nonempty iterable."""
    return statistics.fmean(values)


def _population_std(values: Sequence[float]) -> float:
    """Return population standard deviation, or zero for one value."""
    return statistics.pstdev(values) if len(values) > 1 else 0.0


def summarize_rows(
    rows: Sequence[Mapping[str, object]],
    window_name: str,
    models: Sequence[str] = DEFAULT_MODELS,
    topologies: Sequence[str] = DEFAULT_TOPOLOGIES,
) -> List[Dict[str, object]]:
    """Aggregate episode histories by topology and model."""
    grouped: Dict[tuple[str, str], List[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["topology"]), str(row["model"]))].append(row)

    summary_rows = []
    for topology in topologies:
        for model in models:
            group_rows = grouped[(topology, model)]
            if not group_rows:
                continue
            rewards = [float(row["reward"]) for row in group_rows]
            delays = [float(row["delay_s"]) for row in group_rows]
            energies = [float(row["energy_j"]) for row in group_rows]
            requested_local = sum(
                float(row["requested_local_count"]) for row in group_rows
            )
            requested_edge = sum(
                float(row["requested_edge_count"]) for row in group_rows
            )
            resolved_local = sum(
                float(row["resolved_local_count"]) for row in group_rows
            )
            resolved_edge = sum(
                float(row["resolved_edge_count"]) for row in group_rows
            )
            penalty_count = sum(float(row["penalty_count"]) for row in group_rows)
            summary_rows.append(
                {
                    "window": window_name,
                    "topology": topology,
                    "model": model,
                    "seed": int(group_rows[0]["seed"]),
                    "episode_count": len(group_rows),
                    "reward_mean": _mean(rewards),
                    "reward_std": _population_std(rewards),
                    "delay_mean_s": _mean(delays),
                    "delay_std_s": _population_std(delays),
                    "energy_mean_j": _mean(energies),
                    "energy_std_j": _population_std(energies),
                    "requested_edge_ratio_percent": (
                        100.0
                        * requested_edge
                        / max(requested_local + requested_edge, 1.0)
                    ),
                    "resolved_edge_ratio_percent": (
                        100.0
                        * resolved_edge
                        / max(resolved_local + resolved_edge, 1.0)
                    ),
                    "rejected_edge_ratio_percent": (
                        100.0 * penalty_count / max(requested_edge, 1.0)
                    ),
                    "penalty_count_mean": penalty_count / len(group_rows),
                    "topology_avg_feasible_servers": float(
                        group_rows[0]["topology_avg_feasible_servers"]
                    ),
                    "topology_zero_link_ratio": float(
                        group_rows[0]["topology_zero_link_ratio"]
                    ),
                    "actual_device_compute_power_mean_ghz": float(
                        group_rows[0]["actual_device_compute_power_mean_ghz"]
                    ),
                    "actual_server_compute_power_mean_ghz": float(
                        group_rows[0]["actual_server_compute_power_mean_ghz"]
                    ),
                }
            )
    return summary_rows


def build_summaries(
    rows: Sequence[Mapping[str, object]],
    final_window: int,
    models: Sequence[str] = DEFAULT_MODELS,
    topologies: Sequence[str] = DEFAULT_TOPOLOGIES,
) -> List[Dict[str, object]]:
    """Return all-episode and final-window summary rows."""
    max_episode = max(int(row["episode"]) for row in rows)
    final_rows = [
        row for row in rows if int(row["episode"]) > max_episode - final_window
    ]
    return summarize_rows(
        rows, "all_episodes", models, topologies
    ) + summarize_rows(
        final_rows, f"final_{final_window}", models, topologies
    )


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    """Write dictionaries as JSON Lines."""
    with path.open("w", encoding="utf-8") as output_file:
        for row in rows:
            output_file.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    """Write dictionaries as CSV."""
    if not rows:
        raise ValueError("cannot write an empty CSV")
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _format_mean_std(mean_value: float, std_value: float) -> str:
    """Format a compact mean plus/minus standard deviation value."""
    return f"{mean_value:.4f} +/- {std_value:.4f}"


def _write_markdown(path: Path, summary_rows: Sequence[Mapping[str, object]]) -> None:
    """Write human-readable all-episode and final-window summary tables."""
    windows = []
    for row in summary_rows:
        window = str(row["window"])
        if window not in windows:
            windows.append(window)

    lines = [
        "# Three-Topology Training Summary",
        "",
        "These are stochastic training metrics, matching the baseline paper's ",
        "episode-based plots. Standard deviation is across episodes within one ",
        "training seed, not across independent seeds.",
        "",
        "## Environment snapshot",
        "",
        "| Topology | Avg. feasible servers | Zero-link ratio | Device CPU (GHz) | Server CPU (GHz) |",
        "|---|---:|---:|---:|---:|",
    ]
    seen_topologies = set()
    for row in summary_rows:
        topology = str(row["topology"])
        if topology in seen_topologies:
            continue
        lines.append(
            "| {topology} | {feasible:.4f} | {zero_link:.2f}% | "
            "{device_cpu:.4f} | {server_cpu:.4f} |".format(
                topology=topology,
                feasible=float(row["topology_avg_feasible_servers"]),
                zero_link=100.0 * float(row["topology_zero_link_ratio"]),
                device_cpu=float(row["actual_device_compute_power_mean_ghz"]),
                server_cpu=float(row["actual_server_compute_power_mean_ghz"]),
            )
        )
        seen_topologies.add(topology)
    lines.append("")
    for window in windows:
        lines.extend(
            [
                f"## {window}",
                "",
                "| Topology | Model | Episodes | Reward | Delay (s) | Energy (J) | Edge requested | Edge executed | Rejected edge |",
                "|---|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in summary_rows:
            if row["window"] != window:
                continue
            lines.append(
                "| {topology} | {model} | {episode_count} | {reward} | {delay} | "
                "{energy} | {requested:.2f}% | {resolved:.2f}% | "
                "{rejected:.2f}% |".format(
                    topology=row["topology"],
                    model=row["model"],
                    episode_count=row["episode_count"],
                    reward=_format_mean_std(
                        float(row["reward_mean"]), float(row["reward_std"])
                    ),
                    delay=_format_mean_std(
                        float(row["delay_mean_s"]), float(row["delay_std_s"])
                    ),
                    energy=_format_mean_std(
                        float(row["energy_mean_j"]), float(row["energy_std_j"])
                    ),
                    requested=float(row["requested_edge_ratio_percent"]),
                    resolved=float(row["resolved_edge_ratio_percent"]),
                    rejected=float(row["rejected_edge_ratio_percent"]),
                )
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_outputs(
    output_dir: Path,
    history_rows: Sequence[Mapping[str, object]],
    summary_rows: Sequence[Mapping[str, object]],
) -> None:
    """Write complete histories and summaries in machine/human formats."""
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_dir / "training_history.jsonl", history_rows)
    _write_csv(output_dir / "training_history.csv", history_rows)
    _write_jsonl(output_dir / "summary.jsonl", summary_rows)
    _write_csv(output_dir / "summary.csv", summary_rows)
    _write_markdown(output_dir / "summary.md", summary_rows)


def main() -> None:
    """Export W&B histories and write aggregate result tables."""
    args = parse_args()
    if args.expected_episodes <= 0:
        raise ValueError("expected episodes must be positive")
    if not 0 < args.final_window <= args.expected_episodes:
        raise ValueError("final window must be between 1 and expected episodes")

    if args.source == "local":
        history_rows = fetch_local_histories(
            wandb_dir=args.wandb_dir,
            project_path=args.project_path,
            group=args.group,
            seed=args.seed,
            expected_episodes=args.expected_episodes,
            models=args.models,
            topologies=args.topologies,
        )
    else:
        history_rows = fetch_api_histories(
            project_path=args.project_path,
            group=args.group,
            seed=args.seed,
            expected_episodes=args.expected_episodes,
            models=args.models,
            topologies=args.topologies,
        )
    summary_rows = build_summaries(
        history_rows, args.final_window, args.models, args.topologies
    )
    write_outputs(args.output_dir, history_rows, summary_rows)
    print(
        f"Exported {len(history_rows)} episode rows and "
        f"{len(summary_rows)} summary rows to {args.output_dir}"
    )


if __name__ == "__main__":
    main()
