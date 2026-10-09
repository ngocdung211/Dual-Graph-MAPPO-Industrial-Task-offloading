"""Export subtask offload counts and plot matched training histories.

Read complete W&B histories, or replot the exported CSV without network access.
Successful offloads are resolved edge executions. Rejections are penalty events
under DITEN's strict connection-window fallback, not failures after execution.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from utils.reporting.plot_combined_runs import moving_average
from utils.reporting.summarize_wandb_results import (
    HISTORY_KEYS,
    normalize_history_row,
    select_complete_run,
    validate_run_episodes,
)

MODELS = (
    "e-ATN-MADDPG", "Shared MAPPO", "GATMA-Adapted", "Graph-GAT Mask MAPPO"
)
MODEL_CHOICES = (*MODELS, "Shared Mask MAPPO")
LABELS = {
    "e-ATN-MADDPG": "e-ATN-MADDPG",
    "Shared MAPPO": "MAPPO",
    "GATMA-Adapted": "GATMA",
    "Graph-GAT Mask MAPPO": "TGM-MAPPO",
    "Shared Mask MAPPO": "Mask MAPPO",
}
COLORS = {
    "e-ATN-MADDPG": "#4CA8A0",
    "Shared MAPPO": "#F58518",
    "GATMA-Adapted": "#4C78A8",
    "Graph-GAT Mask MAPPO": "#E45756",
    "Shared Mask MAPPO": "#54A24B",
}
COUNT_FIELDS = (
    "requested_edge_count", "successful_edge_count", "rejected_edge_count"
)
RATE_FIELDS = ("success_rate_percent", "rejection_rate_percent")


def parse_args() -> argparse.Namespace:
    """Parse source, matched-run selection, and reporting settings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-path", default="ObjectPromptDA/industrial-task-offloading"
    )
    parser.add_argument(
        "--group-template", help="W&B group with a {seed} placeholder."
    )
    parser.add_argument(
        "--gatma-group-template", help="Separate GATMA v3 group with {seed}."
    )
    parser.add_argument(
        "--history-csv", type=Path, nargs="+", help="CSV exports with counts."
    )
    parser.add_argument(
        "--run-ids", nargs="+",
        help="Explicit W&B IDs when groups contain repeated runs.",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[190, 191, 192])
    parser.add_argument(
        "--models", nargs="+", choices=MODEL_CHOICES, default=list(MODELS)
    )
    parser.add_argument("--topology", default="modular_cells_30d_9s")
    parser.add_argument("--expected-episodes", type=int, default=1000)
    parser.add_argument("--final-window", type=int, default=100)
    parser.add_argument("--smooth-window", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if (
        args.expected_episodes <= 0
        or not 0 < args.final_window <= args.expected_episodes
    ):
        parser.error("Require 0 < final-window <= expected-episodes.")
    if args.smooth_window <= 0:
        parser.error("smooth-window must be positive.")
    if (
        len(set(args.seeds)) != len(args.seeds)
        or len(set(args.models)) != len(args.models)
    ):
        parser.error("seeds and models must be unique.")
    if args.history_csv:
        if args.group_template or args.gatma_group_template or args.run_ids:
            parser.error("Use history-csv or W&B group templates, not both.")
    elif not args.group_template:
        parser.error("Provide group-template or history-csv.")
    return args


def load_wandb_counts(
    args: argparse.Namespace,
) -> tuple[List[dict], List[dict]]:
    """Fetch full histories with GATMA v3 separate from old baselines."""
    import wandb

    api = wandb.Api(timeout=30)
    histories = []
    sources = []
    for seed in args.seeds:
        groups = defaultdict(list)
        for model in args.models:
            template = (
                args.gatma_group_template
                if model == "GATMA-Adapted" and args.gatma_group_template
                else args.group_template
            )
            groups[template.format(seed=seed)].append(model)
        for group, models in groups.items():
            candidates = defaultdict(list)
            metadata = {}
            filters = {
                "group": group,
                "config.seed": seed,
                "config.topology_scenario": args.topology,
                "config.algorithm": {"$in": models},
            }
            for run in api.runs(args.project_path, filters=filters):
                if args.run_ids and run.id not in args.run_ids:
                    continue
                config = dict(run.config)
                model = str(config["algorithm"])
                version = config.get("agent_kwargs", {}).get(
                    "adaptation_version"
                )
                if model == "GATMA-Adapted" and version != 3:
                    raise ValueError(
                        f"GATMA run {run.id} is not recorded as v3."
                    )
                # Read every logged episode rather than sampled history().
                rows = [
                    normalize_history_row(
                        raw, run_id=run.id, run_url=run.url, model=model,
                        topology=args.topology, seed=seed, topology_metadata={},
                    )
                    for raw in run.scan_history(
                        keys=list(HISTORY_KEYS), page_size=1000
                    )
                ]
                for row in rows:
                    row.update({
                        "group": group, "adaptation_version": version or ""
                    })
                candidates[model].append(rows)
                metadata[run.id] = {
                    "run_id": run.id, "run_url": run.url, "config": config
                }
            for model in models:
                if not candidates[model]:
                    raise ValueError(
                        f"No run for {model}, seed {seed}, group {group}."
                    )
                selected = select_complete_run(
                    candidates[model], (model, seed), args.expected_episodes
                )
                histories.extend(selected)
                sources.append(metadata[selected[0]["run_id"]])
                print(
                    f"Loaded {model}, seed {seed}: {len(selected)} episodes",
                    flush=True,
                )
    return histories, sources


def load_count_csvs(paths: Sequence[Path]) -> tuple[List[dict], List[dict]]:
    """Read count exports, rejecting CSVs that only contain performance."""
    rows = []
    sources = []
    required = {
        "model", "seed", "episode", "requested_edge_count",
        "resolved_edge_count", "penalty_count",
    }
    for path in paths:
        with path.open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            missing = required - set(reader.fieldnames or [])
            if missing:
                raise ValueError(
                    f"{path} lacks counts/metadata: {sorted(missing)}. "
                    "Export from W&B first."
                )
            for original in reader:
                row = dict(original)
                row["topology"] = (
                    row.get("topology") or row.get("topology_scenario", "")
                )
                rows.append(row)
        sources.append({
            "path": str(path.resolve()),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        })
    return rows, sources


def derive_counts(row: Mapping[str, object]) -> Dict[str, object]:
    """Validate count conservation and derive strict-fallback metrics."""
    result = dict(row)
    for name in (
        "requested_edge_count", "resolved_edge_count", "penalty_count"
    ):
        value = float(row[name])
        if not math.isfinite(value) or value < 0 or not value.is_integer():
            raise ValueError(
                f"Invalid {name}={value}; expected a nonnegative integer."
            )
        result[name] = int(value)
    requested = result["requested_edge_count"]
    successful = result["resolved_edge_count"]
    rejected = result["penalty_count"]
    if requested != successful + rejected:
        raise ValueError(
            "Counts require requested edge = resolved edge + penalties "
            "(strict rejection fallback)."
        )
    result["successful_edge_count"] = successful
    result["rejected_edge_count"] = rejected
    # No edge requests means an undefined rate, not 0% or 100% success.
    result["success_rate_percent"] = (
        100 * successful / requested if requested else math.nan
    )
    result["rejection_rate_percent"] = (
        100 * rejected / requested if requested else math.nan
    )
    return result


def validate_histories(
    rows: Sequence[Mapping[str, object]], models: Sequence[str],
    seeds: Sequence[int],
    topology: str, expected_episodes: int,
) -> List[dict]:
    """Require complete, unique model/seed histories and GATMA v3."""
    grouped = defaultdict(list)
    for row in rows:
        model, seed = str(row["model"]), int(row["seed"])
        if (
            model not in models or seed not in seeds
            or row["topology"] != topology
        ):
            continue
        if (
            model == "GATMA-Adapted"
            and str(row.get("adaptation_version")) != "3"
        ):
            raise ValueError(
                "Count CSV must record adaptation_version=3 for GATMA."
            )
        episode = float(row["episode"])
        if not episode.is_integer():
            raise ValueError("Episode numbers must be integers.")
        normalized = derive_counts(row)
        normalized.update({"seed": seed, "episode": int(episode)})
        grouped[(model, seed)].append(normalized)
    result = []
    for model in models:
        for seed in seeds:
            history = grouped[(model, seed)]
            validate_run_episodes(history, expected_episodes)
            result.extend(sorted(history, key=lambda row: row["episode"]))
    return result


def summarize_counts(
    rows: Sequence[Mapping[str, object]], models: Sequence[str],
    seeds: Sequence[int],
    final_window: int,
) -> tuple[List[dict], List[dict]]:
    """Report count averages and request-weighted rates across seeds."""
    per_seed = []
    for model in models:
        for seed in seeds:
            history = sorted(
                [r for r in rows if r["model"] == model and r["seed"] == seed],
                key=lambda r: r["episode"],
            )[-final_window:]
            entry = {"model": model, "seed": seed, "final_window": final_window}
            for field in COUNT_FIELDS:
                entry[field] = statistics.fmean(r[field] for r in history)
            requested = sum(r["requested_edge_count"] for r in history)
            # Weight window rates by requests rather than by episodes.
            for field, count_field in zip(RATE_FIELDS, COUNT_FIELDS[1:]):
                entry[field] = (
                    100 * sum(r[count_field] for r in history) / requested
                    if requested else math.nan
                )
            per_seed.append(entry)
    summary = []
    for model in models:
        entries = [row for row in per_seed if row["model"] == model]
        entry = {
            "model": model, "seed_count": len(entries),
            "seeds": ";".join(map(str, seeds)), "final_window": final_window,
        }
        for field in (*COUNT_FIELDS, *RATE_FIELDS):
            values = [
                row[field] for row in entries if math.isfinite(row[field])
            ]
            entry[field + "_mean"] = (
                statistics.fmean(values) if values else math.nan
            )
            entry[field + "_sd"] = (
                statistics.stdev(values) if len(values) > 1
                else (0.0 if values else math.nan)
            )
        entry["rate_seed_count"] = sum(
            math.isfinite(row["success_rate_percent"]) for row in entries
        )
        summary.append(entry)
    return per_seed, summary


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    """Write stable count exports and summary tables."""
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def save_figure(figure: plt.Figure, output_dir: Path, name: str) -> None:
    """Export both a vector PDF and a high-resolution PNG."""
    figure.tight_layout()
    for extension in ("png", "pdf"):
        figure.savefig(
            output_dir / f"{name}.{extension}", dpi=300, bbox_inches="tight"
        )
    plt.close(figure)


def plot_counts(
    rows: Sequence[Mapping[str, object]],
    summary: Sequence[Mapping[str, object]], models: Sequence[str],
    seeds: Sequence[int], smooth_window: int, output_dir: Path,
) -> None:
    """Plot training count curves and final-window count/rate comparisons."""
    plt.rcParams.update({
        "font.family": "serif", "font.size": 9, "pdf.fonttype": 42
    })
    figure, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for axis, field, title in zip(
        axes, COUNT_FIELDS[1:],
        ("Successful edge executions", "Rejected edge requests"),
    ):
        for model in models:
            color = COLORS[model]
            curves = []
            for seed in seeds:
                history = [
                    r for r in rows
                    if r["model"] == model and r["seed"] == seed
                ]
                curves.append(moving_average(
                    np.asarray([r[field] for r in history]), smooth_window
                ))
            curves = np.asarray(curves)
            mean = curves.mean(axis=0)
            sd = (
                curves.std(axis=0, ddof=1)
                if len(seeds) > 1 else np.zeros_like(mean)
            )
            episodes = np.arange(1, len(mean) + 1)
            axis.plot(episodes, mean, color=color, label=LABELS[model])
            axis.fill_between(
                episodes, mean - sd, mean + sd, color=color, alpha=0.15
            )
        axis.set(
            title=title, xlabel="Training episode",
            ylabel="Subtasks per episode",
        )
        axis.grid(alpha=0.2)
    axes[0].legend(frameon=False, fontsize=8)
    save_figure(figure, output_dir, "offloading_count_curves")

    for fields, legends, ylabel, name in (
        (
            COUNT_FIELDS, ("Requested", "Successful", "Rejected"),
            "Subtasks per episode", "final_offloading_counts",
        ),
        (
            RATE_FIELDS, ("Success", "Rejection"),
            "Percentage of edge requests (%)", "final_offloading_rates",
        ),
    ):
        figure, axis = plt.subplots(
            figsize=(max(6.6, 1.5 * len(models)), 3.8)
        )
        centers = np.arange(len(models))
        width = 0.8 / len(fields)
        for index, (field, legend) in enumerate(zip(fields, legends)):
            means = [row[field + "_mean"] for row in summary]
            deviations = [row[field + "_sd"] for row in summary]
            positions = centers + (index - (len(fields) - 1) / 2) * width
            bars = axis.bar(
                positions, np.nan_to_num(means), width,
                yerr=np.nan_to_num(deviations), capsize=3, label=legend,
            )
            for bar, mean in zip(bars, means):
                if not math.isfinite(mean):
                    axis.text(
                        bar.get_x() + width / 2, 1, "N/A",
                        ha="center", fontsize=7,
                    )
        axis.set_xticks(centers, [LABELS[model] for model in models])
        axis.set_ylabel(ylabel)
        axis.set_title(
            f"Final {summary[0]['final_window']} episodes: mean ± seed SD"
        )
        if fields == RATE_FIELDS:
            axis.set_ylim(0, 105)
        axis.legend(frameon=False)
        axis.grid(axis="y", alpha=0.2)
        save_figure(figure, output_dir, name)


def main() -> None:
    """Validate sources, export complete counts, and generate paper figures."""
    args = parse_args()
    if args.history_csv:
        rows, sources = load_count_csvs(args.history_csv)
    else:
        rows, sources = load_wandb_counts(args)
    rows = validate_histories(
        rows, args.models, args.seeds, args.topology, args.expected_episodes
    )
    per_seed, summary = summarize_counts(
        rows, args.models, args.seeds, args.final_window
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "offloading_history.csv", rows)
    write_csv(args.output_dir / "per_seed_summary.csv", per_seed)
    write_csv(args.output_dir / "final_window_summary.csv", summary)
    plot_counts(
        rows, summary, args.models, args.seeds,
        args.smooth_window, args.output_dir,
    )
    manifest = {
        "sources": sources, "models": args.models, "seeds": args.seeds,
        "topology": args.topology, "expected_episodes": args.expected_episodes,
        "final_window": args.final_window, "smooth_window": args.smooth_window,
        "script_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "git_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
        ).strip(),
        "definitions": {
            "unit": "subtask offloading; not whole-DAG deadline success",
            "successful": "resolved_edge_count", "rejected": "penalty_count",
            "rates": (
                "sum outcomes / sum edge requests within each seed's window"
            ),
            "uncertainty": "sample SD across independent seed summaries",
            "zero_requests": (
                "undefined (N/A); excluded from means; see rate_seed_count"
            ),
        },
    }
    (args.output_dir / "source_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    lines = [
        "# Subtask offloading counts", "",
        f"Final {args.final_window} episodes; mean ± sample SD across seeds.",
        "Rates divide summed outcomes by summed requests within each seed.",
        "Zero-request seeds have undefined rates and are excluded from rates.",
        "", (
            "| Model | Requests/episode | Successes/episode "
            "| Rejections/episode "
            "| Success (%) | Rejection (%) | Rate seeds |"
        ),
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in summary:
        cells = [
            f"{row[f + '_mean']:.2f} ± {row[f + '_sd']:.2f}"
            if math.isfinite(row[f + '_mean']) else "N/A"
            for f in (*COUNT_FIELDS, *RATE_FIELDS)
        ]
        lines.append(
            f"| {LABELS[row['model']]} | " + " | ".join(cells)
            + f" | {row['rate_seed_count']} |"
        )
    (args.output_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print(
        f"Saved {len(rows)} episode rows and count/rate plots "
        f"to {args.output_dir}"
    )


if __name__ == "__main__":
    main()
