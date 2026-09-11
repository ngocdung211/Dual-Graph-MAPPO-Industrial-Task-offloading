"""Combine independent comparison runs into one publication figure.

The script reads per-episode histories from two interchangeable sources:

* local W&B run directories (`wandb/run-*/run-*.wandb`), and
* `*_episode_history.csv` files written by `run_comparision.py`.

Runs are grouped by model, averaged across seeds, and plotted with a
variability band. Summary tables for the paper's result discussion are written
next to the figures.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

METRICS = ("reward", "delay_s", "energy_j")
WANDB_HISTORY_KEYS = {
    "performance/reward": "reward",
    "performance/delay_seconds": "delay_s",
    "performance/energy_joules": "energy_j",
}
MODEL_STYLES = {
    "e-ATN-MADDPG": {"color": "#4C78A8", "linestyle": "--"},
    "Shared MAPPO": {"color": "#F58518", "linestyle": "-."},
    "Shared Mask MAPPO": {"color": "#54A24B", "linestyle": ":"},
    "Graph-GAT MAPPO": {"color": "#B279A2", "linestyle": (0, (6, 2))},
    "Graph-GAT Mask MAPPO": {"color": "#E45756", "linestyle": "-"},
    "MAPPO": {"color": "#79706E", "linestyle": (0, (5, 2))},
    "Mask MAPPO": {"color": "#9D755D", "linestyle": (0, (3, 1, 1, 1))},
    "Graph-GAT Warmup MAPPO": {"color": "#72B7B2", "linestyle": (0, (4, 1, 1, 1))},
    "Graph-GAT Warmup Mask MAPPO": {"color": "#E45756", "linestyle": "-"},
}
MODEL_LABELS = {
    "e-ATN-MADDPG": "e-ATN-MADDPG",
    "Shared MAPPO": "MAPPO",
    "Shared Mask MAPPO": "Mask MAPPO",
    "Graph-GAT MAPPO": "CW-GAT-MAPPO",
    "Graph-GAT Mask MAPPO": "Graph Mask MAPPO",
    "MAPPO": "MAPPO (independent)",
    "Mask MAPPO": "Mask MAPPO (independent)",
    "Graph-GAT Warmup MAPPO": "Graph MAPPO + warmup",
    "Graph-GAT Warmup Mask MAPPO": "Graph Mask MAPPO + warmup",
}
HIGHLIGHT_MODELS = frozenset(
    {"Graph-GAT Mask MAPPO", "Graph-GAT Warmup Mask MAPPO"}
)
PANELS = (
    ("reward", "Reward", "combined_reward"),
    ("delay_s", "Delay (s)", "combined_delay"),
    ("energy_j", "Energy (J)", "combined_energy"),
)


class RunHistory:
    """One training run: a model, a seed, and its per-episode metrics."""

    def __init__(
        self,
        model: str,
        seed: Optional[int],
        source: str,
        episodes: np.ndarray,
        metrics: Dict[str, np.ndarray],
    ):
        """Store one run.

        Args:
            model: Algorithm display name.
            seed: Experiment seed, when the source records one.
            source: Human-readable origin of the run.
            episodes: Episode indices, ascending.
            metrics: Metric name to per-episode values.
        """
        self.model = model
        self.seed = seed
        self.source = source
        self.episodes = episodes
        self.metrics = metrics


def parse_args() -> argparse.Namespace:
    """Parse plotting arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--wandb-dir",
        type=Path,
        default=None,
        help="Directory holding local W&B runs (wandb/run-*/run-*.wandb).",
    )
    parser.add_argument(
        "--history-csv",
        type=Path,
        nargs="*",
        default=(),
        help=(
            "One or more *_episode_history.csv files, or directories that are "
            "searched recursively for them."
        ),
    )
    parser.add_argument(
        "--wandb-group",
        nargs="*",
        default=(),
        help="Keep only W&B runs whose run group matches one of these names.",
    )
    parser.add_argument(
        "--note",
        nargs="*",
        default=(),
        help="Keep only CSV runs whose experiment note matches one of these.",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=(),
        help="Models to plot, in legend order. Default: every model found.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="*",
        default=(),
        help="Seeds to keep. Default: every seed found.",
    )
    parser.add_argument(
        "--topology",
        default="large_30d_9s",
        help="Topology scenario to keep.",
    )
    parser.add_argument(
        "--min-episodes",
        type=int,
        default=0,
        help=(
            "Drop runs with fewer logged episodes than this. Use it to skip "
            "interrupted runs that would otherwise truncate every curve."
        ),
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Truncate every run to this many episodes. Default: shortest run.",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=8,
        help="Trailing moving-average window applied before aggregation.",
    )
    parser.add_argument(
        "--band",
        choices=("auto", "seed_sd", "seed_ci95", "episode_sd", "none"),
        default="auto",
        help=(
            "Shaded band: across-seed SD, across-seed 95%% CI, or the rolling "
            "within-run SD. 'auto' uses across-seed SD when at least two seeds "
            "exist and the rolling within-run SD otherwise."
        ),
    )
    parser.add_argument(
        "--final-window",
        type=int,
        default=100,
        help="Episode count used for the converged-performance tables.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/analysis/combined_runs"),
        help="Directory for figures and summary tables.",
    )
    parser.add_argument(
        "--title",
        default="",
        help="Optional figure title.",
    )
    return parser.parse_args()


def _decode_config(run_record: object) -> Dict[str, object]:
    """Decode JSON-valued W&B run configuration."""
    return {
        update.key: json.loads(update.value_json)
        for update in run_record.config.update
    }


def _decode_history(history_record: object) -> Dict[str, object]:
    """Decode one W&B history record."""
    row: Dict[str, object] = {}
    for item in history_record.item:
        key = item.key if item.key else "/".join(item.nested_key)
        row[key] = json.loads(item.value_json)
    return row


def load_wandb_runs(
    wandb_dir: Path,
    groups: Sequence[str],
    topology: str,
) -> List[RunHistory]:
    """Read matching runs from a local W&B directory.

    Args:
        wandb_dir: Directory containing `run-*/run-*.wandb` files.
        groups: Run groups to keep, or empty to keep every group.
        topology: Topology scenario to keep.

    Returns:
        One RunHistory per matching W&B run.
    """
    from wandb.proto import wandb_internal_pb2
    from wandb.sdk.internal.datastore import DataStore

    runs: List[RunHistory] = []
    for wandb_path in sorted(wandb_dir.glob("run-*/run-*.wandb")):
        datastore = DataStore()
        try:
            datastore.open_for_scan(str(wandb_path))
        except Exception:  # noqa: BLE001 - skip partially written runs
            continue
        config: Dict[str, object] = {}
        group = ""
        rows: List[Dict[str, float]] = []
        while True:
            try:
                record_bytes = datastore.scan_data()
            except Exception:  # noqa: BLE001 - stop at a truncated record
                break
            if record_bytes is None:
                break
            record = wandb_internal_pb2.Record()
            record.ParseFromString(record_bytes)
            if record.HasField("run"):
                config = _decode_config(record.run)
                group = record.run.run_group
            elif record.HasField("history") and config:
                raw_row = _decode_history(record.history)
                if all(
                    key in raw_row
                    for key in ("episode", *WANDB_HISTORY_KEYS)
                ):
                    row = {"episode": float(raw_row["episode"])}
                    row.update(
                        {
                            output_key: float(raw_row[input_key])
                            for input_key, output_key in WANDB_HISTORY_KEYS.items()
                        }
                    )
                    rows.append(row)
        if not rows or not config:
            continue
        if groups and group not in groups:
            continue
        if topology and config.get("topology_scenario") != topology:
            continue
        rows.sort(key=lambda row: row["episode"])
        runs.append(
            RunHistory(
                model=str(config.get("algorithm")),
                seed=(
                    int(config["seed"]) if config.get("seed") is not None else None
                ),
                source=f"wandb:{wandb_path.parent.name}:{group}",
                episodes=np.asarray([row["episode"] for row in rows], dtype=float),
                metrics={
                    metric: np.asarray(
                        [row[metric] for row in rows], dtype=float
                    )
                    for metric in METRICS
                },
            )
        )
    return runs


def _iter_history_csv_paths(paths: Iterable[Path]) -> List[Path]:
    """Expand directories into the episode-history CSV files they contain."""
    csv_paths: List[Path] = []
    for path in paths:
        if path.is_dir():
            csv_paths.extend(sorted(path.rglob("*_episode_history.csv")))
        elif path.is_file():
            csv_paths.append(path)
    return csv_paths


def load_csv_runs(
    paths: Iterable[Path],
    notes: Sequence[str],
    topology: str,
) -> List[RunHistory]:
    """Read runs from `run_comparision.py` episode-history CSV files.

    Args:
        paths: CSV files or directories to search.
        notes: Experiment notes to keep, or empty to keep every note.
        topology: Topology scenario to keep.

    Returns:
        One RunHistory per (file, model) pair.
    """
    runs: List[RunHistory] = []
    for csv_path in _iter_history_csv_paths(paths):
        grouped: Dict[Tuple[str, Optional[int]], List[Dict[str, str]]] = (
            defaultdict(list)
        )
        with csv_path.open(encoding="utf-8", newline="") as csv_file:
            for row in csv.DictReader(csv_file):
                if notes and row.get("experiment_note", "") not in notes:
                    continue
                if topology and row.get("topology_scenario", "") != topology:
                    continue
                seed_text = row.get("seed", "")
                seed = int(seed_text) if seed_text not in ("", None) else None
                grouped[(row["model"], seed)].append(row)
        for (model, seed), rows in grouped.items():
            rows.sort(key=lambda row: int(row["episode"]))
            runs.append(
                RunHistory(
                    model=model,
                    seed=seed,
                    source=f"csv:{csv_path.parent.name}",
                    episodes=np.asarray(
                        [float(row["episode"]) for row in rows], dtype=float
                    ),
                    metrics={
                        metric: np.asarray(
                            [
                                float(row[metric]) if row[metric] else math.nan
                                for row in rows
                            ],
                            dtype=float,
                        )
                        for metric in METRICS
                    },
                )
            )
    return runs


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    """Return a trailing moving average with a growing initial window.

    Args:
        values: Raw per-episode values.
        window: Maximum trailing window length.

    Returns:
        Smoothed values with the same length as `values`.
    """
    if window <= 1:
        return np.asarray(values, dtype=float)
    values_array = np.asarray(values, dtype=float)
    cumulative = np.cumsum(np.insert(values_array, 0, 0.0))
    smoothed = np.empty_like(values_array)
    for index in range(len(values_array)):
        start = max(0, index + 1 - window)
        smoothed[index] = (cumulative[index + 1] - cumulative[start]) / (
            index + 1 - start
        )
    return smoothed


def rolling_std(values: np.ndarray, window: int) -> np.ndarray:
    """Return the trailing rolling standard deviation of raw values.

    Args:
        values: Raw per-episode values.
        window: Maximum trailing window length.

    Returns:
        Rolling standard deviations with the same length as `values`.
    """
    values_array = np.asarray(values, dtype=float)
    deviations = np.zeros_like(values_array)
    effective_window = max(2, window)
    for index in range(len(values_array)):
        start = max(0, index + 1 - effective_window)
        chunk = values_array[start : index + 1]
        deviations[index] = float(chunk.std(ddof=1)) if len(chunk) > 1 else 0.0
    return deviations


def select_runs(
    runs: Sequence[RunHistory],
    models: Sequence[str],
    seeds: Sequence[int],
    min_episodes: int = 0,
) -> Dict[str, List[RunHistory]]:
    """Group runs by model after applying model, seed, and length filters.

    Args:
        runs: Loaded runs.
        models: Models to keep, or empty to keep every model.
        seeds: Seeds to keep, or empty to keep every seed.
        min_episodes: Minimum logged episodes required to keep a run.

    Returns:
        Model name to its runs, ordered by seed.
    """
    grouped: Dict[str, List[RunHistory]] = defaultdict(list)
    for run in runs:
        if models and run.model not in models:
            continue
        if seeds and run.seed not in seeds:
            continue
        if len(run.episodes) < min_episodes:
            print(
                f"  skipping {run.model} ({run.source}): "
                f"{len(run.episodes)} episodes < {min_episodes}"
            )
            continue
        grouped[run.model].append(run)
    for model_runs in grouped.values():
        model_runs.sort(key=lambda run: (run.seed is None, run.seed, run.source))
    return dict(grouped)


def aggregate_model(
    model_runs: Sequence[RunHistory],
    metric: str,
    episode_count: int,
    smooth_window: int,
    band: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """Aggregate one model's runs into a mean curve and a variability band.

    Args:
        model_runs: Runs belonging to one model.
        metric: Metric key to aggregate.
        episode_count: Episodes retained from every run.
        smooth_window: Trailing moving-average window.
        band: Band mode requested on the command line.

    Returns:
        Tuple of (mean, lower, upper, resolved_band_mode).
    """
    smoothed = np.stack(
        [
            moving_average(run.metrics[metric][:episode_count], smooth_window)
            for run in model_runs
        ]
    )
    mean = smoothed.mean(axis=0)
    resolved_band = band
    if band == "auto":
        resolved_band = "seed_sd" if len(model_runs) >= 2 else "episode_sd"
    if resolved_band == "none":
        return mean, mean, mean, resolved_band
    if resolved_band in ("seed_sd", "seed_ci95") and len(model_runs) >= 2:
        deviation = smoothed.std(axis=0, ddof=1)
        if resolved_band == "seed_ci95":
            # Student-t critical value keeps small seed counts honest.
            critical_values = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776}
            critical = critical_values.get(len(model_runs), 1.96)
            deviation = deviation * critical / math.sqrt(len(model_runs))
    else:
        resolved_band = "episode_sd"
        deviation = np.stack(
            [
                rolling_std(run.metrics[metric][:episode_count], smooth_window)
                for run in model_runs
            ]
        ).mean(axis=0)
    return mean, mean - deviation, mean + deviation, resolved_band


def plot_panels(
    grouped_runs: Dict[str, List[RunHistory]],
    model_order: Sequence[str],
    episode_count: int,
    smooth_window: int,
    band: str,
    output_dir: Path,
    title: str,
) -> str:
    """Write one figure per metric and return the resolved band mode.

    Args:
        grouped_runs: Model name to its runs.
        model_order: Legend order.
        episode_count: Episodes retained from every run.
        smooth_window: Trailing moving-average window.
        band: Band mode requested on the command line.
        output_dir: Destination directory.
        title: Optional figure title.

    Returns:
        The band mode actually used for the last model plotted.
    """
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    episodes = np.arange(1, episode_count + 1, dtype=float)
    resolved_band = band
    for metric, ylabel, filename in PANELS:
        figure, axis = plt.subplots(figsize=(5.4, 3.55))
        for model in model_order:
            mean, lower, upper, resolved_band = aggregate_model(
                grouped_runs[model],
                metric,
                episode_count,
                smooth_window,
                band,
            )
            style = MODEL_STYLES.get(
                model, {"color": None, "linestyle": "-"}
            )
            axis.fill_between(
                episodes,
                lower,
                upper,
                color=style["color"],
                alpha=0.17,
                linewidth=0,
            )
            axis.plot(
                episodes,
                mean,
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=1.7 if model in HIGHLIGHT_MODELS else 1.35,
                label=MODEL_LABELS.get(model, model),
            )
        axis.set_xlabel("Training episode")
        axis.set_ylabel(ylabel)
        axis.set_xlim(1, episode_count)
        axis.grid(True, alpha=0.22, linewidth=0.5)
        axis.spines[["top", "right"]].set_visible(False)
        if title:
            axis.set_title(title)
        axis.legend(
            loc="lower center",
            bbox_to_anchor=(0.5, 1.01),
            ncol=2,
            frameon=False,
            columnspacing=1.25,
            handlelength=2.2,
        )
        figure.tight_layout(pad=0.35)
        output_path = output_dir / filename
        figure.savefig(
            output_path.with_suffix(".png"), dpi=300, bbox_inches="tight"
        )
        figure.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(figure)
    return resolved_band


def write_curve_csv(
    path: Path,
    grouped_runs: Dict[str, List[RunHistory]],
    model_order: Sequence[str],
    episode_count: int,
    smooth_window: int,
    band: str,
) -> None:
    """Write the plotted mean and band for every model and episode."""
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(
            ["model", "episode", "run_count", "metric", "mean", "lower", "upper"]
        )
        for model in model_order:
            for metric, _, _ in PANELS:
                mean, lower, upper, _ = aggregate_model(
                    grouped_runs[model],
                    metric,
                    episode_count,
                    smooth_window,
                    band,
                )
                for index in range(episode_count):
                    writer.writerow(
                        [
                            model,
                            index + 1,
                            len(grouped_runs[model]),
                            metric,
                            mean[index],
                            lower[index],
                            upper[index],
                        ]
                    )


def build_final_window_rows(
    grouped_runs: Dict[str, List[RunHistory]],
    model_order: Sequence[str],
    episode_count: int,
    final_window: int,
) -> List[Dict[str, object]]:
    """Summarize converged performance over the final training window.

    Args:
        grouped_runs: Model name to its runs.
        model_order: Row order.
        episode_count: Episodes retained from every run.
        final_window: Episodes averaged at the end of training.

    Returns:
        One summary row per model.
    """
    rows: List[Dict[str, object]] = []
    for model in model_order:
        model_runs = grouped_runs[model]
        row: Dict[str, object] = {
            "model": model,
            "runs": len(model_runs),
            "seeds": ";".join(str(run.seed) for run in model_runs),
            "episodes": episode_count,
            "final_window": final_window,
        }
        for metric, _, _ in PANELS:
            per_seed = [
                float(
                    np.mean(
                        run.metrics[metric][:episode_count][-final_window:]
                    )
                )
                for run in model_runs
            ]
            row[f"{metric}_mean"] = statistics.fmean(per_seed)
            row[f"{metric}_sample_std"] = (
                statistics.stdev(per_seed) if len(per_seed) > 1 else 0.0
            )
            row[f"{metric}_per_seed"] = ";".join(
                f"{value:.4f}" for value in per_seed
            )
        rows.append(row)
    return rows


def build_efficiency_rows(
    grouped_runs: Dict[str, List[RunHistory]],
    model_order: Sequence[str],
    episode_count: int,
    smooth_window: int,
    final_window: int,
) -> List[Dict[str, object]]:
    """Measure convergence speed and stability of the reward curves.

    Args:
        grouped_runs: Model name to its runs.
        model_order: Row order.
        episode_count: Episodes retained from every run.
        smooth_window: Trailing moving-average window.
        final_window: Episodes averaged at the end of training.

    Returns:
        One efficiency row per model.
    """
    rows: List[Dict[str, object]] = []
    for model in model_order:
        model_runs = grouped_runs[model]
        smoothed = np.stack(
            [
                moving_average(
                    run.metrics["reward"][:episode_count], smooth_window
                )
                for run in model_runs
            ]
        )
        mean_curve = smoothed.mean(axis=0)
        final_level = float(np.mean(mean_curve[-final_window:]))
        start_level = float(mean_curve[0])
        rows.append(
            {
                "model": model,
                "runs": len(model_runs),
                "reward_start": start_level,
                "reward_final": final_level,
                "reward_gain": final_level - start_level,
                "episodes_to_90pct_final": _episodes_to_threshold(
                    mean_curve, start_level + 0.90 * (final_level - start_level)
                ),
                "episodes_to_95pct_final": _episodes_to_threshold(
                    mean_curve, start_level + 0.95 * (final_level - start_level)
                ),
                "reward_auc_per_episode": float(np.mean(mean_curve)),
                "final_window_raw_std": float(
                    np.mean(
                        [
                            np.std(
                                run.metrics["reward"][:episode_count][
                                    -final_window:
                                ],
                                ddof=1,
                            )
                            for run in model_runs
                        ]
                    )
                ),
                "cross_seed_spread": float(
                    np.mean(
                        smoothed.std(axis=0, ddof=1)
                        if len(model_runs) > 1
                        else np.zeros_like(mean_curve)
                    )
                ),
            }
        )
    return rows


def _episodes_to_threshold(curve: np.ndarray, threshold: float) -> int:
    """Return the first 1-indexed episode whose smoothed value clears a level."""
    reached = np.nonzero(curve >= threshold)[0]
    return int(reached[0]) + 1 if reached.size else -1


def write_dict_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    """Write dictionary rows to CSV."""
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_report(
    path: Path,
    final_rows: Sequence[Dict[str, object]],
    efficiency_rows: Sequence[Dict[str, object]],
    reference_model: str,
    band_mode: str,
    smooth_window: int,
    episode_count: int,
) -> None:
    """Write a Markdown table set for the paper's result discussion.

    Args:
        path: Destination Markdown file.
        final_rows: Converged-performance rows.
        efficiency_rows: Convergence-speed rows.
        reference_model: Model used as the improvement reference.
        band_mode: Band mode used in the figures.
        smooth_window: Trailing moving-average window.
        episode_count: Episodes retained from every run.
    """
    reference = next(
        (row for row in final_rows if row["model"] == reference_model), None
    )
    lines = [
        "# Combined comparison summary",
        "",
        f"- Episodes per run: {episode_count}",
        f"- Smoothing: trailing moving average, window {smooth_window}",
        f"- Shaded band: {band_mode}",
        f"- Reference model for relative gains: {reference_model}",
        "",
        "## Converged performance (final window)",
        "",
        "| Model | Runs | Seeds | Reward | Delay (s) | Energy (J) |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in final_rows:
        lines.append(
            "| {model} | {runs} | {seeds} | {reward:.3f} ± {reward_sd:.3f} | "
            "{delay:.4f} ± {delay_sd:.4f} | {energy:.4f} ± {energy_sd:.4f} |".format(
                model=row["model"],
                runs=row["runs"],
                seeds=row["seeds"],
                reward=row["reward_mean"],
                reward_sd=row["reward_sample_std"],
                delay=row["delay_s_mean"],
                delay_sd=row["delay_s_sample_std"],
                energy=row["energy_j_mean"],
                energy_sd=row["energy_j_sample_std"],
            )
        )
    if reference is not None:
        lines += [
            "",
            f"## Relative to {reference_model}",
            "",
            "Positive values favour the reference model: reward is higher, "
            "delay and energy are lower.",
            "",
            "| Model | Reward Δ | Reward gain | Delay reduction | "
            "Energy reduction |",
            "| --- | --- | --- | --- | --- |",
        ]
        for row in final_rows:
            reward_delta = float(reference["reward_mean"]) - float(
                row["reward_mean"]
            )
            lines.append(
                "| {model} | {delta:+.3f} | {delta_pct:+.1f}% | "
                "{delay_pct:+.1f}% | {energy_pct:+.1f}% |".format(
                    model=row["model"],
                    delta=reward_delta,
                    delta_pct=_percentage(
                        reference["reward_mean"], row["reward_mean"]
                    ),
                    delay_pct=-_percentage(
                        reference["delay_s_mean"], row["delay_s_mean"]
                    ),
                    energy_pct=-_percentage(
                        reference["energy_j_mean"], row["energy_j_mean"]
                    ),
                )
            )
    lines += [
        "",
        "## Convergence speed and stability (reward)",
        "",
        "| Model | Start | Final | Episodes to 90% | Episodes to 95% | "
        "Mean reward over training | Final-window raw SD | Cross-seed spread |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in efficiency_rows:
        lines.append(
            "| {model} | {start:.3f} | {final:.3f} | {e90} | {e95} | "
            "{auc:.3f} | {raw_sd:.3f} | {spread:.3f} |".format(
                model=row["model"],
                start=row["reward_start"],
                final=row["reward_final"],
                e90=row["episodes_to_90pct_final"],
                e95=row["episodes_to_95pct_final"],
                auc=row["reward_auc_per_episode"],
                raw_sd=row["final_window_raw_std"],
                spread=row["cross_seed_spread"],
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _percentage(reference_value: float, other_value: float) -> float:
    """Return the signed percentage change from `other_value` to the reference."""
    if other_value == 0:
        return 0.0
    return (float(reference_value) - float(other_value)) / abs(
        float(other_value)
    ) * 100.0


def main() -> None:
    """Aggregate runs, write figures, and write summary tables."""
    args = parse_args()
    runs: List[RunHistory] = []
    if args.wandb_dir is not None:
        runs.extend(
            load_wandb_runs(args.wandb_dir, args.wandb_group, args.topology)
        )
    if args.history_csv:
        runs.extend(load_csv_runs(args.history_csv, args.note, args.topology))
    if not runs:
        raise SystemExit("no matching runs found")

    grouped_runs = select_runs(
        runs, args.models, args.seeds, args.min_episodes
    )
    if not grouped_runs:
        raise SystemExit("every run was filtered out")
    model_order = [
        model for model in args.models if model in grouped_runs
    ] or sorted(grouped_runs)

    available_episodes = min(
        len(run.episodes)
        for model in model_order
        for run in grouped_runs[model]
    )
    episode_count = (
        min(available_episodes, args.episodes)
        if args.episodes
        else available_episodes
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Aggregating {episode_count} episodes per run")
    for model in model_order:
        sources = ", ".join(
            f"seed={run.seed}:{run.source}" for run in grouped_runs[model]
        )
        print(f"  {model}: {len(grouped_runs[model])} run(s) [{sources}]")

    band_mode = plot_panels(
        grouped_runs,
        model_order,
        episode_count,
        args.smooth_window,
        args.band,
        args.output_dir,
        args.title,
    )
    write_curve_csv(
        args.output_dir / "episode_curves.csv",
        grouped_runs,
        model_order,
        episode_count,
        args.smooth_window,
        args.band,
    )
    final_rows = build_final_window_rows(
        grouped_runs, model_order, episode_count, args.final_window
    )
    efficiency_rows = build_efficiency_rows(
        grouped_runs,
        model_order,
        episode_count,
        args.smooth_window,
        args.final_window,
    )
    write_dict_csv(args.output_dir / "final_window_summary.csv", final_rows)
    write_dict_csv(args.output_dir / "convergence_summary.csv", efficiency_rows)
    write_report(
        args.output_dir / "summary.md",
        final_rows,
        efficiency_rows,
        reference_model=model_order[-1],
        band_mode=band_mode,
        smooth_window=args.smooth_window,
        episode_count=episode_count,
    )
    print(f"Saved combined outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
