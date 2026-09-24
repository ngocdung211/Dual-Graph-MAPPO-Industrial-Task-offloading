"""Plot large-topology training curves across independent random seeds."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from wandb.proto import wandb_internal_pb2
from wandb.sdk.internal.datastore import DataStore


MODELS = (
    "e-ATN-MADDPG",
    "MAPPO",
    "Graph-GAT MAPPO",
    "Graph-GAT Warmup MAPPO",
)
SEEDS = (75, 175, 190)
TOPOLOGY = "large_30d_9s"
EXPECTED_EPISODES = 1000
HISTORY_KEYS = {
    "performance/reward": "reward",
    "performance/delay_seconds": "delay_s",
    "performance/energy_joules": "energy_j",
}
MODEL_STYLES = {
    "e-ATN-MADDPG": {"color": "#4C78A8", "linestyle": "--"},
    "MAPPO": {"color": "#F58518", "linestyle": "-."},
    "Graph-GAT MAPPO": {"color": "#54A24B", "linestyle": ":"},
    "Graph-GAT Warmup MAPPO": {"color": "#E45756", "linestyle": "-"},
}
MODEL_LABELS = {
    "e-ATN-MADDPG": "e-ATN-MADDPG [1]",
    "MAPPO": "MAPPO",
    "Graph-GAT MAPPO": "Dual-GAT MAPPO w/o warmup",
    "Graph-GAT Warmup MAPPO": "Dual-GAT MAPPO (ours)",
}


def parse_args() -> argparse.Namespace:
    """Parse plotting arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb-dir", type=Path, default=Path("wandb"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/analysis/final_ladder_3008"),
    )
    parser.add_argument("--smooth-window", type=int, default=20)
    return parser.parse_args()


def _decode_config(run_record: object) -> Dict[str, object]:
    """Decode JSON-valued W&B run configuration."""
    return {
        update.key: json.loads(update.value_json)
        for update in run_record.config.update
    }


def _decode_history(history_record: object) -> Dict[str, object]:
    """Decode one W&B history record."""
    row = {}
    for item in history_record.item:
        key = item.key if item.key else "/".join(item.nested_key)
        row[key] = json.loads(item.value_json)
    return row


def _read_wandb_run(
    wandb_path: Path,
) -> Tuple[Dict[str, object], str, str, List[Dict[str, float]]]:
    """Read config and selected episode metrics from one local W&B run."""
    datastore = DataStore()
    datastore.open_for_scan(str(wandb_path))
    config: Dict[str, object] = {}
    group = ""
    run_id = ""
    rows: List[Dict[str, float]] = []
    while True:
        record_bytes = datastore.scan_data()
        if record_bytes is None:
            break
        record = wandb_internal_pb2.Record()
        record.ParseFromString(record_bytes)
        if record.HasField("run"):
            config = _decode_config(record.run)
            group = record.run.run_group
            run_id = record.run.run_id
        elif record.HasField("history") and config:
            raw_row = _decode_history(record.history)
            required_keys = ("episode", *HISTORY_KEYS)
            if all(key in raw_row for key in required_keys):
                row = {"episode": float(raw_row["episode"])}
                row.update(
                    {
                        output_key: float(raw_row[input_key])
                        for input_key, output_key in HISTORY_KEYS.items()
                    }
                )
                rows.append(row)
    return config, group, run_id, rows


def load_histories(
    wandb_dir: Path,
) -> Dict[Tuple[int, str], Tuple[str, List[Dict[str, float]]]]:
    """Load the complete matched large-topology histories."""
    histories = {}
    for wandb_path in sorted(wandb_dir.glob("run-*/run-*.wandb")):
        config, group, run_id, rows = _read_wandb_run(wandb_path)
        seed = config.get("seed")
        model = config.get("algorithm")
        if (
            seed not in SEEDS
            or model not in MODELS
            or config.get("topology_scenario") != TOPOLOGY
            or config.get("episodes") != EXPECTED_EPISODES
            or group != f"final_ladder_s{seed}"
        ):
            continue
        episodes = sorted(int(row["episode"]) for row in rows)
        if episodes != list(range(1, EXPECTED_EPISODES + 1)):
            raise ValueError(
                f"incomplete history for seed={seed}, model={model}: "
                f"found {len(episodes)} episodes"
            )
        key = (int(seed), str(model))
        if key in histories:
            raise ValueError(f"duplicate matched run for {key}")
        histories[key] = (run_id, sorted(rows, key=lambda row: row["episode"]))

    expected = {(seed, model) for seed in SEEDS for model in MODELS}
    missing = sorted(expected - set(histories))
    if missing:
        raise ValueError(f"missing matched W&B histories: {missing}")
    return histories


def moving_average(values: Sequence[float], window: int) -> np.ndarray:
    """Return a trailing moving average with a growing initial window."""
    if window <= 0:
        raise ValueError("smooth window must be positive")
    values_array = np.asarray(values, dtype=float)
    cumulative = np.cumsum(np.insert(values_array, 0, 0.0))
    smoothed = np.empty_like(values_array)
    for index in range(len(values_array)):
        start = max(0, index + 1 - window)
        total = cumulative[index + 1] - cumulative[start]
        smoothed[index] = total / (index + 1 - start)
    return smoothed


def build_output_rows(
    histories: Mapping[Tuple[int, str], Tuple[str, List[Dict[str, float]]]],
    smooth_window: int,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    """Build per-seed histories and episode-wise mean/SD rows."""
    history_rows: List[Dict[str, object]] = []
    summary_rows: List[Dict[str, object]] = []
    smoothed_values: Dict[Tuple[str, str], List[np.ndarray]] = defaultdict(list)

    for model in MODELS:
        for seed in SEEDS:
            run_id, rows = histories[(seed, model)]
            metric_arrays = {
                metric: np.asarray([row[metric] for row in rows], dtype=float)
                for metric in HISTORY_KEYS.values()
            }
            smoothed_arrays = {
                metric: moving_average(values, smooth_window)
                for metric, values in metric_arrays.items()
            }
            for metric, values in smoothed_arrays.items():
                smoothed_values[(model, metric)].append(values)
            for index, row in enumerate(rows):
                output_row: Dict[str, object] = {
                    "run_id": run_id,
                    "seed": seed,
                    "model": model,
                    "topology": TOPOLOGY,
                    "episode": int(row["episode"]),
                }
                for metric in HISTORY_KEYS.values():
                    output_row[metric] = metric_arrays[metric][index]
                    output_row[f"{metric}_smoothed"] = smoothed_arrays[metric][index]
                history_rows.append(output_row)

        for episode_index in range(EXPECTED_EPISODES):
            output_row = {
                "model": model,
                "topology": TOPOLOGY,
                "episode": episode_index + 1,
                "seed_count": len(SEEDS),
                "smooth_window": smooth_window,
            }
            for metric in HISTORY_KEYS.values():
                seed_values = np.asarray(
                    [
                        values[episode_index]
                        for values in smoothed_values[(model, metric)]
                    ]
                )
                output_row[f"{metric}_mean"] = float(seed_values.mean())
                output_row[f"{metric}_std"] = float(seed_values.std(ddof=1))
            summary_rows.append(output_row)
    return history_rows, summary_rows


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    """Write rows to CSV."""
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_final_window_summary(
    path: Path,
    histories: Mapping[Tuple[int, str], Tuple[str, List[Dict[str, float]]]],
) -> None:
    """Write final-100 means and across-seed sample SDs."""
    rows = []
    for model in MODELS:
        seed_means = {}
        for seed in SEEDS:
            final_rows = histories[(seed, model)][1][-100:]
            seed_means[seed] = {
                metric: statistics.fmean(row[metric] for row in final_rows)
                for metric in HISTORY_KEYS.values()
            }
        output_row: Dict[str, object] = {
            "model": model,
            "topology": TOPOLOGY,
            "seeds": "75;175;190",
            "final_window": 100,
        }
        for metric in HISTORY_KEYS.values():
            values = [seed_means[seed][metric] for seed in SEEDS]
            for seed, value in zip(SEEDS, values):
                output_row[f"{metric}_seed_{seed}"] = value
            output_row[f"{metric}_mean"] = statistics.fmean(values)
            output_row[f"{metric}_sample_std"] = statistics.stdev(values)
        rows.append(output_row)
    write_csv(path, rows)


def plot_curves(
    output_dir: Path,
    summary_rows: Sequence[Mapping[str, object]],
    smooth_window: int,
) -> None:
    """Plot separate publication-style mean and SD training figures."""
    rows_by_model = {
        model: [row for row in summary_rows if row["model"] == model]
        for model in MODELS
    }
    panels = (
        ("reward", "Reward", "large_reward_mean_std"),
        ("delay_s", "Delay (s)", "large_delay_mean_std"),
        (
            "energy_j",
            "Energy (J)",
            "large_energy_mean_std",
        ),
    )
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "legend.fontsize": 8,
        }
    )
    for metric, ylabel, filename in panels:
        figure, axis = plt.subplots(figsize=(5.4, 3.55))
        for model in MODELS:
            rows = rows_by_model[model]
            episodes = np.asarray([row["episode"] for row in rows], dtype=float)
            mean = np.asarray([row[f"{metric}_mean"] for row in rows])
            std = np.asarray([row[f"{metric}_std"] for row in rows])
            style = MODEL_STYLES[model]
            axis.fill_between(
                episodes,
                mean - std,
                mean + std,
                color=style["color"],
                alpha=0.13,
                linewidth=0,
            )
            axis.plot(
                episodes,
                mean,
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=1.7 if model == "Graph-GAT Warmup MAPPO" else 1.35,
                label=MODEL_LABELS[model],
            )
        axis.set_xlabel("Training episode")
        axis.set_ylabel(ylabel)
        axis.set_xlim(1, EXPECTED_EPISODES)
        axis.grid(True, alpha=0.22, linewidth=0.5)
        axis.spines[["top", "right"]].set_visible(False)
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


def main() -> None:
    """Export matched histories and create the first multi-seed figure."""
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    histories = load_histories(args.wandb_dir)
    history_rows, summary_rows = build_output_rows(
        histories, args.smooth_window
    )
    write_csv(args.output_dir / "training_history.csv", history_rows)
    write_csv(args.output_dir / "episode_mean_std.csv", summary_rows)
    write_final_window_summary(
        args.output_dir / "final_100_seed_summary.csv", histories
    )
    plot_curves(
        args.output_dir,
        summary_rows,
        args.smooth_window,
    )
    print(f"Saved multi-seed outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
