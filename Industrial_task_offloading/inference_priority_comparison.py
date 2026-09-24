"""Compare saved policies under default and graph-inferred task priorities."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

from baselines.graph_gat_mappo import GraphGATMAPPOAgent
from baselines.gatma import GATMAAgent
from baselines.mappo import MAPPOAgent
from dataset.data_loader import KolektorSDDLoader
from environment.network_env import NetworkEnvironment
from environment.system_model import EdgeServer, IndustrialDevice
from run_comparision import (
    build_devices_for_scenario,
    build_servers_for_scenario,
    evaluate_algorithm_checkpoint,
    set_seed,
)
from utils.experiment_setup import (
    TASK_PRIORITY_FEATURE_DIM,
    build_priorities,
    build_task_priority_model,
    get_priority_checkpoint_path,
    make_priority_dag_sampler,
)
from utils.paper_config import PAPER_PARAMS
from utils.priority_model_training import load_or_train_priority_model
from utils.topology_scenarios_config import TopologyScenario, get_topology_scenario


DEFAULT_CHECKPOINT_DIR = Path(
    "plots/2026-08-12_17-01-21-penalty_0_5_best_medium_1000ep_seed190"
)
CHECKPOINT_FILES = {
    "MAPPO": "MAPPO_checkpoint.pt",
    "Dual-GAT MAPPO w/o warmup": "Graph-GAT_MAPPO_checkpoint.pt",
    "Dual-GAT MAPPO": "Graph-GAT_Warmup_MAPPO_checkpoint.pt",
    "GATMA-Adapted": "GATMA-Adapted_checkpoint.pt",
}
DEFAULT_ALGORITHMS = ("MAPPO", "Dual-GAT MAPPO w/o warmup", "Dual-GAT MAPPO")
AGENT_CLASSES = {
    "MAPPOAgent": MAPPOAgent,
    "GraphGATMAPPOAgent": GraphGATMAPPOAgent,
    "GATMAAgent": GATMAAgent,
}
METRICS = (
    "reward",
    "delay",
    "energy",
    "edge_ratio",
    "resolved_edge_ratio",
    "penalty_count",
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate saved MAPPO, Dual-GAT and GATMA-Adapted checkpoints with default and "
            "Task-GAT-inferred priority orders."
        )
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=DEFAULT_CHECKPOINT_DIR,
        help="Directory containing the selected saved policy checkpoints.",
    )
    parser.add_argument(
        "--algorithms", nargs="+", choices=list(CHECKPOINT_FILES),
        default=list(DEFAULT_ALGORITHMS),
        help="Saved policies to evaluate; include GATMA-Adapted explicitly.",
    )
    parser.add_argument(
        "--device", default="cpu",
        help="Evaluation device for Graph-GAT and GATMA: cpu, auto or cuda.",
    )
    parser.add_argument(
        "--dataset-path",
        default="dataset/KolektorSDD/",
        help="KolektorSDD dataset directory.",
    )
    parser.add_argument(
        "--priority-model",
        choices=("gat", "gcn"),
        default="gat",
        help="Graph model used once to infer the shared priority order.",
    )
    parser.add_argument(
        "--priority-seed",
        type=int,
        default=75,
        help="Seed used for priority-model pretraining and template sampling.",
    )
    parser.add_argument(
        "--system-seed",
        type=int,
        default=None,
        help="Physical compute seed; defaults to checkpoint metadata (190).",
    )
    parser.add_argument(
        "--evaluation-seeds",
        type=int,
        nargs="+",
        default=[75, 175, 190],
        help="Independent task-generation seeds.",
    )
    parser.add_argument(
        "--episodes-per-seed",
        type=int,
        default=5,
        help="Deterministic evaluation episodes for each task seed.",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=None,
        help="Output path without extension; defaults inside checkpoint-dir.",
    )
    return parser.parse_args()


def _load_checkpoints(
    checkpoint_dir: Path,
    algorithms: Sequence[str] = DEFAULT_ALGORITHMS,
) -> Dict[str, Dict[str, object]]:
    """Load and validate the requested checkpoint payloads."""
    checkpoints: Dict[str, Dict[str, object]] = {}
    for display_name in algorithms:
        filename = CHECKPOINT_FILES[display_name]
        checkpoint_path = checkpoint_dir / filename
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"missing checkpoint: {checkpoint_path}")
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=True
        )
        agent_class_name = str(checkpoint.get("agent_class", ""))
        if agent_class_name not in AGENT_CLASSES:
            raise ValueError(
                f"unsupported agent class {agent_class_name!r} in {checkpoint_path}"
            )
        checkpoints[display_name] = checkpoint
    return checkpoints


def _checkpoint_scenario(
    checkpoints: Dict[str, Dict[str, object]],
) -> TopologyScenario:
    """Return the common topology scenario recorded by all checkpoints."""
    topology_configs = {
        (
            str(checkpoint.get("topology_metrics", {}).get("name", "")),
            int(checkpoint.get("topology_metrics", {}).get("topology_seed", 2026)),
            str(
                checkpoint.get("topology_metrics", {}).get(
                    "server_profile", "uniform"
                )
            ),
        )
        for checkpoint in checkpoints.values()
    }
    if len(topology_configs) != 1:
        raise ValueError(f"checkpoints disagree on topology: {topology_configs}")
    scenario_name, topology_seed, server_profile = topology_configs.pop()
    if not scenario_name:
        raise ValueError("checkpoint topology name is missing")
    return get_topology_scenario(
        scenario_name,
        topology_seed=topology_seed,
        server_profile=server_profile,
    )


def _checkpoint_system_seed(
    checkpoints: Dict[str, Dict[str, object]],
) -> int:
    """Read the physical server seed shared by the checkpoint metadata."""
    seeds = {
        int(
            checkpoint["topology_metrics"]["physical_compute"][
                "actual_server_compute_seed"
            ]
        )
        for checkpoint in checkpoints.values()
    }
    if len(seeds) != 1:
        raise ValueError(f"checkpoints disagree on physical seed: {seeds}")
    return seeds.pop()


def _load_priority_model_and_order(
    data_loader: KolektorSDDLoader,
    model_name: str,
    seed: int,
) -> Tuple[torch.nn.Module, List[int]]:
    """Load/train one task graph model and infer one shared priority list."""
    confirmed = PAPER_PARAMS["confirmed"]
    provisional = PAPER_PARAMS["provisional_table2_needed"]
    set_seed(seed)
    priority_model = build_task_priority_model(
        model_name,
        num_features=TASK_PRIORITY_FEATURE_DIM,
        hidden_dim=int(confirmed["gcn_hidden_dim"]),
    )
    dag_sampler = make_priority_dag_sampler(
        data_loader,
        t_max=provisional["t_max"],
        e_max=provisional["e_max"],
        cpu_cycle_scale=provisional["task_cpu_cycle_scale"],
    )
    priority_model = load_or_train_priority_model(
        priority_model=priority_model,
        dag_sampler=dag_sampler,
        checkpoint_path=get_priority_checkpoint_path(model_name),
        epochs=int(provisional["gcn_pretrain_epochs"]),
        samples_per_epoch=int(provisional["gcn_samples_per_epoch"]),
        lr=float(confirmed["gcn_lr"]),
        model_label=model_name.upper(),
    )
    template_dag = dag_sampler()
    priority_order = build_priorities(
        {template_dag.id: template_dag}, priority_model
    )[template_dag.id]
    return priority_model, priority_order


def _build_system(
    scenario: TopologyScenario, system_seed: int
) -> Tuple[List[IndustrialDevice], List[EdgeServer], NetworkEnvironment]:
    """Reconstruct the physical system used by the saved run."""
    confirmed = PAPER_PARAMS["confirmed"]
    provisional = PAPER_PARAMS["provisional_table2_needed"]
    np.random.seed(system_seed)
    servers = build_servers_for_scenario(scenario, confirmed, provisional)
    np.random.seed(system_seed + 1)
    devices = build_devices_for_scenario(scenario, confirmed, provisional)
    network = NetworkEnvironment(
        bandwidth=confirmed["bandwidth_hz"],
        noise_power_dbm=confirmed["noise_power_dbm"],
    )
    return devices, servers, network


def _agent_config(
    checkpoint: Dict[str, object], device: str = "cpu"
) -> Dict[str, object]:
    """Reconstruct the agent constructor configuration from a checkpoint."""
    agent_class_name = str(checkpoint["agent_class"])
    kwargs = dict(checkpoint.get("agent_kwargs", {}))
    if agent_class_name in {"GraphGATMAPPOAgent", "GATMAAgent"}:
        kwargs["device"] = device
    return {
        "class": AGENT_CLASSES[agent_class_name],
        "kwargs": kwargs,
    }


def _evaluate(
    checkpoints: Dict[str, Dict[str, object]],
    scenario: TopologyScenario,
    system_seed: int,
    evaluation_seeds: Sequence[int],
    episodes_per_seed: int,
    data_loader: KolektorSDDLoader,
    priority_model: torch.nn.Module,
    priority_orders: Dict[str, List[int]],
    device: str = "cpu",
) -> List[Dict[str, object]]:
    """Evaluate every checkpoint/order pair under identical conditions."""
    rows: List[Dict[str, object]] = []
    for model_name, checkpoint in checkpoints.items():
        for priority_name, priority_order in priority_orders.items():
            for evaluation_seed in evaluation_seeds:
                devices, servers, network = _build_system(scenario, system_seed)
                history = evaluate_algorithm_checkpoint(
                    agent_config=_agent_config(checkpoint, device),
                    checkpoint=checkpoint,
                    devices=devices,
                    servers=servers,
                    network_env=network,
                    data_loader=data_loader,
                    priority_model=priority_model,
                    num_episodes=episodes_per_seed,
                    experiment_seed=evaluation_seed,
                    priority_mode="gat",
                    topology_scenario=scenario,
                    fixed_priority_order=priority_order,
                )
                for episode_index in range(episodes_per_seed):
                    row: Dict[str, object] = {
                        "model": model_name,
                        "priority": priority_name,
                        "priority_order": list(priority_order),
                        "evaluation_seed": int(evaluation_seed),
                        "episode": episode_index + 1,
                    }
                    for metric in METRICS:
                        row[metric] = float(history[metric][episode_index])
                    rows.append(row)
    return rows


def _summarize(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    """Aggregate seed means using mean and sample standard deviation."""
    summaries: List[Dict[str, object]] = []
    groups = sorted({(str(row["model"]), str(row["priority"])) for row in rows})
    for model_name, priority_name in groups:
        group_rows = [
            row
            for row in rows
            if row["model"] == model_name and row["priority"] == priority_name
        ]
        summary: Dict[str, object] = {
            "model": model_name,
            "priority": priority_name,
            "priority_order": group_rows[0]["priority_order"],
            "seed_count": len(
                {int(row["evaluation_seed"]) for row in group_rows}
            ),
            "episode_count": len(group_rows),
        }
        for metric in METRICS:
            seed_means = np.asarray(
                [
                    np.mean(
                        [
                            float(row[metric])
                            for row in group_rows
                            if int(row["evaluation_seed"]) == evaluation_seed
                        ]
                    )
                    for evaluation_seed in sorted(
                        {int(row["evaluation_seed"]) for row in group_rows}
                    )
                ],
                dtype=float,
            )
            summary[f"{metric}_mean"] = float(np.mean(seed_means))
            summary[f"{metric}_std"] = float(
                np.std(seed_means, ddof=1) if len(seed_means) > 1 else 0.0
            )
        summaries.append(summary)
    return summaries


def _compare_priorities(
    summaries: Sequence[Dict[str, object]],
) -> List[Dict[str, object]]:
    """Return inferred-minus-default metric deltas for every policy."""
    summary_by_key = {
        (str(row["model"]), str(row["priority"])): row for row in summaries
    }
    comparisons: List[Dict[str, object]] = []
    model_names = sorted({str(row["model"]) for row in summaries})
    for model_name in model_names:
        default_row = summary_by_key[(model_name, "default")]
        inferred_row = summary_by_key[(model_name, "inferred")]
        comparison: Dict[str, object] = {"model": model_name}
        for metric in METRICS:
            default_value = float(default_row[f"{metric}_mean"])
            inferred_value = float(inferred_row[f"{metric}_mean"])
            delta = inferred_value - default_value
            comparison[f"{metric}_delta"] = delta
            comparison[f"{metric}_delta_percent"] = (
                100.0 * delta / abs(default_value)
                if abs(default_value) > 1e-12
                else 0.0
            )
        comparisons.append(comparison)
    return comparisons


def _write_outputs(
    output_prefix: Path,
    metadata: Dict[str, object],
    rows: Sequence[Dict[str, object]],
    summaries: Sequence[Dict[str, object]],
    comparisons: Sequence[Dict[str, object]],
) -> Tuple[Path, Path]:
    """Write detailed JSON and summary CSV outputs."""
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = output_prefix.with_suffix(".json")
    csv_path = output_prefix.with_suffix(".csv")
    json_path.write_text(
        json.dumps(
            {
                "metadata": metadata,
                "summary": summaries,
                "inferred_minus_default": comparisons,
                "episodes": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    with csv_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)
    return json_path, csv_path


def _print_summary(
    summaries: Sequence[Dict[str, object]],
    comparisons: Sequence[Dict[str, object]],
) -> None:
    """Print the primary inference metrics as a compact table."""
    print("\nInference comparison (mean ± sample SD across evaluation seeds)")
    print(
        f"{'Model':<31} {'Priority':<10} "
        f"{'Reward':>18} {'Delay (s)':>18} {'Energy (J)':>18}"
    )
    for row in summaries:
        print(
            f"{str(row['model']):<31} {str(row['priority']):<10} "
            f"{row['reward_mean']:8.4f} ± {row['reward_std']:<7.4f} "
            f"{row['delay_mean']:8.4f} ± {row['delay_std']:<7.4f} "
            f"{row['energy_mean']:8.4f} ± {row['energy_std']:<7.4f}"
        )
    print("\nInferred − default priority")
    print(f"{'Model':<31} {'Δ reward':>12} {'Δ delay':>12} {'Δ energy':>12}")
    for row in comparisons:
        print(
            f"{str(row['model']):<31} "
            f"{row['reward_delta']:12.4f} "
            f"{row['delay_delta']:12.4f} "
            f"{row['energy_delta']:12.4f}"
        )


def main() -> None:
    """Run deterministic checkpoint inference and save the comparison."""
    args = parse_args()
    checkpoints = _load_checkpoints(args.checkpoint_dir, args.algorithms)
    scenario = _checkpoint_scenario(checkpoints)
    metadata_system_seed = _checkpoint_system_seed(checkpoints)
    system_seed = (
        metadata_system_seed if args.system_seed is None else args.system_seed
    )
    data_loader = KolektorSDDLoader(args.dataset_path)
    priority_model, inferred_order = _load_priority_model_and_order(
        data_loader, args.priority_model, args.priority_seed
    )
    default_order = sorted(inferred_order)
    priority_orders = {
        "default": default_order,
        "inferred": inferred_order,
    }
    print(f"Default priority:  {default_order}")
    print(f"Inferred priority: {inferred_order}")
    print(
        f"Scenario={scenario.name}, system_seed={system_seed}, "
        f"evaluation_seeds={args.evaluation_seeds}"
    )
    rows = _evaluate(
        checkpoints=checkpoints,
        scenario=scenario,
        system_seed=system_seed,
        evaluation_seeds=args.evaluation_seeds,
        episodes_per_seed=args.episodes_per_seed,
        data_loader=data_loader,
        priority_model=priority_model,
        priority_orders=priority_orders,
        device=args.device,
    )
    summaries = _summarize(rows)
    comparisons = _compare_priorities(summaries)
    output_prefix = args.output_prefix or (
        args.checkpoint_dir / "inference_priority_comparison"
    )
    metadata = {
        "checkpoint_dir": str(args.checkpoint_dir.resolve()),
        "scenario": scenario.name,
        "system_seed": int(system_seed),
        "checkpoint_metadata_system_seed": int(metadata_system_seed),
        "evaluation_seeds": list(args.evaluation_seeds),
        "episodes_per_seed": int(args.episodes_per_seed),
        "time_slots": int(PAPER_PARAMS["confirmed"]["time_slots"]),
        "requested_device": args.device,
        "priority_model": args.priority_model,
        "priority_model_checkpoint": get_priority_checkpoint_path(
            args.priority_model
        ),
        "checkpoint_reported_training_seeds": {
            model_name: checkpoint.get("experiment_seed")
            for model_name, checkpoint in checkpoints.items()
        },
        "default_priority_order": default_order,
        "inferred_priority_order": inferred_order,
        "evaluation_mode": "deterministic greedy, no optimizer updates",
        "interpretation": (
            "Counterfactual evaluation of old policy checkpoints under the "
            "current task profile and priority order; not retrained performance."
        ),
    }
    json_path, csv_path = _write_outputs(
        output_prefix, metadata, rows, summaries, comparisons
    )
    _print_summary(summaries, comparisons)
    print(f"\nSaved JSON: {json_path}")
    print(f"Saved CSV:  {csv_path}")


if __name__ == "__main__":
    main()
