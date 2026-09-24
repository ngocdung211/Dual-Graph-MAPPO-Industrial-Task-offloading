"""Tune unmasked task-offloading agents with persistent Optuna studies."""

from __future__ import annotations

import argparse
import gc
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import torch

try:
    import optuna
except ImportError:  # Optional dependency used only by this entrypoint.
    optuna = None

from baselines.graph_gat_mappo import GraphGATMAPPOAgent
from baselines.mappo import MAPPOAgent
from dataset.data_loader import KolektorSDDLoader
from environment.network_env import NetworkEnvironment
from environment.system_model import EdgeServer, IndustrialDevice
from models.maddpg import EpsilonATNMADDPGAgent
from run_comparision import (
    build_devices_for_scenario,
    build_servers_for_scenario,
    evaluate_algorithm_checkpoint,
    set_seed,
    summarize_physical_compute,
    train_algorithm,
)
from utils.task_priority.experiment_setup import (
    TASK_PRIORITY_FEATURE_DIM,
    build_priorities,
    build_task_priority_model,
    get_priority_checkpoint_path,
    make_priority_dag_sampler,
)
from utils.paper_config import PAPER_PARAMS
from utils.task_priority.priority_model_training import load_or_train_priority_model
from utils.topology.scenarios import (
    TopologyScenario,
    available_topology_scenario_names,
    compute_topology_metrics,
    get_topology_scenario,
)


MODEL_E_ATN_MADDPG = "e-ATN-MADDPG"
MODEL_MAPPO = "MAPPO"
MODEL_GRAPH_GAT_WARMUP_MAPPO = "Graph-GAT Warmup MAPPO"
TUNABLE_MODELS = (
    MODEL_E_ATN_MADDPG,
    MODEL_MAPPO,
    MODEL_GRAPH_GAT_WARMUP_MAPPO,
)


@dataclass(frozen=True)
class TuningResources:
    """Resources shared across sequential trials in one tuning process."""

    scenario: TopologyScenario
    data_loader: KolektorSDDLoader
    priority_model: torch.nn.Module
    priority_mode: str
    fixed_priority_order: List[int]


def parse_args() -> argparse.Namespace:
    """Parse Optuna tuning arguments."""
    parser = argparse.ArgumentParser(
        description="Tune unmasked e-ATN-MADDPG, MAPPO, and Graph-GAT MAPPO."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=TUNABLE_MODELS,
        default=list(TUNABLE_MODELS),
        help="Exact unmasked model names to tune.",
    )
    parser.add_argument(
        "--phase",
        choices=("search", "rerank", "final", "transfer"),
        default="search",
        help=(
            "search runs Optuna, rerank trains each top-three configuration, "
            "final trains the rerank winner, and transfer trains Graph-GAT "
            "with a MAPPO PPO core"
        ),
    )
    parser.add_argument(
        "--topology-scenario",
        choices=available_topology_scenario_names(),
        default="medium_20d_6s",
    )
    parser.add_argument("--trials", type=int, default=25)
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help=(
            "Override the phase budget (search/transfer=150, rerank=500, "
            "final=1000)."
        ),
    )
    parser.add_argument("--train-seed", type=int, default=75)
    parser.add_argument(
        "--train-seeds", nargs="+", type=int, default=[75, 76, 77]
    )
    parser.add_argument(
        "--validation-seeds", nargs="+", type=int, default=[175, 176]
    )
    parser.add_argument("--validation-episodes", type=int, default=5)
    parser.add_argument("--report-interval", type=int, default=10)
    parser.add_argument("--pruning-warmup-episodes", type=int, default=50)
    parser.add_argument("--graph-gat-device", default="auto")
    parser.add_argument("--dataset-path", default="dataset/KolektorSDD/")
    parser.add_argument(
        "--output-dir",
        default="",
        help="Study output folder. A timestamped folder is used when omitted.",
    )
    parser.add_argument(
        "--input-dir",
        default="",
        help=(
            "Folder containing search/rerank JSON. Defaults to output-dir; "
            "required for rerank/final when output-dir is omitted."
        ),
    )
    parser.add_argument(
        "--storage",
        default="",
        help="Optional Optuna storage URL. Defaults to SQLite in output-dir.",
    )
    parser.add_argument(
        "--mappo-params-path",
        default="",
        help="MAPPO best-params JSON used by the Graph-GAT transfer phase.",
    )
    parser.add_argument(
        "--graph-params-path",
        default="",
        help=(
            "Graph-GAT best-params JSON supplying encoder and warmup settings "
            "for the transfer phase."
        ),
    )
    parser.add_argument("--timeout-seconds", type=int, default=None)
    return parser.parse_args()


def phase_episode_budget(phase: str, override: int | None) -> int:
    """Return the agreed episode budget for one tuning phase."""
    if override is not None:
        if override <= 0:
            raise ValueError("episodes must be positive")
        return override
    return {
        "search": 150,
        "rerank": 500,
        "final": 1000,
        "transfer": 150,
    }[phase]


def load_best_params(path: str, expected_model: str) -> Dict[str, object]:
    """Load and validate one tuning best-params JSON file."""
    input_path = Path(path)
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    if payload.get("model") != expected_model:
        raise ValueError(
            f"expected {expected_model} parameters in {input_path}, "
            f"found {payload.get('model')!r}"
        )
    params = payload.get("params")
    if not isinstance(params, dict):
        raise ValueError(f"missing params object in {input_path}")
    return params


def build_graph_transfer_params(
    mappo_params: Dict[str, object],
    graph_params: Dict[str, object],
) -> Dict[str, object]:
    """Combine MAPPO PPO settings with Graph-GAT-specific settings."""
    return {
        "actor_lr": mappo_params["actor_lr"],
        "critic_lr": mappo_params["critic_lr"],
        "gamma": mappo_params["gamma"],
        "clip_param": mappo_params["clip_param"],
        "ppo_epochs": mappo_params["ppo_epochs"],
        "entropy_coef": mappo_params["entropy_coef"],
        "value_loss_coef": mappo_params["value_loss_coef"],
        "max_grad_norm": mappo_params["max_grad_norm"],
        "hidden_dim": mappo_params["hidden_dim"],
        "encoder_lr": graph_params["encoder_lr"],
        "warmup_episodes": graph_params["warmup_episodes"],
        "warmup_updates_per_step": graph_params[
            "warmup_updates_per_step"
        ],
        "warmup_lr": graph_params["warmup_lr"],
    }


def build_default_trial_params(model_name: str) -> Dict[str, object]:
    """Return the current model configuration in Optuna parameter names."""
    confirmed = PAPER_PARAMS["confirmed"]
    provisional = PAPER_PARAMS["provisional_table2_needed"]
    if model_name == MODEL_E_ATN_MADDPG:
        return {
            "actor_lr": provisional["maddpg_actor_lr"],
            "critic_lr": provisional["maddpg_critic_lr"],
            "gamma": provisional["gamma"],
            "tau": provisional["tau_soft_update"],
            "batch_size": int(provisional["batch_size"]),
            "hidden_dim": int(provisional["maddpg_hidden_dim"]),
            "epsilon_final": provisional["maddpg_epsilon_final"],
            "exploration_fraction": provisional[
                "maddpg_exploration_fraction"
            ],
        }
    if model_name == MODEL_MAPPO:
        return {
            "actor_lr": confirmed["rl_lr"],
            "critic_lr": confirmed["rl_lr"],
            "gamma": provisional["gamma"],
            "clip_param": provisional["mappo_clip_param"],
            "ppo_epochs": int(provisional["mappo_ppo_epochs"]),
            "entropy_coef": provisional["mappo_entropy_coef"],
            "value_loss_coef": provisional["mappo_value_loss_coef"],
            "max_grad_norm": provisional["mappo_max_grad_norm"],
            "hidden_dim": int(provisional["mappo_hidden_dim"]),
        }
    if model_name == MODEL_GRAPH_GAT_WARMUP_MAPPO:
        return {
            "lr": confirmed["rl_lr"],
            "encoder_lr": confirmed["rl_lr"],
            "clip_param": provisional["graph_gat_clip_param"],
            "ppo_epochs": int(provisional["graph_gat_ppo_epochs"]),
            "entropy_coef": provisional["graph_gat_entropy_coef"],
            "value_loss_coef": provisional["graph_gat_value_loss_coef"],
            "max_grad_norm": provisional["graph_gat_max_grad_norm"],
            "warmup_episodes": int(
                provisional["graph_gat_topology_warmup_episodes"]
            ),
            "warmup_updates_per_step": int(
                provisional["graph_gat_topology_warmup_updates_per_step"]
            ),
            "warmup_lr": provisional["graph_gat_topology_warmup_lr"],
        }
    raise ValueError(f"unsupported model: {model_name}")


def suggest_agent_config(
    model_name: str, trial: Any, graph_gat_device: str
) -> Dict[str, object]:
    """Suggest one model-specific configuration from an Optuna-like trial."""
    if model_name == MODEL_E_ATN_MADDPG:
        params = {
            "actor_lr": trial.suggest_float(
                "actor_lr", 1e-5, 5e-4, log=True
            ),
            "critic_lr": trial.suggest_float(
                "critic_lr", 5e-5, 1e-3, log=True
            ),
            "gamma": trial.suggest_float("gamma", 0.95, 0.999),
            "tau": trial.suggest_float("tau", 1e-3, 2e-2, log=True),
            "batch_size": trial.suggest_categorical(
                "batch_size", [64, 128, 256]
            ),
            "hidden_dim": trial.suggest_categorical(
                "hidden_dim", [64, 128, 256]
            ),
            "epsilon_final": trial.suggest_float(
                "epsilon_final", 0.01, 0.10
            ),
            "exploration_fraction": trial.suggest_float(
                "exploration_fraction", 0.2, 0.6
            ),
        }
    elif model_name == MODEL_MAPPO:
        params = {
            "actor_lr": trial.suggest_float(
                "actor_lr", 1e-5, 5e-4, log=True
            ),
            "critic_lr": trial.suggest_float(
                "critic_lr", 1e-5, 5e-4, log=True
            ),
            "gamma": trial.suggest_float("gamma", 0.95, 0.999),
            "clip_param": trial.suggest_float("clip_param", 0.1, 0.3),
            "ppo_epochs": trial.suggest_categorical(
                "ppo_epochs", [2, 4, 6, 8]
            ),
            "entropy_coef": trial.suggest_float(
                "entropy_coef", 1e-4, 3e-2, log=True
            ),
            "value_loss_coef": trial.suggest_float(
                "value_loss_coef", 0.25, 1.0
            ),
            "max_grad_norm": trial.suggest_categorical(
                "max_grad_norm", [None, 0.3, 0.5, 1.0]
            ),
            "hidden_dim": trial.suggest_categorical(
                "hidden_dim", [64, 128]
            ),
        }
    elif model_name == MODEL_GRAPH_GAT_WARMUP_MAPPO:
        params = {
            "lr": trial.suggest_float("lr", 3e-5, 3e-4, log=True),
            "encoder_lr": trial.suggest_float(
                "encoder_lr", 1e-5, 1e-4, log=True
            ),
            "clip_param": trial.suggest_float("clip_param", 0.1, 0.3),
            "ppo_epochs": trial.suggest_categorical(
                "ppo_epochs", [2, 4, 6, 8]
            ),
            "entropy_coef": trial.suggest_float(
                "entropy_coef", 1e-4, 3e-2, log=True
            ),
            "value_loss_coef": trial.suggest_float(
                "value_loss_coef", 0.25, 1.0
            ),
            "max_grad_norm": trial.suggest_categorical(
                "max_grad_norm", [None, 0.3, 0.5, 1.0]
            ),
            "warmup_episodes": trial.suggest_categorical(
                "warmup_episodes", [5, 10, 20]
            ),
            "warmup_updates_per_step": trial.suggest_categorical(
                "warmup_updates_per_step", [1, 2, 4, 10]
            ),
            "warmup_lr": trial.suggest_float(
                "warmup_lr", 1e-4, 1e-3, log=True
            ),
        }
    else:
        raise ValueError(f"unsupported model: {model_name}")
    return build_agent_config_from_params(
        model_name, params, graph_gat_device
    )


def build_agent_config_from_params(
    model_name: str,
    params: Dict[str, object],
    graph_gat_device: str,
) -> Dict[str, object]:
    """Build an exact unmasked agent config from stored trial parameters."""
    if model_name == MODEL_E_ATN_MADDPG:
        epsilon_final = float(params["epsilon_final"])
        return {
            "class": EpsilonATNMADDPGAgent,
            "batch_size": int(params["batch_size"]),
            "kwargs": {
                "use_attention": True,
                "use_epsilon_greedy": True,
                "actor_lr": float(params["actor_lr"]),
                "critic_lr": float(params["critic_lr"]),
                "gamma": float(params["gamma"]),
                "tau": float(params["tau"]),
                "hidden_dim": int(params["hidden_dim"]),
                "epsilon_init": 1.0,
                "epsilon_min": epsilon_final,
                "epsilon_final": epsilon_final,
                "exploration_fraction": float(
                    params["exploration_fraction"]
                ),
            },
        }
    if model_name == MODEL_MAPPO:
        return {
            "class": MAPPOAgent,
            "kwargs": {
                "actor_lr": float(params["actor_lr"]),
                "critic_lr": float(params["critic_lr"]),
                "gamma": float(params["gamma"]),
                "clip_param": float(params["clip_param"]),
                "ppo_epochs": int(params["ppo_epochs"]),
                "entropy_coef": float(params["entropy_coef"]),
                "value_loss_coef": float(params["value_loss_coef"]),
                "max_grad_norm": params["max_grad_norm"],
                "hidden_dim": int(params["hidden_dim"]),
                "use_action_mask": False,
            },
        }
    if model_name == MODEL_GRAPH_GAT_WARMUP_MAPPO:
        shared_lr = params.get("lr")
        actor_lr = params.get("actor_lr", shared_lr)
        critic_lr = params.get("critic_lr", shared_lr)
        if actor_lr is None or critic_lr is None:
            raise ValueError(
                "Graph-GAT params require lr or both actor_lr and critic_lr"
            )
        return {
            "class": GraphGATMAPPOAgent,
            "kwargs": {
                "lr": float(actor_lr),
                "actor_lr": float(actor_lr),
                "critic_lr": float(critic_lr),
                "encoder_lr": float(params["encoder_lr"]),
                "gamma": float(params.get("gamma", 0.99)),
                "hidden_dim": int(params.get("hidden_dim", 64)),
                "embedding_dim": 64,
                "clip_param": float(params["clip_param"]),
                "ppo_epochs": int(params["ppo_epochs"]),
                "entropy_coef": float(params["entropy_coef"]),
                "value_loss_coef": float(params["value_loss_coef"]),
                "max_grad_norm": params["max_grad_norm"],
                "use_action_mask": False,
                "topology_warmup_episodes": int(params["warmup_episodes"]),
                "topology_warmup_updates_per_step": int(
                    params["warmup_updates_per_step"]
                ),
                "topology_warmup_lr": float(params["warmup_lr"]),
                "device": graph_gat_device,
            },
        }
    raise ValueError(f"unsupported model: {model_name}")


def load_tuning_resources(
    topology_scenario: str, dataset_path: str
) -> TuningResources:
    """Load dataset and fixed task-priority model once per process."""
    confirmed = PAPER_PARAMS["confirmed"]
    provisional = PAPER_PARAMS["provisional_table2_needed"]
    scenario = get_topology_scenario(topology_scenario)
    data_loader = KolektorSDDLoader(dataset_path=dataset_path)
    priority_mode = str(provisional["priority_model"]).lower()
    priority_model = build_task_priority_model(
        priority_mode,
        num_features=TASK_PRIORITY_FEATURE_DIM,
        hidden_dim=int(confirmed["gcn_hidden_dim"]),
    )
    priority_dag_sampler = make_priority_dag_sampler(
        data_loader,
        t_max=provisional["t_max"],
        e_max=provisional["e_max"],
        cpu_cycle_scale=provisional["task_cpu_cycle_scale"],
    )
    priority_model = load_or_train_priority_model(
        priority_model=priority_model,
        dag_sampler=priority_dag_sampler,
        checkpoint_path=get_priority_checkpoint_path(priority_mode),
        epochs=int(provisional["gcn_pretrain_epochs"]),
        samples_per_epoch=int(provisional["gcn_samples_per_epoch"]),
        lr=confirmed["gcn_lr"],
        model_label=priority_mode.upper(),
    )
    priority_template = priority_dag_sampler()
    fixed_priority_order = build_priorities(
        {priority_template.id: priority_template}, priority_model
    )[priority_template.id]
    return TuningResources(
        scenario=scenario,
        data_loader=data_loader,
        priority_model=priority_model,
        priority_mode=priority_mode,
        fixed_priority_order=fixed_priority_order,
    )


def build_trial_system(
    resources: TuningResources, seed: int
) -> tuple[
    List[IndustrialDevice],
    List[EdgeServer],
    NetworkEnvironment,
    Dict[str, object],
]:
    """Build one reproducible physical system for a training or eval seed."""
    confirmed = PAPER_PARAMS["confirmed"]
    provisional = PAPER_PARAMS["provisional_table2_needed"]
    set_seed(seed)
    np.random.seed(seed)
    servers = build_servers_for_scenario(
        resources.scenario, confirmed, provisional
    )
    np.random.seed(seed + 1)
    devices = build_devices_for_scenario(
        resources.scenario, confirmed, provisional
    )
    topology_metrics = compute_topology_metrics(resources.scenario)
    topology_metrics["physical_compute"] = summarize_physical_compute(
        devices,
        servers,
        device_seed=seed + 1,
        server_seed=seed,
    )
    network_env = NetworkEnvironment(
        bandwidth=confirmed["bandwidth_hz"],
        noise_power_dbm=confirmed["noise_power_dbm"],
    )
    return devices, servers, network_env, topology_metrics


def mean_metric(histories: Sequence[Dict[str, List[float]]], name: str) -> float:
    """Return the mean of one metric across evaluation histories."""
    values = [value for history in histories for value in history[name]]
    return float(np.mean(values))


def _safe_model_name(model_name: str) -> str:
    """Return a filesystem-safe lowercase model label."""
    return model_name.lower().replace(" ", "_").replace("-", "_")


def _write_study_outputs(
    study: Any, output_dir: Path, model_name: str
) -> None:
    """Write resumable study summaries without changing paper defaults."""
    safe_model_name = _safe_model_name(model_name)
    study.trials_dataframe().to_csv(
        output_dir / f"{safe_model_name}_trials.csv", index=False
    )
    completed_trials = [
        trial
        for trial in study.trials
        if trial.state.name == "COMPLETE" and trial.value is not None
    ]
    top_trials = sorted(
        completed_trials,
        key=lambda trial: float(trial.value),
        reverse=True,
    )[:3]
    top_payload = [
        {
            "number": trial.number,
            "value": trial.value,
            "params": trial.params,
            "metrics": trial.user_attrs,
        }
        for trial in top_trials
    ]
    (output_dir / f"{safe_model_name}_top_trials.json").write_text(
        json.dumps(top_payload, indent=2), encoding="utf-8"
    )
    best_trial = study.best_trial
    best_payload = {
        "model": model_name,
        "study_name": study.study_name,
        "trial": best_trial.number,
        "value": best_trial.value,
        "params": best_trial.params,
        "metrics": best_trial.user_attrs,
    }
    (output_dir / f"{safe_model_name}_best_params.json").write_text(
        json.dumps(best_payload, indent=2), encoding="utf-8"
    )


def _load_stage_candidates(
    model_name: str, phase: str, input_dir: Path
) -> List[Dict[str, object]]:
    """Load top search trials or the rerank winner for a later phase."""
    safe_model_name = _safe_model_name(model_name)
    if phase == "rerank":
        input_path = input_dir / f"{safe_model_name}_top_trials.json"
        candidates = json.loads(input_path.read_text(encoding="utf-8"))
        if not candidates:
            raise ValueError(f"no completed search trials in {input_path}")
        return candidates[:3]

    input_path = input_dir / f"{safe_model_name}_rerank.json"
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    winner = payload.get("winner")
    if winner is None:
        raise ValueError(f"no rerank winner in {input_path}")
    return [winner]


def _aggregate_validation_metrics(
    histories: Sequence[Dict[str, List[float]]],
) -> Dict[str, float]:
    """Aggregate the deterministic metrics used to compare candidates."""
    return {
        metric_name: mean_metric(histories, metric_name)
        for metric_name in (
            "reward",
            "delay",
            "energy",
            "edge_ratio",
            "resolved_edge_ratio",
            "penalty_count",
        )
    }


def run_seeded_stage(
    model_name: str,
    phase: str,
    args: argparse.Namespace,
    resources: TuningResources,
    input_dir: Path,
    output_dir: Path,
) -> Dict[str, object]:
    """Run one configured rerank, final, or transfer training stage."""
    if phase == "transfer":
        mappo_params = load_best_params(
            args.mappo_params_path, MODEL_MAPPO
        )
        graph_params = load_best_params(
            args.graph_params_path, MODEL_GRAPH_GAT_WARMUP_MAPPO
        )
        candidates = [
            {
                "number": "MAPPO-PPO-transfer",
                "params": build_graph_transfer_params(
                    mappo_params, graph_params
                ),
            }
        ]
    else:
        candidates = _load_stage_candidates(model_name, phase, input_dir)
    candidate_results = []
    for candidate_index, candidate in enumerate(candidates, start=1):
        params = candidate["params"]
        agent_config = build_agent_config_from_params(
            model_name, params, args.graph_gat_device
        )
        seed_results = []
        for train_seed in args.train_seeds:
            print(
                f"[{model_name}] {phase} candidate={candidate_index}/"
                f"{len(candidates)} train_seed={train_seed}"
            )
            devices, servers, network_env, topology_metrics = (
                build_trial_system(resources, train_seed)
            )
            history, checkpoint = train_algorithm(
                model_name,
                agent_config,
                devices,
                servers,
                network_env,
                resources.data_loader,
                resources.priority_model,
                num_episodes=args.episodes,
                priority_mode=resources.priority_mode,
                topology_scenario=resources.scenario,
                topology_metrics=topology_metrics,
                experiment_seed=train_seed,
                fixed_priority_order=resources.fixed_priority_order,
                show_progress=True,
            )
            if checkpoint is None:
                raise RuntimeError("trainable run did not produce a checkpoint")

            validation_histories = []
            for validation_seed in args.validation_seeds:
                eval_devices, eval_servers, eval_network, _ = (
                    build_trial_system(resources, validation_seed)
                )
                validation_histories.append(
                    evaluate_algorithm_checkpoint(
                        agent_config=agent_config,
                        checkpoint=checkpoint,
                        devices=eval_devices,
                        servers=eval_servers,
                        network_env=eval_network,
                        data_loader=resources.data_loader,
                        priority_model=resources.priority_model,
                        num_episodes=args.validation_episodes,
                        experiment_seed=validation_seed,
                        priority_mode=resources.priority_mode,
                        topology_scenario=resources.scenario,
                        fixed_priority_order=resources.fixed_priority_order,
                    )
                )
            seed_results.append(
                {
                    "train_seed": train_seed,
                    "train_final_20_reward": float(
                        np.mean(history["reward"][-20:])
                    ),
                    "validation": _aggregate_validation_metrics(
                        validation_histories
                    ),
                }
            )
            if phase in ("final", "transfer"):
                checkpoint_path = output_dir / (
                    f"{_safe_model_name(model_name)}_{phase}_seed"
                    f"{train_seed}_checkpoint.pt"
                )
                torch.save(checkpoint, checkpoint_path)
            del devices, servers, network_env, checkpoint
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        validation_rewards = [
            float(seed_result["validation"]["reward"])
            for seed_result in seed_results
        ]
        candidate_results.append(
            {
                "source_trial": candidate.get("number"),
                "params": params,
                "validation_reward_mean": float(
                    np.mean(validation_rewards)
                ),
                "validation_reward_std": float(np.std(validation_rewards)),
                "seed_results": seed_results,
            }
        )

    ranked_results = sorted(
        candidate_results,
        key=lambda result: (
            -float(result["validation_reward_mean"]),
            float(result["validation_reward_std"]),
        ),
    )
    payload = {
        "model": model_name,
        "phase": phase,
        "episodes": args.episodes,
        "train_seeds": args.train_seeds,
        "validation_seeds": args.validation_seeds,
        "validation_episodes": args.validation_episodes,
        "candidates": ranked_results,
        "winner": ranked_results[0],
    }
    if phase == "transfer":
        payload["parameter_sources"] = {
            "mappo_ppo": args.mappo_params_path,
            "graph_encoder_warmup": args.graph_params_path,
        }
    output_path = output_dir / (
        f"{_safe_model_name(model_name)}_{phase}.json"
    )
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def run_study(
    model_name: str,
    args: argparse.Namespace,
    resources: TuningResources,
    output_dir: Path,
    storage_url: str,
) -> Any:
    """Create or resume one model-specific Optuna study."""
    if optuna is None:
        raise RuntimeError(
            "Optuna is not installed. Run: pip install -r requirements-optuna.txt"
        )
    study_name = (
        f"{model_name}-{resources.scenario.name}-"
        f"train{args.train_seed}"
    )
    sampler = optuna.samplers.TPESampler(
        seed=args.train_seed,
        multivariate=True,
    )
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5,
        n_warmup_steps=args.pruning_warmup_episodes,
        interval_steps=args.report_interval,
    )
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_url,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )
    if not study.trials:
        study.enqueue_trial(build_default_trial_params(model_name))

    best_score = study.best_value if study.best_trials else float("-inf")

    def objective(trial: Any) -> float:
        nonlocal best_score
        agent_config = suggest_agent_config(
            model_name, trial, args.graph_gat_device
        )
        devices, servers, network_env, topology_metrics = build_trial_system(
            resources, args.train_seed
        )

        def report_episode(
            episode_number: int, history: Dict[str, List[float]]
        ) -> None:
            if (
                episode_number % args.report_interval != 0
                and episode_number != args.episodes
            ):
                return
            rolling_reward = float(np.mean(history["reward"][-20:]))
            trial.report(rolling_reward, step=episode_number)
            if trial.should_prune():
                raise optuna.TrialPruned()

        try:
            history, checkpoint = train_algorithm(
                model_name,
                agent_config,
                devices,
                servers,
                network_env,
                resources.data_loader,
                resources.priority_model,
                num_episodes=args.episodes,
                priority_mode=resources.priority_mode,
                topology_scenario=resources.scenario,
                topology_metrics=topology_metrics,
                experiment_seed=args.train_seed,
                fixed_priority_order=resources.fixed_priority_order,
                episode_callback=report_episode,
                show_progress=False,
            )
            if checkpoint is None:
                raise RuntimeError("trainable trial did not produce a checkpoint")
            validation_histories = []
            for validation_seed in args.validation_seeds:
                eval_devices, eval_servers, eval_network, _ = build_trial_system(
                    resources, validation_seed
                )
                validation_histories.append(
                    evaluate_algorithm_checkpoint(
                        agent_config=agent_config,
                        checkpoint=checkpoint,
                        devices=eval_devices,
                        servers=eval_servers,
                        network_env=eval_network,
                        data_loader=resources.data_loader,
                        priority_model=resources.priority_model,
                        num_episodes=args.validation_episodes,
                        experiment_seed=validation_seed,
                        priority_mode=resources.priority_mode,
                        topology_scenario=resources.scenario,
                        fixed_priority_order=resources.fixed_priority_order,
                    )
                )

            validation_reward = mean_metric(
                validation_histories, "reward"
            )
            trial.set_user_attr(
                "train_final_20_reward",
                float(np.mean(history["reward"][-20:])),
            )
            for metric_name in (
                "reward",
                "delay",
                "energy",
                "edge_ratio",
                "resolved_edge_ratio",
                "penalty_count",
            ):
                trial.set_user_attr(
                    f"validation_{metric_name}",
                    mean_metric(validation_histories, metric_name),
                )
            trial.set_user_attr(
                "final_epsilon", float(history["epsilon"][-1])
            )
            trial.set_user_attr(
                "update_rounds",
                float(sum(history["maddpg_update_rounds"])),
            )
            if validation_reward > best_score:
                safe_name = model_name.replace(" ", "_")
                checkpoint_path = output_dir / f"{safe_name}_best_checkpoint.pt"
                torch.save(checkpoint, checkpoint_path)
                best_score = validation_reward
            return validation_reward
        finally:
            del devices, servers, network_env
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    finished_trial_count = sum(
        trial.state.is_finished() for trial in study.trials
    )
    remaining_trial_count = max(args.trials - finished_trial_count, 0)
    if remaining_trial_count:
        study.optimize(
            objective,
            n_trials=remaining_trial_count,
            timeout=args.timeout_seconds,
            n_jobs=1,
            gc_after_trial=True,
        )
    _write_study_outputs(study, output_dir, model_name)
    return study


def main() -> None:
    """Run search, reranking, final training, or Graph-GAT transfer."""
    args = parse_args()
    if args.phase == "transfer":
        if args.models != [MODEL_GRAPH_GAT_WARMUP_MAPPO]:
            raise SystemExit(
                "transfer phase requires --models \"Graph-GAT Warmup MAPPO\""
            )
        if not args.mappo_params_path or not args.graph_params_path:
            raise SystemExit(
                "transfer phase requires --mappo-params-path and "
                "--graph-params-path"
            )
    if args.phase == "search" and optuna is None:
        raise SystemExit(
            "Optuna is not installed. Run: pip install -r requirements-optuna.txt"
        )
    args.episodes = phase_episode_budget(args.phase, args.episodes)
    if args.output_dir:
        output_dir = Path(args.output_dir)
    elif args.input_dir:
        output_dir = Path(args.input_dir)
    elif args.phase == "search":
        output_dir = Path(
            "results",
            "optuna",
            datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        )
    else:
        raise SystemExit(
            "--input-dir or --output-dir is required outside search"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    input_dir = Path(args.input_dir) if args.input_dir else output_dir
    resources = load_tuning_resources(
        args.topology_scenario, args.dataset_path
    )
    print(
        f"Tuning phase={args.phase}: topology={resources.scenario.name}, "
        f"episodes={args.episodes}, "
        f"output={output_dir}"
    )
    if args.phase == "search":
        storage_url = args.storage or (
            f"sqlite:///{(output_dir / 'optuna.db').as_posix()}"
        )
        for model_name in args.models:
            study = run_study(
                model_name,
                args,
                resources,
                output_dir,
                storage_url,
            )
            print(
                f"[{model_name}] best trial={study.best_trial.number}, "
                f"validation reward={study.best_value:.6f}"
            )
        return

    for model_name in args.models:
        payload = run_seeded_stage(
            model_name=model_name,
            phase=args.phase,
            args=args,
            resources=resources,
            input_dir=input_dir,
            output_dir=output_dir,
        )
        winner = payload["winner"]
        print(
            f"[{model_name}] {args.phase} validation reward="
            f"{winner['validation_reward_mean']:.6f} +/- "
            f"{winner['validation_reward_std']:.6f}"
        )


if __name__ == "__main__":
    main()
