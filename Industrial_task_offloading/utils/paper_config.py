"""Centralized paper-reproduction parameters and status flags.

This module keeps one source of truth for experiment settings and makes it
explicit which values are confirmed from the paper text versus provisional
defaults that still need Table II confirmation.
"""

from typing import Dict


PAPER_PARAMS: Dict[str, Dict[str, float]] = {
    "confirmed": {
        # Topology / scenario
        "num_devices": 10,
        "num_servers": 3,
        "time_slots": 50,
        "slot_duration_s": 1.0,
        "device_speed_mps": 1.0,
        "coverage_radius_m": 12.0,
        # Communication / system
        "bandwidth_hz": 10e6,
        "noise_power_dbm": -43,
        "device_tx_power_w": 0.5,
        "server_tx_power_w": 1.2,
        "device_energy_coeff": 1e-28,
        "server_energy_coeff": 1e-27,
        # Learning settings confirmed in paper text
        "gcn_hidden_dim": 32,
        "gcn_lr": 0.01,
        "rl_lr": 0.0001,
        "train_episodes_full": 1000,
    },
    "provisional_table2_needed": {
        # Pending Table II verification (kept as current project defaults).
        "gamma": 0.99,
        "batch_size": 64,
        "replay_buffer_capacity": 100000,
        "comparison_full_episodes": 500,
        "baseline_evaluation_episodes": 5,
        "comparison_hyperparameters_dir": (
            "configs/hyperparameters/unmasked_penalty_0_5"
        ),
        "experiment_seed": 75,
        "diagnostic_interval_episodes": 250,
        "subslot_count": 200,
        "lambda1": 5.0,
        "lambda2": 5.0,
        "lambda3": 5.0,
        "lambda4": 5.0,
        "lambda5": 1.0,
        "p_out_value": -1.0,
        "local_estimation_error": 0.05,
        "edge_estimation_error": 0.05,
        "gcn_pretrain_epochs": 200,
        "gcn_samples_per_epoch": 32,
        "priority_model": "gat",
        "topology_scenario": "paper_10d_3s",
        "server_compute_power_min_ghz": 2.3,
        "server_compute_power_max_ghz": 2.5,
        "device_compute_power_min_ghz": 0.8,
        "device_compute_power_max_ghz": 1.2,
        "task_cpu_cycle_scale": 1.4,
        "maddpg_actor_lr": 0.0001,
        "maddpg_critic_lr": 0.0001,
        "maddpg_hidden_dim": 64,
        "maddpg_epsilon_final": 0.03,
        "maddpg_exploration_fraction": 0.4,
        "mappo_clip_param": 0.2,
        "mappo_ppo_epochs": 4,
        "mappo_entropy_coef": 0.01,
        "mappo_value_loss_coef": 1.0,
        "mappo_max_grad_norm": None,
        "mappo_hidden_dim": 64,
        "mappo_use_action_mask": True,
        # Fixed PPO estimator settings shared by every MAPPO-family agent.
        # They are deliberately not tuned so that enabling them changes all
        # variants identically.
        "ppo_gae_lambda": 0.95,
        "ppo_num_minibatches": 4,
        "graph_gat_hidden_dim": 64,
        "graph_gat_embedding_dim": 64,
        "graph_gat_clip_param": 0.2,
        "graph_gat_ppo_epochs": 4,
        "graph_gat_entropy_coef": 0.01,
        "graph_gat_value_loss_coef": 1.0,
        "graph_gat_max_grad_norm": None,
        "graph_gat_use_action_mask": True,
        "graph_gat_topology_warmup_episodes": 5,
        "graph_gat_topology_warmup_updates_per_step": 10,
        "graph_gat_topology_warmup_lr": 0.001,
        "graph_gat_device": "auto",
        "wandb_mode": "disabled",
        "wandb_project": "industrial-task-offloading",
        "wandb_entity": "",
        "epsilon_init": 1.0,
        "epsilon_min": 0.01,
        "epsilon_decay": 0.998,
        "tau_soft_update": 0.01,
        # Task constraints currently used as placeholders.
        "t_max": 1.0,
        "e_max": 1.0,
    },
}
