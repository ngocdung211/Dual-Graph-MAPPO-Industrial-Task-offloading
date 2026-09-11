"""Train and evaluate e-ATN-MADDPG on the DITEN environment."""

from typing import Dict, List, Sequence

import numpy as np
import torch
import time
from tqdm import trange

# Sửa lại import theo đúng cấu trúc thư mục
from environment.network_env import NetworkEnvironment
from environment.system_model import IndustrialDevice, EdgeServer
from environment.diten_env import DITENEnv
from models.replay_buffer import MultiAgentReplayBuffer
from models.maddpg import EpsilonATNMADDPGAgent
from dataset.data_loader import KolektorSDDLoader
from utils.plotter import DITENPlotter
from utils.priority_model_training import load_or_train_priority_model
from utils.experiment_setup import (
    TASK_PRIORITY_FEATURE_DIM,
    broadcast_priority_order,
    build_priorities,
    build_task_priority_model,
    generate_task_dags_for_episode,
    get_priority_checkpoint_path,
    make_priority_dag_sampler,
)
from utils.paper_config import PAPER_PARAMS
from utils.maddpg_training import update_maddpg_agents_from_buffer

import random

def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducible runs.

    Args:
        seed: Random seed value.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

def _collect_joint_actions(
    agents: Sequence[EpsilonATNMADDPGAgent], joint_state: np.ndarray
) -> List[int]:
    """Select one action per agent based on the current joint state.

    Args:
        agents: Agents selecting actions.
        joint_state: Joint state array shaped (num_agents, state_dim).

    Returns:
        List of integer actions, one per agent.
    """
    joint_actions: List[int] = []
    for agent_index, agent in enumerate(agents):
        agent_state = torch.FloatTensor(joint_state[agent_index])
        action = agent.select_action(agent_state)
        joint_actions.append(action)
    return joint_actions


def _update_agents_from_buffer(
    agents: Sequence[EpsilonATNMADDPGAgent],
    replay_buffer: MultiAgentReplayBuffer,
    batch_size: int,
    gamma: float,
) -> Dict[str, float]:
    """Update agents from replay buffer samples when enough data is available.

    Args:
        agents: Agents to update.
        replay_buffer: Shared multi-agent replay buffer.
        batch_size: Batch size for sampling.
        gamma: Discount factor.
    """
    return update_maddpg_agents_from_buffer(
        agents,
        replay_buffer,
        batch_size,
        gamma,
    )


def train_maddpg(
    agents: List[EpsilonATNMADDPGAgent],
    devices: List[IndustrialDevice],
    env: DITENEnv,
    replay_buffer: MultiAgentReplayBuffer,
    priority_model: torch.nn.Module,
    data_loader: KolektorSDDLoader,
    num_episodes: int = 2000,
    batch_size: int = 64,
    gamma: float = 0.99,
    time_slots: int = 50,
) -> Dict[str, List[float]]:
    """Train e-ATN-MADDPG and return reward, delay, and energy histories.

    Args:
        agents: Multi-agent policy set.
        devices: Industrial devices for the episode.
        env: DITEN environment instance.
        replay_buffer: Shared replay buffer for experience sampling.
        priority_model: GCN/GAT model for priority extraction.
        data_loader: Dataset loader for task generation.
        num_episodes: Number of episodes to train.
        batch_size: Batch size for replay sampling.
        gamma: Discount factor.
        time_slots: Time slots per episode.

    Returns:
        Dict with reward, delay, and energy histories.
    """
    num_agents = len(agents)
    history = {"reward": [], "delay": [], "energy": []}
    template_task_dags = generate_task_dags_for_episode(
        devices[:1],
        data_loader,
        cpu_cycle_scale=PAPER_PARAMS["provisional_table2_needed"][
            "task_cpu_cycle_scale"
        ],
    )
    template_device_id = devices[0].id
    shared_priority_order = build_priorities(
        template_task_dags, priority_model
    )[template_device_id]
    fixed_priorities = broadcast_priority_order(
        [device.id for device in devices], shared_priority_order
    )
    print(f"Fixed task-priority order: {shared_priority_order}")

    episode_iterator = trange(num_episodes, desc="Training e-ATN-MADDPG", leave=True)
    for episode in episode_iterator:
        for agent in agents:
            agent.set_training_progress(episode, num_episodes)
        env.reset_episode()
        episode_done = False
        slot_rewards = []
        slot_delays = []
        slot_energies = []

        for _ in range(time_slots):
            if episode_done:
                break
            slot_reward = 0.0
            slot_steps = 0
            prev_delay_mean = np.mean(list(env.device_accumulated_delay.values()))
            prev_energy_mean = np.mean(list(env.device_accumulated_energy.values()))
            task_dags = generate_task_dags_for_episode(
                devices,
                data_loader,
                cpu_cycle_scale=PAPER_PARAMS["provisional_table2_needed"][
                    "task_cpu_cycle_scale"
                ],
            )
            current_joint_state = env.start_time_slot(task_dags, fixed_priorities)
            slot_done = False
            while not slot_done and not episode_done:
                joint_actions = _collect_joint_actions(agents, current_joint_state)

                next_joint_state, joint_rewards, step_episode_done, info = env.step(joint_actions)
                slot_done = info.get("slot_done", False)
                episode_done = step_episode_done
                # Aggregate team reward as average per-agent immediate reward.
                step_reward = sum(joint_rewards) / num_agents
                slot_reward += step_reward
                slot_steps += 1

                replay_buffer.push(
                    current_joint_state,
                    joint_actions,
                    joint_rewards,
                    next_joint_state,
                    done=step_episode_done,
                )
                current_joint_state = next_joint_state

            current_delay_mean = np.mean(list(env.device_accumulated_delay.values()))
            current_energy_mean = np.mean(list(env.device_accumulated_energy.values()))
            slot_rewards.append(slot_reward / max(slot_steps, 1))
            slot_delays.append(max(0.0, current_delay_mean - prev_delay_mean))
            slot_energies.append(max(0.0, current_energy_mean - prev_energy_mean))

        episode_avg_reward = float(np.mean(slot_rewards)) if slot_rewards else 0.0
        episode_avg_delay = float(np.mean(slot_delays)) if slot_delays else 0.0
        episode_avg_energy = float(np.mean(slot_energies)) if slot_energies else 0.0
        history["reward"].append(episode_avg_reward)
        history["delay"].append(episode_avg_delay)
        history["energy"].append(episode_avg_energy)
        episode_iterator.set_postfix(
            reward=f"{episode_avg_reward:.3f}",
            delay=f"{episode_avg_delay:.3f}s",
            energy=f"{episode_avg_energy:.3f}J",
        )
        
        _update_agents_from_buffer(agents, replay_buffer, batch_size, gamma)

        if (episode + 1) % 10 == 0:
            print(
                f"Ep {episode + 1}/{num_episodes} | Avg Reward/Slot: {history['reward'][-1]:.3f} "
                f"| Delay: {history['delay'][-1]:.3f}s | Energy: {history['energy'][-1]:.3f}J"
            )
            
    return history
# ==========================================
# ENTRY POINT: Khởi tạo và chạy mô phỏng
# ==========================================
if __name__ == "__main__":
    print("Initializing Simulation Environment...")
    set_seed(42)  # Đảm bảo tính tái lập
    # 1. Cài đặt thông số mạng (Dựa theo Table II của bài báo)
    confirmed = PAPER_PARAMS["confirmed"]
    provisional = PAPER_PARAMS["provisional_table2_needed"]
    BANDWIDTH = confirmed["bandwidth_hz"]       # 10 MHz
    NOISE_POWER = confirmed["noise_power_dbm"]  # -43 dBm
    NUM_DEVICES = int(confirmed["num_devices"])
    NUM_SERVERS = int(confirmed["num_servers"])
    TIME_SLOTS = int(confirmed["time_slots"])
    
    network_env = NetworkEnvironment(bandwidth=BANDWIDTH, noise_power_dbm=NOISE_POWER)
    
    # 2. Khởi tạo Edge Servers (User-confirmed Fig.4 topology)
    server_locations = [np.array([20.0, 30.0]), np.array([45.0, 50.0]), np.array([70.0, 20.0])]
    servers = []
    for server_index in range(NUM_SERVERS):
        edge_power_hz = np.random.uniform(
            provisional["server_compute_power_min_ghz"],
            provisional["server_compute_power_max_ghz"],
        ) * 1e9
        server = EdgeServer(
            server_id=server_index + 1,
            location=server_locations[server_index],
            compute_power=edge_power_hz,
            transmit_power=confirmed["server_tx_power_w"],
            energy_coeff=confirmed["server_energy_coeff"],
            coverage_radius=confirmed["coverage_radius_m"],
        )
        servers.append(server)

    # 3. Khởi tạo Industrial Devices (2 robots share each fixed rectangle)
    robot_starts = [
        np.array([10.0, 10.0]), np.array([30.0, 30.0]),
        np.array([70.0, 10.0]), np.array([40.0, 20.0]),
        np.array([40.0, 20.0]), np.array([60.0, 50.0]),
        np.array([70.0, 10.0]), np.array([90.0, 40.0]),
        np.array([65.0, 45.0]), np.array([90.0, 55.0]),
    ]
    devices = []
    for device_index in range(NUM_DEVICES):
        local_power_hz = np.random.uniform(
            provisional["device_compute_power_min_ghz"],
            provisional["device_compute_power_max_ghz"],
        ) * 1e9
        device = IndustrialDevice(
            device_id=device_index + 1,
            location=robot_starts[device_index],
            compute_power=local_power_hz,
            transmit_power=confirmed["device_tx_power_w"],
            energy_coeff=confirmed["device_energy_coeff"],
            speed_mps=confirmed["device_speed_mps"],
        )
        devices.append(device)
        
    # 4. Khởi tạo DITEN Environment
    env = DITENEnv(
        devices,
        servers,
        network_env,
        slot_duration=confirmed["slot_duration_s"],
        subslot_count=200,
        time_slots=TIME_SLOTS,
        lambda1=provisional["lambda1"],
        lambda2=provisional["lambda2"],
        lambda3=provisional["lambda3"],
        lambda4=provisional["lambda4"],
        lambda5=provisional["lambda5"],
        p_out_value=provisional["p_out_value"],
        local_estimation_error=provisional["local_estimation_error"],
        edge_estimation_error=provisional["edge_estimation_error"],
    )
    # 5. Khởi tạo DRL Agents (\epsilon-ATN-MADDPG)
    # Xác định kích thước State và Action
    STATE_DIM = env.get_state_dim()
    # Action: 0 (Local) hoặc 1, 2, 3 (Các Server) -> 4 options
    ACTION_DIM = 1 + NUM_SERVERS
    
    
    agents = []
    for _ in range(NUM_DEVICES):
        agent = EpsilonATNMADDPGAgent(
            state_dim=STATE_DIM,
            action_dim=ACTION_DIM,
            num_agents=NUM_DEVICES,
            lr=confirmed["rl_lr"],
            epsilon_init=provisional["epsilon_init"],
            epsilon_min=provisional["epsilon_min"],
            decay=provisional["epsilon_decay"],
        )
        agents.append(agent)

    # Khởi tạo Data Loader
    data_loader = KolektorSDDLoader(dataset_path="dataset/KolektorSDD/")

    # Khởi tạo priority model và Replay Buffer
    priority_model_name = str(provisional["priority_model"]).lower()
    priority_model = build_task_priority_model(
        priority_model_name,
        num_features=TASK_PRIORITY_FEATURE_DIM,
        hidden_dim=int(confirmed["gcn_hidden_dim"]),
    )
    priority_ckpt_path = get_priority_checkpoint_path(priority_model_name)
    sample_training_dag = make_priority_dag_sampler(
        data_loader,
        cpu_cycle_scale=provisional["task_cpu_cycle_scale"],
    )

    priority_model = load_or_train_priority_model(
        priority_model=priority_model,
        dag_sampler=sample_training_dag,
        checkpoint_path=priority_ckpt_path,
        epochs=200,
        samples_per_epoch=32,
        lr=confirmed["gcn_lr"],
        model_label=priority_model_name.upper(),
    )
    replay_buffer = MultiAgentReplayBuffer(
        capacity=int(provisional["replay_buffer_capacity"])
    )
    
    # 6. Bắt đầu Huấn Luyện
    print("\nStarting Training (e-ATN-MADDPG)...")
    EPISODES = int(confirmed["train_episodes_full"])
    
    e_atn_history = train_maddpg(
        agents=agents, 
        devices=devices,
        env=env, 
        replay_buffer=replay_buffer, 
        priority_model=priority_model,
        num_episodes=EPISODES,
        time_slots=TIME_SLOTS,
        data_loader=data_loader,
        batch_size=int(provisional["batch_size"]),
        gamma=provisional["gamma"],
    )
    
    # 7. Vẽ biểu đồ kết quả sau khi train xong
    print("\nTraining complete! Generating plot...")
    plotter = DITENPlotter()
    
    # Tạo dictionary chứa dữ liệu để vẽ (Giả lập các đường baseline cho biểu đồ)
    data_to_plot = {
        "e-ATN-MADDPG": e_atn_history["reward"]
    }
    date_string = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    plotter.plot_training_curve(
        data_dict=data_to_plot,
        title="Training Reward of e-ATN-MADDPG",
        ylabel="Average Reward",
        filename=f"training_reward_curve_{date_string}.png"
    )
    plotter.plot_training_curve(
        data_dict={"e-ATN-MADDPG": e_atn_history["delay"]},
        title="Training Delay of e-ATN-MADDPG",
        ylabel="Average Delay",
        filename=f"training_delay_curve_{date_string}.png"
    )
    plotter.plot_training_curve(
        data_dict={"e-ATN-MADDPG": e_atn_history["energy"]},
        title="Training Energy of e-ATN-MADDPG",
        ylabel="Average Energy",
        filename=f"training_energy_curve_{date_string}.png"
    )
    
    
    print("All processes completed successfully!")
