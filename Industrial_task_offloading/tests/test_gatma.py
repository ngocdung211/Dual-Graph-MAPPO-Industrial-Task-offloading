"""Behavioral checks for the device-agent adaptation of GATMA."""

import copy
import pathlib
import sys

import numpy as np
import pytest
import torch
import torch.nn.functional as F

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from baselines.gatma import GATMAAgent, build_gatma_topology_batch
from dataset.data_loader import KolektorSDDLoader
from environment.network_env import NetworkEnvironment
from environment.system_model import EdgeServer, IndustrialDevice
from inference_priority_comparison import _agent_config, _load_checkpoints
from models.replay_buffer import MultiAgentReplayBuffer
import run_comparision as runner
import utils.training.gatma_training as gatma_training
from utils.training.gatma_training import update_gatma_agents_from_buffer


def _state():
    """Two devices, two servers; device zero can see only server zero."""
    state = torch.rand(2, 18)
    state[:, 14:16] = 0.0
    state[:, 16:18] = torch.tensor([[1.0, 0.0], [1.0, 1.0]])
    return state


def _agents(device="cpu"):
    return [
        GATMAAgent(
            state_dim=18, action_dim=3, num_agents=2, num_servers=2,
            agent_index=index, hidden_dim=8, embedding_dim=8,
            num_heads=2, device=device,
        )
        for index in range(2)
    ]


def test_actor_respects_local_connectivity():
    """Unseen devices/disconnected servers cannot change local decisions."""
    torch.manual_seed(75)
    agent = _agents()[0]
    state = _state()
    original = agent.actor(state)
    changed = state.clone()
    changed[1, :10] += 20.0  # Other device's private task information.
    changed[:, 11] += 20.0  # Disconnected server's compute feature.
    changed[:, 13] += 20.0  # Disconnected server's wait feature.
    torch.testing.assert_close(agent.actor(changed), original)

    changed[:, 10] += 20.0  # Connected server must influence the actor.
    assert not torch.allclose(agent.actor(changed), original)
    state[0, 16:18] = 0.0  # A disconnected device still has its self edge.
    assert torch.isfinite(agent.actor(state)).all()


def test_cached_topology_preserves_outputs_and_gradients():
    """Reusing topology tensors must preserve GATMA math exactly."""
    torch.manual_seed(75)
    agent = _agents()[0]
    states = torch.stack([_state(), _state()])
    action_indices = torch.tensor([[0, 1], [2, 0]])
    joint_actions = F.one_hot(action_indices, num_classes=3).float()

    direct_actor = agent.actor(states)
    direct_critic = agent.critic(states, joint_actions)
    (direct_actor.sum() + direct_critic.sum()).backward()
    direct_gradients = {
        name: parameter.grad.detach().clone()
        for name, parameter in agent.actor.named_parameters()
    }
    direct_gradients.update(
        {
            f"critic.{name}": parameter.grad.detach().clone()
            for name, parameter in agent.critic.named_parameters()
        }
    )
    agent.actor.zero_grad()
    agent.critic.zero_grad()

    topology = build_gatma_topology_batch(states, num_servers=2)
    cached_actor = agent.actor(topology)
    cached_critic = agent.critic(topology, joint_actions)
    (cached_actor.sum() + cached_critic.sum()).backward()

    torch.testing.assert_close(cached_actor, direct_actor)
    torch.testing.assert_close(cached_critic, direct_critic)
    for name, parameter in agent.actor.named_parameters():
        torch.testing.assert_close(parameter.grad, direct_gradients[name])
    for name, parameter in agent.critic.named_parameters():
        torch.testing.assert_close(
            parameter.grad, direct_gradients[f"critic.{name}"]
        )


def test_replay_update_builds_only_state_and_next_state_topologies(monkeypatch):
    """One replay update must share two topology builds across all agents."""
    agents = _agents()
    buffer = MultiAgentReplayBuffer()
    for index in range(4):
        buffer.push(
            _state().numpy(),
            [index % 3, (index + 1) % 3],
            [0.3, 0.7],
            _state().numpy(),
            done=False,
        )
    build_count = 0
    original_builder = build_gatma_topology_batch

    def counting_builder(states, num_servers):
        nonlocal build_count
        build_count += 1
        return original_builder(states, num_servers)

    monkeypatch.setattr(
        gatma_training, "build_gatma_topology_batch", counting_builder
    )

    update_gatma_agents_from_buffer(agents, buffer, 4, 0.95)

    assert build_count == 2


def test_rollout_builds_topology_once_for_all_gatma_agents(monkeypatch):
    """One joint decision must share one topology build across all agents."""
    agents = _agents()
    for agent in agents:
        agent.epsilon = 0.0
    build_count = 0
    original_builder = build_gatma_topology_batch

    def counting_builder(states, num_servers):
        nonlocal build_count
        build_count += 1
        return original_builder(states, num_servers)

    monkeypatch.setattr(
        runner, "build_gatma_topology_batch", counting_builder
    )

    actions, _, _ = runner._collect_joint_actions(agents, _state().numpy())

    assert len(actions) == 2
    assert build_count == 1


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_replay_update_gradients_devices_targets_and_checkpoint(device, tmp_path):
    """One-hot Q inputs retain actor gradients and update all online GATs."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.manual_seed(75)
    agents = _agents(device)
    buffer = MultiAgentReplayBuffer()
    for index in range(4):
        buffer.push(
            _state().numpy(), [index % 3, (index + 1) % 3],
            [0.3, 0.7], _state().numpy(), done=True,
        )
    before = copy.deepcopy(agents[0].actor.state_dict())
    before_target = copy.deepcopy(agents[0].target_actor.state_dict())
    critic_inputs = []

    def capture_actions(module, inputs, output):
        critic_inputs.append(inputs[1].detach().cpu())

    handle = agents[0].critic.register_forward_hook(capture_actions)
    metrics = update_gatma_agents_from_buffer(agents, buffer, 4, 0.95)
    handle.remove()
    assert metrics["update_rounds"] == 1
    assert all(np.isfinite(value) for value in metrics.values())
    for actions in critic_inputs:
        torch.testing.assert_close(
            actions, F.one_hot(actions.argmax(-1), 3).float()
        )
    # The actor update changes only its own action; others come from replay.
    torch.testing.assert_close(critic_inputs[0][:, 1], critic_inputs[1][:, 1])
    for agent in agents:
        for module in (agent.actor, agent.critic, agent.target_actor, agent.target_critic):
            assert next(module.parameters()).device.type == device
        for module in (agent.actor.gat, agent.critic.gat):
            assert any(
                parameter.grad is not None and parameter.grad.abs().sum() > 0
                for parameter in module.parameters()
            )
        assert all(parameter.grad is None for parameter in agent.target_actor.parameters())
        assert all(parameter.grad is None for parameter in agent.target_critic.parameters())
        assert 0 <= agent.select_greedy_action(_state()) < 3
    assert any(not torch.equal(before[key], value) for key, value in agents[0].actor.state_dict().items())
    for key, value in agents[0].target_actor.state_dict().items():
        torch.testing.assert_close(
            value, agents[0].tau * agents[0].actor.state_dict()[key]
            + (1 - agents[0].tau) * before_target[key],
        )
    path = tmp_path / "actor.pt"
    torch.save(agents[0].actor.state_dict(), path)
    restored = _agents("cpu")[0]
    restored.actor.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    state = _state()
    torch.testing.assert_close(
        agents[0].actor(state.to(device)).cpu(), restored.actor(state),
        rtol=1e-4, atol=1e-5,
    )


def test_terminal_target_ignores_next_q(monkeypatch):
    agents = _agents()
    buffer = MultiAgentReplayBuffer()
    for _ in range(2):
        buffer.push(_state().numpy(), [0, 1], [0.3, 0.7], _state().numpy(), True)
    targets = []
    original_mse = F.mse_loss

    def capture_target(prediction, target):
        targets.append(target.detach().clone())
        return original_mse(prediction, target)

    monkeypatch.setattr(F, "mse_loss", capture_target)
    update_gatma_agents_from_buffer(agents, buffer, 2, 0.95)
    torch.testing.assert_close(targets[0], torch.full((2, 1), 0.3))
    torch.testing.assert_close(targets[1], torch.full((2, 1), 0.7))


def test_runner_uses_sixteen_episode_updates_and_continuous_replay(
    monkeypatch, tmp_path
):
    """GATMA must update 16 times after collecting continuous transitions."""
    monkeypatch.setitem(runner.PAPER_PARAMS["confirmed"], "time_slots", 2)
    buffers = []

    class RecordingBuffer(MultiAgentReplayBuffer):
        def __init__(self, capacity):
            super().__init__(capacity)
            buffers.append(self)

    monkeypatch.setattr(runner, "MultiAgentReplayBuffer", RecordingBuffer)
    devices = [IndustrialDevice(1, np.array([10., 10.]), 1e9, 0.5, 1e-28)]
    servers = [EdgeServer(1, np.array([12., 10.]), 2.4e9, 1.2, 1e-27, 12.)]
    network = NetworkEnvironment(bandwidth=10e6, noise_power_dbm=-43)
    loader = KolektorSDDLoader(str(tmp_path / "synthetic-unit-fixture"))
    config = runner.build_algorithm_configs(gatma_device="cpu")["GATMA-Adapted"]
    config["batch_size"] = 2
    config["kwargs"].update(hidden_dim=8, embedding_dim=8, num_heads=2)
    args = dict(
        devices=devices, servers=servers, network_env=network, data_loader=loader,
        priority_model=None, num_episodes=1, experiment_seed=75,
        fixed_priority_order=[1, 2, 3, 4, 5],
    )
    history, checkpoint = runner.train_algorithm(
        "GATMA-Adapted", config, **args, show_progress=False
    )
    transitions = list(buffers[0].buffer)
    assert len(transitions) == 10
    assert sum(transition[4] for transition in transitions) == 1
    assert transitions[-1][4]
    for previous, following in zip(transitions, transitions[1:]):
        np.testing.assert_array_equal(previous[3], following[0])
    assert history["maddpg_update_rounds"] == [16.0]
    assert checkpoint["adaptation"]["time_slots"] == 2
    assert checkpoint["adaptation"]["replay_updates_per_episode"] == 16
    assert "updates_per_slot" not in checkpoint["adaptation"]
    torch.save(checkpoint, tmp_path / "GATMA-Adapted_checkpoint.pt")
    loaded = _load_checkpoints(tmp_path, ["GATMA-Adapted"])["GATMA-Adapted"]
    evaluation = runner.evaluate_algorithm_checkpoint(_agent_config(loaded), loaded, **args)
    assert len(evaluation["reward"]) == 1
    assert all(np.isfinite(values).all() for values in evaluation.values())


def test_gatma_model_settings_come_from_config(monkeypatch) -> None:
    """The comparison builder must use the configured GATMA settings."""
    provisional = runner.PAPER_PARAMS["provisional_table2_needed"]
    monkeypatch.setitem(provisional, "gatma_actor_lr", 2e-4)
    monkeypatch.setitem(provisional, "gatma_batch_size", 256)

    config = runner.build_algorithm_configs(gatma_device="cpu")["GATMA-Adapted"]

    assert config["batch_size"] == 256
    assert config["replay_buffer_capacity"] == provisional[
        "gatma_replay_buffer_capacity"
    ]
    assert config["replay_updates_per_episode"] == provisional[
        "gatma_replay_updates_per_episode"
    ]
    assert config["kwargs"]["actor_lr"] == pytest.approx(2e-4)
    assert config["kwargs"]["critic_lr"] == provisional["gatma_critic_lr"]
    assert config["kwargs"]["gamma"] == provisional["gatma_gamma"]
    assert config["kwargs"]["device"] == "cpu"


def test_legacy_name_selects_one_adapted_baseline():
    configs = runner.build_algorithm_configs(gatma_device="cpu")
    selected = runner.select_algorithm_configs(configs, ["GATMA", "GATMA-Adapted"])
    assert list(selected) == ["GATMA-Adapted"]
    assert selected["GATMA-Adapted"]["kwargs"]["device"] == "cpu"
