"""Tests for MADDPG training update behavior."""

import numpy as np
import pathlib
import pytest
import random
import sys
import torch

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from main import _update_agents_from_buffer
from models.maddpg import EpsilonATNMADDPGAgent
from models.replay_buffer import MultiAgentReplayBuffer


def _build_agents_and_buffer(
    num_agents: int = 2,
    state_dim: int = 4,
    action_dim: int = 3,
    batch_size: int = 4,
):
    """Build deterministic agents and replay samples for update tests."""
    agents = [
        EpsilonATNMADDPGAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            num_agents=num_agents,
            use_attention=False,
            use_epsilon_greedy=False,
        )
        for _ in range(num_agents)
    ]
    replay_buffer = MultiAgentReplayBuffer(capacity=16)

    for sample_index in range(batch_size):
        state = np.full((num_agents, state_dim), 0.1 * (sample_index + 1), dtype=np.float32)
        next_state = state + 0.05
        actions = [
            (sample_index + agent_index) % action_dim for agent_index in range(num_agents)
        ]
        rewards = [
            1.0 - 0.1 * sample_index + 0.05 * agent_index
            for agent_index in range(num_agents)
        ]
        replay_buffer.push(state, actions, rewards, next_state)

    return agents, replay_buffer


def _clone_actor_parameters(agent: EpsilonATNMADDPGAgent):
    """Return detached copies of actor parameters."""
    return [parameter.detach().clone() for parameter in agent.actor.parameters()]


def _clone_critic_parameters(agent: EpsilonATNMADDPGAgent):
    """Return detached copies of critic parameters."""
    return [parameter.detach().clone() for parameter in agent.critic.parameters()]


def _clone_module_parameters(module: torch.nn.Module):
    """Return detached copies of module parameters."""
    return [parameter.detach().clone() for parameter in module.parameters()]


def _parameters_changed(before, after) -> bool:
    """Return whether any parameter tensor changed."""
    return any(not torch.allclose(old, new) for old, new in zip(before, after))


def test_select_action_with_full_epsilon_samples_valid_random_actions() -> None:
    """Full-epsilon policy should explore with valid random discrete actions."""
    random.seed(11)
    agent = EpsilonATNMADDPGAgent(
        state_dim=4,
        action_dim=4,
        num_agents=2,
        epsilon_init=1.0,
        use_epsilon_greedy=True,
    )
    state = torch.zeros(4)

    actions = [agent.select_action(state) for _ in range(20)]

    assert all(0 <= action < agent.action_dim for action in actions)
    assert len(set(actions)) > 1


def test_select_action_without_epsilon_uses_actor_argmax() -> None:
    """Deterministic policy should choose the actor probability argmax."""
    agent = EpsilonATNMADDPGAgent(
        state_dim=4,
        action_dim=4,
        num_agents=2,
        use_epsilon_greedy=False,
    )
    with torch.no_grad():
        for parameter in agent.actor.parameters():
            parameter.zero_()
        agent.actor.fc3.bias.copy_(torch.tensor([-1.0, 0.0, 3.0, 1.0]))

    assert agent.select_action(torch.zeros(4)) == 2


def test_replay_buffer_samples_multi_agent_shapes() -> None:
    """Replay samples should keep batch, agent, state, action, and reward axes."""
    random.seed(13)
    num_agents = 3
    state_dim = 5
    action_dim = 4
    batch_size = 4
    _, replay_buffer = _build_agents_and_buffer(
        num_agents=num_agents,
        state_dim=state_dim,
        action_dim=action_dim,
        batch_size=batch_size,
    )

    state_b, action_b, reward_b, next_state_b, done_b = replay_buffer.sample(
        batch_size
    )

    assert state_b.shape == (batch_size, num_agents, state_dim)
    assert action_b.shape == (batch_size, num_agents)
    assert reward_b.shape == (batch_size, num_agents)
    assert next_state_b.shape == (batch_size, num_agents, state_dim)
    assert done_b.shape == (batch_size, 1)
    assert torch.all(done_b == 0.0)


def test_progress_schedule_reaches_final_epsilon() -> None:
    """Epsilon should depend on episode progress rather than update count."""
    agent = EpsilonATNMADDPGAgent(
        state_dim=4,
        action_dim=3,
        num_agents=2,
        epsilon_init=1.0,
        epsilon_final=0.05,
        exploration_fraction=0.4,
    )

    agent.set_training_progress(completed_episodes=0, total_episodes=500)
    assert agent.epsilon == 1.0
    agent.set_training_progress(completed_episodes=100, total_episodes=500)
    assert agent.epsilon == 0.525
    agent.set_training_progress(completed_episodes=200, total_episodes=500)
    assert agent.epsilon == pytest.approx(0.05)
    agent.set_training_progress(completed_episodes=500, total_episodes=500)
    assert agent.epsilon == pytest.approx(0.05)


def test_actor_parameters_change_after_replay_update() -> None:
    """Actor update should propagate gradients into actor parameters."""
    torch.manual_seed(7)
    np.random.seed(7)

    batch_size = 4
    agents, replay_buffer = _build_agents_and_buffer(batch_size=batch_size)

    before = _clone_actor_parameters(agents[0])

    _update_agents_from_buffer(agents, replay_buffer, batch_size=batch_size, gamma=0.95)

    after = _clone_actor_parameters(agents[0])
    assert _parameters_changed(before, after)


def test_critic_parameters_change_after_replay_update() -> None:
    """Critic update should change critic parameters after replay training."""
    torch.manual_seed(17)
    np.random.seed(17)

    batch_size = 4
    agents, replay_buffer = _build_agents_and_buffer(batch_size=batch_size)
    before = _clone_critic_parameters(agents[0])

    _update_agents_from_buffer(agents, replay_buffer, batch_size=batch_size, gamma=0.95)

    after = _clone_critic_parameters(agents[0])
    assert _parameters_changed(before, after)


def test_terminal_transitions_do_not_bootstrap_target_q() -> None:
    """Terminal replay targets should omit the next-state critic value."""
    torch.manual_seed(37)
    continuing_agents, continuing_buffer = _build_agents_and_buffer(
        batch_size=4
    )
    torch.manual_seed(37)
    terminal_agents, terminal_buffer = _build_agents_and_buffer(batch_size=4)
    continuing_buffer.buffer = type(continuing_buffer.buffer)(
        [(*experience[:-1], False) for experience in continuing_buffer.buffer],
        maxlen=16,
    )
    terminal_buffer.buffer = type(terminal_buffer.buffer)(
        [(*experience[:-1], True) for experience in terminal_buffer.buffer],
        maxlen=16,
    )

    random.seed(41)
    continuing_metrics = _update_agents_from_buffer(
        continuing_agents,
        continuing_buffer,
        batch_size=4,
        gamma=0.95,
    )
    random.seed(41)
    terminal_metrics = _update_agents_from_buffer(
        terminal_agents,
        terminal_buffer,
        batch_size=4,
        gamma=0.95,
    )

    assert continuing_metrics["update_rounds"] == 1.0
    assert terminal_metrics["update_rounds"] == 1.0
    assert continuing_metrics["critic_loss"] != terminal_metrics["critic_loss"]


def test_soft_update_moves_target_network_parameters() -> None:
    """Soft update should move target actor and critic toward source networks."""
    agent = EpsilonATNMADDPGAgent(
        state_dim=4,
        action_dim=3,
        num_agents=2,
        use_attention=False,
        use_epsilon_greedy=False,
    )
    actor_before = _clone_module_parameters(agent.target_actor)
    critic_before = _clone_module_parameters(agent.target_critic)

    with torch.no_grad():
        for parameter in agent.actor.parameters():
            parameter.add_(1.0)
        for parameter in agent.critic.parameters():
            parameter.add_(1.0)

    agent.soft_update(agent.target_actor, agent.actor, tau=0.5)
    agent.soft_update(agent.target_critic, agent.critic, tau=0.5)

    actor_after = _clone_module_parameters(agent.target_actor)
    critic_after = _clone_module_parameters(agent.target_critic)
    assert _parameters_changed(actor_before, actor_after)
    assert _parameters_changed(critic_before, critic_after)
