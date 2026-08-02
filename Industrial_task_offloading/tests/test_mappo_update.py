"""Tests for MAPPO baseline action and update behavior."""

import pathlib
import random
import sys

import numpy as np
import torch

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from baselines.mappo import MAPPOAgent, MultiAgentRolloutBuffer, StochasticActor
from models.replay_buffer import MultiAgentReplayBuffer
from run_comparision import _update_agents_from_rollout


def _build_mappo_agents_and_buffer(
    num_agents: int = 2,
    state_dim: int = 4,
    action_dim: int = 3,
    batch_size: int = 4,
):
    """Build deterministic MAPPO agents and replay samples for update tests."""
    agents = [
        MAPPOAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            num_agents=num_agents,
            ppo_epochs=2,
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


def _clone_module_parameters(module: torch.nn.Module):
    """Return detached copies of module parameters."""
    return [parameter.detach().clone() for parameter in module.parameters()]


def _parameters_changed(before, after) -> bool:
    """Return whether any parameter tensor changed."""
    return any(not torch.allclose(old, new) for old, new in zip(before, after))


def test_stochastic_actor_outputs_valid_probability_distribution() -> None:
    """MAPPO actor should output normalized action probabilities."""
    actor = StochasticActor(state_dim=4, action_dim=3)
    state_batch = torch.zeros((5, 4))

    probabilities = actor(state_batch)

    assert probabilities.shape == (5, 3)
    assert torch.all(probabilities >= 0)
    assert torch.allclose(probabilities.sum(dim=1), torch.ones(5))


def test_select_action_samples_valid_discrete_actions() -> None:
    """MAPPO policy should sample valid discrete offloading actions."""
    random.seed(19)
    torch.manual_seed(19)
    agent = MAPPOAgent(state_dim=4, action_dim=4, num_agents=2)
    state = torch.zeros(4)

    actions = [agent.select_action(state) for _ in range(20)]

    assert all(0 <= action < agent.action_dim for action in actions)
    assert agent.last_action_log_prob is not None


def test_select_action_with_log_prob_returns_rollout_metadata() -> None:
    """MAPPO should expose the sampled action log-prob for PPO rollouts."""
    torch.manual_seed(21)
    agent = MAPPOAgent(state_dim=4, action_dim=4, num_agents=2)

    action, log_prob = agent.select_action_with_log_prob(torch.zeros(4))

    assert 0 <= action < agent.action_dim
    assert isinstance(log_prob, float)


def test_greedy_action_uses_highest_policy_probability() -> None:
    """Deterministic evaluation should not sample the MAPPO policy."""
    agent = MAPPOAgent(state_dim=4, action_dim=3, num_agents=1)
    with torch.no_grad():
        for parameter in agent.actor.parameters():
            parameter.zero_()
        agent.actor.fc3.bias.copy_(torch.tensor([-1.0, 2.0, 0.0]))

    assert agent.select_greedy_action(torch.zeros(4)) == 1


def test_mappo_action_mask_zeros_disconnected_server_probabilities() -> None:
    """MAPPO mask should keep local and remove disconnected server actions."""
    agent = MAPPOAgent(
        state_dim=9,
        action_dim=3,
        num_agents=1,
        use_action_mask=True,
    )
    state = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.4, 1.0, 0.4])
    probabilities = torch.tensor([0.2, 0.3, 0.5])

    masked = agent._masked_action_probabilities(probabilities, state)

    assert masked[2].item() == 0.0
    assert torch.allclose(masked.sum(), torch.tensor(1.0))


def test_mappo_action_mask_can_be_disabled() -> None:
    """MAPPO mask toggle should preserve raw probabilities when disabled."""
    agent = MAPPOAgent(
        state_dim=9,
        action_dim=3,
        num_agents=1,
        use_action_mask=False,
    )
    state = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.4, 1.0, 0.4])
    probabilities = torch.tensor([0.2, 0.3, 0.5])

    masked = agent._masked_action_probabilities(probabilities, state)

    assert torch.allclose(masked, probabilities)


def test_centralized_value_critic_returns_one_value_per_sample() -> None:
    """MAPPO critic should evaluate flattened joint states."""
    agent = MAPPOAgent(state_dim=4, action_dim=3, num_agents=2)
    joint_state_batch = torch.zeros((6, 8))

    values = agent.critic(joint_state_batch)

    assert values.shape == (6, 1)


def test_mappo_update_changes_actor_and_critic_parameters() -> None:
    """MAPPO update_agent should optimize both actor and critic networks."""
    torch.manual_seed(23)
    np.random.seed(23)
    batch_size = 4
    agents, replay_buffer = _build_mappo_agents_and_buffer(batch_size=batch_size)
    state_b, action_b, reward_b, next_state_b, _ = replay_buffer.sample(
        batch_size
    )
    old_log_prob_b = torch.zeros((batch_size, len(agents)))
    done_b = torch.zeros((batch_size, 1))
    actor_before = _clone_module_parameters(agents[0].actor)
    critic_before = _clone_module_parameters(agents[0].critic)

    agents[0].update_agent(
        state_b,
        action_b,
        reward_b,
        next_state_b,
        agent_index=0,
        old_log_prob_b=old_log_prob_b,
        done_b=done_b,
    )

    actor_after = _clone_module_parameters(agents[0].actor)
    critic_after = _clone_module_parameters(agents[0].critic)
    assert _parameters_changed(actor_before, actor_after)
    assert _parameters_changed(critic_before, critic_after)


def test_rollout_buffer_returns_old_log_probs_and_done_shapes() -> None:
    """MAPPO rollout buffer should keep old log-probs and done flags."""
    buffer = MultiAgentRolloutBuffer()
    state = np.zeros((2, 4), dtype=np.float32)
    action = [0, 1]
    reward = [1.0, 0.5]
    next_state = np.ones((2, 4), dtype=np.float32)
    old_log_probs = [-0.2, -0.7]

    buffer.push(state, action, reward, next_state, old_log_probs, done=True)

    state_b, action_b, reward_b, next_state_b, old_log_prob_b, done_b = buffer.as_tensors()
    assert state_b.shape == (1, 2, 4)
    assert action_b.shape == (1, 2)
    assert reward_b.shape == (1, 2)
    assert next_state_b.shape == (1, 2, 4)
    assert old_log_prob_b.shape == (1, 2)
    assert done_b.shape == (1, 1)
    assert done_b.item() == 1.0


def test_rollout_update_helper_updates_mappo_agents_and_clears_buffer() -> None:
    """Comparison rollout helper should train MAPPO from on-policy rollouts."""
    torch.manual_seed(29)
    np.random.seed(29)
    agents, _ = _build_mappo_agents_and_buffer(batch_size=4)
    rollout_buffer = MultiAgentRolloutBuffer()
    for sample_index in range(4):
        state = np.full((2, 4), 0.1 * (sample_index + 1), dtype=np.float32)
        next_state = state + 0.05
        actions = [sample_index % 3, (sample_index + 1) % 3]
        rewards = [1.0 - 0.1 * sample_index, 0.5 + 0.1 * sample_index]
        old_log_probs = [-0.2 - 0.01 * sample_index, -0.4 - 0.01 * sample_index]
        rollout_buffer.push(
            state,
            actions,
            rewards,
            next_state,
            old_log_probs,
            done=sample_index == 3,
        )
    actor_before = _clone_module_parameters(agents[0].actor)
    critic_before = _clone_module_parameters(agents[0].critic)

    _update_agents_from_rollout(agents, rollout_buffer, gamma=0.95)

    actor_after = _clone_module_parameters(agents[0].actor)
    critic_after = _clone_module_parameters(agents[0].critic)
    assert _parameters_changed(actor_before, actor_after)
    assert _parameters_changed(critic_before, critic_after)
    assert len(rollout_buffer) == 0
