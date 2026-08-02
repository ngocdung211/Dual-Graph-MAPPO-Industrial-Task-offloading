"""Shared e-ATN-MADDPG replay update helpers."""

from typing import Dict, Sequence

import torch
import torch.nn.functional as F

from models.maddpg import EpsilonATNMADDPGAgent
from models.replay_buffer import MultiAgentReplayBuffer


def update_maddpg_agents_from_buffer(
    agents: Sequence[EpsilonATNMADDPGAgent],
    replay_buffer: MultiAgentReplayBuffer,
    batch_size: int,
    gamma: float,
) -> Dict[str, float]:
    """Run one synchronized CTDE replay update for all MADDPG agents.

    One joint batch and one target-action snapshot are shared by every agent.
    Critics update first, actors update second, and target networks update only
    after all gradient steps complete.
    """
    if len(replay_buffer) < batch_size:
        return {
            "update_rounds": 0.0,
            "actor_loss": 0.0,
            "critic_loss": 0.0,
            "mean_q": 0.0,
        }

    state_b, action_b, reward_b, next_state_b, done_b = replay_buffer.sample(
        batch_size
    )
    action_dim = agents[0].action_dim
    replay_joint_actions = F.one_hot(
        action_b.long(), num_classes=action_dim
    ).float()

    with torch.no_grad():
        target_joint_action_indices = torch.stack(
            [
                torch.argmax(
                    agent.target_actor(next_state_b[:, agent_index, :]),
                    dim=1,
                )
                for agent_index, agent in enumerate(agents)
            ],
            dim=1,
        )
        target_joint_actions = F.one_hot(
            target_joint_action_indices,
            num_classes=action_dim,
        ).float()
        detached_current_actions = torch.stack(
            [
                agent.actor(state_b[:, agent_index, :]).detach()
                for agent_index, agent in enumerate(agents)
            ],
            dim=1,
        )

    critic_losses = []
    mean_q_values = []
    for agent_index, agent in enumerate(agents):
        agent_rewards = reward_b[:, agent_index].unsqueeze(1)
        with torch.no_grad():
            target_q = agent.target_critic(next_state_b, target_joint_actions)
            critic_target = (
                agent_rewards + gamma * (1.0 - done_b) * target_q
            )

        current_q = agent.critic(state_b, replay_joint_actions)
        critic_loss = F.mse_loss(current_q, critic_target)
        agent.critic_optimizer.zero_grad()
        critic_loss.backward()
        agent.critic_optimizer.step()
        critic_losses.append(float(critic_loss.detach().item()))
        mean_q_values.append(float(current_q.detach().mean().item()))

    actor_losses = []
    for agent_index, agent in enumerate(agents):
        predicted_joint_actions = detached_current_actions.clone()
        predicted_joint_actions[:, agent_index] = agent.actor(
            state_b[:, agent_index, :]
        )
        actor_loss = -agent.critic(
            state_b, predicted_joint_actions
        ).mean()
        agent.actor_optimizer.zero_grad()
        actor_loss.backward()
        agent.actor_optimizer.step()
        actor_losses.append(float(actor_loss.detach().item()))

    for agent in agents:
        agent.soft_update(agent.target_actor, agent.actor)
        agent.soft_update(agent.target_critic, agent.critic)

    return {
        "update_rounds": 1.0,
        "actor_loss": sum(actor_losses) / len(actor_losses),
        "critic_loss": sum(critic_losses) / len(critic_losses),
        "mean_q": sum(mean_q_values) / len(mean_q_values),
    }
