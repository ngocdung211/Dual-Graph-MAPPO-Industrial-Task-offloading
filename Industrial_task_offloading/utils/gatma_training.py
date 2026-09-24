"""Replay-buffer updates for the adapted GATMA baseline."""

from typing import Dict, Sequence

import torch
import torch.nn.functional as F

from baselines.gatma import GATMAAgent, build_gatma_topology_batch
from models.replay_buffer import MultiAgentReplayBuffer


def update_gatma_agents_from_buffer(
    agents: Sequence[GATMAAgent],
    replay_buffer: MultiAgentReplayBuffer,
    batch_size: int,
    gamma: float,
) -> Dict[str, float]:
    """Run one synchronized GATMA critic/actor/target update."""
    if len(replay_buffer) < batch_size:
        return {
            "update_rounds": 0.0,
            "actor_loss": 0.0,
            "critic_loss": 0.0,
            "mean_q": 0.0,
        }

    states, actions, rewards, next_states, dones = replay_buffer.sample(
        batch_size
    )
    # Keep replay storage on the host; transfer one shared batch per update.
    device = agents[0].device
    states, actions, rewards, next_states, dones = (
        tensor.to(device)
        for tensor in (states, actions, rewards, next_states, dones)
    )
    action_dim = agents[0].action_dim
    replay_joint_actions = F.one_hot(
        actions.long(), num_classes=action_dim
    ).float()
    state_topology = build_gatma_topology_batch(
        states, agents[0].num_servers
    )
    next_state_topology = build_gatma_topology_batch(
        next_states, agents[0].num_servers
    )

    with torch.no_grad():
        target_action_indices = torch.stack(
            [
                torch.argmax(
                    agent.target_actor(next_state_topology), dim=1
                )
                for agent in agents
            ],
            dim=1,
        )
        target_joint_actions = F.one_hot(
            target_action_indices, num_classes=action_dim
        ).float()

    critic_losses = []
    mean_q_values = []
    for agent_index, agent in enumerate(agents):
        with torch.no_grad():
            target_q = agent.target_critic(
                next_state_topology, target_joint_actions
            )
            critic_target = rewards[:, agent_index : agent_index + 1]
            critic_target = critic_target + gamma * (1.0 - dones) * target_q

        current_q = agent.critic(state_topology, replay_joint_actions)
        critic_loss = F.mse_loss(current_q, critic_target)
        agent.critic_optimizer.zero_grad()
        critic_loss.backward()
        agent.critic_optimizer.step()
        critic_losses.append(float(critic_loss.detach().item()))
        mean_q_values.append(float(current_q.detach().mean().item()))

    actor_losses = []
    for agent_index, agent in enumerate(agents):
        # The environment executes discrete actions. Use the same one-hot
        # representation for Q evaluation, with a softmax surrogate gradient
        # for this actor only; other agents retain their replayed actions.
        probabilities = agent.actor(state_topology)
        hard_actions = F.one_hot(
            probabilities.argmax(dim=-1), num_classes=action_dim
        ).to(probabilities.dtype)
        predicted_joint_actions = replay_joint_actions.clone()
        predicted_joint_actions[:, agent_index] = (
            hard_actions + (probabilities - probabilities.detach())
        )
        for parameter in agent.critic.parameters():
            parameter.requires_grad_(False)
        actor_loss = -agent.critic(
            state_topology, predicted_joint_actions
        ).mean()
        agent.actor_optimizer.zero_grad()
        actor_loss.backward()
        agent.actor_optimizer.step()
        for parameter in agent.critic.parameters():
            parameter.requires_grad_(True)
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
