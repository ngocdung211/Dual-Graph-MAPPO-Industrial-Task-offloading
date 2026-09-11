"""Canonical MAPPO baseline with one shared actor and one shared critic.

This baseline follows the standard MAPPO recipe for homogeneous agents:
parameter sharing across every device actor, a single centralized value
function over the joint state, and one team advantage per transition. It is the
same centralized-training decentralized-execution structure used by the
Graph-GAT agent, so the two differ only in how each policy encodes its
observation.
"""

from typing import List, Sequence, Tuple

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from baselines.mappo import (
    CentralizedValueCritic,
    MultiAgentRolloutBuffer,
    StochasticActor,
    masked_action_probabilities,
)
from utils.rl_advantages import (
    compute_gae,
    compute_one_step_td,
    minibatch_indices,
    normalize_advantages,
)


class SharedMAPPOAgent:
    """MAPPO with parameter sharing, one centralized critic, team advantage."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        num_agents: int,
        lr: float = 0.0001,
        actor_lr: float | None = None,
        critic_lr: float | None = None,
        gamma: float = 0.99,
        clip_param: float = 0.2,
        ppo_epochs: int = 4,
        entropy_coef: float = 0.01,
        value_loss_coef: float = 1.0,
        max_grad_norm: float | None = None,
        hidden_dim: int = 64,
        use_action_mask: bool = False,
        use_gae: bool = False,
        gae_lambda: float = 0.95,
        num_minibatches: int = 1,
    ):
        """Initialize the shared MAPPO agent.

        Args:
            state_dim: Per-agent state dimension.
            action_dim: Number of actions.
            num_agents: Number of device agents sharing the actor.
            lr: Backward-compatible shared actor/critic learning rate.
            actor_lr: Optional actor learning rate. Uses ``lr`` when omitted.
            critic_lr: Optional critic learning rate. Uses ``lr`` when omitted.
            gamma: Discount factor.
            clip_param: PPO clipping parameter.
            ppo_epochs: PPO epochs per update.
            entropy_coef: Entropy bonus coefficient in the actor loss.
            value_loss_coef: Critic loss multiplier.
            max_grad_norm: Optional gradient clipping norm.
            hidden_dim: Actor and critic hidden width.
            use_action_mask: Whether to mask disconnected edge-server actions.
            use_gae: Use GAE(lambda) instead of one-step TD advantages.
            gae_lambda: GAE trace decay used when ``use_gae`` is set.
            num_minibatches: PPO minibatches per epoch.
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_agents = num_agents
        self.gamma = gamma
        self.clip_param = clip_param
        self.ppo_epochs = ppo_epochs
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm
        self.use_action_mask = use_action_mask
        self.use_gae = use_gae
        self.gae_lambda = gae_lambda
        self.num_minibatches = max(1, int(num_minibatches))

        self.actor = StochasticActor(state_dim, action_dim, hidden_dim=hidden_dim)
        self.critic = CentralizedValueCritic(
            state_dim, num_agents, hidden_dim=hidden_dim
        )
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(), lr=lr if actor_lr is None else actor_lr
        )
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(), lr=lr if critic_lr is None else critic_lr
        )
        self.last_joint_log_probs: List[float] = []

    def action_probabilities(self, joint_state: torch.Tensor) -> torch.Tensor:
        """Return masked action probabilities for every device state."""
        probabilities = self.actor(joint_state)
        if not self.use_action_mask:
            return probabilities
        return masked_action_probabilities(
            probabilities, joint_state, self.action_dim
        )

    def select_joint_actions_with_log_probs(
        self, joint_state: torch.Tensor
    ) -> Tuple[List[int], List[float]]:
        """Sample one action per device from the shared policy.

        Args:
            joint_state: Joint state tensor shaped `(num_agents, state_dim)`.

        Returns:
            Tuple of per-device actions and their old log-probabilities.
        """
        with torch.no_grad():
            probabilities = self.action_probabilities(joint_state)
            distribution = Categorical(probabilities)
            actions = distribution.sample()
            log_probs = distribution.log_prob(actions)
            self.last_joint_log_probs = [
                float(value) for value in log_probs.tolist()
            ]
            return (
                [int(value) for value in actions.tolist()],
                list(self.last_joint_log_probs),
            )

    def select_greedy_joint_actions(
        self, joint_state: torch.Tensor
    ) -> List[int]:
        """Choose the highest-probability action for every device."""
        with torch.no_grad():
            probabilities = self.action_probabilities(joint_state)
            return [
                int(value)
                for value in torch.argmax(probabilities, dim=-1).tolist()
            ]

    def update_from_rollout(self, rollout_buffer: MultiAgentRolloutBuffer) -> None:
        """Update the shared actor and centralized critic from one rollout."""
        if len(rollout_buffer) == 0:
            return

        (
            state_b,
            action_b,
            reward_b,
            next_state_b,
            old_log_prob_b,
            done_b,
        ) = rollout_buffer.as_tensors()

        batch_size = state_b.shape[0]
        joint_state_b = state_b.reshape(batch_size, -1)
        joint_next_state_b = next_state_b.reshape(batch_size, -1)
        actions = action_b.long()
        team_rewards = reward_b.mean(dim=1, keepdim=True)

        with torch.no_grad():
            current_v = self.critic(joint_state_b)
            if self.use_gae:
                last_value = self.critic(joint_next_state_b[-1:])
                advantages, target_v = compute_gae(
                    team_rewards,
                    current_v,
                    last_value,
                    done_b,
                    self.gamma,
                    self.gae_lambda,
                )
            else:
                next_v = self.critic(joint_next_state_b)
                advantages, target_v = compute_one_step_td(
                    team_rewards, current_v, next_v, done_b, self.gamma
                )
            advantages = normalize_advantages(advantages)

        for _ in range(self.ppo_epochs):
            for batch_indices in minibatch_indices(
                batch_size, self.num_minibatches, joint_state_b.device
            ):
                minibatch_states = state_b[batch_indices]
                probabilities = self.action_probabilities(minibatch_states)
                distribution = Categorical(probabilities)
                log_probs = distribution.log_prob(actions[batch_indices])
                entropy = distribution.entropy().mean()

                ratios = torch.exp(log_probs - old_log_prob_b[batch_indices])
                minibatch_advantages = advantages[batch_indices].expand_as(
                    ratios
                )
                unclipped = ratios * minibatch_advantages
                clipped = torch.clamp(
                    ratios, 1.0 - self.clip_param, 1.0 + self.clip_param
                ) * minibatch_advantages
                actor_loss = (
                    -torch.min(unclipped, clipped).mean()
                    - self.entropy_coef * entropy
                )

                values = self.critic(joint_state_b[batch_indices])
                critic_loss = F.mse_loss(values, target_v[batch_indices])

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                if self.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.actor.parameters(), self.max_grad_norm
                    )
                self.actor_optimizer.step()

                self.critic_optimizer.zero_grad()
                (self.value_loss_coef * critic_loss).backward()
                if self.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.critic.parameters(), self.max_grad_norm
                    )
                self.critic_optimizer.step()

        rollout_buffer.clear()


def select_shared_joint_actions(
    agent: SharedMAPPOAgent, joint_state: Sequence[Sequence[float]]
) -> Tuple[List[int], int, int]:
    """Select shared-policy actions and count local versus edge choices."""
    state_tensor = torch.as_tensor(joint_state, dtype=torch.float32)
    actions, _ = agent.select_joint_actions_with_log_probs(state_tensor)
    local_count = sum(1 for action in actions if action == 0)
    return actions, local_count, len(actions) - local_count
