"""MAPPO baseline implementation."""

from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

class CentralizedValueCritic(nn.Module):
    """Evaluate the joint state to estimate the global value V(s)."""

    def __init__(self, state_dim: int, num_agents: int, hidden_dim: int = 64):
        """Initialize the critic network.

        Args:
            state_dim: Per-agent state dimension.
            num_agents: Number of agents in the system.
            hidden_dim: Hidden layer width.
        """
        super(CentralizedValueCritic, self).__init__()
        self.fc1 = nn.Linear(state_dim * num_agents, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, joint_states: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            joint_states: Flattened joint state tensor.

        Returns:
            Estimated state value tensor.
        """
        x = F.relu(self.fc1(joint_states))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class StochasticActor(nn.Module):
    """Output a probability distribution over offloading locations."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        """Initialize the actor network.

        Args:
            state_dim: Per-agent state dimension.
            action_dim: Number of actions.
            hidden_dim: Hidden layer width.
        """
        super(StochasticActor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            state: Per-agent state tensor.

        Returns:
            Action probability distribution.
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return F.softmax(self.fc3(x), dim=-1)


class MultiAgentRolloutBuffer:
    """Store on-policy MAPPO rollout transitions."""

    def __init__(self):
        """Initialize an empty rollout buffer."""
        self.states: List[np.ndarray] = []
        self.actions: List[List[int]] = []
        self.rewards: List[List[float]] = []
        self.next_states: List[np.ndarray] = []
        self.old_log_probs: List[List[float]] = []
        self.dones: List[float] = []

    def push(
        self,
        state,
        action: List[int],
        reward: List[float],
        next_state,
        old_log_probs: List[float],
        done: bool,
    ) -> None:
        """Store one on-policy joint transition."""
        self.states.append(np.asarray(state, dtype=np.float32))
        self.actions.append(list(action))
        self.rewards.append(list(reward))
        self.next_states.append(np.asarray(next_state, dtype=np.float32))
        self.old_log_probs.append(list(old_log_probs))
        self.dones.append(float(done))

    def as_tensors(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the rollout as tensors in insertion order."""
        state_b = torch.FloatTensor(np.array(self.states))
        action_b = torch.FloatTensor(np.array(self.actions))
        reward_b = torch.FloatTensor(np.array(self.rewards))
        next_state_b = torch.FloatTensor(np.array(self.next_states))
        old_log_prob_b = torch.FloatTensor(np.array(self.old_log_probs))
        done_b = torch.FloatTensor(np.array(self.dones)).unsqueeze(1)
        return state_b, action_b, reward_b, next_state_b, old_log_prob_b, done_b

    def clear(self) -> None:
        """Remove all stored rollout transitions."""
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.next_states.clear()
        self.old_log_probs.clear()
        self.dones.clear()

    def __len__(self) -> int:
        """Return the number of stored transitions."""
        return len(self.states)


class MAPPOAgent:
    """Multi-Agent PPO baseline."""

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
    ):
        """Initialize the MAPPO agent.

        Args:
            state_dim: Per-agent state dimension.
            action_dim: Number of actions.
            num_agents: Number of agents in the system.
            lr: Backward-compatible shared actor/critic learning rate.
            actor_lr: Optional actor learning rate. Uses ``lr`` when omitted.
            critic_lr: Optional critic learning rate. Uses ``lr`` when omitted.
            gamma: Discount factor.
            clip_param: PPO clipping parameter.
            ppo_epochs: PPO epochs per update.
            entropy_coef: Entropy bonus coefficient in the actor loss.
            value_loss_coef: Critic loss multiplier.
            max_grad_norm: Optional actor/critic gradient clipping norm.
            hidden_dim: Actor and critic hidden width.
            use_action_mask: Whether to mask disconnected edge-server actions.
        """
        self.action_dim = action_dim
        self.gamma = gamma
        self.clip_param = clip_param
        self.ppo_epochs = ppo_epochs
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm
        self.use_action_mask = use_action_mask
        
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
        self.last_action_log_prob = None

    def select_action(self, state: torch.Tensor) -> int:
        """Sample an action from the policy distribution.

        Args:
            state: Per-agent state tensor.

        Returns:
            Selected action index.
        """
        action, log_prob = self.select_action_with_log_prob(state)
        self.last_action_log_prob = log_prob
        return action

    def select_action_with_log_prob(self, state: torch.Tensor) -> Tuple[int, float]:
        """Sample an action and return its old policy log-probability.

        Args:
            state: Per-agent state tensor.

        Returns:
            Tuple of selected action index and log-probability.
        """
        with torch.no_grad():
            probs = self.actor(state)
            probs = self._masked_action_probabilities(probs, state)
            distribution = Categorical(probs)
            action = distribution.sample()
            log_prob = distribution.log_prob(action)
            return action.item(), float(log_prob.item())

    def select_greedy_action(self, state: torch.Tensor) -> int:
        """Choose the highest-probability action without sampling."""
        with torch.no_grad():
            probabilities = self.actor(state)
            probabilities = self._masked_action_probabilities(
                probabilities, state
            )
            return int(torch.argmax(probabilities).item())

    def _action_mask_from_state(self, state: torch.Tensor) -> torch.Tensor:
        """Return valid local/server actions from flat connection windows."""
        mask = torch.zeros(
            self.action_dim,
            dtype=torch.bool,
            device=state.device,
        )
        mask[0] = True
        num_servers = self.action_dim - 1
        if num_servers <= 0:
            return mask

        window_values = state[-2 * num_servers:]
        window_starts = window_values[:num_servers]
        window_ends = window_values[num_servers:]
        mask[1:] = window_ends > window_starts
        return mask

    def _masked_action_probabilities(
        self, probabilities: torch.Tensor, state: torch.Tensor
    ) -> torch.Tensor:
        """Apply optional action mask and renormalize probabilities."""
        if not self.use_action_mask:
            return probabilities

        if probabilities.dim() == 1:
            mask = self._action_mask_from_state(state).float()
            masked_probabilities = probabilities * mask
            return masked_probabilities / masked_probabilities.sum().clamp_min(1e-8)

        masks = torch.stack(
            [self._action_mask_from_state(row_state) for row_state in state],
            dim=0,
        ).float()
        masked_probabilities = probabilities * masks
        probability_sum = masked_probabilities.sum(dim=1, keepdim=True).clamp_min(1e-8)
        return masked_probabilities / probability_sum

    def update_agent(
        self,
        state_b: torch.Tensor,
        action_b: torch.Tensor,
        reward_b: torch.Tensor,
        next_state_b: torch.Tensor,
        agent_index: int,
        old_log_prob_b: torch.Tensor | None = None,
        done_b: torch.Tensor | None = None,
    ) -> None:
        """Update agent parameters using a batch of experiences.

        Args:
            state_b: Batched joint states.
            action_b: Batched joint actions.
            reward_b: Batched joint rewards.
            next_state_b: Batched next joint states.
            agent_index: Index of the agent being updated.
            old_log_prob_b: Batched old action log-probabilities from rollout collection.
            done_b: Batched terminal flags.
        """
        
        batch_size = state_b.size(0)
        joint_state_b = state_b.view(batch_size, -1)
        joint_next_state_b = next_state_b.view(batch_size, -1)

        agent_rewards = reward_b[:, agent_index].unsqueeze(1)
        agent_states = state_b[:, agent_index, :]
        agent_actions = action_b[:, agent_index].long() 
        done_mask = done_b if done_b is not None else torch.zeros_like(agent_rewards)
        
        # 1. Calculate Advantages and Old Log Probs (Detached)
        with torch.no_grad():
            next_v = self.critic(joint_next_state_b)
            target_v = agent_rewards + self.gamma * next_v * (1.0 - done_mask)
            current_v = self.critic(joint_state_b)
            
            advantages = target_v - current_v
            advantages = (advantages - advantages.mean()) / (
                advantages.std(unbiased=False) + 1e-8
            )
            
            if old_log_prob_b is None:
                probs = self.actor(agent_states)
                distribution = Categorical(probs)
                old_log_probs = distribution.log_prob(agent_actions)
            else:
                old_log_probs = old_log_prob_b[:, agent_index]
            
        # 2. PPO Epochs
        for _ in range(self.ppo_epochs):
            probs = self.actor(agent_states)
            probs = self._masked_action_probabilities(probs, agent_states)
            distribution = Categorical(probs)
            log_probs = distribution.log_prob(agent_actions)
            entropy = distribution.entropy().mean()
            
            current_v_epoch = self.critic(joint_state_b)
            
            # Calculate the ratio (pi_theta / pi_theta_old)
            ratios = torch.exp(log_probs - old_log_probs)
            
            # Calculate Surrogate Losses
            surr1 = ratios * advantages.squeeze()
            surr2 = torch.clamp(ratios, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages.squeeze()
            
            # Actor Loss: Maximize surrogate objective -> minimize negative
            actor_loss = (
                -torch.min(surr1, surr2).mean()
                - self.entropy_coef * entropy
            )
            
            # Critic Loss: Standard MSE
            critic_loss = F.mse_loss(current_v_epoch, target_v)
            
            # Optimize Actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            if self.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.actor.parameters(), self.max_grad_norm
                )
            self.actor_optimizer.step()
            
            # Optimize Critic
            self.critic_optimizer.zero_grad()
            (self.value_loss_coef * critic_loss).backward()
            if self.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.critic.parameters(), self.max_grad_norm
                )
            self.critic_optimizer.step()
