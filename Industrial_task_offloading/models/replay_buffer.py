"""Replay buffer for multi-agent experience storage."""

import random
from collections import deque
from typing import Deque, List, Tuple

import numpy as np
import torch

class MultiAgentReplayBuffer:
    """Store joint experiences for multi-agent training."""

    def __init__(self, capacity: int = 100000):
        """Initialize the replay buffer.

        Args:
            capacity: Maximum number of experiences to store.
        """
        self.buffer: Deque[
            Tuple[List[float], List[int], List[float], List[float], bool]
        ] = deque(maxlen=capacity)

    def push(
        self,
        state: List[float],
        action: List[int],
        reward: List[float],
        next_state: List[float],
        done: bool = False,
    ) -> None:
        """Store a joint experience tuple (S, A, R, S', done).

        Args:
            state: Joint state for all agents.
            action: Joint actions for all agents.
            reward: Joint rewards for all agents.
            next_state: Next joint state for all agents.
            done: Whether the transition terminates the episode.
        """
        self.buffer.append((state, action, reward, next_state, bool(done)))

    def sample(
        self, batch_size: int
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Sample a random mini-batch of experiences.

        Args:
            batch_size: Number of samples to draw.

        Returns:
            Tuple of (states, actions, rewards, next_states, dones).
        """
        batch = random.sample(self.buffer, batch_size)
        
        state_batch = torch.FloatTensor(np.array([exp[0] for exp in batch]))
        action_batch = torch.FloatTensor(np.array([exp[1] for exp in batch]))
        reward_batch = torch.FloatTensor(np.array([exp[2] for exp in batch]))
        next_state_batch = torch.FloatTensor(np.array([exp[3] for exp in batch]))
        done_batch = torch.FloatTensor(
            np.array([exp[4] for exp in batch], dtype=np.float32)
        ).unsqueeze(1)
        
        return state_batch, action_batch, reward_batch, next_state_batch, done_batch

    def __len__(self) -> int:
        """Return the number of stored experiences."""
        return len(self.buffer)
