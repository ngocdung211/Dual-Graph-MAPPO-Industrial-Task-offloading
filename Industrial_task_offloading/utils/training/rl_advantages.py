"""Advantage estimators shared by the on-policy PPO agents.

All estimators expect rollout tensors stored in collection order, which is the
order produced by the comparison training loop: one contiguous episode of
`time_steps` joint transitions with no mid-episode reset.
"""

from typing import Iterator, List, Tuple

import torch


def compute_one_step_td(
    rewards: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return one-step TD advantages and value targets.

    Args:
        rewards: Rewards shaped `(time_steps, channels)`.
        values: Current-state values shaped `(time_steps, channels)`.
        next_values: Next-state values with the same shape as `values`.
        dones: Terminal flags shaped `(time_steps, 1)` or as `rewards`.
        gamma: Discount factor.

    Returns:
        Tuple of advantages and value targets, both shaped like `rewards`.
    """
    targets = rewards + gamma * next_values * (1.0 - dones)
    return targets - values, targets


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    last_value: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return GAE(lambda) advantages and lambda-return value targets.

    The bootstrap value `last_value` is only used for the final transition.
    Every earlier transition reuses `values[t + 1]`, which is the value of the
    state actually visited next, so stale stored next states never enter the
    estimate.

    Args:
        rewards: Rewards shaped `(time_steps, channels)`.
        values: Current-state values shaped `(time_steps, channels)`.
        last_value: Bootstrap value for the state after the final transition,
            broadcastable to one row of `values`.
        dones: Terminal flags shaped `(time_steps, 1)` or as `rewards`.
        gamma: Discount factor.
        gae_lambda: GAE trace decay in `[0, 1]`. Zero reduces to one-step TD.

    Returns:
        Tuple of advantages and lambda-return targets, both shaped like
        `rewards`.

    Raises:
        ValueError: If `gae_lambda` is outside `[0, 1]`.
    """
    if not 0.0 <= gae_lambda <= 1.0:
        raise ValueError("gae_lambda must lie in [0, 1]")

    time_steps = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    running_advantage = torch.zeros_like(rewards[0])
    next_value = last_value.reshape(-1)[: rewards.shape[-1]].reshape_as(values[0])

    for step in range(time_steps - 1, -1, -1):
        non_terminal = 1.0 - dones[step]
        delta = (
            rewards[step]
            + gamma * next_value * non_terminal
            - values[step]
        )
        running_advantage = (
            delta + gamma * gae_lambda * non_terminal * running_advantage
        )
        advantages[step] = running_advantage
        next_value = values[step]

    return advantages, advantages + values


def normalize_advantages(advantages: torch.Tensor) -> torch.Tensor:
    """Standardize advantages over the complete rollout batch."""
    return (advantages - advantages.mean()) / (
        advantages.std(unbiased=False) + 1e-8
    )


def minibatch_indices(
    batch_size: int, num_minibatches: int, device: torch.device
) -> Iterator[torch.Tensor]:
    """Yield index tensors for one shuffled pass over a rollout batch.

    A `num_minibatches` of one yields the unshuffled full batch, which keeps
    the historical full-batch update behaviour bit-for-bit.

    Args:
        batch_size: Number of stored transitions.
        num_minibatches: Requested minibatch count.
        device: Device for the produced index tensors.

    Yields:
        Long tensors of transition indices.
    """
    if num_minibatches <= 1:
        yield torch.arange(batch_size, device=device)
        return

    permutation = torch.randperm(batch_size, device=device)
    chunk_size = max(1, (batch_size + num_minibatches - 1) // num_minibatches)
    chunks: List[torch.Tensor] = list(torch.split(permutation, chunk_size))
    for chunk in chunks:
        if chunk.numel() > 0:
            yield chunk
