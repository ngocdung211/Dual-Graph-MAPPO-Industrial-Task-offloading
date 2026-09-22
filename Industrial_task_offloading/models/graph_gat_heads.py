"""Actor, critic, and auxiliary heads for Graph-GAT MAPPO."""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphGATActor(nn.Module):
    """Score local execution and each device-server pair independently."""

    def __init__(
        self,
        embedding_dim: int,
        edge_feature_dim: int,
        hidden_dim: int = 64,
    ):
        """Initialize the actor head.

        Args:
            embedding_dim: Device and server GAT embedding dimension.
            edge_feature_dim: Device-server edge feature dimension.
            hidden_dim: Hidden layer width.
        """
        super(GraphGATActor, self).__init__()
        self.local_fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.local_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.local_output = nn.Linear(hidden_dim, 1)
        pair_feature_dim = 2 * embedding_dim + edge_feature_dim
        self.server_fc1 = nn.Linear(pair_feature_dim, hidden_dim)
        self.server_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.server_output = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        device_embeddings: torch.Tensor,
        server_embeddings: torch.Tensor,
        edge_features: torch.Tensor,
    ) -> torch.Tensor:
        """Return local and server-conditioned action probabilities."""
        local_hidden = F.relu(self.local_fc1(device_embeddings))
        local_hidden = F.relu(self.local_fc2(local_hidden))
        local_logits = self.local_output(local_hidden)

        expanded_devices = device_embeddings.unsqueeze(-2).expand_as(
            server_embeddings
        )
        pair_features = torch.cat(
            [expanded_devices, server_embeddings, edge_features], dim=-1
        )
        server_hidden = F.relu(self.server_fc1(pair_features))
        server_hidden = F.relu(self.server_fc2(server_hidden))
        server_logits = self.server_output(server_hidden).squeeze(-1)

        action_logits = torch.cat([local_logits, server_logits], dim=-1)
        return F.softmax(action_logits, dim=-1)


class GraphGATValueCritic(nn.Module):
    """Centralized value critic over all device embeddings."""

    def __init__(self, embedding_dim: int, num_devices: int, hidden_dim: int = 64):
        """Initialize the centralized graph critic.

        Args:
            embedding_dim: Device embedding dimension from the GAT encoder.
            num_devices: Number of device agents.
            hidden_dim: Hidden layer width.
        """
        super(GraphGATValueCritic, self).__init__()
        self.fc1 = nn.Linear(embedding_dim * num_devices, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, device_embeddings: torch.Tensor) -> torch.Tensor:
        """Return one scalar value for the joint graph state."""
        if device_embeddings.ndim == 2:
            joint_embedding = device_embeddings.reshape(1, -1)
        else:
            joint_embedding = device_embeddings.reshape(
                *device_embeddings.shape[:-2], -1
            )
        x = F.relu(self.fc1(joint_embedding))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class TopologyWarmupHead(nn.Module):
    """Predict topology targets from each device-server embedding pair."""

    def __init__(self, embedding_dim: int, hidden_dim: int = 64):
        """Initialize topology warmup prediction head.

        Args:
            embedding_dim: Device embedding dimension from local GAT subgraph.
            hidden_dim: Hidden layer width.
        """
        super(TopologyWarmupHead, self).__init__()
        self.fc1 = nn.Linear(2 * embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.feasible_head = nn.Linear(hidden_dim, 1)
        self.window_head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        device_embeddings: torch.Tensor,
        server_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return feasibility logits and window estimates for every pair."""
        expanded_devices = device_embeddings.unsqueeze(-2).expand_as(
            server_embeddings
        )
        pair_embeddings = torch.cat(
            [expanded_devices, server_embeddings], dim=-1
        )
        x = F.relu(self.fc1(pair_embeddings))
        x = F.relu(self.fc2(x))
        feasible_logits = self.feasible_head(x).squeeze(-1)
        window_estimates = torch.sigmoid(self.window_head(x).squeeze(-1))
        return feasible_logits, window_estimates
