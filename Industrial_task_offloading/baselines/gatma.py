"""GATMA baseline adapted to the DITEN device-agent environment."""

import random
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def _build_topology_inputs(
    joint_states: torch.Tensor,
    num_servers: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build shared node features and connectivity from flat joint states."""
    if joint_states.ndim == 2:
        joint_states = joint_states.unsqueeze(0)
    if joint_states.ndim != 3:
        raise ValueError("joint_states must have shape (batch, devices, state_dim)")

    batch_size, num_devices, state_dim = joint_states.shape
    priority_width = state_dim - (5 + 4 * num_servers)
    if priority_width <= 0:
        raise ValueError("joint state width is incompatible with num_servers")

    node_feature_dim = 9 + priority_width
    node_features = joint_states.new_zeros(
        (batch_size, num_devices + num_servers, node_feature_dim)
    )
    device_features = node_features[:, :num_devices]
    device_features[:, :, 0] = 1.0
    device_features[:, :, 2] = joint_states[:, :, 0]
    device_features[:, :, 3:7] = joint_states[:, :, 1:5]
    device_features[:, :, 9:] = joint_states[:, :, 5 : 5 + priority_width]

    edge_power_offset = 5 + priority_width
    edge_wait_offset = edge_power_offset + num_servers
    server_features = node_features[:, num_devices:]
    server_features[:, :, 1] = 1.0
    server_features[:, :, 7] = joint_states[
        :, 0, edge_power_offset : edge_power_offset + num_servers
    ]
    server_features[:, :, 8] = joint_states[
        :, 0, edge_wait_offset : edge_wait_offset + num_servers
    ]

    window_start_offset = edge_wait_offset + num_servers
    window_end_offset = window_start_offset + num_servers
    window_starts = joint_states[
        :, :, window_start_offset : window_start_offset + num_servers
    ]
    window_ends = joint_states[
        :, :, window_end_offset : window_end_offset + num_servers
    ]
    connected = window_ends > window_starts
    return node_features, connected


def _build_local_adjacency(connected: torch.Tensor) -> torch.Tensor:
    """Build device-server adjacency for one device in each batch item."""
    batch_size, num_servers = connected.shape
    num_nodes = num_servers + 1
    adjacency = torch.eye(
        num_nodes,
        dtype=torch.bool,
        device=connected.device,
    ).unsqueeze(0).expand(batch_size, -1, -1).clone()
    adjacency[:, 0, 1:] = connected
    adjacency[:, 1:, 0] = connected
    return adjacency


def _build_global_adjacency(connected: torch.Tensor) -> torch.Tensor:
    """Build batched bipartite device-server adjacency with self edges."""
    batch_size, num_devices, num_servers = connected.shape
    num_nodes = num_devices + num_servers
    adjacency = torch.eye(
        num_nodes,
        dtype=torch.bool,
        device=connected.device,
    ).unsqueeze(0).expand(batch_size, -1, -1).clone()
    adjacency[:, :num_devices, num_devices:] = connected
    adjacency[:, num_devices:, :num_devices] = connected.transpose(1, 2)
    return adjacency


class MultiHeadGraphAttention(nn.Module):
    """Dense multi-head graph attention with concatenated head outputs."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_heads: int = 4,
    ) -> None:
        """Initialize independent graph-attention heads."""
        super().__init__()
        if output_dim % num_heads != 0:
            raise ValueError("output_dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = output_dim // num_heads
        self.projections = nn.ModuleList(
            nn.Linear(input_dim, self.head_dim, bias=False)
            for _ in range(num_heads)
        )
        self.source_attention = nn.Parameter(
            torch.empty(num_heads, self.head_dim)
        )
        self.target_attention = nn.Parameter(
            torch.empty(num_heads, self.head_dim)
        )
        nn.init.xavier_uniform_(self.source_attention)
        nn.init.xavier_uniform_(self.target_attention)

    def forward(
        self,
        node_features: torch.Tensor,
        adjacency: torch.Tensor,
    ) -> torch.Tensor:
        """Aggregate source nodes allowed by ``adjacency[target, source]``."""
        projected = torch.stack(
            [projection(node_features) for projection in self.projections],
            dim=1,
        )
        source_scores = torch.einsum(
            "bhnd,hd->bhn", projected, self.source_attention
        )
        target_scores = torch.einsum(
            "bhnd,hd->bhn", projected, self.target_attention
        )
        attention_logits = F.leaky_relu(
            target_scores.unsqueeze(-1) + source_scores.unsqueeze(-2),
            negative_slope=0.2,
        )
        attention_logits = attention_logits.masked_fill(
            ~adjacency.unsqueeze(1),
            torch.finfo(attention_logits.dtype).min,
        )
        attention_weights = F.softmax(attention_logits, dim=-1)
        aggregated = torch.matmul(attention_weights, projected)
        return F.elu(
            aggregated.transpose(1, 2).reshape(
                node_features.shape[0], node_features.shape[1], -1
            )
        )


class GATMAActor(nn.Module):
    """Local-topology actor for one DITEN device agent."""

    def __init__(
        self,
        node_feature_dim: int,
        action_dim: int,
        num_servers: int,
        agent_index: int,
        hidden_dim: int = 64,
        embedding_dim: int = 64,
        num_heads: int = 4,
    ) -> None:
        """Initialize the input MLP, multi-head GAT, and output MLP."""
        super().__init__()
        self.num_servers = num_servers
        self.agent_index = agent_index
        self.input_mlp = nn.Linear(node_feature_dim, hidden_dim)
        self.gat = MultiHeadGraphAttention(
            input_dim=hidden_dim,
            output_dim=embedding_dim,
            num_heads=num_heads,
        )
        self.output_fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.output_fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, joint_states: torch.Tensor) -> torch.Tensor:
        """Return relaxed categorical offloading actions for this device."""
        single_state = joint_states.ndim == 2
        node_features, connected = _build_topology_inputs(
            joint_states, self.num_servers
        )
        num_devices = connected.shape[1]
        if not 0 <= self.agent_index < num_devices:
            raise ValueError("agent_index is outside the device dimension")

        local_nodes = torch.cat(
            [
                node_features[:, self.agent_index : self.agent_index + 1],
                node_features[:, num_devices:],
            ],
            dim=1,
        )
        adjacency = _build_local_adjacency(
            connected[:, self.agent_index]
        )
        hidden_nodes = F.relu(self.input_mlp(local_nodes))
        device_embedding = self.gat(hidden_nodes, adjacency)[:, 0]
        logits = self.output_fc2(F.relu(self.output_fc1(device_embedding)))
        probabilities = F.softmax(logits, dim=-1)
        return probabilities.squeeze(0) if single_state else probabilities


class GATMACritic(nn.Module):
    """Global-topology Q critic for one DITEN device agent."""

    def __init__(
        self,
        node_feature_dim: int,
        action_dim: int,
        num_servers: int,
        hidden_dim: int = 64,
        embedding_dim: int = 64,
        num_heads: int = 4,
    ) -> None:
        """Initialize the global GAT critic."""
        super().__init__()
        self.num_servers = num_servers
        self.action_dim = action_dim
        self.input_mlp = nn.Linear(
            node_feature_dim + action_dim, hidden_dim
        )
        self.gat = MultiHeadGraphAttention(
            input_dim=hidden_dim,
            output_dim=embedding_dim,
            num_heads=num_heads,
        )
        self.output_fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.output_fc2 = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        joint_states: torch.Tensor,
        joint_actions: torch.Tensor,
    ) -> torch.Tensor:
        """Return one global action-value estimate per batch item."""
        node_features, connected = _build_topology_inputs(
            joint_states, self.num_servers
        )
        num_devices = connected.shape[1]
        if joint_actions.ndim != 3 or joint_actions.shape[1:] != (
            num_devices,
            self.action_dim,
        ):
            raise ValueError(
                "joint_actions must have shape (batch, devices, action_dim)"
            )

        action_features = node_features.new_zeros(
            (*node_features.shape[:2], self.action_dim)
        )
        action_features[:, :num_devices] = joint_actions
        critic_inputs = torch.cat([node_features, action_features], dim=-1)
        hidden_nodes = F.relu(self.input_mlp(critic_inputs))
        adjacency = _build_global_adjacency(connected)
        graph_embeddings = self.gat(hidden_nodes, adjacency)
        graph_embedding = graph_embeddings.mean(dim=1)
        return self.output_fc2(F.relu(self.output_fc1(graph_embedding)))


class GATMAAgent:
    """Paper-inspired GATMA agent adapted to DITEN device decisions."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        num_agents: int,
        num_servers: int,
        agent_index: int,
        actor_lr: float = 1e-4,
        critic_lr: float = 1e-5,
        gamma: float = 0.95,
        tau: float = 0.01,
        hidden_dim: int = 64,
        embedding_dim: int = 64,
        num_heads: int = 4,
        epsilon_init: float = 0.99,
        epsilon_min: float = 0.01,
        exploration_fraction: float = 1.0,
    ) -> None:
        """Initialize online/target GAT actors and critics."""
        del num_agents
        priority_width = state_dim - (5 + 4 * num_servers)
        if priority_width <= 0:
            raise ValueError("state_dim is incompatible with num_servers")
        if not 0.0 < exploration_fraction <= 1.0:
            raise ValueError("exploration_fraction must be in (0, 1]")

        self.action_dim = action_dim
        self.agent_index = agent_index
        self.gamma = gamma
        self.tau = tau
        self.epsilon_init = epsilon_init
        self.epsilon_min = epsilon_min
        self.epsilon = epsilon_init
        self.exploration_fraction = exploration_fraction
        node_feature_dim = 9 + priority_width

        actor_kwargs = {
            "node_feature_dim": node_feature_dim,
            "action_dim": action_dim,
            "num_servers": num_servers,
            "agent_index": agent_index,
            "hidden_dim": hidden_dim,
            "embedding_dim": embedding_dim,
            "num_heads": num_heads,
        }
        critic_kwargs = {
            "node_feature_dim": node_feature_dim,
            "action_dim": action_dim,
            "num_servers": num_servers,
            "hidden_dim": hidden_dim,
            "embedding_dim": embedding_dim,
            "num_heads": num_heads,
        }
        self.actor = GATMAActor(**actor_kwargs)
        self.target_actor = GATMAActor(**actor_kwargs)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.critic = GATMACritic(**critic_kwargs)
        self.target_critic = GATMACritic(**critic_kwargs)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(), lr=actor_lr
        )
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(), lr=critic_lr
        )

    def select_action(self, joint_state: torch.Tensor) -> int:
        """Select an epsilon-greedy offloading action from the joint state."""
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        return self.select_greedy_action(joint_state)

    def select_greedy_action(self, joint_state: torch.Tensor) -> int:
        """Return the deterministic actor argmax action."""
        with torch.no_grad():
            return int(torch.argmax(self.actor(joint_state)).item())

    def set_training_progress(
        self,
        completed_episodes: int,
        total_episodes: int,
    ) -> None:
        """Linearly decay epsilon over the configured training fraction."""
        progress = completed_episodes / max(total_episodes, 1)
        decay_progress = min(progress / self.exploration_fraction, 1.0)
        self.epsilon = self.epsilon_init + decay_progress * (
            self.epsilon_min - self.epsilon_init
        )

    def soft_update(
        self,
        target_network: nn.Module,
        source_network: nn.Module,
    ) -> None:
        """Apply the paper's Polyak target-network update."""
        for target_parameter, source_parameter in zip(
            target_network.parameters(), source_network.parameters()
        ):
            target_parameter.data.copy_(
                self.tau * source_parameter.data
                + (1.0 - self.tau) * target_parameter.data
            )
