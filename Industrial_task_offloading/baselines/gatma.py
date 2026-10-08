"""GATMA baseline adapted to the DITEN device-agent environment."""

import random
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from utils.gpu_readiness import resolve_torch_device


@dataclass(frozen=True)
class GATMATopologyBatch:
    """Reusable topology tensors derived from one joint-state batch."""

    node_features: torch.Tensor
    connected: torch.Tensor
    link_windows: torch.Tensor
    local_adjacency: torch.Tensor
    global_adjacency: torch.Tensor
    single_state: bool


def _build_topology_inputs(
    joint_states: torch.Tensor,
    num_servers: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
    return node_features, connected, torch.stack(
        (window_starts, window_ends), dim=-1
    )


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


def build_gatma_topology_batch(
    joint_states: torch.Tensor,
    num_servers: int,
) -> GATMATopologyBatch:
    """Build topology features and adjacency once for all GATMA networks."""
    single_state = joint_states.ndim == 2
    node_features, connected, link_windows = _build_topology_inputs(
        joint_states, num_servers
    )
    batch_size, num_devices, _ = connected.shape
    local_adjacency = _build_local_adjacency(
        connected.reshape(batch_size * num_devices, num_servers)
    ).reshape(
        batch_size,
        num_devices,
        num_servers + 1,
        num_servers + 1,
    )
    return GATMATopologyBatch(
        node_features=node_features,
        connected=connected,
        link_windows=link_windows,
        local_adjacency=local_adjacency,
        global_adjacency=_build_global_adjacency(connected),
        single_state=single_state,
    )


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
        adaptation_version: int = 3,
    ) -> None:
        """Initialize the input MLP, multi-head GAT, and output MLP."""
        super().__init__()
        self.num_servers = num_servers
        self.agent_index = agent_index
        self.adaptation_version = adaptation_version
        local_feature_dim = node_feature_dim + (
            2 if adaptation_version == 3 else 0
        )
        self.input_mlp = nn.Linear(local_feature_dim, hidden_dim)
        self.gat = MultiHeadGraphAttention(
            input_dim=hidden_dim,
            output_dim=embedding_dim,
            num_heads=num_heads,
        )
        self.output_fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.output_fc2 = nn.Linear(
            hidden_dim, 1 if adaptation_version == 3 else action_dim
        )
        if adaptation_version == 3:
            self.server_fc1 = nn.Linear(2 * embedding_dim + 3, hidden_dim)
            self.server_fc2 = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        joint_states: torch.Tensor | GATMATopologyBatch,
    ) -> torch.Tensor:
        """Return relaxed categorical offloading actions for this device."""
        topology = (
            joint_states
            if isinstance(joint_states, GATMATopologyBatch)
            else build_gatma_topology_batch(joint_states, self.num_servers)
        )
        num_devices = topology.connected.shape[1]
        if not 0 <= self.agent_index < num_devices:
            raise ValueError("agent_index is outside the device dimension")

        local_nodes = torch.cat(
            [
                topology.node_features[
                    :, self.agent_index : self.agent_index + 1
                ],
                topology.node_features[:, num_devices:],
            ],
            dim=1,
        )
        if self.adaptation_version == 3:
            # Windows belong to device-server pairs, not shared server nodes.
            windows = topology.link_windows[:, self.agent_index]
            local_windows = torch.cat(
                [windows.new_zeros((windows.shape[0], 1, 2)), windows], dim=1
            )
            local_nodes = torch.cat([local_nodes, local_windows], dim=-1)
        hidden_nodes = F.relu(self.input_mlp(local_nodes))
        embeddings = self.gat(
            hidden_nodes,
            topology.local_adjacency[:, self.agent_index],
        )
        device_embedding = embeddings[:, 0]
        local_logits = self.output_fc2(
            F.relu(self.output_fc1(device_embedding))
        )
        if self.adaptation_version == 3:
            connected = topology.connected[:, self.agent_index].unsqueeze(-1)
            # Unseen resources cannot affect scores; actions remain available.
            server_embeddings = embeddings[:, 1:] * connected
            candidate_features = torch.cat(
                [
                    device_embedding.unsqueeze(1).expand(
                        -1, self.num_servers, -1
                    ),
                    server_embeddings,
                    windows,
                    connected.to(embeddings.dtype),
                ],
                dim=-1,
            )
            server_logits = self.server_fc2(
                F.relu(self.server_fc1(candidate_features))
            ).squeeze(-1)
            # Candidate j stays aligned with offloading action j + 1.
            logits = torch.cat([local_logits, server_logits], dim=-1)
        else:
            logits = local_logits
        probabilities = F.softmax(logits, dim=-1)
        return probabilities.squeeze(0) if topology.single_state else probabilities


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
        agent_index: int = 0,
        adaptation_version: int = 3,
    ) -> None:
        """Initialize the global GAT critic."""
        super().__init__()
        self.num_servers = num_servers
        self.action_dim = action_dim
        self.agent_index = agent_index
        self.adaptation_version = adaptation_version
        context_dim = 3 * num_servers + 1 if adaptation_version == 3 else 0
        self.input_mlp = nn.Linear(
            node_feature_dim + action_dim + context_dim, hidden_dim
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
        joint_states: torch.Tensor | GATMATopologyBatch,
        joint_actions: torch.Tensor,
    ) -> torch.Tensor:
        """Return one global action-value estimate per batch item."""
        topology = (
            joint_states
            if isinstance(joint_states, GATMATopologyBatch)
            else build_gatma_topology_batch(joint_states, self.num_servers)
        )
        num_devices = topology.connected.shape[1]
        if joint_actions.ndim != 3 or joint_actions.shape[1:] != (
            num_devices,
            self.action_dim,
        ):
            raise ValueError(
                "joint_actions must have shape (batch, devices, action_dim)"
            )

        action_features = topology.node_features.new_zeros(
            (*topology.node_features.shape[:2], self.action_dim)
        )
        action_features[:, :num_devices] = joint_actions
        critic_inputs = torch.cat(
            [topology.node_features, action_features], dim=-1
        )
        if self.adaptation_version == 3:
            num_nodes = topology.node_features.shape[1]
            windows = critic_inputs.new_zeros(
                (critic_inputs.shape[0], num_nodes, 2 * self.num_servers)
            )
            windows[:, :num_devices] = topology.link_windows.flatten(
                start_dim=2
            )
            # Identify resources by the same server slots used by joint actions.
            server_identity = critic_inputs.new_zeros(
                (critic_inputs.shape[0], num_nodes, self.num_servers)
            )
            server_identity[:, num_devices:] = torch.eye(
                self.num_servers, device=critic_inputs.device,
                dtype=critic_inputs.dtype,
            )
            focal_device = critic_inputs.new_zeros(
                (critic_inputs.shape[0], num_nodes, 1)
            )
            focal_device[:, self.agent_index] = 1.0
            critic_inputs = torch.cat(
                [critic_inputs, windows, server_identity, focal_device], dim=-1
            )
        hidden_nodes = F.relu(self.input_mlp(critic_inputs))
        if self.adaptation_version == 3:
            # The collector only reads training state; it adds no cloud action.
            collector = hidden_nodes[:, self.agent_index : self.agent_index + 1]
            hidden_nodes = torch.cat([hidden_nodes, collector], dim=1)
            adjacency = F.pad(topology.global_adjacency, (0, 1, 0, 1))
            adjacency[:, -1, :-1] = True
            graph_embedding = self.gat(hidden_nodes, adjacency)[:, -1]
        else:
            graph_embedding = self.gat(
                hidden_nodes, topology.global_adjacency
            ).mean(dim=1)
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
        device: str = "cpu",
        adaptation_version: int = 3,
    ) -> None:
        """Initialize independent online/target networks on CPU or CUDA."""
        del num_agents
        priority_width = state_dim - (5 + 4 * num_servers)
        if priority_width <= 0:
            raise ValueError("state_dim is incompatible with num_servers")
        if not 0.0 < exploration_fraction <= 1.0:
            raise ValueError("exploration_fraction must be in (0, 1]")
        if adaptation_version not in (2, 3):
            raise ValueError("adaptation_version must be 2 or 3")

        self.action_dim = action_dim
        self.agent_index = agent_index
        self.num_servers = num_servers
        self.gamma = gamma
        self.tau = tau
        self.epsilon_init = epsilon_init
        self.epsilon_min = epsilon_min
        self.epsilon = epsilon_init
        self.exploration_fraction = exploration_fraction
        self.device = resolve_torch_device(device)
        self.adaptation_version = adaptation_version
        node_feature_dim = 9 + priority_width

        actor_kwargs = {
            "node_feature_dim": node_feature_dim,
            "action_dim": action_dim,
            "num_servers": num_servers,
            "agent_index": agent_index,
            "hidden_dim": hidden_dim,
            "embedding_dim": embedding_dim,
            "num_heads": num_heads,
            "adaptation_version": adaptation_version,
        }
        critic_kwargs = {
            "node_feature_dim": node_feature_dim,
            "action_dim": action_dim,
            "num_servers": num_servers,
            "hidden_dim": hidden_dim,
            "embedding_dim": embedding_dim,
            "num_heads": num_heads,
            "agent_index": agent_index,
            "adaptation_version": adaptation_version,
        }
        self.actor = GATMAActor(**actor_kwargs).to(self.device)
        self.target_actor = GATMAActor(**actor_kwargs).to(self.device)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.critic = GATMACritic(**critic_kwargs).to(self.device)
        self.target_critic = GATMACritic(**critic_kwargs).to(self.device)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.target_actor.requires_grad_(False)
        self.target_critic.requires_grad_(False)
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(), lr=actor_lr
        )
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(), lr=critic_lr
        )

    def select_action(
        self,
        joint_state: torch.Tensor | GATMATopologyBatch,
    ) -> int:
        """Select an epsilon-greedy offloading action from the joint state."""
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        return self.select_greedy_action(joint_state)

    def select_greedy_action(
        self,
        joint_state: torch.Tensor | GATMATopologyBatch,
    ) -> int:
        """Return the deterministic actor argmax action."""
        if not isinstance(joint_state, GATMATopologyBatch):
            joint_state = torch.as_tensor(
                joint_state, dtype=torch.float32, device=self.device
            )
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
