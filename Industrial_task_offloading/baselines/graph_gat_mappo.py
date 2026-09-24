"""Graph-GAT MAPPO controller and graph-specific training workflow.

The current 30-device/9-server experiment enables GAE and four PPO minibatches
through runner flags. Warmup, lightweight topology, and greedy action selection
are optional variants; the constructor defaults remain available for older runs.
"""

from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from baselines.graph_gat_rollout import GraphGATRolloutBuffer, GraphGATTransition
from models.graph_gat_heads import (
    GraphGATActor,
    GraphGATValueCritic,
    TopologyWarmupHead,
)
from models.topology_gat import TopologyGATEncoder
from utils.gpu_readiness import resolve_torch_device
from utils.training.rl_advantages import (
    compute_gae,
    compute_one_step_td,
    minibatch_indices,
    normalize_advantages,
)
from utils.topology.graph_state import TopologyGraphState


__all__ = [
    "GraphGATActor",
    "GraphGATValueCritic",
    "TopologyWarmupHead",
    "GraphGATTransition",
    "GraphGATRolloutBuffer",
    "GraphGATMAPPOAgent",
]


class GraphGATMAPPOAgent:
    """Shared Graph-GAT MAPPO controller for all device agents."""

    def __init__(
        self,
        num_devices: int,
        num_servers: int,
        node_feature_dim: int,
        edge_feature_dim: int,
        embedding_dim: int = 64,
        hidden_dim: int = 64,
        lr: float = 0.0001,
        actor_lr: Optional[float] = None,
        critic_lr: Optional[float] = None,
        encoder_lr: Optional[float] = None,
        gamma: float = 0.99,
        clip_param: float = 0.2,
        ppo_epochs: int = 4,
        entropy_coef: float = 0.01,
        value_loss_coef: float = 1.0,
        max_grad_norm: Optional[float] = None,
        use_action_mask: bool = True,
        use_gae: bool = False,
        gae_lambda: float = 0.95,
        num_minibatches: int = 1,
        topology_warmup_episodes: int = 0,
        topology_warmup_updates_per_step: int = 0,
        topology_warmup_lr: float = 0.001,
        lightweight_topology: bool = False,
        device: str = "cpu",
    ):
        """Initialize Graph-GAT MAPPO.

        Args:
            num_devices: Number of device agents.
            num_servers: Number of edge servers.
            node_feature_dim: Input topology node feature dimension.
            edge_feature_dim: Input topology edge feature dimension.
            embedding_dim: GAT device embedding dimension.
            hidden_dim: Hidden layer width.
            lr: Fallback actor, critic, and encoder learning rate.
            actor_lr: Optional actor-specific learning rate. Uses ``lr`` when
                omitted.
            critic_lr: Optional critic-specific learning rate. Uses ``lr``
                when omitted.
            encoder_lr: Optional encoder-specific learning rate. Uses ``lr``
                when omitted.
            gamma: Discount factor.
            clip_param: PPO clipping parameter.
            ppo_epochs: PPO epochs per rollout update.
            entropy_coef: Entropy bonus coefficient in the actor loss.
            value_loss_coef: Critic loss coefficient in the joint PPO loss.
            max_grad_norm: Optional maximum PPO gradient norm.
            use_action_mask: Whether to mask disconnected edge-server actions.
            use_gae: Use GAE(lambda) instead of one-step TD advantages.
            gae_lambda: GAE trace decay used when ``use_gae`` is set.
            num_minibatches: PPO minibatches per epoch. One keeps the
                historical single full-batch update per epoch.
            topology_warmup_episodes: Number of first episodes using online GAT warmup.
            topology_warmup_updates_per_step: Auxiliary GAT updates before action selection.
            topology_warmup_lr: Learning rate for topology warmup optimizer.
            lightweight_topology: Use one server-to-device edge per pair,
                three edge features, and one topology GAT layer.
            device: PyTorch device request: ``cpu``, ``cuda``, ``cuda:<index>``,
                or ``auto``.
        """
        self.num_devices = num_devices
        self.num_servers = num_servers
        self.action_dim = num_servers + 1
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
        self.topology_warmup_episodes = topology_warmup_episodes
        self.topology_warmup_updates_per_step = topology_warmup_updates_per_step
        self.lightweight_topology = lightweight_topology
        self.connected_feature_index = 0 if lightweight_topology else 2
        self.window_length_feature_index = 2 if lightweight_topology else 6
        self.device = resolve_torch_device(device)

        # PPO shares this encoder between the local actor and global critic.
        self.encoder = TopologyGATEncoder(
            node_feature_dim=node_feature_dim,
            edge_feature_dim=edge_feature_dim,
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            single_layer_one_way=lightweight_topology,
        )
        self.actor = GraphGATActor(
            embedding_dim=embedding_dim,
            edge_feature_dim=edge_feature_dim,
            hidden_dim=hidden_dim,
        )
        self.critic = GraphGATValueCritic(
            embedding_dim=embedding_dim,
            num_devices=num_devices,
            hidden_dim=hidden_dim,
        )
        # Only Warmup variants train this auxiliary head before action selection.
        self.topology_warmup_head = TopologyWarmupHead(
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
        )
        self.encoder.to(self.device)
        self.actor.to(self.device)
        self.critic.to(self.device)
        self.topology_warmup_head.to(self.device)
        self.ppo_parameters = (
            list(self.encoder.parameters())
            + list(self.actor.parameters())
            + list(self.critic.parameters())
        )
        self.optimizer = optim.Adam(
            [
                {
                    "params": self.encoder.parameters(),
                    "lr": lr if encoder_lr is None else encoder_lr,
                },
                {
                    "params": self.actor.parameters(),
                    "lr": lr if actor_lr is None else actor_lr,
                },
                {
                    "params": self.critic.parameters(),
                    "lr": lr if critic_lr is None else critic_lr,
                },
            ]
        )
        self.topology_warmup_optimizer = optim.Adam(
            list(self.encoder.parameters()) + list(self.topology_warmup_head.parameters()),
            lr=topology_warmup_lr,
        )

    # Main rollout path: sample device actions and keep behavior log-probabilities.
    def select_actions_with_log_probs(
        self, graph_state: TopologyGraphState
    ) -> Tuple[List[int], List[float]]:
        """Sample one offloading action per device from graph embeddings."""
        with torch.no_grad():
            probabilities = self._actor_probabilities_for_graph_state(graph_state)
            distribution = Categorical(probabilities)
            actions = distribution.sample()
            log_probs = distribution.log_prob(actions)
            return (
                actions.detach().cpu().tolist(),
                [float(value) for value in log_probs.detach().cpu().tolist()],
            )

    # Evaluation variant: choose the highest-probability action without sampling.
    def select_greedy_actions(
        self, graph_state: TopologyGraphState
    ) -> List[int]:
        """Choose highest-probability actions without policy sampling."""
        with torch.no_grad():
            probabilities = self._actor_probabilities_for_graph_state(graph_state)
            return torch.argmax(probabilities, dim=-1).detach().cpu().tolist()

    def update_from_rollout(self, rollout_buffer: GraphGATRolloutBuffer) -> None:
        """Update encoder, actor, and critic from on-policy graph rollouts."""
        transitions = rollout_buffer.as_transitions()
        if not transitions:
            return

        actions = torch.tensor(
            [transition.actions for transition in transitions],
            dtype=torch.long,
            device=self.device,
        )
        rewards = torch.tensor(
            [transition.rewards for transition in transitions],
            dtype=torch.float32,
            device=self.device,
        )
        old_log_probs = torch.tensor(
            [transition.old_log_probs for transition in transitions],
            dtype=torch.float32,
            device=self.device,
        )
        dones = torch.tensor(
            [[float(transition.done)] for transition in transitions],
            dtype=torch.float32,
            device=self.device,
        )
        team_rewards = rewards.mean(dim=1, keepdim=True)
        graph_states = [transition.graph_state for transition in transitions]
        next_graph_states = [transition.next_graph_state for transition in transitions]

        stacked_features = self._stack_graph_features(graph_states)
        batch_size = len(transitions)

        with torch.no_grad():
            current_values = self._values_from_stacked(*stacked_features)
            # The current 30/9 command sets --use-gae; the other branch keeps
            # the historical one-step TD behavior when that flag is omitted.
            if self.use_gae:
                last_value = self._values_from_stacked(
                    *self._stack_graph_features(next_graph_states[-1:])
                )
                advantages, target_values = compute_gae(
                    team_rewards,
                    current_values,
                    last_value,
                    dones,
                    self.gamma,
                    self.gae_lambda,
                )
            else:
                next_values = self._values_from_stacked(
                    *self._stack_graph_features(next_graph_states)
                )
                advantages, target_values = compute_one_step_td(
                    team_rewards, current_values, next_values, dones, self.gamma
                )
            # Normalize across the whole rollout before splitting PPO minibatches.
            advantages = normalize_advantages(advantages)

        # The current 30/9 run uses four minibatches; one means a full-batch step.
        for _ in range(self.ppo_epochs):
            for batch_indices in minibatch_indices(
                batch_size, self.num_minibatches, self.device
            ):
                minibatch_features = tuple(
                    feature[batch_indices] for feature in stacked_features
                )
                log_probs, entropy, values = self._policy_and_values_from_stacked(
                    *minibatch_features, actions[batch_indices]
                )

                ratios = torch.exp(log_probs - old_log_probs[batch_indices])
                expanded_advantages = advantages[batch_indices].expand_as(ratios)
                unclipped = ratios * expanded_advantages
                clipped = torch.clamp(
                    ratios, 1.0 - self.clip_param, 1.0 + self.clip_param
                ) * expanded_advantages
                actor_loss = (
                    -torch.min(unclipped, clipped).mean()
                    - self.entropy_coef * entropy
                )
                critic_loss = F.mse_loss(values, target_values[batch_indices])

                self.optimizer.zero_grad()
                (actor_loss + self.value_loss_coef * critic_loss).backward()
                if self.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.ppo_parameters, self.max_grad_norm
                    )
                self.optimizer.step()

        rollout_buffer.clear()

    # Optional single-state helper; PPO uses the batched encoder below.
    def _encode_graph(self, graph_state: TopologyGraphState) -> torch.Tensor:
        """Encode the complete topology into global device embeddings."""
        device_features = self._selected_node_features(
            graph_state, graph_state.device_node_indices
        )
        server_features = self._selected_node_features(
            graph_state, graph_state.server_node_indices
        )
        forward_edge_features, backward_edge_features = (
            self._encoder_edge_features(graph_state)
        )
        return self.encoder.forward_batched_global(
            device_features=device_features,
            server_features=server_features,
            forward_edge_features=forward_edge_features,
            backward_edge_features=backward_edge_features,
        )

    # Warmup variants: train the encoder on topology labels during early episodes.
    def should_warmup_topology(self, episode_index: int) -> bool:
        """Return whether online topology warmup is active for this episode."""
        return (
            self.topology_warmup_episodes > 0
            and self.topology_warmup_updates_per_step > 0
            and episode_index < self.topology_warmup_episodes
        )

    def warmup_topology_encoder(
        self, graph_state: TopologyGraphState, update_count: int
    ) -> float:
        """Train topology encoder on link feasibility/window labels."""
        if update_count <= 0:
            return 0.0

        feasible_targets, window_targets = self._topology_warmup_targets(graph_state)
        last_loss = 0.0
        for _ in range(update_count):
            device_embeddings, server_embeddings = (
                self._encode_local_actor_nodes(graph_state)
            )
            feasible_logits, window_estimates = self.topology_warmup_head(
                device_embeddings, server_embeddings
            )
            feasible_loss = F.binary_cross_entropy_with_logits(
                feasible_logits, feasible_targets
            )
            window_loss = F.mse_loss(window_estimates, window_targets)
            loss = feasible_loss + 0.5 * window_loss

            self.topology_warmup_optimizer.zero_grad()
            loss.backward()
            self.topology_warmup_optimizer.step()
            last_loss = float(loss.detach().item())

        return last_loss

    def _topology_warmup_targets(
        self, graph_state: TopologyGraphState
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build feasible-link and window-length labels from graph edges."""
        forward_edge_features, _ = self._encoder_edge_features(graph_state)
        return (
            forward_edge_features[:, :, self.connected_feature_index],
            forward_edge_features[:, :, self.window_length_feature_index],
        )

    # Main action path: one local graph per device, scored by the shared actor.
    def _actor_probabilities_for_graph_state(
        self, graph_state: TopologyGraphState
    ) -> torch.Tensor:
        """Return actor probabilities from per-device local subgraph embeddings."""
        device_embeddings, server_embeddings = (
            self._encode_local_actor_nodes(graph_state)
        )
        forward_edge_features, _ = self._encoder_edge_features(graph_state)
        probabilities = self.actor(
            device_embeddings,
            server_embeddings,
            forward_edge_features,
        )
        return self._masked_action_probabilities(probabilities, graph_state)

    def _encode_local_actor_embeddings(
        self, graph_state: TopologyGraphState
    ) -> torch.Tensor:
        """Return device embeddings from local device-server graphs."""
        device_embeddings, _ = self._encode_local_actor_nodes(graph_state)
        return device_embeddings

    def _encode_local_actor_nodes(
        self, graph_state: TopologyGraphState
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return device and server embeddings from batched local graphs."""
        device_features = self._selected_node_features(
            graph_state, graph_state.device_node_indices
        )
        server_features = self._selected_node_features(
            graph_state, graph_state.server_node_indices
        )
        forward_edge_features, backward_edge_features = (
            self._encoder_edge_features(graph_state)
        )
        return self.encoder.forward_batched_local_nodes(
            device_features=device_features,
            server_features=server_features,
            forward_edge_features=forward_edge_features,
            backward_edge_features=backward_edge_features,
        )

    def _selected_node_features(
        self, graph_state: TopologyGraphState, node_indices: torch.Tensor
    ) -> torch.Tensor:
        """Select graph nodes and move their features to the agent device."""
        graph_device = graph_state.node_features.device
        selected_features = graph_state.node_features[
            node_indices.to(graph_device)
        ]
        return selected_features.to(self.device)

    def _device_server_edge_features(
        self, graph_state: TopologyGraphState
    ) -> torch.Tensor:
        """Return ordered link features shaped ``(N, S, directions, E)``."""
        direction_count = 1 if self.lightweight_topology else 2
        expected_edge_count = direction_count * self.num_devices * self.num_servers
        if graph_state.edge_features.shape[0] != expected_edge_count:
            raise ValueError(
                "topology graph edge count does not match the configured "
                "directions per device-server pair"
            )
        return graph_state.edge_features.reshape(
            self.num_devices,
            self.num_servers,
            direction_count,
            graph_state.edge_features.shape[-1],
        )

    def _encoder_edge_features(
        self, graph_state: TopologyGraphState
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return actor-facing and server-to-device encoder edge tensors."""
        pair_features = self._device_server_edge_features(graph_state).to(
            self.device
        )
        forward_edge_features = pair_features[:, :, 0, :]
        if self.lightweight_topology:
            # One-way topology has no reverse edge; the encoder accepts two tensors.
            return forward_edge_features, forward_edge_features
        return forward_edge_features, pair_features[:, :, 1, :]

    # Single-device graph helper retained for callers; action sampling is batched.
    def _local_subgraph_for_device(
        self, graph_state: TopologyGraphState, device_index: int
    ) -> TopologyGraphState:
        """Build one actor subgraph with one device node and all server nodes."""
        device = graph_state.node_features.device
        source_device_node = int(graph_state.device_node_indices[device_index])
        source_server_nodes = [
            int(server_node) for server_node in graph_state.server_node_indices.tolist()
        ]
        source_nodes = [source_device_node] + source_server_nodes
        local_index_by_source = {
            source_node: local_index
            for local_index, source_node in enumerate(source_nodes)
        }

        node_features = graph_state.node_features[
            torch.tensor(source_nodes, dtype=torch.long, device=device)
        ]
        edge_pairs: List[Tuple[int, int]] = []
        edge_features = []
        for edge_position, (source_node, target_node) in enumerate(
            graph_state.edge_index.transpose(0, 1).tolist()
        ):
            if source_node not in local_index_by_source:
                continue
            if target_node not in local_index_by_source:
                continue
            if source_device_node not in (source_node, target_node):
                continue
            edge_pairs.append(
                (
                    local_index_by_source[source_node],
                    local_index_by_source[target_node],
                )
            )
            edge_features.append(graph_state.edge_features[edge_position])

        if edge_pairs:
            edge_index = torch.tensor(
                edge_pairs, dtype=torch.long, device=device
            ).transpose(0, 1)
            edge_feature_tensor = torch.stack(edge_features, dim=0)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
            edge_feature_tensor = graph_state.edge_features.new_empty(
                (0, graph_state.edge_features.shape[1])
            )

        return TopologyGraphState(
            node_features=node_features,
            edge_index=edge_index,
            edge_features=edge_feature_tensor,
            device_node_indices=torch.tensor([0], dtype=torch.long, device=device),
            server_node_indices=torch.arange(
                1, self.num_servers + 1, dtype=torch.long, device=device
            ),
        )

    def _action_mask_for_graph_state(
        self, graph_state: TopologyGraphState
    ) -> torch.Tensor:
        """Return valid local/server actions for each device."""
        mask = torch.zeros(
            (self.num_devices, self.action_dim),
            dtype=torch.bool,
            device=self.device,
        )
        mask[:, 0] = True
        forward_edge_features, _ = self._encoder_edge_features(graph_state)
        mask[:, 1:] = (
            forward_edge_features[:, :, self.connected_feature_index] > 0.5
        )
        return mask

    def _masked_action_probabilities(
        self, probabilities: torch.Tensor, graph_state: TopologyGraphState
    ) -> torch.Tensor:
        """Zero disconnected server probabilities and renormalize actions."""
        if not self.use_action_mask:
            return probabilities

        forward_edge_features, _ = self._encoder_edge_features(graph_state)
        return self._masked_action_probabilities_for_edges(
            probabilities, forward_edge_features
        )

    # Mask variants drop disconnected servers; unmasked variants return actor output.
    def _masked_action_probabilities_for_edges(
        self,
        probabilities: torch.Tensor,
        forward_edge_features: torch.Tensor,
    ) -> torch.Tensor:
        """Apply link masks to single-state or rollout action probabilities."""
        if not self.use_action_mask:
            return probabilities

        action_mask = torch.zeros_like(probabilities, dtype=torch.bool)
        action_mask[..., 0] = True  # Local execution remains valid for every device.
        action_mask[..., 1:] = (
            forward_edge_features[..., self.connected_feature_index] > 0.2
        )
        masked_probabilities = probabilities * action_mask.float()
        probability_sum = masked_probabilities.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-8)
        return masked_probabilities / probability_sum

    # Main PPO path: stack rollout graphs once, then slice them into minibatches.
    def _stack_graph_features(
        self, graph_states: List[TopologyGraphState]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Stack rollout graph features into fixed-shape time-batch tensors."""
        device_features = torch.stack(
            [
                graph_state.node_features[graph_state.device_node_indices]
                for graph_state in graph_states
            ]
        )
        server_features = torch.stack(
            [
                graph_state.node_features[graph_state.server_node_indices]
                for graph_state in graph_states
            ]
        )
        directed_edge_features = [
            self._encoder_edge_features(graph_state)
            for graph_state in graph_states
        ]
        forward_edge_features = torch.stack(
            [edge_pair[0] for edge_pair in directed_edge_features]
        )
        backward_edge_features = torch.stack(
            [edge_pair[1] for edge_pair in directed_edge_features]
        )
        return (
            device_features.to(self.device),
            server_features.to(self.device),
            forward_edge_features.to(self.device),
            backward_edge_features.to(self.device),
        )

    def _values_for_graphs(
        self, graph_states: List[TopologyGraphState]
    ) -> torch.Tensor:
        """Return centralized values for a batch of graph states."""
        return self._values_from_stacked(
            *self._stack_graph_features(graph_states)
        )

    def _values_from_stacked(
        self,
        device_features: torch.Tensor,
        server_features: torch.Tensor,
        forward_edge_features: torch.Tensor,
        backward_edge_features: torch.Tensor,
    ) -> torch.Tensor:
        """Return centralized values for pre-stacked rollout features."""
        device_embeddings = self.encoder.forward_batched_global_rollout(
            device_features,
            server_features,
            forward_edge_features,
            backward_edge_features,
        )
        return self.critic(device_embeddings)

    def _policy_and_values_for_graphs(
        self,
        graph_states: List[TopologyGraphState],
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return batched rollout action log-probabilities and values."""
        return self._policy_and_values_from_stacked(
            *self._stack_graph_features(graph_states), actions
        )

    def _policy_and_values_from_stacked(
        self,
        device_features: torch.Tensor,
        server_features: torch.Tensor,
        forward_edge_features: torch.Tensor,
        backward_edge_features: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return log-probabilities, entropy, and values for stacked features."""
        actions = actions.to(self.device)
        device_embeddings, server_embeddings = (
            self.encoder.forward_batched_local_rollout_nodes(
                device_features,
                server_features,
                forward_edge_features,
                backward_edge_features,
            )
        )
        probabilities = self._masked_action_probabilities_for_edges(
            self.actor(
                device_embeddings,
                server_embeddings,
                forward_edge_features,
            ),
            forward_edge_features,
        )
        distribution = Categorical(probabilities)
        # The critic sees all device embeddings, while the actor uses local graphs.
        global_embeddings = self.encoder.forward_batched_global_rollout(
            device_features,
            server_features,
            forward_edge_features,
            backward_edge_features,
        )
        return (
            distribution.log_prob(actions),
            distribution.entropy().mean(),
            self.critic(global_embeddings),
        )

    def synchronize_device(self) -> None:
        """Wait for pending CUDA work so wall-clock timings are accurate."""
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
