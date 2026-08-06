"""Graph attention encoder for device-server topology states."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TopologyGraphAttentionLayer(nn.Module):
    """Single graph-attention layer over directed topology edges."""

    def __init__(self, node_feature_dim: int, edge_feature_dim: int, output_dim: int):
        """Initialize the graph-attention layer.

        Args:
            node_feature_dim: Input node feature dimension.
            edge_feature_dim: Input edge feature dimension.
            output_dim: Output node embedding dimension.
        """
        super(TopologyGraphAttentionLayer, self).__init__()
        self.edge_feature_dim = edge_feature_dim
        self.node_projection = nn.Linear(node_feature_dim, output_dim, bias=False)
        self.edge_projection = nn.Linear(edge_feature_dim, output_dim, bias=False)
        self.attention_projection = nn.Linear(output_dim * 3, 1, bias=False)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
    ) -> torch.Tensor:
        """Encode node features with attention over incoming edges.

        Args:
            node_features: Tensor shaped `(num_nodes, node_feature_dim)`.
            edge_index: Directed edges shaped `(2, num_edges)`.
            edge_features: Edge features shaped `(num_edges, edge_feature_dim)`.

        Returns:
            Updated node embeddings shaped `(num_nodes, output_dim)`.
        """
        num_nodes = node_features.shape[0]
        edge_index, edge_features = self._add_self_edges(
            edge_index, edge_features, num_nodes, node_features.device
        )

        projected_nodes = self.node_projection(node_features)
        projected_edges = self.edge_projection(edge_features)
        source_nodes = edge_index[0]
        target_nodes = edge_index[1]

        source_embeddings = projected_nodes[source_nodes]
        target_embeddings = projected_nodes[target_nodes]
        attention_input = torch.cat(
            [source_embeddings, target_embeddings, projected_edges], dim=-1
        )
        attention_logits = self.leaky_relu(
            self.attention_projection(attention_input)
        ).squeeze(-1)
        messages = source_embeddings + projected_edges

        node_outputs = []
        for node_index in range(num_nodes):
            incoming_mask = target_nodes == node_index
            incoming_logits = attention_logits[incoming_mask]
            incoming_messages = messages[incoming_mask]
            attention_weights = F.softmax(incoming_logits, dim=0).unsqueeze(-1)
            node_outputs.append(torch.sum(attention_weights * incoming_messages, dim=0))

        return torch.stack(node_outputs, dim=0)

    def forward_batched_local(
        self,
        device_features: torch.Tensor,
        server_features: torch.Tensor,
        forward_edge_features: torch.Tensor,
        backward_edge_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode independent device-server local graphs in one batch.

        Each batch item contains one device and all servers. Server features may
        be shared across devices on the first layer or device-specific on later
        layers. The calculation is equivalent to calling ``forward`` once per
        local graph, but it avoids Python graph construction and encoder loops.

        Args:
            device_features: Device features shaped ``(num_devices, input_dim)``.
            server_features: Shared server features shaped
                ``(num_servers, input_dim)`` or batched features shaped
                ``(num_devices, num_servers, input_dim)``.
            forward_edge_features: Device-to-server features shaped
                ``(num_devices, num_servers, edge_feature_dim)``.
            backward_edge_features: Server-to-device features with the same
                shape as ``forward_edge_features``.

        Returns:
            Device and server outputs shaped ``(num_devices, output_dim)`` and
            ``(num_devices, num_servers, output_dim)``.
        """
        num_devices = device_features.shape[0]
        num_servers = forward_edge_features.shape[1]
        if server_features.ndim == 2:
            server_features = server_features.unsqueeze(0).expand(
                num_devices, -1, -1
            )

        projected_devices = self.node_projection(device_features)
        projected_servers = self.node_projection(server_features)
        projected_forward_edges = self.edge_projection(forward_edge_features)
        projected_backward_edges = self.edge_projection(backward_edge_features)

        device_outputs = self._aggregate_device_messages(
            projected_devices,
            projected_servers,
            projected_backward_edges,
        )
        server_outputs = self._aggregate_server_messages(
            projected_devices,
            projected_servers,
            projected_forward_edges,
            num_servers,
        )
        return device_outputs, server_outputs

    def forward_batched_global(
        self,
        device_features: torch.Tensor,
        server_features: torch.Tensor,
        forward_edge_features: torch.Tensor,
        backward_edge_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode one complete bipartite topology without per-node loops.

        Args:
            device_features: Device features shaped
                ``(num_devices, input_dim)``.
            server_features: Server features shaped
                ``(num_servers, input_dim)``.
            forward_edge_features: Device-to-server features shaped
                ``(num_devices, num_servers, edge_feature_dim)``.
            backward_edge_features: Server-to-device features with the same
                shape as ``forward_edge_features``.

        Returns:
            Device and server outputs shaped ``(num_devices, output_dim)`` and
            ``(num_servers, output_dim)``.
        """
        num_devices = device_features.shape[0]
        projected_devices = self.node_projection(device_features)
        projected_servers = self.node_projection(server_features)
        projected_forward_edges = self.edge_projection(forward_edge_features)
        projected_backward_edges = self.edge_projection(backward_edge_features)

        device_outputs = self._aggregate_device_messages(
            projected_devices,
            projected_servers.unsqueeze(0).expand(num_devices, -1, -1),
            projected_backward_edges,
        )
        server_outputs = self._aggregate_global_server_messages(
            projected_devices,
            projected_servers,
            projected_forward_edges,
        )
        return device_outputs, server_outputs

    def forward_batched_global_rollout(
        self,
        device_features: torch.Tensor,
        server_features: torch.Tensor,
        forward_edge_features: torch.Tensor,
        backward_edge_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode a time batch of complete bipartite topology graphs.

        Args:
            device_features: Device features shaped
                ``(time_steps, num_devices, input_dim)``.
            server_features: Server features shaped
                ``(time_steps, num_servers, input_dim)``.
            forward_edge_features: Device-to-server features shaped
                ``(time_steps, num_devices, num_servers, edge_feature_dim)``.
            backward_edge_features: Server-to-device features with the same
                shape as ``forward_edge_features``.

        Returns:
            Device and server outputs shaped
            ``(time_steps, num_devices, output_dim)`` and
            ``(time_steps, num_servers, output_dim)``.
        """
        num_devices = device_features.shape[1]
        num_servers = server_features.shape[1]
        output_dim = self.node_projection.out_features
        projected_devices = self.node_projection(device_features)
        projected_servers = self.node_projection(server_features)
        projected_forward_edges = self.edge_projection(forward_edge_features)
        projected_backward_edges = self.edge_projection(backward_edge_features)

        device_server_sources = projected_servers.unsqueeze(1).expand(
            -1, num_devices, -1, -1
        )
        device_sources = torch.cat(
            [device_server_sources, projected_devices.unsqueeze(2)], dim=2
        )
        device_targets = projected_devices.unsqueeze(2).expand_as(device_sources)
        device_self_edges = projected_backward_edges.new_zeros(
            (*projected_devices.shape[:2], 1, output_dim)
        )
        device_edges = torch.cat(
            [projected_backward_edges, device_self_edges], dim=2
        )
        device_outputs = self._aggregate_rollout_messages(
            device_sources, device_targets, device_edges
        )

        server_device_sources = projected_devices.unsqueeze(1).expand(
            -1, num_servers, -1, -1
        )
        server_sources = torch.cat(
            [server_device_sources, projected_servers.unsqueeze(2)], dim=2
        )
        server_targets = projected_servers.unsqueeze(2).expand_as(server_sources)
        server_self_edges = projected_forward_edges.new_zeros(
            (*projected_servers.shape[:2], 1, output_dim)
        )
        server_edges = torch.cat(
            [projected_forward_edges.transpose(1, 2), server_self_edges], dim=2
        )
        server_outputs = self._aggregate_rollout_messages(
            server_sources, server_targets, server_edges
        )
        return device_outputs, server_outputs

    def _aggregate_rollout_messages(
        self,
        source_embeddings: torch.Tensor,
        target_embeddings: torch.Tensor,
        projected_edges: torch.Tensor,
    ) -> torch.Tensor:
        """Aggregate messages over the neighbor dimension of rollout tensors."""
        attention_logits = self.leaky_relu(
            self.attention_projection(
                torch.cat(
                    [source_embeddings, target_embeddings, projected_edges],
                    dim=-1,
                )
            )
        ).squeeze(-1)
        attention_weights = F.softmax(attention_logits, dim=2).unsqueeze(-1)
        return torch.sum(
            attention_weights * (source_embeddings + projected_edges), dim=2
        )

    def _aggregate_device_messages(
        self,
        projected_devices: torch.Tensor,
        projected_servers: torch.Tensor,
        projected_backward_edges: torch.Tensor,
    ) -> torch.Tensor:
        """Aggregate server and self messages into every device."""
        num_devices = projected_devices.shape[0]
        output_dim = projected_devices.shape[-1]
        self_edges = projected_backward_edges.new_zeros(
            (num_devices, 1, output_dim)
        )
        source_embeddings = torch.cat(
            [projected_servers, projected_devices.unsqueeze(1)], dim=1
        )
        target_embeddings = projected_devices.unsqueeze(1).expand_as(
            source_embeddings
        )
        projected_edges = torch.cat(
            [projected_backward_edges, self_edges], dim=1
        )
        attention_logits = self.leaky_relu(
            self.attention_projection(
                torch.cat(
                    [source_embeddings, target_embeddings, projected_edges],
                    dim=-1,
                )
            )
        ).squeeze(-1)
        attention_weights = F.softmax(attention_logits, dim=1).unsqueeze(-1)
        return torch.sum(
            attention_weights * (source_embeddings + projected_edges), dim=1
        )

    def _aggregate_server_messages(
        self,
        projected_devices: torch.Tensor,
        projected_servers: torch.Tensor,
        projected_forward_edges: torch.Tensor,
        num_servers: int,
    ) -> torch.Tensor:
        """Aggregate one device message and one self message into each server."""
        num_devices = projected_devices.shape[0]
        output_dim = projected_devices.shape[-1]
        device_sources = projected_devices.unsqueeze(1).expand(
            -1, num_servers, -1
        )
        source_embeddings = torch.stack(
            [device_sources, projected_servers], dim=2
        )
        target_embeddings = projected_servers.unsqueeze(2).expand_as(
            source_embeddings
        )
        self_edges = projected_forward_edges.new_zeros(
            (num_devices, num_servers, output_dim)
        )
        projected_edges = torch.stack(
            [projected_forward_edges, self_edges], dim=2
        )
        attention_logits = self.leaky_relu(
            self.attention_projection(
                torch.cat(
                    [source_embeddings, target_embeddings, projected_edges],
                    dim=-1,
                )
            )
        ).squeeze(-1)
        attention_weights = F.softmax(attention_logits, dim=2).unsqueeze(-1)
        return torch.sum(
            attention_weights * (source_embeddings + projected_edges), dim=2
        )

    def _aggregate_global_server_messages(
        self,
        projected_devices: torch.Tensor,
        projected_servers: torch.Tensor,
        projected_forward_edges: torch.Tensor,
    ) -> torch.Tensor:
        """Aggregate every device and one self message into each server."""
        num_servers = projected_servers.shape[0]
        output_dim = projected_servers.shape[-1]
        device_sources = projected_devices.unsqueeze(0).expand(
            num_servers, -1, -1
        )
        source_embeddings = torch.cat(
            [device_sources, projected_servers.unsqueeze(1)], dim=1
        )
        target_embeddings = projected_servers.unsqueeze(1).expand_as(
            source_embeddings
        )
        self_edges = projected_forward_edges.new_zeros(
            (num_servers, 1, output_dim)
        )
        projected_edges = torch.cat(
            [projected_forward_edges.transpose(0, 1), self_edges], dim=1
        )
        attention_logits = self.leaky_relu(
            self.attention_projection(
                torch.cat(
                    [source_embeddings, target_embeddings, projected_edges],
                    dim=-1,
                )
            )
        ).squeeze(-1)
        attention_weights = F.softmax(attention_logits, dim=1).unsqueeze(-1)
        return torch.sum(
            attention_weights * (source_embeddings + projected_edges), dim=1
        )

    def _add_self_edges(
        self,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
        num_nodes: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Add self edges so every node keeps its own feature context."""
        edge_index = edge_index.to(device=device, dtype=torch.long)
        edge_features = edge_features.to(device=device, dtype=torch.float32)
        node_indices = torch.arange(num_nodes, device=device, dtype=torch.long)
        self_edge_index = torch.stack([node_indices, node_indices], dim=0)
        self_edge_features = torch.zeros(
            (num_nodes, self.edge_feature_dim), device=device, dtype=torch.float32
        )

        if edge_index.numel() == 0:
            return self_edge_index, self_edge_features

        return (
            torch.cat([edge_index, self_edge_index], dim=1),
            torch.cat([edge_features, self_edge_features], dim=0),
        )


class TopologyGATEncoder(nn.Module):
    """Encode topology graph tensors into one embedding per device node."""

    def __init__(
        self,
        node_feature_dim: int,
        edge_feature_dim: int,
        hidden_dim: int = 64,
        embedding_dim: int = 64,
    ):
        """Initialize the topology GAT encoder.

        Args:
            node_feature_dim: Input node feature dimension.
            edge_feature_dim: Input edge feature dimension.
            hidden_dim: Hidden graph-attention dimension.
            embedding_dim: Final device embedding dimension.
        """
        super(TopologyGATEncoder, self).__init__()
        self.gat1 = TopologyGraphAttentionLayer(
            node_feature_dim=node_feature_dim,
            edge_feature_dim=edge_feature_dim,
            output_dim=hidden_dim,
        )
        self.gat2 = TopologyGraphAttentionLayer(
            node_feature_dim=hidden_dim,
            edge_feature_dim=edge_feature_dim,
            output_dim=embedding_dim,
        )

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
        device_node_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Return device-node embeddings from graph tensors.

        Args:
            node_features: Tensor shaped `(num_nodes, node_feature_dim)`.
            edge_index: Directed edges shaped `(2, num_edges)`.
            edge_features: Edge features shaped `(num_edges, edge_feature_dim)`.
            device_node_indices: Indices of device nodes in `node_features`.

        Returns:
            Tensor shaped `(num_devices, embedding_dim)`.
        """
        hidden_nodes = F.elu(self.gat1(node_features, edge_index, edge_features))
        encoded_nodes = self.gat2(hidden_nodes, edge_index, edge_features)
        return encoded_nodes[device_node_indices.to(encoded_nodes.device)]

    def forward_batched_local(
        self,
        device_features: torch.Tensor,
        server_features: torch.Tensor,
        forward_edge_features: torch.Tensor,
        backward_edge_features: torch.Tensor,
    ) -> torch.Tensor:
        """Return local device embeddings for all devices in one batched call.

        Args:
            device_features: Device node features shaped
                ``(num_devices, node_feature_dim)``.
            server_features: Server node features shaped
                ``(num_servers, node_feature_dim)``.
            forward_edge_features: Device-to-server edge features shaped
                ``(num_devices, num_servers, edge_feature_dim)``.
            backward_edge_features: Server-to-device edge features with the
                same shape as ``forward_edge_features``.

        Returns:
            Device embeddings shaped ``(num_devices, embedding_dim)``.
        """
        encoded_devices, _ = self.forward_batched_local_nodes(
            device_features,
            server_features,
            forward_edge_features,
            backward_edge_features,
        )
        return encoded_devices

    def forward_batched_local_nodes(
        self,
        device_features: torch.Tensor,
        server_features: torch.Tensor,
        forward_edge_features: torch.Tensor,
        backward_edge_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return device and per-device server embeddings for local graphs.

        Returns:
            Device embeddings shaped ``(num_devices, embedding_dim)`` and
            server embeddings shaped
            ``(num_devices, num_servers, embedding_dim)``.
        """
        hidden_devices, hidden_servers = self.gat1.forward_batched_local(
            device_features,
            server_features,
            forward_edge_features,
            backward_edge_features,
        )
        return self.gat2.forward_batched_local(
            F.elu(hidden_devices),
            F.elu(hidden_servers),
            forward_edge_features,
            backward_edge_features,
        )

    def forward_batched_global(
        self,
        device_features: torch.Tensor,
        server_features: torch.Tensor,
        forward_edge_features: torch.Tensor,
        backward_edge_features: torch.Tensor,
    ) -> torch.Tensor:
        """Return device embeddings from one complete bipartite topology.

        Args:
            device_features: Device node features shaped
                ``(num_devices, node_feature_dim)``.
            server_features: Server node features shaped
                ``(num_servers, node_feature_dim)``.
            forward_edge_features: Device-to-server edge features shaped
                ``(num_devices, num_servers, edge_feature_dim)``.
            backward_edge_features: Server-to-device edge features with the
                same shape as ``forward_edge_features``.

        Returns:
            Global-context device embeddings shaped
            ``(num_devices, embedding_dim)``.
        """
        hidden_devices, hidden_servers = self.gat1.forward_batched_global(
            device_features,
            server_features,
            forward_edge_features,
            backward_edge_features,
        )
        encoded_devices, _ = self.gat2.forward_batched_global(
            F.elu(hidden_devices),
            F.elu(hidden_servers),
            forward_edge_features,
            backward_edge_features,
        )
        return encoded_devices

    def forward_batched_local_rollout(
        self,
        device_features: torch.Tensor,
        server_features: torch.Tensor,
        forward_edge_features: torch.Tensor,
        backward_edge_features: torch.Tensor,
    ) -> torch.Tensor:
        """Return local device embeddings for a rollout time batch.

        Args:
            device_features: Device features shaped
                ``(time_steps, num_devices, node_feature_dim)``.
            server_features: Server features shaped
                ``(time_steps, num_servers, node_feature_dim)``.
            forward_edge_features: Device-to-server features shaped
                ``(time_steps, num_devices, num_servers, edge_feature_dim)``.
            backward_edge_features: Server-to-device features with the same
                shape as ``forward_edge_features``.

        Returns:
            Local device embeddings shaped
            ``(time_steps, num_devices, embedding_dim)``.
        """
        encoded_devices, _ = self.forward_batched_local_rollout_nodes(
            device_features,
            server_features,
            forward_edge_features,
            backward_edge_features,
        )
        return encoded_devices

    def forward_batched_local_rollout_nodes(
        self,
        device_features: torch.Tensor,
        server_features: torch.Tensor,
        forward_edge_features: torch.Tensor,
        backward_edge_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return device and server embeddings for rollout local graphs.

        Returns:
            Device embeddings shaped
            ``(time_steps, num_devices, embedding_dim)`` and server embeddings
            shaped ``(time_steps, num_devices, num_servers, embedding_dim)``.
        """
        time_steps, num_devices = device_features.shape[:2]
        num_servers = server_features.shape[1]
        batched_servers = server_features.unsqueeze(1).expand(
            -1, num_devices, -1, -1
        )
        encoded_devices, encoded_servers = self.forward_batched_local_nodes(
            device_features.reshape(time_steps * num_devices, -1),
            batched_servers.reshape(
                time_steps * num_devices, num_servers, -1
            ),
            forward_edge_features.reshape(
                time_steps * num_devices, num_servers, -1
            ),
            backward_edge_features.reshape(
                time_steps * num_devices, num_servers, -1
            ),
        )
        return (
            encoded_devices.reshape(time_steps, num_devices, -1),
            encoded_servers.reshape(
                time_steps, num_devices, num_servers, -1
            ),
        )

    def forward_batched_global_rollout(
        self,
        device_features: torch.Tensor,
        server_features: torch.Tensor,
        forward_edge_features: torch.Tensor,
        backward_edge_features: torch.Tensor,
    ) -> torch.Tensor:
        """Return global-context device embeddings for a rollout time batch."""
        hidden_devices, hidden_servers = (
            self.gat1.forward_batched_global_rollout(
                device_features,
                server_features,
                forward_edge_features,
                backward_edge_features,
            )
        )
        encoded_devices, _ = self.gat2.forward_batched_global_rollout(
            F.elu(hidden_devices),
            F.elu(hidden_servers),
            forward_edge_features,
            backward_edge_features,
        )
        return encoded_devices
