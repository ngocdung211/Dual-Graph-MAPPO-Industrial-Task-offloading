"""Separate edge-aware PyG GAT encoder for the topology ablation."""

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GATConv

from models.topology_gat import TopologyGATEncoder


class PyGTopologyGATEncoder(TopologyGATEncoder):
    """Use library GAT layers with the existing local/global encoder interface.

    Edge attributes affect attention, while messages contain projected nodes.
    The inherited local rollout helpers only reshape tensors; all attention
    and aggregation in this encoder are performed by GATConv.
    """

    def __init__(
        self,
        node_feature_dim: int,
        edge_feature_dim: int,
        hidden_dim: int = 64,
        embedding_dim: int = 64,
        single_layer_one_way: bool = False,
    ):
        """Build two single-head layers matching the full custom encoder widths."""
        nn.Module.__init__(self)
        if single_layer_one_way:
            raise ValueError(
                "PyG topology experiment requires the full two-layer graph"
            )
        self.single_layer_one_way = False
        layer_kwargs = {
            "heads": 1,
            "dropout": 0.0,
            "negative_slope": 0.2,
            "edge_dim": edge_feature_dim,
            "add_self_loops": True,
            "fill_value": 0.0,
            "bias": False,
        }
        self.gat1 = GATConv(node_feature_dim, hidden_dim, **layer_kwargs)
        self.gat2 = GATConv(hidden_dim, embedding_dim, **layer_kwargs)

    def _encode_nodes(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
    ) -> torch.Tensor:
        """Run PyG attention with ELU between the two layers."""
        hidden = self.gat1(node_features, edge_index, edge_features)
        return self.gat2(F.elu(hidden), edge_index, edge_features)

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
        device_node_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Return device embeddings for a single explicit topology graph."""
        return self._encode_nodes(node_features, edge_index, edge_features)[
            device_node_indices
        ]

    def forward_batched_local_nodes(
        self,
        device_features: torch.Tensor,
        server_features: torch.Tensor,
        forward_edge_features: torch.Tensor,
        backward_edge_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode disjoint one-device/all-server graphs without cross-device edges."""
        num_devices, num_servers = forward_edge_features.shape[:2]
        if server_features.ndim == 2:
            server_features = server_features.unsqueeze(0).expand(
                num_devices, -1, -1
            )
        nodes = torch.cat([device_features.unsqueeze(1), server_features], dim=1)
        # Give each local graph its own node range, including server copies.
        offsets = torch.arange(num_devices, device=nodes.device) * (
            num_servers + 1
        )
        sources = offsets[:, None].expand(-1, num_servers).reshape(-1)
        targets = (
            offsets[:, None]
            + torch.arange(1, num_servers + 1, device=nodes.device)
        ).reshape(-1)
        edge_index = torch.stack([
            torch.cat([sources, targets]), torch.cat([targets, sources]),
        ])
        edge_features = torch.cat([
            forward_edge_features.reshape(-1, forward_edge_features.shape[-1]),
            backward_edge_features.reshape(-1, backward_edge_features.shape[-1]),
        ])
        encoded = self._encode_nodes(nodes.flatten(0, 1), edge_index, edge_features)
        encoded = encoded.reshape(num_devices, num_servers + 1, -1)
        return encoded[:, 0], encoded[:, 1:]

    def forward_batched_global(
        self,
        device_features: torch.Tensor,
        server_features: torch.Tensor,
        forward_edge_features: torch.Tensor,
        backward_edge_features: torch.Tensor,
    ) -> torch.Tensor:
        """Encode a single full topology using the rollout batching path."""
        return self.forward_batched_global_rollout(
            device_features.unsqueeze(0),
            server_features.unsqueeze(0),
            forward_edge_features.unsqueeze(0),
            backward_edge_features.unsqueeze(0),
        )[0]

    def forward_batched_global_rollout(
        self,
        device_features: torch.Tensor,
        server_features: torch.Tensor,
        forward_edge_features: torch.Tensor,
        backward_edge_features: torch.Tensor,
    ) -> torch.Tensor:
        """Batch disjoint full topologies so different timesteps cannot interact."""
        time_steps, num_devices, num_servers = forward_edge_features.shape[:3]
        nodes = torch.cat([device_features, server_features], dim=1)
        # Disjoint node ranges keep message passing inside each timestep.
        offsets = torch.arange(time_steps, device=nodes.device) * (
            num_devices + num_servers
        )
        sources = (
            offsets[:, None, None]
            + torch.arange(num_devices, device=nodes.device)[None, :, None]
        ).expand(-1, -1, num_servers).reshape(-1)
        targets = (
            offsets[:, None, None]
            + num_devices
            + torch.arange(num_servers, device=nodes.device)[None, None, :]
        ).expand(-1, num_devices, -1).reshape(-1)
        edge_index = torch.stack([
            torch.cat([sources, targets]), torch.cat([targets, sources]),
        ])
        edge_features = torch.cat([
            forward_edge_features.reshape(-1, forward_edge_features.shape[-1]),
            backward_edge_features.reshape(-1, backward_edge_features.shape[-1]),
        ])
        encoded = self._encode_nodes(nodes.flatten(0, 1), edge_index, edge_features)
        encoded = encoded.reshape(time_steps, num_devices + num_servers, -1)
        return encoded[:, :num_devices]
