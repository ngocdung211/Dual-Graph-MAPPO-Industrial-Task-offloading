"""Build topology graph tensors from the flat DITEN joint state."""

from dataclasses import dataclass

import torch


STANDARD_TOPOLOGY_EDGE_FEATURE_DIM = 7
LIGHTWEIGHT_TOPOLOGY_EDGE_FEATURE_DIM = 3


@dataclass(frozen=True)
class TopologyGraphState:
    """Graph tensors for device-server topology state."""

    node_features: torch.Tensor
    edge_index: torch.Tensor
    edge_features: torch.Tensor
    device_node_indices: torch.Tensor
    server_node_indices: torch.Tensor


def build_topology_graph_state(
    joint_state: torch.Tensor,
    num_devices: int,
    num_servers: int,
    lightweight: bool = False,
) -> TopologyGraphState:
    """Convert the flat DITEN joint state into graph tensors.

    Args:
        joint_state: Flat state tensor shaped `(num_devices, state_dim)`.
        num_devices: Number of device nodes.
        num_servers: Number of edge-server nodes.
        lightweight: Build one server-to-device edge per pair with three
            nonredundant connection features instead of two directed edges.

    Returns:
        Topology graph state with shared node features and configured links.
        Edges are ordered by device and server; standard graphs then alternate
        device-to-server and server-to-device directions.

    Raises:
        ValueError: If the flat state dimensions are incompatible.
    """
    state_tensor = torch.as_tensor(joint_state, dtype=torch.float32)
    if state_tensor.ndim != 2 or state_tensor.shape[0] != num_devices:
        raise ValueError("joint_state must have shape (num_devices, state_dim)")

    state_width = int(state_tensor.shape[1])
    fixed_width = 5 + 4 * num_servers
    priority_width = state_width - fixed_width
    if priority_width <= 0:
        raise ValueError("flat state width is incompatible with num_servers")

    edge_power_offset = 5 + priority_width
    edge_wait_offset = edge_power_offset + num_servers
    window_start_offset = edge_wait_offset + num_servers
    window_end_offset = window_start_offset + num_servers

    device_features = _build_device_node_features(state_tensor, priority_width)
    server_features = _build_server_node_features(
        state_tensor, edge_power_offset, edge_wait_offset, num_servers, priority_width
    )
    node_features = torch.cat([device_features, server_features], dim=0)
    edge_builder = (
        _build_lightweight_connection_edges
        if lightweight
        else _build_valid_connection_edges
    )
    edge_index, edge_features = edge_builder(
        state_tensor,
        num_devices,
        num_servers,
        window_start_offset,
        window_end_offset,
    )

    return TopologyGraphState(
        node_features=node_features,
        edge_index=edge_index,
        edge_features=edge_features,
        device_node_indices=torch.arange(num_devices, dtype=torch.long),
        server_node_indices=torch.arange(
            num_devices, num_devices + num_servers, dtype=torch.long
        ),
    )


def _build_device_node_features(
    state_tensor: torch.Tensor, priority_width: int
) -> torch.Tensor:
    """Build padded node features for device nodes."""
    num_devices = state_tensor.shape[0]
    features = torch.zeros((num_devices, 9 + priority_width), dtype=torch.float32)
    features[:, 0] = 1.0
    features[:, 2] = state_tensor[:, 0]
    features[:, 3] = state_tensor[:, 1]
    features[:, 4] = state_tensor[:, 2]
    features[:, 5] = state_tensor[:, 3]
    features[:, 6] = state_tensor[:, 4]
    features[:, 9:] = state_tensor[:, 5 : 5 + priority_width]
    return features


def _build_server_node_features(
    state_tensor: torch.Tensor,
    edge_power_offset: int,
    edge_wait_offset: int,
    num_servers: int,
    priority_width: int,
) -> torch.Tensor:
    """Build padded node features for server nodes."""
    features = torch.zeros((num_servers, 9 + priority_width), dtype=torch.float32)
    features[:, 1] = 1.0
    features[:, 7] = state_tensor[
        0, edge_power_offset : edge_power_offset + num_servers
    ]
    features[:, 8] = state_tensor[
        0, edge_wait_offset : edge_wait_offset + num_servers
    ]
    return features


def _build_valid_connection_edges(
    state_tensor: torch.Tensor,
    num_devices: int,
    num_servers: int,
    window_start_offset: int,
    window_end_offset: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build bidirectional edges for all links with connection flags."""
    edge_pairs: list[tuple[int, int]] = []
    edge_features: list[list[float]] = []

    for device_index in range(num_devices):
        for server_index in range(num_servers):
            window_start = float(
                state_tensor[device_index, window_start_offset + server_index]
            )
            window_end = float(
                state_tensor[device_index, window_end_offset + server_index]
            )
            window_length = max(0.0, window_end - window_start)
            is_connected = 1.0 if window_length > 0.0 else 0.0
            is_disconnected = 1.0 - is_connected

            device_node = device_index
            server_node = num_devices + server_index
            forward_features = [
                1.0,
                0.0,
                is_connected,
                is_disconnected,
                window_start,
                window_end,
                window_length,
            ]
            backward_features = [
                0.0,
                1.0,
                is_connected,
                is_disconnected,
                window_start,
                window_end,
                window_length,
            ]

            edge_pairs.append((device_node, server_node))
            edge_features.append(forward_features)
            edge_pairs.append((server_node, device_node))
            edge_features.append(backward_features)

    if not edge_pairs:
        return (
            torch.empty((2, 0), dtype=torch.long),
            torch.empty((0, 7), dtype=torch.float32),
        )

    return (
        torch.tensor(edge_pairs, dtype=torch.long).transpose(0, 1),
        torch.tensor(edge_features, dtype=torch.float32),
    )


def _build_lightweight_connection_edges(
    state_tensor: torch.Tensor,
    num_devices: int,
    num_servers: int,
    window_start_offset: int,
    window_end_offset: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build one server-to-device edge with three features for every pair."""
    edge_pairs: list[tuple[int, int]] = []
    edge_features: list[list[float]] = []

    for device_index in range(num_devices):
        for server_index in range(num_servers):
            window_start = float(
                state_tensor[device_index, window_start_offset + server_index]
            )
            window_end = float(
                state_tensor[device_index, window_end_offset + server_index]
            )
            window_length = max(0.0, window_end - window_start)
            is_connected = 1.0 if window_length > 0.0 else 0.0
            device_node = device_index
            server_node = num_devices + server_index

            edge_pairs.append((server_node, device_node))
            edge_features.append(
                [is_connected, window_start, window_length]
            )

    if not edge_pairs:
        return (
            torch.empty((2, 0), dtype=torch.long),
            torch.empty(
                (0, LIGHTWEIGHT_TOPOLOGY_EDGE_FEATURE_DIM),
                dtype=torch.float32,
            ),
        )

    return (
        torch.tensor(edge_pairs, dtype=torch.long).transpose(0, 1),
        torch.tensor(edge_features, dtype=torch.float32),
    )
