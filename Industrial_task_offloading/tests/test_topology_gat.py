"""Tests for topology graph attention encoding."""

import pathlib
import sys

import torch

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.topology_gat import TopologyGATEncoder
from utils.topology_graph_state import build_topology_graph_state


def _make_joint_state() -> torch.Tensor:
    """Create a two-device, two-server flat state with valid graph links."""
    return torch.tensor(
        [
            [
                1.0,
                0.1,
                0.5,
                0.2,
                0.05,
                0.2,
                0.4,
                0.6,
                0.8,
                1.0,
                2.3,
                2.5,
                0.1,
                0.2,
                0.0,
                1.0,
                0.8,
                1.0,
            ],
            [
                0.9,
                0.3,
                0.7,
                0.4,
                0.06,
                0.2,
                0.4,
                0.6,
                0.8,
                1.0,
                2.3,
                2.5,
                0.1,
                0.2,
                1.0,
                0.25,
                1.0,
                0.75,
            ],
        ],
        dtype=torch.float32,
    )


def test_topology_gat_encoder_returns_device_embeddings() -> None:
    """Encoder output should contain one embedding per device node."""
    graph_state = build_topology_graph_state(
        _make_joint_state(), num_devices=2, num_servers=2
    )
    encoder = TopologyGATEncoder(
        node_feature_dim=14,
        edge_feature_dim=7,
        hidden_dim=16,
        embedding_dim=8,
    )

    device_embeddings = encoder(
        graph_state.node_features,
        graph_state.edge_index,
        graph_state.edge_features,
        graph_state.device_node_indices,
    )

    assert device_embeddings.shape == (2, 8)
    assert torch.isfinite(device_embeddings).all()


def test_topology_gat_encoder_backpropagates_gradients() -> None:
    """A loss on device embeddings should update GAT encoder parameters."""
    graph_state = build_topology_graph_state(
        _make_joint_state(), num_devices=2, num_servers=2
    )
    encoder = TopologyGATEncoder(
        node_feature_dim=14,
        edge_feature_dim=7,
        hidden_dim=16,
        embedding_dim=8,
    )

    device_embeddings = encoder(
        graph_state.node_features,
        graph_state.edge_index,
        graph_state.edge_features,
        graph_state.device_node_indices,
    )
    loss = device_embeddings.pow(2).mean()
    loss.backward()

    gradients = [
        parameter.grad
        for parameter in encoder.parameters()
        if parameter.requires_grad
    ]
    assert gradients
    assert all(gradient is not None for gradient in gradients)
    assert any(torch.any(gradient != 0.0) for gradient in gradients)


def test_topology_gat_batched_local_encoder_backpropagates_gradients() -> None:
    """Batched local encoding should retain gradients through both GAT layers."""
    graph_state = build_topology_graph_state(
        _make_joint_state(), num_devices=2, num_servers=2
    )
    encoder = TopologyGATEncoder(
        node_feature_dim=14,
        edge_feature_dim=7,
        hidden_dim=16,
        embedding_dim=8,
    )
    edge_features = graph_state.edge_features.reshape(2, 2, 2, 7)

    device_embeddings = encoder.forward_batched_local(
        device_features=graph_state.node_features[:2],
        server_features=graph_state.node_features[2:],
        forward_edge_features=edge_features[:, :, 0, :],
        backward_edge_features=edge_features[:, :, 1, :],
    )
    loss = device_embeddings.pow(2).mean()
    loss.backward()

    gradients = [
        parameter.grad
        for parameter in encoder.parameters()
        if parameter.requires_grad
    ]
    assert gradients
    assert all(gradient is not None for gradient in gradients)
    assert any(torch.any(gradient != 0.0) for gradient in gradients)


def test_topology_gat_batched_local_returns_server_embeddings() -> None:
    """Local encoding should preserve one server embedding per device pair."""
    graph_state = build_topology_graph_state(
        _make_joint_state(), num_devices=2, num_servers=2
    )
    encoder = TopologyGATEncoder(
        node_feature_dim=14,
        edge_feature_dim=7,
        hidden_dim=16,
        embedding_dim=8,
    )
    edge_features = graph_state.edge_features.reshape(2, 2, 2, 7)

    device_embeddings, server_embeddings = (
        encoder.forward_batched_local_nodes(
            device_features=graph_state.node_features[:2],
            server_features=graph_state.node_features[2:],
            forward_edge_features=edge_features[:, :, 0, :],
            backward_edge_features=edge_features[:, :, 1, :],
        )
    )

    assert device_embeddings.shape == (2, 8)
    assert server_embeddings.shape == (2, 2, 8)
    assert torch.isfinite(server_embeddings).all()


def test_batched_global_embeddings_match_graph_encoder() -> None:
    """Dense global encoding should preserve the complete graph calculation."""
    torch.manual_seed(53)
    graph_state = build_topology_graph_state(
        _make_joint_state(), num_devices=2, num_servers=2
    )
    encoder = TopologyGATEncoder(
        node_feature_dim=14,
        edge_feature_dim=7,
        hidden_dim=16,
        embedding_dim=8,
    )
    edge_features = graph_state.edge_features.reshape(2, 2, 2, 7)

    graph_embeddings = encoder(
        graph_state.node_features,
        graph_state.edge_index,
        graph_state.edge_features,
        graph_state.device_node_indices,
    )
    batched_embeddings = encoder.forward_batched_global(
        device_features=graph_state.node_features[:2],
        server_features=graph_state.node_features[2:],
        forward_edge_features=edge_features[:, :, 0, :],
        backward_edge_features=edge_features[:, :, 1, :],
    )

    assert torch.allclose(batched_embeddings, graph_embeddings, atol=1e-6)


def test_batched_global_gradients_match_graph_encoder() -> None:
    """Dense global encoding should preserve complete-graph gradients."""
    torch.manual_seed(59)
    graph_state = build_topology_graph_state(
        _make_joint_state(), num_devices=2, num_servers=2
    )
    graph_encoder = TopologyGATEncoder(
        node_feature_dim=14,
        edge_feature_dim=7,
        hidden_dim=16,
        embedding_dim=8,
    )
    batched_encoder = TopologyGATEncoder(
        node_feature_dim=14,
        edge_feature_dim=7,
        hidden_dim=16,
        embedding_dim=8,
    )
    batched_encoder.load_state_dict(graph_encoder.state_dict())
    edge_features = graph_state.edge_features.reshape(2, 2, 2, 7)

    graph_embeddings = graph_encoder(
        graph_state.node_features,
        graph_state.edge_index,
        graph_state.edge_features,
        graph_state.device_node_indices,
    )
    graph_embeddings.pow(2).mean().backward()
    batched_embeddings = batched_encoder.forward_batched_global(
        device_features=graph_state.node_features[:2],
        server_features=graph_state.node_features[2:],
        forward_edge_features=edge_features[:, :, 0, :],
        backward_edge_features=edge_features[:, :, 1, :],
    )
    batched_embeddings.pow(2).mean().backward()

    for graph_parameter, batched_parameter in zip(
        graph_encoder.parameters(), batched_encoder.parameters()
    ):
        assert torch.allclose(
            graph_parameter.grad, batched_parameter.grad, atol=1e-6
        )
