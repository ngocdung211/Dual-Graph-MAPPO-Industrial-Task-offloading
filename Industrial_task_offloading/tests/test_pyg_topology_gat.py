"""Checks for the separate library-GAT experiment and graph isolation."""

import copy
import importlib.util

import pytest
import torch

if importlib.util.find_spec("torch_geometric") is None:
    pytest.skip("optional PyG dependency is not installed", allow_module_level=True)

from baselines.graph_gat_mappo import GraphGATMAPPOAgent, GraphGATRolloutBuffer
from models.pyg_topology_gat import PyGTopologyGATEncoder
from models.topology_gat import TopologyGATEncoder
from utils.comparison.algorithm_config import (
    build_algorithm_configs,
    select_algorithm_configs,
)
from utils.topology.graph_state import build_topology_graph_state


def _make_graph_state(offset=0.0):
    """Build a two-device/two-server graph with unequal link attributes."""
    state = torch.tensor([
        [1.0 + offset, 0.1, 0.5, 0.2, 0.05, 0.2, 0.4, 0.6, 0.8,
         1.0, 2.3, 2.5, 0.1, 0.2, 0.0, 1.0, 0.8, 1.0],
        [0.9 + offset, 0.3, 0.7, 0.4, 0.06, 0.2, 0.4, 0.6, 0.8,
         1.0, 2.3, 2.5, 0.1, 0.2, 1.0, 0.25, 1.0, 0.75],
    ], dtype=torch.float32)
    return build_topology_graph_state(state, num_devices=2, num_servers=2)


def _policy_and_values_sequential_reference(agent, graphs, actions):
    """Evaluate each timestep independently using an explicit global graph."""
    log_probs, entropies, values = [], [], []
    for graph, action in zip(graphs, actions):
        distribution = torch.distributions.Categorical(
            agent._actor_probabilities_for_graph_state(graph)
        )
        log_probs.append(distribution.log_prob(action))
        entropies.append(distribution.entropy())
        embeddings = agent.encoder(
            graph.node_features, graph.edge_index, graph.edge_features,
            graph.device_node_indices,
        )
        values.append(agent.critic(embeddings))
    return torch.stack(log_probs), torch.stack(entropies).mean(), torch.cat(values)


def _make_agent(**kwargs):
    """Create a small CPU PyG controller with GAE and minibatches."""
    return GraphGATMAPPOAgent(
        num_devices=2, num_servers=2, node_feature_dim=14,
        edge_feature_dim=7, hidden_dim=8, embedding_dim=8,
        encoder_backend="pyg", use_gae=True, num_minibatches=2,
        **kwargs,
    )


def test_pyg_registry_preserves_custom_and_default_selection():
    """PyG is separately selectable without adding a default dependency."""
    configs = build_algorithm_configs(use_gae=True, num_minibatches=4)
    custom = configs["Graph-GAT MAPPO"]["kwargs"]
    pyg = configs["PyG-GAT MAPPO"]["kwargs"]
    assert pyg == {**custom, "encoder_backend": "pyg"}
    assert "PyG-GAT MAPPO" not in select_algorithm_configs(configs, None)
    assert "PyG-GAT MAPPO" in select_algorithm_configs(configs, ["PyG-GAT MAPPO"])
    masked = configs["PyG-GAT Mask MAPPO"]["kwargs"]
    assert masked == {
        **configs["Graph-GAT Mask MAPPO"]["kwargs"], "encoder_backend": "pyg",
    }
    assert masked == {**pyg, "use_action_mask": True}
    assert "PyG-GAT Mask MAPPO" not in select_algorithm_configs(configs, None)
    selected = select_algorithm_configs(
        configs, ["Graph-GAT Mask MAPPO", "PyG-GAT Mask MAPPO"],
    )
    assert list(selected) == ["Graph-GAT Mask MAPPO", "PyG-GAT Mask MAPPO"]
    agent = GraphGATMAPPOAgent(
        num_devices=2, num_servers=2, node_feature_dim=14, edge_feature_dim=7,
    )
    assert type(agent.encoder) is TopologyGATEncoder


def test_pyg_local_batch_matches_independent_explicit_graphs():
    """Local batching cannot leak other devices into the actor context."""
    torch.manual_seed(17)
    agent = _make_agent()
    graph = _make_graph_state()
    devices = graph.node_features[graph.device_node_indices]
    servers = graph.node_features[graph.server_node_indices]
    forward_edges, backward_edges = agent._encoder_edge_features(graph)
    batched_devices, batched_servers = agent.encoder.forward_batched_local_nodes(
        devices, servers, forward_edges, backward_edges,
    )
    for device_index in range(2):
        local = agent._local_subgraph_for_device(graph, device_index)
        explicit = agent.encoder._encode_nodes(
            local.node_features, local.edge_index, local.edge_features,
        )
        torch.testing.assert_close(batched_devices[device_index], explicit[0])
        torch.testing.assert_close(batched_servers[device_index], explicit[1:])


def test_pyg_rollout_outputs_and_gradients_match_sequential_graphs():
    """Disjoint rollout graphs preserve independent outputs and gradients."""
    torch.manual_seed(23)
    batched = _make_agent()
    sequential = copy.deepcopy(batched)
    graphs = [_make_graph_state(offset=0.2 * step) for step in range(3)]
    actions = torch.tensor([[0, 1], [1, 2], [2, 0]])
    actual = batched._policy_and_values_from_stacked(
        *batched._stack_graph_features(graphs), actions,
    )
    expected = _policy_and_values_sequential_reference(sequential, graphs, actions)
    for batch_output, reference in zip(actual, expected):
        torch.testing.assert_close(batch_output, reference)
    sum(output.sum() for output in actual).backward()
    sum(output.sum() for output in expected).backward()
    for batch_parameter, reference in zip(
        batched.ppo_parameters, sequential.ppo_parameters,
    ):
        assert batch_parameter.grad is not None
        assert torch.isfinite(batch_parameter.grad).all()
        torch.testing.assert_close(
            batch_parameter.grad, reference.grad, atol=1e-5, rtol=1e-4,
        )


@pytest.mark.parametrize("use_action_mask", [False, True])
def test_pyg_ppo_update_and_checkpoint_round_trip(tmp_path, use_action_mask):
    """GAE training changes all PPO modules and preserves saved predictions."""
    torch.manual_seed(29)
    agent = _make_agent(ppo_epochs=2, use_action_mask=use_action_mask)
    before = {
        name: [parameter.detach().clone() for parameter in module.parameters()]
        for name, module in (
            ("encoder", agent.encoder), ("actor", agent.actor),
            ("critic", agent.critic),
        )
    }
    buffer = GraphGATRolloutBuffer()
    for step in range(4):
        graph = _make_graph_state(offset=step * 0.1)
        actions, log_probs = agent.select_actions_with_log_probs(graph)
        buffer.push(
            graph_state=graph, actions=actions, rewards=[1.0, 0.5 + step * 0.1],
            next_graph_state=_make_graph_state(offset=(step + 1) * 0.1),
            old_log_probs=log_probs, done=step == 3,
        )
    agent.update_from_rollout(buffer)
    assert len(buffer) == 0
    checkpoint = {}
    for name, old_parameters in before.items():
        module = getattr(agent, name)
        assert any(
            not torch.equal(old, new)
            for old, new in zip(old_parameters, module.parameters())
        )
        checkpoint[name] = module.state_dict()
    checkpoint_path = tmp_path / "pyg_checkpoint.pt"
    torch.save(checkpoint, checkpoint_path)
    restored = _make_agent(use_action_mask=use_action_mask)
    for name, state in torch.load(checkpoint_path, weights_only=True).items():
        getattr(restored, name).load_state_dict(state)
    assert isinstance(restored.encoder, PyGTopologyGATEncoder)
    graph = _make_graph_state()
    torch.testing.assert_close(
        agent._actor_probabilities_for_graph_state(graph),
        restored._actor_probabilities_for_graph_state(graph),
    )
    torch.testing.assert_close(agent._encode_graph(graph), restored._encode_graph(graph))


def test_pyg_mask_blocks_disconnected_servers_and_keeps_local_execution():
    """Registered mask variant samples only feasible actions, even without links."""
    config = build_algorithm_configs()["PyG-GAT Mask MAPPO"]
    agent = config["class"](
        num_devices=2, num_servers=2, node_feature_dim=14, edge_feature_dim=7,
        **config["kwargs"],
    )
    graph = _make_graph_state()
    # Device 0 can reach only server 2; device 1 has no connected servers.
    connected = torch.tensor([[0.0, 1.0], [0.0, 0.0]])
    graph.edge_features.reshape(2, 2, 2, 7)[..., 2] = connected.unsqueeze(-1)
    probabilities = agent._actor_probabilities_for_graph_state(graph)
    torch.testing.assert_close(probabilities.sum(dim=-1), torch.ones(2))
    assert probabilities[0, 1] == 0
    assert probabilities[0, 0] > 0
    assert probabilities[0, 2] > 0
    torch.testing.assert_close(probabilities[1], torch.tensor([1.0, 0.0, 0.0]))
    for _ in range(10):
        actions, log_probs = agent.select_actions_with_log_probs(graph)
        assert actions[0] in (0, 2)
        assert actions[1] == 0
        assert torch.isfinite(torch.tensor(log_probs)).all()
