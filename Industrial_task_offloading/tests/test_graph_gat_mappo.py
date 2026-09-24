"""Tests for Graph-GAT MAPPO ablation behavior."""

import pathlib
import sys

import pytest
import torch
from torch.distributions import Categorical

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from baselines.gatma import GATMAAgent
from baselines.graph_gat_mappo import GraphGATMAPPOAgent, GraphGATRolloutBuffer
from run_comparision import (
    _collect_graph_gat_actions,
    _update_graph_gat_mappo_from_rollout,
    build_algorithm_configs,
    select_algorithm_configs,
)
from utils.paper_config import PAPER_PARAMS
from utils.topology_graph_state import build_topology_graph_state


def _make_joint_state(offset: float = 0.0) -> torch.Tensor:
    """Create a two-device, two-server flat state with valid graph links."""
    return torch.tensor(
        [
            [
                1.0 + offset,
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
                0.9 + offset,
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


def _make_graph_state(offset: float = 0.0):
    """Create a topology graph state fixture."""
    return build_topology_graph_state(
        _make_joint_state(offset), num_devices=2, num_servers=2
    )


def _make_lightweight_graph_state(offset: float = 0.0):
    """Create a one-way, three-edge-feature topology graph fixture."""
    return build_topology_graph_state(
        _make_joint_state(offset),
        num_devices=2,
        num_servers=2,
        lightweight=True,
    )


def _clone_module_parameters(module: torch.nn.Module):
    """Return detached copies of module parameters."""
    return [parameter.detach().clone() for parameter in module.parameters()]


def _parameters_changed(before, after) -> bool:
    """Return whether any parameter tensor changed."""
    return any(not torch.allclose(old, new) for old, new in zip(before, after))


def _encode_sequential_local_embeddings(
    agent: GraphGATMAPPOAgent, graph_state
) -> torch.Tensor:
    """Encode local graphs through the original sequential reference path."""
    embeddings = []
    for device_index in range(agent.num_devices):
        local_graph = agent._local_subgraph_for_device(graph_state, device_index)
        embeddings.append(
            agent.encoder(
                local_graph.node_features,
                local_graph.edge_index,
                local_graph.edge_features,
                local_graph.device_node_indices,
            )[0]
        )
    return torch.stack(embeddings)


def _policy_and_values_sequential_reference(
    agent: GraphGATMAPPOAgent, graph_states, actions: torch.Tensor
):
    """Evaluate rollout graphs through the previous transition loop."""
    log_prob_rows = []
    entropy_rows = []
    value_rows = []
    for graph_index, graph_state in enumerate(graph_states):
        probabilities = agent._actor_probabilities_for_graph_state(graph_state)
        distribution = Categorical(probabilities)
        log_prob_rows.append(distribution.log_prob(actions[graph_index]))
        entropy_rows.append(distribution.entropy())
        value_rows.append(agent.critic(agent._encode_graph(graph_state)))
    return (
        torch.stack(log_prob_rows),
        torch.stack(entropy_rows).mean(),
        torch.cat(value_rows),
    )


def test_graph_gat_mappo_samples_valid_actions_from_graph_state() -> None:
    """Graph-GAT MAPPO should sample one valid offloading action per device."""
    torch.manual_seed(31)
    agent = GraphGATMAPPOAgent(
        num_devices=2,
        num_servers=2,
        node_feature_dim=14,
        edge_feature_dim=7,
        embedding_dim=8,
    )

    actions, log_probs = agent.select_actions_with_log_probs(_make_graph_state())

    assert len(actions) == 2
    assert len(log_probs) == 2
    assert all(0 <= action < agent.action_dim for action in actions)
    assert all(isinstance(log_prob, float) for log_prob in log_probs)


def test_graph_gat_mappo_cpu_device_keeps_modules_and_outputs_on_cpu() -> None:
    """CPU mode should keep model computation on the control device."""
    agent = GraphGATMAPPOAgent(
        num_devices=2,
        num_servers=2,
        node_feature_dim=14,
        edge_feature_dim=7,
        embedding_dim=8,
        device="cpu",
    )
    graph_state = _make_graph_state()

    probabilities = agent._actor_probabilities_for_graph_state(graph_state)
    values = agent._values_for_graphs([graph_state])

    assert agent.device == torch.device("cpu")
    assert next(agent.encoder.parameters()).device.type == "cpu"
    assert next(agent.actor.parameters()).device.type == "cpu"
    assert next(agent.critic.parameters()).device.type == "cpu"
    assert probabilities.device.type == "cpu"
    assert values.device.type == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_graph_gat_mappo_cuda_smoke_update() -> None:
    """CUDA mode should select actions and complete one rollout update."""
    torch.manual_seed(47)
    agent = GraphGATMAPPOAgent(
        num_devices=2,
        num_servers=2,
        node_feature_dim=14,
        edge_feature_dim=7,
        embedding_dim=8,
        ppo_epochs=1,
        device="cuda",
    )
    buffer = GraphGATRolloutBuffer()
    for step_index in range(2):
        graph_state = _make_graph_state(offset=0.05 * step_index)
        actions, old_log_probs = agent.select_actions_with_log_probs(graph_state)
        buffer.push(
            graph_state=graph_state,
            actions=actions,
            rewards=[1.0, 0.5],
            next_graph_state=_make_graph_state(offset=0.05 * (step_index + 1)),
            old_log_probs=old_log_probs,
            done=step_index == 1,
        )

    agent.update_from_rollout(buffer)
    agent.synchronize_device()

    assert agent.device.type == "cuda"
    assert next(agent.encoder.parameters()).device.type == "cuda"
    assert len(buffer) == 0


def test_graph_gat_mappo_masks_disconnected_server_actions() -> None:
    """Graph-GAT MAPPO should assign zero probability to disconnected servers."""
    torch.manual_seed(33)
    agent = GraphGATMAPPOAgent(
        num_devices=2,
        num_servers=2,
        node_feature_dim=14,
        edge_feature_dim=7,
        embedding_dim=8,
    )
    graph_state = _make_graph_state()
    device_embeddings, server_embeddings = agent._encode_local_actor_nodes(
        graph_state
    )
    forward_edge_features = agent._device_server_edge_features(
        graph_state
    )[:, :, 0, :]
    probabilities = agent.actor(
        device_embeddings, server_embeddings, forward_edge_features
    )

    mask = agent._action_mask_for_graph_state(graph_state)
    masked_probabilities = agent._masked_action_probabilities(
        probabilities, graph_state
    )

    assert mask.tolist() == [[True, True, False], [True, False, True]]
    assert masked_probabilities[0, 2].item() == 0.0
    assert masked_probabilities[1, 1].item() == 0.0
    assert torch.allclose(masked_probabilities.sum(dim=1), torch.ones(2))


def test_graph_gat_mappo_action_mask_can_be_disabled() -> None:
    """Graph-GAT MAPPO mask toggle should preserve raw probabilities."""
    torch.manual_seed(34)
    agent = GraphGATMAPPOAgent(
        num_devices=2,
        num_servers=2,
        node_feature_dim=14,
        edge_feature_dim=7,
        embedding_dim=8,
        use_action_mask=False,
    )
    graph_state = _make_graph_state()
    device_embeddings, server_embeddings = agent._encode_local_actor_nodes(
        graph_state
    )
    forward_edge_features = agent._device_server_edge_features(
        graph_state
    )[:, :, 0, :]
    probabilities = agent.actor(
        device_embeddings, server_embeddings, forward_edge_features
    )

    masked_probabilities = agent._masked_action_probabilities(
        probabilities, graph_state
    )

    assert torch.allclose(masked_probabilities, probabilities)


def test_graph_gat_actor_is_equivariant_to_server_permutation() -> None:
    """Swapping server inputs should swap only the corresponding actions."""
    torch.manual_seed(45)
    agent = GraphGATMAPPOAgent(
        num_devices=2,
        num_servers=2,
        node_feature_dim=14,
        edge_feature_dim=7,
        embedding_dim=8,
        use_action_mask=False,
    )
    graph_state = _make_graph_state()
    device_features = agent._selected_node_features(
        graph_state, graph_state.device_node_indices
    )
    server_features = agent._selected_node_features(
        graph_state, graph_state.server_node_indices
    )
    edge_features = agent._device_server_edge_features(graph_state)
    forward_edge_features = edge_features[:, :, 0, :]
    backward_edge_features = edge_features[:, :, 1, :]

    device_embeddings, server_embeddings = (
        agent.encoder.forward_batched_local_nodes(
            device_features,
            server_features,
            forward_edge_features,
            backward_edge_features,
        )
    )
    probabilities = agent.actor(
        device_embeddings, server_embeddings, forward_edge_features
    )

    server_order = torch.tensor([1, 0])
    permuted_device_embeddings, permuted_server_embeddings = (
        agent.encoder.forward_batched_local_nodes(
            device_features,
            server_features[server_order],
            forward_edge_features[:, server_order],
            backward_edge_features[:, server_order],
        )
    )
    permuted_probabilities = agent.actor(
        permuted_device_embeddings,
        permuted_server_embeddings,
        forward_edge_features[:, server_order],
    )
    expected_probabilities = torch.cat(
        [probabilities[:, :1], probabilities[:, 1:][:, server_order]],
        dim=-1,
    )

    assert torch.allclose(
        permuted_probabilities, expected_probabilities, atol=1e-6
    )


def test_graph_gat_topology_warmup_updates_encoder_parameters() -> None:
    """Topology warmup should update GAT encoder before PPO action selection."""
    torch.manual_seed(34)
    agent = GraphGATMAPPOAgent(
        num_devices=2,
        num_servers=2,
        node_feature_dim=14,
        edge_feature_dim=7,
        embedding_dim=8,
        topology_warmup_lr=0.01,
    )
    graph_state = _make_graph_state()
    encoder_before = _clone_module_parameters(agent.encoder)

    loss_value = agent.warmup_topology_encoder(graph_state, update_count=2)

    assert loss_value > 0.0
    assert _parameters_changed(encoder_before, _clone_module_parameters(agent.encoder))


def test_lightweight_graph_gat_runs_one_layer_warmup_and_policy() -> None:
    """Lightweight Graph-GAT should use one layer and compact edge features."""
    torch.manual_seed(46)
    agent = GraphGATMAPPOAgent(
        num_devices=2,
        num_servers=2,
        node_feature_dim=14,
        edge_feature_dim=3,
        embedding_dim=8,
        lightweight_topology=True,
        topology_warmup_lr=0.01,
    )
    graph_state = _make_lightweight_graph_state()

    actions, log_probs = agent.select_actions_with_log_probs(graph_state)
    warmup_loss = agent.warmup_topology_encoder(graph_state, update_count=1)

    assert agent.encoder.gat2 is None
    assert graph_state.edge_features.shape == (4, 3)
    assert len(actions) == 2
    assert len(log_probs) == 2
    assert warmup_loss > 0.0


def test_graph_gat_mappo_actor_uses_local_subgraph_embeddings() -> None:
    """One device actor input should not depend on another device feature."""
    torch.manual_seed(35)
    agent = GraphGATMAPPOAgent(
        num_devices=2,
        num_servers=2,
        node_feature_dim=14,
        edge_feature_dim=7,
        embedding_dim=8,
    )
    base_state = _make_joint_state()
    changed_state = base_state.clone()
    changed_state[1, 0:10] = changed_state[1, 0:10] + 7.0

    base_probabilities = agent._actor_probabilities_for_graph_state(
        build_topology_graph_state(base_state, num_devices=2, num_servers=2)
    )
    changed_probabilities = agent._actor_probabilities_for_graph_state(
        build_topology_graph_state(changed_state, num_devices=2, num_servers=2)
    )

    assert torch.allclose(base_probabilities[0], changed_probabilities[0], atol=1e-6)


def test_batched_local_embeddings_match_sequential_reference() -> None:
    """Batched local encoding should preserve the previous graph calculation."""
    torch.manual_seed(36)
    agent = GraphGATMAPPOAgent(
        num_devices=2,
        num_servers=2,
        node_feature_dim=14,
        edge_feature_dim=7,
        embedding_dim=8,
    )
    graph_state = _make_graph_state()
    sequential_embeddings = _encode_sequential_local_embeddings(
        agent, graph_state
    )

    batched_embeddings = agent._encode_local_actor_embeddings(graph_state)

    assert torch.allclose(batched_embeddings, sequential_embeddings, atol=1e-6)


def test_batched_local_encoder_gradients_match_sequential_reference() -> None:
    """Batched local encoding should preserve encoder training gradients."""
    torch.manual_seed(38)
    sequential_agent = GraphGATMAPPOAgent(
        num_devices=2,
        num_servers=2,
        node_feature_dim=14,
        edge_feature_dim=7,
        embedding_dim=8,
    )
    batched_agent = GraphGATMAPPOAgent(
        num_devices=2,
        num_servers=2,
        node_feature_dim=14,
        edge_feature_dim=7,
        embedding_dim=8,
    )
    batched_agent.encoder.load_state_dict(sequential_agent.encoder.state_dict())
    graph_state = _make_graph_state()

    sequential_embeddings = _encode_sequential_local_embeddings(
        sequential_agent, graph_state
    )
    sequential_embeddings.pow(2).mean().backward()

    batched_embeddings = batched_agent._encode_local_actor_embeddings(graph_state)
    batched_embeddings.pow(2).mean().backward()

    for sequential_parameter, batched_parameter in zip(
        sequential_agent.encoder.parameters(), batched_agent.encoder.parameters()
    ):
        assert torch.allclose(
            sequential_parameter.grad, batched_parameter.grad, atol=1e-6
        )


def test_batched_global_critic_value_matches_graph_reference() -> None:
    """Vectorized global topology should preserve centralized critic values."""
    torch.manual_seed(39)
    agent = GraphGATMAPPOAgent(
        num_devices=2,
        num_servers=2,
        node_feature_dim=14,
        edge_feature_dim=7,
        embedding_dim=8,
    )
    graph_state = _make_graph_state()
    reference_embeddings = agent.encoder(
        graph_state.node_features,
        graph_state.edge_index,
        graph_state.edge_features,
        graph_state.device_node_indices,
    )

    reference_value = agent.critic(reference_embeddings)
    batched_value = agent.critic(agent._encode_graph(graph_state))

    assert torch.allclose(batched_value, reference_value, atol=1e-6)


def test_batched_rollout_policy_and_values_match_sequential_reference() -> None:
    """Time-batched PPO evaluation should preserve policy and critic outputs."""
    torch.manual_seed(40)
    agent = GraphGATMAPPOAgent(
        num_devices=2,
        num_servers=2,
        node_feature_dim=14,
        edge_feature_dim=7,
        embedding_dim=8,
    )
    graph_states = [
        _make_graph_state(offset=0.05 * step_index) for step_index in range(4)
    ]
    actions = torch.tensor([[0, 2], [1, 0], [0, 2], [1, 0]])

    reference = _policy_and_values_sequential_reference(
        agent, graph_states, actions
    )
    batched = agent._policy_and_values_for_graphs(graph_states, actions)

    for batched_value, reference_value in zip(batched, reference):
        assert torch.allclose(batched_value, reference_value, atol=1e-6)


def test_batched_rollout_gradients_match_sequential_reference() -> None:
    """Time-batched PPO evaluation should preserve training gradients."""
    torch.manual_seed(42)
    sequential_agent = GraphGATMAPPOAgent(
        num_devices=2,
        num_servers=2,
        node_feature_dim=14,
        edge_feature_dim=7,
        embedding_dim=8,
    )
    batched_agent = GraphGATMAPPOAgent(
        num_devices=2,
        num_servers=2,
        node_feature_dim=14,
        edge_feature_dim=7,
        embedding_dim=8,
    )
    for sequential_module, batched_module in (
        (sequential_agent.encoder, batched_agent.encoder),
        (sequential_agent.actor, batched_agent.actor),
        (sequential_agent.critic, batched_agent.critic),
    ):
        batched_module.load_state_dict(sequential_module.state_dict())
    graph_states = [
        _make_graph_state(offset=0.05 * step_index) for step_index in range(4)
    ]
    actions = torch.tensor([[0, 2], [1, 0], [0, 2], [1, 0]])

    sequential_outputs = _policy_and_values_sequential_reference(
        sequential_agent, graph_states, actions
    )
    sequential_loss = (
        sequential_outputs[0].pow(2).mean()
        + sequential_outputs[1]
        + sequential_outputs[2].pow(2).mean()
    )
    sequential_loss.backward()
    batched_outputs = batched_agent._policy_and_values_for_graphs(
        graph_states, actions
    )
    batched_loss = (
        batched_outputs[0].pow(2).mean()
        + batched_outputs[1]
        + batched_outputs[2].pow(2).mean()
    )
    batched_loss.backward()

    for sequential_module, batched_module in (
        (sequential_agent.encoder, batched_agent.encoder),
        (sequential_agent.actor, batched_agent.actor),
        (sequential_agent.critic, batched_agent.critic),
    ):
        for sequential_parameter, batched_parameter in zip(
            sequential_module.parameters(), batched_module.parameters()
        ):
            assert torch.allclose(
                sequential_parameter.grad, batched_parameter.grad, atol=1e-6
            )


def test_graph_gat_rollout_buffer_keeps_graph_transitions() -> None:
    """Graph-GAT rollout buffer should keep graph states and PPO metadata."""
    buffer = GraphGATRolloutBuffer()

    buffer.push(
        graph_state=_make_graph_state(),
        actions=[0, 1],
        rewards=[1.0, 0.5],
        next_graph_state=_make_graph_state(offset=0.1),
        old_log_probs=[-0.2, -0.4],
        done=True,
    )

    assert len(buffer) == 1
    transitions = buffer.as_transitions()
    assert transitions[0].actions == [0, 1]
    assert transitions[0].rewards == [1.0, 0.5]
    assert transitions[0].old_log_probs == [-0.2, -0.4]
    assert transitions[0].done is True


def test_graph_gat_mappo_update_changes_actor_critic_and_gat_parameters() -> None:
    """One PPO update should optimize actor, critic, and GAT parameters."""
    torch.manual_seed(37)
    agent = GraphGATMAPPOAgent(
        num_devices=2,
        num_servers=2,
        node_feature_dim=14,
        edge_feature_dim=7,
        embedding_dim=8,
        ppo_epochs=2,
        max_grad_norm=0.5,
    )
    buffer = GraphGATRolloutBuffer()
    for step_index in range(4):
        graph_state = _make_graph_state(offset=0.05 * step_index)
        actions, old_log_probs = agent.select_actions_with_log_probs(graph_state)
        buffer.push(
            graph_state=graph_state,
            actions=actions,
            rewards=[1.0 - 0.1 * step_index, 0.6 + 0.05 * step_index],
            next_graph_state=_make_graph_state(offset=0.05 * (step_index + 1)),
            old_log_probs=old_log_probs,
            done=step_index == 3,
        )
    encoder_before = _clone_module_parameters(agent.encoder)
    actor_before = _clone_module_parameters(agent.actor)
    critic_before = _clone_module_parameters(agent.critic)

    agent.update_from_rollout(buffer)

    assert _parameters_changed(encoder_before, _clone_module_parameters(agent.encoder))
    assert _parameters_changed(actor_before, _clone_module_parameters(agent.actor))
    assert _parameters_changed(critic_before, _clone_module_parameters(agent.critic))
    assert len(buffer) == 0


def test_graph_gat_mappo_is_registered_as_separate_comparison_model() -> None:
    """Comparison config should expose Graph-GAT MAPPO without replacing MAPPO."""
    configs = build_algorithm_configs()

    assert configs["Graph-GAT MAPPO"]["class"] is GraphGATMAPPOAgent
    assert configs["Graph-GAT Warmup MAPPO"]["class"] is GraphGATMAPPOAgent
    assert configs["Graph-GAT Warmup Mask MAPPO"]["class"] is GraphGATMAPPOAgent
    assert (
        configs["Graph-GAT Warmup MAPPO"]["kwargs"][
            "topology_warmup_episodes"
        ]
        == 5
    )
    assert (
        configs["Graph-GAT Warmup MAPPO"]["kwargs"][
            "topology_warmup_updates_per_step"
        ]
        == 10
    )
    assert configs["MAPPO"]["class"] is not GraphGATMAPPOAgent


def test_graph_gat_device_override_only_applies_to_graph_agents() -> None:
    """Comparison config should route the device override to Graph-GAT only."""
    configs = build_algorithm_configs(graph_gat_device="cpu")

    for config in configs.values():
        kwargs = config["kwargs"]
        if config["class"] is GraphGATMAPPOAgent:
            assert kwargs["device"] == "cpu"
        elif config["class"] is GATMAAgent:
            assert kwargs["device"] == "auto"
        else:
            assert "device" not in kwargs


def test_graph_gat_config_is_scoped_to_graph_variants(monkeypatch) -> None:
    """A graph setting in config must not change flat MAPPO."""
    provisional = PAPER_PARAMS["provisional_table2_needed"]
    monkeypatch.setitem(provisional, "graph_gat_entropy_coef", 0.005)

    configs = build_algorithm_configs(graph_gat_device="cpu")

    for config in configs.values():
        if config["class"] is GraphGATMAPPOAgent:
            assert config["kwargs"]["entropy_coef"] == 0.005
    assert configs["Shared MAPPO"]["kwargs"]["entropy_coef"] != 0.005


def test_comparison_builder_rejects_removed_model_override() -> None:
    """Model hyperparameters must come from config, not builder overrides."""
    with pytest.raises(TypeError):
        build_algorithm_configs(graph_gat_lr=8e-5)


def test_graph_gat_optimizer_supports_encoder_specific_learning_rate() -> None:
    """Encoder, actor, and critic should use their configured PPO rates."""
    agent = GraphGATMAPPOAgent(
        num_devices=2,
        num_servers=2,
        node_feature_dim=14,
        edge_feature_dim=7,
        embedding_dim=8,
        lr=8e-5,
        actor_lr=7e-5,
        critic_lr=9e-5,
        encoder_lr=3e-5,
        entropy_coef=0.005,
        value_loss_coef=0.5,
        max_grad_norm=0.5,
    )

    assert [group["lr"] for group in agent.optimizer.param_groups] == [
        3e-5,
        7e-5,
        9e-5,
    ]
    assert agent.entropy_coef == 0.005
    assert agent.value_loss_coef == 0.5
    assert agent.max_grad_norm == 0.5


def test_algorithm_selection_runs_only_requested_models_in_order() -> None:
    """Targeted speed tests should avoid rerunning the full ablation matrix."""
    configs = build_algorithm_configs(graph_gat_device="cpu")

    selected = select_algorithm_configs(
        configs, ["Graph-GAT MAPPO", "Local Only"]
    )

    assert list(selected) == ["Graph-GAT MAPPO", "Local Only"]
    assert selected["Graph-GAT MAPPO"] is configs["Graph-GAT MAPPO"]


def test_algorithm_selection_rejects_unknown_name() -> None:
    """A typo in a targeted benchmark must fail before training starts."""
    with pytest.raises(ValueError, match="unknown algorithm"):
        select_algorithm_configs(build_algorithm_configs(), ["Graph GAT"])


def test_collect_graph_gat_actions_builds_graph_from_joint_state() -> None:
    """Comparison loop should collect Graph-GAT actions from topology graph state."""
    torch.manual_seed(41)
    agent = GraphGATMAPPOAgent(
        num_devices=2,
        num_servers=2,
        node_feature_dim=14,
        edge_feature_dim=7,
        embedding_dim=8,
    )

    (
        actions,
        local_count,
        edge_count,
        old_log_probs,
        graph_state,
        graph_build_time,
        graph_warmup_time,
        graph_warmup_loss,
        graph_action_time,
    ) = (
        _collect_graph_gat_actions(
            agent=agent,
            joint_state=_make_joint_state(),
            num_devices=2,
            num_servers=2,
            episode_index=0,
        )
    )

    assert len(actions) == 2
    assert len(old_log_probs) == 2
    assert local_count + edge_count == 2
    assert graph_state.node_features.shape == (4, 14)
    assert all(0 <= action < agent.action_dim for action in actions)
    assert graph_build_time >= 0.0
    assert graph_warmup_time == 0.0
    assert graph_warmup_loss == 0.0
    assert graph_action_time >= 0.0


def test_graph_gat_rollout_update_helper_updates_agent_and_clears_buffer() -> None:
    """Comparison loop helper should train Graph-GAT MAPPO from graph rollouts."""
    torch.manual_seed(43)
    agent = GraphGATMAPPOAgent(
        num_devices=2,
        num_servers=2,
        node_feature_dim=14,
        edge_feature_dim=7,
        embedding_dim=8,
        ppo_epochs=2,
    )
    buffer = GraphGATRolloutBuffer()
    for step_index in range(4):
        graph_state = _make_graph_state(offset=0.05 * step_index)
        actions, old_log_probs = agent.select_actions_with_log_probs(graph_state)
        buffer.push(
            graph_state=graph_state,
            actions=actions,
            rewards=[1.0 - 0.1 * step_index, 0.6 + 0.05 * step_index],
            next_graph_state=_make_graph_state(offset=0.05 * (step_index + 1)),
            old_log_probs=old_log_probs,
            done=step_index == 3,
        )
    actor_before = _clone_module_parameters(agent.actor)

    update_time = _update_graph_gat_mappo_from_rollout(agent, buffer, gamma=0.95)

    assert _parameters_changed(actor_before, _clone_module_parameters(agent.actor))
    assert len(buffer) == 0
    assert update_time >= 0.0


def test_graph_gat_mappo_batches_rollout_encoder_calls() -> None:
    """PPO should use one local/global encoder call per rollout evaluation."""
    agent = GraphGATMAPPOAgent(
        num_devices=2,
        num_servers=2,
        node_feature_dim=14,
        edge_feature_dim=7,
        embedding_dim=8,
        ppo_epochs=2,
    )
    buffer = GraphGATRolloutBuffer()
    transition_count = 4
    for step_index in range(transition_count):
        buffer.push(
            graph_state=_make_graph_state(offset=0.05 * step_index),
            actions=[0, 1],
            rewards=[1.0 - 0.1 * step_index, 0.6 + 0.05 * step_index],
            next_graph_state=_make_graph_state(offset=0.05 * (step_index + 1)),
            old_log_probs=[-0.5, -0.5],
            done=step_index == transition_count - 1,
        )
    original_global_rollout = agent.encoder.forward_batched_global_rollout
    original_local_rollout = (
        agent.encoder.forward_batched_local_rollout_nodes
    )
    global_encode_call_count = 0
    local_batch_call_count = 0

    def counted_global_rollout(*args, **kwargs):
        nonlocal global_encode_call_count
        global_encode_call_count += 1
        return original_global_rollout(*args, **kwargs)

    def counted_local_rollout(*args, **kwargs):
        nonlocal local_batch_call_count
        local_batch_call_count += 1
        return original_local_rollout(*args, **kwargs)

    agent.encoder.forward_batched_global_rollout = counted_global_rollout
    agent.encoder.forward_batched_local_rollout_nodes = counted_local_rollout

    agent.update_from_rollout(buffer)

    expected_global_calls = 2 + agent.ppo_epochs
    expected_local_batch_calls = agent.ppo_epochs
    assert global_encode_call_count == expected_global_calls
    assert local_batch_call_count == expected_local_batch_calls
