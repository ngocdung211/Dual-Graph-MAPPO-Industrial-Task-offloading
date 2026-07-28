"""Tests for simple offloading baseline policies."""

import pathlib
import random
import sys
from types import SimpleNamespace

import numpy as np

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from baselines.offloading_baselines import (
    EdgeOnlyAgent,
    FeatureExtractionEdgeAgent,
    LocalOnlyAgent,
    RandomOffloadingAgent,
)
from run_comparision import _collect_joint_actions, build_algorithm_configs


def test_local_only_agent_always_selects_local_action() -> None:
    """Local-only baseline should always select action 0."""
    agent = LocalOnlyAgent(state_dim=17, action_dim=4, num_agents=10)

    assert agent.select_action(np.zeros(17, dtype=np.float32)) == 0


def test_random_offloading_agent_returns_valid_action() -> None:
    """Random baseline should sample within the discrete action space."""
    random.seed(3)
    agent = RandomOffloadingAgent(state_dim=17, action_dim=4, num_agents=10)

    actions = {agent.select_action(np.zeros(17, dtype=np.float32)) for _ in range(20)}

    assert actions
    assert all(0 <= action < 4 for action in actions)


def test_edge_only_agent_chooses_first_connected_edge() -> None:
    """Edge-only baseline should prefer the first edge with l_end > l_start."""
    agent = EdgeOnlyAgent(state_dim=17, action_dim=4, num_agents=10)
    state = np.zeros(17, dtype=np.float32)
    state[2] = 0.2
    state[5:8] = 2.0
    # State tail encodes three l_start values then three l_end values.
    state[-6:] = np.array([1.0, 0.2, 0.4, 1.0, 0.8, 0.9], dtype=np.float32)

    assert agent.select_action(state) == 2


def test_edge_only_agent_uses_local_when_no_edge_is_feasible() -> None:
    """Edge-only baseline should not request an impossible edge action."""
    agent = EdgeOnlyAgent(state_dim=17, action_dim=4, num_agents=10)

    assert agent.select_action(np.zeros(17, dtype=np.float32)) == 0


def test_feature_extraction_edge_agent_uses_edge_only_for_subtask_four() -> None:
    """Feature-extraction-only edge baseline should offload only subtask 4."""
    agent = FeatureExtractionEdgeAgent(state_dim=17, action_dim=4, num_agents=10)
    state = np.zeros(17, dtype=np.float32)
    state[2] = 0.2
    state[5:8] = 2.0
    state[-6:] = np.array([1.0, 0.2, 0.4, 1.0, 0.8, 0.9], dtype=np.float32)

    assert agent.select_action_for_subtask(state, subtask_id=1) == 0
    assert agent.select_action_for_subtask(state, subtask_id=4) == 2


def test_collect_joint_actions_passes_current_subtask_to_feature_baseline() -> None:
    """Comparison loop should provide current subtask id to the feature baseline."""
    agent = FeatureExtractionEdgeAgent(state_dim=17, action_dim=4, num_agents=1)
    joint_state = np.zeros((1, 17), dtype=np.float32)
    joint_state[0, 2] = 0.2
    joint_state[0, 5:8] = 2.0
    joint_state[0, -6:] = np.array([1.0, 0.2, 0.4, 1.0, 0.8, 0.9], dtype=np.float32)
    env = SimpleNamespace(
        devices=[SimpleNamespace(id=1)],
        current_step={1: 1},
        priorities={1: [1, 4]},
    )

    joint_actions, local_count, edge_count = _collect_joint_actions([agent], joint_state, env)

    assert joint_actions == [2]
    assert local_count == 0
    assert edge_count == 1


def test_comparison_config_includes_simple_offloading_baselines() -> None:
    """Comparison setup should expose requested simple baselines."""
    configs = build_algorithm_configs()

    assert configs["Local Only"]["class"] is LocalOnlyAgent
    assert configs["Edge Only"]["class"] is EdgeOnlyAgent
    assert configs["Random Offloading"]["class"] is RandomOffloadingAgent
    assert configs["Feature Extraction Edge"]["class"] is FeatureExtractionEdgeAgent
