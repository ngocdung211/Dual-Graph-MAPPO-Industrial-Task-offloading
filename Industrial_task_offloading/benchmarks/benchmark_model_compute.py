"""Benchmark model size and inference cost for the large topology."""

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Callable, Dict, Iterable, List

import numpy as np
import torch
from torch.profiler import ProfilerActivity, profile

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from baselines.graph_gat_mappo import GraphGATMAPPOAgent
from baselines.gatma import GATMAAgent
from baselines.mappo import MAPPOAgent
from models.maddpg import EpsilonATNMADDPGAgent
from utils.topology_graph_state import (
    LIGHTWEIGHT_TOPOLOGY_EDGE_FEATURE_DIM,
    STANDARD_TOPOLOGY_EDGE_FEATURE_DIM,
    build_topology_graph_state,
)


NUM_DEVICES = 30
NUM_SERVERS = 9
NUM_SUBTASKS = 5
STATE_DIM = 5 + NUM_SUBTASKS + 4 * NUM_SERVERS
ACTION_DIM = NUM_SERVERS + 1
NODE_FEATURE_DIM = 9 + NUM_SUBTASKS
EDGE_FEATURE_DIM = STANDARD_TOPOLOGY_EDGE_FEATURE_DIM


def _parameter_count(parameters: Iterable[torch.nn.Parameter]) -> int:
    """Count unique scalar parameters."""
    unique_parameters = {id(parameter): parameter for parameter in parameters}
    return sum(parameter.numel() for parameter in unique_parameters.values())


def _profile_flops(function: Callable[[], object]) -> int:
    """Return PyTorch-profiler FLOPs for one inference call."""
    with torch.inference_mode(), profile(
        activities=[ProfilerActivity.CPU], with_flops=True
    ) as profiler:
        function()
    return int(sum(event.flops for event in profiler.key_averages()))


def _latency_samples_ms(
    function: Callable[[], object], warmup: int, repeats: int, trials: int
) -> List[float]:
    """Measure mean milliseconds per call over independent trials."""
    with torch.inference_mode():
        for _ in range(warmup):
            function()

        samples = []
        for _ in range(trials):
            start_time = time.perf_counter_ns()
            for _ in range(repeats):
                function()
            elapsed_ns = time.perf_counter_ns() - start_time
            samples.append(elapsed_ns / repeats / 1_000_000)
    return samples


def _latency_summary(samples: List[float]) -> Dict[str, float]:
    """Summarize independent latency trials."""
    return {
        "median_ms": statistics.median(samples),
        "mean_ms": statistics.mean(samples),
        "sample_sd_ms": statistics.stdev(samples) if len(samples) > 1 else 0.0,
    }


def benchmark(warmup: int, repeats: int, trials: int) -> Dict[str, object]:
    """Benchmark all learned methods used in the large-topology comparison."""
    torch.manual_seed(75)
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    joint_state = torch.randn(NUM_DEVICES, STATE_DIM)
    joint_state_array = joint_state.numpy()
    graph_state = build_topology_graph_state(
        joint_state, num_devices=NUM_DEVICES, num_servers=NUM_SERVERS
    )
    lightweight_graph_state = build_topology_graph_state(
        joint_state,
        num_devices=NUM_DEVICES,
        num_servers=NUM_SERVERS,
        lightweight=True,
    )

    maddpg_agents = [
        EpsilonATNMADDPGAgent(
            state_dim=STATE_DIM,
            action_dim=ACTION_DIM,
            num_agents=NUM_DEVICES,
            hidden_dim=128,
        )
        for _ in range(NUM_DEVICES)
    ]
    mappo_agents = [
        MAPPOAgent(
            state_dim=STATE_DIM,
            action_dim=ACTION_DIM,
            num_agents=NUM_DEVICES,
            hidden_dim=128,
        )
        for _ in range(NUM_DEVICES)
    ]
    gatma_agents = [
        GATMAAgent(
            state_dim=STATE_DIM,
            action_dim=ACTION_DIM,
            num_agents=NUM_DEVICES,
            num_servers=NUM_SERVERS,
            agent_index=agent_index,
        )
        for agent_index in range(NUM_DEVICES)
    ]
    graph_agent = GraphGATMAPPOAgent(
        num_devices=NUM_DEVICES,
        num_servers=NUM_SERVERS,
        node_feature_dim=NODE_FEATURE_DIM,
        edge_feature_dim=EDGE_FEATURE_DIM,
        embedding_dim=64,
        hidden_dim=64,
        device="cpu",
    )
    lightweight_graph_agent = GraphGATMAPPOAgent(
        num_devices=NUM_DEVICES,
        num_servers=NUM_SERVERS,
        node_feature_dim=NODE_FEATURE_DIM,
        edge_feature_dim=LIGHTWEIGHT_TOPOLOGY_EDGE_FEATURE_DIM,
        embedding_dim=64,
        hidden_dim=64,
        lightweight_topology=True,
        device="cpu",
    )

    for agent in maddpg_agents:
        agent.actor.eval()
        agent.critic.eval()
    for agent in mappo_agents:
        agent.actor.eval()
        agent.critic.eval()
    for agent in gatma_agents:
        agent.actor.eval()
        agent.critic.eval()
    graph_agent.encoder.eval()
    graph_agent.actor.eval()
    graph_agent.critic.eval()
    lightweight_graph_agent.encoder.eval()
    lightweight_graph_agent.actor.eval()
    lightweight_graph_agent.critic.eval()

    def maddpg_joint_inference() -> List[int]:
        return [
            int(torch.argmax(agent.actor(joint_state[index])).item())
            for index, agent in enumerate(maddpg_agents)
        ]

    def mappo_joint_inference() -> List[int]:
        return [
            int(torch.argmax(agent.actor(joint_state[index])).item())
            for index, agent in enumerate(mappo_agents)
        ]

    def maddpg_end_to_end_inference() -> List[int]:
        return [
            int(
                torch.argmax(
                    agent.actor(torch.as_tensor(joint_state_array[index]))
                ).item()
            )
            for index, agent in enumerate(maddpg_agents)
        ]

    def mappo_end_to_end_inference() -> List[int]:
        return [
            int(
                torch.argmax(
                    agent.actor(torch.as_tensor(joint_state_array[index]))
                ).item()
            )
            for index, agent in enumerate(mappo_agents)
        ]

    def gatma_joint_inference() -> List[int]:
        return [
            int(torch.argmax(agent.actor(joint_state)).item())
            for agent in gatma_agents
        ]

    def gatma_end_to_end_inference() -> List[int]:
        current_joint_state = torch.as_tensor(joint_state_array)
        return [
            int(torch.argmax(agent.actor(current_joint_state)).item())
            for agent in gatma_agents
        ]

    def graph_joint_inference() -> List[int]:
        probabilities = graph_agent._actor_probabilities_for_graph_state(graph_state)
        return torch.argmax(probabilities, dim=-1).tolist()

    def graph_end_to_end_inference() -> List[int]:
        current_graph = build_topology_graph_state(
            joint_state_array, num_devices=NUM_DEVICES, num_servers=NUM_SERVERS
        )
        probabilities = graph_agent._actor_probabilities_for_graph_state(current_graph)
        return torch.argmax(probabilities, dim=-1).tolist()

    def lightweight_graph_joint_inference() -> List[int]:
        probabilities = (
            lightweight_graph_agent._actor_probabilities_for_graph_state(
                lightweight_graph_state
            )
        )
        return torch.argmax(probabilities, dim=-1).tolist()

    def lightweight_graph_end_to_end_inference() -> List[int]:
        current_graph = build_topology_graph_state(
            joint_state_array,
            num_devices=NUM_DEVICES,
            num_servers=NUM_SERVERS,
            lightweight=True,
        )
        probabilities = (
            lightweight_graph_agent._actor_probabilities_for_graph_state(
                current_graph
            )
        )
        return torch.argmax(probabilities, dim=-1).tolist()

    maddpg_optimized_parameters = [
        parameter
        for agent in maddpg_agents
        for module in (agent.actor, agent.critic)
        for parameter in module.parameters()
    ]
    mappo_optimized_parameters = [
        parameter
        for agent in mappo_agents
        for module in (agent.actor, agent.critic)
        for parameter in module.parameters()
    ]
    gatma_optimized_parameters = [
        parameter
        for agent in gatma_agents
        for module in (agent.actor, agent.critic)
        for parameter in module.parameters()
    ]
    graph_ppo_parameters = list(graph_agent.ppo_parameters)
    graph_warmup_parameters = list(graph_agent.topology_warmup_head.parameters())
    lightweight_graph_ppo_parameters = list(
        lightweight_graph_agent.ppo_parameters
    )
    lightweight_graph_warmup_parameters = list(
        lightweight_graph_agent.topology_warmup_head.parameters()
    )

    maddpg_flops = _profile_flops(maddpg_joint_inference)
    mappo_flops = _profile_flops(mappo_joint_inference)
    gatma_flops = _profile_flops(gatma_joint_inference)
    graph_flops = _profile_flops(graph_joint_inference)
    lightweight_graph_flops = _profile_flops(lightweight_graph_joint_inference)

    maddpg_latency = _latency_summary(
        _latency_samples_ms(maddpg_joint_inference, warmup, repeats, trials)
    )
    mappo_latency = _latency_summary(
        _latency_samples_ms(mappo_joint_inference, warmup, repeats, trials)
    )
    maddpg_end_to_end_latency = _latency_summary(
        _latency_samples_ms(
            maddpg_end_to_end_inference, warmup, repeats, trials
        )
    )
    mappo_end_to_end_latency = _latency_summary(
        _latency_samples_ms(mappo_end_to_end_inference, warmup, repeats, trials)
    )
    gatma_latency = _latency_summary(
        _latency_samples_ms(gatma_joint_inference, warmup, repeats, trials)
    )
    gatma_end_to_end_latency = _latency_summary(
        _latency_samples_ms(
            gatma_end_to_end_inference, warmup, repeats, trials
        )
    )
    graph_latency = _latency_summary(
        _latency_samples_ms(graph_joint_inference, warmup, repeats, trials)
    )
    graph_end_to_end_latency = _latency_summary(
        _latency_samples_ms(graph_end_to_end_inference, warmup, repeats, trials)
    )
    lightweight_graph_latency = _latency_summary(
        _latency_samples_ms(
            lightweight_graph_joint_inference, warmup, repeats, trials
        )
    )
    lightweight_graph_end_to_end_latency = _latency_summary(
        _latency_samples_ms(
            lightweight_graph_end_to_end_inference,
            warmup,
            repeats,
            trials,
        )
    )

    graph_common = {
        "ppo_trainable_parameters": _parameter_count(graph_ppo_parameters),
        "deployment_parameters": _parameter_count(
            list(graph_agent.encoder.parameters())
            + list(graph_agent.actor.parameters())
        ),
        "inference_flops": graph_flops,
        "neural_inference_latency": graph_latency,
        "end_to_end_policy_latency": graph_end_to_end_latency,
    }
    lightweight_graph_common = {
        "ppo_trainable_parameters": _parameter_count(
            lightweight_graph_ppo_parameters
        ),
        "deployment_parameters": _parameter_count(
            list(lightweight_graph_agent.encoder.parameters())
            + list(lightweight_graph_agent.actor.parameters())
        ),
        "inference_flops": lightweight_graph_flops,
        "neural_inference_latency": lightweight_graph_latency,
        "end_to_end_policy_latency": lightweight_graph_end_to_end_latency,
    }
    return {
        "metadata": {
            "device": "cpu",
            "processor": platform.processor(),
            "platform": platform.platform(),
            "torch_version": torch.__version__,
            "torch_threads": torch.get_num_threads(),
            "num_devices": NUM_DEVICES,
            "num_servers": NUM_SERVERS,
            "state_dim": STATE_DIM,
            "warmup_calls": warmup,
            "calls_per_trial": repeats,
            "trials": trials,
            "flop_definition": (
                "PyTorch profiler estimate; multiply-add counts as two FLOPs; "
                "unsupported elementwise operations may be omitted"
            ),
        },
        "models": {
            "e-ATN-MADDPG": {
                "optimized_trainable_parameters": _parameter_count(
                    maddpg_optimized_parameters
                ),
                "deployment_parameters": _parameter_count(
                    parameter
                    for agent in maddpg_agents
                    for parameter in agent.actor.parameters()
                ),
                "inference_flops": maddpg_flops,
                "neural_inference_latency": maddpg_latency,
                "end_to_end_policy_latency": maddpg_end_to_end_latency,
            },
            "MAPPO": {
                "optimized_trainable_parameters": _parameter_count(
                    mappo_optimized_parameters
                ),
                "deployment_parameters": _parameter_count(
                    parameter
                    for agent in mappo_agents
                    for parameter in agent.actor.parameters()
                ),
                "inference_flops": mappo_flops,
                "neural_inference_latency": mappo_latency,
                "end_to_end_policy_latency": mappo_end_to_end_latency,
            },
            "GATMA": {
                "optimized_trainable_parameters": _parameter_count(
                    gatma_optimized_parameters
                ),
                "deployment_parameters": _parameter_count(
                    parameter
                    for agent in gatma_agents
                    for parameter in agent.actor.parameters()
                ),
                "target_network_parameters": _parameter_count(
                    parameter
                    for agent in gatma_agents
                    for module in (agent.target_actor, agent.target_critic)
                    for parameter in module.parameters()
                ),
                "inference_flops": gatma_flops,
                "neural_inference_latency": gatma_latency,
                "end_to_end_policy_latency": gatma_end_to_end_latency,
            },
            "Dual-GAT MAPPO w/o warmup": {
                **graph_common,
                "auxiliary_trainable_parameters": 0,
                "total_trainable_parameters": _parameter_count(
                    graph_ppo_parameters
                ),
            },
            "Dual-GAT MAPPO": {
                **graph_common,
                "auxiliary_trainable_parameters": _parameter_count(
                    graph_warmup_parameters
                ),
                "total_trainable_parameters": _parameter_count(
                    graph_ppo_parameters + graph_warmup_parameters
                ),
            },
            "Lightweight Dual-GAT MAPPO": {
                **lightweight_graph_common,
                "auxiliary_trainable_parameters": _parameter_count(
                    lightweight_graph_warmup_parameters
                ),
                "total_trainable_parameters": _parameter_count(
                    lightweight_graph_ppo_parameters
                    + lightweight_graph_warmup_parameters
                ),
            },
        },
    }


def main() -> None:
    """Run the benchmark and write machine-readable results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=1000)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/analysis/large_multiseed/model_compute.json"),
    )
    args = parser.parse_args()

    results = benchmark(args.warmup, args.repeats, args.trials)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
