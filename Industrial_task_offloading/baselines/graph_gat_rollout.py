"""On-policy graph transitions and rollout storage for Graph-GAT MAPPO."""

from dataclasses import dataclass
from typing import List

from utils.topology.graph_state import TopologyGraphState


@dataclass(frozen=True)
class GraphGATTransition:
    """One on-policy Graph-GAT MAPPO transition."""

    graph_state: TopologyGraphState
    actions: List[int]
    rewards: List[float]
    next_graph_state: TopologyGraphState
    old_log_probs: List[float]
    done: bool


class GraphGATRolloutBuffer:
    """Store on-policy graph transitions for Graph-GAT MAPPO."""

    def __init__(self):
        """Initialize an empty graph rollout buffer."""
        self.transitions: List[GraphGATTransition] = []

    def push(
        self,
        graph_state: TopologyGraphState,
        actions: List[int],
        rewards: List[float],
        next_graph_state: TopologyGraphState,
        old_log_probs: List[float],
        done: bool,
    ) -> None:
        """Store one graph transition."""
        self.transitions.append(
            GraphGATTransition(
                graph_state=graph_state,
                actions=list(actions),
                rewards=list(rewards),
                next_graph_state=next_graph_state,
                old_log_probs=list(old_log_probs),
                done=bool(done),
            )
        )

    def as_transitions(self) -> List[GraphGATTransition]:
        """Return stored transitions in insertion order."""
        return list(self.transitions)

    def clear(self) -> None:
        """Remove all stored transitions."""
        self.transitions.clear()

    def __len__(self) -> int:
        """Return number of stored graph transitions."""
        return len(self.transitions)
