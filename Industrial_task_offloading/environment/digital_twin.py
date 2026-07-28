"""Minimal digital-twin state for compute-capability observations."""

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np

from environment.system_model import EdgeServer, IndustrialDevice


@dataclass(frozen=True)
class DeviceTwinState:
    """Estimated compute state for one industrial device."""

    device_id: int
    estimated_compute_power: float


@dataclass(frozen=True)
class ServerTwinState:
    """Estimated compute state for one edge server."""

    server_id: int
    estimated_compute_power: float


@dataclass(frozen=True)
class DigitalTwinSnapshot:
    """Compute-capability snapshot synchronized for one time slot."""

    time_slot_index: int
    device_states: Dict[int, DeviceTwinState]
    server_states: Dict[int, ServerTwinState]


class DigitalTwin:
    """Create simple noisy CPU estimates from the physical network."""

    def __init__(
        self,
        local_estimation_error: float = 0.0,
        edge_estimation_error: float = 0.0,
        random_generator: Optional[np.random.Generator] = None,
    ) -> None:
        """Initialize CPU-estimation error ranges.

        Args:
            local_estimation_error: Maximum relative device CPU error.
            edge_estimation_error: Maximum relative server CPU error.
            random_generator: Optional NumPy generator for isolated testing.
        """
        self.local_estimation_error = max(0.0, local_estimation_error)
        self.edge_estimation_error = max(0.0, edge_estimation_error)
        self.random_generator = random_generator
        self.latest_snapshot: Optional[DigitalTwinSnapshot] = None

    def synchronize(
        self,
        devices: Sequence[IndustrialDevice],
        servers: Sequence[EdgeServer],
        time_slot_index: int,
    ) -> DigitalTwinSnapshot:
        """Synchronize one twin snapshot from current physical entities."""
        device_states = {
            device.id: DeviceTwinState(
                device_id=device.id,
                estimated_compute_power=self._sample_estimated_power(
                    device.compute_power, self.local_estimation_error
                ),
            )
            for device in devices
        }
        server_states = {
            server.id: ServerTwinState(
                server_id=server.id,
                estimated_compute_power=self._sample_estimated_power(
                    server.compute_power, self.edge_estimation_error
                ),
            )
            for server in servers
        }
        self.latest_snapshot = DigitalTwinSnapshot(
            time_slot_index=time_slot_index,
            device_states=device_states,
            server_states=server_states,
        )
        return self.latest_snapshot

    def reset(self) -> None:
        """Discard the previous episode's snapshot."""
        self.latest_snapshot = None

    def _sample_estimated_power(
        self, actual_power: float, error_ratio: float
    ) -> float:
        """Sample an estimate within the configured relative error range."""
        if error_ratio == 0.0:
            return actual_power
        if self.random_generator is None:
            noise = np.random.uniform(-error_ratio, error_ratio)
        else:
            noise = self.random_generator.uniform(-error_ratio, error_ratio)
        return max(1e-9, actual_power * (1.0 + noise))
