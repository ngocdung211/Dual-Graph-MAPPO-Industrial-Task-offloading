"""Communication and computation models for the industrial network."""

import math
from typing import Tuple

import numpy as np

class NetworkEnvironment:
    """Handle the communication and computation mathematical models."""

    def __init__(self, bandwidth: float, noise_power_dbm: float):
        """Initialize network parameters.

        Args:
            bandwidth: Subchannel bandwidth (Hz).
            noise_power_dbm: Background noise power in dBm.
        """
        # Convert background noise from dBm to linear scale (Watts).
        self.bandwidth_hz = bandwidth
        self.noise_power_w = (10 ** (noise_power_dbm / 10)) / 1000

    def calculate_channel_gain(self, loc1: np.ndarray, loc2: np.ndarray) -> float:
        """Calculate path-loss based channel gain between two locations.

        Args:
            loc1: First location (x, y).
            loc2: Second location (x, y).

        Returns:
            Channel gain value.
        """
        distance = np.linalg.norm(loc1 - loc2)
        # Using a standard simplified path loss model; adjust exponent as needed
        return distance ** (-2) if distance > 0 else 1.0

    # --- COMMUNICATION MODEL ---
    
    def get_uplink_rate(self, transmit_power: float, channel_gain: float) -> float:
        """Calculate uplink data transmission rate.

        Args:
            transmit_power: Device transmit power (W).
            channel_gain: Channel gain between device and server.

        Returns:
            Uplink data rate (bps).
        """
        sinr = (transmit_power * channel_gain) / self.noise_power_w
        return self.bandwidth_hz * math.log2(1 + sinr)
        
    def get_downlink_rate(self, transmit_power: float, channel_gain: float) -> float:
        """Calculate downlink data transmission rate.

        Args:
            transmit_power: Server transmit power (W).
            channel_gain: Channel gain between server and device.

        Returns:
            Downlink data rate (bps).
        """
        sinr = (transmit_power * channel_gain) / self.noise_power_w
        return self.bandwidth_hz * math.log2(1 + sinr)

    # --- COMPUTATION MODEL ---

    @staticmethod
    def _realized_computation(
        cpu_cycles: float, f_est: float, f_actual: float
    ) -> Tuple[float, float]:
        """Return realized delay and reconstructed physical compute power."""
        estimated_power = max(f_est, 1e-9)
        compute_deviation = estimated_power - f_actual
        physical_power = max(1e-9, estimated_power - compute_deviation)
        estimated_delay = cpu_cycles / estimated_power
        delay_deviation = (
            cpu_cycles * compute_deviation / (estimated_power * physical_power)
        )
        return estimated_delay + delay_deviation, physical_power

    def calculate_local_computation(
        self, cpu_cycles: float, energy_coeff: float, f_est: float, f_actual: float
    ) -> Tuple[float, float]:
        """Calculate delay and energy for local computing.

        Args:
            cpu_cycles: CPU cycles required by the subtask.
            energy_coeff: Local computing energy coefficient.
            f_est: Estimated computing power.
            f_actual: Actual computing power.

        Returns:
            Tuple of (actual_delay, energy_consumption).
        """
        actual_delay, physical_power = self._realized_computation(
            cpu_cycles, f_est, f_actual
        )
        energy_consumption = energy_coeff * cpu_cycles * (physical_power ** 2)

        return actual_delay, energy_consumption

    def calculate_edge_computation(
        self, cpu_cycles: float, energy_coeff: float, f_est: float, f_actual: float
    ) -> Tuple[float, float]:
        """Calculate delay and energy for edge computing.

        Args:
            cpu_cycles: CPU cycles required by the subtask.
            energy_coeff: Edge computing energy coefficient.
            f_est: Estimated computing power.
            f_actual: Actual computing power.

        Returns:
            Tuple of (actual_delay, energy_consumption).
        """
        actual_delay, _ = self._realized_computation(
            cpu_cycles, f_est, f_actual
        )
        # Temporarily exclude edge-server computation energy from the
        # device-side offloading objective. Transmission energy is accounted
        # for separately by DITENEnv.
        del energy_coeff
        energy_consumption = 0.0
        return actual_delay, energy_consumption
