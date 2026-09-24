"""Digital twin environment for industrial task offloading."""

import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from environment.digital_twin import DigitalTwin, DigitalTwinSnapshot
from environment.network_env import NetworkEnvironment
from environment.system_model import EdgeServer, IndustrialDevice, Subtask, TaskDAG


class DITENEnv:
    """Digital twin industrial task offloading environment."""

    def __init__(
        self,
        devices: List[IndustrialDevice],
        servers: List[EdgeServer],
        network_env: NetworkEnvironment,
        slot_duration: float = 1.0,
        subslot_count: int = 100,
        lambda1: float = 3.0,
        lambda2: float = 3.0,
        lambda3: float = 3.0,
        lambda4: float = 3.0,
        lambda5: float = 0.5,
        p_out_value: float = -1.0,
        local_estimation_error: float = 0.05,
        edge_estimation_error: float = 0.05,
        time_slots: int = 100,
        strict_connection_window: bool = True,
        route_rectangles: Optional[Sequence[Sequence[Sequence[float]]]] = None,
        world_size: Sequence[float] = (100.0, 100.0),
    ):
        """Initialize the DITEN environment.

        Args:
            devices: Industrial devices participating in the episode.
            servers: Edge servers available for offloading.
            network_env: Communication and computation model.
            slot_duration: Duration of one time slot (seconds).
            subslot_count: Number of subslots used to detect coverage windows.
            lambda1: Reward coefficient for immediate delay.
            lambda2: Reward coefficient for accumulated delay.
            lambda3: Reward coefficient for immediate energy.
            lambda4: Reward coefficient for accumulated energy.
            lambda5: Reward coefficient for outage penalty.
            p_out_value: Penalty value applied on connection window violations.
            local_estimation_error: Relative error for local power estimation.
            edge_estimation_error: Relative error for edge power estimation.
            time_slots: Number of time slots per episode.
            strict_connection_window: Enforce strict coverage window constraint.
            route_rectangles: Optional custom rectangular mobility routes.
            world_size: Physical map width and height in meters.
        """
        self.devices: List[IndustrialDevice] = devices
        self.servers: List[EdgeServer] = servers
        self.network_env: NetworkEnvironment = network_env
        self.num_agents: int = len(devices)

        self.slot_duration: float = slot_duration
        self.subslot_count: int = subslot_count
        self.time_slots: int = max(1, int(time_slots))

        self.lambda1: float = lambda1
        self.lambda2: float = lambda2
        self.lambda3: float = lambda3
        self.lambda4: float = lambda4
        self.lambda5: float = lambda5
        self.p_out_value: float = p_out_value
        self.local_estimation_error: float = max(0.0, local_estimation_error)
        self.edge_estimation_error: float = max(0.0, edge_estimation_error)
        self.digital_twin = DigitalTwin(
            local_estimation_error=self.local_estimation_error,
            edge_estimation_error=self.edge_estimation_error,
        )
        self.digital_twin_snapshot: Optional[DigitalTwinSnapshot] = None
        self.strict_connection_window: bool = bool(strict_connection_window)

        self.device_accumulated_delay: Dict[int, float] = {}
        self.device_accumulated_energy: Dict[int, float] = {}
        self.slot_accumulated_delay: Dict[int, float] = {}
        self.slot_accumulated_energy: Dict[int, float] = {}
        self.subtask_finish_times: Dict[int, Dict[int, float]] = {}
        self.subtask_locations: Dict[int, Dict[int, int]] = {}
        self.local_finish_time: Dict[int, float] = {}
        self.server_finish_time: Dict[int, float] = {}

        self.task_dags: Dict[int, TaskDAG] = {}
        self.priorities: Dict[int, List[int]] = {}
        self.current_step: Dict[int, int] = {d.id: 0 for d in self.devices}
        self.current_slot_index: int = 0
        self.current_slot: float = 0.0
        self.connection_windows: Dict[Tuple[int, int], Tuple[float, float]] = {}
        self.last_step_metrics: List[Dict[str, float]] = []
        self.device_estimated_power: Dict[int, float] = {}
        self.server_estimated_power: Dict[int, float] = {}
        self.device_waypoints: Dict[int, List[np.ndarray]] = {}
        self.device_waypoint_idx: Dict[int, int] = {}
        self.route_rectangles = route_rectangles
        resolved_world_size = np.asarray(world_size, dtype=float)
        if resolved_world_size.shape != (2,) or np.any(resolved_world_size <= 0.0):
            raise ValueError("world_size must contain two positive dimensions")
        self.world_min: np.ndarray = np.array([0.0, 0.0], dtype=float)
        self.world_max: np.ndarray = resolved_world_size
        self._connection_window_cache_key: Optional[Tuple[object, ...]] = None
        subslot_duration = self.slot_duration / float(self.subslot_count)
        self._connection_window_time_offsets = (
            np.arange(self.subslot_count + 1, dtype=float) * subslot_duration
        )
        self.runtime_metrics: Dict[str, float] = {}
        self._reset_runtime_metrics()
        self._build_predetermined_paths()

    def reset_episode(self) -> None:
        """Reset episode state, device positions, and connection windows."""
        self._reset_runtime_metrics()
        self._connection_window_cache_key = None
        self.current_step = {d.id: 0 for d in self.devices}
        self.current_slot_index = 0
        self.current_slot = 0.0

        self.device_accumulated_delay = {d.id: 0.0 for d in self.devices}
        self.device_accumulated_energy = {d.id: 0.0 for d in self.devices}
        self.slot_accumulated_delay = {d.id: 0.0 for d in self.devices}
        self.slot_accumulated_energy = {d.id: 0.0 for d in self.devices}
        self.task_dags = {}
        self.priorities = {}
        self.subtask_finish_times = {}
        self.subtask_locations = {}
        self.local_finish_time = {}
        self.server_finish_time = {}
        self.last_step_metrics = []
        self.digital_twin.reset()
        self.digital_twin_snapshot = None

        for device in self.devices:
            waypoint_index = self.device_waypoint_idx[device.id]
            waypoints = self.device_waypoints[device.id]
            device.update_location(waypoints[waypoint_index].copy())
            next_waypoint = waypoints[(waypoint_index + 1) % (len(waypoints) - 1)]
            direction = next_waypoint - waypoints[waypoint_index]
            device.set_direction(direction)

        self._update_connection_windows()

    def _reset_runtime_metrics(self) -> None:
        """Reset cumulative wall-clock counters for one episode."""
        self.runtime_metrics = {
            "connection_window_seconds": 0.0,
            "connection_window_requests": 0.0,
            "connection_window_updates": 0.0,
            "connection_window_samples": 0.0,
            "joint_state_seconds": 0.0,
            "joint_state_calls": 0.0,
        }

    def get_runtime_metrics(self) -> Dict[str, float]:
        """Return a copy of cumulative environment wall-clock counters."""
        return dict(self.runtime_metrics)

    def start_time_slot(
        self, task_dags: Dict[int, TaskDAG], priorities: Dict[int, List[int]]
    ) -> np.ndarray:
        """Initialize a new time slot with task DAGs and priority orders.

        Args:
            task_dags: Mapping of device IDs to TaskDAG instances.
            priorities: Mapping of device IDs to ordered subtask IDs.

        Returns:
            Joint state array for all devices.
        """
        self._validate_priority_orders(task_dags, priorities)
        self.task_dags = task_dags
        self.priorities = priorities
        self.current_step = {d.id: 0 for d in self.devices}

        self.subtask_finish_times = {d.id: {} for d in self.devices}
        self.subtask_locations = {d.id: {} for d in self.devices}
        # self.local_finish_time = {d.id: 0.0 for d in self.devices}
        # self.server_finish_time = {s.id: 0.0 for s in self.servers}
        self.slot_accumulated_delay = {d.id: 0.0 for d in self.devices}
        self.slot_accumulated_energy = {d.id: 0.0 for d in self.devices}
        slot_start = self.current_slot_index * self.slot_duration

        if not self.local_finish_time:
            self.local_finish_time = {d.id: slot_start for d in self.devices}
        else:
            self.local_finish_time = {
                d.id: max(self.local_finish_time.get(d.id, slot_start), slot_start)
                for d in self.devices
            }

        if not self.server_finish_time:
            self.server_finish_time = {s.id: slot_start for s in self.servers}
        else:
            self.server_finish_time = {
                s.id: max(self.server_finish_time.get(s.id, slot_start), slot_start)
                for s in self.servers
            }
        self.last_step_metrics = []

        self.digital_twin_snapshot = self.digital_twin.synchronize(
            self.devices,
            self.servers,
            time_slot_index=self.current_slot_index,
        )
        self.device_estimated_power = {
            device_id: state.estimated_compute_power
            for device_id, state in self.digital_twin_snapshot.device_states.items()
        }
        self.server_estimated_power = {
            server_id: state.estimated_compute_power
            for server_id, state in self.digital_twin_snapshot.server_states.items()
        }

        self._update_connection_windows()
        return self._get_joint_state()

    def _validate_priority_orders(
        self, task_dags: Dict[int, TaskDAG], priorities: Dict[int, List[int]]
    ) -> None:
        """Validate that each priority order respects DAG dependencies.

        Args:
            task_dags: Mapping of device IDs to TaskDAG instances.
            priorities: Mapping of device IDs to ordered subtask IDs.

        Raises:
            ValueError: If a priority order is missing subtasks or schedules a
                successor before its predecessor.
        """
        for device_id, task_dag in task_dags.items():
            priority_order = priorities.get(device_id, [])
            expected_subtasks = set(task_dag.subtasks.keys())
            ordered_subtasks = set(priority_order)
            if ordered_subtasks != expected_subtasks:
                raise ValueError(
                    f"Priority order for device {device_id} must contain every subtask exactly once."
                )

            priority_index = {
                subtask_id: order_index for order_index, subtask_id in enumerate(priority_order)
            }
            for predecessor_id, successor_id in task_dag.edges:
                if priority_index[predecessor_id] > priority_index[successor_id]:
                    raise ValueError(
                        f"Priority order schedules predecessor {predecessor_id} after successor "
                        f"{successor_id} for device {device_id}."
                    )

    def reset(self, task_dags: Dict[int, TaskDAG], priorities: Dict[int, List[int]]) -> np.ndarray:
        """Reset the episode and start the first time slot.

        Args:
            task_dags: Mapping of device IDs to TaskDAG instances.
            priorities: Mapping of device IDs to ordered subtask IDs.

        Returns:
            Joint state array for all devices.
        """
        # Backward-compatible entrypoint: reset full episode then initialize first slot.
        self.reset_episode()
        return self.start_time_slot(task_dags, priorities)

    def step(self, joint_actions: List[int]) -> Tuple[np.ndarray, List[float], bool, Dict[str, bool]]:
        """Advance one scheduling step.

        Args:
            joint_actions: One action per device (0 for local, >0 for edge server).

        Returns:
            Tuple of (next_state, rewards, episode_done, info).
        """
        self.last_step_metrics = []

        local_finish_snapshot = dict(self.local_finish_time)
        server_finish_snapshot = dict(self.server_finish_time)

        pending, server_groups = self._build_pending_items(joint_actions)
        self._schedule_local_items(pending, local_finish_snapshot)
        self._schedule_edge_items(server_groups, local_finish_snapshot, server_finish_snapshot)

        self.local_finish_time = local_finish_snapshot
        self.server_finish_time = server_finish_snapshot

        rewards = self._finalize_pending_items(pending)

        subtask_count = len(self.task_dags[self.devices[0].id].subtasks)
        slot_done = all(self.current_step[d.id] >= subtask_count for d in self.devices)
        # self._advance_mobility_within_slot(subtask_count)

        if slot_done:
            self.current_slot_index += 1
            self.current_slot = self.current_slot_index * self.slot_duration
            for device in self.devices:
                self._move_device_on_path(device, self.slot_duration)
            self._update_connection_windows()

        episode_done = self.current_slot_index >= self.time_slots
        info = {"slot_done": slot_done, "episode_done": episode_done}
        return self._get_joint_state(), rewards, episode_done, info

    def _build_pending_items(self, joint_actions: List[int]) -> Tuple[List[dict], Dict[int, List[dict]]]:
        """Create per-device execution items and group edge actions by server.

        Args:
            joint_actions: One action per device.

        Returns:
            Tuple of (pending_items, server_groups).
        """
        pending: List[dict] = []
        server_groups: Dict[int, List[dict]] = {}

        for device_index, requested_action in enumerate(joint_actions):
            device = self.devices[device_index]
            task_dag = self.task_dags[device.id]
            step_idx = self.current_step[device.id]
            priority_list = self.priorities.get(device.id, [])
            if step_idx >= len(priority_list):
                pending.append(
                    {
                        "idx": device_index,
                        "device": device,
                        "completed": True,
                        "requested_action": None,
                        "p_out": 0.0,
                    }
                )
                continue
            current_subtask_id = priority_list[step_idx]
            subtask = task_dag.subtasks[current_subtask_id]

            predecessor_ready_time, tx_energy, transfer_time = self._resolve_predecessor_ready_time(
                device=device, task_dag=task_dag, current_subtask_id=current_subtask_id, action=requested_action
            )

            item = {
                "idx": device_index,
                "device": device,
                "task_dag": task_dag,
                "subtask_id": current_subtask_id,
                "subtask": subtask,
                "requested_action": requested_action,
                "predecessor_ready_time": predecessor_ready_time,
                "tx_energy": tx_energy,
                "transfer_time": transfer_time,
                "p_out": 0.0,
                "rejected": False,
            }
            pending.append(item)
            if requested_action > 0:
                server_groups.setdefault(requested_action, []).append(item)

        return pending, server_groups

    def _schedule_local_items(self, pending: List[dict], local_finish_snapshot: Dict[int, float]) -> None:
        """Schedule local execution items and update their timings.

        Args:
            pending: Items awaiting scheduling.
            local_finish_snapshot: Copy of device local finish times.
        """
        for item in pending:
            if item["requested_action"] != 0:
                continue
            device = item["device"]
            subtask = item["subtask"]
            est_power = self.device_estimated_power[device.id]
            actual_power = device.compute_power
            comp_time, comp_energy = self.network_env.calculate_local_computation(
                subtask.cpu_cycles, device.energy_coeff, est_power, actual_power
            )
            local_wait = max(0.0, local_finish_snapshot[device.id] - self.current_slot)
            start_time = max(self.current_slot + local_wait, item["predecessor_ready_time"])
            finish_time = start_time + comp_time
            local_finish_snapshot[device.id] = finish_time

            item.update(
                {
                    "resolved_action": 0,
                    "start_time": start_time,
                    "finish_time": finish_time,
                    "local_time": comp_time,
                    "server_time": 0.0,
                    "attempted_server_time": 0.0,
                    "queue_or_wait_time": max(0.0, start_time - self.current_slot),
                    "penalty_time": 0.0,
                    "f_est": est_power,
                    "f_actual": actual_power,
                    "e_comp": comp_energy,
                }
            )

    def _schedule_edge_items(
        self,
        server_groups: Dict[int, List[dict]],
        local_finish_snapshot: Dict[int, float],
        server_finish_snapshot: Dict[int, float],
    ) -> None:
        """Schedule edge execution items, enforcing connection windows.

        Args:
            server_groups: Items grouped by requested edge server index.
            local_finish_snapshot: Copy of device local finish times.
            server_finish_snapshot: Copy of server finish times.
        """
        for action, group in server_groups.items():
            server = self.servers[action - 1]
            server_available_time = server_finish_snapshot[server.id]
            est_power = self.server_estimated_power[server.id]
            actual_power = server.compute_power

            for item in group:
                window_start, window_end = self.connection_windows[(item["device"].id, server.id)]
                item["l_start"] = window_start
                item["l_end"] = window_end

            # Deterministic FIFO-like queue discipline by earliest feasible start
            # (dependency-ready and link-available), then device id tie-break.
            group.sort(
                key=lambda item: (
                    max(item["predecessor_ready_time"], item["l_start"] + 1e-9),
                    item["device"].id,
                )
            )

            for item in group:
                subtask = item["subtask"]
                comp_time, comp_energy = self.network_env.calculate_edge_computation(
                    subtask.cpu_cycles, server.energy_coeff, est_power, actual_power
                )
                start_time = max(
                    self.current_slot,
                    server_available_time,
                    item["predecessor_ready_time"],
                    item["l_start"] + 1e-9,
                )
                finish_time = start_time + comp_time
                within_window = start_time > item["l_start"] and finish_time < item["l_end"]
                if (not within_window) and self.strict_connection_window:
                    # Reject invalid offload and fallback to local execution with penalty.
                    device = item["device"]
                    (
                        predecessor_ready_time,
                        tx_energy,
                        transfer_time,
                    ) = self._resolve_predecessor_ready_time(
                        device=device,
                        task_dag=item["task_dag"],
                        current_subtask_id=item["subtask_id"],
                        action=0,
                    )
                    item["predecessor_ready_time"] = predecessor_ready_time
                    item["tx_energy"] = tx_energy
                    item["transfer_time"] = transfer_time
                    local_est_power = self.device_estimated_power[device.id]
                    local_actual_power = device.compute_power
                    local_time, local_energy = self.network_env.calculate_local_computation(
                        subtask.cpu_cycles, device.energy_coeff, local_est_power, local_actual_power
                    )
                    local_wait = max(0.0, local_finish_snapshot[device.id] - self.current_slot)
                    local_start = max(self.current_slot + local_wait, predecessor_ready_time)
                    local_finish = local_start + local_time
                    local_finish_snapshot[device.id] = local_finish
                    item["rejected"] = True
                    item["p_out"] = self.p_out_value
                    item["attempted_start_time"] = start_time
                    item["attempted_finish_time"] = finish_time
                    item["penalty_time"] = (
                        max(0.0, item["l_start"] - start_time)
                        + max(0.0, finish_time - item["l_end"])
                    )
                    item.update(
                        {
                            "resolved_action": 0,
                            "start_time": local_start,
                            "finish_time": local_finish,
                            "local_time": local_time,
                            "server_time": 0.0,
                            "attempted_server_time": comp_time,
                            "queue_or_wait_time": max(0.0, local_start - self.current_slot),
                            "f_est": local_est_power,
                            "f_actual": local_actual_power,
                            "e_comp": local_energy,
                        }
                    )
                    continue

                item["p_out"] = self.p_out_value if not within_window else 0.0
                if item["rejected"]:
                    item.update(
                        {
                            "resolved_action": action,
                            "f_est": est_power,
                            "f_actual": actual_power,
                        }
                    )
                else:
                    item.update(
                        {
                            "resolved_action": action,
                            "start_time": start_time,
                            "finish_time": finish_time,
                            "local_time": 0.0,
                            "server_time": comp_time,
                            "attempted_server_time": comp_time,
                            "queue_or_wait_time": max(0.0, start_time - self.current_slot),
                            "penalty_time": 0.0,
                            "f_est": est_power,
                            "f_actual": actual_power,
                            "e_comp": comp_energy,
                        }
                    )
                    server_available_time = finish_time
                    server_finish_snapshot[server.id] = server_available_time

    def _finalize_pending_items(self, pending: List[dict]) -> List[float]:
        """Compute rewards and update per-device metrics for scheduled items.

        Args:
            pending: Items scheduled in the current step.

        Returns:
            Reward list aligned with device order.
        """
        rewards: List[float] = []
        pending.sort(key=lambda item: item["idx"])

        for item in pending:
            device = item["device"]
            if item.get("completed"):
                rewards.append(0.0)
                self.last_step_metrics.append(
                    {
                        "device_id": float(device.id),
                        "subtask_id": -1.0,
                        "action": -1.0,
                        "requested_action": -1.0,
                        "f_est": 0.0,
                        "f_actual": 0.0,
                        "current_slot": float(self.current_slot),
                        "start_time": 0.0,
                        "finish_time": 0.0,
                        "execution_time": 0.0,
                        "local_time": 0.0,
                        "server_time": 0.0,
                        "attempted_server_time": 0.0,
                        "queue_or_wait_time": 0.0,
                        "transfer_time": 0.0,
                        "delay": 0.0,
                        "energy": 0.0,
                        "tx_energy": 0.0,
                        "comp_energy": 0.0,
                        "reward": 0.0,
                        "p_out": 0.0,
                        "penalty_applied": 0.0,
                        "penalty_time": 0.0,
                        "fallback_local": 0.0,
                    }
                )
                continue
            task_dag = item["task_dag"]
            current_subtask_id = item["subtask_id"]

            subtask_latency = item["transfer_time"] + item["queue_or_wait_time"] + (
                item["finish_time"] - item["start_time"]
            )
            instant_energy = item["tx_energy"] + item["e_comp"]

            self.subtask_finish_times[device.id][current_subtask_id] = item["finish_time"]
            self.subtask_locations[device.id][current_subtask_id] = item["resolved_action"]

            accum_delay = self.slot_accumulated_delay[device.id]
            accum_energy = self.slot_accumulated_energy[device.id]
            finish_elapsed = max(0.0, item["finish_time"] - self.current_slot)
            dag_makespan = max(accum_delay, finish_elapsed)
            instant_delay = dag_makespan - accum_delay
            self.slot_accumulated_delay[device.id] += instant_delay
            self.slot_accumulated_energy[device.id] += instant_energy
            self.device_accumulated_delay[device.id] += instant_delay
            self.device_accumulated_energy[device.id] += instant_energy

            reward = self._calculate_reward(
                instant_delay,
                instant_energy,
                self.slot_accumulated_delay[device.id],
                self.slot_accumulated_energy[device.id],
                item["p_out"],
                task_dag,
            )
            rewards.append(reward)
            self.last_step_metrics.append(
                {
                    "device_id": float(device.id),
                    "subtask_id": float(current_subtask_id),
                    "action": float(item["resolved_action"]),
                    "requested_action": float(item["requested_action"]),
                    "f_est": float(item["f_est"]),
                    "f_actual": float(item["f_actual"]),
                    "f_deviation": float(item["f_est"] - item["f_actual"]),
                    "current_slot": float(self.current_slot),
                    "start_time": float(item["start_time"]),
                    "finish_time": float(item["finish_time"]),
                    "execution_time": float(item["finish_time"] - item["start_time"]),
                    "local_time": float(item["local_time"]),
                    "server_time": float(item["server_time"]),
                    "attempted_server_time": float(item["attempted_server_time"]),
                    "queue_or_wait_time": float(item["queue_or_wait_time"]),
                    "transfer_time": float(item["transfer_time"]),
                    "delay": float(instant_delay),
                    "subtask_latency": float(subtask_latency),
                    "dag_makespan": float(dag_makespan),
                    "energy": float(instant_energy),
                    "tx_energy": float(item["tx_energy"]),
                    "comp_energy": float(item["e_comp"]),
                    "reward": float(reward),
                    "p_out": float(item["p_out"]),
                    "penalty_applied": float(1.0 if item["p_out"] != 0.0 else 0.0),
                    "penalty_time": float(item["penalty_time"]),
                    "fallback_local": float(1.0 if item.get("rejected", False) else 0.0),
                }
            )

            self.current_step[device.id] += 1

        return rewards

    def _advance_mobility_within_slot(self, subtask_count: int) -> None:
        """Advance device mobility within the current time slot.

        Args:
            subtask_count: Total subtasks per device in this slot.
        """
        slot_end_time = (self.current_slot_index + 1) * self.slot_duration
        elapsed_per_subtask = self.slot_duration / max(subtask_count, 1)
        remaining_in_slot = max(0.0, slot_end_time - self.current_slot)
        elapsed = min(elapsed_per_subtask, remaining_in_slot)
        if elapsed > 0.0:
            for device in self.devices:
                self._move_device_on_path(device, elapsed)
            self.current_slot += elapsed
            self._update_connection_windows()

    def _resolve_predecessor_ready_time(
        self, device: IndustrialDevice, task_dag: TaskDAG, current_subtask_id: int, action: int
    ) -> Tuple[float, float, float]:
        """Compute earliest start time and transfer energy from predecessors.

        Args:
            device: Device executing the subtask.
            task_dag: Task DAG for the device.
            current_subtask_id: Current subtask identifier.
            action: Requested execution location (0 local, >0 edge).

        Returns:
            Tuple of (predecessor_ready_time, transfer_energy, transfer_time).
        """
        predecessors = [pred for pred, succ in task_dag.edges if succ == current_subtask_id]
        # predecessor_ready_time = 0.0
        slot_start = self.current_slot_index * self.slot_duration
        predecessor_ready_time = slot_start
        tx_energy = 0.0
        transfer_time = 0.0
        for pred_id in predecessors:
            pred_finish = self.subtask_finish_times[device.id].get(pred_id, 0.0)
            pred_location = self.subtask_locations[device.id].get(pred_id, 0)
            if pred_location != action:
                trans_delay, trans_energy = self._calculate_result_transfer(
                    device, task_dag, pred_id, pred_location, action
                )
                pred_finish += trans_delay
                tx_energy += trans_energy
                transfer_time += trans_delay
            predecessor_ready_time = max(predecessor_ready_time, pred_finish)

        # Eq. (15)-(17) term for first offloaded subtask: upload raw subtask input to edge server.
        if action > 0 and len(predecessors) == 0:
            input_delay, input_energy = self._calculate_input_upload(
                device, action, task_dag.subtasks[current_subtask_id].data_size
            )
            # predecessor_ready_time += input_delay
            predecessor_ready_time = slot_start + input_delay
            tx_energy += input_energy
            transfer_time += input_delay

        return predecessor_ready_time, tx_energy, transfer_time

    def _calculate_result_transfer(
        self, device: IndustrialDevice, task_dag: TaskDAG, pred_id: int, pred_location: int, action: int
    ) -> Tuple[float, float]:
        """Return transfer delay and energy when predecessor results move nodes.

        Args:
            device: Device executing the current subtask.
            task_dag: Task DAG containing subtask metadata.
            pred_id: Predecessor subtask ID.
            pred_location: Execution location of predecessor.
            action: Execution location for current subtask.

        Returns:
            Tuple of (transfer_delay, transfer_energy).
        """
        if pred_location == action:
            return 0.0, 0.0
        if action == 0 and pred_location > 0:
            pred_server = self.servers[pred_location - 1]
            gain = self.network_env.calculate_channel_gain(device.location, pred_server.location)
            down_rate = max(self.network_env.get_downlink_rate(pred_server.transmit_power, gain), 1e-9)
            trans_delay = task_dag.subtasks[pred_id].result_size / down_rate
            return trans_delay, pred_server.transmit_power * trans_delay
        if action > 0 and pred_location == 0:
            target_server = self.servers[action - 1]
            gain = self.network_env.calculate_channel_gain(device.location, target_server.location)
            up_rate = max(self.network_env.get_uplink_rate(device.transmit_power, gain), 1e-9)
            trans_delay = task_dag.subtasks[pred_id].result_size / up_rate
            return trans_delay, device.transmit_power * trans_delay
        return 0.0, 0.0

    def _calculate_input_upload(self, device: IndustrialDevice, action: int, data_size: float) -> Tuple[float, float]:
        """Return upload delay and energy for initial subtask input.

        Args:
            device: Device uploading input data.
            action: Target execution location (0 local, >0 edge).
            data_size: Input data size (bytes).

        Returns:
            Tuple of (upload_delay, upload_energy).
        """
        if action <= 0:
            return 0.0, 0.0
        target_server = self.servers[action - 1]
        gain = self.network_env.calculate_channel_gain(device.location, target_server.location)
        up_rate = max(self.network_env.get_uplink_rate(device.transmit_power, gain), 1e-9)
        upload_delay = data_size / up_rate
        return upload_delay, device.transmit_power * upload_delay

    def get_state_dim(self) -> int:
        """Return the flattened state vector dimension.

        Returns:
            State vector length for a single device.
        """
        # Eq. (23): {f_loc, W_d, subtask features(C,D,R), priority sequence, F_edge, W_s, l_start, l_end}
        # Priority sequence is represented as a normalized full vector with one value per subtask.
        subtask_count = 0
        if self.task_dags:
            first_dag = self.task_dags[next(iter(self.task_dags))]
            subtask_count = len(first_dag.subtasks)
        if subtask_count == 0:
            subtask_count = 5  # project uses 5-stage KolektorSDD pipeline
        return (2 + 3) + subtask_count + 4 * len(self.servers)

    def _update_connection_windows(self) -> None:
        """Recompute connection windows for every device-server pair."""
        self.runtime_metrics["connection_window_requests"] += 1.0
        cache_key = self._connection_window_signature()
        if cache_key == self._connection_window_cache_key:
            return

        update_start = time.perf_counter()
        device_locations = np.stack([device.location for device in self.devices])
        device_velocities = np.stack(
            [device.direction * device.speed_mps for device in self.devices]
        )
        server_locations = np.stack([server.location for server in self.servers])
        coverage_radii = np.asarray(
            [server.coverage_radius for server in self.servers], dtype=float
        )

        sampled_locations = (
            device_locations[:, np.newaxis, :]
            + device_velocities[:, np.newaxis, :]
            * self._connection_window_time_offsets[np.newaxis, :, np.newaxis]
        )
        location_deltas = (
            sampled_locations[:, np.newaxis, :, :]
            - server_locations[np.newaxis, :, np.newaxis, :]
        )
        distances = np.linalg.norm(location_deltas, axis=3)
        inside = distances <= coverage_radii[np.newaxis, :, np.newaxis]

        has_inside_sample = np.any(inside, axis=2)
        first_inside_index = np.argmax(inside, axis=2)
        sample_indices = np.arange(inside.shape[2], dtype=int)
        after_first_inside = (
            sample_indices[np.newaxis, np.newaxis, :]
            > first_inside_index[:, :, np.newaxis]
        )
        first_exit_candidates = (~inside) & after_first_inside
        has_exit_sample = np.any(first_exit_candidates, axis=2)
        first_exit_index = np.argmax(first_exit_candidates, axis=2)

        horizon = self.slot_duration
        start_offsets = self._connection_window_time_offsets[first_inside_index]
        end_offsets = np.where(
            has_exit_sample,
            self._connection_window_time_offsets[first_exit_index],
            horizon,
        )
        start_offsets = np.where(has_inside_sample, start_offsets, horizon)
        end_offsets = np.where(has_inside_sample, end_offsets, horizon)

        self.connection_windows = {
            (device.id, server.id): (
                self.current_slot + float(start_offsets[device_index, server_index]),
                self.current_slot + float(end_offsets[device_index, server_index]),
            )
            for device_index, device in enumerate(self.devices)
            for server_index, server in enumerate(self.servers)
        }
        self._connection_window_cache_key = cache_key
        self.runtime_metrics["connection_window_seconds"] += (
            time.perf_counter() - update_start
        )
        self.runtime_metrics["connection_window_updates"] += 1.0
        self.runtime_metrics["connection_window_samples"] += float(inside.size)

    def _connection_window_signature(self) -> Tuple[object, ...]:
        """Return state fields that determine all sampled connection windows."""
        device_state = tuple(
            (
                device.id,
                *np.asarray(device.location, dtype=float).tolist(),
                *np.asarray(device.direction, dtype=float).tolist(),
                float(device.speed_mps),
            )
            for device in self.devices
        )
        server_state = tuple(
            (
                server.id,
                *np.asarray(server.location, dtype=float).tolist(),
                float(server.coverage_radius),
            )
            for server in self.servers
        )
        return (float(self.current_slot), device_state, server_state)

    def _build_predetermined_paths(self) -> None:
        """Initialize fixed mobility paths for industrial devices."""
        if self.device_waypoints:
            return
        if self.route_rectangles is None:
            # 5 fixed rectangular routes from user-confirmed Fig.4 topology.
            rectangles = [
                [np.array([10.0, 10.0]), np.array([10.0, 30.0]), np.array([30.0, 30.0]), np.array([30.0, 10.0])],
                [np.array([70.0, 10.0]), np.array([40.0, 10.0]), np.array([40.0, 20.0]), np.array([70.0, 20.0])],
                [np.array([40.0, 20.0]), np.array([40.0, 50.0]), np.array([60.0, 50.0]), np.array([60.0, 20.0])],
                [np.array([70.0, 10.0]), np.array([70.0, 40.0]), np.array([90.0, 40.0]), np.array([90.0, 10.0])],
                [np.array([65.0, 45.0]), np.array([65.0, 55.0]), np.array([90.0, 55.0]), np.array([90.0, 45.0])],
            ]
        else:
            rectangles = [
                [np.asarray(point, dtype=float) for point in rectangle]
                for rectangle in self.route_rectangles
            ]
        route_count_needed = (len(self.devices) + 1) // 2
        if len(rectangles) < route_count_needed:
            raise ValueError("route_rectangles must provide one route per two devices")
        for device_index, device in enumerate(self.devices):
            route_index = device_index // 2
            route = [
                np.clip(point.copy(), self.world_min, self.world_max)
                for point in rectangles[route_index]
            ]
            route.append(route[0].copy())
            self.device_waypoints[device.id] = route
            # Two robots on same rectangle start from different corners.
            self.device_waypoint_idx[device.id] = 0 if (device_index % 2 == 0) else 2

    def _move_device_on_path(self, device: IndustrialDevice, elapsed: float) -> None:
        """Move a device along its path for the given elapsed time.

        Args:
            device: Device to move.
            elapsed: Elapsed time in seconds.
        """
        remaining = elapsed
        waypoints = self.device_waypoints[device.id]
        waypoint_index = self.device_waypoint_idx[device.id]
        pos = device.location.copy()
        speed = max(device.speed_mps, 1e-9)

        while remaining > 0:
            target_index = (waypoint_index + 1) % len(waypoints)
            target = waypoints[target_index]
            vec = target - pos
            dist = np.linalg.norm(vec)

            if dist < 1e-9:
                waypoint_index = target_index
                continue

            dir_vec = vec / dist
            travel = min(dist, speed * remaining)
            pos = pos + dir_vec * travel
            remaining -= travel / speed

            if travel >= dist - 1e-9:
                waypoint_index = target_index
            else:
                break

        device.update_location(np.clip(pos, self.world_min, self.world_max))
        self.device_waypoint_idx[device.id] = waypoint_index
        next_index = (waypoint_index + 1) % len(waypoints)
        device.set_direction(waypoints[next_index] - device.location)

    def _compute_waiting_delay_local(self, device: IndustrialDevice) -> float:
        """Return queueing delay at the local device.

        Args:
            device: Device whose local queue is measured.

        Returns:
            Waiting delay in seconds.
        """
        # Eq. (7): queueing delay at local node
        return max(0.0, self.local_finish_time[device.id] - self.current_slot)

    def _compute_waiting_delay_server(self, server: EdgeServer) -> float:
        """Return queueing delay at the edge server.

        Args:
            server: Edge server whose queue is measured.

        Returns:
            Waiting delay in seconds.
        """
        # Eq. (12): queueing delay at edge server
        return max(0.0, self.server_finish_time[server.id] - self.current_slot)

    def _get_joint_state(self) -> np.ndarray:
        """Build the normalized joint state vector for all devices.

        Returns:
            Joint state array shaped (num_devices, state_dim).
        """
        state_start = time.perf_counter()
        joint_state = []
        for device in self.devices:
            task_dag = self.task_dags[device.id]
            state_step = min(self.current_step[device.id], len(self.priorities[device.id]) - 1)
            current_subtask_id = self.priorities[device.id][state_step]
            subtask = task_dag.subtasks[current_subtask_id]

            local_power_norm = self.device_estimated_power[device.id] / 1e9
            local_wait_norm = self._compute_waiting_delay_local(device) / max(self.slot_duration, 1e-6)
            cpu_norm = subtask.cpu_cycles / 1e9
            data_norm = subtask.data_size / 1e6
            result_norm = subtask.result_size / 1e6
            # Paper Eq.(23) uses priority information R_di^t. We encode this as
            # a normalized per-subtask rank vector over the full priority sequence.
            priority_order = self.priorities[device.id]
            total_subtasks = max(len(priority_order), 1)
            priority_rank = {
                sid: rank for rank, sid in enumerate(priority_order, start=1)
            }
            priority_vector = [
                priority_rank.get(subtask_id, total_subtasks) / total_subtasks
                for subtask_id in sorted(task_dag.subtasks.keys())
            ]
            edge_power_norm = [self.server_estimated_power[server.id] / 1e9 for server in self.servers]
            edge_wait_norm = [
                self._compute_waiting_delay_server(server) / max(self.slot_duration, 1e-6)
                for server in self.servers
            ]
            window_starts = []
            window_ends = []
            horizon = self.slot_duration
            for server in self.servers:
                l_start, l_end = self.connection_windows[(device.id, server.id)]
                window_starts.append((l_start - self.current_slot) / max(horizon, 1e-6))
                window_ends.append((l_end - self.current_slot) / max(horizon, 1e-6))

            state_vector = (
                [local_power_norm, local_wait_norm, cpu_norm, data_norm, result_norm]
                + priority_vector
                + edge_power_norm
                + edge_wait_norm
                + window_starts
                + window_ends
            )
            joint_state.append(state_vector)
        joint_state_array = np.array(joint_state, dtype=np.float32)
        self.runtime_metrics["joint_state_seconds"] += time.perf_counter() - state_start
        self.runtime_metrics["joint_state_calls"] += 1.0
        return joint_state_array

    def _calculate_reward(
        self, t_im: float, e_im: float, t_accm: float, e_accm: float, p_out: float, task_dag: TaskDAG
    ) -> float:
        """Compute reward per Equation (24).

        Args:
            t_im: Immediate processing delay.
            e_im: Immediate energy consumption.
            t_accm: Accumulated delay in the slot.
            e_accm: Accumulated energy in the slot.
            p_out: Connection window penalty.
            task_dag: Task DAG with t_max and e_max.

        Returns:
            Reward value.
        """
        # Eq. (24)
        t_max = task_dag.t_max
        e_max = task_dag.e_max
        m_total = len(task_dag.subtasks)
        return (
            self.lambda1 * ((t_max / m_total) - t_im)
            + self.lambda2 * (t_max - t_accm)
            + self.lambda3 * ((e_max / m_total) - e_im)
            + self.lambda4 * (e_max - e_accm)
            + self.lambda5 * p_out
        )
