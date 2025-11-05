"""Deep Q-Learning agent and RMS scheduling environment."""
from __future__ import annotations

import math
import random
from collections import Counter
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import CONFIG
from data_preprocessing import JobRecord, Scenario


# ---------------------------------------------------------------------------
# Replay buffer with prioritised sampling
# ---------------------------------------------------------------------------


class PrioritizedReplayBuffer:
    """Experience replay buffer supporting proportional prioritisation."""

    def __init__(self, capacity: int, alpha: float = 0.6, epsilon: float = 1e-4):
        self.capacity = capacity
        self.alpha = alpha
        self.epsilon = epsilon
        self.buffer: List[Tuple[np.ndarray, int, float, np.ndarray, bool, np.ndarray]] = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.position = 0

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        next_valid_mask: np.ndarray,
    ) -> None:
        max_priority = self.priorities.max() if self.buffer else 1.0
        experience = (state, action, reward, next_state, done, next_valid_mask)
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
        self.priorities[self.position] = max(max_priority, self.epsilon)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int, beta: float) -> Tuple[List, np.ndarray, np.ndarray]:
        if len(self.buffer) == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[: len(self.buffer)]

        scaled = priorities ** self.alpha
        probs = scaled / scaled.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        weights = (len(self.buffer) * probs[indices]) ** (-beta)
        weights /= weights.max() + 1e-8
        return samples, indices, weights.astype(np.float32)

    def update_priorities(self, indices: Iterable[int], priorities: Iterable[float]) -> None:
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = max(float(priority), self.epsilon)

    def __len__(self) -> int:  # pragma: no cover - tiny wrapper
        return len(self.buffer)


# ---------------------------------------------------------------------------
# Neural network policy
# ---------------------------------------------------------------------------


class RMSDuelingNetwork(nn.Module):
    """Dueling network used as the function approximator."""

    def __init__(self, state_dim: int, action_dim: int, config=CONFIG):
        super().__init__()
        self.dueling = config.network.dueling
        self.layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()

        input_dim = state_dim
        for hidden_units in config.network.hidden_units:
            layer = nn.Linear(input_dim, hidden_units)
            nn.init.kaiming_uniform_(layer.weight, a=math.sqrt(5))
            nn.init.zeros_(layer.bias)
            self.layers.append(layer)
            if config.network.layer_norm:
                self.norm_layers.append(nn.LayerNorm(hidden_units))
            else:
                self.norm_layers.append(nn.Identity())
            input_dim = hidden_units

        self.activation = self._resolve_activation(config.network.activation)
        self.dropout = (
            nn.Dropout(config.network.dropout)
            if config.network.dropout > 0
            else nn.Identity()
        )

        if self.dueling:
            shared_dim = input_dim
            hidden_dim = max(shared_dim // 2, 32)
            self.value_stream = nn.Sequential(
                nn.Linear(shared_dim, hidden_dim),
                self.activation,
                nn.Linear(hidden_dim, 1),
            )
            self.advantage_stream = nn.Sequential(
                nn.Linear(shared_dim, hidden_dim),
                self.activation,
                nn.Linear(hidden_dim, action_dim),
            )
        else:
            self.head = nn.Linear(input_dim, action_dim)

    @staticmethod
    def _resolve_activation(name: str) -> nn.Module:
        mapping = {
            "relu": nn.ReLU(),
            "elu": nn.ELU(),
            "gelu": nn.GELU(),
            "silu": nn.SiLU(),
        }
        return mapping.get(name.lower(), nn.ELU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer, norm in zip(self.layers, self.norm_layers):
            x = layer(x)
            x = norm(x)
            x = self.activation(x)
            x = self.dropout(x)
        if self.dueling:
            value = self.value_stream(x)
            advantage = self.advantage_stream(x)
            return value + advantage - advantage.mean(dim=1, keepdim=True)
        return self.head(x)


# ---------------------------------------------------------------------------
# RMS environment
# ---------------------------------------------------------------------------


class RMSSchedulingEnvironment:
    """Reconfigurable manufacturing environment with dense state feedback."""

    def __init__(
        self,
        config=CONFIG,
        normalization_stats: Optional[Dict[str, float]] = None,
        machine_groups: Optional[Sequence[str]] = None,
        product_families: Optional[Sequence[str]] = None,
    ):
        self.config = config
        self.norms = normalization_stats or {
            "max_processing": 1.0,
            "max_due": config.environment.horizon_hours,
            "max_energy": 1.0,
            "max_setup": 0.1,
            "max_reconfig": 0.1,
            "max_slack": 1.0,
            "max_arrival": 1.0,
        }
        self.machine_ids = list(machine_groups or [])
        self.product_families = list(product_families or [])
        self.machine_to_index = {m: idx for idx, m in enumerate(self.machine_ids)}
        self.family_to_index = {p: idx for idx, p in enumerate(self.product_families)}
        self.machine_feature_count = 6
        self.job_feature_count = 10
        self.global_feature_count = 9
        self.action_size = config.environment.candidate_pool
        self.state_dim = (
            len(self.machine_ids) * self.machine_feature_count
            + self.action_size * self.job_feature_count
            + self.global_feature_count
        )
        self.random = np.random.default_rng(config.synthetic.random_seed)
        self.reset_state()

    # ------------------------------------------------------------------
    def reset_state(self) -> None:
        self.pending_jobs: List[JobRecord] = []
        self.total_jobs = 0
        self.completed_jobs = 0
        self.current_time = 0.0
        self.machine_available: Dict[str, float] = {machine: 0.0 for machine in self.machine_ids}
        self.machine_busy_time: Dict[str, float] = {machine: 0.0 for machine in self.machine_ids}
        self.machine_energy: Dict[str, float] = {machine: 0.0 for machine in self.machine_ids}
        self.machine_last_family: Dict[str, Optional[str]] = {machine: None for machine in self.machine_ids}
        self.reconfigurations = 0
        self.invalid_actions = 0
        self.total_energy = 0.0
        self.total_tardiness = 0.0
        self.total_setup = 0.0
        self.total_idle = 0.0
        self.reward_component_totals = Counter()
        self.step_reward_breakdown: List[Dict[str, float]] = []
        self.step_rewards: List[float] = []
        self.slack_samples: List[float] = []
        self.backlog_trace: List[float] = []
        self.gini_trace: List[float] = []
        self.machine_energy_trace: List[Dict[str, float]] = []
        self.machine_util_trace: List[Dict[str, float]] = []
        self.candidate_jobs: List[Optional[JobRecord]] = []
        self.valid_action_mask = np.zeros(self.action_size, dtype=bool)
        self.schedule_log: List[Dict[str, float]] = []
        self.steps = 0
        self.scenario_id = ""

    def reset(self, scenario: Scenario, deterministic: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        self.reset_state()
        self.scenario_id = scenario.scenario_id
        self.pending_jobs = sorted(scenario.jobs, key=lambda job: (job.arrival_time, job.due_time))
        self.total_jobs = len(self.pending_jobs)
        self.current_time = min(job.arrival_time for job in self.pending_jobs) if self.pending_jobs else 0.0
        return self._build_state()

    # ------------------------------------------------------------------
    def step(self, action_index: int) -> Tuple[np.ndarray, float, bool, Dict[str, np.ndarray]]:
        self.steps += 1
        reward_components: Dict[str, float] = {}

        if action_index >= self.action_size:
            action_index = int(action_index % self.action_size)

        if not self.valid_action_mask[action_index]:
            penalty = self.config.reward.invalid_action_penalty
            reward_components["invalid"] = penalty
            self.invalid_actions += 1
            reward = penalty
            self.current_time += 0.05
            next_state, mask = self._build_state()
            info = self._build_info(None, reward_components, mask)
            return next_state, float(reward), False, info

        selected_job = self.candidate_jobs[action_index]
        if selected_job is None:
            penalty = self.config.reward.invalid_action_penalty
            reward_components["invalid"] = penalty
            self.invalid_actions += 1
            reward = penalty
            next_state, mask = self._build_state()
            info = self._build_info(None, reward_components, mask)
            return next_state, float(reward), False, info

        self.pending_jobs = [job for job in self.pending_jobs if job.job_id != selected_job.job_id]

        machine = selected_job.machine_group
        machine_available = self.machine_available.get(machine, 0.0)
        start_time = max(machine_available, selected_job.arrival_time, self.current_time)

        requires_reconfig = (
            self.machine_last_family[machine] is not None
            and self.machine_last_family[machine] != selected_job.product_family
        )
        setup_time = selected_job.setup_time
        if requires_reconfig:
            setup_time += selected_job.reconfiguration_cost
        idle_time = max(0.0, start_time - machine_available)
        processing_time = selected_job.processing_time
        finish_time = start_time + setup_time + processing_time
        tardiness = max(0.0, finish_time - selected_job.due_time)
        slack_after = max(0.0, selected_job.due_time - finish_time)
        energy = selected_job.energy_cost

        reward_components["completion"] = self.config.reward.completion_bonus
        throughput_progress = (self.completed_jobs + 1) / max(1, self.total_jobs)
        reward_components["throughput"] = self.config.reward.throughput_weight * throughput_progress
        if tardiness <= 1e-6:
            reward_components["on_time"] = self.config.reward.on_time_bonus
        else:
            reward_components["tardiness"] = self.config.reward.tardiness_penalty * tardiness

        energy_gap = energy - self.config.reward.energy_threshold
        reward_components["energy"] = self.config.reward.energy_penalty * energy_gap
        reward_components["setup"] = self.config.reward.setup_penalty * setup_time
        if requires_reconfig:
            reward_components["reconfiguration"] = (
                self.config.reward.reconfiguration_penalty * selected_job.reconfiguration_cost
            )
        if idle_time > 0:
            reward_components["idle"] = self.config.reward.idle_penalty * idle_time
        if slack_after > 0:
            window = selected_job.due_time - selected_job.arrival_time
            window = window if window > 0 else 1.0
            reward_components["slack"] = self.config.reward.slack_bonus * (slack_after / window)

        reward = float(sum(reward_components.values()))

        self.machine_available[machine] = finish_time
        self.machine_busy_time[machine] += processing_time + setup_time
        self.machine_energy[machine] += energy
        self.machine_last_family[machine] = selected_job.product_family
        if requires_reconfig:
            self.reconfigurations += 1

        self.completed_jobs += 1
        self.total_energy += energy
        self.total_tardiness += tardiness
        self.total_setup += setup_time
        self.total_idle += idle_time
        self.reward_component_totals.update(reward_components)
        self.step_reward_breakdown.append(reward_components)
        self.step_rewards.append(reward)
        self.slack_samples.append(slack_after)

        if self.pending_jobs:
            next_machine_ready = min(self.machine_available.values())
            next_arrival = min(job.arrival_time for job in self.pending_jobs)
            self.current_time = min(next_machine_ready, next_arrival)
        else:
            self.current_time = min(self.machine_available.values())

        self.schedule_log.append(
            {
                "job_id": selected_job.job_id,
                "machine": machine,
                "product_family": selected_job.product_family,
                "start_time": float(start_time),
                "finish_time": float(finish_time),
                "tardiness": float(tardiness),
                "energy": float(energy),
                "setup_time": float(setup_time),
                "requires_reconfig": bool(requires_reconfig),
                "reward": float(reward),
                "slack_after": float(slack_after),
            }
        )

        next_state, mask = self._build_state()
        done = self.completed_jobs >= self.total_jobs or self.steps >= self.config.training.max_steps_per_episode
        info = self._build_info(
            selected_job, reward_components, mask, tardiness, energy, slack_after, idle_time
        )
        return next_state, reward, done, info

    # ------------------------------------------------------------------
    def _build_state(self) -> Tuple[np.ndarray, np.ndarray]:
        candidates, valid_flags = self._select_candidate_jobs()
        features: List[float] = []

        for machine in self.machine_ids:
            available = self.machine_available.get(machine, 0.0)
            busy = self.machine_busy_time.get(machine, 0.0)
            energy = self.machine_energy.get(machine, 0.0)
            last_family = self.machine_last_family.get(machine)
            queue_ratio = self._pending_ratio(machine)
            utilization = busy / max(1.0, self.current_time + 1.0)
            energy_norm = energy / max(self.norms["max_energy"], 1.0)
            last_idx = self.family_to_index.get(last_family, -1)
            last_ratio = 0.0 if last_idx < 0 else last_idx / max(1, len(self.family_to_index) - 1)
            features.extend(
                [
                    available / max(self.config.environment.horizon_hours, 1.0),
                    utilization,
                    energy_norm,
                    1.0 if last_family is not None else 0.0,
                    queue_ratio,
                    last_ratio,
                ]
            )

        self.candidate_jobs = []
        mask = np.zeros(self.action_size, dtype=bool)
        for idx in range(self.action_size):
            if idx < len(candidates):
                job = candidates[idx]
                available_flag = valid_flags[idx]
                mask[idx] = available_flag
                self.candidate_jobs.append(job)
                features.extend(self._encode_job(job, available_flag))
            else:
                self.candidate_jobs.append(None)
                features.extend([0.0] * self.job_feature_count)

        self.valid_action_mask = mask

        pending_ratio = len(self.pending_jobs) / max(1, self.total_jobs)
        completed_ratio = self.completed_jobs / max(1, self.total_jobs)
        slack_values = [
            job.due_time - max(job.arrival_time, self.current_time) - job.processing_time
            for job in self.pending_jobs
        ]
        slack_mean = np.mean(slack_values) if slack_values else 0.0
        slack_std = np.std(slack_values) if slack_values else 0.0
        energy_pending = [job.energy_cost for job in self.pending_jobs]
        energy_mean = np.mean(energy_pending) if energy_pending else 0.0
        energy_std = np.std(energy_pending) if energy_pending else 0.0
        priority_mean = np.mean([job.priority for job in self.pending_jobs]) if self.pending_jobs else 0.0
        gini = self._gini(list(self.machine_busy_time.values()))

        self.backlog_trace.append(pending_ratio)
        self.gini_trace.append(gini)
        self.machine_energy_trace.append(self.machine_energy.copy())
        self.machine_util_trace.append(self._compute_machine_utilization())

        features.extend(
            [
                self.current_time / max(self.config.environment.horizon_hours, 1.0),
                pending_ratio,
                completed_ratio,
                slack_mean / max(self.norms["max_slack"], 1.0),
                slack_std / max(self.norms["max_slack"], 1.0),
                energy_mean / max(self.norms["max_energy"], 1.0),
                energy_std / max(self.norms["max_energy"], 1.0),
                priority_mean / 5.0,
                gini,
            ]
        )

        state = np.asarray(features, dtype=np.float32)
        return state, mask.astype(bool)

    def _select_candidate_jobs(self) -> Tuple[List[JobRecord], List[bool]]:
        lookahead = self.config.environment.lookahead_horizon
        available: List[JobRecord] = []
        upcoming: List[JobRecord] = []
        for job in self.pending_jobs:
            if job.arrival_time <= self.current_time + 1e-6:
                available.append(job)
            elif job.arrival_time <= self.current_time + lookahead:
                upcoming.append(job)

        available.sort(key=lambda job: (-job.priority, job.due_time))
        upcoming.sort(key=lambda job: (job.arrival_time, job.due_time))
        candidates = (available + upcoming)[: self.action_size]
        valid_flags = [job.arrival_time <= self.current_time + 1e-6 for job in candidates]
        return candidates, valid_flags

    def _encode_job(self, job: JobRecord, available: bool) -> List[float]:
        slack = job.due_time - max(self.current_time, job.arrival_time) - job.processing_time
        machine_idx = self.machine_to_index.get(job.machine_group, 0)
        machine_norm = machine_idx / max(1, len(self.machine_to_index) - 1)
        return [
            job.processing_time / max(self.norms["max_processing"], 1.0),
            job.due_time / max(self.norms["max_due"], 1.0),
            job.arrival_time / max(self.norms["max_arrival"], 1.0),
            job.energy_cost / max(self.norms["max_energy"], 1.0),
            job.priority / 5.0,
            job.setup_time / max(self.norms["max_setup"], 0.1),
            job.reconfiguration_cost / max(self.norms["max_reconfig"], 0.1),
            slack / max(self.norms["max_slack"], 1.0),
            machine_norm,
            1.0 if available else 0.0,
        ]

    def _pending_ratio(self, machine: str) -> float:
        total = len(self.pending_jobs)
        if total == 0:
            return 0.0
        matching = sum(1 for job in self.pending_jobs if job.machine_group == machine)
        return matching / total

    def _compute_machine_utilization(self) -> Dict[str, float]:
        horizon = max(self.config.environment.horizon_hours, 1.0)
        return {
            machine: min(1.0, busy / horizon) for machine, busy in self.machine_busy_time.items()
        }

    @staticmethod
    def _gini(values: Sequence[float]) -> float:
        arr = np.asarray(values, dtype=np.float64)
        if np.allclose(arr, 0):
            return 0.0
        arr = np.abs(arr) + 1e-9
        arr = np.sort(arr)
        index = np.arange(1, arr.shape[0] + 1)
        n = arr.shape[0]
        return (np.sum((2 * index - n - 1) * arr)) / (n * np.sum(arr))

    def _build_info(
        self,
        job: Optional[JobRecord],
        reward_components: Dict[str, float],
        valid_mask: np.ndarray,
        tardiness: float = 0.0,
        energy: float = 0.0,
        slack_after: float = 0.0,
        idle_time: float = 0.0,
    ) -> Dict[str, np.ndarray]:
        info = {
            "reward_components": reward_components,
            "valid_action_mask": valid_mask.astype(bool),
            "pending_jobs": len(self.pending_jobs),
            "completed_jobs": self.completed_jobs,
            "tardiness": tardiness,
            "energy": energy,
            "slack": slack_after,
            "idle_time": idle_time,
            "backlog_ratio": len(self.pending_jobs) / max(1, self.total_jobs),
            "gini_workload": self.gini_trace[-1] if self.gini_trace else 0.0,
            "machine_energy": self.machine_energy.copy(),
            "machine_utilization": self._compute_machine_utilization(),
            "reconfigurations": self.reconfigurations,
            "invalid_actions": self.invalid_actions,
            "candidate_jobs": [j.job_id if j else None for j in self.candidate_jobs],
        }
        if job is not None:
            info.update(
                {
                    "job_id": job.job_id,
                    "machine": job.machine_group,
                    "product_family": job.product_family,
                }
            )
        return info

    def get_episode_metrics(self) -> Dict[str, float]:
        completed = max(1, self.completed_jobs)
        return {
            "completion_rate": self.completed_jobs / max(1, self.total_jobs),
            "avg_tardiness": self.total_tardiness / completed,
            "avg_energy": self.total_energy / completed,
            "avg_setup_time": self.total_setup / completed,
            "avg_idle_time": self.total_idle / completed,
            "reconfigurations": self.reconfigurations,
            "invalid_actions": self.invalid_actions,
            "reward_components": dict(self.reward_component_totals),
            "machine_energy": self.machine_energy.copy(),
            "machine_utilization": self._compute_machine_utilization(),
            "slack_mean": np.mean(self.slack_samples) if self.slack_samples else 0.0,
            "slack_std": np.std(self.slack_samples) if self.slack_samples else 0.0,
            "backlog_trace": list(self.backlog_trace),
            "gini_trace": list(self.gini_trace),
            "schedule": list(self.schedule_log),
        }


# ---------------------------------------------------------------------------
# DQN agent
# ---------------------------------------------------------------------------


class RMSDQNAgent:
    """Double DQN agent with enhanced exploration and diagnostics."""

    def __init__(self, config=CONFIG, state_dim: Optional[int] = None, action_dim: Optional[int] = None):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.random = np.random.default_rng(config.synthetic.random_seed)

        self.state_dim = state_dim or 1
        self.action_dim = action_dim or config.environment.candidate_pool

        self.policy_net = RMSDuelingNetwork(self.state_dim, self.action_dim, config).to(self.device)
        self.target_net = RMSDuelingNetwork(self.state_dim, self.action_dim, config).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.AdamW(
            self.policy_net.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=20, T_mult=2
        )

        self.replay_buffer = PrioritizedReplayBuffer(config.training.replay_capacity)
        self.training_steps = 0
        self.epsilon = config.exploration.epsilon_start
        self.temperature = config.exploration.initial_temperature
        self.beta = 0.6
        total_updates = max(1, config.training.episodes * config.training.max_steps_per_episode)
        self.beta_increment = (1.0 - self.beta) / total_updates

        self.debug_history: Dict[str, List] = {
            "episode_rewards": [],
            "loss": [],
            "epsilon": [],
            "temperature": [],
            "q_values": [],
            "td_errors": [],
            "grad_norms": [],
            "learning_rates": [],
            "buffer_size": [],
            "action_modes": [],
            "step_rewards": [],
        }

    # ------------------------------------------------------------------
    def train(self, env: RMSSchedulingEnvironment, scenarios: Sequence[Scenario]) -> List[Dict[str, float]]:
        history: List[Dict[str, float]] = []
        for episode in range(self.config.training.episodes):
            scenario = random.choice(scenarios)
            state, mask = env.reset(scenario)
            episode_reward = 0.0
            episode_losses: List[float] = []
            action_counter: Counter = Counter()
            q_max_trace: List[float] = []
            q_mean_trace: List[float] = []
            q_std_trace: List[float] = []
            softmax_count = 0
            step = 0

            while True:
                action, q_values, mode = self._select_action(state, mask, training=True)
                action_counter[action] += 1
                q_max_trace.append(float(np.max(q_values)))
                q_mean_trace.append(float(np.mean(q_values)))
                q_std_trace.append(float(np.std(q_values)))
                if mode == "boltzmann":
                    softmax_count += 1

                next_state, reward, done, info = env.step(action)
                episode_reward += reward

                self.debug_history["step_rewards"].append(reward)
                self.debug_history["action_modes"].append(mode)
                self.debug_history["q_values"].append(q_values.tolist())
                self.debug_history["buffer_size"].append(len(self.replay_buffer))

                self.replay_buffer.push(
                    state,
                    action,
                    reward,
                    next_state,
                    done,
                    info["valid_action_mask"].astype(np.float32),
                )

                loss_value = self._optimise_model()
                if loss_value is not None:
                    episode_losses.append(loss_value)

                state, mask = next_state, info["valid_action_mask"]
                step += 1
                if done:
                    break

            metrics = env.get_episode_metrics()
            mean_loss = float(np.mean(episode_losses)) if episode_losses else 0.0

            self.debug_history["episode_rewards"].append(episode_reward)
            self.debug_history["loss"].append(mean_loss)
            self.debug_history["epsilon"].append(self.epsilon)
            self.debug_history["temperature"].append(self.temperature)

            history.append(
                {
                    "episode": episode + 1,
                    "reward": episode_reward,
                    "loss": mean_loss,
                    "epsilon": self.epsilon,
                    "temperature": self.temperature,
                    "steps": step,
                    "q_max": float(np.mean(q_max_trace)) if q_max_trace else 0.0,
                    "q_mean": float(np.mean(q_mean_trace)) if q_mean_trace else 0.0,
                    "q_std": float(np.mean(q_std_trace)) if q_std_trace else 0.0,
                    "softmax_usage": softmax_count / max(1, step),
                    "action_counts": dict(action_counter),
                    "scenario_id": scenario.scenario_id,
                    "slack_samples": list(env.slack_samples),
                    **metrics,
                }
            )

            if (episode + 1) % self.config.exploration.epsilon_restart_interval == 0:
                self.epsilon = max(
                    self.config.exploration.epsilon_end,
                    self.epsilon * self.config.exploration.epsilon_restart_scale,
                )

        return history

    # ------------------------------------------------------------------
    def evaluate(
        self, env: RMSSchedulingEnvironment, scenarios: Sequence[Scenario]
    ) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
        self.policy_net.eval()
        with torch.no_grad():
            results: List[Dict[str, float]] = []
            reward_accumulator = Counter()
            for scenario in scenarios:
                state, mask = env.reset(scenario, deterministic=True)
                while True:
                    action, _, _ = self._select_action(state, mask, training=False)
                    next_state, reward, done, info = env.step(action)
                    state, mask = next_state, info["valid_action_mask"]
                    if done:
                        break
                metrics = env.get_episode_metrics()
                metrics.update({"scenario_id": scenario.scenario_id})
                results.append(metrics)
                reward_accumulator.update(metrics["reward_components"])

            summary = {
                "episodes": len(results),
                "avg_completion_rate": float(np.mean([r["completion_rate"] for r in results])),
                "avg_tardiness": float(np.mean([r["avg_tardiness"] for r in results])),
                "avg_energy": float(np.mean([r["avg_energy"] for r in results])),
                "avg_setup": float(np.mean([r["avg_setup_time"] for r in results])),
                "avg_idle": float(np.mean([r["avg_idle_time"] for r in results])),
                "reward_components": dict(reward_accumulator),
            }
        self.policy_net.train()
        return summary, results

    # ------------------------------------------------------------------
    def _select_action(self, state: np.ndarray, valid_mask: np.ndarray, training: bool) -> Tuple[int, np.ndarray, str]:
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor).cpu().numpy()[0]
        q_values = q_values.copy()
        q_values += self.random.normal(0.0, self.config.exploration.noise_std, size=q_values.shape) if training else 0.0
        q_values[~valid_mask] = -1e9
        valid_indices = np.where(valid_mask)[0]
        explore_mode = "greedy"

        if len(valid_indices) == 0:
            action = int(np.argmax(q_values))
            return action, q_values, "forced"

        if training:
            if self.random.random() < self.epsilon:
                action = int(self.random.choice(valid_indices))
                explore_mode = "epsilon"
            else:
                if self.random.random() < self.config.exploration.boltzmann_ratio:
                    logits = q_values[valid_indices] / max(self.temperature, 1e-3)
                    logits = logits - logits.max()
                    probs = np.exp(logits)
                    probs /= probs.sum()
                    action = int(self.random.choice(valid_indices, p=probs))
                    explore_mode = "boltzmann"
                else:
                    action = int(valid_indices[np.argmax(q_values[valid_indices])])
            self.epsilon = max(
                self.config.exploration.epsilon_end,
                self.epsilon * self.config.exploration.epsilon_decay,
            )
            self.temperature = max(
                self.config.exploration.min_temperature,
                self.temperature * self.config.exploration.temperature_decay,
            )
        else:
            action = int(valid_indices[np.argmax(q_values[valid_indices])])

        return action, q_values, explore_mode

    # ------------------------------------------------------------------
    def _optimise_model(self) -> Optional[float]:
        if len(self.replay_buffer) < self.config.training.min_buffer_size:
            return None
        batch_size = self.config.training.batch_size
        samples, indices, weights = self.replay_buffer.sample(batch_size, self.beta)

        states = torch.from_numpy(np.stack([sample[0] for sample in samples])).float().to(self.device)
        actions = torch.tensor([sample[1] for sample in samples], dtype=torch.long, device=self.device).unsqueeze(1)
        rewards = torch.tensor([sample[2] for sample in samples], dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states = torch.from_numpy(np.stack([sample[3] for sample in samples])).float().to(self.device)
        dones = torch.tensor([sample[4] for sample in samples], dtype=torch.float32, device=self.device).unsqueeze(1)
        next_masks = torch.from_numpy(np.stack([sample[5] for sample in samples])).bool().to(self.device)
        weights_tensor = torch.from_numpy(weights).float().unsqueeze(1).to(self.device)

        current_q = self.policy_net(states).gather(1, actions)

        with torch.no_grad():
            next_q_main = self.policy_net(next_states)
            next_q_main = next_q_main.masked_fill(~next_masks, -1e9)
            next_actions = next_q_main.argmax(dim=1, keepdim=True)
            next_q_target = self.target_net(next_states)
            next_q_target = next_q_target.masked_fill(~next_masks, -1e9)
            target_q = rewards + (1 - dones) * self.config.training.gamma * next_q_target.gather(1, next_actions)

        td_errors = target_q - current_q
        loss = F.smooth_l1_loss(current_q, target_q, reduction="none", beta=self.config.training.huber_delta)
        loss = (loss * weights_tensor).mean()

        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.policy_net.parameters(), self.config.training.max_grad_norm
        )
        self.optimizer.step()
        self.scheduler.step(self.training_steps / max(1, self.config.training.max_steps_per_episode))

        priorities = td_errors.detach().abs().cpu().numpy().flatten() + 1e-3
        self.replay_buffer.update_priorities(indices, priorities)

        self.beta = min(1.0, self.beta + self.beta_increment)
        self.training_steps += 1
        if self.training_steps % self.config.training.target_update_interval == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        self.debug_history["td_errors"].extend(priorities.tolist())
        self.debug_history["grad_norms"].append(float(grad_norm))
        self.debug_history["learning_rates"].append(self.optimizer.param_groups[0]["lr"])

        return float(loss.item())
