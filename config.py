"""Configuration settings for the RMS Deep Q-Learning project."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple


@dataclass
class PathConfig:
    """Container for filesystem paths used in the project."""

    base_dir: Path = Path(__file__).resolve().parent
    data_dir: Path = base_dir / "data"
    results_dir: Path = base_dir / "results"
    real_dataset: Path = data_dir / "rms_real_data.csv"


@dataclass
class TrainingConfig:
    """Hyperparameters that drive the learning process."""

    episodes: int = 120
    max_steps_per_episode: int = 180
    warmup_episodes: int = 5
    batch_size: int = 128
    gamma: float = 0.985
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    max_grad_norm: float = 1.5
    target_update_interval: int = 6
    replay_capacity: int = 100_000
    min_buffer_size: int = 2_000
    huber_delta: float = 1.0
    gradient_accumulation_steps: int = 1


@dataclass
class ExplorationConfig:
    """Parameters for balancing exploration and exploitation."""

    epsilon_start: float = 1.0
    epsilon_end: float = 0.04
    epsilon_decay: float = 0.992
    epsilon_restart_interval: int = 35
    epsilon_restart_scale: float = 0.65
    boltzmann_ratio: float = 0.35
    initial_temperature: float = 1.5
    min_temperature: float = 0.2
    temperature_decay: float = 0.995
    noise_std: float = 0.05


@dataclass
class NetworkConfig:
    """Definition of the neural network policy architecture."""

    hidden_units: Tuple[int, ...] = (320, 256, 192, 128)
    dropout: float = 0.15
    activation: str = "elu"
    dueling: bool = True
    layer_norm: bool = True


@dataclass
class RewardConfig:
    """Weights that shape the multi-objective reward."""

    completion_bonus: float = 12.0
    throughput_weight: float = 3.0
    on_time_bonus: float = 8.0
    tardiness_penalty: float = -0.8
    energy_penalty: float = -0.35
    setup_penalty: float = -0.25
    reconfiguration_penalty: float = -1.2
    idle_penalty: float = -0.2
    invalid_action_penalty: float = -5.0
    slack_bonus: float = 0.15
    energy_threshold: float = 6.0


@dataclass
class EnvironmentConfig:
    """Parameters that describe the RMS environment."""

    candidate_pool: int = 12
    lookahead_horizon: float = 1.5
    horizon_hours: float = 32.0
    max_machines: int = 10
    setup_time_mean: float = 0.25
    setup_time_std: float = 0.08
    min_batch_size: int = 40


@dataclass
class SyntheticConfig:
    """Controls generation of additional RMS scenarios."""

    scenarios: int = 30
    jobs_per_scenario: Tuple[int, int] = (60, 140)
    random_seed: int = 42
    arrival_rate: float = 0.65
    high_priority_share: float = 0.25
    rush_job_share: float = 0.18
    energy_profile: Tuple[float, float] = (4.0, 9.5)


@dataclass
class VisualizationConfig:
    """Settings controlling diagnostics and figure generation."""

    dpi: int = 180
    style: str = "darkgrid"
    figure_size: Tuple[int, int] = (10, 6)


@dataclass
class Config:
    """Aggregated configuration wrapper used throughout the project."""

    paths: PathConfig = field(default_factory=PathConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    exploration: ExplorationConfig = field(default_factory=ExplorationConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    synthetic: SyntheticConfig = field(default_factory=SyntheticConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)


CONFIG = Config()

# Ensure default directories exist to avoid runtime surprises.
CONFIG.paths.data_dir.mkdir(parents=True, exist_ok=True)
CONFIG.paths.results_dir.mkdir(parents=True, exist_ok=True)
