"""Data ingestion, cleaning, and synthetic scenario generation for RMS DQN."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

from config import CONFIG


@dataclass
class JobRecord:
    """Canonical representation of a job in the RMS scheduling problem."""

    job_id: int
    product_family: str
    machine_group: str
    processing_time: float
    arrival_time: float
    due_time: float
    energy_cost: float
    priority: int
    setup_time: float
    reconfiguration_cost: float


@dataclass
class Scenario:
    """A collection of jobs that share the same production horizon."""

    scenario_id: str
    jobs: List[JobRecord]


@dataclass
class DatasetBundle:
    """All artefacts derived from the real and synthetic data sources."""

    train_scenarios: List[Scenario]
    test_scenarios: List[Scenario]
    normalization_stats: Dict[str, float]
    machine_groups: List[str]
    product_families: List[str]
    real_dataframe: pd.DataFrame
    synthetic_scenarios: List[Scenario]


class RMSDataManager:
    """Prepare and enrich RMS scheduling data for the learning pipeline."""

    def __init__(self, config=CONFIG):
        self.config = config
        self.random_state = np.random.default_rng(config.synthetic.random_seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def prepare_datasets(self) -> DatasetBundle:
        """Load the real CSV file, craft synthetic scenarios, and build splits."""

        raw_real = self._load_real_dataset()
        cleaned_real = self._clean_dataframe(raw_real)
        engineered_real = self._engineer_features(cleaned_real)

        synthetic_scenarios = self._generate_synthetic_scenarios(engineered_real)

        all_scenarios = [Scenario("real-001", self._dataframe_to_jobs(engineered_real))]
        all_scenarios.extend(synthetic_scenarios)
        self.random_state.shuffle(all_scenarios)

        split_index = int(len(all_scenarios) * 0.8)
        train = all_scenarios[:split_index]
        test = all_scenarios[split_index:] or all_scenarios[-2:]

        normalization = self._compute_normalization_stats(all_scenarios)
        machines = self._extract_unique(engineered_real, [s for s in synthetic_scenarios])
        families = sorted(engineered_real["product_family"].unique().tolist())

        return DatasetBundle(
            train_scenarios=train,
            test_scenarios=test,
            normalization_stats=normalization,
            machine_groups=machines,
            product_families=families,
            real_dataframe=engineered_real,
            synthetic_scenarios=synthetic_scenarios,
        )

    # ------------------------------------------------------------------
    # Real data utilities
    # ------------------------------------------------------------------
    def _load_real_dataset(self) -> pd.DataFrame:
        path = self.config.paths.real_dataset
        if not path.exists():
            raise FileNotFoundError(
                f"Expected real dataset at {path}, but the file was not found."
            )
        df = pd.read_csv(path)
        df.columns = [c.strip().lower() for c in df.columns]
        required = {
            "job_id",
            "product_family",
            "machine_group",
            "processing_time",
            "arrival_time",
            "due_time",
            "energy_cost",
            "priority",
            "setup_time",
            "reconfiguration_cost",
        }
        missing = required.difference(set(df.columns))
        if missing:
            raise ValueError(
                "The real dataset is missing expected columns: " + ", ".join(sorted(missing))
            )
        return df

    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        numeric_cols = [
            "processing_time",
            "arrival_time",
            "due_time",
            "energy_cost",
            "priority",
            "setup_time",
            "reconfiguration_cost",
        ]
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
        df = df.dropna(subset=numeric_cols)
        df = df.sort_values("arrival_time").reset_index(drop=True)

        # Ensure due dates are always after arrival + processing.
        min_due = df["arrival_time"] + df["processing_time"] + 0.5
        df["due_time"] = np.maximum(df["due_time"], min_due)

        df["priority"] = df["priority"].clip(lower=1, upper=5).astype(int)
        df["setup_time"] = df["setup_time"].clip(lower=0.01)
        df["reconfiguration_cost"] = df["reconfiguration_cost"].clip(lower=0.1)
        return df

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["processing_density"] = df["processing_time"] / (
            df["due_time"] - df["arrival_time"]
        )
        df["slack_time"] = df["due_time"] - df["arrival_time"] - df["processing_time"]
        df["critical_ratio"] = (
            (df["due_time"] - df["arrival_time"]) / df["processing_time"]
        ).clip(upper=10.0)
        df["energy_intensity"] = df["energy_cost"] / df["processing_time"]
        df["urgency_flag"] = (df["slack_time"] <= 2.0).astype(int)
        return df

    # ------------------------------------------------------------------
    # Synthetic data
    # ------------------------------------------------------------------
    def _generate_synthetic_scenarios(self, real_df: pd.DataFrame) -> List[Scenario]:
        config = self.config.synthetic
        scenarios: List[Scenario] = []

        processing_mu = max(real_df["processing_time"].mean(), 0.1)
        processing_sigma = real_df["processing_time"].std(ddof=0) or 1.0
        energy_mu = max(real_df["energy_cost"].mean(), 0.1)
        energy_sigma = real_df["energy_cost"].std(ddof=0) or 1.0
        setup_mu = max(real_df["setup_time"].mean(), 0.05)
        setup_sigma = real_df["setup_time"].std(ddof=0) or 0.1
        reconfig_mu = max(real_df["reconfiguration_cost"].mean(), 0.05)
        reconfig_sigma = real_df["reconfiguration_cost"].std(ddof=0) or 0.2

        machine_groups = real_df["machine_group"].unique().tolist()
        product_families = real_df["product_family"].unique().tolist()

        for idx in range(config.scenarios):
            n_jobs = int(self.random_state.integers(*config.jobs_per_scenario))
            jobs: List[JobRecord] = []
            current_time = 0.0

            for job_index in range(n_jobs):
                product = self.random_state.choice(product_families)
                machine = self.random_state.choice(machine_groups)
                processing = max(
                    0.3,
                    float(
                        self.random_state.lognormal(
                            mean=math.log(processing_mu), sigma=0.35
                        )
                    ),
                )
                inter_arrival = self.random_state.exponential(1.0 / config.arrival_rate)
                arrival = current_time + inter_arrival
                slack = max(
                    1.0,
                    float(self.random_state.normal(loc=5.0, scale=2.0)),
                )
                due = arrival + processing + slack
                energy = float(
                    np.clip(
                        self.random_state.normal(energy_mu, energy_sigma),
                        config.energy_profile[0],
                        config.energy_profile[1],
                    )
                )
                priority = int(
                    np.clip(
                        self.random_state.choice(
                            [1, 2, 3, 4, 5],
                            p=[
                                1 - config.high_priority_share - config.rush_job_share,
                                0.25,
                                0.22,
                                config.high_priority_share,
                                config.rush_job_share,
                            ],
                        ),
                        1,
                        5,
                    )
                )
                setup = max(0.05, float(self.random_state.normal(setup_mu, setup_sigma)))
                reconfig = max(
                    0.15,
                    float(self.random_state.normal(reconfig_mu, reconfig_sigma)),
                )

                jobs.append(
                    JobRecord(
                        job_id=job_index + 1,
                        product_family=str(product),
                        machine_group=str(machine),
                        processing_time=float(processing),
                        arrival_time=float(arrival),
                        due_time=float(due),
                        energy_cost=float(energy),
                        priority=priority,
                        setup_time=float(setup),
                        reconfiguration_cost=float(reconfig),
                    )
                )
                current_time = arrival

            scenarios.append(
                Scenario(scenario_id=f"synthetic-{idx:03d}", jobs=jobs)
            )

        return scenarios

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _dataframe_to_jobs(self, df: pd.DataFrame) -> List[JobRecord]:
        return [
            JobRecord(
                job_id=int(row.job_id),
                product_family=str(row.product_family),
                machine_group=str(row.machine_group),
                processing_time=float(row.processing_time),
                arrival_time=float(row.arrival_time),
                due_time=float(row.due_time),
                energy_cost=float(row.energy_cost),
                priority=int(row.priority),
                setup_time=float(row.setup_time),
                reconfiguration_cost=float(row.reconfiguration_cost),
            )
            for row in df.itertuples(index=False)
        ]

    def _compute_normalization_stats(self, scenarios: Sequence[Scenario]) -> Dict[str, float]:
        processing = []
        due = []
        energy = []
        setup = []
        reconfig = []
        slack = []
        arrival = []

        for scenario in scenarios:
            for job in scenario.jobs:
                processing.append(job.processing_time)
                due.append(job.due_time)
                energy.append(job.energy_cost)
                setup.append(job.setup_time)
                reconfig.append(job.reconfiguration_cost)
                slack.append(job.due_time - job.arrival_time - job.processing_time)
                arrival.append(job.arrival_time)

        def _safe_max(values: Iterable[float], default: float) -> float:
            return float(max(max(values), default)) if values else default

        return {
            "max_processing": _safe_max(processing, 1.0),
            "max_due": _safe_max(due, self.config.environment.horizon_hours),
            "max_energy": _safe_max(energy, 1.0),
            "max_setup": _safe_max(setup, 0.1),
            "max_reconfig": _safe_max(reconfig, 0.1),
            "max_slack": _safe_max(slack, 1.0),
            "max_arrival": _safe_max(arrival, 1.0),
        }

    def _extract_unique(self, real_df: pd.DataFrame, synthetic: Sequence[Scenario]) -> List[str]:
        machines = set(real_df["machine_group"].tolist())
        for scenario in synthetic:
            machines.update(job.machine_group for job in scenario.jobs)
        machines = sorted(machines)
        max_machines = self.config.environment.max_machines
        return machines[:max_machines]


def prepare_data_bundle(config=CONFIG) -> DatasetBundle:
    """Convenience wrapper used by the main entrypoint."""

    manager = RMSDataManager(config)
    return manager.prepare_datasets()
