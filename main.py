"""Entry point for training and evaluating the RMS Double DQN agent."""
from __future__ import annotations

import logging
import random
from typing import Sequence

import numpy as np
import torch

from config import CONFIG
from data_preprocessing import DatasetBundle, prepare_data_bundle
from dqn_agent import RMSSchedulingEnvironment, RMSDQNAgent
from visualization import RMSVisualizer


def _set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # pragma: no cover - depends on runtime
        torch.cuda.manual_seed_all(seed)


def _log_summary(training_history: Sequence[dict], evaluation_summary: dict) -> None:
    logging.info("Training episodes completed: %d", len(training_history))
    if training_history:
        best_episode = max(training_history, key=lambda record: record["reward"])
        logging.info(
            "Best episode %d | Reward %.2f | Completion %.2f%%",
            best_episode["episode"],
            best_episode["reward"],
            best_episode["completion_rate"] * 100,
        )
    if evaluation_summary:
        logging.info("Evaluation summary: %s", evaluation_summary)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    config = CONFIG
    _set_global_seed(config.synthetic.random_seed)

    logging.info("Preparing datasets from %s", config.paths.real_dataset)
    data_bundle: DatasetBundle = prepare_data_bundle(config)
    logging.info(
        "Loaded %d training scenarios and %d testing scenarios",
        len(data_bundle.train_scenarios),
        len(data_bundle.test_scenarios),
    )

    environment = RMSSchedulingEnvironment(
        config=config,
        normalization_stats=data_bundle.normalization_stats,
        machine_groups=data_bundle.machine_groups,
        product_families=data_bundle.product_families,
    )
    agent = RMSDQNAgent(config=config, state_dim=environment.state_dim, action_dim=environment.action_size)

    logging.info("Beginning training (%d episodes)", config.training.episodes)
    training_history = agent.train(environment, data_bundle.train_scenarios)

    logging.info("Evaluating agent on hold-out scenarios")
    evaluation_summary, evaluation_records = agent.evaluate(environment, data_bundle.test_scenarios)

    visualizer = RMSVisualizer(config)
    logging.info("Generating analytical visualisations in %s", visualizer.results_dir)
    visualizer.generate_all(training_history, evaluation_summary, evaluation_records, agent.debug_history, data_bundle)

    _log_summary(training_history, evaluation_summary)
    logging.info("Pipeline completed successfully. Results stored in %s", config.paths.results_dir)


if __name__ == "__main__":
    main()
