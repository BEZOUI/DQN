"""Visualization utilities for analysing RMS DQN training and evaluation."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from config import CONFIG


class RMSVisualizer:
    """Generate diagnostic plots that illuminate agent behaviour."""

    def __init__(self, config=CONFIG):
        self.config = config
        self.results_dir = Path(config.paths.results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        sns.set_theme(style=config.visualization.style)

    # ------------------------------------------------------------------
    def generate_all(
        self,
        training_history: Sequence[Dict],
        evaluation_summary: Dict,
        evaluation_records: Sequence[Dict],
        debug_history: Dict[str, List],
        dataset_bundle,
    ) -> None:
        """Produce the complete suite of 30 analysis figures."""

        if not training_history:
            return
        history_df = pd.DataFrame(training_history)
        history_df.sort_values("episode", inplace=True)
        component_df = pd.DataFrame([
            pd.Series(record.get("reward_components", {})) for record in training_history
        ]).fillna(0.0)
        component_df.index = history_df["episode"].values

        figure_builders = [
            self._plot_episode_rewards,
            self._plot_reward_moving_average,
            self._plot_loss_curve,
            self._plot_epsilon_decay,
            self._plot_episode_lengths,
            self._plot_completion_rate,
            self._plot_avg_tardiness,
            self._plot_avg_energy,
            self._plot_reconfigurations,
            self._plot_invalid_actions,
            lambda *args: self._plot_reward_components(component_df),
            self._plot_action_distribution,
            self._plot_q_statistics,
            self._plot_q_histogram,
            self._plot_td_errors,
            self._plot_buffer_occupancy,
            self._plot_softmax_usage,
            self._plot_temperature_schedule,
            self._plot_slack_distribution,
            self._plot_backlog_profile,
            self._plot_machine_utilization_heatmap,
            self._plot_machine_energy_heatmap,
            self._plot_gini_trace,
            self._plot_gradient_norms,
            self._plot_learning_rate,
            self._plot_cumulative_reward,
            self._plot_step_reward_distribution,
            self._plot_evaluation_summary,
            self._plot_schedule_gantt,
            self._plot_priority_tardiness_scatter,
        ]

        for index, builder in enumerate(figure_builders, start=1):
            try:
                builder(history_df, evaluation_summary, evaluation_records, debug_history, dataset_bundle)
            except Exception as exc:  # pragma: no cover - defensive logging
                print(f"[visualization] Skipped plot {index:02d}: {exc}")

    # ------------------------------------------------------------------
    def _save_current_figure(self, name: str) -> None:
        path = self.results_dir / name
        plt.tight_layout()
        plt.savefig(path, dpi=self.config.visualization.dpi)
        plt.close()

    # Individual plot implementations ---------------------------------
    def _plot_episode_rewards(self, history_df, *_):
        plt.figure(figsize=self.config.visualization.figure_size)
        sns.lineplot(data=history_df, x="episode", y="reward", marker="o")
        plt.title("Episode reward progression")
        plt.xlabel("Episode")
        plt.ylabel("Total reward")
        self._save_current_figure("01_episode_rewards.png")

    def _plot_reward_moving_average(self, history_df, *_):
        plt.figure(figsize=self.config.visualization.figure_size)
        window = max(5, int(len(history_df) * 0.1))
        rolling = history_df["reward"].rolling(window=window, min_periods=1).mean()
        sns.lineplot(x=history_df["episode"], y=rolling)
        plt.title(f"Reward moving average (window={window})")
        plt.xlabel("Episode")
        plt.ylabel("Reward (moving average)")
        self._save_current_figure("02_reward_moving_average.png")

    def _plot_loss_curve(self, history_df, *_):
        plt.figure(figsize=self.config.visualization.figure_size)
        sns.lineplot(data=history_df, x="episode", y="loss", marker="o", color="#E76F51")
        plt.title("Training loss per episode")
        plt.xlabel("Episode")
        plt.ylabel("Huber loss")
        self._save_current_figure("03_loss_curve.png")

    def _plot_epsilon_decay(self, history_df, *_):
        plt.figure(figsize=self.config.visualization.figure_size)
        sns.lineplot(data=history_df, x="episode", y="epsilon", marker="o")
        plt.title("ε-greedy exploration schedule")
        plt.xlabel("Episode")
        plt.ylabel("Epsilon")
        self._save_current_figure("04_epsilon_decay.png")

    def _plot_episode_lengths(self, history_df, *_):
        plt.figure(figsize=self.config.visualization.figure_size)
        sns.lineplot(data=history_df, x="episode", y="steps", marker="o")
        plt.title("Episode durations")
        plt.xlabel("Episode")
        plt.ylabel("Steps")
        self._save_current_figure("05_episode_lengths.png")

    def _plot_completion_rate(self, history_df, *_):
        plt.figure(figsize=self.config.visualization.figure_size)
        sns.lineplot(data=history_df, x="episode", y="completion_rate", marker="o", color="#264653")
        plt.title("Completion rate per episode")
        plt.xlabel("Episode")
        plt.ylabel("Completion rate")
        self._save_current_figure("06_completion_rate.png")

    def _plot_avg_tardiness(self, history_df, *_):
        plt.figure(figsize=self.config.visualization.figure_size)
        sns.lineplot(data=history_df, x="episode", y="avg_tardiness", marker="o", color="#2A9D8F")
        plt.title("Average tardiness per episode")
        plt.xlabel("Episode")
        plt.ylabel("Tardiness (hours)")
        self._save_current_figure("07_avg_tardiness.png")

    def _plot_avg_energy(self, history_df, *_):
        plt.figure(figsize=self.config.visualization.figure_size)
        sns.lineplot(data=history_df, x="episode", y="avg_energy", marker="o", color="#F4A261")
        plt.title("Average energy consumption")
        plt.xlabel("Episode")
        plt.ylabel("Energy units")
        self._save_current_figure("08_avg_energy.png")

    def _plot_reconfigurations(self, history_df, *_):
        plt.figure(figsize=self.config.visualization.figure_size)
        sns.lineplot(data=history_df, x="episode", y="reconfigurations", marker="o", color="#E9C46A")
        plt.title("Reconfiguration counts per episode")
        plt.xlabel("Episode")
        plt.ylabel("Reconfigurations")
        self._save_current_figure("09_reconfiguration_counts.png")

    def _plot_invalid_actions(self, history_df, *_):
        plt.figure(figsize=self.config.visualization.figure_size)
        sns.lineplot(data=history_df, x="episode", y="invalid_actions", marker="o", color="#E63946")
        plt.title("Invalid action frequency")
        plt.xlabel("Episode")
        plt.ylabel("Invalid actions")
        self._save_current_figure("10_invalid_actions.png")

    def _plot_reward_components(self, component_df):
        if component_df.empty:
            return
        plt.figure(figsize=self.config.visualization.figure_size)
        component_df.plot.area(ax=plt.gca(), cmap="viridis")
        plt.title("Reward component contributions")
        plt.xlabel("Episode")
        plt.ylabel("Reward component value")
        self._save_current_figure("11_reward_components.png")

    def _plot_action_distribution(self, history_df, *_):
        counter = Counter()
        for record in history_df["action_counts"]:
            counter.update(record)
        if not counter:
            return
        actions, counts = zip(*sorted(counter.items()))
        plt.figure(figsize=self.config.visualization.figure_size)
        sns.barplot(x=list(map(str, actions)), y=np.array(counts) / max(1, sum(counts)))
        plt.title("Action selection distribution")
        plt.xlabel("Action index")
        plt.ylabel("Selection share")
        self._save_current_figure("12_action_distribution.png")

    def _plot_q_statistics(self, history_df, *_):
        plt.figure(figsize=self.config.visualization.figure_size)
        plt.plot(history_df["episode"], history_df["q_max"], label="Q max", marker="o")
        plt.plot(history_df["episode"], history_df["q_mean"], label="Q mean", marker="o")
        plt.plot(history_df["episode"], history_df["q_std"], label="Q std", marker="o")
        plt.legend()
        plt.title("Q-value statistics")
        plt.xlabel("Episode")
        plt.ylabel("Value")
        self._save_current_figure("13_q_statistics.png")

    def _plot_q_histogram(self, _history_df, _summary, _records, debug_history, *_):
        values = np.array(debug_history.get("q_values", []), dtype=object)
        if values.size == 0:
            return
        flat = np.concatenate(values).astype(float)
        plt.figure(figsize=self.config.visualization.figure_size)
        sns.histplot(flat, bins=40, kde=True, color="#457B9D")
        plt.title("Distribution of predicted Q-values")
        plt.xlabel("Q-value")
        plt.ylabel("Density")
        self._save_current_figure("14_q_histogram.png")

    def _plot_td_errors(self, _history_df, _summary, _records, debug_history, *_):
        errors = debug_history.get("td_errors", [])
        if not errors:
            return
        plt.figure(figsize=self.config.visualization.figure_size)
        sns.histplot(errors, bins=40, kde=True, color="#8E5E2E")
        plt.title("TD error distribution")
        plt.xlabel("|TD error|")
        plt.ylabel("Frequency")
        self._save_current_figure("15_td_error_distribution.png")

    def _plot_buffer_occupancy(self, _history_df, _summary, _records, debug_history, *_):
        occupancy = debug_history.get("buffer_size", [])
        if not occupancy:
            return
        plt.figure(figsize=self.config.visualization.figure_size)
        sns.lineplot(x=range(len(occupancy)), y=occupancy)
        plt.title("Replay buffer fill level")
        plt.xlabel("Update step")
        plt.ylabel("Buffer size")
        self._save_current_figure("16_replay_buffer_occupancy.png")

    def _plot_softmax_usage(self, history_df, *_):
        plt.figure(figsize=self.config.visualization.figure_size)
        sns.lineplot(data=history_df, x="episode", y="softmax_usage", marker="o")
        plt.title("Boltzmann exploration utilisation")
        plt.xlabel("Episode")
        plt.ylabel("Usage ratio")
        self._save_current_figure("17_softmax_usage.png")

    def _plot_temperature_schedule(self, history_df, *_):
        plt.figure(figsize=self.config.visualization.figure_size)
        sns.lineplot(data=history_df, x="episode", y="temperature", marker="o", color="#2F3E46")
        plt.title("Softmax temperature schedule")
        plt.xlabel("Episode")
        plt.ylabel("Temperature")
        self._save_current_figure("18_temperature_schedule.png")

    def _plot_slack_distribution(self, history_df, *_):
        slack_samples = [value for episode in history_df["slack_samples"] for value in episode]
        if not slack_samples:
            return
        plt.figure(figsize=self.config.visualization.figure_size)
        sns.boxplot(x=slack_samples, color="#2A9D8F")
        plt.title("Slack distribution across steps")
        plt.xlabel("Slack (hours)")
        self._save_current_figure("19_slack_distribution.png")

    def _plot_backlog_profile(self, history_df, *_):
        backlog_matrix = self._pad_sequences(history_df["backlog_trace"].tolist())
        plt.figure(figsize=self.config.visualization.figure_size)
        sns.heatmap(backlog_matrix, cmap="mako", cbar_kws={"label": "Backlog ratio"})
        plt.title("Backlog ratio evolution per episode")
        plt.xlabel("Step")
        plt.ylabel("Episode index")
        self._save_current_figure("20_backlog_profile.png")

    def _plot_machine_utilization_heatmap(self, history_df, _summary, _records, _debug, dataset_bundle):
        machines = dataset_bundle.machine_groups
        data = [
            [record.get(machine, 0.0) for machine in machines]
            for record in history_df["machine_utilization"]
        ]
        if not data:
            return
        plt.figure(figsize=(max(8, len(machines) * 0.8), 6))
        sns.heatmap(data, xticklabels=machines, yticklabels=history_df["episode"], cmap="crest")
        plt.title("Machine utilisation heatmap")
        plt.xlabel("Machine")
        plt.ylabel("Episode")
        self._save_current_figure("21_machine_utilization_heatmap.png")

    def _plot_machine_energy_heatmap(self, history_df, _summary, _records, _debug, dataset_bundle):
        machines = dataset_bundle.machine_groups
        data = [
            [record.get(machine, 0.0) for machine in machines]
            for record in history_df["machine_energy"]
        ]
        if not data:
            return
        plt.figure(figsize=(max(8, len(machines) * 0.8), 6))
        sns.heatmap(data, xticklabels=machines, yticklabels=history_df["episode"], cmap="rocket_r")
        plt.title("Machine energy consumption heatmap")
        plt.xlabel("Machine")
        plt.ylabel("Episode")
        self._save_current_figure("22_machine_energy_heatmap.png")

    def _plot_gini_trace(self, history_df, *_):
        plt.figure(figsize=self.config.visualization.figure_size)
        gini_matrix = self._pad_sequences(history_df["gini_trace"].tolist())
        sns.heatmap(gini_matrix, cmap="viridis", cbar_kws={"label": "Gini"})
        plt.title("Workload balance evolution")
        plt.xlabel("Step")
        plt.ylabel("Episode")
        self._save_current_figure("23_gini_trace.png")

    def _plot_gradient_norms(self, _history_df, _summary, _records, debug_history, *_):
        norms = debug_history.get("grad_norms", [])
        if not norms:
            return
        plt.figure(figsize=self.config.visualization.figure_size)
        sns.lineplot(x=range(len(norms)), y=norms)
        plt.title("Gradient norm trajectory")
        plt.xlabel("Update step")
        plt.ylabel("Gradient norm")
        self._save_current_figure("24_gradient_norms.png")

    def _plot_learning_rate(self, _history_df, _summary, _records, debug_history, *_):
        lrs = debug_history.get("learning_rates", [])
        if not lrs:
            return
        plt.figure(figsize=self.config.visualization.figure_size)
        sns.lineplot(x=range(len(lrs)), y=lrs)
        plt.title("Learning rate schedule")
        plt.xlabel("Update step")
        plt.ylabel("Learning rate")
        self._save_current_figure("25_learning_rate.png")

    def _plot_cumulative_reward(self, history_df, *_):
        plt.figure(figsize=self.config.visualization.figure_size)
        cumulative = history_df["reward"].cumsum()
        sns.lineplot(x=history_df["episode"], y=cumulative)
        plt.title("Cumulative reward")
        plt.xlabel("Episode")
        plt.ylabel("Cumulative reward")
        self._save_current_figure("26_cumulative_reward.png")

    def _plot_step_reward_distribution(self, _history_df, _summary, _records, debug_history, *_):
        rewards = debug_history.get("step_rewards", [])
        if not rewards:
            return
        plt.figure(figsize=self.config.visualization.figure_size)
        sns.violinplot(x=rewards, color="#6D597A")
        plt.title("Distribution of step rewards")
        plt.xlabel("Reward")
        self._save_current_figure("27_step_reward_violin.png")

    def _plot_evaluation_summary(self, _history_df, evaluation_summary, *_):
        if not evaluation_summary:
            return
        plt.figure(figsize=self.config.visualization.figure_size)
        keys = [
            "avg_completion_rate",
            "avg_tardiness",
            "avg_energy",
            "avg_setup",
            "avg_idle",
        ]
        values = [evaluation_summary.get(key, 0.0) for key in keys]
        sns.barplot(x=keys, y=values, palette="muted")
        plt.xticks(rotation=30, ha="right")
        plt.title("Evaluation summary metrics")
        self._save_current_figure("28_evaluation_summary.png")

    def _plot_schedule_gantt(self, _history_df, _summary, evaluation_records, *_):
        if not evaluation_records:
            return
        schedule = evaluation_records[0]["schedule"]
        if not schedule:
            return
        plt.figure(figsize=(12, 6))
        machines = sorted({entry["machine"] for entry in schedule})
        machine_to_y = {machine: idx for idx, machine in enumerate(machines)}
        for entry in schedule:
            y = machine_to_y[entry["machine"]]
            plt.barh(
                y=y,
                width=entry["finish_time"] - entry["start_time"],
                left=entry["start_time"],
                height=0.6,
                color="#90BE6D" if not entry["requires_reconfig"] else "#F94144",
            )
        plt.yticks(range(len(machines)), machines)
        plt.xlabel("Time (hours)")
        plt.ylabel("Machine")
        plt.title("Sample evaluation schedule")
        self._save_current_figure("29_schedule_gantt.png")

    def _plot_priority_tardiness_scatter(self, _history_df, _summary, evaluation_records, *_):
        if not evaluation_records:
            return
        data = []
        for record in evaluation_records:
            for entry in record.get("schedule", []):
                data.append((entry.get("product_family", "NA"), entry.get("tardiness", 0.0)))
        if not data:
            return
        families, tardiness = zip(*data)
        plt.figure(figsize=self.config.visualization.figure_size)
        sns.stripplot(x=families, y=tardiness, color="#577590")
        plt.xticks(rotation=30, ha="right")
        plt.title("Tardiness by product family")
        plt.ylabel("Tardiness (hours)")
        self._save_current_figure("30_priority_tardiness_scatter.png")

    # ------------------------------------------------------------------
    @staticmethod
    def _pad_sequences(sequences: Iterable[Sequence[float]]) -> np.ndarray:
        seq_list = list(sequences)
        max_len = max((len(seq) for seq in seq_list), default=0)
        if max_len == 0:
            return np.zeros((len(seq_list), 0))
        matrix = np.zeros((len(seq_list), max_len))
        for idx, sequence in enumerate(seq_list):
            matrix[idx, : len(sequence)] = sequence
        return matrix
