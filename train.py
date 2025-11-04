"""
Training and Evaluation Script for DQN Manufacturing System
Author: Professor of Operations Research & Deep Learning
Version: 3.0 - Production Implementation (NumPy-based)

This script orchestrates training, evaluation, and visualization.
Pure NumPy + Matplotlib implementation - guaranteed to work.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from datetime import datetime
from manufacturing_dqn import (
    DQNAgent, ManufacturingEnvironment, Job, generate_synthetic_jobs
)

# Set style
plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'default')
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10


def simple_progress_bar(current, total, prefix='', bar_length=50):
    """Simple progress bar without external dependencies"""
    percent = current / total
    filled_length = int(bar_length * percent)
    bar = '█' * filled_length + '-' * (bar_length - filled_length)
    print(f'\r{prefix} |{bar}| {current}/{total} ({percent*100:.1f}%)', end='', flush=True)
    if current == total:
        print()


class ManufacturingOptimizer:
    """
    Main orchestrator for DQN-based manufacturing optimization.
    Handles training, evaluation, and comprehensive visualization.
    """
    
    def __init__(
        self,
        n_jobs: int = 200,
        n_machines: int = 15,
        n_episodes: int = 150,
        max_steps: int = 250,
        test_split: float = 0.2,
        output_dir: str = "results"
    ):
        self.n_jobs = n_jobs
        self.n_machines = n_machines
        self.n_episodes = n_episodes
        self.max_steps = max_steps
        self.test_split = test_split
        self.output_dir = output_dir
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate or load data
        print("🚀 Deep Q-Learning for Reconfigurable Manufacturing Systems")
        print("=" * 80)
        print("Version 3.0 - Python Production Implementation\n")
        
        print("📂 Generating synthetic manufacturing data...")
        all_jobs = generate_synthetic_jobs(n_jobs, n_machines)
        
        # Split into train/test
        n_test = int(n_jobs * test_split)
        self.test_jobs = all_jobs[:n_test]
        self.train_jobs = all_jobs[n_test:]
        
        print(f"✓ Generated {n_jobs} jobs")
        print(f"📊 Training: {len(self.train_jobs)} | Testing: {len(self.test_jobs)}\n")
        
        # Initialize environment
        self.train_env = ManufacturingEnvironment(self.train_jobs, n_machines)
        self.test_env = ManufacturingEnvironment(self.test_jobs, n_machines)
        
        # Initialize agent
        state_size = len(self.train_env.get_state())
        action_size = len(self.train_jobs) + 1  # +1 for wait action
        
        self.agent = DQNAgent(
            state_size=state_size,
            action_size=action_size,
            learning_rate=0.001,
            gamma=0.99,
            epsilon_start=1.0,
            epsilon_end=0.01,
            epsilon_decay=0.995,
            buffer_size=10000,
            batch_size=64,
            target_update=100,
            max_grad_norm=10.0
        )
        
        # Training metrics
        self.episode_rewards = []
        self.episode_completions = []
        self.episode_losses = []
        self.episode_epsilons = []
        
    def train(self):
        """Train the DQN agent"""
        print("🎯 Training Phase...")
        print("Architecture: 4-layer DQN with gradient clipping (Pure NumPy)")
        print("Features: Double DQN, Experience Replay, ε-greedy exploration\n")
        
        for episode in range(self.n_episodes):
            state = self.train_env.reset(self.train_jobs)
            episode_reward = 0
            episode_loss = []
            
            for step in range(self.max_steps):
                # Get valid actions
                valid_actions = self.train_env.get_valid_actions()
                
                # Select action
                action = self.agent.act(state, valid_actions, training=True)
                
                # Execute action
                next_state, reward, done, info = self.train_env.step(action)
                
                # Store experience
                next_valid_actions = self.train_env.get_valid_actions()
                self.agent.remember(state, action, reward, next_state, done, next_valid_actions)
                
                # Train agent
                loss = self.agent.train_step()
                if loss > 0:
                    episode_loss.append(loss)
                
                episode_reward += reward
                state = next_state
                
                if done:
                    break
            
            # Track metrics
            kpis = self.train_env.get_kpis()
            self.episode_rewards.append(episode_reward)
            self.episode_completions.append(kpis['jobs_completed'])
            self.episode_losses.append(np.mean(episode_loss) if episode_loss else 0)
            self.episode_epsilons.append(self.agent.epsilon)
            
            # Progress report
            simple_progress_bar(episode + 1, self.n_episodes, prefix='Training')
            
            # Detailed report every 15 episodes
            if (episode + 1) % 15 == 0:
                avg_reward = np.mean(self.episode_rewards[-15:])
                completion_rate = kpis['completion_rate'] * 100
                print(f"\nEpisode {episode + 1}/{self.n_episodes} | "
                      f"Reward: {episode_reward:.1f} | "
                      f"Avg: {avg_reward:.1f} | "
                      f"Completed: {kpis['jobs_completed']} ({completion_rate:.1f}%) | "
                      f"ε: {self.agent.epsilon:.3f}")
        
        print("\n✓ Training complete!\n")
    
    def evaluate(self):
        """Evaluate trained agent on test set"""
        print("🧪 Evaluation Phase on Test Set...")
        print("=" * 70)
        
        state = self.test_env.reset(self.test_jobs)
        
        for step in range(self.max_steps):
            valid_actions = self.test_env.get_valid_actions()
            action = self.agent.act(state, valid_actions, training=False)
            next_state, reward, done, info = self.test_env.step(action)
            state = next_state
            
            if done:
                break
        
        # Get final KPIs
        kpis = self.test_env.get_kpis()
        
        print("EVALUATION RESULTS")
        print("=" * 70)
        print(f"Completed: {kpis['jobs_completed']} / {kpis['total_jobs']} "
              f"({kpis['completion_rate'] * 100:.1f}%)")
        print(f"Avg Energy: {kpis['avg_energy']:.2f} units/job")
        print(f"Makespan: {kpis['makespan']:.1f} minutes")
        print(f"Utilization: {kpis['utilization']:.1f}%")
        print(f"On-time Rate: {kpis['on_time_rate'] * 100:.1f}%")
        print(f"Total Reward: {kpis['total_reward']:.1f}")
        print("=" * 70)
        print()
        
        return kpis
    
    def visualize_training(self):
        """Create comprehensive training visualizations"""
        print("📊 Generating training visualizations...")
        
        # 1. Episode Rewards
        fig, ax = plt.subplots(figsize=(12, 6))
        episodes = np.arange(1, len(self.episode_rewards) + 1)
        ax.plot(episodes, self.episode_rewards, alpha=0.6, label='Episode Reward', linewidth=1)
        
        # Moving average
        window = 10
        if len(self.episode_rewards) >= window:
            moving_avg = np.convolve(self.episode_rewards, np.ones(window)/window, mode='valid')
            ax.plot(episodes[window-1:], moving_avg, 'r-', linewidth=2, label=f'{window}-Episode MA')
        
        ax.set_xlabel('Episode', fontsize=12, fontweight='bold')
        ax.set_ylabel('Total Reward', fontsize=12, fontweight='bold')
        ax.set_title('Training Rewards Over Episodes', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/01_training_rewards.png")
        plt.close()
        
        # 2. Job Completions
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(episodes, self.episode_completions, 'g-', linewidth=2)
        ax.fill_between(episodes, 0, self.episode_completions, alpha=0.3, color='green')
        ax.set_xlabel('Episode', fontsize=12, fontweight='bold')
        ax.set_ylabel('Jobs Completed', fontsize=12, fontweight='bold')
        ax.set_title('Job Completion Progress', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/02_job_completion.png")
        plt.close()
        
        # 3. Training Loss
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(episodes, self.episode_losses, 'orange', alpha=0.7, linewidth=1)
        if len(self.episode_losses) >= window:
            loss_ma = np.convolve(self.episode_losses, np.ones(window)/window, mode='valid')
            ax.plot(episodes[window-1:], loss_ma, 'r-', linewidth=2, label=f'{window}-Episode MA')
        ax.set_xlabel('Episode', fontsize=12, fontweight='bold')
        ax.set_ylabel('Loss (Huber)', fontsize=12, fontweight='bold')
        ax.set_title('Training Loss Convergence', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/03_training_loss.png")
        plt.close()
        
        # 4. Epsilon Decay
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(episodes, self.episode_epsilons, 'purple', linewidth=2)
        ax.fill_between(episodes, 0, self.episode_epsilons, alpha=0.3, color='purple')
        ax.set_xlabel('Episode', fontsize=12, fontweight='bold')
        ax.set_ylabel('Epsilon (ε)', fontsize=12, fontweight='bold')
        ax.set_title('Exploration Rate Decay', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/04_epsilon_decay.png")
        plt.close()
        
        # 5. Training Dashboard
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
        
        # Rewards
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(episodes, self.episode_rewards, alpha=0.6, linewidth=1)
        if len(self.episode_rewards) >= window:
            moving_avg = np.convolve(self.episode_rewards, np.ones(window)/window, mode='valid')
            ax1.plot(episodes[window-1:], moving_avg, 'r-', linewidth=2)
        ax1.set_xlabel('Episode', fontweight='bold')
        ax1.set_ylabel('Reward', fontweight='bold')
        ax1.set_title('Episode Rewards', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Completions
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(episodes, self.episode_completions, 'g-', linewidth=2)
        ax2.fill_between(episodes, 0, self.episode_completions, alpha=0.3, color='green')
        ax2.set_xlabel('Episode', fontweight='bold')
        ax2.set_ylabel('Jobs Completed', fontweight='bold')
        ax2.set_title('Job Completions', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Loss
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.plot(episodes, self.episode_losses, 'orange', alpha=0.7, linewidth=1)
        if len(self.episode_losses) >= window:
            loss_ma = np.convolve(self.episode_losses, np.ones(window)/window, mode='valid')
            ax3.plot(episodes[window-1:], loss_ma, 'r-', linewidth=2)
        ax3.set_xlabel('Episode', fontweight='bold')
        ax3.set_ylabel('Loss', fontweight='bold')
        ax3.set_title('Training Loss', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # Epsilon
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.plot(episodes, self.episode_epsilons, 'purple', linewidth=2)
        ax4.fill_between(episodes, 0, self.episode_epsilons, alpha=0.3, color='purple')
        ax4.set_xlabel('Episode', fontweight='bold')
        ax4.set_ylabel('Epsilon', fontweight='bold')
        ax4.set_title('Exploration Rate', fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        fig.suptitle('Training Dashboard', fontsize=16, fontweight='bold', y=0.995)
        plt.savefig(f"{self.output_dir}/05_training_dashboard.png")
        plt.close()
        
        print(f"✓ Training visualizations saved in {self.output_dir}/")
    
    def visualize_evaluation(self):
        """Create comprehensive evaluation visualizations"""
        print("📊 Generating evaluation visualizations...")
        
        completed_jobs = self.test_env.completed_jobs
        
        if len(completed_jobs) == 0:
            print("⚠️  No jobs completed in evaluation. Skipping visualizations.")
            return
        
        # 1. Energy Distribution
        fig, ax = plt.subplots(figsize=(12, 6))
        energies = [j.energy_consumption for j in completed_jobs]
        ax.hist(energies, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
        ax.axvline(np.mean(energies), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(energies):.2f}')
        ax.set_xlabel('Energy Consumption (units)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax.set_title('Energy Consumption Distribution', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/06_energy_distribution.png")
        plt.close()
        
        # 2. Processing Times
        fig, ax = plt.subplots(figsize=(12, 6))
        proc_times = [j.processing_time for j in completed_jobs]
        ax.hist(proc_times, bins=30, color='lightgreen', edgecolor='black', alpha=0.7)
        ax.axvline(np.mean(proc_times), color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: {np.mean(proc_times):.1f}')
        ax.set_xlabel('Processing Time (minutes)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax.set_title('Processing Time Distribution', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/07_processing_times.png")
        plt.close()
        
        # 3. Lateness Analysis
        fig, ax = plt.subplots(figsize=(12, 6))
        lateness = [j.lateness for j in completed_jobs]
        on_time = sum(1 for l in lateness if l == 0)
        late = len(lateness) - on_time
        
        ax.bar(['On-Time', 'Late'], [on_time, late], color=['green', 'red'], alpha=0.7, edgecolor='black')
        ax.set_ylabel('Number of Jobs', fontsize=12, fontweight='bold')
        ax.set_title(f'On-Time Delivery Performance ({on_time}/{len(lateness)} on-time)', 
                     fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/08_lateness.png")
        plt.close()
        
        # 4. Machine Workload
        fig, ax = plt.subplots(figsize=(12, 6))
        machine_jobs = self.test_env.machines_jobs_completed
        machines = np.arange(1, self.n_machines + 1)
        colors = plt.cm.viridis(np.linspace(0, 1, self.n_machines))
        ax.bar(machines, machine_jobs, color=colors, edgecolor='black', alpha=0.8)
        ax.set_xlabel('Machine ID', fontsize=12, fontweight='bold')
        ax.set_ylabel('Jobs Completed', fontsize=12, fontweight='bold')
        ax.set_title('Machine Workload Distribution', fontsize=14, fontweight='bold')
        ax.set_xticks(machines)
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/09_machine_workload.png")
        plt.close()
        
        # 5. Machine Energy Consumption
        fig, ax = plt.subplots(figsize=(12, 6))
        machine_energy = self.test_env.machines_energy
        ax.bar(machines, machine_energy, color=colors, edgecolor='black', alpha=0.8)
        ax.set_xlabel('Machine ID', fontsize=12, fontweight='bold')
        ax.set_ylabel('Total Energy (units)', fontsize=12, fontweight='bold')
        ax.set_title('Energy Consumption by Machine', fontsize=14, fontweight='bold')
        ax.set_xticks(machines)
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/10_machine_energy.png")
        plt.close()
        
        # 6. KPI Summary
        kpis = self.test_env.get_kpis()
        fig, ax = plt.subplots(figsize=(10, 8))
        
        kpi_names = ['Completion\nRate (%)', 'On-Time\nDelivery (%)', 
                     'Utilization (%)', 'Avg Energy\n(units/job)']
        kpi_values = [
            kpis['completion_rate'] * 100,
            kpis['on_time_rate'] * 100,
            kpis['utilization'],
            kpis['avg_energy'] * 10  # Scale for visibility
        ]
        
        colors_kpi = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12']
        bars = ax.barh(kpi_names, kpi_values, color=colors_kpi, edgecolor='black', alpha=0.8)
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, kpi_values)):
            if i == 3:  # Avg energy (scaled)
                label_val = kpis['avg_energy']
                ax.text(bar.get_width() + 2, bar.get_y() + bar.get_height()/2, 
                       f'{label_val:.2f}', va='center', fontweight='bold', fontsize=11)
            else:
                ax.text(bar.get_width() + 2, bar.get_y() + bar.get_height()/2, 
                       f'{val:.1f}%', va='center', fontweight='bold', fontsize=11)
        
        ax.set_xlabel('Value', fontsize=12, fontweight='bold')
        ax.set_title('Key Performance Indicators', fontsize=14, fontweight='bold')
        ax.set_xlim(0, max(kpi_values) * 1.15)
        ax.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/11_kpi_summary.png")
        plt.close()
        
        # 7. Evaluation Dashboard
        fig = plt.figure(figsize=(18, 12))
        gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.3)
        
        # Energy distribution
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.hist(energies, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
        ax1.axvline(np.mean(energies), color='red', linestyle='--', linewidth=2)
        ax1.set_xlabel('Energy (units)', fontweight='bold')
        ax1.set_ylabel('Frequency', fontweight='bold')
        ax1.set_title('Energy Distribution', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Processing times
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.hist(proc_times, bins=20, color='lightgreen', edgecolor='black', alpha=0.7)
        ax2.axvline(np.mean(proc_times), color='red', linestyle='--', linewidth=2)
        ax2.set_xlabel('Time (min)', fontweight='bold')
        ax2.set_ylabel('Frequency', fontweight='bold')
        ax2.set_title('Processing Times', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # On-time delivery
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.bar(['On-Time', 'Late'], [on_time, late], color=['green', 'red'], 
                alpha=0.7, edgecolor='black')
        ax3.set_ylabel('Jobs', fontweight='bold')
        ax3.set_title(f'Delivery Performance', fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Machine workload
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.bar(machines, machine_jobs, color=colors, edgecolor='black', alpha=0.8)
        ax4.set_xlabel('Machine', fontweight='bold')
        ax4.set_ylabel('Jobs', fontweight='bold')
        ax4.set_title('Machine Workload', fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')
        
        # Machine energy
        ax5 = fig.add_subplot(gs[2, 0])
        ax5.bar(machines, machine_energy, color=colors, edgecolor='black', alpha=0.8)
        ax5.set_xlabel('Machine', fontweight='bold')
        ax5.set_ylabel('Energy (units)', fontweight='bold')
        ax5.set_title('Energy by Machine', fontweight='bold')
        ax5.grid(True, alpha=0.3, axis='y')
        
        # KPIs
        ax6 = fig.add_subplot(gs[2, 1])
        kpi_display = ['Completion', 'On-Time', 'Utilization']
        kpi_display_values = kpi_values[:3]
        bars = ax6.barh(kpi_display, kpi_display_values, color=colors_kpi[:3], 
                        edgecolor='black', alpha=0.8)
        for bar, val in zip(bars, kpi_display_values):
            ax6.text(bar.get_width() + 2, bar.get_y() + bar.get_height()/2, 
                    f'{val:.1f}%', va='center', fontweight='bold')
        ax6.set_xlabel('Percentage', fontweight='bold')
        ax6.set_title('Key Metrics (%)', fontweight='bold')
        ax6.set_xlim(0, max(kpi_display_values) * 1.15)
        ax6.grid(True, alpha=0.3, axis='x')
        
        fig.suptitle('Evaluation Dashboard', fontsize=16, fontweight='bold', y=0.995)
        plt.savefig(f"{self.output_dir}/12_evaluation_dashboard.png")
        plt.close()
        
        print(f"✓ Evaluation visualizations saved in {self.output_dir}/")
    
    def run(self):
        """Execute complete optimization pipeline"""
        # Train
        self.train()
        
        # Evaluate
        kpis = self.evaluate()
        
        # Visualize
        self.visualize_training()
        self.visualize_evaluation()
        
        # Final summary
        print("\n" + "=" * 80)
        print("🏆 OPTIMIZATION COMPLETE - RESULTS SUMMARY")
        print("=" * 80)
        print(f"✓ Completion Rate:     {kpis['completion_rate'] * 100:.2f}%")
        print(f"✓ Average Energy:      {kpis['avg_energy']:.2f} units/job")
        print(f"✓ Makespan:            {kpis['makespan']:.2f} minutes")
        print(f"✓ Machine Utilization: {kpis['utilization']:.2f}%")
        print(f"✓ On-Time Delivery:    {kpis['on_time_rate'] * 100:.2f}%")
        print(f"✓ Total Reward:        {kpis['total_reward']:.1f}")
        print(f"✓ Jobs Completed:      {kpis['jobs_completed']} / {kpis['total_jobs']}")
        print("=" * 80)
        print(f"📁 Results saved in: {self.output_dir}/")
        print("=" * 80)
        
        # Save agent
        model_path = f"{self.output_dir}/dqn_agent.npy"
        self.agent.save(model_path)
        print(f"💾 Model saved: {model_path}\n")
        
        return self.agent, kpis


def main():
    """Main execution function"""
    optimizer = ManufacturingOptimizer(
        n_jobs=200,
        n_machines=15,
        n_episodes=150,
        max_steps=250,
        test_split=0.2,
        output_dir="results"
    )
    
    agent, kpis = optimizer.run()
    
    return agent, kpis


if __name__ == "__main__":
    agent, results = main()
