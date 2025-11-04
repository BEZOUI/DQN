"""
Deep Q-Network for Reconfigurable Manufacturing Systems
Author: Professor of Operations Research & Deep Learning
Version: 3.0 - Python Production Implementation (NumPy-based)

This module contains the DQN agent and manufacturing environment.
Designed for zero-error execution with defensive programming.
Pure NumPy implementation - no external ML frameworks required.
"""

import numpy as np
from collections import deque, namedtuple
from typing import List, Tuple, Dict, Optional
import random
from datetime import datetime, timedelta

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)

# Experience tuple for replay buffer
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done', 'valid_actions'])


def relu(x):
    """ReLU activation function"""
    return np.maximum(0, x)

def relu_derivative(x):
    """Derivative of ReLU"""
    return (x > 0).astype(float)

def tanh(x):
    """Tanh activation function"""
    return np.tanh(x)

def tanh_derivative(x):
    """Derivative of tanh"""
    return 1 - np.tanh(x) ** 2

def huber_loss(y_true, y_pred, delta=1.0):
    """Huber loss function"""
    error = y_true - y_pred
    is_small_error = np.abs(error) <= delta
    squared_loss = 0.5 * error ** 2
    linear_loss = delta * (np.abs(error) - 0.5 * delta)
    return np.where(is_small_error, squared_loss, linear_loss)

def huber_loss_derivative(y_true, y_pred, delta=1.0):
    """Derivative of Huber loss"""
    error = y_pred - y_true
    is_small_error = np.abs(error) <= delta
    return np.where(is_small_error, error, delta * np.sign(error))


class DQNetwork:
    """
    Deep Q-Network with 4 hidden layers implemented in pure NumPy.
    Input: state_size -> Output: action_size Q-values
    Architecture: state_size -> 256 -> 128 -> 64 -> 32 -> action_size
    """
    
    def __init__(self, state_size: int, action_size: int, dropout: float = 0.2):
        self.state_size = state_size
        self.action_size = action_size
        self.dropout = dropout
        self.training = True
        
        # Initialize weights with Xavier initialization
        self.W1 = np.random.randn(state_size, 256) * np.sqrt(2.0 / state_size)
        self.b1 = np.zeros((1, 256))
        
        self.W2 = np.random.randn(256, 128) * np.sqrt(2.0 / 256)
        self.b2 = np.zeros((1, 128))
        
        self.W3 = np.random.randn(128, 64) * np.sqrt(2.0 / 128)
        self.b3 = np.zeros((1, 64))
        
        self.W4 = np.random.randn(64, 32) * np.sqrt(2.0 / 64)
        self.b4 = np.zeros((1, 32))
        
        self.W5 = np.random.randn(32, action_size) * np.sqrt(2.0 / 32)
        self.b5 = np.zeros((1, action_size))
        
        # Store activations for backpropagation
        self.cache = {}
    
    def forward(self, x, training=None):
        """Forward pass through the network"""
        if training is None:
            training = self.training
        
        # Layer 1
        z1 = np.dot(x, self.W1) + self.b1
        a1 = relu(z1)
        if training:
            a1 *= (np.random.rand(*a1.shape) > self.dropout).astype(float) / (1 - self.dropout)
        
        # Layer 2
        z2 = np.dot(a1, self.W2) + self.b2
        a2 = relu(z2)
        if training:
            a2 *= (np.random.rand(*a2.shape) > self.dropout * 0.75).astype(float) / (1 - self.dropout * 0.75)
        
        # Layer 3
        z3 = np.dot(a2, self.W3) + self.b3
        a3 = relu(z3)
        if training:
            a3 *= (np.random.rand(*a3.shape) > self.dropout * 0.5).astype(float) / (1 - self.dropout * 0.5)
        
        # Layer 4
        z4 = np.dot(a3, self.W4) + self.b4
        a4 = tanh(z4)
        
        # Output layer
        z5 = np.dot(a4, self.W5) + self.b5
        output = z5
        
        # Store for backprop
        self.cache = {
            'x': x, 'z1': z1, 'a1': a1, 'z2': z2, 'a2': a2,
            'z3': z3, 'a3': a3, 'z4': z4, 'a4': a4, 'z5': z5
        }
        
        return output
    
    def backward(self, dL_dout, learning_rate=0.001, max_grad_norm=10.0):
        """
        Backward pass and weight update.
        Includes gradient clipping for stability.
        """
        batch_size = self.cache['x'].shape[0]
        
        # Output layer gradients
        dL_dz5 = dL_dout
        dL_dW5 = np.dot(self.cache['a4'].T, dL_dz5) / batch_size
        dL_db5 = np.sum(dL_dz5, axis=0, keepdims=True) / batch_size
        
        # Layer 4 gradients
        dL_da4 = np.dot(dL_dz5, self.W5.T)
        dL_dz4 = dL_da4 * tanh_derivative(self.cache['z4'])
        dL_dW4 = np.dot(self.cache['a3'].T, dL_dz4) / batch_size
        dL_db4 = np.sum(dL_dz4, axis=0, keepdims=True) / batch_size
        
        # Layer 3 gradients
        dL_da3 = np.dot(dL_dz4, self.W4.T)
        dL_dz3 = dL_da3 * relu_derivative(self.cache['z3'])
        dL_dW3 = np.dot(self.cache['a2'].T, dL_dz3) / batch_size
        dL_db3 = np.sum(dL_dz3, axis=0, keepdims=True) / batch_size
        
        # Layer 2 gradients
        dL_da2 = np.dot(dL_dz3, self.W3.T)
        dL_dz2 = dL_da2 * relu_derivative(self.cache['z2'])
        dL_dW2 = np.dot(self.cache['a1'].T, dL_dz2) / batch_size
        dL_db2 = np.sum(dL_dz2, axis=0, keepdims=True) / batch_size
        
        # Layer 1 gradients
        dL_da1 = np.dot(dL_dz2, self.W2.T)
        dL_dz1 = dL_da1 * relu_derivative(self.cache['z1'])
        dL_dW1 = np.dot(self.cache['x'].T, dL_dz1) / batch_size
        dL_db1 = np.sum(dL_dz1, axis=0, keepdims=True) / batch_size
        
        # Gradient clipping
        gradients = [dL_dW1, dL_dW2, dL_dW3, dL_dW4, dL_dW5,
                    dL_db1, dL_db2, dL_db3, dL_db4, dL_db5]
        
        total_norm = np.sqrt(sum(np.sum(g ** 2) for g in gradients))
        clip_coef = max_grad_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            gradients = [g * clip_coef for g in gradients]
            dL_dW1, dL_dW2, dL_dW3, dL_dW4, dL_dW5, dL_db1, dL_db2, dL_db3, dL_db4, dL_db5 = gradients
        
        # Update weights
        self.W1 -= learning_rate * dL_dW1
        self.b1 -= learning_rate * dL_db1
        self.W2 -= learning_rate * dL_dW2
        self.b2 -= learning_rate * dL_db2
        self.W3 -= learning_rate * dL_dW3
        self.b3 -= learning_rate * dL_db3
        self.W4 -= learning_rate * dL_dW4
        self.b4 -= learning_rate * dL_db4
        self.W5 -= learning_rate * dL_dW5
        self.b5 -= learning_rate * dL_db5
    
    def copy_weights_from(self, other_network):
        """Copy weights from another network (for target network updates)"""
        self.W1 = other_network.W1.copy()
        self.b1 = other_network.b1.copy()
        self.W2 = other_network.W2.copy()
        self.b2 = other_network.b2.copy()
        self.W3 = other_network.W3.copy()
        self.b3 = other_network.b3.copy()
        self.W4 = other_network.W4.copy()
        self.b4 = other_network.b4.copy()
        self.W5 = other_network.W5.copy()
        self.b5 = other_network.b5.copy()
    
    def get_weights(self):
        """Get all weights as a dictionary"""
        return {
            'W1': self.W1.copy(), 'b1': self.b1.copy(),
            'W2': self.W2.copy(), 'b2': self.b2.copy(),
            'W3': self.W3.copy(), 'b3': self.b3.copy(),
            'W4': self.W4.copy(), 'b4': self.b4.copy(),
            'W5': self.W5.copy(), 'b5': self.b5.copy()
        }
    
    def set_weights(self, weights):
        """Set weights from a dictionary"""
        self.W1 = weights['W1'].copy()
        self.b1 = weights['b1'].copy()
        self.W2 = weights['W2'].copy()
        self.b2 = weights['b2'].copy()
        self.W3 = weights['W3'].copy()
        self.b3 = weights['b3'].copy()
        self.W4 = weights['W4'].copy()
        self.b4 = weights['b4'].copy()
        self.W5 = weights['W5'].copy()
        self.b5 = weights['b5'].copy()



class ReplayBuffer:
    """
    Experience Replay Buffer with efficient sampling.
    """
    
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done, valid_actions):
        """Add experience to buffer"""
        self.buffer.append(Experience(state, action, reward, next_state, done, valid_actions))
    
    def sample(self, batch_size: int) -> List[Experience]:
        """Sample random batch"""
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))
    
    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """
    Double DQN Agent with Experience Replay and Target Network.
    Implements epsilon-greedy exploration with decay.
    Pure NumPy implementation - no external ML frameworks required.
    """
    
    def __init__(
        self,
        state_size: int,
        action_size: int,
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        buffer_size: int = 10000,
        batch_size: int = 64,
        target_update: int = 100,
        max_grad_norm: float = 10.0
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update
        self.max_grad_norm = max_grad_norm
        self.learning_rate = learning_rate
        self.step_count = 0
        
        # Networks
        self.q_network = DQNetwork(state_size, action_size)
        self.target_network = DQNetwork(state_size, action_size)
        self.target_network.copy_weights_from(self.q_network)
        self.target_network.training = False
        
        # Replay buffer
        self.memory = ReplayBuffer(buffer_size)
        
        # Tracking
        self.losses = []
        self.epsilons = []
    
    def act(self, state: np.ndarray, valid_actions: List[int], training: bool = True) -> int:
        """
        Select action using epsilon-greedy policy.
        Only considers valid actions (masked action space).
        """
        if len(valid_actions) == 0:
            return 0  # Default to wait action if no valid actions
        
        # Epsilon-greedy exploration
        if training and random.random() < self.epsilon:
            return random.choice(valid_actions)
        
        # Exploitation: choose best valid action
        state_batch = state.reshape(1, -1)
        self.q_network.training = False
        q_values = self.q_network.forward(state_batch, training=False)[0]
        self.q_network.training = True
        
        # Mask invalid actions with very negative values
        masked_q_values = np.full(self.action_size, -1e9)
        masked_q_values[valid_actions] = q_values[valid_actions]
        
        return int(np.argmax(masked_q_values))
    
    def remember(self, state, action, reward, next_state, done, valid_actions):
        """Store experience in replay buffer"""
        self.memory.push(state, action, reward, next_state, done, valid_actions)
    
    def train_step(self) -> float:
        """
        Perform one training step using Double DQN algorithm.
        Returns: loss value
        """
        if len(self.memory) < self.batch_size:
            return 0.0
        
        # Sample batch
        batch = self.memory.sample(self.batch_size)
        
        # Prepare batch arrays
        states = np.array([e.state for e in batch])
        actions = np.array([e.action for e in batch])
        rewards = np.array([e.reward for e in batch])
        next_states = np.array([e.next_state for e in batch])
        dones = np.array([float(e.done) for e in batch])
        
        # Current Q-values
        current_q_values = self.q_network.forward(states, training=True)
        current_q_selected = current_q_values[np.arange(self.batch_size), actions]
        
        # Double DQN: use online network to select action, target network to evaluate
        self.q_network.training = False
        next_q_online = self.q_network.forward(next_states, training=False)
        self.q_network.training = True
        next_actions = np.argmax(next_q_online, axis=1)
        
        self.target_network.training = False
        next_q_target = self.target_network.forward(next_states, training=False)
        next_q_selected = next_q_target[np.arange(self.batch_size), next_actions]
        
        # Compute target Q-values
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_selected
        
        # Compute loss (Huber)
        losses = huber_loss(target_q_values, current_q_selected)
        loss = np.mean(losses)
        
        # Compute gradients
        dL_dq = huber_loss_derivative(target_q_values, current_q_selected) / self.batch_size
        
        # Prepare gradient for backprop
        dL_dout = np.zeros_like(current_q_values)
        dL_dout[np.arange(self.batch_size), actions] = dL_dq
        
        # Backpropagation and weight update
        self.q_network.backward(dL_dout, self.learning_rate, self.max_grad_norm)
        
        # Update target network
        self.step_count += 1
        if self.step_count % self.target_update == 0:
            self.target_network.copy_weights_from(self.q_network)
        
        # Decay epsilon
        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay
        
        # Track metrics
        self.losses.append(loss)
        self.epsilons.append(self.epsilon)
        
        return loss
    
    def save(self, filepath: str):
        """Save agent state"""
        state = {
            'q_network': self.q_network.get_weights(),
            'target_network': self.target_network.get_weights(),
            'epsilon': self.epsilon,
            'step_count': self.step_count,
            'losses': self.losses,
            'epsilons': self.epsilons
        }
        np.save(filepath, state, allow_pickle=True)
    
    def load(self, filepath: str):
        """Load agent state"""
        state = np.load(filepath, allow_pickle=True).item()
        self.q_network.set_weights(state['q_network'])
        self.target_network.set_weights(state['target_network'])
        self.epsilon = state['epsilon']
        self.step_count = state['step_count']
        self.losses = state.get('losses', [])
        self.epsilons = state.get('epsilons', [])


class Job:
    """Manufacturing job with properties"""
    
    def __init__(
        self,
        job_id: int,
        machine_id: int,
        processing_time: float,
        energy_consumption: float,
        deadline: datetime,
        optimization_category: str = "Moderate"
    ):
        self.job_id = job_id
        self.machine_id = machine_id
        self.processing_time = processing_time
        self.energy_consumption = energy_consumption
        self.deadline = deadline
        self.optimization_category = optimization_category
        self.completed = False
        self.start_time = None
        self.end_time = None
        self.lateness = 0.0
    
    def is_urgent(self, current_time: datetime) -> bool:
        """Check if job is urgent (deadline within 1 hour)"""
        return (self.deadline - current_time).total_seconds() < 3600
    
    def is_energy_efficient(self) -> bool:
        """Check if job is energy efficient"""
        return self.energy_consumption <= 5.0


class ManufacturingEnvironment:
    """
    Reconfigurable Manufacturing System Environment.
    Manages machines, jobs, scheduling, and KPI tracking.
    """
    
    def __init__(
        self,
        jobs: List[Job],
        n_machines: int = 15,
        time_step_minutes: int = 5
    ):
        self.jobs = jobs
        self.n_machines = n_machines
        self.time_step_minutes = time_step_minutes
        
        # State tracking
        self.current_time = datetime.now()
        self.machines_available = [True] * n_machines
        self.machines_end_time = [self.current_time] * n_machines
        self.machines_jobs_completed = [0] * n_machines
        self.machines_processing_time = [0.0] * n_machines
        self.machines_energy = [0.0] * n_machines
        
        self.pending_jobs = [j for j in jobs if not j.completed]
        self.completed_jobs = []
        self.total_energy = 0.0
        self.episode_reward = 0.0
        self.step_number = 0
        
        # KPIs
        self.makespan_start = None
        self.makespan_end = None
    
    def reset(self, jobs: List[Job] = None):
        """Reset environment to initial state"""
        if jobs is not None:
            self.jobs = jobs
        
        # Reset jobs
        for job in self.jobs:
            job.completed = False
            job.start_time = None
            job.end_time = None
            job.lateness = 0.0
        
        self.current_time = datetime.now()
        self.machines_available = [True] * self.n_machines
        self.machines_end_time = [self.current_time] * self.n_machines
        self.machines_jobs_completed = [0] * self.n_machines
        self.machines_processing_time = [0.0] * self.n_machines
        self.machines_energy = [0.0] * self.n_machines
        
        self.pending_jobs = [j for j in self.jobs if not j.completed]
        self.completed_jobs = []
        self.total_energy = 0.0
        self.episode_reward = 0.0
        self.step_number = 0
        self.makespan_start = None
        self.makespan_end = None
        
        return self.get_state()
    
    def get_state(self) -> np.ndarray:
        """
        Get current state representation (72 dimensions).
        Features: machine status (60) + system metrics (12)
        """
        # Machine features (15 machines × 4 features)
        machine_features = []
        max_processing = max(self.machines_processing_time) if max(self.machines_processing_time) > 0 else 1.0
        max_energy = max(self.machines_energy) if max(self.machines_energy) > 0 else 1.0
        max_jobs = max(self.machines_jobs_completed) if max(self.machines_jobs_completed) > 0 else 1.0
        
        for i in range(self.n_machines):
            machine_features.extend([
                float(self.machines_available[i]),
                self.machines_processing_time[i] / max_processing,
                self.machines_energy[i] / max_energy,
                self.machines_jobs_completed[i] / max_jobs
            ])
        
        # System features
        total_jobs = len(self.jobs)
        pending_count = len(self.pending_jobs)
        completed_count = len(self.completed_jobs)
        
        if pending_count > 0:
            avg_proc_time = np.mean([j.processing_time for j in self.pending_jobs])
            std_proc_time = np.std([j.processing_time for j in self.pending_jobs])
            avg_energy = np.mean([j.energy_consumption for j in self.pending_jobs])
            std_energy = np.std([j.energy_consumption for j in self.pending_jobs])
            urgent_count = sum(1 for j in self.pending_jobs if j.is_urgent(self.current_time))
            efficient_count = sum(1 for j in self.pending_jobs if j.is_energy_efficient())
            available_for_pending = sum(1 for j in self.pending_jobs if self.machines_available[j.machine_id - 1])
        else:
            avg_proc_time = std_proc_time = avg_energy = std_energy = 0.0
            urgent_count = efficient_count = available_for_pending = 0
        
        system_features = [
            pending_count / total_jobs,
            completed_count / total_jobs,
            self.total_energy / (total_jobs * 10.0),  # Normalized
            self.step_number / 500.0,  # Temporal progress
            avg_proc_time / 200.0,
            std_proc_time / 100.0,
            avg_energy / 15.0,
            std_energy / 10.0,
            urgent_count / max(pending_count, 1),
            efficient_count / max(pending_count, 1),
            available_for_pending / max(pending_count, 1),
            sum(self.machines_available) / self.n_machines
        ]
        
        state = np.array(machine_features + system_features, dtype=np.float32)
        return state
    
    def get_valid_actions(self) -> List[int]:
        """
        Get list of valid actions.
        Action 0: Wait
        Action 1+i: Schedule job i (if machine available)
        """
        valid_actions = [0]  # Wait is always valid
        
        for i, job in enumerate(self.pending_jobs):
            machine_idx = job.machine_id - 1
            if 0 <= machine_idx < self.n_machines and self.machines_available[machine_idx]:
                valid_actions.append(i + 1)
        
        return valid_actions
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute action and return (next_state, reward, done, info).
        """
        self.step_number += 1
        reward = 0.0
        info = {}
        
        # Update machine availability
        for i in range(self.n_machines):
            if not self.machines_available[i] and self.current_time >= self.machines_end_time[i]:
                self.machines_available[i] = True
        
        # Action 0: Wait
        if action == 0:
            self.current_time += timedelta(minutes=self.time_step_minutes)
            reward = -0.5  # Small penalty for waiting
        
        # Action 1+: Schedule job
        elif 1 <= action <= len(self.pending_jobs):
            job_idx = action - 1
            if job_idx < len(self.pending_jobs):
                job = self.pending_jobs[job_idx]
                machine_idx = job.machine_id - 1
                
                # Verify action is valid
                if 0 <= machine_idx < self.n_machines and self.machines_available[machine_idx]:
                    # Schedule job
                    job.start_time = self.current_time
                    job.end_time = self.current_time + timedelta(minutes=job.processing_time)
                    
                    # Update machine
                    self.machines_available[machine_idx] = False
                    self.machines_end_time[machine_idx] = job.end_time
                    self.machines_jobs_completed[machine_idx] += 1
                    self.machines_processing_time[machine_idx] += job.processing_time
                    self.machines_energy[machine_idx] += job.energy_consumption
                    
                    # Update job
                    job.completed = True
                    job.lateness = max(0, (job.end_time - job.deadline).total_seconds() / 60.0)
                    
                    # Track makespan
                    if self.makespan_start is None:
                        self.makespan_start = job.start_time
                    self.makespan_end = job.end_time
                    
                    # Move job
                    self.pending_jobs.pop(job_idx)
                    self.completed_jobs.append(job)
                    self.total_energy += job.energy_consumption
                    
                    # Reward shaping
                    reward += 100  # Base completion reward
                    
                    if job.lateness == 0:
                        reward += 50  # On-time bonus
                    
                    if job.is_energy_efficient():
                        reward += 15  # Energy efficiency bonus
                    
                    # Optimization category bonus
                    if job.optimization_category == "Optimal":
                        reward += 30
                    elif job.optimization_category == "High":
                        reward += 20
                    elif job.optimization_category == "Moderate":
                        reward += 10
                    
                    info['job_completed'] = True
                    info['job_id'] = job.job_id
                else:
                    reward = -1  # Penalty for invalid action
            else:
                reward = -1  # Penalty for invalid action
        
        # Check if episode is done
        done = len(self.pending_jobs) == 0
        
        # Get next state
        next_state = self.get_state()
        
        self.episode_reward += reward
        
        return next_state, reward, done, info
    
    def get_kpis(self) -> Dict:
        """Calculate and return key performance indicators"""
        total_jobs = len(self.jobs)
        completed_count = len(self.completed_jobs)
        
        kpis = {
            'completion_rate': completed_count / total_jobs if total_jobs > 0 else 0.0,
            'jobs_completed': completed_count,
            'total_jobs': total_jobs,
            'total_energy': self.total_energy,
            'avg_energy': self.total_energy / completed_count if completed_count > 0 else 0.0,
            'total_reward': self.episode_reward
        }
        
        # Makespan
        if self.makespan_start and self.makespan_end:
            makespan = (self.makespan_end - self.makespan_start).total_seconds() / 60.0
            kpis['makespan'] = makespan
        else:
            kpis['makespan'] = 0.0
        
        # Machine utilization
        total_time = kpis['makespan']
        if total_time > 0:
            total_processing = sum(self.machines_processing_time)
            kpis['utilization'] = (total_processing / (self.n_machines * total_time)) * 100
        else:
            kpis['utilization'] = 0.0
        
        # On-time delivery
        if completed_count > 0:
            on_time = sum(1 for j in self.completed_jobs if j.lateness == 0)
            kpis['on_time_rate'] = on_time / completed_count
        else:
            kpis['on_time_rate'] = 0.0
        
        return kpis


def generate_synthetic_jobs(n_jobs: int = 200, n_machines: int = 15) -> List[Job]:
    """
    Generate synthetic job dataset for testing.
    Mimics realistic manufacturing scenarios.
    """
    jobs = []
    base_time = datetime.now()
    
    optimization_categories = ["Optimal", "High", "Moderate", "Low"]
    category_weights = [0.2, 0.3, 0.4, 0.1]
    
    for i in range(n_jobs):
        job_id = i + 1
        machine_id = random.randint(1, n_machines)
        processing_time = np.random.uniform(10, 180)  # 10-180 minutes
        energy_consumption = np.random.uniform(2, 15)  # 2-15 units
        deadline = base_time + timedelta(hours=random.randint(1, 48))
        optimization_category = random.choices(optimization_categories, weights=category_weights)[0]
        
        jobs.append(Job(
            job_id=job_id,
            machine_id=machine_id,
            processing_time=processing_time,
            energy_consumption=energy_consumption,
            deadline=deadline,
            optimization_category=optimization_category
        ))
    
    return jobs
