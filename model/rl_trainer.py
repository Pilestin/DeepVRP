"""
Reinforcement Learning training framework for VRP.
Implements REINFORCE and Actor-Critic algorithms.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Optional, Tuple, Dict, List
import numpy as np
from collections import deque


class RLTrainer:
    """
    Reinforcement Learning trainer for VRP models.
    Supports REINFORCE and Actor-Critic methods.
    """
    
    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 1e-4,
        baseline_type: str = 'exponential',  # 'exponential', 'critic', 'rollout'
        baseline_alpha: float = 0.95,
        use_critic: bool = False,
        critic_learning_rate: float = 1e-4,
        max_grad_norm: float = 1.0
    ):
        """
        Args:
            model: Policy network (AttentionModel or GNNModel)
            learning_rate: Learning rate for policy network
            baseline_type: Type of baseline for variance reduction
            baseline_alpha: Decay factor for exponential baseline
            use_critic: Whether to use critic network (Actor-Critic)
            critic_learning_rate: Learning rate for critic
            max_grad_norm: Maximum gradient norm for clipping
        """
        self.model = model
        self.device = next(model.parameters()).device
        
        # Optimizer
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Baseline
        self.baseline_type = baseline_type
        self.baseline_alpha = baseline_alpha
        self.baseline_value = None
        
        # Critic network (for Actor-Critic)
        self.use_critic = use_critic
        if use_critic:
            self.critic = ValueNetwork(input_dim=128).to(self.device)
            self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_learning_rate)
        else:
            self.critic = None
        
        self.max_grad_norm = max_grad_norm
        
        # Training statistics
        self.stats = {
            'loss': [],
            'reward': [],
            'baseline': [],
            'advantage': []
        }
    
    def compute_baseline(self, rewards: torch.Tensor) -> torch.Tensor:
        """
        Compute baseline for variance reduction.
        
        Args:
            rewards: (batch_size,) tensor of rewards
        
        Returns:
            baseline: (batch_size,) tensor
        """
        if self.baseline_type == 'exponential':
            # Exponential moving average
            mean_reward = rewards.mean().item()
            if self.baseline_value is None:
                self.baseline_value = mean_reward
            else:
                self.baseline_value = (
                    self.baseline_alpha * self.baseline_value +
                    (1 - self.baseline_alpha) * mean_reward
                )
            return torch.full_like(rewards, self.baseline_value)
        
        elif self.baseline_type == 'mean':
            # Batch mean
            return rewards.mean().expand_as(rewards)
        
        elif self.baseline_type == 'critic':
            # Use critic network (must have use_critic=True)
            if not self.use_critic:
                raise ValueError("Critic baseline requires use_critic=True")
            return self.baseline_value  # Computed in train_step
        
        else:
            # No baseline
            return torch.zeros_like(rewards)
    
    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
        sample_size: int = 1,
        temperature: float = 1.0
    ) -> Dict[str, float]:
        """
        Single training step with policy gradient.
        
        Args:
            batch: Dictionary containing:
                - node_features: (batch_size, num_nodes, feature_dim)
                - distance_matrix: (batch_size, num_nodes, num_nodes)
                - demands: (batch_size, num_nodes)
                - capacity: scalar or (batch_size,)
            sample_size: Number of trajectories to sample per instance
            temperature: Sampling temperature for exploration
        
        Returns:
            Dictionary with training statistics
        """
        self.model.train()
        
        # Sample trajectories and compute rewards
        log_probs_list, rewards, trajectories = self.sample_trajectories(
            batch, sample_size, temperature
        )
        
        # Compute baseline
        if self.use_critic and self.baseline_type == 'critic':
            # Get state value estimates from critic
            state_values = self.compute_state_values(batch)
            baseline = state_values
            self.baseline_value = baseline
        else:
            baseline = self.compute_baseline(rewards)
        
        # Compute advantages
        advantages = rewards - baseline
        
        # Policy gradient loss
        policy_loss = -(log_probs_list * advantages.detach()).mean()
        
        # Update policy
        self.optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()
        
        # Update critic if using Actor-Critic
        if self.use_critic:
            critic_loss = F.mse_loss(state_values, rewards.detach())
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.critic_optimizer.step()
        
        # Statistics
        stats = {
            'loss': policy_loss.item(),
            'reward': rewards.mean().item(),
            'baseline': baseline.mean().item(),
            'advantage': advantages.mean().item(),
            'reward_std': rewards.std().item()
        }
        
        if self.use_critic:
            stats['critic_loss'] = critic_loss.item()
        
        # Update running statistics
        for key in ['loss', 'reward', 'baseline', 'advantage']:
            self.stats[key].append(stats.get(key, 0))
        
        return stats
    
    def sample_trajectories(
        self,
        batch: Dict[str, torch.Tensor],
        sample_size: int = 1,
        temperature: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor, List]:
        """
        Sample trajectories using current policy.
        
        Returns:
            log_probs: (batch_size * sample_size,) total log probability
            rewards: (batch_size * sample_size,) negative tour lengths
            trajectories: List of sampled tours
        """
        batch_size = batch['node_features'].size(0)
        num_nodes = batch['node_features'].size(1)
        device = batch['node_features'].device
        
        # Expand batch for multiple samples
        if sample_size > 1:
            expanded_batch = {
                key: val.repeat_interleave(sample_size, dim=0)
                for key, val in batch.items()
            }
        else:
            expanded_batch = batch
        
        effective_batch_size = batch_size * sample_size
        
        # Initialize
        current_node = torch.zeros(effective_batch_size, dtype=torch.long, device=device)
        first_node = current_node.clone()
        visited = torch.zeros(effective_batch_size, num_nodes, dtype=torch.bool, device=device)
        visited[:, 0] = True  # Mark depot as visited initially
        
        # Track trajectory
        trajectories = [current_node.clone()]
        log_probs_list = []
        
        # Vehicle state
        if 'capacity' in batch:
            capacity = batch['capacity']
            if isinstance(capacity, (int, float)):
                remaining_capacity = torch.full(
                    (effective_batch_size, 1), capacity, dtype=torch.float32, device=device
                )
            else:
                remaining_capacity = capacity.repeat_interleave(sample_size, dim=0).unsqueeze(1)
        else:
            remaining_capacity = torch.full(
                (effective_batch_size, 1), 200.0, dtype=torch.float32, device=device
            )
        
        demands = expanded_batch.get('demands', expanded_batch['node_features'][:, :, 2])
        
        # Construct tour
        for step in range(num_nodes - 1):
            # Create mask for valid nodes
            mask = ~visited
            
            # Check capacity constraints
            can_serve = demands <= remaining_capacity
            mask = mask & can_serve
            
            # Always allow return to depot
            mask[:, 0] = True
            
            # Get action probabilities
            log_probs = self.model(
                expanded_batch['node_features'],
                current_node,
                first_node,
                remaining_capacity / 200.0,  # Normalize
                mask,
                expanded_batch.get('distance_matrix')
            )
            
            # Sample action
            if temperature != 1.0:
                probs = (log_probs / temperature).exp()
                probs = probs / probs.sum(dim=-1, keepdim=True)
                selected = torch.multinomial(probs, 1).squeeze(-1)
                selected_log_probs = log_probs.gather(1, selected.unsqueeze(1)).squeeze(1)
            else:
                probs = log_probs.exp()
                selected = torch.multinomial(probs, 1).squeeze(-1)
                selected_log_probs = log_probs.gather(1, selected.unsqueeze(1)).squeeze(1)
            
            # Update state
            visited.scatter_(1, selected.unsqueeze(1), True)
            current_node = selected
            trajectories.append(current_node.clone())
            log_probs_list.append(selected_log_probs)
            
            # Update capacity
            selected_demands = demands.gather(1, selected.unsqueeze(1))
            remaining_capacity = remaining_capacity - selected_demands
            
            # Reset capacity when returning to depot
            at_depot = (selected == 0)
            remaining_capacity[at_depot] = 200.0 if isinstance(capacity, (int, float)) else capacity[at_depot]
        
        # Return to depot
        trajectories.append(torch.zeros_like(current_node))
        
        # Compute total log probability
        total_log_probs = torch.stack(log_probs_list, dim=1).sum(dim=1)
        
        # Compute rewards (negative tour length)
        tour_lengths = self.compute_tour_length(trajectories, expanded_batch['distance_matrix'])
        rewards = -tour_lengths
        
        return total_log_probs, rewards, trajectories
    
    def compute_tour_length(
        self,
        trajectories: List[torch.Tensor],
        distance_matrix: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute total tour length from trajectories.
        
        Args:
            trajectories: List of tensors, each (batch_size,)
            distance_matrix: (batch_size, num_nodes, num_nodes)
        
        Returns:
            tour_lengths: (batch_size,)
        """
        batch_size = distance_matrix.size(0)
        device = distance_matrix.device
        
        tour_length = torch.zeros(batch_size, device=device)
        
        for i in range(len(trajectories) - 1):
            from_node = trajectories[i]
            to_node = trajectories[i + 1]
            
            # Get distances
            distances = distance_matrix[
                torch.arange(batch_size, device=device),
                from_node,
                to_node
            ]
            tour_length += distances
        
        return tour_length
    
    def compute_state_values(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute state values using critic network.
        
        Returns:
            values: (batch_size,)
        """
        if not self.use_critic:
            raise ValueError("Critic not initialized")
        
        # Encode problem instance
        with torch.no_grad():
            node_embeddings = self.model.encode(batch['node_features'])
            graph_embedding = node_embeddings.mean(dim=1)
        
        values = self.critic(graph_embedding).squeeze(-1)
        return values


class ValueNetwork(nn.Module):
    """Critic network for Actor-Critic."""
    
    def __init__(self, input_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, input_dim)
        
        Returns:
            value: (batch_size, 1)
        """
        return self.net(x)
