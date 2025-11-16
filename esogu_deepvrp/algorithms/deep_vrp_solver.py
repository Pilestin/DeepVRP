"""
Generic VRP solver that works with any deep learning model.
Supports: Attention, Pointer Network, Seq2Seq, GCN, GAT, Hybrid
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Optional
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from esogu_deepvrp.data_classes import VRPProblem


class DeepVRPSolver:
    """Generic deep learning VRP solver."""
    
    def __init__(
        self,
        model: nn.Module,
        model_type: str,
        device: str = 'cpu'
    ):
        """
        Args:
            model: PyTorch model
            model_type: 'attention', 'pointer', 'seq2seq', 'gcn', 'gat', 'hybrid'
            device: 'cpu' or 'cuda'
        """
        self.model = model.to(device)
        self.model_type = model_type.lower()
        self.device = device
        self.model.eval()
    
    def solve(
        self,
        problem: VRPProblem,
        node_features: torch.Tensor,
        graph_data: Optional[object] = None,
        greedy: bool = True
    ) -> Tuple[List[List[str]], float, float, float]:
        """
        VRP problemini çöz.
        
        Args:
            problem: VRPProblem instance
            node_features: (num_nodes, 7) normalized features
            graph_data: PyTorch Geometric Data object (for GNN models)
            greedy: True for greedy decoding, False for sampling
        
        Returns:
            routes: List of routes
            total_distance: Total distance
            total_time: Total time  
            total_energy: Total energy
        """
        with torch.no_grad():
            routes = []
            total_distance = 0.0
            total_time = 0.0
            total_energy = 0.0
            
            unvisited = set(range(1, problem.num_nodes))
            
            for vehicle_idx in range(problem.num_vehicles):
                if not unvisited:
                    break
                
                route, dist, time, energy = self._construct_route(
                    problem, node_features, graph_data,
                    unvisited, vehicle_idx, greedy
                )
                
                if len(route) > 2:
                    routes.append(route)
                    total_distance += dist
                    total_time += time
                    total_energy += energy
            
            return routes, total_distance, total_time, total_energy
    
    def _construct_route(
        self,
        problem: VRPProblem,
        node_features: torch.Tensor,
        graph_data: Optional[object],
        unvisited: set,
        vehicle_idx: int,
        greedy: bool
    ) -> Tuple[List[str], float, float, float]:
        """Construct a single route."""
        
        vehicle = problem.vehicles[vehicle_idx]
        vehicle.reset()
        
        current_idx = 0
        route_names = [problem.depot.name]
        
        total_distance = 0.0
        total_time = 0.0
        total_energy = 0.0
        current_time = 0.0
        
        # Add batch dimension
        node_features_batch = node_features.unsqueeze(0).to(self.device)
        
        # Encode once for efficiency (if model supports it)
        encoded_features = None
        if hasattr(self.model, 'encode'):
            encoded_features = self.model.encode(node_features_batch)
        
        while unvisited:
            # Create mask
            mask = self._create_mask(
                problem, vehicle, current_idx, current_time, unvisited
            )
            
            if not mask.any():
                break
            
            # Get next node from model
            next_idx = self._get_next_node(
                node_features_batch, encoded_features,
                current_idx, vehicle, mask, greedy
            )
            
            if next_idx == 0 or next_idx not in unvisited:
                break
            
            # Visit customer
            customer = problem.customers[next_idx - 1]
            distance = problem.get_distance(current_idx, next_idx)
            energy = problem.get_energy(current_idx, next_idx)
            
            vehicle.load(customer.weight)
            vehicle.consume_battery(energy)
            
            total_distance += distance
            total_energy += energy
            
            # Time calculation (12.5 m/s)
            travel_time = distance / 12.5
            current_time += travel_time
            
            wait_time = 0.0
            if current_time < customer.ready_time:
                wait_time = customer.ready_time - current_time
                current_time = customer.ready_time
            
            current_time += customer.service_time
            total_time += travel_time + wait_time + customer.service_time
            
            route_names.append(customer.name)
            unvisited.remove(next_idx)
            current_idx = next_idx
        
        # Return to depot
        if current_idx != 0:
            distance = problem.get_distance(current_idx, 0)
            energy = problem.get_energy(current_idx, 0)
            total_distance += distance
            total_energy += energy
            total_time += distance / 12.5
            route_names.append(problem.depot.name)
        
        return route_names, total_distance, total_time, total_energy
    
    def _get_next_node(
        self,
        node_features: torch.Tensor,
        encoded_features: Optional[torch.Tensor],
        current_idx: int,
        vehicle,
        mask: torch.Tensor,
        greedy: bool
    ) -> int:
        """Get next node from model."""
        
        current_node_tensor = torch.tensor([current_idx], dtype=torch.long, device=self.device)
        remaining_capacity = torch.tensor(
            [[vehicle.capacity - vehicle.current_load]],
            dtype=torch.float32,
            device=self.device
        )
        mask_batch = mask.unsqueeze(0)
        
        # Model-specific forward pass
        if self.model_type in ['attention', 'hybrid']:
            # These models need first_node_idx
            first_node_tensor = torch.tensor([0], dtype=torch.long, device=self.device)
            log_probs = self.model(
                node_features,
                current_node_tensor,
                first_node_tensor,
                remaining_capacity,
                mask_batch
            )
        elif self.model_type in ['pointer', 'seq2seq']:
            log_probs = self.model(
                node_features,
                current_node_tensor,
                remaining_capacity,
                mask_batch
            )
        elif self.model_type in ['gcn', 'gat']:
            # GNN models need adjacency/edge_index
            # Create fully connected graph
            num_nodes = node_features.size(1)
            adjacency = torch.ones(1, num_nodes, num_nodes, device=self.device)
            log_probs = self.model(
                node_features,
                adjacency,
                current_node_tensor,
                remaining_capacity,
                mask_batch
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        # Select next node
        if greedy:
            next_idx = log_probs.argmax(dim=-1).item()
        else:
            probs = torch.exp(log_probs)
            next_idx = torch.multinomial(probs, 1).item()
        
        return next_idx
    
    def _create_mask(
        self,
        problem: VRPProblem,
        vehicle,
        current_idx: int,
        current_time: float,
        unvisited: set
    ) -> torch.Tensor:
        """Create valid node mask."""
        
        num_nodes = problem.num_nodes
        mask = torch.zeros(num_nodes, dtype=torch.bool, device=self.device)
        
        for customer_idx in unvisited:
            customer = problem.customers[customer_idx - 1]
            
            # Capacity check
            if not vehicle.can_load(customer.weight):
                continue
            
            # Distance and energy
            distance = problem.get_distance(current_idx, customer_idx)
            energy = problem.get_energy(current_idx, customer_idx)
            
            # Battery check (including return to depot)
            distance_to_depot = problem.get_distance(customer_idx, 0)
            energy_to_depot = problem.get_energy(customer_idx, 0)
            
            if vehicle.current_battery < (energy + energy_to_depot):
                continue
            
            # Time window check
            travel_time = distance / 12.5
            arrival_time = current_time + travel_time
            service_start_time = max(arrival_time, customer.ready_time)
            
            if service_start_time > customer.due_date:
                continue
            
            mask[customer_idx] = True
        
        return mask
