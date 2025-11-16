"""
Attention Model kullanarak VRP çözümü.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.attention_model import AttentionModel
from esogu_deepvrp.data_classes import VRPProblem


class AttentionVRPSolver:
    """Attention model ile VRP çözümü."""
    
    def __init__(
        self,
        model: AttentionModel,
        device: str = 'cpu'
    ):
        self.model = model.to(device)
        self.device = device
        self.model.eval()
    
    def solve(
        self,
        problem: VRPProblem,
        node_features: torch.Tensor,
        greedy: bool = True
    ) -> Tuple[List[List[str]], float, float, float]:
        """
        VRP problemini çöz.
        
        Args:
            problem: VRPProblem instance
            node_features: (num_nodes, 7) normalized features
            greedy: True ise greedy decoding, False ise sampling
        
        Returns:
            routes: List of routes
            total_distance: Total distance
            total_time: Total time
            total_energy: Total energy
        """
        with torch.no_grad():
            # Add batch dimension
            node_features = node_features.unsqueeze(0).to(self.device)  # (1, num_nodes, 7)
            
            # Encode once
            node_embeddings = self.model.encode(node_features)
            
            # Solve for each vehicle
            routes = []
            total_distance = 0.0
            total_time = 0.0
            total_energy = 0.0
            
            unvisited = set(range(1, problem.num_nodes))  # Depot hariç
            
            for vehicle_idx in range(problem.num_vehicles):
                if not unvisited:
                    break
                
                route, dist, time, energy = self._construct_route(
                    problem, node_features, node_embeddings,
                    unvisited, vehicle_idx, greedy
                )
                
                if len(route) > 2:  # At least depot-customer-depot
                    routes.append(route)
                    total_distance += dist
                    total_time += time
                    total_energy += energy
            
            return routes, total_distance, total_time, total_energy
    
    def _construct_route(
        self,
        problem: VRPProblem,
        node_features: torch.Tensor,
        node_embeddings: torch.Tensor,
        unvisited: set,
        vehicle_idx: int,
        greedy: bool
    ) -> Tuple[List[str], float, float, float]:
        """Tek bir rota oluştur."""
        
        vehicle = problem.vehicles[vehicle_idx]
        vehicle.reset()
        
        current_idx = 0  # Depot
        route_names = [problem.depot.name]
        
        total_distance = 0.0
        total_time = 0.0
        total_energy = 0.0
        current_time = 0.0
        
        first_idx = 0  # İlk node (depot)
        
        while unvisited:
            # Mask oluştur (valid nodes)
            mask = self._create_mask(
                problem, vehicle, current_idx, current_time, unvisited
            )
            
            if not mask.any():
                break
            
            # Remaining capacity
            remaining_capacity = torch.tensor([[vehicle.capacity - vehicle.current_load]], 
                                             dtype=torch.float32, device=self.device)
            
            # Model ile next node seç
            current_node_tensor = torch.tensor([current_idx], dtype=torch.long, device=self.device)
            first_node_tensor = torch.tensor([first_idx], dtype=torch.long, device=self.device)
            
            log_probs = self.model(
                node_features,
                current_node_tensor,
                first_node_tensor,
                remaining_capacity,
                mask.unsqueeze(0)
            )
            
            # Node seç
            if greedy:
                next_idx = log_probs.argmax(dim=-1).item()
            else:
                probs = torch.exp(log_probs)
                next_idx = torch.multinomial(probs, 1).item()
            
            # Feasibility check
            if next_idx == 0 or next_idx not in unvisited:
                break
            
            # Müşteriyi ziyaret et
            customer = problem.customers[next_idx - 1]
            distance = problem.get_distance(current_idx, next_idx)
            energy = problem.get_energy(current_idx, next_idx)
            
            vehicle.load(customer.weight)
            vehicle.consume_battery(energy)
            
            total_distance += distance
            total_energy += energy
            
            # Zaman hesapla (12.5 m/s)
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
        
        # Depoya dön
        if current_idx != 0:
            distance = problem.get_distance(current_idx, 0)
            energy = problem.get_energy(current_idx, 0)
            total_distance += distance
            total_energy += energy
            total_time += distance / 12.5
            route_names.append(problem.depot.name)
        
        return route_names, total_distance, total_time, total_energy
    
    def _create_mask(
        self,
        problem: VRPProblem,
        vehicle,
        current_idx: int,
        current_time: float,
        unvisited: set
    ) -> torch.Tensor:
        """Valid node mask oluştur."""
        
        num_nodes = problem.num_nodes
        mask = torch.zeros(num_nodes, dtype=torch.bool, device=self.device)
        
        for customer_idx in unvisited:
            customer = problem.customers[customer_idx - 1]
            
            # Kapasite
            if not vehicle.can_load(customer.weight):
                continue
            
            # Mesafe ve enerji
            distance = problem.get_distance(current_idx, customer_idx)
            energy = problem.get_energy(current_idx, customer_idx)
            
            # Batarya (depoya dönüş dahil)
            distance_to_depot = problem.get_distance(customer_idx, 0)
            energy_to_depot = problem.get_energy(customer_idx, 0)
            
            if vehicle.current_battery < (energy + energy_to_depot):
                continue
            
            # Zaman penceresi
            travel_time = distance / 12.5
            arrival_time = current_time + travel_time
            service_start_time = max(arrival_time, customer.ready_time)
            
            if service_start_time > customer.due_date:
                continue
            
            mask[customer_idx] = True
        
        return mask


def create_random_attention_model(input_dim: int = 7, embed_dim: int = 128) -> AttentionModel:
    """Random başlatılmış attention model oluştur."""
    model = AttentionModel(
        input_dim=input_dim,
        embed_dim=embed_dim,
        num_heads=8,
        num_encoder_layers=3,
        ff_dim=512
    )
    return model
