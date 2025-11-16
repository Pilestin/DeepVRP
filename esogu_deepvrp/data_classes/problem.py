"""
VRP Problem instance - tüm problem verilerini tutan sınıf.
"""

from typing import List, Dict, Tuple
import numpy as np
from .node import Depot, Customer
from .vehicle import Vehicle


class VRPProblem:
    """Complete VRP problem instance."""
    
    def __init__(
        self,
        depot: Depot,
        customers: List[Customer],
        distance_matrix: np.ndarray,
        energy_matrix: np.ndarray,
        location_paths: Dict[Tuple[int, int], np.ndarray],
        num_vehicles: int = 5,
        vehicle_capacity: float = 350.0,
        battery_capacity: float = 15600.0
    ):
        self.depot = depot
        self.customers = customers
        self.distance_matrix = distance_matrix
        self.energy_matrix = energy_matrix
        self.location_paths = location_paths
        
        # Create vehicles
        self.num_vehicles = num_vehicles
        self.vehicles = [
            Vehicle(
                vehicle_id=i,
                capacity=vehicle_capacity,
                battery_capacity=battery_capacity
            )
            for i in range(num_vehicles)
        ]
        
        # Problem dimensions
        self.num_customers = len(customers)
        self.num_nodes = 1 + self.num_customers  # depot + customers
        
    def get_node_by_index(self, idx: int):
        """Get node by index (0=depot, 1..n=customers)."""
        if idx == 0:
            return self.depot
        return self.customers[idx - 1]
    
    def get_distance(self, from_idx: int, to_idx: int) -> float:
        """Get distance between two nodes."""
        from_no = int(self.get_node_by_index(from_idx).no)
        to_no = int(self.get_node_by_index(to_idx).no)
        return self.distance_matrix[from_no - 1, to_no - 1]
    
    def get_energy(self, from_idx: int, to_idx: int) -> float:
        """Get energy consumption between two nodes."""
        from_no = int(self.get_node_by_index(from_idx).no)
        to_no = int(self.get_node_by_index(to_idx).no)
        return self.energy_matrix[from_no - 1, to_no - 1]
    
    def get_location_path(self, from_idx: int, to_idx: int) -> np.ndarray:
        """Get GPS path between two nodes."""
        from_no = int(self.get_node_by_index(from_idx).no)
        to_no = int(self.get_node_by_index(to_idx).no)
        return self.location_paths.get((from_no, to_no), np.array([]))
    
    def reset_vehicles(self):
        """Reset all vehicles to initial state."""
        for vehicle in self.vehicles:
            vehicle.reset()
    
    def get_total_demand(self) -> float:
        """Calculate total customer demand."""
        return sum(c.weight for c in self.customers)
    
    def __repr__(self):
        return (f"VRPProblem(customers={self.num_customers}, vehicles={self.num_vehicles}, "
                f"total_demand={self.get_total_demand():.1f})")
