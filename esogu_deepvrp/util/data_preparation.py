"""
Okunan raw veriyi VRPProblem ve DL tensörlerine dönüştüren utility fonksiyonlar.
"""

import sys
import os
import numpy as np
import torch
from typing import Tuple, Dict

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from esogu_deepvrp.data_classes import Depot, Customer, VRPProblem
from model.embeddings import create_node_features_from_problem
from model.transforms import normalize_features, to_graph_data


def create_problem_from_raw_data(
    problem_data: dict,
    distance_matrix: np.ndarray,
    energy_matrix: np.ndarray,
    location_paths: dict,
    num_vehicles: int = 5,
    vehicle_capacity: float = 200.0,
    battery_capacity: float = 100.0
) -> VRPProblem:
    """
    Raw veriyi VRPProblem nesnesine dönüştürür.
    
    Args:
        problem_data: read_problem_instance_file() çıktısı
        distance_matrix: Distance matrix
        energy_matrix: Energy matrix
        location_paths: GPS path data
        num_vehicles: Araç sayısı
        vehicle_capacity: Araç kapasitesi (kg)
        battery_capacity: Batarya kapasitesi (kWh)
    
    Returns:
        VRPProblem instance
    """
    # Depot oluştur
    depot_data = problem_data['depot']
    depot = Depot(
        no=depot_data['No'],
        name=depot_data['Name'],
        latitude=depot_data['Latitude'],
        longitude=depot_data['Longitude'],
        x=depot_data['X'],
        y=depot_data['Y']
    )
    
    # Customer'ları oluştur
    customers = []
    for cust_data in problem_data['customers']:
        customer = Customer(
            no=cust_data['No'],
            name=cust_data['Name'],
            node_type=cust_data['Type'],
            latitude=cust_data['Latitude'],
            longitude=cust_data['Longitude'],
            x=cust_data['X'],
            y=cust_data['Y'],
            weight=cust_data['Weight'],
            quantity=cust_data['Quantity'],
            ready_time=cust_data['ReadyTime'],
            service_time=cust_data['ServiceTime'],
            due_date=cust_data['DueDate']
        )
        customers.append(customer)
    
    # VRPProblem oluştur
    problem = VRPProblem(
        depot=depot,
        customers=customers,
        distance_matrix=distance_matrix,
        energy_matrix=energy_matrix,
        location_paths=location_paths,
        num_vehicles=num_vehicles,
        vehicle_capacity=vehicle_capacity,
        battery_capacity=battery_capacity
    )
    
    return problem


def prepare_for_deep_learning(
    problem: VRPProblem,
    normalize: bool = True,
    create_graph: bool = False,
    k_neighbors: int = None
) -> Dict:
    """
    VRPProblem'i derin öğrenme için hazırlar.
    
    Args:
        problem: VRPProblem instance
        normalize: Feature normalizasyonu yapılsın mı
        create_graph: PyTorch Geometric graph oluşturulsun mu
        k_neighbors: Graph için k-nearest neighbor (None = fully connected)
    
    Returns:
        Dictionary with:
            - node_features: (num_nodes, 7) tensor
            - distance_matrix: (num_nodes, num_nodes) tensor
            - energy_matrix: (num_nodes, num_nodes) tensor
            - normalization_stats: Feature normalization parameters
            - graph_data: PyTorch Geometric Data (optional)
    """
    # Node features oluştur
    node_features = create_node_features_from_problem(problem)
    
    # Normalizasyon
    normalization_stats = None
    if normalize:
        node_features, normalization_stats = normalize_features(node_features, method='minmax')
    
    # Extract relevant parts of distance/energy matrices
    # Problem'deki node'ların no'larını kullanarak ilgili kısımları al
    node_numbers = [int(problem.depot.no)] + [int(c.no) for c in problem.customers]
    
    # Matrix indexleme (node numbers 1-based, matrix 0-based)
    indices = [n - 1 for n in node_numbers]
    
    distance_submatrix = problem.distance_matrix[np.ix_(indices, indices)]
    energy_submatrix = problem.energy_matrix[np.ix_(indices, indices)]
    
    # Tensörlere dönüştür
    result = {
        'node_features': torch.from_numpy(node_features).float(),
        'distance_matrix': torch.from_numpy(distance_submatrix).float(),
        'energy_matrix': torch.from_numpy(energy_submatrix).float(),
        'normalization_stats': normalization_stats,
        'problem': problem  # Orjinal problem referansı
    }
    
    # Graph data oluştur (PyTorch Geometric için)
    if create_graph:
        graph_data = to_graph_data(
            node_features,
            distance_submatrix,
            energy_submatrix,
            k_neighbors=k_neighbors
        )
        result['graph_data'] = graph_data
    
    return result


def create_batch(problems: list, **kwargs) -> Dict:
    """
    Birden fazla problem'i batch haline getirir.
    
    Args:
        problems: List of VRPProblem instances
        **kwargs: prepare_for_deep_learning() parametreleri
    
    Returns:
        Batched tensors
    """
    batch_data = [prepare_for_deep_learning(p, **kwargs) for p in problems]
    
    # Stack tensors
    node_features = torch.stack([d['node_features'] for d in batch_data])
    distance_matrices = torch.stack([d['distance_matrix'] for d in batch_data])
    energy_matrices = torch.stack([d['energy_matrix'] for d in batch_data])
    
    result = {
        'node_features': node_features,
        'distance_matrix': distance_matrices,
        'energy_matrix': energy_matrices,
        'normalization_stats': [d['normalization_stats'] for d in batch_data],
        'problems': [d['problem'] for d in batch_data]
    }
    
    if 'graph_data' in batch_data[0]:
        result['graph_data'] = [d['graph_data'] for d in batch_data]
    
    return result
