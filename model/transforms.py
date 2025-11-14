"""
Veri dönüşüm fonksiyonları - normalizasyon, masking, graph conversion.
"""

import torch
import numpy as np
from typing import Tuple, Optional
from torch_geometric.data import Data


def normalize_features(features: np.ndarray, method: str = 'minmax') -> Tuple[np.ndarray, dict]:
    """
    Feature normalization.
    
    Args:
        features: (num_nodes, num_features) array
        method: 'minmax' or 'standard'
    
    Returns:
        normalized_features: (num_nodes, num_features)
        stats: normalization parameters for inverse transform
    """
    if method == 'minmax':
        min_vals = features.min(axis=0)
        max_vals = features.max(axis=0)
        # Avoid division by zero
        range_vals = max_vals - min_vals
        range_vals[range_vals == 0] = 1.0
        
        normalized = (features - min_vals) / range_vals
        stats = {'min': min_vals, 'max': max_vals, 'method': 'minmax'}
        
    elif method == 'standard':
        mean = features.mean(axis=0)
        std = features.std(axis=0)
        std[std == 0] = 1.0
        
        normalized = (features - mean) / std
        stats = {'mean': mean, 'std': std, 'method': 'standard'}
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return normalized, stats


def denormalize_features(normalized: np.ndarray, stats: dict) -> np.ndarray:
    """Inverse normalization."""
    if stats['method'] == 'minmax':
        return normalized * (stats['max'] - stats['min']) + stats['min']
    elif stats['method'] == 'standard':
        return normalized * stats['std'] + stats['mean']
    else:
        raise ValueError(f"Unknown normalization method: {stats['method']}")


def create_attention_mask(num_nodes: int, visited: Optional[np.ndarray] = None) -> torch.Tensor:
    """
    Attention mask for Transformer models.
    
    Args:
        num_nodes: Total number of nodes
        visited: (num_nodes,) boolean array indicating visited nodes
    
    Returns:
        mask: (num_nodes,) boolean tensor (True = can attend, False = masked)
    """
    mask = torch.ones(num_nodes, dtype=torch.bool)
    
    if visited is not None:
        mask = torch.from_numpy(~visited)
    
    return mask


def create_distance_mask(
    distance_matrix: np.ndarray,
    threshold: float = float('inf')
) -> np.ndarray:
    """
    Create mask for feasible edges based on distance threshold.
    
    Returns:
        mask: (num_nodes, num_nodes) boolean array
    """
    return distance_matrix <= threshold


def to_graph_data(
    node_features: np.ndarray,
    distance_matrix: np.ndarray,
    energy_matrix: np.ndarray,
    k_neighbors: Optional[int] = None
) -> Data:
    """
    Convert problem data to PyTorch Geometric Data object.
    
    Args:
        node_features: (num_nodes, num_features)
        distance_matrix: (num_nodes, num_nodes)
        energy_matrix: (num_nodes, num_nodes)
        k_neighbors: If specified, only keep k nearest neighbors
    
    Returns:
        PyTorch Geometric Data object
    """
    num_nodes = node_features.shape[0]
    
    # Create edge list
    if k_neighbors is None:
        # Fully connected graph
        edge_index = []
        edge_attr = []
        
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    edge_index.append([i, j])
                    edge_attr.append([distance_matrix[i, j], energy_matrix[i, j]])
    
    else:
        # K-nearest neighbors
        edge_index = []
        edge_attr = []
        
        for i in range(num_nodes):
            # Get k nearest neighbors
            distances = distance_matrix[i]
            nearest = np.argsort(distances)[1:k_neighbors+1]  # Exclude self
            
            for j in nearest:
                edge_index.append([i, j])
                edge_attr.append([distance_matrix[i, j], energy_matrix[i, j]])
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float32)
    node_features = torch.tensor(node_features, dtype=torch.float32)
    
    data = Data(
        x=node_features,
        edge_index=edge_index,
        edge_attr=edge_attr,
        num_nodes=num_nodes
    )
    
    return data


def batch_to_tensor(batch_data: list) -> dict:
    """
    Convert batch of problem instances to tensors.
    
    Args:
        batch_data: List of (node_features, distance_matrix, energy_matrix) tuples
    
    Returns:
        Dictionary with batched tensors
    """
    node_features = []
    distance_matrices = []
    energy_matrices = []
    
    for nf, dm, em in batch_data:
        node_features.append(torch.from_numpy(nf))
        distance_matrices.append(torch.from_numpy(dm))
        energy_matrices.append(torch.from_numpy(em))
    
    return {
        'node_features': torch.stack(node_features),
        'distance_matrix': torch.stack(distance_matrices),
        'energy_matrix': torch.stack(energy_matrices)
    }
