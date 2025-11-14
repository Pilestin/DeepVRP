"""
Node ve Graph embedding sınıfları - DL modelleri için feature extraction.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple


class NodeEmbedding(nn.Module):
    """
    Node (müşteri/depot) özelliklerini embedding'e dönüştürür.
    
    Features:
        - Koordinatlar (x, y)
        - Talep (demand/weight)
        - Zaman penceresi (ready_time, due_date, service_time)
    """
    
    def __init__(self, embedding_dim: int = 128):
        super().__init__()
        self.embedding_dim = embedding_dim
        
        # Feature dimensions
        # [x, y, demand, ready_time, due_date, service_time, is_depot]
        self.input_dim = 7
        
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )
    
    def forward(self, node_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            node_features: (batch_size, num_nodes, input_dim)
        
        Returns:
            embeddings: (batch_size, num_nodes, embedding_dim)
        """
        return self.encoder(node_features)


class GraphEmbedding(nn.Module):
    """
    Graph (problem instance) için global embedding.
    Distance ve energy matrislerini de kullanır.
    """
    
    def __init__(self, node_embedding_dim: int = 128, graph_embedding_dim: int = 128):
        super().__init__()
        self.node_embedding_dim = node_embedding_dim
        self.graph_embedding_dim = graph_embedding_dim
        
        # Node embeddings
        self.node_embedder = NodeEmbedding(embedding_dim=node_embedding_dim)
        
        # Graph-level aggregation
        self.graph_encoder = nn.Sequential(
            nn.Linear(node_embedding_dim, 256),
            nn.ReLU(),
            nn.Linear(256, graph_embedding_dim),
            nn.LayerNorm(graph_embedding_dim)
        )
        
        # Edge feature encoder (distance, energy)
        self.edge_encoder = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, node_embedding_dim)
        )
    
    def forward(
        self,
        node_features: torch.Tensor,
        distance_matrix: torch.Tensor,
        energy_matrix: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            node_features: (batch_size, num_nodes, input_dim)
            distance_matrix: (batch_size, num_nodes, num_nodes)
            energy_matrix: (batch_size, num_nodes, num_nodes)
        
        Returns:
            node_embeddings: (batch_size, num_nodes, node_embedding_dim)
            graph_embedding: (batch_size, graph_embedding_dim)
        """
        # Node embeddings
        node_embeddings = self.node_embedder(node_features)
        
        # Edge features (distance + energy)
        edge_features = torch.stack([distance_matrix, energy_matrix], dim=-1)
        
        # Graph-level embedding (mean pooling over nodes)
        graph_embedding = self.graph_encoder(node_embeddings.mean(dim=1))
        
        return node_embeddings, graph_embedding
    
    def encode_edges(
        self,
        distance_matrix: torch.Tensor,
        energy_matrix: torch.Tensor
    ) -> torch.Tensor:
        """
        Encode edge features.
        
        Returns:
            edge_embeddings: (batch_size, num_nodes, num_nodes, node_embedding_dim)
        """
        edge_features = torch.stack([distance_matrix, energy_matrix], dim=-1)
        return self.edge_encoder(edge_features)


def create_node_features_from_problem(problem) -> np.ndarray:
    """
    VRPProblem'den node feature matrix oluşturur.
    
    Returns:
        node_features: (num_nodes, 7) numpy array
    """
    num_nodes = problem.num_nodes
    features = np.zeros((num_nodes, 7), dtype=np.float32)
    
    # Depot (index 0)
    depot = problem.depot
    features[0] = [
        depot.x,
        depot.y,
        0.0,  # demand
        0.0,  # ready_time
        10000.0,  # due_date (large value)
        0.0,  # service_time
        1.0   # is_depot flag
    ]
    
    # Customers
    for i, customer in enumerate(problem.customers, start=1):
        features[i] = [
            customer.x,
            customer.y,
            customer.weight,
            customer.ready_time,
            customer.due_date,
            customer.service_time,
            0.0  # is_depot flag
        ]
    
    return features
