"""
Hybrid model combining GNN and Attention mechanisms.
"""

import torch
import torch.nn as nn
from .gnn_model import GNN
from .attention_model import TransformerEncoderLayer, AttentionDecoder


class HybridEncoder(nn.Module):
    """
    Hybrid encoder combining GNN for structural learning
    and Transformer for sequence modeling.
    """
    
    def __init__(
        self,
        input_dim: int = 7,
        embed_dim: int = 128,
        num_gnn_layers: int = 2,
        num_transformer_layers: int = 2,
        gnn_type: str = 'gat',
        num_heads: int = 8,
        k_neighbors: int = None
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.k_neighbors = k_neighbors
        
        # GNN for structural graph features
        self.gnn = GNN(
            input_dim=input_dim,
            hidden_dim=embed_dim,
            output_dim=embed_dim,
            num_layers=num_gnn_layers,
            layer_type=gnn_type,
            num_heads=num_heads // 2
        )
        
        # Transformer for refining representations
        self.transformer_layers = nn.ModuleList([
            TransformerEncoderLayer(
                embed_dim=embed_dim,
                num_heads=num_heads,
                ff_dim=embed_dim * 4
            )
            for _ in range(num_transformer_layers)
        ])
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        
    def forward(
        self,
        node_features: torch.Tensor,
        distance_matrix: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            node_features: (batch_size, num_nodes, input_dim)
            distance_matrix: (batch_size, num_nodes, num_nodes)
        
        Returns:
            node_embeddings: (batch_size, num_nodes, embed_dim)
        """
        # GNN encoding (graph structure)
        gnn_embeddings = self.gnn(node_features, distance_matrix, self.k_neighbors)
        
        # Transformer encoding (sequence modeling)
        transformer_embeddings = gnn_embeddings
        for layer in self.transformer_layers:
            transformer_embeddings = layer(transformer_embeddings)
        
        # Fuse GNN and Transformer representations
        combined = torch.cat([gnn_embeddings, transformer_embeddings], dim=-1)
        fused_embeddings = self.fusion(combined)
        
        return fused_embeddings


class HybridModel(nn.Module):
    """
    Hybrid VRP model combining GNN and Attention mechanisms.
    
    Architecture:
    1. GNN extracts graph structural features
    2. Transformer refines node representations
    3. Attention-based decoder for action selection
    """
    
    def __init__(
        self,
        input_dim: int = 7,
        embed_dim: int = 128,
        num_gnn_layers: int = 2,
        num_transformer_layers: int = 2,
        gnn_type: str = 'gat',
        num_heads: int = 8,
        k_neighbors: int = None
    ):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Hybrid encoder
        self.encoder = HybridEncoder(
            input_dim=input_dim,
            embed_dim=embed_dim,
            num_gnn_layers=num_gnn_layers,
            num_transformer_layers=num_transformer_layers,
            gnn_type=gnn_type,
            num_heads=num_heads,
            k_neighbors=k_neighbors
        )
        
        # Decoder
        self.decoder = AttentionDecoder(embed_dim=embed_dim, num_heads=num_heads)
        
        # Graph pooling for global context
        self.graph_pooling = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
    def forward(
        self,
        node_features: torch.Tensor,
        current_node_idx: torch.Tensor,
        first_node_idx: torch.Tensor,
        remaining_capacity: torch.Tensor,
        mask: torch.Tensor,
        distance_matrix: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Single decoding step.
        
        Returns:
            log_probs: (batch_size, num_nodes)
        """
        batch_size = node_features.size(0)
        
        # Encode
        node_embeddings = self.encoder(node_features, distance_matrix)
        
        # Graph embedding
        graph_embedding = self.graph_pooling(node_embeddings.mean(dim=1))
        
        # Get current and first node embeddings
        current_node_emb = node_embeddings[torch.arange(batch_size), current_node_idx]
        first_node_emb = node_embeddings[torch.arange(batch_size), first_node_idx]
        
        # Decode
        log_probs = self.decoder(
            node_embeddings,
            graph_embedding,
            current_node_emb,
            first_node_emb,
            remaining_capacity,
            mask
        )
        
        return log_probs
    
    def encode(
        self,
        node_features: torch.Tensor,
        distance_matrix: torch.Tensor = None
    ) -> torch.Tensor:
        """Encode node features."""
        return self.encoder(node_features, distance_matrix)
