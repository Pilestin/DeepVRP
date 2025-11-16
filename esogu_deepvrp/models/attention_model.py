"""
Attention-based model for VRP following Kool et al. (2019).
Multi-head attention encoder-decoder architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism."""
    
    def __init__(self, embed_dim: int = 128, num_heads: int = 8):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = math.sqrt(self.head_dim)
        
        # Linear projections
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: (batch_size, query_len, embed_dim)
            key: (batch_size, key_len, embed_dim)
            value: (batch_size, key_len, embed_dim)
            mask: (batch_size, query_len, key_len) - True for valid positions
        
        Returns:
            output: (batch_size, query_len, embed_dim)
            attention_weights: (batch_size, num_heads, query_len, key_len)
        """
        batch_size = query.size(0)
        
        # Project and reshape: (batch, length, embed) -> (batch, num_heads, length, head_dim)
        Q = self.q_proj(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        # (batch, num_heads, query_len, head_dim) @ (batch, num_heads, head_dim, key_len)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # Apply mask if provided
        if mask is not None:
            # Expand mask for all heads: (batch, 1, query_len, key_len)
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(~mask, float('-inf'))
        
        # Attention weights
        attn_weights = F.softmax(scores, dim=-1)
        
        # Apply attention to values
        output = torch.matmul(attn_weights, V)
        
        # Reshape: (batch, num_heads, query_len, head_dim) -> (batch, query_len, embed_dim)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
        
        # Final projection
        output = self.out_proj(output)
        
        return output, attn_weights


class FeedForward(nn.Module):
    """Position-wise feed-forward network."""
    
    def __init__(self, embed_dim: int = 128, ff_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerEncoderLayer(nn.Module):
    """Single transformer encoder layer with multi-head attention and FFN."""
    
    def __init__(self, embed_dim: int = 128, num_heads: int = 8, ff_dim: int = 512):
        super().__init__()
        self.attention = MultiHeadAttention(embed_dim, num_heads)
        self.ffn = FeedForward(embed_dim, ff_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, embed_dim)
            mask: (batch_size, seq_len, seq_len)
        
        Returns:
            output: (batch_size, seq_len, embed_dim)
        """
        # Self-attention with residual connection
        attn_output, _ = self.attention(x, x, x, mask)
        x = self.norm1(x + attn_output)
        
        # Feed-forward with residual connection
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)
        
        return x


class AttentionEncoder(nn.Module):
    """Transformer encoder for VRP."""
    
    def __init__(
        self,
        input_dim: int = 7,
        embed_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 3,
        ff_dim: int = 512
    ):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Input embedding
        self.input_embedding = nn.Linear(input_dim, embed_dim)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, ff_dim)
            for _ in range(num_layers)
        ])
        
    def forward(self, node_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            node_features: (batch_size, num_nodes, input_dim)
        
        Returns:
            node_embeddings: (batch_size, num_nodes, embed_dim)
        """
        # Initial embedding
        x = self.input_embedding(node_features)
        
        # Pass through transformer layers
        for layer in self.layers:
            x = layer(x)
        
        return x


class AttentionDecoder(nn.Module):
    """
    Autoregressive decoder for VRP using attention mechanism.
    Selects next node to visit based on current context.
    """
    
    def __init__(self, embed_dim: int = 128, num_heads: int = 8):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        # Context embedding (graph context + current node + first node + capacity info)
        self.context_embedding = nn.Linear(embed_dim * 3 + 1, embed_dim)
        
        # Attention for action selection
        self.attention = MultiHeadAttention(embed_dim, num_heads)
        
        # Pointer mechanism
        self.pointer = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.Tanh(),
            nn.Linear(embed_dim, 1)
        )
        
    def forward(
        self,
        node_embeddings: torch.Tensor,
        graph_embedding: torch.Tensor,
        current_node_emb: torch.Tensor,
        first_node_emb: torch.Tensor,
        remaining_capacity: torch.Tensor,
        mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            node_embeddings: (batch_size, num_nodes, embed_dim)
            graph_embedding: (batch_size, embed_dim)
            current_node_emb: (batch_size, embed_dim)
            first_node_emb: (batch_size, embed_dim)
            remaining_capacity: (batch_size, 1)
            mask: (batch_size, num_nodes) - True for valid nodes
        
        Returns:
            log_probs: (batch_size, num_nodes)
            selected_nodes: (batch_size,)
        """
        batch_size, num_nodes, _ = node_embeddings.size()
        
        # Create context vector
        context = torch.cat([
            graph_embedding,
            current_node_emb,
            first_node_emb,
            remaining_capacity
        ], dim=-1)
        
        # Embed context
        context_emb = self.context_embedding(context)  # (batch_size, embed_dim)
        context_emb = context_emb.unsqueeze(1)  # (batch_size, 1, embed_dim)
        
        # Compute attention scores
        scores = self.pointer(node_embeddings).squeeze(-1)  # (batch_size, num_nodes)
        
        # Apply mask (set invalid nodes to -inf)
        scores = scores.masked_fill(~mask, float('-inf'))
        
        # Compute probabilities
        log_probs = F.log_softmax(scores, dim=-1)
        
        return log_probs


class AttentionModel(nn.Module):
    """
    Complete Attention Model for VRP.
    Encoder-decoder architecture with multi-head attention.
    """
    
    def __init__(
        self,
        input_dim: int = 7,
        embed_dim: int = 128,
        num_heads: int = 8,
        num_encoder_layers: int = 3,
        ff_dim: int = 512
    ):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Encoder
        self.encoder = AttentionEncoder(
            input_dim=input_dim,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_encoder_layers,
            ff_dim=ff_dim
        )
        
        # Decoder
        self.decoder = AttentionDecoder(embed_dim=embed_dim, num_heads=num_heads)
        
        # Graph-level embedding (mean pooling)
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
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Single decoding step.
        
        Args:
            node_features: (batch_size, num_nodes, input_dim)
            current_node_idx: (batch_size,)
            first_node_idx: (batch_size,)
            remaining_capacity: (batch_size, 1)
            mask: (batch_size, num_nodes)
        
        Returns:
            log_probs: (batch_size, num_nodes)
        """
        batch_size = node_features.size(0)
        
        # Encode all nodes
        node_embeddings = self.encoder(node_features)
        
        # Graph embedding (mean pooling)
        graph_embedding = self.graph_pooling(node_embeddings.mean(dim=1))
        
        # Get current node and first node embeddings
        current_node_emb = node_embeddings[torch.arange(batch_size), current_node_idx]
        first_node_emb = node_embeddings[torch.arange(batch_size), first_node_idx]
        
        # Decode (select next node)
        log_probs = self.decoder(
            node_embeddings,
            graph_embedding,
            current_node_emb,
            first_node_emb,
            remaining_capacity,
            mask
        )
        
        return log_probs
    
    def encode(self, node_features: torch.Tensor) -> torch.Tensor:
        """
        Encode node features (can be called once for efficiency).
        
        Returns:
            node_embeddings: (batch_size, num_nodes, embed_dim)
        """
        return self.encoder(node_features)
