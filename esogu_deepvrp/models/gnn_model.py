"""
Graph Neural Network models for VRP.
Includes GCN, GAT, and hybrid architectures.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class GCNLayer(nn.Module):
    """
    Graph Convolutional Network layer (Kipf & Welling, 2017).
    Performs spectral convolution on graphs with degree normalization.
    """
    
    def __init__(self, in_dim: int, out_dim: int, activation: bool = True):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.activation = nn.ReLU() if activation else nn.Identity()
        
    def forward(
        self,
        node_features: torch.Tensor,
        adjacency: torch.Tensor,
        edge_weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            node_features: (batch_size, num_nodes, in_dim)
            adjacency: (batch_size, num_nodes, num_nodes) - adjacency matrix
            edge_weights: (batch_size, num_nodes, num_nodes) - optional edge weights
        
        Returns:
            output: (batch_size, num_nodes, out_dim)
        """
        # Add self-loops
        batch_size, num_nodes, _ = node_features.size()
        identity = torch.eye(num_nodes, device=node_features.device).unsqueeze(0).expand(batch_size, -1, -1)
        adjacency_with_self = adjacency + identity
        
        # Compute degree matrix for normalization
        degree = adjacency_with_self.sum(dim=-1, keepdim=True).clamp(min=1)
        degree_inv_sqrt = torch.pow(degree, -0.5)
        
        # Symmetric normalization: D^(-1/2) A D^(-1/2)
        norm_adj = adjacency_with_self * degree_inv_sqrt * degree_inv_sqrt.transpose(-2, -1)
        
        # Apply edge weights if provided
        if edge_weights is not None:
            norm_adj = norm_adj * edge_weights
        
        # Message passing: aggregate neighbor features
        aggregated = torch.matmul(norm_adj, node_features)
        
        # Linear transformation and activation
        output = self.activation(self.linear(aggregated))
        
        return output


class GATLayer(nn.Module):
    """
    Graph Attention Network layer (Veličković et al., 2018).
    Uses attention mechanism to learn importance of neighboring nodes.
    """
    
    def __init__(self, in_dim: int, out_dim: int, num_heads: int = 4, concat: bool = True):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.concat = concat
        
        # Linear transformations for each head
        self.W = nn.ModuleList([
            nn.Linear(in_dim, out_dim, bias=False) for _ in range(num_heads)
        ])
        
        # Attention mechanism parameters
        self.a = nn.ModuleList([
            nn.Linear(2 * out_dim, 1, bias=False) for _ in range(num_heads)
        ])
        
        self.leaky_relu = nn.LeakyReLU(0.2)
        
        # Output transformation if heads are concatenated
        if concat:
            self.out_proj = nn.Linear(num_heads * out_dim, out_dim)
        
    def forward(
        self,
        node_features: torch.Tensor,
        adjacency: torch.Tensor,
        edge_features: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            node_features: (batch_size, num_nodes, in_dim)
            adjacency: (batch_size, num_nodes, num_nodes) - binary adjacency
            edge_features: (batch_size, num_nodes, num_nodes, edge_dim) - optional
        
        Returns:
            output: (batch_size, num_nodes, out_dim)
            attention_weights: (batch_size, num_heads, num_nodes, num_nodes)
        """
        batch_size, num_nodes, _ = node_features.size()
        
        multi_head_outputs = []
        multi_head_attentions = []
        
        for head_idx in range(self.num_heads):
            # Linear transformation
            h = self.W[head_idx](node_features)  # (batch, num_nodes, out_dim)
            
            # Prepare for attention computation
            # Repeat h for source and target nodes
            h_i = h.unsqueeze(2).expand(-1, -1, num_nodes, -1)  # (batch, num_nodes, num_nodes, out_dim)
            h_j = h.unsqueeze(1).expand(-1, num_nodes, -1, -1)  # (batch, num_nodes, num_nodes, out_dim)
            
            # Concatenate source and target features
            concat_features = torch.cat([h_i, h_j], dim=-1)  # (batch, num_nodes, num_nodes, 2*out_dim)
            
            # Compute attention coefficients
            e = self.leaky_relu(self.a[head_idx](concat_features).squeeze(-1))  # (batch, num_nodes, num_nodes)
            
            # Mask attention for non-existent edges
            e = e.masked_fill(adjacency == 0, float('-inf'))
            
            # Normalize attention coefficients
            alpha = F.softmax(e, dim=-1)  # (batch, num_nodes, num_nodes)
            
            # Aggregate neighbor features
            output = torch.matmul(alpha, h)  # (batch, num_nodes, out_dim)
            
            multi_head_outputs.append(output)
            multi_head_attentions.append(alpha)
        
        # Combine multi-head outputs
        if self.concat:
            output = torch.cat(multi_head_outputs, dim=-1)
            output = self.out_proj(output)
        else:
            output = torch.stack(multi_head_outputs, dim=0).mean(dim=0)
        
        attention_weights = torch.stack(multi_head_attentions, dim=1)
        
        return output, attention_weights


class GNN(nn.Module):
    """
    General Graph Neural Network for VRP.
    Supports GCN and GAT layers.
    """
    
    def __init__(
        self,
        input_dim: int = 7,
        hidden_dim: int = 128,
        output_dim: int = 128,
        num_layers: int = 3,
        layer_type: str = 'gat',
        num_heads: int = 4
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.layer_type = layer_type.lower()
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # GNN layers
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for i in range(num_layers):
            in_dim = hidden_dim
            out_dim = hidden_dim if i < num_layers - 1 else output_dim
            
            if self.layer_type == 'gcn':
                self.layers.append(GCNLayer(in_dim, out_dim, activation=(i < num_layers - 1)))
            elif self.layer_type == 'gat':
                self.layers.append(GATLayer(in_dim, out_dim, num_heads=num_heads))
            else:
                raise ValueError(f"Unknown layer type: {layer_type}")
            
            self.norms.append(nn.LayerNorm(out_dim))
        
    def create_adjacency_matrix(
        self,
        num_nodes: int,
        batch_size: int,
        device: torch.device,
        k_neighbors: Optional[int] = None,
        distance_matrix: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Create adjacency matrix.
        If k_neighbors is None, creates fully connected graph.
        Otherwise, creates k-nearest neighbor graph.
        """
        if k_neighbors is None:
            # Fully connected (excluding self-loops initially)
            adj = torch.ones(batch_size, num_nodes, num_nodes, device=device)
            adj = adj - torch.eye(num_nodes, device=device).unsqueeze(0)
        else:
            # k-nearest neighbors
            if distance_matrix is None:
                raise ValueError("distance_matrix required for k-NN graph")
            
            # Get k nearest neighbors for each node
            _, indices = torch.topk(distance_matrix, k=k_neighbors + 1, dim=-1, largest=False)
            indices = indices[:, :, 1:]  # Exclude self
            
            # Create adjacency matrix
            adj = torch.zeros(batch_size, num_nodes, num_nodes, device=device)
            batch_indices = torch.arange(batch_size, device=device).view(-1, 1, 1).expand(-1, num_nodes, k_neighbors)
            node_indices = torch.arange(num_nodes, device=device).view(1, -1, 1).expand(batch_size, -1, k_neighbors)
            
            adj[batch_indices, node_indices, indices] = 1
        
        return adj
    
    def forward(
        self,
        node_features: torch.Tensor,
        distance_matrix: Optional[torch.Tensor] = None,
        k_neighbors: Optional[int] = None
    ) -> torch.Tensor:
        """
        Args:
            node_features: (batch_size, num_nodes, input_dim)
            distance_matrix: (batch_size, num_nodes, num_nodes) - optional
            k_neighbors: int - number of nearest neighbors (None for fully connected)
        
        Returns:
            node_embeddings: (batch_size, num_nodes, output_dim)
        """
        batch_size, num_nodes, _ = node_features.size()
        device = node_features.device
        
        # Create adjacency matrix
        adjacency = self.create_adjacency_matrix(
            num_nodes, batch_size, device, k_neighbors, distance_matrix
        )
        
        # Input projection
        x = self.input_proj(node_features)
        
        # Apply GNN layers
        for i, (layer, norm) in enumerate(zip(self.layers, self.norms)):
            if self.layer_type == 'gat':
                x_new, _ = layer(x, adjacency)
            else:
                x_new = layer(x, adjacency)
            
            # Residual connection and normalization
            if i > 0 and x.size(-1) == x_new.size(-1):
                x = norm(x + x_new)
            else:
                x = norm(x_new)
            
            # Apply ReLU except for last layer
            if i < self.num_layers - 1:
                x = F.relu(x)
        
        return x


class GNNDecoder(nn.Module):
    """
    Decoder for GNN-based VRP solver.
    Uses node embeddings to compute action probabilities.
    """
    
    def __init__(self, embed_dim: int = 128):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Context encoder
        self.context_encoder = nn.Sequential(
            nn.Linear(embed_dim * 3 + 1, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
        # Action scorer
        self.scorer = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1)
        )
        
    def forward(
        self,
        node_embeddings: torch.Tensor,
        current_node_idx: torch.Tensor,
        first_node_idx: torch.Tensor,
        remaining_capacity: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            node_embeddings: (batch_size, num_nodes, embed_dim)
            current_node_idx: (batch_size,)
            first_node_idx: (batch_size,)
            remaining_capacity: (batch_size, 1)
            mask: (batch_size, num_nodes)
        
        Returns:
            log_probs: (batch_size, num_nodes)
        """
        batch_size, num_nodes, _ = node_embeddings.size()
        
        # Get embeddings
        graph_emb = node_embeddings.mean(dim=1)
        current_emb = node_embeddings[torch.arange(batch_size), current_node_idx]
        first_emb = node_embeddings[torch.arange(batch_size), first_node_idx]
        
        # Create context
        context = torch.cat([graph_emb, current_emb, first_emb, remaining_capacity], dim=-1)
        context_emb = self.context_encoder(context)  # (batch_size, embed_dim)
        
        # Expand context for all nodes
        context_expanded = context_emb.unsqueeze(1).expand(-1, num_nodes, -1)
        
        # Compute scores for each node
        combined = torch.cat([node_embeddings, context_expanded], dim=-1)
        scores = self.scorer(combined).squeeze(-1)  # (batch_size, num_nodes)
        
        # Apply mask
        scores = scores.masked_fill(~mask, float('-inf'))
        
        # Log probabilities
        log_probs = F.log_softmax(scores, dim=-1)
        
        return log_probs


class GNNModel(nn.Module):
    """
    Complete GNN-based model for VRP.
    """
    
    def __init__(
        self,
        input_dim: int = 7,
        embed_dim: int = 128,
        num_layers: int = 3,
        layer_type: str = 'gat',
        num_heads: int = 4,
        k_neighbors: Optional[int] = None
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.k_neighbors = k_neighbors
        
        # GNN encoder
        self.encoder = GNN(
            input_dim=input_dim,
            hidden_dim=embed_dim,
            output_dim=embed_dim,
            num_layers=num_layers,
            layer_type=layer_type,
            num_heads=num_heads
        )
        
        # Decoder
        self.decoder = GNNDecoder(embed_dim=embed_dim)
        
    def forward(
        self,
        node_features: torch.Tensor,
        current_node_idx: torch.Tensor,
        first_node_idx: torch.Tensor,
        remaining_capacity: torch.Tensor,
        mask: torch.Tensor,
        distance_matrix: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Single decoding step.
        
        Returns:
            log_probs: (batch_size, num_nodes)
        """
        # Encode
        node_embeddings = self.encoder(node_features, distance_matrix, self.k_neighbors)
        
        # Decode
        log_probs = self.decoder(
            node_embeddings,
            current_node_idx,
            first_node_idx,
            remaining_capacity,
            mask
        )
        
        return log_probs
    
    def encode(
        self,
        node_features: torch.Tensor,
        distance_matrix: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Encode node features."""
        return self.encoder(node_features, distance_matrix, self.k_neighbors)


class GNNVRPModel(nn.Module):
    """
    GNN-based model for VRP (GCN or GAT).
    """
    
    def __init__(
        self,
        input_dim: int = 7,
        hidden_dim: int = 128,
        num_layers: int = 3,
        layer_type: str = 'gat',
        num_heads: int = 4
    ):
        super().__init__()
        self.gnn = GNN(
            input_dim, hidden_dim, hidden_dim,
            num_layers, layer_type, num_heads
        )
        
        # Context embedding
        self.context_proj = nn.Linear(hidden_dim + 1, hidden_dim)
        
        # Pointer mechanism
        self.W_q = nn.Linear(hidden_dim, hidden_dim)
        self.W_k = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1)
        
    def forward(
        self,
        node_features: torch.Tensor,
        adjacency: torch.Tensor,
        current_node_idx: torch.Tensor,
        remaining_capacity: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            node_features: (batch, num_nodes, input_dim)
            adjacency: (batch, num_nodes, num_nodes)
            current_node_idx: (batch,)
            remaining_capacity: (batch, 1)
            mask: (batch, num_nodes)
        
        Returns:
            log_probs: (batch, num_nodes)
        """
        batch_size, num_nodes, _ = node_features.size()
        
        # Encode with GNN
        node_embeddings = self.gnn(node_features, adjacency)
        
        # Current node embedding
        current_emb = node_embeddings[torch.arange(batch_size), current_node_idx]
        
        # Context with capacity
        context = torch.cat([current_emb, remaining_capacity], dim=-1)
        context_emb = self.context_proj(context)
        
        # Attention scores
        Q = self.W_q(context_emb).unsqueeze(1)  # (batch, 1, hidden)
        K = self.W_k(node_embeddings)  # (batch, num_nodes, hidden)
        
        energies = torch.tanh(Q + K)
        scores = self.v(energies).squeeze(-1)  # (batch, num_nodes)
        
        # Apply mask
        scores = scores.masked_fill(~mask, float('-inf'))
        
        log_probs = F.log_softmax(scores, dim=-1)
        
        return log_probs
