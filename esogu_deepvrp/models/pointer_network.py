"""
Pointer Network for VRP (Vinyals et al., 2015).
Uses attention mechanism to point to input elements.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class PointerNetwork(nn.Module):
    """
    Pointer Network - pointing to input sequence.
    Original paper: "Pointer Networks" (Vinyals et al., 2015)
    """
    
    def __init__(
        self,
        input_dim: int = 7,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Encoder LSTM
        self.encoder = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Decoder LSTM
        self.decoder = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Attention layers
        self.W1 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v = nn.Linear(hidden_dim, 1, bias=False)
        
        # Context embedding for decoder input
        self.context_embed = nn.Linear(hidden_dim + 1, input_dim)
        
    def encode(self, node_features: torch.Tensor) -> Tuple[torch.Tensor, Tuple]:
        """
        Encode input sequence.
        
        Args:
            node_features: (batch_size, num_nodes, input_dim)
        
        Returns:
            encoder_outputs: (batch_size, num_nodes, hidden_dim)
            encoder_state: (h_n, c_n) for LSTM
        """
        encoder_outputs, encoder_state = self.encoder(node_features)
        return encoder_outputs, encoder_state
    
    def decode_step(
        self,
        decoder_input: torch.Tensor,
        decoder_state: Tuple,
        encoder_outputs: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Tuple]:
        """
        Single decoder step with pointing.
        
        Args:
            decoder_input: (batch_size, 1, input_dim)
            decoder_state: (h, c) from previous step
            encoder_outputs: (batch_size, num_nodes, hidden_dim)
            mask: (batch_size, num_nodes) - True for valid positions
        
        Returns:
            log_probs: (batch_size, num_nodes)
            decoder_state: Updated (h, c)
        """
        # Decoder step
        decoder_output, decoder_state = self.decoder(decoder_input, decoder_state)
        # decoder_output: (batch_size, 1, hidden_dim)
        
        # Attention mechanism
        # u_i = v^T tanh(W1 * e_i + W2 * d_t)
        encoder_transform = self.W1(encoder_outputs)  # (batch, num_nodes, hidden)
        decoder_transform = self.W2(decoder_output)    # (batch, 1, hidden)
        
        # Broadcast and combine
        energies = torch.tanh(encoder_transform + decoder_transform)
        scores = self.v(energies).squeeze(-1)  # (batch, num_nodes)
        
        # Apply mask
        if mask is not None:
            scores = scores.masked_fill(~mask, float('-inf'))
        
        # Compute probabilities
        log_probs = F.log_softmax(scores, dim=-1)
        
        return log_probs, decoder_state
    
    def forward(
        self,
        node_features: torch.Tensor,
        current_node_idx: torch.Tensor,
        remaining_capacity: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass for single decoding step.
        
        Args:
            node_features: (batch_size, num_nodes, input_dim)
            current_node_idx: (batch_size,) - currently selected node
            remaining_capacity: (batch_size, 1)
            mask: (batch_size, num_nodes)
        
        Returns:
            log_probs: (batch_size, num_nodes)
        """
        batch_size = node_features.size(0)
        
        # Encode
        encoder_outputs, encoder_state = self.encode(node_features)
        
        # Prepare decoder input (current node embedding + capacity)
        current_node_emb = node_features[torch.arange(batch_size), current_node_idx]
        context = torch.cat([current_node_emb, remaining_capacity], dim=-1)
        decoder_input = self.context_embed(context).unsqueeze(1)
        
        # Decode step
        log_probs, _ = self.decode_step(
            decoder_input, encoder_state, encoder_outputs, mask
        )
        
        return log_probs
