"""
Sequence-to-Sequence Model for VRP.
Encoder-Decoder architecture with attention.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class Seq2SeqEncoder(nn.Module):
    """Encoder with LSTM/GRU."""
    
    def __init__(
        self,
        input_dim: int = 7,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = True,
        rnn_type: str = 'LSTM'
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        RNN = nn.LSTM if rnn_type == 'LSTM' else nn.GRU
        self.rnn = RNN(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Project bidirectional outputs to hidden_dim
        if bidirectional:
            self.projection = nn.Linear(hidden_dim * 2, hidden_dim)
        else:
            self.projection = nn.Identity()
    
    def forward(self, node_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            node_features: (batch, num_nodes, input_dim)
        
        Returns:
            outputs: (batch, num_nodes, hidden_dim)
            hidden: (batch, hidden_dim) - last hidden state
        """
        outputs, state = self.rnn(node_features)
        
        # Project if bidirectional
        outputs = self.projection(outputs)
        
        # Get last hidden state
        if isinstance(state, tuple):  # LSTM
            hidden = state[0]  # h_n: (num_layers * num_directions, batch, hidden)
        else:  # GRU
            hidden = state
        
        # Take last layer, combine directions if bidirectional
        if self.bidirectional:
            hidden = hidden[-2:]  # Last layer both directions
            hidden = hidden.transpose(0, 1).contiguous().view(hidden.size(1), -1)
            hidden = self.projection(hidden)  # (batch, hidden_dim)
        else:
            hidden = hidden[-1]  # (batch, hidden_dim)
        
        return outputs, hidden


class BahdanauAttention(nn.Module):
    """Bahdanau attention mechanism."""
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.W_h = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_s = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v = nn.Linear(hidden_dim, 1, bias=False)
    
    def forward(
        self,
        encoder_outputs: torch.Tensor,
        decoder_state: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            encoder_outputs: (batch, num_nodes, hidden_dim)
            decoder_state: (batch, hidden_dim)
            mask: (batch, num_nodes)
        
        Returns:
            context: (batch, hidden_dim)
            attention_weights: (batch, num_nodes)
        """
        # Compute attention scores
        encoder_transform = self.W_h(encoder_outputs)  # (batch, num_nodes, hidden)
        decoder_transform = self.W_s(decoder_state).unsqueeze(1)  # (batch, 1, hidden)
        
        energies = torch.tanh(encoder_transform + decoder_transform)
        scores = self.v(energies).squeeze(-1)  # (batch, num_nodes)
        
        # Apply mask
        if mask is not None:
            scores = scores.masked_fill(~mask, float('-inf'))
        
        # Attention weights
        attention_weights = F.softmax(scores, dim=-1)
        
        # Context vector
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        
        return context, attention_weights


class Seq2SeqDecoder(nn.Module):
    """Decoder with attention."""
    
    def __init__(
        self,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        rnn_type: str = 'LSTM'
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Attention
        self.attention = BahdanauAttention(hidden_dim)
        
        # RNN (input: context + capacity)
        RNN = nn.LSTM if rnn_type == 'LSTM' else nn.GRU
        self.rnn = RNN(
            hidden_dim + 1,  # context + capacity
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Output projection
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(
        self,
        encoder_outputs: torch.Tensor,
        decoder_hidden: torch.Tensor,
        remaining_capacity: torch.Tensor,
        mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            encoder_outputs: (batch, num_nodes, hidden_dim)
            decoder_hidden: (batch, hidden_dim)
            remaining_capacity: (batch, 1)
            mask: (batch, num_nodes)
        
        Returns:
            log_probs: (batch, num_nodes)
            attention_weights: (batch, num_nodes)
        """
        # Get context via attention
        context, attention_weights = self.attention(
            encoder_outputs, decoder_hidden, mask
        )
        
        # Prepare RNN input
        rnn_input = torch.cat([context, remaining_capacity], dim=-1).unsqueeze(1)
        
        # RNN step
        output, _ = self.rnn(rnn_input)
        output = output.squeeze(1)  # (batch, hidden_dim)
        
        # Compute scores for all nodes
        output = self.out_proj(output)  # (batch, hidden_dim)
        scores = torch.bmm(output.unsqueeze(1), encoder_outputs.transpose(1, 2)).squeeze(1)
        # (batch, num_nodes)
        
        # Apply mask
        scores = scores.masked_fill(~mask, float('-inf'))
        
        # Log probabilities
        log_probs = F.log_softmax(scores, dim=-1)
        
        return log_probs, attention_weights


class Seq2SeqModel(nn.Module):
    """
    Sequence-to-Sequence model for VRP.
    Encoder-Decoder with Bahdanau attention.
    """
    
    def __init__(
        self,
        input_dim: int = 7,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = True,
        rnn_type: str = 'LSTM'
    ):
        super().__init__()
        self.encoder = Seq2SeqEncoder(
            input_dim, hidden_dim, num_layers, dropout, bidirectional, rnn_type
        )
        self.decoder = Seq2SeqDecoder(hidden_dim, num_layers, dropout, rnn_type)
        self.hidden_dim = hidden_dim
    
    def forward(
        self,
        node_features: torch.Tensor,
        current_node_idx: torch.Tensor,
        remaining_capacity: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            node_features: (batch, num_nodes, input_dim)
            current_node_idx: (batch,)
            remaining_capacity: (batch, 1)
            mask: (batch, num_nodes)
        
        Returns:
            log_probs: (batch, num_nodes)
        """
        batch_size = node_features.size(0)
        
        # Encode
        encoder_outputs, encoder_hidden = self.encoder(node_features)
        
        # Use current node as decoder state
        decoder_hidden = encoder_outputs[torch.arange(batch_size), current_node_idx]
        
        # Decode
        log_probs, _ = self.decoder(
            encoder_outputs, decoder_hidden, remaining_capacity, mask
        )
        
        return log_probs
