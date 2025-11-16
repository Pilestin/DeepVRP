"""
Model factory for creating VRP models.
"""

import torch.nn as nn
from models.attention_model import AttentionModel
from models.gnn_model import GNNVRPModel
from models.hybrid_model import HybridModel
from models.pointer_network import PointerNetwork
from models.seq2seq_model import Seq2SeqModel


def create_vrp_model(
    model_type: str,
    input_dim: int = 7,
    hidden_dim: int = 128,
    **kwargs
) -> nn.Module:
    """
    Create a VRP model.
    
    Args:
        model_type: 'attention', 'gcn', 'gat', 'hybrid', 'pointer', 'seq2seq'
        input_dim: Input feature dimension
        hidden_dim: Hidden dimension
        **kwargs: Model-specific arguments
    
    Returns:
        model: PyTorch model
    """
    model_type = model_type.lower()
    
    if model_type == 'attention':
        model = AttentionModel(
            input_dim=input_dim,
            embed_dim=hidden_dim,
            num_heads=kwargs.get('num_heads', 8),
            num_encoder_layers=kwargs.get('num_layers', 3),
            ff_dim=kwargs.get('ff_dim', 512)
        )
    
    elif model_type == 'gcn':
        model = GNNVRPModel(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=kwargs.get('num_layers', 3),
            layer_type='gcn'
        )
    
    elif model_type == 'gat':
        model = GNNVRPModel(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=kwargs.get('num_layers', 3),
            layer_type='gat',
            num_heads=kwargs.get('num_heads', 4)
        )
    
    elif model_type == 'hybrid':
        model = HybridModel(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_gnn_layers=kwargs.get('num_gnn_layers', 2),
            num_attention_layers=kwargs.get('num_attention_layers', 2),
            num_heads=kwargs.get('num_heads', 8)
        )
    
    elif model_type == 'pointer':
        model = PointerNetwork(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=kwargs.get('num_layers', 2),
            dropout=kwargs.get('dropout', 0.1)
        )
    
    elif model_type == 'seq2seq':
        model = Seq2SeqModel(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=kwargs.get('num_layers', 2),
            dropout=kwargs.get('dropout', 0.1),
            bidirectional=kwargs.get('bidirectional', True),
            rnn_type=kwargs.get('rnn_type', 'LSTM')
        )
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model


def get_model_params(model: nn.Module) -> int:
    """Count model parameters."""
    return sum(p.numel() for p in model.parameters())
