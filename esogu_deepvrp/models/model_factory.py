"""
Model factory for creating different VRP solving models.
"""

import torch
import torch.nn as nn
from typing import Dict, Any
from .attention_model import AttentionModel
from .gnn_model import GNNModel
from .hybrid_model import HybridModel


def create_model(model_type: str, config: Dict[str, Any] = None) -> nn.Module:
    """
    Factory function to create VRP solving models.
    
    Args:
        model_type: Type of model ('attention', 'gnn_gcn', 'gnn_gat', 'hybrid')
        config: Model configuration dictionary
    
    Returns:
        Initialized model
    
    Examples:
        >>> model = create_model('attention', {'embed_dim': 128, 'num_heads': 8})
        >>> model = create_model('gnn_gat', {'embed_dim': 128, 'num_layers': 3})
        >>> model = create_model('hybrid', {'embed_dim': 128})
    """
    if config is None:
        config = {}
    
    # Default configurations
    default_config = {
        'input_dim': 7,
        'embed_dim': 128,
        'num_heads': 8,
        'num_layers': 3,
        'num_encoder_layers': 3,
        'ff_dim': 512,
        'k_neighbors': None
    }
    
    # Merge with provided config
    final_config = {**default_config, **config}
    
    model_type = model_type.lower()
    
    if model_type == 'attention':
        model = AttentionModel(
            input_dim=final_config['input_dim'],
            embed_dim=final_config['embed_dim'],
            num_heads=final_config['num_heads'],
            num_encoder_layers=final_config['num_encoder_layers'],
            ff_dim=final_config['ff_dim']
        )
    
    elif model_type in ['gnn_gcn', 'gcn']:
        model = GNNModel(
            input_dim=final_config['input_dim'],
            embed_dim=final_config['embed_dim'],
            num_layers=final_config['num_layers'],
            layer_type='gcn',
            k_neighbors=final_config['k_neighbors']
        )
    
    elif model_type in ['gnn_gat', 'gat']:
        model = GNNModel(
            input_dim=final_config['input_dim'],
            embed_dim=final_config['embed_dim'],
            num_layers=final_config['num_layers'],
            layer_type='gat',
            num_heads=final_config.get('num_heads', 4),
            k_neighbors=final_config['k_neighbors']
        )
    
    elif model_type == 'hybrid':
        model = HybridModel(
            input_dim=final_config['input_dim'],
            embed_dim=final_config['embed_dim'],
            num_gnn_layers=final_config.get('num_gnn_layers', 2),
            num_transformer_layers=final_config.get('num_transformer_layers', 2),
            gnn_type=final_config.get('gnn_type', 'gat'),
            num_heads=final_config['num_heads'],
            k_neighbors=final_config['k_neighbors']
        )
    
    else:
        raise ValueError(f"Unknown model type: {model_type}. "
                        f"Choose from ['attention', 'gnn_gcn', 'gnn_gat', 'hybrid']")
    
    return model


def get_model_info(model_type: str) -> Dict[str, Any]:
    """
    Get information about a model type.
    
    Args:
        model_type: Type of model
    
    Returns:
        Dictionary with model information
    """
    info = {
        'attention': {
            'name': 'Attention Model',
            'description': 'Transformer-based encoder-decoder with multi-head attention',
            'reference': 'Kool et al. (2019) - Attention, Learn to Solve Routing Problems!',
            'advantages': [
                'Effective sequence modeling',
                'Parallelizable training',
                'Good generalization',
                'Interpretable attention weights'
            ],
            'hyperparameters': {
                'embed_dim': 'Embedding dimension (128-256)',
                'num_heads': 'Number of attention heads (8-16)',
                'num_encoder_layers': 'Number of transformer layers (3-6)',
                'ff_dim': 'Feed-forward dimension (512-2048)'
            }
        },
        'gnn_gcn': {
            'name': 'Graph Convolutional Network',
            'description': 'Spectral graph convolution with degree normalization',
            'reference': 'Kipf & Welling (2017) - Semi-Supervised Classification with GCN',
            'advantages': [
                'Exploits graph structure',
                'Efficient message passing',
                'Good for spatial problems',
                'Fast inference'
            ],
            'hyperparameters': {
                'embed_dim': 'Embedding dimension (128-256)',
                'num_layers': 'Number of GCN layers (2-5)',
                'k_neighbors': 'k-NN graph or fully connected (None)'
            }
        },
        'gnn_gat': {
            'name': 'Graph Attention Network',
            'description': 'Graph neural network with learned attention weights',
            'reference': 'Veličković et al. (2018) - Graph Attention Networks',
            'advantages': [
                'Adaptive edge importance',
                'Handles varying neighborhoods',
                'Interpretable attention',
                'Better than GCN for irregular graphs'
            ],
            'hyperparameters': {
                'embed_dim': 'Embedding dimension (128-256)',
                'num_layers': 'Number of GAT layers (2-5)',
                'num_heads': 'Number of attention heads (4-8)',
                'k_neighbors': 'k-NN graph or fully connected (None)'
            }
        },
        'hybrid': {
            'name': 'Hybrid GNN-Attention Model',
            'description': 'Combines GNN structural learning with Transformer refinement',
            'reference': 'Custom architecture',
            'advantages': [
                'Best of both worlds',
                'Graph structure + sequence modeling',
                'Highly expressive',
                'State-of-the-art potential'
            ],
            'hyperparameters': {
                'embed_dim': 'Embedding dimension (128-256)',
                'num_gnn_layers': 'Number of GNN layers (2-3)',
                'num_transformer_layers': 'Number of transformer layers (2-3)',
                'gnn_type': 'GNN type (gcn or gat)',
                'num_heads': 'Number of attention heads (8-16)'
            }
        }
    }
    
    return info.get(model_type.lower(), {'error': 'Unknown model type'})


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_summary(model: nn.Module, model_type: str = None):
    """
    Print a summary of the model.
    
    Args:
        model: PyTorch model
        model_type: Type of model (optional)
    """
    print("=" * 70)
    print("MODEL SUMMARY")
    print("=" * 70)
    
    if model_type:
        info = get_model_info(model_type)
        print(f"\nModel: {info.get('name', 'Unknown')}")
        print(f"Description: {info.get('description', 'N/A')}")
        print(f"Reference: {info.get('reference', 'N/A')}")
    
    print(f"\nArchitecture:")
    print(model)
    
    num_params = count_parameters(model)
    print(f"\nTotal Parameters: {num_params:,}")
    print(f"Model Size: ~{num_params * 4 / 1024 / 1024:.2f} MB (FP32)")
    
    print("=" * 70)
