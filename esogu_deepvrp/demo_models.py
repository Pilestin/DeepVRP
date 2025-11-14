"""
Model architecture demonstration and testing.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from model.model_factory import create_model, print_model_summary, get_model_info


def demo_all_models():
    """Demonstrate all available model architectures."""
    
    print("\n" + "=" * 70)
    print("DEEP LEARNING MODELS FOR VRP - ARCHITECTURE DEMONSTRATION")
    print("=" * 70)
    
    # Model types to demonstrate
    model_types = ['attention', 'gnn_gcn', 'gnn_gat', 'hybrid']
    
    # Sample input
    batch_size = 4
    num_nodes = 11  # 1 depot + 10 customers
    input_dim = 7
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Create sample data
    node_features = torch.randn(batch_size, num_nodes, input_dim).to(device)
    distance_matrix = torch.rand(batch_size, num_nodes, num_nodes).to(device)
    current_node_idx = torch.zeros(batch_size, dtype=torch.long).to(device)
    first_node_idx = torch.zeros(batch_size, dtype=torch.long).to(device)
    remaining_capacity = torch.ones(batch_size, 1).to(device)
    mask = torch.ones(batch_size, num_nodes, dtype=torch.bool).to(device)
    
    print(f"\nInput shapes:")
    print(f"  Node features: {node_features.shape}")
    print(f"  Distance matrix: {distance_matrix.shape}")
    
    for model_type in model_types:
        print("\n" + "=" * 70)
        
        # Get model information
        info = get_model_info(model_type)
        print(f"\n{info['name']}")
        print("-" * 70)
        print(f"Description: {info['description']}")
        print(f"Reference: {info['reference']}")
        print(f"\nAdvantages:")
        for adv in info['advantages']:
            print(f"  - {adv}")
        
        # Create model
        print(f"\nInitializing {model_type} model...")
        model = create_model(model_type, {'embed_dim': 128, 'num_heads': 8})
        model = model.to(device)
        model.eval()
        
        # Count parameters
        num_params = sum(p.numel() for p in model.parameters())
        print(f"Parameters: {num_params:,}")
        print(f"Model size: ~{num_params * 4 / 1024 / 1024:.2f} MB")
        
        # Forward pass
        print(f"\nTesting forward pass...")
        with torch.no_grad():
            try:
                if model_type in ['gnn_gcn', 'gnn_gat', 'hybrid']:
                    log_probs = model(
                        node_features,
                        current_node_idx,
                        first_node_idx,
                        remaining_capacity,
                        mask,
                        distance_matrix
                    )
                else:
                    log_probs = model(
                        node_features,
                        current_node_idx,
                        first_node_idx,
                        remaining_capacity,
                        mask
                    )
                
                print(f"Output shape: {log_probs.shape}")
                print(f"Output range: [{log_probs.min():.3f}, {log_probs.max():.3f}]")
                print(f"Sum of probabilities: {log_probs.exp().sum(dim=1).mean():.3f} (should be ~1.0)")
                print(f"SUCCESS")
            
            except Exception as e:
                print(f"ERROR: {str(e)}")
    
    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)


def compare_models():
    """Compare different model architectures."""
    
    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test configuration
    configs = [
        ('attention', {'embed_dim': 128, 'num_heads': 8, 'num_encoder_layers': 3}),
        ('gnn_gcn', {'embed_dim': 128, 'num_layers': 3}),
        ('gnn_gat', {'embed_dim': 128, 'num_layers': 3, 'num_heads': 4}),
        ('hybrid', {'embed_dim': 128, 'num_gnn_layers': 2, 'num_transformer_layers': 2})
    ]
    
    print(f"\n{'Model':<20} {'Parameters':<15} {'Size (MB)':<12} {'Inference Time (ms)':<20}")
    print("-" * 70)
    
    # Sample input
    batch_size = 8
    num_nodes = 21
    node_features = torch.randn(batch_size, num_nodes, 7).to(device)
    distance_matrix = torch.rand(batch_size, num_nodes, num_nodes).to(device)
    current_node_idx = torch.zeros(batch_size, dtype=torch.long).to(device)
    first_node_idx = torch.zeros(batch_size, dtype=torch.long).to(device)
    remaining_capacity = torch.ones(batch_size, 1).to(device)
    mask = torch.ones(batch_size, num_nodes, dtype=torch.bool).to(device)
    
    for model_type, config in configs:
        model = create_model(model_type, config).to(device)
        model.eval()
        
        num_params = sum(p.numel() for p in model.parameters())
        size_mb = num_params * 4 / 1024 / 1024
        
        # Measure inference time
        import time
        with torch.no_grad():
            # Warmup
            for _ in range(5):
                if model_type in ['gnn_gcn', 'gnn_gat', 'hybrid']:
                    _ = model(node_features, current_node_idx, first_node_idx, 
                            remaining_capacity, mask, distance_matrix)
                else:
                    _ = model(node_features, current_node_idx, first_node_idx,
                            remaining_capacity, mask)
            
            # Measure
            start = time.time()
            for _ in range(20):
                if model_type in ['gnn_gcn', 'gnn_gat', 'hybrid']:
                    _ = model(node_features, current_node_idx, first_node_idx,
                            remaining_capacity, mask, distance_matrix)
                else:
                    _ = model(node_features, current_node_idx, first_node_idx,
                            remaining_capacity, mask)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            elapsed = (time.time() - start) / 20 * 1000
        
        print(f"{model_type:<20} {num_params:<15,} {size_mb:<12.2f} {elapsed:<20.2f}")
    
    print("=" * 70)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Model demonstration')
    parser.add_argument('--mode', choices=['demo', 'compare', 'both'], default='both',
                       help='Demonstration mode')
    
    args = parser.parse_args()
    
    if args.mode in ['demo', 'both']:
        demo_all_models()
    
    if args.mode in ['compare', 'both']:
        compare_models()
