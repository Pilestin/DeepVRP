"""
Model ve embedding testleri.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from model.embeddings import NodeEmbedding, GraphEmbedding
from model.transforms import create_attention_mask


def test_embeddings(dl_data):
    """Embedding modellerini test et."""
    
    print("\n" + "="*60)
    print("TESTING EMBEDDINGS")
    print("="*60)
    
    # Node features
    node_features = dl_data['node_features'].unsqueeze(0)  # Add batch dimension
    distance_matrix = dl_data['distance_matrix'].unsqueeze(0)
    energy_matrix = dl_data['energy_matrix'].unsqueeze(0)
    
    print(f"\nInput Shapes:")
    print(f"  Node Features: {node_features.shape}")
    print(f"  Distance Matrix: {distance_matrix.shape}")
    print(f"  Energy Matrix: {energy_matrix.shape}")
    
    # Test NodeEmbedding
    print("\n--- NodeEmbedding Test ---")
    node_embedder = NodeEmbedding(embedding_dim=128)
    node_embeddings = node_embedder(node_features)
    print(f"✓ Node Embeddings: {node_embeddings.shape}")
    print(f"  Expected: (1, 11, 128)")
    
    # Test GraphEmbedding
    print("\n--- GraphEmbedding Test ---")
    graph_embedder = GraphEmbedding(node_embedding_dim=128, graph_embedding_dim=128)
    node_emb, graph_emb = graph_embedder(node_features, distance_matrix, energy_matrix)
    print(f"✓ Node Embeddings: {node_emb.shape}")
    print(f"✓ Graph Embedding: {graph_emb.shape}")
    print(f"  Expected: (1, 11, 128) and (1, 128)")
    
    # Test Attention Mask
    print("\n--- Attention Mask Test ---")
    mask = create_attention_mask(11)
    print(f"✓ Attention Mask: {mask.shape}")
    print(f"  All True (unvisited): {mask.all()}")
    
    # Visited mask test
    import numpy as np
    visited = np.array([True, False, False, True, False, False, False, False, False, False, False])
    mask_with_visited = create_attention_mask(11, visited)
    print(f"✓ Mask with 2 visited: {mask_with_visited.sum().item()} available nodes")
    
    print("\n" + "="*60)
    print("ALL TESTS PASSED!")
    print("="*60 + "\n")
    
    return {
        'node_embedder': node_embedder,
        'graph_embedder': graph_embedder,
        'sample_node_embeddings': node_embeddings,
        'sample_graph_embedding': graph_emb
    }


if __name__ == "__main__":
    print("Please run main.py first to generate dl_data, then call this function.")
