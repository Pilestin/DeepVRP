# Deep Learning Models Implementation Guide

## Overview

This document provides a comprehensive guide to the implementation of deep learning models for solving the Capacitated Electric Vehicle Routing Problem with Time Windows (CEVRPTW). Four distinct architectures are implemented, each offering unique advantages for different problem characteristics.

## Available Models

### 1. Attention Model

**Implementation:** `model/attention_model.py`

**Architecture:**
- Transformer-based encoder-decoder
- Multi-head self-attention mechanism
- Positional encoding through feature embedding
- Autoregressive decoding

**Key Components:**
- `MultiHeadAttention`: Scaled dot-product attention with multiple heads
- `TransformerEncoderLayer`: Self-attention + feed-forward with residual connections
- `AttentionEncoder`: Stack of transformer layers for node encoding
- `AttentionDecoder`: Attention-based next-node selection

**Hyperparameters:**
- `embed_dim`: 128 (embedding dimension)
- `num_heads`: 8 (number of attention heads)
- `num_encoder_layers`: 3 (transformer depth)
- `ff_dim`: 512 (feed-forward hidden dimension)

**Computational Complexity:**
- Encoding: O(n² d) where n = number of nodes, d = embedding dimension
- Decoding per step: O(n d)
- Total for one solution: O(n³ d)

**Advantages:**
- Captures long-range dependencies
- Parallelizable training
- Interpretable attention weights
- Good generalization across problem sizes

**When to Use:**
- General-purpose VRP solving
- When interpretability is important
- When training data is abundant
- Problems with complex constraint patterns

---

### 2. Graph Convolutional Network (GCN)

**Implementation:** `model/gnn_model.py` (layer_type='gcn')

**Architecture:**
- Spectral graph convolution
- Symmetric normalization by node degree
- Message passing on graph structure
- Stacked GCN layers with residual connections

**Key Components:**
- `GCNLayer`: Single graph convolution layer with degree normalization
- `GNN`: Multi-layer GCN encoder
- `GNNDecoder`: MLP-based action selection from node embeddings

**Mathematical Formulation:**
```
h_v^(k) = σ(Σ_{u∈N(v)∪{v}} (1/√(d_u d_v)) W^(k) h_u^(k-1))
```

**Hyperparameters:**
- `embed_dim`: 128
- `num_layers`: 3 (GCN depth)
- `k_neighbors`: None (fully connected) or int (k-NN graph)

**Computational Complexity:**
- Encoding: O(|E| d) where |E| = number of edges
- For fully connected: O(n² d)
- For k-NN graph: O(n k d)

**Advantages:**
- Efficient for sparse graphs
- Exploits spatial structure
- Fast inference
- Good inductive bias for routing problems

**When to Use:**
- Spatially structured problems
- Large-scale instances (with k-NN)
- When computational efficiency is critical
- Problems with clear spatial patterns

---

### 3. Graph Attention Network (GAT)

**Implementation:** `model/gnn_model.py` (layer_type='gat')

**Architecture:**
- Attention-based message passing
- Learns edge importance adaptively
- Multi-head attention for graphs
- Layer-wise attention coefficients

**Key Components:**
- `GATLayer`: Graph attention layer with learned attention weights
- Multi-head mechanism for diverse relationship learning
- LeakyReLU activation for attention computation

**Mathematical Formulation:**
```
α_vu = softmax(LeakyReLU(a^T [W h_v || W h_u]))
h_v^(k) = σ(Σ_{u∈N(v)} α_vu W^(k) h_u^(k-1))
```

**Hyperparameters:**
- `embed_dim`: 128
- `num_layers`: 3
- `num_heads`: 4 (attention heads per layer)
- `k_neighbors`: None or int

**Computational Complexity:**
- Encoding: O(|E| d H) where H = number of heads
- Attention computation: O(|E| d)

**Advantages:**
- Adaptive to neighborhood importance
- Better than GCN for irregular graphs
- Interpretable attention weights
- Handles varying node degrees well

**When to Use:**
- Heterogeneous problem structures
- When edge importance varies significantly
- Problems with irregular spatial distributions
- When interpretability of relationships is needed

---

### 4. Hybrid Model (GNN + Attention)

**Implementation:** `model/hybrid_model.py`

**Architecture:**
- GNN encoder for structural features
- Transformer encoder for sequence refinement
- Fusion layer combining both representations
- Attention-based decoder

**Key Components:**
- `HybridEncoder`: Sequential GNN → Transformer → Fusion
- Feature fusion with concatenation and MLP
- Combines inductive biases from both architectures

**Information Flow:**
```
Input → GNN (structure) → Transformer (refinement) → Fusion → Decoder
         ↓                    ↓
    Spatial features    Sequential features
```

**Hyperparameters:**
- `embed_dim`: 128
- `num_gnn_layers`: 2
- `num_transformer_layers`: 2
- `gnn_type`: 'gat' or 'gcn'
- `num_heads`: 8

**Computational Complexity:**
- GNN encoding: O(|E| d H)
- Transformer encoding: O(n² d)
- Total: O(|E| d H + n² d)

**Advantages:**
- Combines strengths of GNN and Attention
- Captures both local and global patterns
- State-of-the-art potential
- Flexible architecture

**When to Use:**
- Maximum performance priority
- Sufficient computational resources
- Complex problems with multiple patterns
- When you want best of both worlds

---

## Model Selection Guide

| Criteria | Attention | GCN | GAT | Hybrid |
|----------|-----------|-----|-----|--------|
| **Accuracy** | High | Medium | High | Very High |
| **Speed** | Medium | Fast | Medium | Slow |
| **Memory** | Medium | Low | Medium | High |
| **Interpretability** | High | Low | High | Medium |
| **Scalability** | Medium | High | Medium | Low |
| **Training Stability** | High | High | Medium | Medium |

**Recommendations:**
- **Small to medium problems (n < 100)**: Attention or GAT
- **Large problems (n > 100)**: GCN with k-NN or Attention
- **Maximum accuracy**: Hybrid or GAT
- **Fast inference**: GCN
- **Research/experimentation**: Try all, compare results

---

## Training Procedure

### Reinforcement Learning Setup

All models are trained using the REINFORCE algorithm with baseline for variance reduction.

**Training Loop:**
```python
from model.model_factory import create_model
from model.rl_trainer import RLTrainer

# Create model
model = create_model('attention', {'embed_dim': 128})

# Initialize trainer
trainer = RLTrainer(
    model=model,
    learning_rate=1e-4,
    baseline_type='exponential',
    baseline_alpha=0.95
)

# Training loop
for epoch in range(num_epochs):
    for batch in dataloader:
        stats = trainer.train_step(
            batch=batch,
            sample_size=1,
            temperature=1.0
        )
        
        print(f"Loss: {stats['loss']:.4f}, "
              f"Reward: {stats['reward']:.2f}")
```

### Baseline Strategies

**1. Exponential Moving Average (Default)**
```python
baseline = α × baseline_old + (1-α) × reward_mean
```
- Simple and effective
- Low memory overhead
- Smooth baseline evolution

**2. Critic Network (Actor-Critic)**
```python
trainer = RLTrainer(model, use_critic=True)
```
- Learns state value function
- Lower variance than exponential
- Requires more computation

**3. Rollout Baseline**
- Use greedy rollout as baseline
- Most accurate but slowest
- Recommended for final evaluation

### Hyperparameter Tuning

**Learning Rate Schedule:**
```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=100, eta_min=1e-5
)
```

**Temperature Annealing:**
- Start: T = 2.0 (high exploration)
- End: T = 1.0 (exploitation)
- Schedule: Linear or exponential decay

**Batch Size:**
- Small problems (n < 50): 256-512
- Medium problems (n < 100): 128-256
- Large problems (n > 100): 64-128

---

## Evaluation Metrics

### Solution Quality

**1. Tour Length**
```python
tour_length = sum(distance[tour[i], tour[i+1]] 
                 for i in range(len(tour)-1))
```

**2. Optimality Gap**
```python
gap = (dl_solution - optimal_solution) / optimal_solution × 100%
```

**3. Feasibility Rate**
```python
feasibility = (valid_solutions / total_solutions) × 100%
```

### Computational Efficiency

**1. Inference Time**
- Measure per instance
- Average over test set
- Compare with classical methods

**2. Solution Quality vs. Time Trade-off**
```python
efficiency = solution_quality / inference_time
```

---

## Advanced Techniques

### 1. Data Augmentation

**Symmetry Augmentation:**
```python
def augment_8_way(problem):
    # Original + 7 transformations
    augmented = [
        problem,
        rotate_90(problem),
        rotate_180(problem),
        rotate_270(problem),
        flip_horizontal(problem),
        flip_vertical(problem),
        flip_diagonal_1(problem),
        flip_diagonal_2(problem)
    ]
    return augmented
```

**Benefits:**
- 8x more training data
- Better generalization
- Rotation and reflection invariance

### 2. Beam Search Decoding

```python
def beam_search(model, problem, beam_width=5):
    # Instead of sampling, keep top-k candidates
    beams = [initial_state]
    
    for step in range(num_nodes):
        candidates = []
        for beam in beams:
            probs = model.get_action_probs(beam)
            top_k = probs.topk(beam_width)
            candidates.extend(expand_beam(beam, top_k))
        
        beams = select_top_k(candidates, beam_width)
    
    return best_beam(beams)
```

**Advantages:**
- Better solutions at inference
- Controllable quality-speed trade-off
- No retraining required

### 3. Fine-tuning Strategies

**Instance-Specific Fine-tuning:**
```python
# Pre-train on diverse problems
pretrained_model = train_on_random_instances()

# Fine-tune on target distribution
finetuned_model = finetune(pretrained_model, target_instances)
```

**Transfer Learning:**
- Train on small problems
- Transfer to larger problems
- Fine-tune embedding dimensions

---

## Implementation Best Practices

### 1. Numerical Stability

```python
# Use log-softmax for probabilities
log_probs = F.log_softmax(scores, dim=-1)

# Avoid exp(large_number)
# Use log-space computations when possible
```

### 2. Gradient Clipping

```python
torch.nn.utils.clip_grad_norm_(
    model.parameters(), 
    max_norm=1.0
)
```

### 3. Batch Normalization vs. Layer Normalization

- **BatchNorm**: Good for convolutional layers
- **LayerNorm**: Better for sequence models (Transformer)
- **Recommendation**: Use LayerNorm for VRP models

### 4. Checkpoint Management

```python
# Save best model
if val_reward > best_reward:
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'reward': val_reward
    }, 'best_model.pt')
```

---

## Troubleshooting

### Common Issues

**1. Model Not Learning**
- Check learning rate (try 1e-4 to 1e-3)
- Verify gradient flow (use gradient clipping)
- Ensure proper masking of invalid actions
- Check baseline convergence

**2. Unstable Training**
- Reduce learning rate
- Increase batch size
- Use gradient clipping
- Try different baseline type

**3. Poor Generalization**
- Add data augmentation
- Reduce model complexity
- Increase training data diversity
- Use dropout (0.1-0.2)

**4. Slow Inference**
- Use GCN instead of Attention
- Reduce embedding dimension
- Use k-NN graphs instead of fully connected
- Enable mixed precision (FP16)

---

## Performance Benchmarks

### Expected Results (20-node problems)

| Model | Avg Gap to Optimal | Inference Time (ms) | Parameters |
|-------|-------------------|---------------------|------------|
| Attention | 2-5% | 15-25 | 760K |
| GCN | 5-8% | 5-10 | 150K |
| GAT | 2-4% | 20-30 | 890K |
| Hybrid | 1-3% | 30-40 | 880K |

Note: Results vary based on problem characteristics and hyperparameters.

---

## References

1. Kool, W., van Hoof, H., & Welling, M. (2019). Attention, Learn to Solve Routing Problems! ICLR.
2. Kipf, T. N., & Welling, M. (2017). Semi-Supervised Classification with Graph Convolutional Networks. ICLR.
3. Veličković, P., et al. (2018). Graph Attention Networks. ICLR.
4. Nazari, M., et al. (2018). Reinforcement Learning for Solving the Vehicle Routing Problem. NeurIPS.
5. Bello, I., et al. (2017). Neural Combinatorial Optimization with Reinforcement Learning. ICLR.

---

## Code Examples

### Example 1: Train Attention Model

```python
import torch
from model.model_factory import create_model
from model.rl_trainer import RLTrainer

# Create model
model = create_model('attention', {
    'embed_dim': 128,
    'num_heads': 8,
    'num_encoder_layers': 3
})

# Setup trainer
trainer = RLTrainer(
    model=model,
    learning_rate=1e-4,
    baseline_type='exponential'
)

# Training
for epoch in range(100):
    for batch in dataloader:
        stats = trainer.train_step(batch)
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Reward = {stats['reward']:.2f}")
```

### Example 2: Compare All Models

```python
from model.model_factory import create_model

models = {
    'Attention': create_model('attention'),
    'GCN': create_model('gnn_gcn'),
    'GAT': create_model('gnn_gat'),
    'Hybrid': create_model('hybrid')
}

for name, model in models.items():
    # Evaluate
    score = evaluate(model, test_set)
    print(f"{name}: {score:.2f}")
```

### Example 3: Inference with Beam Search

```python
def solve_with_beam_search(model, problem, beam_width=10):
    model.eval()
    with torch.no_grad():
        solutions = beam_search_decode(
            model, problem, beam_width
        )
    return best_solution(solutions)
```

---

## Conclusion

This implementation provides a comprehensive suite of deep learning models for VRP solving, each with distinct advantages. The choice of model should be guided by problem characteristics, computational constraints, and performance requirements. For research purposes, we recommend experimenting with all models and analyzing their behavior on your specific problem instances.

For production deployment, start with the Attention model for its balance of accuracy and interpretability, then optimize based on specific requirements.
