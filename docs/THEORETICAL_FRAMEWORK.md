# Deep Learning Approaches for Vehicle Routing Problems: Theoretical Framework

## Table of Contents
1. [Introduction](#introduction)
2. [Problem Representation](#problem-representation)
3. [Embedding Mechanisms](#embedding-mechanisms)
4. [Model Architectures](#model-architectures)
5. [Training Methodology](#training-methodology)
6. [References](#references)

---

## 1. Introduction

This document presents a comprehensive theoretical framework for applying deep learning methodologies to the Capacitated Electric Vehicle Routing Problem with Time Windows (CEVRPTW). The approaches discussed leverage recent advances in neural network architectures, particularly attention mechanisms and graph neural networks, to learn effective routing policies.

### 1.1 Problem Formulation

The CEVRPTW can be formulated as follows:

**Given:**
- A depot location d with coordinates (x_d, y_d)
- A set of n customers C = {c_1, c_2, ..., c_n}
- For each customer c_i:
  - Location coordinates (x_i, y_i)
  - Demand q_i
  - Time window [e_i, l_i] where e_i is ready time and l_i is due date
  - Service time s_i
- Distance matrix D ∈ R^(n+1)×(n+1)
- Energy consumption matrix E ∈ R^(n+1)×(n+1)
- Fleet of m electric vehicles with:
  - Capacity Q
  - Battery capacity B

**Objective:**
Minimize total travel distance while satisfying:
- Each customer visited exactly once
- Vehicle capacity constraints
- Time window constraints
- Battery constraints

---

## 2. Problem Representation

### 2.1 Node Feature Representation

Each node (depot or customer) is represented as a feature vector f_i ∈ R^7:

```
f_i = [x_i, y_i, q_i, e_i, l_i, s_i, δ_i]
```

where:
- x_i, y_i: Normalized spatial coordinates
- q_i: Demand (0 for depot)
- e_i: Ready time (0 for depot)
- l_i: Due date (large value for depot)
- s_i: Service time (0 for depot)
- δ_i: Binary depot indicator (1 for depot, 0 otherwise)

### 2.2 Graph Representation

The problem instance is represented as a complete directed graph G = (V, E) where:
- V = {v_0, v_1, ..., v_n} is the set of nodes (depot + customers)
- E = V × V is the set of directed edges
- Each edge (i,j) has attributes:
  - Distance: d_ij from distance matrix D
  - Energy: e_ij from energy matrix E

### 2.3 Normalization

Feature normalization is critical for neural network training. Min-max normalization is applied:

```
f'_i = (f_i - f_min) / (f_max - f_min)
```

This ensures all features are in the range [0, 1], preventing certain features from dominating the learning process.

---

## 3. Embedding Mechanisms

### 3.1 Node Embedding

The node embedding layer transforms raw node features into a dense, learned representation:

**Architecture:**
```
h_i = LayerNorm(ReLU(W_2 × ReLU(W_1 × f_i + b_1) + b_2))
```

where:
- W_1 ∈ R^64×7, b_1 ∈ R^64
- W_2 ∈ R^128×64, b_2 ∈ R^128
- h_i ∈ R^128 is the node embedding

**Rationale:**
- Two-layer MLP captures non-linear relationships between features
- ReLU activation introduces non-linearity
- LayerNorm stabilizes training and improves generalization
- 128-dimensional embedding provides sufficient representational capacity

### 3.2 Edge Embedding

Edge features (distance and energy) are encoded similarly:

```
e_ij = ReLU(W_e × [d_ij, e_ij]^T + b_e)
```

where:
- W_e ∈ R^128×2, b_e ∈ R^128
- Edge embedding dimension matches node embedding for consistency

### 3.3 Graph-Level Embedding

A global graph representation is obtained through aggregation:

```
h_graph = f_agg({h_1, h_2, ..., h_n})
```

Common aggregation functions:
- Mean pooling: h_graph = (1/n) Σ h_i
- Max pooling: h_graph = max{h_1, ..., h_n}
- Attention-based: h_graph = Σ α_i h_i

The graph embedding is then processed through an MLP:

```
g = LayerNorm(ReLU(W_g2 × ReLU(W_g1 × h_graph + b_g1) + b_g2))
```

where g ∈ R^128 represents the entire problem instance.

**Purpose:**
- Captures global problem characteristics
- Used for context in decision-making
- Enables generalization across problem sizes

---

## 4. Model Architectures

### 4.1 Attention Mechanism for VRP

#### 4.1.1 Theoretical Foundation

The attention mechanism, introduced by Bahdanau et al. (2014) and popularized by Vaswani et al. (2017) in the Transformer architecture, computes a weighted combination of values based on learned compatibility between queries and keys.

**Scaled Dot-Product Attention:**

```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

where:
- Q ∈ R^n×d_k: Query matrix
- K ∈ R^n×d_k: Key matrix
- V ∈ R^n×d_v: Value matrix
- d_k: Key/query dimension
- √d_k: Scaling factor to prevent gradient vanishing

#### 4.1.2 Multi-Head Attention

Multi-head attention allows the model to attend to different representation subspaces:

```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h)W^O

where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
```

Parameters:
- h: Number of attention heads (typically 8)
- W_i^Q, W_i^K ∈ R^d_model×d_k
- W_i^V ∈ R^d_model×d_v
- W^O ∈ R^hd_v×d_model

**Advantages:**
- Captures different types of relationships simultaneously
- Improves model capacity without excessive parameters
- Provides interpretability through attention weights

#### 4.1.3 Attention Model for VRP

The Attention Model (Kool et al., 2019) uses an encoder-decoder architecture:

**Encoder:**
```
for l = 1 to L:
    h^(l) = MultiHeadAttention(h^(l-1), h^(l-1), h^(l-1))
    h^(l) = LayerNorm(h^(l-1) + h^(l))
    h^(l) = LayerNorm(h^(l) + FFN(h^(l)))
```

**Decoder (Autoregressive):**
At each step t:
1. Context embedding: c_t = [h_graph, h_current, h_first, remaining_capacity]
2. Query computation: q_t = W_q c_t
3. Compatibility scores: u_tj = (q_t^T k_j) / √d_k for all unvisited nodes j
4. Apply mask to visited nodes: u_tj = -∞ for visited j
5. Action probabilities: π_t(j) = softmax(u_t)_j

**Masking Strategy:**
- Visited nodes: probability = 0
- Capacity-infeasible nodes: probability = 0
- Time-window-infeasible nodes: probability = 0

### 4.2 Graph Neural Networks (GNN)

#### 4.2.1 Message Passing Framework

GNNs operate on graph-structured data through iterative message passing:

**General Framework:**
```
for k = 1 to K:
    for each node v:
        m_v^(k) = AGGREGATE({h_u^(k-1) : u ∈ N(v)})
        h_v^(k) = UPDATE(h_v^(k-1), m_v^(k))
```

where:
- N(v): Neighbors of node v
- m_v^(k): Aggregated message at layer k
- h_v^(k): Node representation at layer k

#### 4.2.2 Graph Convolutional Networks (GCN)

GCN (Kipf & Welling, 2017) performs spectral convolution on graphs:

```
h_v^(k) = σ(Σ_{u∈N(v)∪{v}} (1/√(d_u d_v)) W^(k) h_u^(k-1))
```

where:
- d_v: Degree of node v
- W^(k): Learnable weight matrix
- σ: Non-linear activation (ReLU)

**Normalization:** The term 1/√(d_u d_v) normalizes by node degrees, preventing feature explosion in high-degree nodes.

#### 4.2.3 Graph Attention Networks (GAT)

GAT (Veličković et al., 2018) uses attention to learn edge weights:

```
α_vu = softmax(LeakyReLU(a^T [W h_v || W h_u]))

h_v^(k) = σ(Σ_{u∈N(v)} α_vu W^(k) h_u^(k-1))
```

where:
- a: Attention weight vector
- ||: Concatenation
- α_vu: Learned attention coefficient from u to v

**Multi-Head GAT:**
```
h_v^(k) = ||_{i=1}^H σ(Σ_{u∈N(v)} α_vu^(i) W^(k,i) h_u^(k-1))
```

**Advantages over GCN:**
- Learns edge importance adaptively
- Better handles varying neighborhood sizes
- Provides interpretable attention weights

#### 4.2.4 GNN for VRP

**Node Update with Edge Features:**
```
m_v = Σ_{u∈N(v)} α_vu (W_n h_u + W_e e_uv)
h_v' = LayerNorm(h_v + ReLU(W_m m_v))
```

**Action Selection:**
After K GNN layers, select next node based on:
```
score_v = MLP([h_v^(K), h_current, edge_features])
π(v) = softmax(score / temperature)
```

### 4.3 Reinforcement Learning Integration

#### 4.3.1 Markov Decision Process Formulation

The VRP is formulated as an MDP:
- State s_t: Partial tour, current location, remaining capacity, current time
- Action a_t: Select next customer to visit
- Reward r_t: Negative distance traveled (or 0 for intermediate steps)
- Terminal reward: Negative total tour length

#### 4.3.2 REINFORCE Algorithm

Policy gradient method for optimizing the routing policy:

**Objective:**
```
J(θ) = E_{π_θ}[R(τ)]
```

where:
- θ: Policy network parameters
- τ = (s_0, a_0, s_1, ..., s_T): Trajectory
- R(τ): Total reward (negative tour length)

**Gradient Estimation:**
```
∇_θ J(θ) = E_{τ∼π_θ}[R(τ) Σ_t ∇_θ log π_θ(a_t|s_t)]
```

**REINFORCE with Baseline:**
```
∇_θ J(θ) ≈ (1/B) Σ_{i=1}^B [(R(τ_i) - b) Σ_t ∇_θ log π_θ(a_t^i|s_t^i)]
```

where b is a baseline (e.g., exponential moving average of rewards) to reduce variance.

#### 4.3.3 Actor-Critic

Combines policy gradient (actor) with value function estimation (critic):

**Actor (Policy Network):**
```
π_θ(a|s): Probability distribution over actions
```

**Critic (Value Network):**
```
V_ϕ(s): Estimated value of state s
```

**Update Rules:**
```
δ_t = r_t + γV_ϕ(s_{t+1}) - V_ϕ(s_t)  (TD error)
θ ← θ + α_θ δ_t ∇_θ log π_θ(a_t|s_t)
ϕ ← ϕ + α_ϕ δ_t ∇_ϕ V_ϕ(s_t)
```

**Advantages:**
- Lower variance than pure REINFORCE
- Faster convergence
- Better sample efficiency

### 4.4 Hybrid Approaches

#### 4.4.1 GNN + Attention

Combines GNN's graph structure exploitation with attention's sequence modeling:

**Architecture:**
1. GNN Encoder: Extract structural graph features
2. Attention Encoder: Refine node representations
3. Attention Decoder: Autoregressive route construction

**Information Flow:**
```
h_initial = NodeEmbedding(features)
h_structural = GNN(h_initial, edges)
h_refined = Transformer(h_structural)
π_t(a) = AttentionDecoder(h_refined, context_t)
```

#### 4.4.2 GNN + RL

Uses GNN as the policy network in RL framework:

**Policy Network:**
```
π_θ(a|s) = softmax(GNN_θ(graph_state))
```

**Training:**
- Sample trajectories using current policy
- Compute rewards (negative tour lengths)
- Update GNN parameters via policy gradient

**Advantages:**
- GNN captures spatial relationships
- RL optimizes for task-specific objective
- No need for labeled optimal solutions

---

## 5. Training Methodology

### 5.1 Data Generation

**Training Set Construction:**
- Generate random problem instances
- Vary problem size: n ∈ {10, 20, 50, 100}
- Vary customer distributions: clustered, random, mixed
- Create time windows with varying tightness

**Augmentation:**
- 8-way symmetry augmentation (rotations and reflections)
- Instance size generalization: train on smaller, test on larger

### 5.2 Optimization

**Loss Functions:**

1. **Supervised Learning:**
   ```
   L = -Σ_t log π_θ(a_t*|s_t)
   ```
   where a_t* is the optimal action (requires labeled data)

2. **Reinforcement Learning:**
   ```
   L = -E[(R(τ) - b) Σ_t log π_θ(a_t|s_t)]
   ```

3. **Hybrid:**
   ```
   L = L_supervised + λ L_RL
   ```

**Optimizers:**
- Adam optimizer with learning rate 1e-4
- Learning rate scheduling: decay or warmup + cosine annealing
- Gradient clipping: ||∇|| ≤ 1.0

**Batch Training:**
- Batch size: 128-512 instances
- Parallel trajectory sampling in RL
- Distributed training for large models

### 5.3 Evaluation Metrics

**Solution Quality:**
- Tour length: Total distance traveled
- Optimality gap: (solution - optimal) / optimal × 100%
- Feasibility rate: Percentage of valid solutions

**Computational Efficiency:**
- Inference time per instance
- Number of model parameters
- GPU memory consumption

**Generalization:**
- Zero-shot transfer to larger instances
- Performance on different distributions
- Robustness to constraint variations

### 5.4 Baseline Comparisons

**Classical Methods:**
- Clarke-Wright Savings Algorithm
- Sweep Algorithm
- 2-opt, 3-opt local search

**Metaheuristics:**
- Genetic Algorithm (GA)
- Simulated Annealing (SA)
- Tabu Search
- Ant Colony Optimization (ACO)

**Learning-based Methods:**
- Pointer Networks
- Attention Model
- Graph Attention Networks
- Hybrid approaches

---

## 6. Implementation Considerations

### 6.1 Constraint Handling

**Hard Constraints:**
- Masking infeasible actions in action distribution
- Penalty terms in reward function
- Repair mechanisms for constraint violations

**Soft Constraints:**
- Weighted penalty in objective function
- Lagrangian relaxation
- Constraint as additional features

### 6.2 Scalability

**Techniques:**
- Mini-batch processing
- Sparse attention for large graphs
- Hierarchical approaches for large-scale problems
- Parallelization across problem instances

### 6.3 Interpretability

**Analysis Tools:**
- Attention weight visualization
- GNN layer activation analysis
- Decision trajectory analysis
- Ablation studies

---

## 7. References

1. Vaswani, A., et al. (2017). "Attention is all you need." NeurIPS.
2. Kool, W., van Hoof, H., & Welling, M. (2019). "Attention, learn to solve routing problems!" ICLR.
3. Kipf, T. N., & Welling, M. (2017). "Semi-supervised classification with graph convolutional networks." ICLR.
4. Veličković, P., et al. (2018). "Graph attention networks." ICLR.
5. Nazari, M., et al. (2018). "Reinforcement learning for solving the vehicle routing problem." NeurIPS.
6. Vinyals, O., Fortunato, M., & Jaitly, N. (2015). "Pointer networks." NeurIPS.
7. Bello, I., et al. (2017). "Neural combinatorial optimization with reinforcement learning." ICLR.
8. Joshi, C. K., et al. (2019). "Learning TSP requires rethinking generalization." CP.

---

## Appendix A: Mathematical Notation

| Symbol | Description |
|--------|-------------|
| n | Number of customers |
| m | Number of vehicles |
| V | Set of nodes (vertices) |
| E | Set of edges |
| f_i | Feature vector for node i |
| h_i | Embedding for node i |
| d_ij | Distance from node i to j |
| e_ij | Energy consumption from i to j |
| Q | Vehicle capacity |
| B | Battery capacity |
| θ | Neural network parameters |
| π_θ | Policy parameterized by θ |
| τ | Trajectory (sequence of states and actions) |
| R(τ) | Total reward for trajectory τ |

---

## Appendix B: Hyperparameter Guidelines

| Parameter | Recommended Range | Notes |
|-----------|------------------|-------|
| Embedding dimension | 128-256 | Higher for larger problems |
| Number of attention heads | 8-16 | Multi-head attention |
| Number of encoder layers | 3-6 | Balance depth and computation |
| Learning rate | 1e-4 to 1e-3 | With scheduler |
| Batch size | 128-512 | Depends on GPU memory |
| Temperature (sampling) | 1.0-2.0 | Higher for exploration |
| Baseline decay | 0.9-0.99 | Exponential moving average |

