# Deep Learning Models for Vehicle Routing Problems

This directory contains implementations of state-of-the-art deep learning models for solving the Capacitated Electric Vehicle Routing Problem with Time Windows (CEVRPTW).

## Implemented Models

### 1. Attention Model (`attention_model.py`)
Transformer-based encoder-decoder with multi-head attention mechanism. Based on Kool et al. (2019).

**Parameters:** ~760K | **Inference:** ~20ms | **Accuracy:** High

### 2. Graph Convolutional Network (`gnn_model.py`)
Spectral graph convolution with degree normalization. Based on Kipf & Welling (2017).

**Parameters:** ~150K | **Inference:** ~8ms | **Accuracy:** Medium-High

### 3. Graph Attention Network (`gnn_model.py`)
Attention-based graph neural network with learned edge weights. Based on Veličković et al. (2018).

**Parameters:** ~890K | **Inference:** ~25ms | **Accuracy:** High

### 4. Hybrid Model (`hybrid_model.py`)
Combines GNN structural learning with Transformer sequence modeling.

**Parameters:** ~880K | **Inference:** ~35ms | **Accuracy:** Very High

## Quick Start

```python
from model.model_factory import create_model
from model.rl_trainer import RLTrainer

# Create a model
model = create_model('attention', {'embed_dim': 128, 'num_heads': 8})

# Setup training
trainer = RLTrainer(model, learning_rate=1e-4)

# Train
stats = trainer.train_step(batch)
```

## Model Selection

| Use Case | Recommended Model |
|----------|------------------|
| Best Accuracy | Hybrid or GAT |
| Fastest Inference | GCN |
| General Purpose | Attention |
| Large Scale | GCN with k-NN |

## Components

- **Embeddings** (`embeddings.py`): Node and graph feature encoding
- **Transforms** (`transforms.py`): Data preprocessing and normalization
- **RL Trainer** (`rl_trainer.py`): REINFORCE and Actor-Critic training
- **Model Factory** (`model_factory.py`): Easy model creation and comparison

## Documentation

- `../docs/THEORETICAL_FRAMEWORK.md` - Mathematical foundations and theory
- `../docs/IMPLEMENTATION_GUIDE.md` - Detailed implementation guide and best practices

## Testing

```bash
# Demo all models
python ../esogu_deepvrp/demo_models.py --mode demo

# Compare performance
python ../esogu_deepvrp/demo_models.py --mode compare
```

## Requirements

- PyTorch >= 2.0.0
- torch-geometric >= 2.3.0
- numpy >= 1.24.0

## Citation

If you use these implementations in your research, please cite the original papers:

```bibtex
@inproceedings{kool2019attention,
  title={Attention, learn to solve routing problems!},
  author={Kool, Wouter and Van Hoof, Herke and Welling, Max},
  booktitle={ICLR},
  year={2019}
}

@inproceedings{kipf2017semi,
  title={Semi-supervised classification with graph convolutional networks},
  author={Kipf, Thomas N and Welling, Max},
  booktitle={ICLR},
  year={2017}
}

@inproceedings{velivckovic2018graph,
  title={Graph attention networks},
  author={Veli{\v{c}}kovi{\'c}, Petar and Cucurull, Guillem and Casanova, Arantxa and Romero, Adriana and Lio, Pietro and Bengio, Yoshua},
  booktitle={ICLR},
  year={2018}
}
```
