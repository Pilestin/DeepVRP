import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from _1_instance_creator import generate_vrptw_instance
from _2_ortools_solver import solve_vrptw_ortools
from _3_supervised import build_training_samples_from_route
from _4_GNN import VRPGNN

from utils import compute_route_cost, generate_dataset



# Dataset
all_samples = generate_dataset(num_instances=5, num_customers=8)
print("Num training samples:", len(all_samples))

# Model
in_features = all_samples[0][0].shape[1]
model = VRPGNN(in_features=in_features, hidden_dim=64)

optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# Train
for epoch in range(10):
    total_loss = 0.0
    for x_step, edge_index, label in all_samples:
        model.train()
        optimizer.zero_grad()

        scores = model(x_step, edge_index)  # [N]
        scores_batch = scores.unsqueeze(0)          # [1, N]
        target = torch.tensor([label], dtype=torch.long)  # [1]

        loss = criterion(scores_batch, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, loss = {total_loss:.4f}")
