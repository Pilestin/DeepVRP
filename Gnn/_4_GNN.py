import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class VRPGNN(nn.Module):
    def __init__(self, in_features, hidden_dim=64):
        super().__init__()
        self.conv1 = GCNConv(in_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, 1)  # [N,1] -> skor

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = self.conv2(h, edge_index)
        h = F.relu(h)
        scores = self.out(h).squeeze(-1)  # [N]
        return scores
