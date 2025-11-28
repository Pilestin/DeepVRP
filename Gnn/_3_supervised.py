import torch
from torch_geometric.data import Data
import math

def build_graph_from_nodes(nodes):
    # Statik kısım (TW, demand, coord, is_depot)
    static_feats = []
    for i, n in enumerate(nodes):
        is_depot = 1.0 if i == 0 else 0.0
        static_feats.append([
            n["x"], n["y"],
            n["demand"],
            n["tw_early"], n["tw_late"],
            is_depot
        ])
    x_static = torch.tensor(static_feats, dtype=torch.float)

    num_nodes = len(nodes)
    edge_index_list = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i == j:
                continue
            edge_index_list.append([i, j])
    edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
    return x_static, edge_index

def build_step_features(x_static, current_node, visited_mask):
    """
    x_static: [N, 6]
    current_node: int
    visited_mask: [N] bool (True=visited)
    """
    N = x_static.size(0)
    is_visited = torch.tensor(visited_mask, dtype=torch.float).unsqueeze(-1)  # [N,1]
    is_current = torch.zeros((N, 1), dtype=torch.float)
    is_current[current_node, 0] = 1.0

    # [N, 6+1+1] = [N, 8]
    x_step = torch.cat([x_static, is_visited, is_current], dim=1)
    return x_step

def build_training_samples_from_route(nodes, route):
    """
    route: örn [0, 5, 2, 8, 1, 0]
    """
    x_static, edge_index = build_graph_from_nodes(nodes)
    N = len(nodes)

    samples = []

    visited = [False] * N
    # Depo başlangıçta visited kabul edilebilir
    visited[route[0]] = True

    for t in range(len(route) - 1):
        current_node = route[t]
        next_node = route[t+1]

        x_step = build_step_features(x_static, current_node, visited)
        label = next_node  # CrossEntropy için target class

        samples.append((x_step, edge_index, label))

        visited[next_node] = True

    return samples
