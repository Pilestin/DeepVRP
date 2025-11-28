from _1_instance_creator import generate_vrptw_instance
from _2_ortools_solver import solve_vrptw_ortools
from _3_supervised import build_training_samples_from_route

import numpy as np 

def compute_route_cost(route, dist):
    """route: [n0, n1, ..., nk], dist: np.array"""
    cost = 0.0
    for i in range(len(route) - 1):
        cost += dist[route[i], route[i+1]]
    return cost


def generate_dataset(num_instances=10, num_customers=10, seed=0):
    rng = np.random.default_rng(seed)
    all_samples = []

    for k in range(num_instances):
        # 1) Instance üret
        nodes, dist = generate_vrptw_instance(
            num_customers=num_customers,
            seed=int(rng.integers(0, 1e9))
        )

        # 2) OR-Tools ile çöz
        route = solve_vrptw_ortools(nodes, dist, vehicle_capacity=40)
        if route is None:
            print(f"[Instance {k}] OR-Tools çözüm bulamadı, atlanıyor.")
            continue

        # 3) BEFORE rota: basit sıralı tur (0,1,2,...,N-1,0)
        num_nodes = len(nodes)
        route_before = list(range(num_nodes)) + [0]

        cost_before = compute_route_cost(route_before, dist)
        cost_after = compute_route_cost(route, dist)

        print(f"\n=== Instance {k} ===")
        print("Nodes (id, x, y, demand, tw_e, tw_l):")
        for n in nodes:
            print(
                f"  {n['id']:2d} | "
                f"x={n['x']:.1f}, y={n['y']:.1f}, "
                f"d={n['demand']}, "
                f"[{n['tw_early']}, {n['tw_late']}]"
            )

        print("\nBefore route (naive):", route_before)
        print(f"Before cost: {cost_before:.2f}")

        print("\nAfter route (OR-Tools):", route)
        print(f"After cost:  {cost_after:.2f}")
        print("=" * 40)

        # 4) GNN için supervised örnekleri oluştur
        samples = build_training_samples_from_route(nodes, route)
        all_samples.extend(samples)

    return all_samples


def generate_training_samples(num_instances=200, num_customers=20, seed=0):
    rng = np.random.default_rng(seed)
    all_samples = []

    for k in range(num_instances):
        nodes, dist = generate_vrptw_instance(
            num_customers=num_customers,
            seed=int(rng.integers(0, 1e9))
        )

        route = solve_vrptw_ortools(nodes, dist, vehicle_capacity=40)
        if route is None:
            continue

        samples = build_training_samples_from_route(nodes, route)
        all_samples.extend(samples)

    return all_samples
