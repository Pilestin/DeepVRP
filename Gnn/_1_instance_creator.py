import numpy as np

def generate_vrptw_instance(num_customers=20, seed=0):
    rng = np.random.default_rng(seed)

    # Depo
    depot = {
        "id": 0,
        "x": 50.0,
        "y": 50.0,
        "demand": 0,
        "tw_early": 0,
        "tw_late": 1000
    }

    customers = []
    for i in range(1, num_customers + 1):
        x = rng.uniform(0, 100)
        y = rng.uniform(0, 100)

        demand = rng.integers(1, 10)
        tw_e = rng.integers(0, 600)
        tw_l = tw_e + rng.integers(50, 200)

        customers.append({
            "id": i,
            "x": x,
            "y": y,
            "demand": demand,
            "tw_early": tw_e,
            "tw_late": tw_l
        })

    nodes = [depot] + customers

    N = len(nodes)
    dist = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            dx = nodes[i]["x"] - nodes[j]["x"]
            dy = nodes[i]["y"] - nodes[j]["y"]
            dist[i, j] = np.sqrt(dx*dx + dy*dy)

    return nodes, dist
