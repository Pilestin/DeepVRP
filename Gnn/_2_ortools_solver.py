from ortools.constraint_solver import pywrapcp, routing_enums_pb2

def solve_vrptw_ortools(nodes, dist, vehicle_capacity=40, vehicle_speed=1.0):
    num_nodes = len(nodes)
    num_vehicles = 1
    depot_index = 0

    manager = pywrapcp.RoutingIndexManager(num_nodes, num_vehicles, depot_index)
    routing = pywrapcp.RoutingModel(manager)

    # --- Mesafe callback ---
    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return int(dist[from_node, to_node])  # tamsayı olmalı

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # --- Kapasite kısıtı ---
    demands = [n["demand"] for n in nodes]

    def demand_callback(from_index):
        from_node = manager.IndexToNode(from_index)
        return demands[from_node]

    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,                      # slack
        [vehicle_capacity],     # araç kapasitesi
        True,                   # depot'ta sıfırdan başla
        "Capacity"
    )

    # --- Zaman penceresi kısıtı ---
    # Mesafeyi süre olarak sayıyoruz (speed=1)
    def time_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        travel_time = dist[from_node, to_node] / vehicle_speed
        return int(travel_time)

    time_callback_index = routing.RegisterTransitCallback(time_callback)

    # Add time dimension.
    horizon = 1000
    routing.AddDimension(
        time_callback_index,
        1000,      # waiting / slack
        horizon,   # max süre
        False,     # start cumul to zero
        "Time"
    )
    time_dimension = routing.GetDimensionOrDie("Time")

    # TW’leri bağla
    for node_idx, node in enumerate(nodes):
        index = manager.NodeToIndex(node_idx)
        e = int(node["tw_early"])
        l = int(node["tw_late"])
        time_dimension.CumulVar(index).SetRange(e, l)

    # Depo için geniş aralık
    depot_e = 0
    depot_l = horizon
    depot_index_ortools = manager.NodeToIndex(depot_index)
    time_dimension.CumulVar(depot_index_ortools).SetRange(depot_e, depot_l)

    # Arama parametreleri
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    search_parameters.time_limit.seconds = 3

    solution = routing.SolveWithParameters(search_parameters)

    if solution is None:
        return None  # feasible çözüm bulunamadıysa

    # Çözümden rota çıkar
    route = []
    index = routing.Start(0)
    while not routing.IsEnd(index):
        node = manager.IndexToNode(index)
        route.append(node)
        index = solution.Value(routing.NextVar(index))
    route.append(depot_index)  # depo’ya dönüş
    return route
