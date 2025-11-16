

import os 
import sys
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# UTILITIES
from util.printer_utils import sleep_with_message, print_paths
from util.read_problem_instance import select_problemset
from util.data_preparation import create_problem_from_raw_data, prepare_for_deep_learning
# PROCESS STARTER
from start_process import start_process
# ALGORITHMS
from algorithms import solve_vrp_problem


current_dir = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.abspath(os.path.join(current_dir, '..'))                                                 #
problem_files = os.path.join(project_path, 'dataset', 'esogu', 'problems')                                      # problem dosyalarının bulunduğu klasör     
distance_matrix_file = os.path.join(project_path, 'dataset', 'esogu', 'matrix', 'Distance_matrix_v3.2.xlsx')    # mesafe matrisi
energy_matrix_file = os.path.join(project_path, 'dataset', 'esogu', 'matrix', 'Energy_matrix_v3.2.xlsx')        # enerji matrisi
location_matrix_file = os.path.join(project_path, 'dataset', 'esogu', 'matrix', 'Location_matrix_v3.2.xlsx')    # lokasyon matrisi


""" Bu kısımda öncelikle seçilen problem okunacak ve Modellenecek 
    - Arından matrisler excel içerisinden okunacak 
    - Deep Learning modellerine uygun embedding işlemleri yapılacak?
    - Modeller eğitilip test edilecek
    - Yol verileri Route4Vehicle'a uygun çıkarılacak. Aynı zamanda rota bilgisi yazılacak [[cs5, 1, 2, 3, cs5], [cs5, 4, 5, 6, cs5], ...]
"""

print("\n--- Deep VRP Main Execution (En Derini) ---\n")

def main():
    
    # HİKAYE Printleri
    print_paths(project_path, problem_files, distance_matrix_file, energy_matrix_file, location_matrix_file)
    
    # Problemi input() olarak seçx 
    selected_key, problemset_dict = select_problemset()
    
    # Seçilen problem ile işlemleri başlat
    file_name = f"newesoguv32-{problemset_dict[selected_key].lower()}-ds1.xml"
    problem_file = os.path.join(problem_files, file_name)
    
    # Verileri oku
    problem_data, distance_matrix, energy_matrix, location_matrix = start_process(
        problem_file, distance_matrix_file, energy_matrix_file, location_matrix_file
    )
    
    # VRPProblem nesnesi oluştur
    print("\n--- Creating VRP Problem Instance ---")
    vrp_problem = create_problem_from_raw_data(
        problem_data=problem_data,
        distance_matrix=distance_matrix,
        energy_matrix=energy_matrix,
        location_paths=location_matrix,
        num_vehicles=5,
        vehicle_capacity=350.0,
        battery_capacity=15600.0
    )
    
    print(f"✓ {vrp_problem}")
    print(f"  Depot: {vrp_problem.depot}")
    print(f"  Sample Customer: {vrp_problem.customers[0]}")
    print(f"  Sample Vehicle: {vrp_problem.vehicles[0]}")
    
    # Derin öğrenme için hazırla
    print("\n--- Preparing for Deep Learning ---")
    dl_data = prepare_for_deep_learning(
        problem=vrp_problem,
        normalize=True,
        create_graph=True,
        k_neighbors=10  # 10 en yakın komşu
    )
    
    print(f"✓ Node Features Shape: {dl_data['node_features'].shape}")
    print(f"✓ Distance Matrix Shape: {dl_data['distance_matrix'].shape}")
    print(f"✓ Energy Matrix Shape: {dl_data['energy_matrix'].shape}")
    if 'graph_data' in dl_data:
        print(f"✓ Graph Data: {dl_data['graph_data'].num_nodes} nodes, {dl_data['graph_data'].num_edges} edges")
    
    # Veri özetini göster
    print(f"\n--- Problem Summary ---")
    print(f"  - Depot: {problem_data['depot']['Name']} (Node {problem_data['depot']['No']})")
    print(f"  - Total Customers: {len(problem_data['customers'])}")
    print(f"  - Delivery Nodes: {problem_data['num_delivery']}")
    print(f"  - Pickup Nodes: {problem_data['num_pickup']}")
    print(f"\nMatrix Shapes:")
    print(f"  - Distance Matrix: {distance_matrix.shape}")
    print(f"  - Energy Matrix: {energy_matrix.shape}")
    print(f"  - Location Paths: {len(location_matrix)} routes")
    print(f"\nVehicle Configuration:")
    print(f"  - Number of Vehicles: {vrp_problem.num_vehicles}")
    print(f"  - Capacity: {vrp_problem.vehicles[0].capacity} kg")
    print(f"  - Battery Capacity: {vrp_problem.vehicles[0].battery_capacity} kWh")
    
    # Deep Learning Modelleri ile Çözüm
    print("\n" + "="*70)
    print("TESTING DEEP LEARNING MODELS")
    print("="*70)
    
    import sys
    sys.path.insert(0, os.path.join(project_path, 'model'))
    from vrp_models import create_vrp_model, get_model_params
    from algorithms.deep_vrp_solver import DeepVRPSolver
    
    models_to_test = [
        ('Attention', 'attention'),
        ('Pointer Network', 'pointer'),
        ('Seq2Seq', 'seq2seq'),
        ('GCN', 'gcn'),
        ('GAT', 'gat'),
        ('Hybrid', 'hybrid')
    ]
    
    results = {}
    
    for model_name, model_type in models_to_test:
        print(f"\n--- {model_name} Model ---")
        
        try:
            # Create model
            model = create_vrp_model(model_type, input_dim=7, hidden_dim=128)
            num_params = get_model_params(model)
            print(f"✓ Model created: {num_params:,} parameters")
            
            # Solve
            solver = DeepVRPSolver(model, model_type, device='cpu')
            routes, total_distance, total_time, total_energy = solver.solve(
                problem=vrp_problem,
                node_features=dl_data['node_features'],
                graph_data=dl_data.get('graph_data'),
                greedy=True
            )
            
            # Store results
            results[model_name] = {
                'routes': routes,
                'distance': total_distance,
                'time': total_time,
                'energy': total_energy,
                'num_vehicles': len(routes),
                'params': num_params
            }
            
            print(f"  Vehicles used: {len(routes)}")
            print(f"  Total Distance: {total_distance:.2f} m")
            print(f"  Total Time: {total_time:.2f} s ({total_time/60:.2f} min)")
            print(f"  Total Energy: {total_energy:.2f} kWh")
        except Exception as e:
            print(f"  ❌ Error: {str(e)}")
            results[model_name] = None
    
    # Comparison Table
    print("\n" + "="*70)
    print("MODEL COMPARISON")
    print("="*70)
    print(f"\n{'Model':<20} {'Params':<12} {'Vehicles':<10} {'Distance':<12} {'Time':<12} {'Energy':<12}")
    print("-"*90)
    
    for model_name in results:
        if results[model_name]:
            r = results[model_name]
            print(f"{model_name:<20} {r['params']:<12,} {r['num_vehicles']:<10} "
                  f"{r['distance']:<12.2f} {r['time']/60:<12.2f} {r['energy']:<12.2f}")
    
    print("="*90)
    
    # Save results to file
    results_file = os.path.join(current_dir, 'deep_vrp_results.txt')
    with open(results_file, 'w') as f:
        f.write(f"MODEL COMPARISON {datetime.now()}\n")
        f.write(f"\n{'Model':<20} {'Params':<12} {'Vehicles':<10} {'Distance':<12} {'Time':<12} {'Energy':<12}\n")
        f.write("-"*90 + "\n")
        
        for model_name in results:
            if results[model_name]:
                r = results[model_name]
                f.write(f"{model_name:<20} {r['params']:<12,} {r['num_vehicles']:<10} "
                        f"{r['distance']:<12.2f} {r['time']/60:<12.2f} {r['energy']:<12.2f}\n")
                f.write("="*90 + "\n")
                f.write("\nRoutes:\n")
                for i, route in enumerate(r['routes']):
                    f.write(f"  Vehicle {i+1}: {route}\n")
                f.write("\n")
    print(f"\n✓ Results saved to {results_file}")
    print("="*70)


    # Show best model routes
    if results:
        best_model = min(
            [(name, r) for name, r in results.items() if r],
            key=lambda x: x[1]['distance']
        )
        
        print(f"\n--- Best Model: {best_model[0]} (Distance: {best_model[1]['distance']:.2f} m) ---")
        print("\nRoutes:")
        routes_str = "["
        for i, route in enumerate(best_model[1]['routes']):
            if i > 0:
                routes_str += ", "
            routes_str += "[" + ", ".join(route) + "]"
        routes_str += "]"
        print(f"Routes = {routes_str}")
    
    return vrp_problem, dl_data, results

if __name__ == "__main__":
    vrp_problem, dl_data, results = main()
    
    print("\n" + "="*70)
    print("✓ ALL MODELS TESTED!")
    print("="*70)