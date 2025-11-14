

import os 
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# UTILITIES
from util.printer_utils import sleep_with_message, print_paths
from util.read_problem_instance import select_problemset
from util.data_preparation import create_problem_from_raw_data, prepare_for_deep_learning
# PROCESS STARTER
from start_process import start_process


current_dir = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.abspath(os.path.join(current_dir, '..'))
problem_files = os.path.join(project_path, 'dataset', 'esogu', 'problems')
distance_matrix_file = os.path.join(project_path, 'dataset', 'esogu', 'matrix', 'Distance_matrix_v3.2.xlsx')
energy_matrix_file = os.path.join(project_path, 'dataset', 'esogu', 'matrix', 'Energy_matrix_v3.2.xlsx')
location_matrix_file = os.path.join(project_path, 'dataset', 'esogu', 'matrix', 'Location_matrix_v3.2.xlsx')


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
    
    # Problemi input() olarak seç 
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
        vehicle_capacity=200.0,
        battery_capacity=100.0
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
    
    return vrp_problem, dl_data

if __name__ == "__main__":
    vrp_problem, dl_data = main()
    
    # Embeddings test
    print("\n" + "="*60)
    print("Would you like to test embeddings? (y/n)")
    choice = input("Choice: ").strip().lower()
    
    if choice == 'y':
        from test_embeddings import test_embeddings
        embedding_results = test_embeddings(dl_data)
        print("Embedding models are ready for training!")