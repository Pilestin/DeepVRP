

import os 

# UTILITIES
from util.printer_utils import sleep_with_message, print_paths
from util.read_problem_instance import select_problemset
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
    
    # Veri özetini göster
    print(f"Problem Summary:")
    print(f"  - Depot: {problem_data['depot']['Name']} (Node {problem_data['depot']['No']})")
    print(f"  - Total Customers: {len(problem_data['customers'])}")
    print(f"  - Delivery Nodes: {problem_data['num_delivery']}")
    print(f"  - Pickup Nodes: {problem_data['num_pickup']}")
    print(f"\nMatrix Shapes:")
    print(f"  - Distance Matrix: {distance_matrix.shape}")
    print(f"  - Energy Matrix: {energy_matrix.shape}")
    print(f"  - Location Paths: {len(location_matrix)} routes")
    
    return problem_data, distance_matrix, energy_matrix, location_matrix

if __name__ == "__main__":
    main()