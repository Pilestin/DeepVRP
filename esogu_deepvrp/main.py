

import os 

# UTILITIES
from util.printer_utils import sleep_with_message, print_paths
from util.read_problem_instance import select_problemset
# PROCESS STARTER
from start_process import start_process


current_dir = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.abspath(os.path.join(current_dir, '..', '..'))
problem_files = project_path + '/dataset/esogu/problems'                                    # problem dosyalarının bulunduğu klasör
distance_matrix_file = project_path + '/dataset/esogu/matrix/Distance_matrix_v3.2.xlsx'     # mesafe matrisi 
energy_matrix_file = project_path + '/dataset/esogu/matrix/Energy_matrix_v3.2.xlsx'         # enerji matrisi
location_matrix_file = project_path + '/dataset/esogu/matrix/Location_matrix_v3.2.xlsx'     # lokasyon matrisi


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
    start_process(problem_file, distance_matrix_file, energy_matrix_file, location_matrix_file)

if __name__ == "__main__":
    main()