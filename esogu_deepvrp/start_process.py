


# UTILITIES
from util.read_problem_instance import select_problemset, read_problem_instance_file
from util.read_matrix_files import read_distance_matrix, read_energy_matrix, read_location_matrix
from util.printer_utils import sleep_with_message


def start_process(problem_file, distance_matrix_file, energy_matrix_file, location_matrix_file):

    try:
        # ------------------------problem instance oku----------------------------------------------
        sleep_with_message(message=f"Reading problem file: {problem_file}", seconds=0.5)
        read_problem_instance_file(problem_file)
        # ------------------------matrixler oku----------------------------------------------
        sleep_with_message(message=f"Reading distance matrix from: {distance_matrix_file}", seconds=0.5)
        read_distance_matrix(distance_matrix_file)
        sleep_with_message(message=f"Reading energy matrix from: {energy_matrix_file}", seconds=0.5)
        read_energy_matrix(energy_matrix_file)
        sleep_with_message(message=f"Reading location matrix from: {location_matrix_file}", seconds=0.5)
        read_location_matrix(location_matrix_file)

    except Exception as e:
        print("An error occurred during the start process:", str(e))
        sleep_with_message(message="Error occurred. Please check the logs.", seconds=2)
        