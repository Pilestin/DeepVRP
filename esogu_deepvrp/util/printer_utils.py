

import time

def sleep(seconds):
    time.sleep(seconds)
    
    
def sleep_with_message(seconds, message: str):
    """ Bir mesaj ile birlikte belirli bir süre uyutur """
    print(message)
    time.sleep(seconds)
    
    
def sleep_with_messagess(seconds, messages : list[str]):
    """ Birden fazla mesaj ile birlikte belirli bir süre uyuturss """
    for msg in messages:
        print(msg)
        time.sleep(seconds)
        

def print_paths(project_path, problem_files, distance_matrix_file, energy_matrix_file, location_matrix_file):
    print("------------------ Paths Information ------------------")
    print("Project Path:", project_path)
    print("Problem Files Path:", problem_files)
    print("Distance Matrix File:", distance_matrix_file)
    print("Energy Matrix File:", energy_matrix_file)
    print("Location Matrix File:", location_matrix_file)
    print("-------------------------------------------------------")