


""" Buradaki fonksiyon(lar) verilen problem instance xml dosyalarını okuyup dictionary olarak tutar ve döndürür."""



def select_problemset():
    problemset_dict = {
        '1': 'C05',
        '2': 'C10',
        '3': 'C20',
        '4': 'C40',
        '5': 'C60',
        '6': 'R05',
        '7': 'R10',
        '8': 'R20',
        '9': 'R40',
        '10': 'R60',
        '11': 'RC05',
        '12': 'RC10',
        '13': 'RC20',
        '14': 'RC40',
        '15': 'RC60'
    }
    print("Select problemset to process from the 'dataset/esogu/problems' directory.")
    
    # for key, value in problemset_dict.items():
    #     print(f"{value} : {key}")
    # selected_key = input("Your choice: ")
    
    selected_key = "3"
    print(f"Selected problemset {problemset_dict[selected_key]} will be processed")
    
    return selected_key, problemset_dict


def read_problem_instance_file(file_path: str):
    pass 