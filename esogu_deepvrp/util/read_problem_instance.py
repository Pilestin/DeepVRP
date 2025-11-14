


""" Buradaki fonksiyon(lar) verilen problem instance xml dosyalarını okuyup dictionary olarak tutar ve döndürür."""

import xml.etree.ElementTree as ET



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
    
    for key, value in problemset_dict.items():
        print(f"  {key}: {value}")
    
    selected_key = input("\nYour choice: ").strip()
    
    if selected_key not in problemset_dict:
        print(f"Invalid choice. Defaulting to C20.")
        selected_key = "3"
    
    print(f"Selected problemset {problemset_dict[selected_key]} will be processed\n")
    
    return selected_key, problemset_dict


def read_problem_instance_file(file_path: str):
    """
    XML dosyasından problem verilerini okur.
    
    Returns:
        dict: {
            'depot': {'No': str, 'Name': str, 'Latitude': float, 'Longitude': float, 'X': float, 'Y': float},
            'customers': [
                {
                    'No': str,
                    'Name': str,
                    'Type': str,
                    'Latitude': float,
                    'Longitude': float,
                    'X': float,
                    'Y': float,
                    'Weight': int,
                    'Quantity': int,
                    'ReadyTime': int,
                    'ServiceTime': int,
                    'DueDate': int
                }, ...
            ],
            'num_delivery': int,
            'num_pickup': int
        }
    """
    tree = ET.parse(file_path)
    root = tree.getroot()
    
    # CEVRPTW elementini bul
    cevrptw = root.find('.//CEVRPTW')
    num_delivery = int(cevrptw.get('NumberOfDelivery', 0))
    num_pickup = int(cevrptw.get('NumberOfPickup', 0))
    
    problem_data = {
        'depot': None,
        'customers': [],
        'num_delivery': num_delivery,
        'num_pickup': num_pickup
    }
    
    # Tüm node'ları oku
    nodes = cevrptw.findall('.//Node')
    
    for node in nodes:
        node_type = node.get('Type')
        
        # Depo bilgisi (Entrance kullanıyoruz)
        if node_type == 'Entrance':
            location = node.find('Location')
            problem_data['depot'] = {
                'No': node.get('No'),
                'Name': node.get('Name'),
                'Latitude': float(location.find('Latitude').text),
                'Longitude': float(location.find('Longitude').text),
                'X': float(location.find('X_Coordinates').text),
                'Y': float(location.find('Y_Coordinates').text)
            }
        
        # Müşteri noktaları (Delivery veya Pickup)
        elif node_type in ['Delivery', 'Pickup']:
            location = node.find('Location')
            request = node.find('.//Request')
            load_info = request.find('LoadInformation')
            
            customer = {
                'No': node.get('No'),
                'Name': node.get('Name'),
                'Type': node_type,
                'Latitude': float(location.find('Latitude').text),
                'Longitude': float(location.find('Longitude').text),
                'X': float(location.find('X_Coordinates').text),
                'Y': float(location.find('Y_Coordinates').text),
                'Weight': int(load_info.find('Weight').text),
                'Quantity': int(load_info.find('Quantity').text),
                'ReadyTime': int(request.get('ReadyTime')),
                'ServiceTime': int(request.get('ServiceTime')),
                'DueDate': int(request.get('DueDate'))
            }
            problem_data['customers'].append(customer)
    
    print(f"✓ Problem loaded: {len(problem_data['customers'])} customers, Depot: {problem_data['depot']['Name']}")
    return problem_data 