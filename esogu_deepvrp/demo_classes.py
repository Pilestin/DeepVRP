"""
VRPProblem örneği ve temel kullanım.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from data_classes import Depot, Customer, Vehicle, VRPProblem


def demo_data_classes():
    """Veri sınıflarının kullanımını gösterir."""
    
    print("\n" + "="*60)
    print("DATA CLASSES DEMO")
    print("="*60)
    
    # Depot oluştur
    depot = Depot(
        no="121",
        name="cs5",
        latitude=39.751377,
        longitude=30.481888,
        x=5307.25,
        y=2900.51
    )
    print(f"\n{depot}")
    print(f"  GPS: {depot.get_gps()}")
    print(f"  Coordinates: {depot.get_coordinates()}")
    
    # Customer oluştur
    customer = Customer(
        no="98",
        name="22A",
        node_type="Delivery",
        latitude=39.750739,
        longitude=30.489428,
        x=5951.61,
        y=2812.234,
        weight=95,
        quantity=5,
        ready_time=1476,
        service_time=240,
        due_date=1540
    )
    print(f"\n{customer}")
    print(f"  Time Window: {customer.get_time_window()}")
    print(f"  Is Delivery: {customer.is_delivery()}")
    
    # Vehicle oluştur
    vehicle = Vehicle(
        vehicle_id=1,
        capacity=200.0,
        battery_capacity=100.0
    )
    print(f"\n{vehicle}")
    
    # Yük ekle
    vehicle.load(50)
    print(f"  After loading 50kg: {vehicle}")
    
    # Batarya tüket
    vehicle.consume_battery(15.5)
    print(f"  After consuming 15.5kWh: {vehicle}")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    demo_data_classes()
