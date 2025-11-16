"""
Basit greedy algoritma ile VRP çözümü.
Nearest neighbor heuristic kullanılarak rota oluşturur.
"""

import numpy as np
from typing import List, Tuple, Dict
from data_classes import VRPProblem, Vehicle


class VRPSolution:
    """VRP çözüm sonuçlarını tutan sınıf."""
    
    def __init__(self):
        self.routes: List[List[str]] = []  # Her araç için rota (node isimleri)
        self.route_distances: List[float] = []  # Her rotanın toplam mesafesi
        self.route_times: List[float] = []  # Her rotanın toplam süresi
        self.route_energies: List[float] = []  # Her rotanın toplam enerjisi
        self.total_distance: float = 0.0
        self.total_time: float = 0.0
        self.total_energy: float = 0.0
        self.num_vehicles_used: int = 0
    
    def add_route(self, route: List[str], distance: float, time: float, energy: float):
        """Rota ekle."""
        self.routes.append(route)
        self.route_distances.append(distance)
        self.route_times.append(time)
        self.route_energies.append(energy)
        self.total_distance += distance
        self.total_time += time
        self.total_energy += energy
        self.num_vehicles_used += 1
    
    def print_solution(self):
        """Çözümü formatlanmış şekilde yazdır."""
        print("\n" + "="*70)
        print("VRP SOLUTION RESULTS")
        print("="*70)
        
        print(f"\nNumber of vehicles used: {self.num_vehicles_used}")
        print(f"\nRoutes:")
        print("-"*70)
        
        for i, route in enumerate(self.routes, 1):
            route_str = " → ".join(route)
            print(f"\nVehicle {i}:")
            print(f"  Route: {route_str}")
            print(f"  Distance: {self.route_distances[i-1]:.2f} m")
            print(f"  Time: {self.route_times[i-1]:.2f} s ({self.route_times[i-1]/60:.2f} min)")
            print(f"  Energy: {self.route_energies[i-1]:.2f} kWh")
        
        print("\n" + "-"*70)
        print("TOTAL SUMMARY")
        print("-"*70)
        print(f"Total Distance: {self.total_distance:.2f} m")
        print(f"Total Time: {self.total_time:.2f} s ({self.total_time/60:.2f} min, {self.total_time/3600:.2f} hours)")
        print(f"Total Energy: {self.total_energy:.2f} kWh")
        print("="*70 + "\n")
    
    def get_routes_list(self) -> List[List[str]]:
        """Rota listesini döndür (format: [[cs5, 1, 2, cs5], ...])"""
        return self.routes


class GreedyVRPSolver:
    """
    Greedy Nearest Neighbor algoritması ile VRP çözümü.
    """
    
    def __init__(self, problem: VRPProblem):
        self.problem = problem
        self.solution = VRPSolution()
    
    def solve(self) -> VRPSolution:
        """
        VRP problemini greedy nearest neighbor ile çöz.
        """
        # Müşteri listesi (depot hariç)
        unvisited = set(range(1, self.problem.num_nodes))  # 0 = depot
        
        vehicle_idx = 0
        
        while unvisited and vehicle_idx < self.problem.num_vehicles:
            # Yeni araç için rota oluştur
            route, distance, time, energy = self._construct_route(unvisited, vehicle_idx)
            
            if route and len(route) > 2:  # En az depot-customer-depot olmalı
                self.solution.add_route(route, distance, time, energy)
                vehicle_idx += 1
            else:
                break
        
        # Eğer tüm müşteriler ziyaret edilmediyse uyarı ver
        if unvisited:
            print(f"\nWARNING: {len(unvisited)} customers could not be visited!")
            print(f"Unvisited customers: {unvisited}")
        
        return self.solution
    
    def _construct_route(
        self, 
        unvisited: set, 
        vehicle_idx: int
    ) -> Tuple[List[str], float, float, float]:
        """
        Bir araç için rota oluştur.
        
        Returns:
            route: Rota (node isimleri)
            distance: Toplam mesafe
            time: Toplam süre
            energy: Toplam enerji
        """
        vehicle = self.problem.vehicles[vehicle_idx]
        vehicle.reset()
        
        current_idx = 0  # Depot'tan başla
        route_indices = [0]
        route_names = [self.problem.depot.name]
        
        total_distance = 0.0
        total_time = 0.0
        total_energy = 0.0
        current_time = 0.0
        
        while unvisited:
            # En yakın uygun müşteriyi bul
            best_customer = None
            best_distance = float('inf')
            
            for customer_idx in unvisited:
                customer = self.problem.customers[customer_idx - 1]
                
                # Kapasite kontrolü
                if not vehicle.can_load(customer.weight):
                    continue
                
                # Mesafe hesapla
                distance = self.problem.get_distance(current_idx, customer_idx)
                energy = self.problem.get_energy(current_idx, customer_idx)
                
                # Batarya kontrolü
                # Müşteriye git + depoya dön için yeterli batarya var mı?
                distance_to_depot = self.problem.get_distance(customer_idx, 0)
                energy_to_depot = self.problem.get_energy(customer_idx, 0)
                
                if vehicle.current_battery < (energy + energy_to_depot):
                    continue
                
                # Zaman penceresi kontrolü - 12.5 m/s hız
                travel_time = distance / 12.5  # Seyahat süresi (saniye)
                arrival_time = current_time + travel_time
                
                # Ready time'dan önce gelirsek bekle
                service_start_time = max(arrival_time, customer.ready_time)
                service_end_time = service_start_time + customer.service_time
                
                # Due date'i aşmamalı (servis başlama zamanı due_date'ten önce olmalı)
                if service_start_time > customer.due_date:
                    continue
                
                # En yakın müşteriyi seç
                if distance < best_distance:
                    best_distance = distance
                    best_customer = customer_idx
            
            # Uygun müşteri bulunamadıysa döngüden çık
            if best_customer is None:
                break
            
            # Müşteriyi ziyaret et
            customer = self.problem.customers[best_customer - 1]
            distance = self.problem.get_distance(current_idx, best_customer)
            energy = self.problem.get_energy(current_idx, best_customer)
            
            # Güncelleme
            vehicle.load(customer.weight)
            vehicle.consume_battery(energy)
            
            total_distance += distance
            total_energy += energy
            
            # Zaman hesaplaması - 12.5 m/s hız
            travel_time = distance / 12.5  # Seyahat süresi (saniye)
            current_time += travel_time
            
            # Hazır olmasını bekle (ready_time'dan önce gelirsek)
            wait_time = 0.0
            if current_time < customer.ready_time:
                wait_time = customer.ready_time - current_time
                current_time = customer.ready_time
            
            # Servis süresi ekle
            current_time += customer.service_time
            total_time += travel_time + wait_time + customer.service_time
            
            # Rotaya ekle
            route_indices.append(best_customer)
            route_names.append(customer.name)
            
            # Ziyaret edildi olarak işaretle
            unvisited.remove(best_customer)
            current_idx = best_customer
        
        # Depoya dön
        if current_idx != 0:
            distance = self.problem.get_distance(current_idx, 0)
            energy = self.problem.get_energy(current_idx, 0)
            
            total_distance += distance
            total_energy += energy
            total_time += distance / 12.5  # 12.5 m/s hız
            
            route_indices.append(0)
            route_names.append(self.problem.depot.name)
        
        return route_names, total_distance, total_time, total_energy


def solve_vrp_problem(problem: VRPProblem) -> VRPSolution:
    """
    VRP problemini çöz ve sonuçları döndür.
    
    Args:
        problem: VRPProblem instance
    
    Returns:
        VRPSolution with routes and statistics
    """
    solver = GreedyVRPSolver(problem)
    solution = solver.solve()
    return solution
