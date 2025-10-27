"""
VRP Environment - Temel VRP Ortam Tanımı

Bu modül, Vehicle Routing Problem (VRP) ve Traveling Salesman Problem (TSP) 
için temel ortam yapısını sağlar.

Autor: Yasin
Tarih: 28 Ekim 2025
"""

import numpy as np
import torch
from typing import Tuple, List, Optional
from dataclasses import dataclass


@dataclass
class VRPInstance:
    """
    Bir VRP problem örneğini temsil eder.
    
    Attributes:
        depot: Depo koordinatları (x, y)
        customers: Müşteri koordinatları liste [(x1, y1), (x2, y2), ...]
        demands: Her müşterinin talep miktarı (opsiyonel, CVRP için)
        capacity: Araç kapasitesi (opsiyonel, CVRP için)
        time_windows: Zaman pencereleri (opsiyonel, VRPTW için)
    """
    depot: Tuple[float, float]
    customers: List[Tuple[float, float]]
    demands: Optional[List[float]] = None
    capacity: Optional[float] = None
    time_windows: Optional[List[Tuple[float, float]]] = None
    
    @property
    def num_customers(self) -> int:
        return len(self.customers)
    
    def to_tensor(self) -> torch.Tensor:
        """Problem örneğini PyTorch tensor'üne çevir"""
        all_coords = [self.depot] + self.customers
        return torch.FloatTensor(all_coords)
    
    def distance_matrix(self) -> np.ndarray:
        """Tüm noktalar arası Öklid mesafe matrisini hesapla"""
        all_coords = np.array([self.depot] + self.customers)
        n = len(all_coords)
        dist_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                dist_matrix[i, j] = np.linalg.norm(all_coords[i] - all_coords[j])
        
        return dist_matrix


class VRPEnvironment:
    """
    VRP/TSP için Reinforcement Learning ortamı.
    
    Bu sınıf, RL agent'ının etkileşim kuracağı ortamı tanımlar.
    State (durum), action (aksiyon), reward (ödül) kavramlarını içerir.
    """
    
    def __init__(self, instance: VRPInstance):
        """
        Args:
            instance: Çözülecek VRP problem örneği
        """
        self.instance = instance
        self.num_customers = instance.num_customers
        self.dist_matrix = instance.distance_matrix()
        
        # Ortam durumu
        self.current_location = 0  # Başlangıç: depo (index 0)
        self.visited = set([0])    # Ziyaret edilen lokasyonlar
        self.tour = [0]            # Tur sırası
        self.total_distance = 0.0  # Toplam mesafe
        
    def reset(self) -> torch.Tensor:
        """
        Ortamı başlangıç durumuna getir.
        
        Returns:
            Initial state (başlangıç durumu)
        """
        self.current_location = 0
        self.visited = set([0])
        self.tour = [0]
        self.total_distance = 0.0
        
        return self._get_state()
    
    def _get_state(self) -> torch.Tensor:
        """
        Mevcut durumu tensor formatında döndür.
        
        State representation:
        - Tüm lokasyon koordinatları
        - Hangi lokasyonların ziyaret edildiği (binary mask)
        - Mevcut lokasyon
        """
        coords = self.instance.to_tensor()
        
        # Visited mask: 1 = ziyaret edildi, 0 = ziyaret edilmedi
        visited_mask = torch.zeros(len(coords))
        for idx in self.visited:
            visited_mask[idx] = 1
        
        # Current location one-hot encoding
        current_mask = torch.zeros(len(coords))
        current_mask[self.current_location] = 1
        
        # State'i birleştir
        state = {
            'coords': coords,
            'visited': visited_mask,
            'current': current_mask
        }
        
        return state
    
    def step(self, action: int) -> Tuple[torch.Tensor, float, bool, dict]:
        """
        Bir aksiyon uygula ve ortamı güncelle.
        
        Args:
            action: Gidilecek lokasyon index'i (0: depo, 1-n: müşteriler)
        
        Returns:
            next_state: Yeni durum
            reward: Ödül (genelde negatif mesafe)
            done: Episode bitti mi?
            info: Ek bilgiler
        """
        # Aksiyonun geçerli olup olmadığını kontrol et
        if action in self.visited and action != 0:
            # Geçersiz aksiyon: zaten ziyaret edilmiş (depo hariç)
            reward = -1000.0  # Büyük ceza
            done = False
            info = {'error': 'Invalid action: already visited'}
            return self._get_state(), reward, done, info
        
        # Mesafeyi hesapla
        distance = self.dist_matrix[self.current_location, action]
        
        # Ortamı güncelle
        self.tour.append(action)
        self.visited.add(action)
        self.total_distance += distance
        self.current_location = action
        
        # Reward: negatif mesafe (daha kısa = daha iyi)
        reward = -distance
        
        # Episode bitti mi kontrol et
        # Tüm müşteriler ziyaret edildi ve depoya dönüldü mü?
        done = (len(self.visited) == self.num_customers + 1 and action == 0)
        
        info = {
            'total_distance': self.total_distance,
            'tour_length': len(self.tour),
            'visited_count': len(self.visited)
        }
        
        return self._get_state(), reward, done, info
    
    def get_valid_actions(self) -> List[int]:
        """
        Mevcut durumda yapılabilecek geçerli aksiyonları döndür.
        
        Returns:
            Geçerli aksiyon indeksleri listesi
        """
        valid = []
        
        # Eğer tüm müşteriler ziyaret edildiyse, sadece depoya dönülebilir
        if len(self.visited) == self.num_customers + 1:
            return [0]
        
        # Ziyaret edilmemiş müşteriler
        for i in range(1, self.num_customers + 1):
            if i not in self.visited:
                valid.append(i)
        
        return valid
    
    def get_tour_info(self) -> dict:
        """
        Mevcut turun detaylı bilgisini döndür.
        
        Returns:
            Tur bilgileri (sıra, mesafe, vb.)
        """
        return {
            'tour': self.tour.copy(),
            'total_distance': self.total_distance,
            'num_visited': len(self.visited),
            'is_complete': len(self.visited) == self.num_customers + 1
        }


def generate_random_vrp_instance(num_customers: int = 20, 
                                 grid_size: float = 1.0,
                                 seed: Optional[int] = None) -> VRPInstance:
    """
    Rastgele bir VRP/TSP problem örneği üret.
    
    Args:
        num_customers: Müşteri sayısı
        grid_size: Koordinat alanı boyutu [0, grid_size] x [0, grid_size]
        seed: Random seed (tekrarlanabilirlik için)
    
    Returns:
        VRPInstance objesi
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Depo: merkeze yakın rastgele bir nokta
    depot = (np.random.uniform(0.4, 0.6) * grid_size, 
             np.random.uniform(0.4, 0.6) * grid_size)
    
    # Müşteriler: rastgele dağılmış noktalar
    customers = [(np.random.uniform(0, grid_size), 
                  np.random.uniform(0, grid_size)) 
                 for _ in range(num_customers)]
    
    return VRPInstance(depot=depot, customers=customers)


if __name__ == "__main__":
    # Test kodu
    print("=" * 60)
    print("VRP Environment Test")
    print("=" * 60)
    
    # Rastgele bir problem örneği oluştur
    instance = generate_random_vrp_instance(num_customers=5, seed=42)
    print(f"\n📦 Problem: {instance.num_customers} müşteri")
    print(f"   Depo: {instance.depot}")
    print(f"   İlk 3 müşteri: {instance.customers[:3]}")
    
    # Ortamı başlat
    env = VRPEnvironment(instance)
    state = env.reset()
    print(f"\n🎯 Başlangıç durumu:")
    print(f"   Mevcut lokasyon: {env.current_location}")
    print(f"   Geçerli aksiyonlar: {env.get_valid_actions()}")
    
    # Basit bir tur simülasyonu (greedy nearest neighbor)
    print(f"\n🚗 Greedy Nearest Neighbor Turu:")
    while not env.get_tour_info()['is_complete']:
        valid_actions = env.get_valid_actions()
        
        if len(valid_actions) == 1 and valid_actions[0] == 0:
            # Depoya dön
            action = 0
        else:
            # En yakın müşteriyi seç
            distances = [env.dist_matrix[env.current_location, a] 
                        for a in valid_actions]
            action = valid_actions[np.argmin(distances)]
        
        state, reward, done, info = env.step(action)
        print(f"   Adım {len(env.tour)-1}: Lokasyon {action}, "
              f"Mesafe: {-reward:.3f}, Toplam: {info['total_distance']:.3f}")
        
        if done:
            break
    
    # Final sonuç
    tour_info = env.get_tour_info()
    print(f"\n✅ Tur tamamlandı!")
    print(f"   Tur sırası: {tour_info['tour']}")
    print(f"   Toplam mesafe: {tour_info['total_distance']:.3f}")
    print("=" * 60)
