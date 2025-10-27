"""
Metrics - VRP Çözüm Kalitesini Değerlendirme

Bu modül, VRP çözümlerinin kalitesini ölçmek için metrikler içerir.

Autor: Yasin
Tarih: 28 Ekim 2025
"""

import numpy as np
from typing import List, Tuple


def calculate_tour_length(coordinates: List[Tuple[float, float]], 
                          tour: List[int]) -> float:
    """
    Bir turun toplam uzunluğunu hesapla.
    
    Args:
        coordinates: Nokta koordinatları [(x, y), ...]
        tour: Ziyaret sırası [0, 2, 1, 3, 0]
    
    Returns:
        Toplam tur uzunluğu (Öklid mesafesi)
    """
    coords = np.array(coordinates)
    total_length = 0.0
    
    for i in range(len(tour) - 1):
        start = coords[tour[i]]
        end = coords[tour[i + 1]]
        total_length += np.linalg.norm(end - start)
    
    return total_length


def calculate_gap(solution_value: float, optimal_value: float) -> float:
    """
    Çözüm ile optimal arasındaki gap'i hesapla (%).
    
    Args:
        solution_value: Bulunan çözümün değeri
        optimal_value: Optimal çözümün değeri (biliniyor ise)
    
    Returns:
        Gap yüzdesi
    """
    return ((solution_value - optimal_value) / optimal_value) * 100


def evaluate_multiple_solutions(coordinates: List[Tuple[float, float]],
                                tours: List[List[int]]) -> dict:
    """
    Birden fazla çözümü değerlendir ve istatistikler ver.
    
    Args:
        coordinates: Problem koordinatları
        tours: Çözüm turları listesi
    
    Returns:
        İstatistikler dict'i
    """
    lengths = [calculate_tour_length(coordinates, tour) for tour in tours]
    
    return {
        'mean': np.mean(lengths),
        'std': np.std(lengths),
        'min': np.min(lengths),
        'max': np.max(lengths),
        'median': np.median(lengths),
        'best_tour_idx': np.argmin(lengths),
        'all_lengths': lengths
    }


def is_valid_vrp_tour(tour: List[int], num_customers: int) -> bool:
    """
    Bir turun geçerli olup olmadığını kontrol et.
    
    Geçerli bir tur:
    - Depo (0) ile başlar ve biter
    - Her müşteri tam bir kez ziyaret edilir
    - Ardışık ziyaretler arasında tekrar yok (depo hariç)
    
    Args:
        tour: Kontrol edilecek tur
        num_customers: Toplam müşteri sayısı
    
    Returns:
        Geçerli ise True, değilse False
    """
    # Depo ile başlayıp bitiyor mu?
    if tour[0] != 0 or tour[-1] != 0:
        return False
    
    # Tüm müşteriler ziyaret edilmiş mi?
    visited = set(tour[1:-1])  # İlk ve son depoyu çıkar
    expected = set(range(1, num_customers + 1))
    
    if visited != expected:
        return False
    
    # Her müşteri sadece bir kez mi?
    if len(tour[1:-1]) != len(visited):
        return False
    
    return True


if __name__ == "__main__":
    # Test
    print("=" * 60)
    print("Metrics Test")
    print("=" * 60)
    
    coords = [(0.5, 0.5), (0.2, 0.8), (0.7, 0.9), (0.9, 0.3)]
    tour = [0, 1, 2, 3, 0]
    
    length = calculate_tour_length(coords, tour)
    print(f"\n📏 Tour Length: {length:.3f}")
    
    is_valid = is_valid_vrp_tour(tour, num_customers=3)
    print(f"✓ Valid Tour: {is_valid}")
    
    # Multiple solutions
    tours = [
        [0, 1, 2, 3, 0],
        [0, 3, 2, 1, 0],
        [0, 2, 1, 3, 0]
    ]
    
    stats = evaluate_multiple_solutions(coords, tours)
    print(f"\n📊 Statistics:")
    print(f"   Mean: {stats['mean']:.3f}")
    print(f"   Std: {stats['std']:.3f}")
    print(f"   Best: {stats['min']:.3f} (tour {stats['best_tour_idx']})")
    
    print("\n" + "=" * 60)
