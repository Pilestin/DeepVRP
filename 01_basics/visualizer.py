"""
Visualizer - VRP Çözümlerini Görselleştirme

Bu modül, VRP turlarını ve öğrenme sürecini görselleştirmek için 
fonksiyonlar içerir.

Autor: Yasin
Tarih: 28 Ekim 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
import os


def plot_vrp_tour(coordinates: List[Tuple[float, float]], 
                  tour: List[int],
                  title: str = "VRP Tour",
                  save_path: Optional[str] = None,
                  show: bool = True):
    """
    VRP turunu görselleştir.
    
    Args:
        coordinates: Tüm nokta koordinatları [(x, y), ...] (depo + müşteriler)
        tour: Ziyaret sırası [0, 3, 1, 2, 0]
        title: Grafik başlığı
        save_path: Kaydetme yolu (opsiyonel)
        show: Göster (True) veya sadece kaydet (False)
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    coords = np.array(coordinates)
    
    # Depoyu kırmızı ile çiz
    ax.scatter(coords[0, 0], coords[0, 1], c='red', s=200, marker='s', 
               label='Depot', zorder=3, edgecolors='black', linewidth=2)
    
    # Müşterileri mavi ile çiz
    ax.scatter(coords[1:, 0], coords[1:, 1], c='blue', s=100, 
               label='Customers', zorder=2, edgecolors='black', linewidth=1)
    
    # Nokta numaralarını yaz
    for idx, (x, y) in enumerate(coords):
        ax.annotate(str(idx), (x, y), fontsize=10, ha='center', va='center',
                   color='white', weight='bold', zorder=4)
    
    # Turu çiz (oklar ile)
    total_distance = 0
    for i in range(len(tour) - 1):
        start = coords[tour[i]]
        end = coords[tour[i + 1]]
        
        # Mesafe hesapla
        distance = np.linalg.norm(end - start)
        total_distance += distance
        
        # Ok çiz
        ax.annotate('', xy=end, xytext=start,
                   arrowprops=dict(arrowstyle='->', lw=2, color='green', 
                                 alpha=0.7, shrinkA=15, shrinkB=15))
        
        # Mesafeyi yazdır (ortada)
        mid_point = (start + end) / 2
        ax.text(mid_point[0], mid_point[1], f'{distance:.2f}', 
               fontsize=8, ha='center', bbox=dict(boxstyle='round,pad=0.3', 
               facecolor='yellow', alpha=0.5))
    
    ax.set_xlabel('X Coordinate', fontsize=12)
    ax.set_ylabel('Y Coordinate', fontsize=12)
    ax.set_title(f'{title}\nTotal Distance: {total_distance:.2f}', 
                fontsize=14, weight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Grafik kaydedildi: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_training_progress(rewards: List[float],
                          losses: Optional[List[float]] = None,
                          title: str = "Training Progress",
                          save_path: Optional[str] = None,
                          show: bool = True):
    """
    Eğitim sürecindeki ödül ve loss değerlerini görselleştir.
    
    Args:
        rewards: Episode başına toplam ödüller
        losses: Episode başına loss değerleri (opsiyonel)
        title: Grafik başlığı
        save_path: Kaydetme yolu (opsiyonel)
        show: Göster (True) veya sadece kaydet (False)
    """
    if losses is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(10, 5))
    
    # Rewards grafiği
    episodes = range(1, len(rewards) + 1)
    ax1.plot(episodes, rewards, label='Episode Reward', alpha=0.6)
    
    # Moving average (hareketli ortalama)
    window = min(50, len(rewards) // 10)
    if window > 1:
        moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        ax1.plot(range(window, len(rewards) + 1), moving_avg, 
                label=f'Moving Avg (w={window})', linewidth=2, color='red')
    
    ax1.set_xlabel('Episode', fontsize=12)
    ax1.set_ylabel('Total Reward', fontsize=12)
    ax1.set_title('Reward Progress', fontsize=13, weight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Loss grafiği (varsa)
    if losses is not None:
        ax2.plot(episodes, losses, label='Loss', alpha=0.6, color='orange')
        
        if window > 1:
            moving_avg_loss = np.convolve(losses, np.ones(window)/window, mode='valid')
            ax2.plot(range(window, len(losses) + 1), moving_avg_loss,
                    label=f'Moving Avg (w={window})', linewidth=2, color='red')
        
        ax2.set_xlabel('Episode', fontsize=12)
        ax2.set_ylabel('Loss', fontsize=12)
        ax2.set_title('Loss Progress', fontsize=13, weight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=15, weight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Grafik kaydedildi: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_multiple_tours(coordinates: List[Tuple[float, float]],
                       tours: List[List[int]],
                       labels: List[str],
                       title: str = "Tour Comparison",
                       save_path: Optional[str] = None,
                       show: bool = True):
    """
    Birden fazla turu yan yana karşılaştır.
    
    Args:
        coordinates: Nokta koordinatları
        tours: Tur listesi
        labels: Her tur için etiket
        title: Ana başlık
        save_path: Kaydetme yolu
        show: Göster veya kaydet
    """
    num_tours = len(tours)
    fig, axes = plt.subplots(1, num_tours, figsize=(8*num_tours, 7))
    
    if num_tours == 1:
        axes = [axes]
    
    coords = np.array(coordinates)
    
    for idx, (tour, label, ax) in enumerate(zip(tours, labels, axes)):
        # Depo
        ax.scatter(coords[0, 0], coords[0, 1], c='red', s=200, marker='s',
                  label='Depot', zorder=3, edgecolors='black', linewidth=2)
        
        # Müşteriler
        ax.scatter(coords[1:, 0], coords[1:, 1], c='blue', s=100,
                  label='Customers', zorder=2, edgecolors='black', linewidth=1)
        
        # Numaralar
        for i, (x, y) in enumerate(coords):
            ax.annotate(str(i), (x, y), fontsize=9, ha='center', va='center',
                       color='white', weight='bold', zorder=4)
        
        # Tur çiz
        total_distance = 0
        for i in range(len(tour) - 1):
            start = coords[tour[i]]
            end = coords[tour[i + 1]]
            distance = np.linalg.norm(end - start)
            total_distance += distance
            
            ax.annotate('', xy=end, xytext=start,
                       arrowprops=dict(arrowstyle='->', lw=2, color='green',
                                     alpha=0.7, shrinkA=15, shrinkB=15))
        
        ax.set_xlabel('X Coordinate', fontsize=11)
        ax.set_ylabel('Y Coordinate', fontsize=11)
        ax.set_title(f'{label}\nDistance: {total_distance:.2f}',
                    fontsize=12, weight='bold')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
    
    plt.suptitle(title, fontsize=15, weight='bold')
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Grafik kaydedildi: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


if __name__ == "__main__":
    # Test kodu
    print("=" * 60)
    print("Visualizer Test")
    print("=" * 60)
    
    # Test verisi
    coords = [(0.5, 0.5), (0.2, 0.8), (0.7, 0.9), (0.9, 0.3), (0.3, 0.2)]
    tour1 = [0, 1, 2, 3, 4, 0]
    tour2 = [0, 4, 3, 2, 1, 0]
    
    # Tek tur
    print("\n🎨 Tek tur görselleştirme...")
    plot_vrp_tour(coords, tour1, title="Test Tour 1", 
                  save_path="experiments/results/test_tour.png", show=False)
    
    # Karşılaştırma
    print("\n🎨 Çoklu tur karşılaştırma...")
    plot_multiple_tours(coords, [tour1, tour2], 
                       labels=["Tour 1", "Tour 2"],
                       title="Tour Comparison Test",
                       save_path="experiments/results/test_comparison.png",
                       show=False)
    
    # Training progress
    print("\n🎨 Eğitim ilerleme grafiği...")
    rewards = [-np.random.exponential(10) + i*0.1 for i in range(200)]
    losses = [100 * np.exp(-i/50) + np.random.random() for i in range(200)]
    plot_training_progress(rewards, losses, title="Test Training",
                          save_path="experiments/results/test_training.png",
                          show=False)
    
    print("\n✅ Tüm testler tamamlandı!")
    print("=" * 60)
