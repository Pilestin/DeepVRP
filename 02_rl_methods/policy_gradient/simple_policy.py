"""
Simple Policy Gradient - TSP/VRP için Temel RL Modeli

Bu modül, REINFORCE algoritmasını kullanarak basit bir policy network 
implementasyonu içerir. VRP öğrenmeye giriş için idealdir.

⭐ TEMEL KAVRAMLAR:
- Policy Network: Durumdan aksiyona mapping yapan sinir ağı
- REINFORCE: Monte Carlo policy gradient algoritması
- Baseline: Varyansı azaltmak için kullanılan referans değer

Autor: Yasin
Tarih: 28 Ekim 2025
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Tuple, Dict
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from basics.vrp_environment import VRPEnvironment, VRPInstance, generate_random_vrp_instance
from basics.visualizer import plot_vrp_tour, plot_training_progress


class SimplePolicyNetwork(nn.Module):
    """
    Basit Policy Network
    
    Bu ağ, mevcut durumu (state) alır ve hangi aksiyonun seçileceğine
    dair bir olasılık dağılımı (probability distribution) üretir.
    
    Mimari:
    - Input: Koordinatlar + visited mask + current location
    - Hidden Layers: Fully connected layers
    - Output: Her lokasyon için aksiyon olasılığı
    """
    
    def __init__(self, input_dim: int = 2, hidden_dim: int = 128, 
                 num_locations: int = 11):
        """
        Args:
            input_dim: Her lokasyonun özellik boyutu (x, y için 2)
            hidden_dim: Gizli katman boyutu
            num_locations: Toplam lokasyon sayısı (depo + müşteriler)
        """
        super(SimplePolicyNetwork, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_locations = num_locations
        
        # Feature encoder: Her lokasyonu ayrı ayrı encode et
        self.location_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Context encoder: Global durum bilgisi
        # visited + current için (num_locations * 2)
        self.context_encoder = nn.Sequential(
            nn.Linear(num_locations * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Policy head: Aksiyon olasılıklarını üret
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_locations)
        )
        
    def forward(self, coords: torch.Tensor, 
                visited: torch.Tensor, 
                current: torch.Tensor,
                valid_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass: Durumdan aksiyon olasılıklarına
        
        Args:
            coords: Koordinatlar [num_locations, 2]
            visited: Ziyaret mask [num_locations]
            current: Mevcut lokasyon mask [num_locations]
            valid_mask: Geçerli aksiyonlar için mask [num_locations]
                       1 = geçerli, 0 = geçersiz
        
        Returns:
            action_probs: Her aksiyon için olasılık [num_locations]
        """
        batch_size = coords.size(0) if len(coords.shape) > 2 else 1
        
        # Eğer batch değilse, batch dimension ekle
        if len(coords.shape) == 2:
            coords = coords.unsqueeze(0)
            visited = visited.unsqueeze(0)
            current = current.unsqueeze(0)
            if valid_mask is not None:
                valid_mask = valid_mask.unsqueeze(0)
        
        # Location features encode et
        # coords: [batch, num_locations, 2]
        loc_features = self.location_encoder(coords)  # [batch, num_locations, hidden]
        
        # Current location'ın feature'ını al
        current_expanded = current.unsqueeze(-1)  # [batch, num_locations, 1]
        current_feature = (loc_features * current_expanded).sum(dim=1)  # [batch, hidden]
        
        # Context encode et
        context_input = torch.cat([visited, current], dim=-1)  # [batch, num_locations*2]
        context_feature = self.context_encoder(context_input)  # [batch, hidden]
        
        # Policy head'e gir
        combined = torch.cat([current_feature, context_feature], dim=-1)
        logits = self.policy_head(combined)  # [batch, num_locations]
        
        # Geçersiz aksiyonları maskele (çok büyük negatif değer)
        if valid_mask is not None:
            logits = logits.masked_fill(valid_mask == 0, -1e9)
        
        # Softmax ile olasılıklara çevir
        action_probs = torch.softmax(logits, dim=-1)
        
        # Eğer tek örnekse, batch dimension'ı kaldır
        if batch_size == 1 and len(action_probs.shape) > 1:
            action_probs = action_probs.squeeze(0)
        
        return action_probs


class REINFORCEAgent:
    """
    REINFORCE Algoritması ile VRP Çözücü
    
    REINFORCE, Monte Carlo policy gradient yöntemidir:
    1. Policy network ile bir episode (tur) oluştur
    2. Episode tamamlandığında toplam ödülü hesapla
    3. Her adım için gradient hesapla ve network'ü güncelle
    
    Matematiksel olarak:
    ∇θ J(θ) = E[∑t ∇θ log π(at|st) * Gt]
    
    Burada:
    - θ: Network parametreleri
    - π: Policy (olasılık dağılımı)
    - Gt: t anından sonraki toplam ödül (return)
    """
    
    def __init__(self, num_locations: int, learning_rate: float = 1e-3):
        """
        Args:
            num_locations: Toplam lokasyon sayısı
            learning_rate: Öğrenme hızı
        """
        self.num_locations = num_locations
        self.policy_net = SimplePolicyNetwork(num_locations=num_locations)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
        # Episode history
        self.log_probs = []  # Her adımdaki log olasılıklar
        self.rewards = []    # Her adımdaki ödüller
        
    def select_action(self, state: Dict, valid_actions: List[int]) -> int:
        """
        Mevcut policy'ye göre bir aksiyon seç.
        
        Args:
            state: Ortam durumu (coords, visited, current)
            valid_actions: Geçerli aksiyon listesi
        
        Returns:
            Seçilen aksiyon index'i
        """
        coords = state['coords']
        visited = state['visited']
        current = state['current']
        
        # Valid mask oluştur
        valid_mask = torch.zeros(self.num_locations)
        for action in valid_actions:
            valid_mask[action] = 1
        
        # Policy network'ten olasılıkları al
        with torch.no_grad():
            action_probs = self.policy_net(coords, visited, current, valid_mask)
        
        # Olasılık dağılımından sample et
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        
        # Log probability'yi kaydet (eğitim için)
        self.log_probs.append(dist.log_prob(action))
        
        return action.item()
    
    def store_reward(self, reward: float):
        """Episode sırasında ödül kaydet"""
        self.rewards.append(reward)
    
    def train_episode(self):
        """
        Episode tamamlandıktan sonra policy'yi güncelle.
        
        REINFORCE algoritması:
        1. Returns (Gt) hesapla: Her adım için gelecekteki toplam ödül
        2. Policy gradient hesapla: ∇θ J(θ) = ∑ log_prob * (Gt - baseline)
        3. Gradient descent ile parametreleri güncelle
        """
        # Returns hesapla (discounted cumulative rewards)
        returns = []
        G = 0
        gamma = 0.99  # Discount factor
        
        # Geriye doğru git
        for reward in reversed(self.rewards):
            G = reward + gamma * G
            returns.insert(0, G)
        
        returns = torch.tensor(returns)
        
        # Normalize et (variance azaltma)
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Policy loss hesapla
        policy_loss = []
        for log_prob, G in zip(self.log_probs, returns):
            # Negative gradient: gradient ascent için minimize ediyoruz
            policy_loss.append(-log_prob * G)
        
        policy_loss = torch.stack(policy_loss).sum()
        
        # Backpropagation
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()
        
        # History'yi temizle
        loss_value = policy_loss.item()
        self.log_probs = []
        self.rewards = []
        
        return loss_value


def train_vrp_solver(num_episodes: int = 1000,
                    num_customers: int = 10,
                    print_every: int = 100,
                    save_dir: str = "experiments"):
    """
    Policy Gradient ile VRP solver'ı eğit.
    
    Args:
        num_episodes: Toplam episode sayısı
        num_customers: Problem boyutu (müşteri sayısı)
        print_every: Her kaç episode'da bir progress yazdır
        save_dir: Model ve sonuçların kaydedileceği klasör
    """
    print("=" * 70)
    print("🚀 REINFORCE ile VRP Eğitimi Başlıyor")
    print("=" * 70)
    print(f"📊 Parametreler:")
    print(f"   - Episode sayısı: {num_episodes}")
    print(f"   - Müşteri sayısı: {num_customers}")
    print(f"   - Toplam lokasyon: {num_customers + 1} (depo dahil)")
    print("=" * 70)
    
    # Agent oluştur
    agent = REINFORCEAgent(num_locations=num_customers + 1, learning_rate=1e-3)
    
    # Training history
    episode_rewards = []
    episode_losses = []
    episode_distances = []
    
    best_distance = float('inf')
    best_tour = None
    best_coords = None
    
    for episode in range(num_episodes):
        # Yeni problem instance oluştur
        instance = generate_random_vrp_instance(num_customers=num_customers)
        env = VRPEnvironment(instance)
        
        # Episode başlat
        state = env.reset()
        done = False
        episode_reward = 0
        
        # Episode boyunca aksiyonlar al
        while not done:
            valid_actions = env.get_valid_actions()
            action = agent.select_action(state, valid_actions)
            
            next_state, reward, done, info = env.step(action)
            agent.store_reward(reward)
            
            episode_reward += reward
            state = next_state
        
        # Episode tamamlandı, eğit
        loss = agent.train_episode()
        
        # Metrikleri kaydet
        tour_info = env.get_tour_info()
        episode_rewards.append(episode_reward)
        episode_losses.append(loss)
        episode_distances.append(tour_info['total_distance'])
        
        # En iyi turu kaydet
        if tour_info['total_distance'] < best_distance:
            best_distance = tour_info['total_distance']
            best_tour = tour_info['tour']
            best_coords = [instance.depot] + instance.customers
        
        # Progress yazdır
        if (episode + 1) % print_every == 0:
            avg_reward = np.mean(episode_rewards[-print_every:])
            avg_distance = np.mean(episode_distances[-print_every:])
            print(f"Episode {episode+1}/{num_episodes} | "
                  f"Avg Reward: {avg_reward:.2f} | "
                  f"Avg Distance: {avg_distance:.2f} | "
                  f"Best Distance: {best_distance:.2f}")
    
    print("\n" + "=" * 70)
    print("✅ Eğitim Tamamlandı!")
    print("=" * 70)
    print(f"🏆 En İyi Tur Mesafesi: {best_distance:.3f}")
    print(f"📍 Tur Sırası: {best_tour}")
    
    # Sonuçları kaydet ve görselleştir
    os.makedirs(f"{save_dir}/results", exist_ok=True)
    os.makedirs(f"{save_dir}/models", exist_ok=True)
    
    # Model kaydet
    torch.save(agent.policy_net.state_dict(), 
              f"{save_dir}/models/policy_net_simple.pth")
    print(f"\n💾 Model kaydedildi: {save_dir}/models/policy_net_simple.pth")
    
    # Grafikleri çiz
    plot_training_progress(episode_rewards, episode_losses,
                          title="REINFORCE Training Progress",
                          save_path=f"{save_dir}/results/training_progress.png",
                          show=False)
    
    if best_tour and best_coords:
        plot_vrp_tour(best_coords, best_tour,
                     title=f"Best Tour (Distance: {best_distance:.2f})",
                     save_path=f"{save_dir}/results/best_tour.png",
                     show=False)
    
    print(f"📊 Grafikler kaydedildi: {save_dir}/results/")
    print("=" * 70)
    
    return agent, episode_rewards, episode_distances


if __name__ == "__main__":
    # Eğitimi başlat
    agent, rewards, distances = train_vrp_solver(
        num_episodes=1000,
        num_customers=10,
        print_every=100,
        save_dir="experiments"
    )
    
    print("\n🎓 Eğitim tamamlandı. Model hazır!")
    print("💡 Sonuçları 'experiments/results/' klasöründe inceleyebilirsiniz.")
