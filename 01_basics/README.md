# 🎓 Temel Kavramlar - VRP ve Reinforcement Learning'e Giriş

## 📖 İçindekiler

1. [VRP Nedir?](#vrp-nedir)
2. [VRP Ortam Yapısı](#vrp-ortam-yapısı)
3. [Reinforcement Learning Temelleri](#rl-temelleri)
4. [Modül Açıklamaları](#modül-açıklamaları)

---

## 🚗 VRP Nedir?

**Vehicle Routing Problem (VRP)**, kombinatoryal optimizasyon problemlerinden biridir. Temel olarak:

> Bir depodan başlayarak, belirli sayıda müşteriyi ziyaret edip tekrar depoya dönen optimal rotayı bulma problemi.

### VRP Varyantları

1. **TSP (Traveling Salesman Problem)**
   - En basit form: tek araç, tüm noktaları ziyaret et
   - VRP'nin temel hali

2. **CVRP (Capacitated VRP)**
   - Her müşterinin talebi var
   - Araç kapasitesi sınırlı
   - Birden fazla tur gerekebilir

3. **VRPTW (VRP with Time Windows)**
   - Her müşterinin zaman penceresi var
   - Belirli saatler arasında ziyaret edilmeli

4. **EVRP (Electric VRP)**
   - Elektrikli araçlar
   - Şarj istasyonları
   - Batarya kısıtları

### Neden Zor?

- **NP-Hard** problem: n müşteri için n! olası permütasyon
- 20 müşteri = 2.43 × 10^18 olası tur!
- Exact çözümler büyük problemlerde imkansız
- Metasezgiseller (ALNS, Tabu Search) kullanılır

### Deep Learning ile Çözüm

**Geleneksel Yöntemler:**
- Matematiksel optimizasyon (MIP)
- Metasezgiseller (ALNS, VNS, Tabu Search)
- Her problem için yeniden çalıştır

**Deep Learning Avantajları:**
- ✅ Öğrenilmiş bir model: bir kez eğit, hızlıca çöz
- ✅ Generalization: farklı boyutlarda çalışabilir
- ✅ End-to-end: feature engineering yok
- ✅ Scalability: büyük problemlere adapt olabilir

**Dezavantajlar:**
- ❌ Eğitim süresi uzun
- ❌ Optimal garanti yok
- ❌ Kısıtlı problemlerde (VRPTW) daha zor

---

## 🏗️ VRP Ortam Yapısı

### VRPInstance Sınıfı

Problem tanımını tutar:

```python
instance = VRPInstance(
    depot=(0.5, 0.5),          # Depo koordinatları
    customers=[(x1,y1), ...],   # Müşteri koordinatları
    demands=[d1, d2, ...],      # Opsiyonel: müşteri talepleri
    capacity=100                # Opsiyonel: araç kapasitesi
)
```

**Özellikler:**
- `num_customers`: Müşteri sayısı
- `to_tensor()`: PyTorch tensor'üne çevir
- `distance_matrix()`: Mesafe matrisini hesapla

### VRPEnvironment Sınıfı

RL agent'ının etkileşim kurduğu ortam:

```python
env = VRPEnvironment(instance)
state = env.reset()                           # Başlangıç
action = agent.select_action(state)           # Aksiyon seç
next_state, reward, done, info = env.step(action)  # Adım at
```

**State (Durum):**
```python
state = {
    'coords': Tensor[N, 2],      # Koordinatlar
    'visited': Tensor[N],         # Ziyaret mask (0/1)
    'current': Tensor[N]          # Mevcut lokasyon (one-hot)
}
```

**Action (Aksiyon):**
- Integer: 0 (depo), 1, 2, ..., N (müşteriler)
- Gidilecek lokasyonun index'i

**Reward (Ödül):**
- Negatif mesafe: `reward = -distance`
- Daha kısa mesafe = daha yüksek ödül
- Geçersiz aksiyon = büyük ceza (-1000)

**Done (Bitti mi):**
- `True`: Tüm müşteriler ziyaret edildi ve depoya dönüldü
- `False`: Episode devam ediyor

---

## 🤖 RL Temelleri

### Markov Decision Process (MDP)

VRP bir MDP olarak modellenir:

**Tanım:** MDP = (S, A, P, R, γ)
- **S**: State space (durum uzayı)
- **A**: Action space (aksiyon uzayı)
- **P**: Transition probability (geçiş olasılığı) - VRP'de deterministik
- **R**: Reward function (ödül fonksiyonu)
- **γ**: Discount factor (indirim faktörü)

### VRP MDP Representation

| Bileşen | VRP'de Karşılığı |
|---------|------------------|
| State | Koordinatlar + ziyaret durumu + mevcut lokasyon |
| Action | Bir sonraki ziyaret edilecek müşteri |
| Reward | Negatif mesafe (kısa tur = yüksek ödül) |
| Done | Tüm müşteriler ziyaret edildi mi? |

### RL Öğrenme Süreci

```
1. Agent durumu gözlemler (observe state)
2. Policy'ye göre aksiyon seçer (select action)
3. Ortam aksiyonu uygular (execute action)
4. Ödül alır (receive reward)
5. Yeni duruma geçer (transition to new state)
6. Deneyimden öğrenir (learn from experience)
```

### Exploration vs Exploitation

**Exploration:** Yeni aksiyonlar dene, bilinmeyeni keşfet
**Exploitation:** Bilinen en iyi aksiyonu kullan

**Stochastic Policy:** π(a|s) olasılıksal
- Doğal exploration sağlar
- Deterministic policy'den daha robust

---

## 📁 Modül Açıklamaları

### 1. `vrp_environment.py`

**Ana Sınıflar:**
- `VRPInstance`: Problem tanımı
- `VRPEnvironment`: RL ortamı
- `generate_random_vrp_instance()`: Rastgele problem üretici

**Kullanım:**
```python
from vrp_environment import *

# Problem oluştur
instance = generate_random_vrp_instance(num_customers=10, seed=42)

# Ortamı başlat
env = VRPEnvironment(instance)
state = env.reset()

# Greedy nearest neighbor ile çöz
while not env.get_tour_info()['is_complete']:
    valid = env.get_valid_actions()
    # En yakın müşteriyi seç
    distances = [env.dist_matrix[env.current_location, a] for a in valid]
    action = valid[np.argmin(distances)]
    state, reward, done, info = env.step(action)

print(f"Tour: {env.tour}")
print(f"Total Distance: {env.total_distance:.3f}")
```

### 2. `visualizer.py`

**Fonksiyonlar:**
- `plot_vrp_tour()`: Tek tur görselleştir
- `plot_training_progress()`: Eğitim grafikleri
- `plot_multiple_tours()`: Turları karşılaştır

**Kullanım:**
```python
from visualizer import plot_vrp_tour

coords = [(0.5, 0.5), (0.2, 0.8), (0.7, 0.3)]
tour = [0, 1, 2, 0]

plot_vrp_tour(coords, tour, 
              title="My VRP Solution",
              save_path="my_tour.png",
              show=True)
```

---

## 🧪 İlk Deneyler

### 1. Ortamı Test Et

```bash
cd 01_basics
python vrp_environment.py
```

**Beklenen Çıktı:**
- Problem detayları
- Greedy NN ile oluşturulmuş bir tur
- Toplam mesafe

### 2. Görselleştirmeyi Test Et

```bash
python visualizer.py
```

**Beklenen Çıktı:**
- `experiments/results/` klasöründe grafikler
- Test turları ve karşılaştırmalar

### 3. Kendi Probleminizi Oluşturun

```python
import numpy as np
from vrp_environment import VRPInstance, VRPEnvironment
from visualizer import plot_vrp_tour

# Manuel problem tanımla
instance = VRPInstance(
    depot=(50, 50),
    customers=[
        (20, 80), (80, 20), (30, 30), 
        (70, 70), (50, 90)
    ]
)

env = VRPEnvironment(instance)

# Random policy ile çöz
state = env.reset()
import random
while not env.get_tour_info()['is_complete']:
    valid = env.get_valid_actions()
    action = random.choice(valid)
    state, reward, done, info = env.step(action)

# Görselleştir
coords = [instance.depot] + instance.customers
plot_vrp_tour(coords, env.tour, title="Random Policy Tour")
```

---

## ❓ Sık Sorulan Sorular

**S: VRP'yi neden RL ile çözüyoruz?**
**C:** RL, sequential decision making problemleri için idealdir. VRP'de her adımda "sonraki hangi müşteri?" kararı veriyoruz - bu bir RL problemidir.

**S: Neden state representation bu şekilde?**
**C:** Koordinatlar spatial bilgi, visited mask geçmiş bilgi, current mask mevcut durumu verir. Agent'ın karar vermesi için yeterli bilgi.

**S: Reward neden negatif mesafe?**
**C:** Daha kısa tur = daha iyi. Negatif mesafe kullanarak, maksimizasyon problemi haline getiriyoruz (RL standard'ı).

**S: Ortam deterministik mi?**
**C:** Evet, aynı aksiyonu aynı durumda uygularsanız aynı sonucu alırsınız. Stochasticity policy'de (agent'ta).

---

## 🚀 Sonraki Adımlar

Temelleri öğrendiniz! Şimdi:

1. ✅ **Policy Gradient** (02_rl_methods/policy_gradient)
   - Basit RL ile VRP çözümü
   - REINFORCE algoritması

2. ⏭️ **Actor-Critic** (02_rl_methods/actor_critic)
   - Daha kararlı öğrenme
   - Value function entegrasyonu

3. ⏭️ **Attention Mechanism** (02_rl_methods/attention_rl)
   - Modern VRP çözümleri
   - State-of-the-art performans

---

**Hazır mısınız?** İlk RL modelinizi eğitmeye başlayalım! 🎯

```bash
cd ../02_rl_methods/policy_gradient
python simple_policy.py
```
