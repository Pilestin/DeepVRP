# 🎯 Policy Gradient ile VRP Çözümü

## 📚 Teorik Arka Plan

### Reinforcement Learning Temelleri

**Reinforcement Learning (RL)**, bir agent'ın bir ortamda aksiyonlar alarak maksimum ödül elde etmeyi öğrendiği bir makine öğrenmesi yaklaşımıdır.

#### Temel Bileşenler:

1. **State (Durum - s)**: Ortamın mevcut durumu
   - VRP'de: Hangi müşterilerin ziyaret edildiği, mevcut konum, vb.

2. **Action (Aksiyon - a)**: Agent'ın alabileceği eylemler
   - VRP'de: Bir sonraki ziyaret edilecek müşteri

3. **Reward (Ödül - r)**: Bir aksiyonun ne kadar iyi olduğunu gösteren sinyal
   - VRP'de: Negatif mesafe (daha kısa = daha iyi)

4. **Policy (Politika - π)**: Durumdan aksiyona mapping
   - π(a|s): s durumunda a aksiyonunu alma olasılığı

5. **Value Function (Değer Fonksiyonu - V)**: Bir durumun ne kadar iyi olduğu
   - V(s): s durumundan başlayarak elde edilebilecek beklenen toplam ödül

### Policy Gradient Yöntemleri

Policy Gradient, policy'yi **doğrudan** parametreleştirip optimize eder.

**Amaç:** Policy parametrelerini (θ) optimize et
```
J(θ) = E[∑t γ^t * r_t]  (Beklenen toplam ödül)
```

**Gradient:**
```
∇θ J(θ) = E[∑t ∇θ log π_θ(a_t|s_t) * G_t]
```

Burada:
- `G_t` = ∑k γ^k * r_(t+k) (t anından sonraki toplam ödül)
- `γ` = discount factor (gelecek ödüllerinin önemi)

### REINFORCE Algoritması

**REINFORCE**, en basit policy gradient algoritmasıdır:

1. Policy network ile bir episode oynat
2. Her adım için log probability kaydet: `log π(a_t|s_t)`
3. Episode sonunda returns hesapla: `G_t`
4. Loss hesapla: `L = -∑t log π(a_t|s_t) * G_t`
5. Backpropagation ile gradient descent yap

**Avantajlar:**
- ✅ Basit ve anlaşılır
- ✅ Sürekli aksiyon uzaylarında çalışır
- ✅ Stochastic policy öğrenir (exploration doğal)

**Dezavantajlar:**
- ❌ Yüksek variance (çözüm: baseline kullan)
- ❌ Sample inefficient (çok deneyim gerekir)
- ❌ Yavaş öğrenme

## 🏗️ Mimari

### SimplePolicyNetwork

```
Input: 
  - Coordinates [N, 2]
  - Visited mask [N]
  - Current location [N]

↓

Location Encoder (MLP):
  - Linear(2, 128)
  - ReLU
  - Linear(128, 128)

↓

Context Encoder (MLP):
  - Linear(2N, 128)
  - ReLU
  - Linear(128, 128)

↓

Policy Head:
  - Concat(location_feature, context_feature)
  - Linear(256, 128)
  - ReLU
  - Linear(128, N)

↓

Output: Action probabilities [N]
```

### Neden Bu Mimari?

1. **Location Encoder**: Her lokasyonun spatial özelliklerini öğrenir
2. **Context Encoder**: Global durum bilgisini (visited, current) encode eder
3. **Policy Head**: İkisini birleştirip aksiyon olasılıklarını üretir

## 💻 Kullanım

### Temel Eğitim

```python
from simple_policy import train_vrp_solver

# 10 müşterili VRP için 1000 episode eğit
agent, rewards, distances = train_vrp_solver(
    num_episodes=1000,
    num_customers=10,
    print_every=100
)
```

### Kendi Probleminizi Çözme

```python
from simple_policy import REINFORCEAgent
from vrp_environment import VRPInstance, VRPEnvironment

# Problem tanımla
instance = VRPInstance(
    depot=(0.5, 0.5),
    customers=[(0.2, 0.3), (0.8, 0.7), (0.4, 0.9)]
)

# Agent oluştur ve eğit
agent = REINFORCEAgent(num_locations=4)  # 3 müşteri + 1 depo
env = VRPEnvironment(instance)

# Bir çözüm üret
state = env.reset()
done = False
while not done:
    valid_actions = env.get_valid_actions()
    action = agent.select_action(state, valid_actions)
    state, reward, done, info = env.step(action)

tour_info = env.get_tour_info()
print(f"Tour: {tour_info['tour']}")
print(f"Distance: {tour_info['total_distance']}")
```

## 📊 Beklenen Sonuçlar

10 müşterili TSP için:
- **İlk 100 episode**: Random turlar, mesafe ~7-10
- **200-500 episode**: Öğrenme başlar, mesafe ~5-7
- **500-1000 episode**: İyi çözümler, mesafe ~4-5
- **Optimal çözüm**: ~3.5-4.5 (problem instance'a bağlı)

**Not:** Bu basit model, attention mekanizması olmadan sınırlı performans gösterir. İlerleyen modüllerde çok daha iyi sonuçlar alacağız!

## 🔬 Deneyler ve İyileştirmeler

### Hiperparametre Ayarları

```python
# Learning rate
lr = 1e-3  # Varsayılan, 1e-4 veya 1e-2 deneyin

# Network boyutu
hidden_dim = 128  # 64, 256 deneyin

# Discount factor
gamma = 0.99  # 0.95, 1.0 deneyin

# Episode sayısı
num_episodes = 1000  # Daha büyük problemler için artırın
```

### Variance Azaltma

1. **Baseline ekle**: `G_t - V(s_t)` kullan (Actor-Critic'e geçiş)
2. **Returns normalize et**: Mean/std ile normalize
3. **Advantage function**: `A(s,a) = Q(s,a) - V(s)`

### İleri Seviye

- [ ] Baseline ekle (value network)
- [ ] Entropy regularization (exploration için)
- [ ] Multiple workers (paralel örnekleme)
- [ ] Attention mechanism ekle

## 📖 Öğrenme Soruları

1. **Policy Gradient neden "gradient" diyor?**
   - Cevap: Policy parametrelerinin gradient'ini hesaplayıp, ödülü artıracak yönde güncelliyoruz.

2. **Neden log probability kullanıyoruz?**
   - Cevap: Matematiksel olarak türev almayı kolaylaştırır ve numerically stable.

3. **Variance problemi nedir?**
   - Cevap: Her episode farklı returns verir, bu da gradient'lerde büyük değişiklikler yaratır.

4. **Bu yöntem neden diğer ML yöntemlerinden farklı?**
   - Cevap: Supervised learning gibi doğru cevapları yok, agent kendi deneyimlerinden öğreniyor.

## 🔗 İleri Okuma

- **Sutton & Barto**: "Reinforcement Learning: An Introduction" (Kitap)
- **Karpathy Blog**: "Deep Reinforcement Learning: Pong from Pixels"
- **OpenAI Spinning Up**: RL dokümantasyonu
- **Williams 1992**: "Simple Statistical Gradient-Following Algorithms..." (REINFORCE paper)

---

**Sonraki Adım:** Actor-Critic mimarisine geçelim! 🚀
