# 🚀 Hızlı Başlangıç Rehberi

## 🎯 Hoş Geldiniz!

DeepVRP projenize hoş geldiniz! Bu rehber, projeyi kurmanız ve ilk derin öğrenme modelinizi eğitmeniz için adım adım yol gösterecek.

## 📋 Önkoşullar

✅ Python 3.11.9 yüklü (kontrol edildi)
✅ Temel Python bilgisi
✅ PyTorch öğrenme motivasyonu

## 🔧 Kurulum Adımları

### 1. Sanal Ortam Oluşturma

```powershell
# DeepVRP klasöründeyken
python -m venv venv
```

### 2. Sanal Ortamı Aktifleştirme

```powershell
.\venv\Scripts\Activate
```

**Not:** Eğer execution policy hatası alırsanız:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### 3. Gereksinimleri Yükleme

```powershell
pip install --upgrade pip
pip install torch torchvision numpy matplotlib networkx tqdm tensorboard pandas seaborn
```

**Alternatif:** requirements.txt'den yükleyin:
```powershell
pip install -r requirements.txt
```

## 🧪 İlk Testler

### Test 1: VRP Ortamı

```powershell
python 01_basics\vrp_environment.py
```

**Beklenen Çıktı:**
- Problem detayları
- Greedy nearest neighbor çözümü
- Toplam mesafe hesaplaması

### Test 2: Görselleştirme

```powershell
python 01_basics\visualizer.py
```

**Beklenen Çıktı:**
- `experiments/results/` klasöründe grafikler oluşturulması

### Test 3: Utilities

```powershell
python utils\metrics.py
python utils\helpers.py
```

## 🎓 İlk RL Modelini Eğitme

### Policy Gradient ile TSP Çözümü

```powershell
python 02_rl_methods\policy_gradient\simple_policy.py
```

**Bu script:**
- 1000 episode boyunca bir REINFORCE agent'ı eğitir
- 10 müşterili TSP problemleri çözer
- Training progress grafiklerini kaydeder
- En iyi turu görselleştirir

**Eğitim süresi:** ~5-10 dakika (CPU'da)

**Sonuçlar:**
- `experiments/models/policy_net_simple.pth` - Eğitilmiş model
- `experiments/results/training_progress.png` - Eğitim grafikleri
- `experiments/results/best_tour.png` - En iyi tur görselleştirmesi

## 📚 Öğrenme Yol Haritası

### 🔰 Seviye 1: Temel Kavramlar (Hafta 1-2)

**Okumanız Gerekenler:**
1. `readme.md` - Genel proje yapısı
2. `01_basics/README.md` - VRP ve RL temelleri
3. `02_rl_methods/policy_gradient/README.md` - Policy Gradient

**Yapmanız Gerekenler:**
- [ ] VRP ortamını anlayın (`vrp_environment.py`)
- [ ] Kendi problem instance'larınızı oluşturun
- [ ] Basit heuristic'lerle (greedy, random) çözümler üretin
- [ ] Görselleştirmeleri inceleyin

**Pratik Egzersizler:**
```python
# Egzersiz 1: Kendi probleminizi oluşturun
from basics.vrp_environment import VRPInstance, VRPEnvironment

instance = VRPInstance(
    depot=(50, 50),
    customers=[(20, 80), (80, 20), (30, 30), (70, 70)]
)

# Egzersiz 2: Farklı çözüm stratejileri deneyin
# - Random policy
# - Nearest neighbor
# - Farthest insertion

# Egzersiz 3: Sonuçları karşılaştırın
```

### ⚡ Seviye 2: Policy Gradient (Hafta 3-4)

**Okumanız Gerekenler:**
- REINFORCE algoritması teorisi
- Policy gradient matematiksel türetme
- Variance reduction teknikleri

**Yapmanız Gerekenler:**
- [ ] Simple policy modelini eğitin
- [ ] Hiperparametre deneyimleri yapın
- [ ] Farklı problem boyutlarını deneyin (5, 10, 20 müşteri)
- [ ] Training curves'leri analiz edin

**Sorular:**
1. Policy gradient neden "gradient" terimi kullanıyor?
2. Log probability neden kullanılıyor?
3. Baseline variance'ı nasıl azaltıyor?
4. Exploration vs exploitation trade-off'u nedir?

### 🚀 Seviye 3: İleri Seviye (Hafta 5+)

**Gelecek Modüller:**
- Actor-Critic (daha stabil öğrenme)
- Attention Mechanism (modern VRP çözümleri)
- Graph Neural Networks (graph representation)
- Pointer Networks (sequence-to-sequence)
- Transformer (state-of-the-art)

## 🔍 Proje Yapısı

```
DeepVRP/
├── 📄 readme.md              # Ana dokümantasyon
├── 📄 QUICKSTART.md          # Bu dosya
├── 📄 requirements.txt       # Python paketleri
│
├── 📁 01_basics/             # ⭐ BURADAN BAŞLAYIN
│   ├── vrp_environment.py    # VRP ortam tanımı
│   ├── visualizer.py         # Görselleştirme
│   └── README.md             # Detaylı açıklamalar
│
├── 📁 02_rl_methods/         # ⭐ İKİNCİ ADIM
│   └── policy_gradient/      # Basit RL
│       ├── simple_policy.py  # REINFORCE implementasyonu
│       └── README.md         # Teorik açıklamalar
│
├── 📁 03_graph_methods/      # 🔜 YAKINDA
│   ├── gcn/                  # Graph Convolutional Network
│   ├── gat/                  # Graph Attention Network
│   └── gnn_vrp/              # GNN ile VRP
│
├── 📁 04_sequence_methods/   # 🔜 YAKINDA
│   ├── seq2seq/              # Seq2Seq + Attention
│   ├── pointer_network/      # Pointer Network
│   └── transformer/          # Transformer
│
├── 📁 utils/                 # Yardımcı fonksiyonlar
│   ├── metrics.py            # Değerlendirme metrikleri
│   └── helpers.py            # Genel yardımcılar
│
└── 📁 experiments/           # Sonuçlar ve modeller
    ├── logs/                 # Training logları
    ├── models/               # Kaydedilen modeller
    └── results/              # Grafikler ve tablolar
```

## 💡 İpuçları ve Püf Noktaları

### Debugging

```python
# VRP ortamını adım adım inceleyin
env = VRPEnvironment(instance)
state = env.reset()

print("Initial state:", state)
print("Valid actions:", env.get_valid_actions())

# Her adımı izleyin
for i in range(5):
    action = env.get_valid_actions()[0]
    state, reward, done, info = env.step(action)
    print(f"Step {i}: action={action}, reward={reward:.3f}, done={done}")
```

### Performans İyileştirme

1. **GPU kullanın** (varsa):
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
```

2. **Batch processing**:
- Birden fazla problem instance'ını paralel çözün
- Eğitim süresini kısaltın

3. **Checkpoint kaydetme**:
```python
from utils.helpers import save_checkpoint

save_checkpoint(model, optimizer, epoch, 
                'experiments/models/checkpoint.pth')
```

### Sonuçları Analiz Etme

```python
from utils.metrics import evaluate_multiple_solutions

# Birden fazla çalıştırmanın istatistiklerini alın
stats = evaluate_multiple_solutions(coords, tours)
print(f"Mean: {stats['mean']:.3f}")
print(f"Std: {stats['std']:.3f}")
print(f"Best: {stats['min']:.3f}")
```

## 🐛 Sık Karşılaşılan Sorunlar

### Problem: ModuleNotFoundError

**Çözüm:**
```powershell
# Sanal ortamın aktif olduğundan emin olun
.\venv\Scripts\Activate

# Paketleri yeniden yükleyin
pip install -r requirements.txt
```

### Problem: "Import Error" alıyorum

**Çözüm:**
```python
# Script'leri proje kök dizininden çalıştırın
cd C:\Users\Yasin\Desktop\Kodlar\DeepVRP
python 01_basics\vrp_environment.py
```

### Problem: Eğitim çok yavaş

**Çözüm:**
- Episode sayısını azaltın (ilk testler için 100-200)
- Daha küçük problem boyutu kullanın (5-10 müşteri)
- GPU kullanın (varsa)

### Problem: Model öğrenmiyor

**Kontrol Edin:**
- Learning rate'i ayarlayın (1e-4 veya 1e-2 deneyin)
- Network boyutunu değiştirin (hidden_dim)
- Reward function'ın doğru çalıştığından emin olun

## 📞 Yardım ve Kaynaklar

### Önerilen Okumalar

1. **Kitaplar:**
   - "Reinforcement Learning: An Introduction" - Sutton & Barto
   - "Deep Learning" - Goodfellow, Bengio, Courville

2. **Online Kaynaklar:**
   - OpenAI Spinning Up (RL tutorial)
   - PyTorch Tutorials
   - Papers with Code (VRP başlığı)

3. **Önemli Makaleler:**
   - "Attention, Learn to Solve Routing Problems!" (Kool et al., 2019)
   - "Pointer Networks" (Vinyals et al., 2015)
   - "Neural Combinatorial Optimization" (Bello et al., 2016)

### Deney Fikirleri

1. **Farklı problem boyutları**:
   - 5, 10, 20, 50, 100 müşteri
   - Generalization yeteneğini test edin

2. **Transfer learning**:
   - Küçük problemde eğitin
   - Büyük problemde test edin

3. **Hibrit yaklaşımlar**:
   - RL + local search
   - RL ile initial solution, ALNS ile improve

## 🎯 Sonraki Adımlar

Şimdi şunları yapmalısınız:

1. ✅ Kurulumu tamamlayın
2. ✅ Temel testleri çalıştırın
3. ✅ İlk RL modelinizi eğitin
4. 📖 README dosyalarını okuyun
5. 💻 Kod üzerinde denemeler yapın
6. 📊 Sonuçlarınızı analiz edin
7. 🚀 İleri seviye modüllere geçin

---

**Başarılar! 🎓**

Sorularınız için kod üzerinde notlar alın ve kendi deneyimlerinizi dokümante edin. Bu proje sizin araştırma portföyünüzün temelini oluşturacak.
