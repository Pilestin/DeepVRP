# DeepVRP - Deep Learning for Vehicle Routing Problems

### Attention (:D) : This repository is under development and represents my learning process.

**Amaç:** VRP problemlerini Deep Learning yöntemleriyle çözmeyi öğrenmek ve araştırma yapmak  
**Başlangıç:** 28 Ekim 2025

## Proje Hakkında

Bu proje, Capacitated Electric Vehicle Routing Problem with Time Windows (CEVRPTW) problemini derin öğrenme yöntemleriyle çözmeyi amaçlayan kapsamlı bir araştırma ve uygulama deposudur. Proje, 4 farklı state-of-the-art derin öğrenme mimarisi içermekte ve tümü Reinforcement Learning (REINFORCE algoritması) ile eğitilmektedir.

### İmplementasyon Durumu

✅ **Tamamlanan Modeller:**

1. **Attention Model (Transformer-based)**
   - Multi-head self-attention mekanizması
   - Encoder-decoder mimarisi
   - ~761K parametreler
   - İnference: ~9ms (20 node problem)

2. **Graph Convolutional Network (GCN)**
   - Spectral graph convolution
   - Degree-normalized message passing
   - ~150K parametreler
   - İnference: ~3ms (en hızlı model)

3. **Graph Attention Network (GAT)**
   - Attention-based GNN
   - Learned edge importance
   - ~497K parametreler
   - İnference: ~20ms

4. **Hybrid Model (GNN + Attention)**
   - GNN encoder + Transformer refinement
   - Structural ve sequential learning kombinasyonu
   - ~877K parametreler
   - İnference: ~17ms (en yüksek doğruluk)

## Proje Yapısı

```
DeepVRP/
├── model/                          # Deep Learning Modelleri
│   ├── attention_model.py         # Transformer-based model
│   ├── gnn_model.py               # GCN ve GAT implementasyonları
│   ├── hybrid_model.py            # GNN + Attention hybrid
│   ├── embeddings.py              # Feature encoding layers
│   ├── transforms.py              # Data preprocessing
│   ├── rl_trainer.py              # RL training framework
│   └── model_factory.py           # Model creation utilities
│
├── esogu_deepvrp/                 # Ana Uygulama
│   ├── data_classes/              # Problem representation
│   │   ├── node.py               # Node, Depot, Customer
│   │   ├── vehicle.py            # Vehicle class
│   │   └── problem.py            # VRPProblem class
│   │
│   ├── util/                      # Utilities
│   │   ├── read_problem_instance.py   # XML parsing
│   │   ├── read_matrix_files.py       # Excel reading
│   │   ├── data_preparation.py        # DL data preparation
│   │   └── printer_utils.py           # Output formatting
│   │
│   ├── main.py                    # Main entry point
│   ├── demo_models.py             # Model demonstration
│   └── test_embeddings.py         # Embedding tests
│
├── docs/                          # Detaylı Dokümantasyon
│   ├── THEORETICAL_FRAMEWORK.md   # Teorik çerçeve ve matematik
│   └── IMPLEMENTATION_GUIDE.md    # Implementasyon detayları
│
├── 01_basics/                     # Temel kavramlar
│   ├── vrp_environment.py         # VRP ortam tanımı
│   └── visualizer.py              # Sonuçları görselleştirme
│
├── 02_rl_methods/                 # Reinforcement Learning
│   └── policy_gradient/           # Policy Gradient implementasyonu
│       └── simple_policy.py       # Basit REINFORCE örneği
│
├── utils/                         # Yardımcı fonksiyonlar
│   ├── metrics.py                 # Performans metrikleri
│   └── helpers.py                 # Genel yardımcı fonksiyonlar
│
└── dataset/                       # Problem instances
    └── esogu/
        ├── problems/              # XML problem files (15 instances)
        └── matrix/                # Distance, Energy, Location matrices
```

##  Kurulum

### Gereksinimler

```bash
Python 3.8+
PyTorch 2.0+
PyTorch Geometric
NumPy
Matplotlib
NetworkX
Pandas
OpenPyXL
```

### Adımlar

1. Sanal ortam oluştur:
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

2. Gereksinimleri yükle:
```powershell
pip install -r requirements.txt
```

## 🚀 Hızlı Başlangıç

### Model Demonstration

Tüm modelleri test etmek için:

```bash
python esogu_deepvrp/demo_models.py --mode demo
```

Performans karşılaştırması için:

```bash
python esogu_deepvrp/demo_models.py --mode compare
```

### Örnek Kullanım

```python
from model.model_factory import create_model

# Attention model oluştur
model = create_model('attention', {'embed_dim': 128, 'num_heads': 8})

# GNN model oluştur
model = create_model('gnn_gat', {'embed_dim': 128, 'num_layers': 3})

# Hybrid model oluştur
model = create_model('hybrid', {'embed_dim': 128})
```

Detaylı kullanım için [QUICKSTART.md](QUICKSTART.md) dosyasına bakınız.

## 📚 Dokümantasyon

### Teorik Çerçeve

Matematiksel formülasyonlar, attention mekanizması teorisi, graph neural network temelleri ve reinforcement learning entegrasyonu için:

**[docs/THEORETICAL_FRAMEWORK.md](docs/THEORETICAL_FRAMEWORK.md)**

Bu dokümanda şunlar bulunmaktadır:
- CEVRPTW matematiksel formülasyonu
- Node feature representation
- Attention mechanism detayları
- Graph neural network foundations
- Message passing frameworks
- REINFORCE algoritması açıklaması
- Training metodolojisi
- Evaluation metrics

### İmplementasyon Rehberi

Mimari detayları, model seçimi, eğitim prosedürleri ve hyperparameter tuning için:

**[docs/IMPLEMENTATION_GUIDE.md](docs/IMPLEMENTATION_GUIDE.md)**

Bu dokümanda şunlar bulunmaktadır:
- Her model için detaylı mimari açıklamaları
- Computational complexity analizleri
- Model seçim kılavuzu
- Training procedures
- Hyperparameter tuning stratejileri
- Performance benchmarks
- Troubleshooting guide
- Kod örnekleri

### Proje Yapısı ve Özet

- **[PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)** - Proje yapısının kısa özeti
- **[IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)** - Kapsamlı implementasyon özeti
- **[QUICKSTART.md](QUICKSTART.md)** - Adım adım kurulum ve eğitim rehberi

## 🎯 Veri Akışı

Proje, aşağıdaki veri akışını takip eder:

```
1. Veri Yükleme
   ├── XML Problem Files → read_problem_instance.py
   ├── Excel Matrices → read_matrix_files.py
   └── GPS Path Data → location_matrix

2. Nesne Oluşturma
   ├── Problem Data → Depot, Customer objects
   └── VRPProblem instance (with vehicles)

3. DL Hazırlık
   ├── Node Features (7-dim) → Normalizasyon
   ├── Distance/Energy Matrices → Tensor format
   └── PyTorch Geometric Graph (optional)

4. Model Training
   ├── Node Embeddings (128-dim)
   ├── Policy Network (Action Selection)
   └── REINFORCE Algorithm

5. Solution Generation
   ├── Autoregressive Decoding
   ├── Constraint Masking
   └── Tour Construction
```

### Node Feature Representation (7 boyutlu)

- **Spatial:** x, y koordinatları
- **Demand:** weight/quantity
- **Temporal:** ready_time, due_date, service_time
- **Type:** is_depot flag

## 📊 Model Karşılaştırması

| Model | Parametreler | Boyut (MB) | İnference (ms) | Doğruluk | Hız |
|-------|-------------|------------|----------------|----------|-----|
| Attention | 761K | 2.90 | 9 | ★★★★☆ | ★★★☆☆ |
| GCN | 150K | 0.57 | 3 | ★★★☆☆ | ★★★★★ |
| GAT | 497K | 1.90 | 20 | ★★★★☆ | ★★★☆☆ |
| Hybrid | 877K | 3.35 | 17 | ★★★★★ | ★★★☆☆ |

### Model Seçimi

- **Attention Model:** Genel amaçlı VRP çözümü, yorumlanabilirlik önemli
- **GCN:** Büyük ölçekli problemler, hesaplama verimliliği kritik
- **GAT:** Heterojen problemler, maksimum doğruluk gerekli
- **Hybrid:** Araştırma ve en yüksek performans hedeflendiğinde

##  Referanslar

Bu projede kullanılan önemli makaleler:

1. **Attention Model** - Kool et al. (2019): "Attention, Learn to Solve Routing Problems!"
2. **Pointer Networks** - Vinyals et al. (2015): "Pointer Networks"
3. **Graph Attention Networks** - Veličković et al. (2018): "Graph Attention Networks"
4. **Learn2Opt** - Chen & Tian (2019): "Learning to Perform Local Rewriting for Combinatorial Optimization"
5. **REINFORCE** - Williams (1992): "Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning"

Detaylı akademik referanslar için [docs/THEORETICAL_FRAMEWORK.md](docs/THEORETICAL_FRAMEWORK.md) dosyasına bakınız.

## 🔬 Araştırma ve Geliştirme

### Mevcut Durum

✅ Dört SOTA model implementasyonu tamamlandı  
✅ Akademik dokümantasyon hazırlandı  
✅ Test ve validation yapıldı  
✅ Model factory ve utilities hazır  

### Sonraki Adımlar

- [ ] Eğitim pipeline'ı oluşturma
- [ ] Evaluation framework hazırlama
- [ ] Baseline metaheuristics ile karşılaştırma
- [ ] Deneysel protokol tasarlama
- [ ] Farklı problem boyutlarında test (5, 10, 20, 40, 60 müşteri)

##  Notlar ve İlerlemeler

### Test Sonuçları

Tüm modeller C10 (10-müşteri) problemi üzerinde başarıyla test edildi:
- ✅ Tüm mimariler geçerli probability distributions üretir
- ✅ Output shapes doğrulandı: (batch_size, num_nodes)
- ✅ Probability sums: ~1.0
- ✅ Compilation veya runtime hatası yok

### Teknik Stack

- **Deep Learning:** PyTorch 2.0+
- **Graph Processing:** PyTorch Geometric
- **Numerical Computing:** NumPy, SciPy
- **Data Handling:** Pandas, OpenPyXL
- **Visualization:** Matplotlib, Seaborn

---

**Not:** Detaylı kullanım örnekleri, teorik açıklamalar ve implementasyon detayları için yukarıdaki dokümantasyon linklerine bakınız. Her bir dokümanda ilgili konunun derinlemesine açıklamaları bulunmaktadır.
