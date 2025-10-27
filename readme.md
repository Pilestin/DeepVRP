# 🚗 DeepVRP - Deep Learning for Vehicle Routing Problems

**Araştırmacı:** Yasin  
**Amaç:** VRP problemlerini Deep Learning yöntemleriyle çözmeyi öğrenmek ve araştırma yapmak  
**Başlangıç:** 28 Ekim 2025

## 📖 Proje Hakkında

Bu proje, Vehicle Routing Problem (VRP) ve türevlerini (TSP, CVRP, VRPTW, vb.) derin öğrenme yöntemleriyle çözmeyi öğrenmek için hazırlanmış kapsamlı bir eğitim ve araştırma deposudur.

### 🎯 Kapsanan Yöntemler

1. **Reinforcement Learning**
   - Policy Gradient
   - Actor-Critic
   - REINFORCE algoritması

2. **Graph Neural Networks**
   - Graph Convolutional Networks (GCN)
   - Graph Attention Networks (GAT)
   - Message Passing Neural Networks

3. **Sequence Models**
   - Seq2Seq with Attention
   - Pointer Networks
   - Transformer Architecture

4. **Attention Mechanisms**
   - Self-Attention
   - Multi-Head Attention
   - Cross-Attention

## 📁 Proje Yapısı

```
DeepVRP/
├── 01_basics/              # Temel kavramlar ve ortam kurulumu
│   ├── vrp_environment.py  # VRP ortam tanımı
│   ├── data_generator.py   # Problem instance'ları üretme
│   └── visualizer.py       # Sonuçları görselleştirme
│
├── 02_rl_methods/          # Reinforcement Learning yöntemleri
│   ├── policy_gradient/    # Policy Gradient implementasyonu
│   ├── actor_critic/       # Actor-Critic implementasyonu
│   └── attention_rl/       # RL + Attention mekanizması
│
├── 03_graph_methods/       # Graph-based yöntemler
│   ├── gcn/               # Graph Convolutional Network
│   ├── gat/               # Graph Attention Network
│   └── gnn_vrp/           # GNN ile VRP çözümü
│
├── 04_sequence_methods/    # Sequence-based yöntemler
│   ├── seq2seq/           # Seq2Seq + Attention
│   ├── pointer_network/   # Pointer Network
│   └── transformer/       # Transformer mimarisi
│
├── 05_hybrid_methods/      # Hibrit yaklaşımlar
│   └── attention_model/   # Attention Model (Kool et al., 2019)
│
├── experiments/           # Deneyler ve sonuçlar
│   ├── logs/             # Training logları
│   ├── models/           # Kaydedilen modeller
│   └── results/          # Sonuç grafikleri ve tablolar
│
├── docs/                 # Detaylı dokümantasyon
│   ├── theory/          # Teorik açıklamalar
│   └── tutorials/       # Adım adım öğreticiler
│
└── utils/               # Yardımcı fonksiyonlar
    ├── metrics.py       # Performans metrikleri
    └── helpers.py       # Genel yardımcı fonksiyonlar
```

## 🛠️ Kurulum

### Gereksinimler

```bash
Python 3.8+
PyTorch 2.0+
NumPy
Matplotlib
NetworkX
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

## 📚 Öğrenme Sırası (Önerilen)

### Hafta 1-2: Temel Kavramlar
- [ ] VRP ortamı ve problem tanımı
- [ ] Basit Policy Gradient ile TSP
- [ ] Görselleştirme ve değerlendirme

### Hafta 3-4: Gelişmiş RL
- [ ] Actor-Critic mimarisi
- [ ] Attention mekanizması entegrasyonu
- [ ] CVRP'ye genişletme

### Hafta 5-6: Graph Methods
- [ ] GCN temelleri
- [ ] GAT ve attention graphs
- [ ] VRP'de graph representation

### Hafta 7-8: Sequence Models
- [ ] Seq2Seq + Attention
- [ ] Pointer Networks
- [ ] Transformer mimarisi

### Hafta 9+: İleri Seviye
- [ ] Hibrit modeller
- [ ] VRPTW, EVRP gibi varyantlar
- [ ] Kendi araştırma fikirlerinizi geliştirme

## 📊 Referanslar

Bu projede kullanılan önemli makaleler:

1. **Attention Model** - Kool et al. (2019): "Attention, Learn to Solve Routing Problems!"
2. **Pointer Networks** - Vinyals et al. (2015): "Pointer Networks"
3. **Graph Attention Networks** - Veličković et al. (2018): "Graph Attention Networks"
4. **Learn2Opt** - Chen & Tian (2019): "Learning to Perform Local Rewriting for Combinatorial Optimization"

## 📝 Notlar ve İlerlemeler

### Günlük Notlar
Bu bölümü kendi öğrenme notlarınızla doldurun.

---

**Not:** Her modül kendi README.md dosyasını içerir ve detaylı açıklamalar sunar.
