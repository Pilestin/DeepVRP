# DeepVRP - Data Structures & Embeddings

## 📦 Veri Sınıfları (`data_classes/`)

### Node Hiyerarşisi
- **`Node`**: Base class (koordinatlar, GPS)
- **`Depot`**: Depo/şarj istasyonu
- **`Customer`**: Müşteri noktası (talep, zaman penceresi)

### Araç
- **`Vehicle`**: Elektrikli araç (kapasite, batarya)

### Problem
- **`VRPProblem`**: Tüm problem verilerini tutan ana sınıf

## 🧠 Deep Learning Modülleri (`model/`)

### Embeddings
- **`NodeEmbedding`**: Node features → 128-dim embeddings
- **`GraphEmbedding`**: Graph-level representation

### Transforms
- `normalize_features()`: MinMax / Standard normalization
- `to_graph_data()`: PyTorch Geometric conversion
- `create_attention_mask()`: Transformer masks

## 🚀 Kullanım

```python
# 1. Veri oku
problem_data, distance_matrix, energy_matrix, location_matrix = start_process(...)

# 2. VRPProblem oluştur
vrp_problem = create_problem_from_raw_data(problem_data, distance_matrix, ...)

# 3. DL için hazırla
dl_data = prepare_for_deep_learning(vrp_problem, normalize=True, create_graph=True)

# 4. Model kullan
embedder = GraphEmbedding()
node_emb, graph_emb = embedder(dl_data['node_features'], ...)
```

## 📊 Veri Formatı

**Node Features** (7 boyut):
```
[x, y, demand, ready_time, due_date, service_time, is_depot]
```

**Matrisler**:
- Distance: (N×N) mesafe
- Energy: (N×N) enerji tüketimi

**Graph**: PyTorch Geometric Data object
