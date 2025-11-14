# Proje Yapısı

```
DeepVRP/
├── esogu_deepvrp/              # Ana uygulama
│   ├── data_classes/           # Veri sınıfları
│   │   ├── __init__.py
│   │   ├── node.py            # Node, Depot, Customer
│   │   ├── vehicle.py         # Vehicle
│   │   └── problem.py         # VRPProblem
│   │
│   ├── util/                   # Yardımcı fonksiyonlar
│   │   ├── read_problem_instance.py   # XML okuma
│   │   ├── read_matrix_files.py       # Excel okuma
│   │   ├── data_preparation.py        # DL hazırlık
│   │   └── printer_utils.py
│   │
│   ├── main.py                # Ana program
│   ├── start_process.py       # Veri okuma koordinasyonu
│   ├── test_embeddings.py     # Embedding testleri
│   ├── demo_classes.py        # Sınıf örnekleri
│   └── DATA_GUIDE.md          # Kullanım kılavuzu
│
├── model/                      # Deep Learning modülleri
│   ├── __init__.py
│   ├── embeddings.py          # NodeEmbedding, GraphEmbedding
│   └── transforms.py          # Normalizasyon, graph conversion
│
└── dataset/                    # Veri dosyaları
    └── esogu/
        ├── problems/          # XML problem dosyaları
        └── matrix/            # Distance, Energy, Location matrices
```

## Veri Akışı

1. **Okuma** (`util/read_*.py`)
   - XML → dict
   - Excel → numpy arrays

2. **Nesne Oluşturma** (`data_classes/`)
   - dict → Depot, Customer objects
   - → VRPProblem instance

3. **DL Hazırlık** (`util/data_preparation.py`)
   - VRPProblem → normalized tensors
   - → PyTorch Geometric graphs

4. **Embedding** (`model/embeddings.py`)
   - Tensors → 128-dim embeddings
   - Node + Graph level

## Kullanım

```bash
cd esogu_deepvrp
python main.py          # Ana program
python demo_classes.py  # Sınıf demo
```
