# VRP Çözüm Sistemi - Kullanım Kılavuzu

## Sistem Özellikleri

### Araç Parametreleri
- **Kapasite:** 350 kg
- **Batarya Kapasitesi:** 15,600 kWh
- **Hız:** 12.5 m/s (45 km/h)

### Kısıtlar
1. **Kapasite Kısıtı:** Araç kapasitesi aşılmamalı
2. **Batarya Kısıtı:** Müşteriye gidip depoya dönecek kadar batarya olmalı
3. **Zaman Penceresi Kısıtı:** 
   - Müşteriye varış zamanı `ready_time` ile `due_date` arasında olmalı
   - Erken varılırsa bekleme yapılır
   - Geç varış durumunda müşteri ziyaret edilmez

### Hesaplamalar

#### 1. Seyahat Süresi (Travel Time)
```
travel_time (saniye) = distance (metre) / 12.5 (m/s)
```

#### 2. Toplam Süre (Total Time)
Her müşteri ziyaretinde:
```
total_time = travel_time + wait_time + service_time
```

Nerede:
- `travel_time`: Seyahat süresi
- `wait_time`: Bekleme süresi (eğer ready_time'dan önce varılırsa)
- `service_time`: Müşteriye hizmet verme süresi

#### 3. Enerji Tüketimi
Enerji matrisi kullanılarak her segment için enerji tüketimi hesaplanır.

## Çıktı Formatı

### Rota Formatı (Tırnaksız)
```
Routes = [[cs5, 31, 14, cs5], [cs5, 45, 22A, cs5], [cs5, 119, 113, 13, cs5]]
```

### Detaylı Sonuçlar
Her rota için:
- Mesafe (metre)
- Süre (saniye, dakika)
- Enerji (kWh)

Toplam özet:
- Toplam Mesafe
- Toplam Süre (saniye, dakika, saat)
- Toplam Enerji

## Kullanım

### Basit Test
```bash
cd esogu_deepvrp
python test_vrp.py
```

### Manuel Çalıştırma
```bash
cd esogu_deepvrp
python main.py
```
Problem setini seçin (1-15 arası):
- 1: C05 (5 müşteri, clustered)
- 2: C10 (10 müşteri, clustered)
- 3: C20 (20 müşteri, clustered)
- vb.

## Gelecek Güncellemeler (Test İçin)

### Tank Capacity Testi
Daha sonra test için:
- **Tank Capacity:** 3500 metre
- **Energy Consumption:** 1 kWh/metre

Bu değişiklik için `create_problem_from_raw_data` fonksiyonunda:
```python
vrp_problem = create_problem_from_raw_data(
    problem_data=problem_data,
    distance_matrix=distance_matrix,
    energy_matrix=energy_matrix,
    location_paths=location_matrix,
    num_vehicles=5,
    vehicle_capacity=350.0,
    battery_capacity=3500.0  # 3500m için test
)
```

Ve `greedy_solver.py`'de enerji hesaplaması:
```python
# Normal: energy = energy_matrix değeri
# Test için: energy = distance * 1.0  # 1 kWh per metre
```

## Örnek Çıktı

```
======================================================================
VRP SOLUTION RESULTS
======================================================================

Number of vehicles used: 5

Routes:
----------------------------------------------------------------------

Vehicle 1:
  Route: cs5 → 31 → 14 → cs5
  Distance: 3016.77 m
  Time: 1832.90 s (30.55 min)
  Energy: 318.52 kWh

Vehicle 2:
  Route: cs5 → 45 → 22A → cs5
  Distance: 1801.24 m
  Time: 1783.66 s (29.73 min)
  Energy: 168.17 kWh

----------------------------------------------------------------------
TOTAL SUMMARY
----------------------------------------------------------------------
Total Distance: 11900.63 m
Total Time: 6420.36 s (107.01 min, 1.78 hours)
Total Energy: 1256.05 kWh
======================================================================

--- Routes in Requested Format ---
Routes = [[cs5, 31, 14, cs5], [cs5, 45, 22A, cs5], [cs5, 119, 113, 13, cs5], [cs5, 34, 24, cs5], [cs5, 26, cs5]]
```

## Kontroller

✅ Araç kapasitesi: 350 kg  
✅ Batarya kapasitesi: 15,600 kWh  
✅ Hız: 12.5 m/s  
✅ Zaman pencereleri kontrol ediliyor  
✅ Service time hesaplamada  
✅ Rota formatı tırnaksız  
✅ Mesafe, zaman, enerji raporlanıyor  
