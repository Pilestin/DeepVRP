import numpy as np
import matplotlib.pyplot as plt

class NeuralVRPSolver:
    def __init__(self, cities, learning_rate=0.8, num_neurons_factor=2.5):
        """
        Modeli ve parametreleri başlatır.
        """
        self.cities = cities
        self.num_cities = len(cities)
        # Nöron sayısı şehir sayısından biraz fazla olmalı ki esnek olsun
        self.num_neurons = int(self.num_cities * num_neurons_factor)
        
        # Nöronları (Ağırlıkları) başlat: Rastgele bir çember şeklinde
        # Bu bizim 'Modelimiz'dir.
        self.neurons = np.random.rand(self.num_neurons, 2)
        
        self.learning_rate = learning_rate
        self.loss_history = [] # Mesafeyi takip etmek için

    def normalize_data(self):
        """Şehir koordinatlarını işlemlerin sağlıklı olması için normalize eder."""
        pass # Bu örnekte veriyi zaten 0-1 arasında üreteceğiz.

    def train(self, epochs=1000):
        """
        Modelin eğitimi (Training Loop).
        Her epoch'ta model şehirlere biraz daha yaklaşır.
        """
        print(f"Eğitim başlıyor: {self.num_cities} şehir, {epochs} iterasyon...")
        
        for epoch in range(epochs):
            # Rastgele bir şehir seç (Batch size = 1 gibi düşünebilirsin)
            city_idx = np.random.randint(0, self.num_cities)
            city = self.cities[city_idx]
            
            # 1. En yakın nöronu (Winner Neuron) bul
            # Bu işlem 'Forward Pass' gibidir.
            dists = np.linalg.norm(self.neurons - city, axis=1)
            winner_idx = np.argmin(dists)
            
            # 2. Neighborhood (Komşuluk) fonksiyonu
            # Kazanan nöronu ve komşularını şehre doğru çek (Update Weights)
            # Gaussian fonksiyonu kullanılarak etki alanı belirlenir
            radius = self.num_neurons * np.exp(-epoch / (epochs / 10)) # Zamanla yarıçap azalır
            learning_rate = self.learning_rate * np.exp(-epoch / epochs) # Zamanla öğrenme hızı düşer
            
            # Tüm nöronların kazanana olan uzaklığı (index bazında, ring yapısı)
            neuron_indices = np.arange(self.num_neurons)
            dist_from_winner = np.abs(neuron_indices - winner_idx)
            # Ring (Halka) yapısı olduğu için diğer taraftan yakınlık hesabı
            dist_from_winner = np.minimum(dist_from_winner, self.num_neurons - dist_from_winner)
            
            # Gaussian etkisi (Neighborhood function)
            influence = np.exp(-(dist_from_winner**2) / (2 * (radius**2)))
            
            # 3. Ağırlıkları güncelle (Backpropagation mantığı)
            # Nöronları şehre doğru hareket ettir
            # new_pos = old_pos + (city - old_pos) * learning_rate * influence
            self.neurons += (city - self.neurons) * learning_rate * influence[:, np.newaxis]

        print("Eğitim tamamlandı.")

    def get_route(self):
        """
        Eğitilmiş modelden rotayı çıkarır (Inference).
        Her şehir için ona en yakın nöronu bulur ve sıraya dizersiniz.
        """
        winner_neurons = []
        for city in self.cities:
            dists = np.linalg.norm(self.neurons - city, axis=1)
            winner_idx = np.argmin(dists)
            winner_neurons.append(winner_idx)
        
        # Şehirleri nöron sırasına göre diz (argsort)
        route_indices = np.argsort(winner_neurons)
        return self.cities[route_indices]

    def plot_route(self, title="Sonuç"):
        """
        Rotayı ve nöronları görselleştirir.
        """
        ordered_cities = self.get_route()
        # Döngüyü kapatmak için başa dön
        ordered_cities = np.vstack([ordered_cities, ordered_cities[0]])
        
        plt.figure(figsize=(10, 6))
        
        # Şehirler
        plt.scatter(self.cities[:, 0], self.cities[:, 1], c='red', s=50, label='Müşteriler/Şehirler')
        
        # Modelin (Nöronların) son hali
        # plt.plot(self.neurons[:, 0], self.neurons[:, 1], 'g.', alpha=0.3, label='Nöronlar (Model)')
        
        # Çizilen Rota
        plt.plot(ordered_cities[:, 0], ordered_cities[:, 1], c='blue', linewidth=2, label='Optimize Edilen Rota')
        
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.show()

# --- ANA PROGRAM ---

# 1. Veri Seti Oluşturma (Data Generation)
num_cities = 50
# [0, 1] aralığında rastgele koordinatlar
dataset = np.random.rand(num_cities, 2)

# 2. Modeli Başlatma
# Girdi olarak veri setimizi veriyoruz
model = NeuralVRPSolver(dataset, learning_rate=0.8)

# Görselleştirme: Eğitimden Önce (Model henüz hiçbir şey bilmiyor)
# Başlangıçta nöronlar rastgele olduğu için rota karışıktır.
# model.plot_route("Eğitimden Önceki Rastgele Rota")

# 3. Eğitimi Başlat (Train)
# Epoch sayısı arttıkça rota daha pürüzsüz olur
model.train(epochs=2000) 

# 4. Sonucu Test Et ve Görselleştir (Test/Inference)
model.plot_route(f"Derin Öğrenme (SOM) ile Çözülen VRP - {num_cities} Nokta")