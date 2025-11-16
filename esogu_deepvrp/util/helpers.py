"""
Helpers - Genel Yardımcı Fonksiyonlar

Bu modül, proje genelinde kullanılan yardımcı fonksiyonları içerir.

Autor: Yasin
Tarih: 28 Ekim 2025
"""

import torch
import numpy as np
import random
import os
from typing import Optional


def set_seed(seed: int = 42):
    """
    Tüm random seed'leri ayarla (reproducibility için).
    
    Args:
        seed: Random seed değeri
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """
    Kullanılabilir en iyi device'ı döndür (GPU varsa GPU).
    
    Returns:
        torch.device: 'cuda' veya 'cpu'
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return device


def save_checkpoint(model: torch.nn.Module, 
                   optimizer: torch.optim.Optimizer,
                   epoch: int,
                   filepath: str,
                   additional_info: Optional[dict] = None):
    """
    Model checkpoint'i kaydet.
    
    Args:
        model: PyTorch modeli
        optimizer: Optimizer
        epoch: Mevcut epoch
        filepath: Kayıt yolu
        additional_info: Ek bilgiler (loss, metrics, vb.)
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    
    if additional_info:
        checkpoint.update(additional_info)
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(checkpoint, filepath)
    print(f"💾 Checkpoint kaydedildi: {filepath}")


def load_checkpoint(model: torch.nn.Module,
                   optimizer: Optional[torch.optim.Optimizer],
                   filepath: str) -> dict:
    """
    Model checkpoint'i yükle.
    
    Args:
        model: PyTorch modeli
        optimizer: Optimizer (opsiyonel)
        filepath: Checkpoint yolu
    
    Returns:
        Checkpoint bilgileri
    """
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"📂 Checkpoint yüklendi: {filepath}")
    return checkpoint


def count_parameters(model: torch.nn.Module) -> int:
    """
    Model parametrelerinin sayısını hesapla.
    
    Args:
        model: PyTorch modeli
    
    Returns:
        Toplam parametre sayısı
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_summary(model: torch.nn.Module, model_name: str = "Model"):
    """
    Model özetini yazdır.
    
    Args:
        model: PyTorch modeli
        model_name: Model adı
    """
    print("=" * 60)
    print(f"📊 {model_name} Özeti")
    print("=" * 60)
    print(model)
    print("-" * 60)
    total_params = count_parameters(model)
    print(f"Toplam Parametre Sayısı: {total_params:,}")
    print("=" * 60)


class AverageMeter:
    """
    Ortalama hesaplama için yardımcı sınıf.
    
    Eğitim sırasında loss, metric gibi değerlerin hareketli ortalamasını
    tutmak için kullanılır.
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Tüm değerleri sıfırla"""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        """
        Yeni değer ekle.
        
        Args:
            val: Eklenecek değer
            n: Kaç örnekten elde edildi (batch size)
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def __str__(self):
        return f'{self.avg:.4f}'


if __name__ == "__main__":
    # Test
    print("=" * 60)
    print("Helpers Test")
    print("=" * 60)
    
    # Seed test
    print("\n🌱 Setting seed...")
    set_seed(42)
    print(f"Random: {random.random():.5f}")
    print(f"NumPy: {np.random.random():.5f}")
    
    # Device test
    print(f"\n💻 Device: {get_device()}")
    
    # AverageMeter test
    print("\n📊 AverageMeter test:")
    meter = AverageMeter()
    for i in range(5):
        meter.update(i * 2)
        print(f"  Update {i}: avg = {meter.avg:.2f}")
    
    print("\n" + "=" * 60)
