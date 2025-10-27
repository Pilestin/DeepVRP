# DeepVRP Başlangıç Scripti
# Bu script, projeyi kurup ilk temel testleri çalıştırır

Write-Host "================================" -ForegroundColor Cyan
Write-Host "🚗 DeepVRP Projesine Hoş Geldiniz" -ForegroundColor Cyan
Write-Host "================================" -ForegroundColor Cyan
Write-Host ""

# Gerekli klasörleri oluştur
Write-Host "📁 Klasör yapısı oluşturuluyor..." -ForegroundColor Yellow
$folders = @(
    "experiments/logs",
    "experiments/models", 
    "experiments/results",
    "docs/theory",
    "docs/tutorials"
)

foreach ($folder in $folders) {
    if (!(Test-Path $folder)) {
        New-Item -ItemType Directory -Path $folder -Force | Out-Null
        Write-Host "  ✓ $folder oluşturuldu" -ForegroundColor Green
    }
}

Write-Host ""
Write-Host "================================" -ForegroundColor Cyan
Write-Host "🔧 Kurulum Talimatları" -ForegroundColor Cyan
Write-Host "================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "1️⃣ Sanal ortam oluşturun:" -ForegroundColor White
Write-Host "   python -m venv venv" -ForegroundColor Gray
Write-Host ""

Write-Host "2️⃣ Sanal ortamı aktifleştirin:" -ForegroundColor White
Write-Host "   .\venv\Scripts\Activate.ps1" -ForegroundColor Gray
Write-Host ""

Write-Host "3️⃣ Gereksinimleri yükleyin:" -ForegroundColor White
Write-Host "   pip install -r requirements.txt" -ForegroundColor Gray
Write-Host ""

Write-Host "4️⃣ Temel testleri çalıştırın:" -ForegroundColor White
Write-Host "   python 01_basics\vrp_environment.py" -ForegroundColor Gray
Write-Host ""

Write-Host "5️⃣ İlk RL modelini eğitin:" -ForegroundColor White
Write-Host "   python 02_rl_methods\policy_gradient\simple_policy.py" -ForegroundColor Gray
Write-Host ""

Write-Host "================================" -ForegroundColor Cyan
Write-Host "📚 Öğrenme Yol Haritası" -ForegroundColor Cyan
Write-Host "================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Hafta 1-2: Temel Kavramlar" -ForegroundColor Yellow
Write-Host "  → 01_basics/README.md dosyasını okuyun" -ForegroundColor Gray
Write-Host "  → VRP ortamını anlayın ve test edin" -ForegroundColor Gray
Write-Host ""
Write-Host "Hafta 3-4: Policy Gradient" -ForegroundColor Yellow
Write-Host "  → 02_rl_methods/policy_gradient/README.md" -ForegroundColor Gray
Write-Host "  → REINFORCE algoritmasını çalıştırın" -ForegroundColor Gray
Write-Host ""
Write-Host "Hafta 5+: İleri Seviye" -ForegroundColor Yellow
Write-Host "  → Actor-Critic, Attention, GNN modelleri" -ForegroundColor Gray
Write-Host ""

Write-Host "================================" -ForegroundColor Cyan
Write-Host "✅ Başlangıç scripti tamamlandı!" -ForegroundColor Green
Write-Host "================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "💡 İpucu: readme.md dosyasını inceleyerek devam edin." -ForegroundColor Cyan
Write-Host ""
