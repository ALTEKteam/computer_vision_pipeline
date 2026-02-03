import torch

print("--- GPU DOĞRULAMA TESTİ ---")
try:
    # 1. GPU var mı?
    is_available = torch.cuda.is_available()
    print(f"CUDA Erişimi: {is_available}")
    
    if is_available:
        # 2. Cihaz Adı
        print(f"Ekran Kartı: {torch.cuda.get_device_name(0)}")
        
        # 3. Basit Bir Tensör İşlemi (Çöküyor mu?)
        x = torch.rand(5, 3).cuda()
        y = torch.rand(5, 3).cuda()
        z = x + y
        print(f"Hesaplama Testi: BAŞARILI ✅ (Sonuç boyutu: {z.shape})")
        print("Sürüm Uyumu: MÜKEMMEL.")
    else:
        print("HATA: GPU hala görünmüyor.")
        
except Exception as e:
    print(f"HATA: {e}")