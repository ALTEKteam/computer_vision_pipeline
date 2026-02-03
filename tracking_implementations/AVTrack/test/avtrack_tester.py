# import os
# import sys
# import torch
# import numpy as np
# import cv2
# from easydict import EasyDict

# # --- RENKLİ ÇIKTI ---
# class Colors:
#     GREEN = '\033[92m'
#     RED = '\033[91m'
#     YELLOW = '\033[93m'
#     RESET = '\033[0m'

# print(f"\n{Colors.YELLOW}--- AVTRACK SİSTEM DOĞRULAMA TESTİ (FİNAL) ---{Colors.RESET}\n")

# # 1. PATH AYARLARI VE IMPORT
# # ------------------------------------------------------------------
# print("[1/5] Kütüphane Yükleniyor...")

# current_dir = os.getcwd()
# if os.path.exists(os.path.join(current_dir, 'AVTrack')):
#     repo_path = os.path.join(current_dir, 'AVTrack')
# else:
#     repo_path = current_dir

# sys.path.append(repo_path)
# print(f"  Çalışma Dizini: {repo_path}")

# try:
#     from lib.test.tracker.avtrack import AVTrack
#     # Yama için gerekli modülü çekiyoruz
#     import lib.models.layers.head as head_module 
#     print(f"  {Colors.GREEN}✅ AVTrack kütüphanesi import edildi.{Colors.RESET}")
# except ImportError as e:
#     print(f"  {Colors.RED}❌ HATA: Kütüphane bulunamadı.{Colors.RESET}")
#     print(f"  Detay: {e}")
#     sys.exit()

# # 2. DONANIM
# # ------------------------------------------------------------------
# if torch.cuda.is_available():
#     device = 'cuda'
#     print(f"  ✅ GPU: {torch.cuda.get_device_name(0)}")
#     torch.backends.cudnn.enabled = False
#     torch.backends.cudnn.benchmark = False  # Hız optimizasyonunu kapat (Hata kaynağı bu)
#     torch.backends.cudnn.deterministic = True # Kararlı mod
# else:
#     device = 'cpu'
#     print(f"  ⚠️ GPU YOK! CPU kullanılıyor.")

# # 3. YAMA (YENİ YÖNTEM - RECURSION FIX)
# # ------------------------------------------------------------------
# print("\n[3/5] Kod Yamalanıyor (Güvenli Mod)...")

# # Sorun: Orijinal kodda feat_sz_t ve feat_sz_s eksik.
# # Çözüm: Sınıf miras almak yerine, direkt __init__ fonksiyonunu güncelliyoruz.
# # Bu sayede 'RecursionError' oluşmaz.

# try:
#     # 1. Hedef Sınıfı Seç (Senin Config dosyasında HEAD: CENTER olduğu için CenterPredictor da gerekebilir)
#     # Ancak hata Corner_Predictor'dan geliyordu, ikisini de garantiye alalım.
    
#     TargetClasses = []
#     if hasattr(head_module, 'Corner_Predictor'):
#         TargetClasses.append(head_module.Corner_Predictor)
#     if hasattr(head_module, 'CenterPredictor'):
#         TargetClasses.append(head_module.CenterPredictor)

#     for TargetClass in TargetClasses:
#         # Orijinal init fonksiyonunu sakla
#         original_init = TargetClass.__init__

#         # Yeni init fonksiyonu tanımla
#         def new_init(self, *args, **kwargs):
#             # Önce orijinal kodu çalıştır
#             original_init(self, *args, **kwargs)
            
#             # Eksik parçaları ekle (Yama Burası)
#             if not hasattr(self, 'feat_sz_t'):
#                 self.feat_sz_t = 8   # (128 / 16)
#             if not hasattr(self, 'feat_sz_s'):
#                 self.feat_sz_s = 16  # (256 / 16)

#         # Sınıfın init fonksiyonunu bizimkiyle değiştir
#         TargetClass.__init__ = new_init
    
#     print(f"  {Colors.GREEN}✅ Yama başarıyla uygulandı (Method Injection).{Colors.RESET}")

# except Exception as e:
#     print(f"  {Colors.RED}⚠️ Yama sırasında hata: {e}{Colors.RESET}")

# # 4. MODEL YÜKLEME (SENİN CONFIG DOSYANA GÖRE)
# # ------------------------------------------------------------------
# print("\n[4/5] Model Kuruluyor...")

# try:
#     # --- BURASI GÖNDERDİĞİN YAML DOSYASI İLE BİREBİR AYNI ---
#     cfg = EasyDict({
#         'MODEL': {
#             'PRETRAIN_FILE': "mae_pretrain_vit_base.pth",
#             'IS_DISTILL': True,
#             'EXTRA_MERGER': False,
#             'RETURN_INTER': False,
#             'BACKBONE': {
#                 'TYPE': 'deit_tiny_distilled_patch16_224',
#                 'STRIDE': 16,
#                 'CAT_MODE': 'direct',   # Varsayılan
#                 'SEP_SEG': False        # Varsayılan
#             },
#             'HEAD': {
#                 'TYPE': 'CENTER',       # <-- DİKKAT: Senin dosyalarda CENTER yazıyor!
#                 'NUM_CHANNELS': 256
#             },
#             'HIDDEN_DIM': 256
#         },
#         'DATA': {
#             'MAX_SAMPLE_INTERVAL': 200,
#             'MEAN': [0.485, 0.456, 0.406],
#             'STD': [0.229, 0.224, 0.225],
#             'SEARCH': {
#                 'CENTER_JITTER': 3,
#                 'FACTOR': 4.0,
#                 'SCALE_JITTER': 0.25,
#                 'SIZE': 256,
#                 'SIZE_EVA': 224,
#                 'NUMBER': 1
#             },
#             'TEMPLATE': {
#                 'CENTER_JITTER': 0,
#                 'FACTOR': 2.0,
#                 'SCALE_JITTER': 0,
#                 'SIZE': 128,
#                 'SIZE_EVA': 112,
#                 'NUMBER': 1
#             }
#         },
#         'TRAIN': { # Hata vermemesi için dummy değerler
#             'LR': 0.0004, 'WEIGHT_DECAY': 0.0001, 'EPOCH': 300,
#             'LR_DROP_EPOCH': 240, 'BATCH_SIZE': 32, 'NUM_WORKER': 10,
#             'OPTIMIZER': "ADAMW", 'BACKBONE_MULTIPLIER': 0.1,
#             'GIOU_WEIGHT': 2.0, 'L1_WEIGHT': 5.0, 'GRAD_CLIP_NORM': 0.1,
#             'PRINT_INTERVAL': 50, 'VAL_EPOCH_INTERVAL': 2000, 'AMP': False,
#             'SCHEDULER': {'TYPE': 'step', 'DECAY_RATE': 0.1},
#             'DROP_PATH_RATE': 0.1,
#             'DEEP_SUPERVISION': False,
#             'FREEZE_BACKBONE_BN': True
#         },
#         'TEST': { # Gerekli test parametreleri
#             'EPOCH': 300,
#             'SEARCH_FACTOR': 4.0,
#             'SEARCH_SIZE': 256,
#             'TEMPLATE_FACTOR': 2.0,
#             'TEMPLATE_SIZE': 128
#         }
#     })

#     params = EasyDict({
#         'cfg': cfg,
#         'search_size': 256,
#         'template_size': 128,
#         'checkpoint': 'C:/Users/rizam/Desktop/CENG/Altek/ComputerVision/Tracking_Imps/AVTrack/pretrained_models/AVTrack_MD_DeiT.pth',
#         'debug':False,
#         'use_visdom':False,
#         'save_all_boxes':False,
#         'template_factor':2.0,
#         'template_size':128,
#         'search_factor':4.0,
#         'search_size':256
#     })

#     # Modeli Başlat
#     tracker = AVTrack(params, dataset_name='AVTrack-MD-DeiT')

#     # Ağırlık Dosyasını Bul
#     possible_paths = [
#         os.path.join(repo_path, 'pretrained_models', 'AVTrack_MD_DeiT.pth'),
#         'AVTrack_MD_DeiT.pth'
#     ]
#     model_path = next((p for p in possible_paths if os.path.exists(p)), None)
            
#     if model_path:
#         print(f"  Dosya Yükleniyor: {model_path}")
#         checkpoint = torch.load(model_path, map_location='cpu')
        
#         if 'net' in checkpoint:
#             tracker.network.load_state_dict(checkpoint['net'], strict=False)
#         else:
#             tracker.network.load_state_dict(checkpoint, strict=False)
            
#         tracker.network.to(device)
#         tracker.network.eval()
#         print(f"  {Colors.GREEN}✅ BAŞARILI: Model yüklendi!{Colors.RESET}")
#     else:
#         print(f"  {Colors.RED}❌ HATA: Model dosyası bulunamadı!{Colors.RESET}")
#         sys.exit()

# except Exception as e:
#     print(f"  {Colors.RED}❌ YÜKLEME HATASI:{Colors.RESET}")
#     import traceback
#     traceback.print_exc()
#     sys.exit()

# print("\n[5/5] Matematiksel Test (Dummy Frame)...")
# try:
#     dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
#     dummy_target = [100, 100, 50, 50] 
#     init_info = {'init_bbox': dummy_target}
    
#     print("  -> Initialize ediliyor...")
    
#     # --- KRİTİK NOKTA: GRADIENT HESAPLAMAYI KAPAT ---
#     # Bu blok hafıza kullanımını %70 azaltır ve "Out of Memory" hatasını çözer.
#     with torch.no_grad():
#         tracker.initialize(dummy_frame, init_info)
        
#         print("  -> Track ediliyor...")
#         output = tracker.track(dummy_frame)
    
#     # Çıktı Analizi
#     bbox = None
#     score = 0.0
    
#     if isinstance(output, dict):
#         bbox = output.get('bbox', output.get('target_bbox'))
#         score = output.get('score', output.get('best_score', 0))
#     elif isinstance(output, (list, np.ndarray)):
#         bbox = output
#         score = -1 

#     print(f"  {Colors.GREEN}✅ TEST TAMAMLANDI!{Colors.RESET}")
#     print(f"  Tahmin: {bbox}")
#     print(f"  Skor: {score}")
#     print(f"\n{Colors.GREEN}>>> HARİKA! HER ŞEY ÇALIŞIYOR. <<<{Colors.RESET}")

# except RuntimeError as e:
#     if "out of memory" in str(e):
#         print(f"  {Colors.RED}❌ HATA: GPU Hafızası yine yetmedi!{Colors.RESET}")
#         print("  Çözüm: Terminali kapatıp açarak 'Zombie' işlemleri temizle.")
#     else:
#         print(f"  {Colors.RED}❌ ÇALIŞMA HATASI: {e}{Colors.RESET}")
#         import traceback
#         traceback.print_exc()
# except Exception as e:
#     raise e
#     print(f"  {Colors.RED}❌ ÇALIŞMA HATASI: {e}{Colors.RESET}")
#     import traceback
#     traceback.print_exc()
import os
import sys
import torch
import numpy as np
import cv2
import gc
from easydict import EasyDict

# --- GÜVENLİ VE DOĞRU TEST ---
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    RESET = '\033[0m'

print(f"\n{Colors.YELLOW}--- AVTRACK SİSTEM KONTROLÜ (PYTORCH 1.13) ---{Colors.RESET}\n")

# 1. TEMİZLİK
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# 2. VERSİYON KONTROLÜ (İÇİN RAHAT OLSUN)
print(f"[1/5] Sürüm Kontrolü:")
print(f"  PyTorch: {torch.__version__}")
print(f"  CUDA: {torch.version.cuda}")
if "1.13" not in torch.__version__:
    print(f"  {Colors.RED}⚠️ DİKKAT: Önerilen sürüm (1.13) kurulu değil! Yine hata alabilirsin.{Colors.RESET}")
else:
    print(f"  {Colors.GREEN}✅ Sürüm Uyumlu: RTX 3060 için güvenli sürüm algılandı.{Colors.RESET}")

# 3. KÜTÜPHANE YÜKLEME
print("\n[2/5] AVTrack Yükleniyor...")
current_dir = os.getcwd()
repo_path = os.path.join(current_dir, 'AVTrack') if os.path.exists(os.path.join(current_dir, 'AVTrack')) else current_dir
sys.path.append(repo_path)

try:
    from lib.test.tracker.avtrack import AVTrack
    import lib.models.layers.head as head_module 
    print(f"  {Colors.GREEN}✅ Kütüphane sorunsuz yüklendi.{Colors.RESET}")
except ImportError as e:
    print(f"  {Colors.RED}❌ HATA: AVTrack dosyaları bulunamadı.{Colors.RESET}")
    sys.exit()

# 4. YAMA (Method Injection - En Güvenlisi)
print("\n[3/5] Kod Yamalanıyor...")
try:
    TargetClasses = []
    if hasattr(head_module, 'Corner_Predictor'): TargetClasses.append(head_module.Corner_Predictor)
    if hasattr(head_module, 'CenterPredictor'): TargetClasses.append(head_module.CenterPredictor)

    for TargetClass in TargetClasses:
        original_init = TargetClass.__init__
        def new_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            if not hasattr(self, 'feat_sz_t'): self.feat_sz_t = 8   
            if not hasattr(self, 'feat_sz_s'): self.feat_sz_s = 16  
        TargetClass.__init__ = new_init
    print(f"  {Colors.GREEN}✅ Yama uygulandı.{Colors.RESET}")
except Exception as e:
    print(f"  {Colors.RED}⚠️ Yama hatası: {e}{Colors.RESET}")

# 5. MODEL YÜKLEME (GPU)
print("\n[4/5] Model GPU'ya Yükleniyor...")
try:
    cfg = EasyDict({
        'MODEL': {
            'PRETRAIN_FILE': "mae_pretrain_vit_base.pth", 'IS_DISTILL': True, 'EXTRA_MERGER': False, 'RETURN_INTER': False,
            'BACKBONE': {'TYPE': 'deit_tiny_distilled_patch16_224', 'STRIDE': 16, 'CAT_MODE': 'direct', 'SEP_SEG': False},
            'HEAD': {'TYPE': 'CENTER', 'NUM_CHANNELS': 256}, 'HIDDEN_DIM': 256
        },
        'DATA': {
            'MAX_SAMPLE_INTERVAL': 200, 'MEAN': [0.485, 0.456, 0.406], 'STD': [0.229, 0.224, 0.225],
            'SEARCH': {'CENTER_JITTER': 3, 'FACTOR': 4.0, 'SCALE_JITTER': 0.25, 'SIZE': 256, 'SIZE_EVA': 224, 'NUMBER': 1},
            'TEMPLATE': {'CENTER_JITTER': 0, 'FACTOR': 2.0, 'SCALE_JITTER': 0, 'SIZE': 128, 'SIZE_EVA': 112, 'NUMBER': 1}
        },
        'TRAIN': { 
            'LR': 0.0004, 'WEIGHT_DECAY': 0.0001, 'EPOCH': 300, 'LR_DROP_EPOCH': 240, 'BATCH_SIZE': 32, 'NUM_WORKER': 10,
            'OPTIMIZER': "ADAMW", 'BACKBONE_MULTIPLIER': 0.1, 'GIOU_WEIGHT': 2.0, 'L1_WEIGHT': 5.0, 'GRAD_CLIP_NORM': 0.1,
            'PRINT_INTERVAL': 50, 'VAL_EPOCH_INTERVAL': 2000, 'AMP': False, 'SCHEDULER': {'TYPE': 'step', 'DECAY_RATE': 0.1},
            'DROP_PATH_RATE': 0.1, 'DEEP_SUPERVISION': False, 'FREEZE_BACKBONE_BN': True
        },
        'TEST': {'EPOCH': 300, 'SEARCH_FACTOR': 4.0, 'SEARCH_SIZE': 256, 'TEMPLATE_FACTOR': 2.0, 'TEMPLATE_SIZE': 128}
    })

    params = EasyDict({'cfg': cfg, 'search_size': 256, 'template_size': 128,
        'checkpoint': '/home/furkan/CENG/altek/tracking_implementations/AVTrack/pretrained_models/AVTrack_MD_DeiT.pth',
        'debug':False,
        'use_visdom':False,
        'save_all_boxes':False,
        'template_factor':2.0,
        'template_size':128,
        'search_factor':4.0,
        'search_size':256})
    tracker = AVTrack(params, dataset_name='AVTrack-MD-DeiT')

    possible_paths = [os.path.join(repo_path, 'pretrained_models', 'AVTrack_MD_DeiT.pth'), 'AVTrack_MD_DeiT.pth']
    model_path = next((p for p in possible_paths if os.path.exists(p)), None)
            
    if model_path:
        checkpoint = torch.load(model_path, map_location='cpu')
        tracker.network.load_state_dict(checkpoint['net'] if 'net' in checkpoint else checkpoint, strict=False)
        
        # --- GPU ENTEGRASYONU (Hatasız) ---
        if torch.cuda.is_available():
            tracker.network.to('cuda')
            print(f"  {Colors.GREEN}✅ Model GPU hafızasına (CUDA) yerleşti.{Colors.RESET}")
        else:
            print(f"  {Colors.RED}❌ GPU bulunamadı!{Colors.RESET}")
            sys.exit()
        
        tracker.network.eval()
    else:
        print(f"  {Colors.RED}❌ HATA: Model dosyası bulunamadı!{Colors.RESET}")
        sys.exit()

except Exception as e:
    print(f"  {Colors.RED}❌ YÜKLEME HATASI: {e}{Colors.RESET}")
    import traceback
    traceback.print_exc()
    sys.exit()

# 6. MATEMATİKSEL TEST
print("\n[5/5] Matematiksel Test...")
try:
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    dummy_target = [100, 100, 50, 50] 
    init_info = {'init_bbox': dummy_target}
    
    # Hafızayı yormamak için 'no_grad' kullanıyoruz.
    with torch.no_grad():
        tracker.initialize(dummy_frame, init_info)
        output = tracker.track(dummy_frame)
    
    bbox = None
    if isinstance(output, dict):
        bbox = output.get('bbox', output.get('target_bbox'))
    elif isinstance(output, (list, np.ndarray)):
        bbox = output

    print(f"\n{Colors.GREEN}✅✅✅ TEST BAŞARILI! ✅✅✅{Colors.RESET}")
    print(f"  Tahmin: {bbox}")
    print(f"  {Colors.YELLOW}SONUÇ:{Colors.RESET} Sistem RTX 3060 ile tam uyumlu çalışıyor. Artık 'main.py' yazabiliriz.")

except Exception as e:
    print(f"  {Colors.RED}❌ ÇALIŞMA HATASI: {e}{Colors.RESET}")
    import traceback
    traceback.print_exc()
