import os
import sys
import torch
import numpy as np
import cv2
import gc
from easydict import EasyDict

# --- RENKLİ ÇIKTI ---
class Colors:
    GREEN = '\033[92m'
    CYAN = '\033[96m'
    YELLOW = '\033[93m'
    RESET = '\033[0m'

print(f"\n{Colors.CYAN}=== AVTRACK KAMERA SİSTEMİ BAŞLATILIYOR ==={Colors.RESET}")

# 1. ORTAM KURULUMU (TESTER'DAN ALINDI)
# ------------------------------------------------------------------
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    device = 'cuda'
else:
    print("HATA: GPU yok! Çok yavaş çalışır.")
    sys.exit()

current_dir = os.getcwd()
if os.path.exists(os.path.join("/home/furkan/Desktop/CS/altek/tracking_implementations/", 'ORTrack')):
    repo_path = os.path.join("/home/furkan/Desktop/CS/altek/tracking_implementations/", 'ORTrack')
else:
    repo_path = current_dir
print("Repo yolu:", repo_path)
sys.path.append(repo_path)

try:
    from lib.test.tracker.ortrack import ORTrack
    import lib.config.ortrack.config as config_module
    import lib.models.layers.head as head_module 
except ImportError as er:
    print("Kütüphane bulunamadı.",er)
    sys.exit()

# 2. YAMA (KOD DÜZELTME)
# ------------------------------------------------------------------
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
except Exception:
    pass

# 3. MODELİ YÜKLE
# ------------------------------------------------------------------
print(f"{Colors.YELLOW}[Sistem]{Colors.RESET} Model GPU'ya yükleniyor...")

cfg = EasyDict({
        'MODEL': {
            'PRETRAIN_FILE': "mae_pretrain_vit_base.pth", 
            'IS_DISTILL': True, 
            'EXTRA_MERGER': False, 
            'RETURN_INTER': False,
            'BACKBONE': {
                'TYPE': 'deit_tiny_distilled_patch16_224', # Tiny model kullanıyorsan burası doğru
                'STRIDE': 16, 
                'CAT_MODE': 'direct', 
                'SEP_SEG': False,
                'CE_LOC': [],
                'CE_KEEP_RATIO': [],
                'CE_TEMPLATE_RANGE': 'ALL'
            },
            'HEAD': {'TYPE': 'CENTER', 'NUM_CHANNELS': 256}, 
            'HIDDEN_DIM': 256
        },
        'DATA': {
            'MAX_SAMPLE_INTERVAL': 200, 
            'MEAN': [0.485, 0.456, 0.406], 
            'STD': [0.229, 0.224, 0.225],
            'SEARCH': {'CENTER_JITTER': 3, 'FACTOR': 4.0, 'SCALE_JITTER': 0.25, 'SIZE': 256, 'SIZE_EVA': 224, 'NUMBER': 1},
            'TEMPLATE': {'CENTER_JITTER': 0, 'FACTOR': 2.0, 'SCALE_JITTER': 0, 'SIZE': 128, 'SIZE_EVA': 112, 'NUMBER': 1}
        },
        'TRAIN': { 
            'LR': 0.0004, 'WEIGHT_DECAY': 0.0001, 'EPOCH': 300, 'LR_DROP_EPOCH': 240, 'BATCH_SIZE': 32, 'NUM_WORKER': 10,
            'OPTIMIZER': "ADAMW", 'BACKBONE_MULTIPLIER': 0.1, 'GIOU_WEIGHT': 2.0, 'L1_WEIGHT': 5.0, 'GRAD_CLIP_NORM': 0.1,
            'PRINT_INTERVAL': 50, 'VAL_EPOCH_INTERVAL': 2000, 'AMP': False, 'SCHEDULER': {'TYPE': 'step', 'DECAY_RATE': 0.1},
            'DROP_PATH_RATE': 0.1, 'DEEP_SUPERVISION': False, 'FREEZE_BACKBONE_BN': True
        },
        'TEST': {
            'EPOCH': 300, 
            'SEARCH_FACTOR': 4.0, 
            'SEARCH_SIZE': 256, 
            'TEMPLATE_FACTOR': 2.0, 
            'TEMPLATE_SIZE': 128
        }
    })

params = EasyDict({'cfg': cfg, 'search_size': 256, 'template_size': 128,
    'checkpoint': '/home/furkan/Desktop/CS/altek/tracking_implementations/ORTrack/pretrained_models/ORTrack_ep0300.pth',
    'debug':False,
    'use_visdom':False,
    'save_all_boxes':False,
    'template_factor':2.0,
    'template_size':128,
    'search_factor':4.0,
    'search_size':256})
tracker = ORTrack(params, dataset_name='ORTrack-MD-DeiT')

possible_paths = [os.path.join(repo_path, 'pretrained_models', 'ORTrack_ep0300.pth'), 'ORTrack_ep0300.pth']
model_path = next((p for p in possible_paths if os.path.exists(p)), None)
        
if model_path:
    checkpoint = torch.load(model_path, map_location='cpu')
    tracker.network.load_state_dict(checkpoint['net'] if 'net' in checkpoint else checkpoint, strict=False)
    tracker.network.to(device)
    tracker.network.eval()
    print(f"{Colors.GREEN}[Başarılı]{Colors.RESET} Model Hazır!")
else:
    print("Model dosyası yok!")
    sys.exit()

# 4. KAMERA VE TAKİP DÖNGÜSÜ
cap = cv2.VideoCapture(0)

print(f"\n{Colors.CYAN}=== TALİMATLAR ==={Colors.RESET}")
print("1. Mouse ile nesneyi seç ve ENTER'a bas.")
print("2. Çıkmak için 'q'.\n")

ret, frame = cap.read()
init_rect = cv2.selectROI("Hedef Secimi", frame, fromCenter=False, showCrosshair=True)
cv2.destroyWindow("Hedef Secimi")

if init_rect[2] > 0 and init_rect[3] > 0:
    init_bbox = list(init_rect)
    
    with torch.no_grad():
        tracker.initialize(frame, {'init_bbox': init_bbox})
    
    print(f"{Colors.GREEN}>>> TAKİP BAŞLADI! <<<{Colors.RESET}")
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        # FPS Ölçümü Başlangıç
        timer = cv2.getTickCount()
        
        with torch.no_grad():
            output = tracker.track(frame)
        
        # FPS Hesabı
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
        
        # --- DÜZELTME BURADA YAPILDI ---
        # Artık 'target_bbox' kullanıyoruz
        bbox = output.get('target_bbox', None)
        
        # Eğer 'avtrack.py' dosyasını düzenlediysen 'best_score' gelir.
        # Düzenlemediysen 1.0 varsayarız (Kutu yine de çizilir).
        score = output.get('best_score', 1.0)
        
        if bbox is not None:
            x, y, w, h = [int(v) for v in bbox]
            
            # Skor kontrolü (Opsiyonel: Skor çok düşükse kırmızı yap)
            # Eğer avtrack.py'yi düzenlemediysen score hep 1.0 olur, hep yeşil yanar.
            color = (0, 255, 0) if score > 0.3 else (0, 0, 255)
            
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
            
            # Bilgi Yazıları
            cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.putText(frame, f"Skor: {float(score):.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Terminale de yaz (Canlı olduğunu görmek için)
            if int(fps) > 0:
                print(f"\rFPS: {int(fps)} | Skor: {score:.2f} | Kutu: {x},{y},{w},{h}   ", end="")
        
        cv2.imshow("AVTrack Live", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
