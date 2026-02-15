import os
import sys
import torch
import cv2
import gc
import argparse
from easydict import EasyDict

# --- RENKLİ ÇIKTI ---
class Colors:
    GREEN = '\033[92m'
    CYAN = '\033[96m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    RESET = '\033[0m'

print(f"\n{Colors.CYAN}=== MIXFORMERV2 SİSTEMİ BAŞLATILIYOR ==={Colors.RESET}")

# 1. ORTAM AYARLARI
# ------------------------------------------------------------------
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    device = 'cuda'
else:
    print("HATA: GPU yok! MixFormerV2 GPU olmadan çok yavaş çalışır.")
    sys.exit()

current_dir = os.getcwd()
sys.path.append(current_dir)

# ==============================================================================
# KRİTİK HATA ÇÖZÜCÜ: LOCAL.PY DOSYALARINI ONARMA
# ==============================================================================

# 1. TEST İÇİN LOCAL.PY (Senin aldığın 'AttributeError: prj_dir' hatasının çözümü)
test_local_path = os.path.join(current_dir, 'lib', 'test', 'evaluation', 'local.py')

# Dosyayı her seferinde yeniden yazıyoruz ki 'prj_dir' kesin olsun.
print(f"{Colors.YELLOW}Ayarlar güncelleniyor: {test_local_path}{Colors.RESET}")
with open(test_local_path, 'w') as f:
    f.write("from lib.test.evaluation.environment import EnvSettings\n\n")
    f.write("def local_env_settings():\n")
    f.write("    settings = EnvSettings()\n")
    f.write(f"    settings.prj_dir = '{current_dir}'\n")  # <-- İŞTE BU EKSİKTİ
    f.write(f"    settings.save_dir = '{os.path.join(current_dir, 'output')}'\n")
    f.write(f"    settings.results_path = '{os.path.join(current_dir, 'output', 'test', 'results')}'\n")
    f.write(f"    settings.segmentation_path = '{os.path.join(current_dir, 'output', 'test', 'segmentation_results')}'\n")
    f.write("    return settings\n")

# 2. TRAIN İÇİN LOCAL.PY (PyTorch model yükleme hatasını önlemek için)
train_admin_dir = os.path.join(current_dir, 'lib', 'train', 'admin')
train_local_path = os.path.join(train_admin_dir, 'local.py')

if not os.path.exists(train_admin_dir):
    os.makedirs(train_admin_dir)

if not os.path.exists(train_local_path):
    with open(train_local_path, 'w') as f:
        f.write("class EnvironmentSettings:\n")
        f.write("    def __init__(self):\n")
        f.write("        self.workspace_dir = ''\n")

# ==============================================================================

# İthalatlar (Dosyalar oluştuktan sonra yapılmalı)
try:
    from lib.test.tracker.mixformer2_vit import MixFormer   
    from lib.test.parameter.mixformer2_vit import parameters
except ImportError as er:
    print(f"{Colors.RED}İMPORT HATASI: {er}{Colors.RESET}")
    print("Lütfen klasör yapısının bozulmadığından emin olun.")
    sys.exit()

def run_tracker():
    # 3. CONFIG: ORTAK AYARLAR (HARDCODED)
    # YAML dosyası okumak yerine parametreleri buradan zorluyoruz.
    # Böylece 'yaml dosyası bulunamadı' hatalarından da kurtuluruz.
    
    # === PARAMETRE AYARLARI ===
    # Eğer indirdiğin dosya adı farklıysa burayı güncelle!
    model_name = "mixformerv2_small.pth" 
    model_path = os.path.join(current_dir, "pretrained_models", model_name)
    
    # 288'lik Student Modeli İçin Config
# === FRANKENSTEIN CONFIG (Dosyaya Tam Oturan Ayarlar) ===
    mixformer_config = EasyDict({
        "DATA": {
              "MAX_SAMPLE_INTERVAL": 200,
            "SEARCH": {
                "SIZE": 224,        # Backbone 224px (14x16) istiyor
                "FACTOR": 4.0,
                "CENTER_JITTER": 3,
                "SCALE_JITTER": 0.25
            },
            "TEMPLATE": {
                "SIZE": 112,        # Backbone 112px (7x16) istiyor
                "FACTOR": 2.0,
                "CENTER_JITTER": 0,
                "SCALE_JITTER": 0
            }
        },
        "MODEL": {
            # KODUN KABUL ETTİĞİ TEK İSİM:
            "VIT_TYPE": "base_patch16", 
            
            # --- KAFA AYARLARI (Hata Burada Çözülüyor) ---
            "HEAD_TYPE": "MLP",    # "CORNER" değil "MLP" yazmalısın!
            "FEAT_SZ": 96,         # Checkpoint [96, 768] bekliyor (Önceki hatadan yakaladık)
            
            "BACKBONE": {
                "VIT_TYPE": "base_patch16",
                "STRIDE": 16,
                "DEPTH": 4,         # Dosya sadece 4 katmanlı
                "MLP_RATIO": 1.0,   # Dosya ince (Ratio 1)
                "PRETRAINED": False
            },
            "HEAD_TYPE": "MLP",     # Garanti olsun diye buraya da ekledim
            "HIDDEN_DIM": 768,      # Dosya 768 kanal genişliğinde
            "PREDICT_MASK": False
        },
        "TRAIN": { "EPOCH": 500 },
        "TEST": {
            "EPOCH": 300,
            "SEARCH_FACTOR": 4.0,
            "SEARCH_SIZE": 224,
            "TEMPLATE_FACTOR": 2.0,
            "TEMPLATE_SIZE": 112,
            "UPDATE_INTERVALS": {
                "LASOT": [200],
                "GOT10K_TEST": [200],
                "TRACKINGNET": [25],
                "VOT20": [10],
                "VOT20LT": [200]
            }
        }
    })

    if not os.path.exists(model_path):
        print(f"{Colors.RED}HATA: Model dosyası bulunamadı!{Colors.RESET}")
        print(f"Aranan: {model_path}")
        return

    print(f"Model Yükleniyor... ({model_name})")
    
    params = EasyDict()
    params.cfg = mixformer_config
    params.checkpoint = model_path
    
    # Kısa yollar (Tracker kodunun içinde buralara erişiliyor)
    params.search_factor = mixformer_config.TEST.SEARCH_FACTOR
    params.search_size = mixformer_config.TEST.SEARCH_SIZE
    params.template_factor = mixformer_config.TEST.TEMPLATE_FACTOR
    params.template_size = mixformer_config.TEST.TEMPLATE_SIZE
    params.debug = False
    params.save_all_boxes = False

    # Tracker Başlatılıyor
    tracker = MixFormer(params, dataset_name='MixFormer-Live')
    
    # 4. KAMERA
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("HATA: Kamera açılamadı!")
        return

    cv2.namedWindow("MixFormerV2", cv2.WINDOW_NORMAL)
    print(f"{Colors.GREEN}HAZIR! Nesne seçmek için bir tuşa basın.{Colors.RESET}")

    ret, frame = cap.read()
    if not ret: return

    # ROI Seçimi
    init_box = cv2.selectROI("MixFormerV2", frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("MixFormerV2") # ROI penceresini kapat
    
    # Tracker Initialize
    tracker.initialize(frame, {'init_bbox': list(init_box)})
    
    print("Takip Başladı. Çıkış: 'q'")
    
    timer = cv2.TickMeter()

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        timer.start()
        out = tracker.track(frame)
        timer.stop()
        
        fps = 1.0 / (timer.getTimeSec() + 1e-8)
        timer.reset()
        
        bbox = out['target_bbox']
        x, y, w, h = [int(i) for i in bbox]
        
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f"MixFormerV2 | FPS: {int(fps)}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow("MixFormerV2 Live", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    run_tracker()