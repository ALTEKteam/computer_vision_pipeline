import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import cv2 as cv # type: ignore
import sys
import time

# Modülleri import ediyoruz
# Eğer __init__.py dosyalarını oluşturduysan bu şekilde temiz import yapabilirsin:
from modules.yolo_engine import YoloDetector
from modules.tracker_adapter import TrackerAdapter
from core.pipeline import DronePipeline
from params.tracker_types import TRACKERS

def main():
    # --- 1. AYARLAR ---
    # Model ve Video yolları (Senin bilgisayarındaki yollara göre düzenle)
    MODEL_PATH = r"/home/furkan/Desktop/CS/altek/pipeline/models/best_new_fp16.onnx"
    VIDEO_PATH = r"/home/furkan/Desktop/CS/altek/pipeline/videos/Talon_video.mp4"

    if not os.path.exists(MODEL_PATH):
        print(f"HATA: Model dosyası bulunamadı -> {MODEL_PATH}")
        return
    if not os.path.exists(VIDEO_PATH):
        print(f"HATA: Video dosyası bulunamadı -> {VIDEO_PATH}")
        return

    # --- 2. SİSTEMİ BAŞLATMA (INITIALIZATION) ---
    print("Sistem Başlatılıyor...")
    
    print("1. YOLO Dedektörü Yükleniyor...")
    yolo_engine = YoloDetector(model_path=MODEL_PATH, conf_thres=0.5)
    
    print("2. AVTrack Takipçisi Hazırlanıyor...")
    tracker_engine = TrackerAdapter(tracker_model=TRACKERS.ORTrack)  # VitTracker kullanılıyor
    
    print("3. Pipeline (Beyin) Kuruluyor...")
    pipeline = DronePipeline(yolo_engine, tracker_engine)

    # --- 3. ANA DÖNGÜ (MAIN LOOP) ---
    cap = cv.VideoCapture(VIDEO_PATH)
    
    # Tam ekran pencere ayarı (Opsiyonel)
    cv.namedWindow("TEKNOFEST 2026 - TARGET SYSTEM", cv.WINDOW_NORMAL)
    
    print("Sistem Hazır! Başlatılıyor...")
    
    prev_time = 0
    
    while True:
        ret, frame = cap.read()
        if not ret: break
            
        # Pipeline işlemi
        processed_frame = pipeline.run_step(frame)
        
        # --- 3. EKLEME: FPS Hesaplama ve Ekrana Yazma ---
        curr_time = time.time()
        time_diff = curr_time - prev_time
        
        # Sıfıra bölünme hatasını önlemek için küçük kontrol
        if time_diff > 0:
            fps = 1 / time_diff
        else:
            fps = 0
            
        prev_time = curr_time
        
        # FPS Metnini Oluştur
        fps_text = f"FPS: {int(fps)}"
        
        # Ekrana Yaz (Sağ alt köşe, Camgöbeği rengi)
        cv.putText(processed_frame, fps_text, (1050, 680), 
                   cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        # ------------------------------------------------
        
        cv.imshow("TEKNOFEST 2026 - TARGET SYSTEM", processed_frame)
        
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()