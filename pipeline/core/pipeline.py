import time
import cv2 as cv
from enum import Enum

# Durum Yönetimi için Enum (Okunabilirliği artırır)
class SystemState(Enum):
    SEARCHING = 1   # YOLO ile etrafı tara
    TRACKING = 2    # Hedefi bulduk, takip et
    LOST = 3        # Hedef anlık kayboldu, bekle

class DronePipeline:
    def __init__(self, yolo_engine, tracker_engine):
        self.yolo = yolo_engine
        self.tracker = tracker_engine
        
        # Başlangıç durumu
        self.state = SystemState.SEARCHING
        
        # Zamanlayıcılar
        self.lock_start_time = None  # Kilitlenme süresi sayacı
        self.lost_start_time = None  # Kayıp süresi sayacı
        
        # Hedef Bilgisi
        self.bbox = None # [x, y, w, h]

    def run_step(self, frame):
        output_frame = frame.copy()
        
        # --- DURUM 1: ARAMA MODU ---
        if self.state == SystemState.SEARCHING:
            detected_bbox = self.yolo.detect(frame)
            
            if detected_bbox is not None:
                print(f"HEDEF BULUNDU! Takip Başlıyor...")
                self.tracker.initialize(frame, detected_bbox)
                self.state = SystemState.TRACKING
                self.bbox = detected_bbox
                self.lock_start_time = time.time()
                self._draw_status(output_frame, self.bbox, (0, 0, 255), "DETECTED")
            else:
                cv.putText(output_frame, "SEARCHING (YOLO)...", (50, 50), 
                           cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 4)

        # --- DURUM 2: TAKİP MODU ---
        elif self.state == SystemState.TRACKING:
            success, new_bbox = self.tracker.update(frame)
            
            if success:
                self.bbox = new_bbox
                self.lost_start_time = None # Hedef var, kayıp sayacını sıfırla
                
                # Kilitlenme Süresi Hesabı
                lock_duration = time.time() - self.lock_start_time
                
                # --- SENARYO: 4 SANİYE SONRASI ---
                if lock_duration > 4.0:
                    status_text = f"TARGETING IS SUCCESSFUL! ({lock_duration:.1f}s)"
                    
                    # Görsel Efekt: Kalın Kırmızı Çerçeve
                    cv.rectangle(output_frame, (0,0), (output_frame.shape[1], output_frame.shape[0]), (255,0,0), 15)
                    self._draw_status(output_frame, self.bbox, (255,0,0), status_text)
                    
                    # İSTEĞE BAĞLI: 4 saniye dolunca ne olsun?
                    # Eğer "Bıraksın ve tekrar aramaya dönsün" istiyorsan şu bloğu aç:
                    if lock_duration > 5.0: # 1 saniye de kilitli göstersin sonra sıfırlasın
                        print("Kilitlenme Tamamlandı. Sistem Sıfırlanıyor...")
                        self.state = SystemState.SEARCHING
                        self.tracker.clear_initialization()
                else:
                    # Henüz 4 saniye olmadı, saymaya devam
                    status_text = f"TRACKING... {4.0 - lock_duration:.1f}s"
                    cv.putText(output_frame, "TRACKING (AVTRACK)...", (50, 50), 
                           cv.FONT_HERSHEY_SIMPLEX, 1, (138,168,0), 4)
                    self._draw_status(output_frame, self.bbox, (0, 255, 0), status_text)

            else:
                # Takip koptu! Hemen YOLO'ya dönme, LOST moduna geç
                print("Takip Koptu -> LOST State")
                self.state = SystemState.LOST
                self.lost_start_time = time.time()

        # --- DURUM 3: KAYIP MODU (1 Saniye Bekleme) ---
        elif self.state == SystemState.LOST:
            elapsed_lost_time = time.time() - self.lost_start_time
            
            # 1. Tracker'a hala şans veriyoruz (Belki görüntü düzelir)
            success, new_bbox = self.tracker.update(frame)
            
            if success:
                print("Hedef LOST modunda geri kazanıldı!")
                self.state = SystemState.TRACKING
                self.bbox = new_bbox
                self.lost_start_time = None
                # Not: lock_start_time'ı sıfırlamıyoruz ki süre kaldığı yerden (veya yaklaştığı yerden) devam etsin
            
            # 2. Eğer 1 saniyeden fazla süredir kayıpsa -> ARTIK PES ET
            elif elapsed_lost_time > 1.0:
                print(f"{elapsed_lost_time:.1f}s geçti, hedef gelmedi. YOLO devreye giriyor.")
                self.state = SystemState.SEARCHING
                self.bbox = None
                self.lock_start_time = None
            
            # 3. Bekleme süresindeyiz
            else:
                cv.putText(output_frame, f"LOST! Waiting... {1.0 - elapsed_lost_time:.1f}s", 
                           (50, 100), cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 4)
                
                # Son bilinen konumu hayalet (gri) çizelim
                if self.bbox:
                    self._draw_status(output_frame, self.bbox, (128, 128, 128), "LOST?")

        return output_frame

    def _draw_status(self, img, bbox, color, text):
        """Yardımcı fonksiyon: Kutu ve yazı çizer"""
        x, y, w, h = [int(v) for v in bbox]
        # Kutu
        cv.rectangle(img, (x, y), (x + w, y + h), color, 3)
        # Yazı arka planı (okunurluk için)
        (tw, th), _ = cv.getTextSize(text, cv.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv.rectangle(img, (x, y - 25), (x + tw, y), color, -1)
        # Yazı
        cv.putText(img, text, (x, y - 5), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)