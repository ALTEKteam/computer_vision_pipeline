import sys
import cv2 as cv
import numpy as np

# Change the dependency paths according to the current tracker you want to use.
# import params.tracker.av_track_params as avtrack_params
# import params.tracker.or_track_params as ortrack_params
import params.tracker.mixformer_params as mixformer_params

# sys.path.append(r'/home/furkan/Desktop/CS/altek/tracking_implementations/AVTrack')
# sys.path.append(r'/home/furkan/Desktop/CS/altek/tracking_implementations/ORTrack')
sys.path.append(r'/home/furkan/Desktop/CS/altek/tracking_implementations/MixFormerV2')

# from lib.test.tracker.avtrack import AVTrack
# from lib.test.tracker.ortrack import ORTrack
# from .builtin.vittracker import VitTracker
from lib.test.tracker.mixformer2_vit import MixFormer   
from params.tracker_types import TRACKERS

class TrackerAdapter:

    def __init__(self, model_path=None, tracker_model = TRACKERS.ORTrack):
        """
        Tracker modelini yükler ve hazırlar.
        """
        # Burada AVTracker sınıfını başlatıyoruz.
        # Eğer parametre (cfg) istiyorsa buraya eklemelisin.
        
        # self.tracker = AVTracker(params=...) 
        if (tracker_model == TRACKERS.AVTrack):
            pass
            # self.tracker = AVTrack(avtrack_params.params, dataset_name=avtrack_params.dataset_name) 
        elif (tracker_model == TRACKERS.ORTrack):
            pass
            # self.tracker = ORTrack(ortrack_params.params, dataset_name=ortrack_params.dataset_name)
        elif (tracker_model == TRACKERS.MixFormerV2):
            self.tracker = MixFormer(mixformer_params.params,dataset_name="MixFormer-Live")
        else:
            print(f"Yeni tracker'a geçildi: {tracker_model}")
            self.tracker = VitTracker()  # VitTracker sınıfını kullan        
        self.is_initialized = False
        self.tracker_model = tracker_model
        print("Tracker Modülü Hazır (Model yüklendi)")

    def initialize(self, frame, bbox):
        """
        YOLO'dan gelen kutu ile takibi başlatır.
        bbox formatı: [x, y, w, h]
        """
        if self.tracker is None:
            print("UYARI: Tracker sınıfı başlatılmamış (Import hatası olabilir)")
            return

        # AVTrack genelde init fonksiyonu ile başlar
        # Bazen 'image' ve 'info' (bbox) ister.
        
        # Bbox'ı int yapalım ne olur ne olmaz
        bbox = [int(x) for x in bbox]
        if (self.tracker_model != TRACKERS.VitTracker):
            # AVTrack init çağrısı (Senin koduna özel):
            self.tracker.initialize(frame, {'init_bbox': bbox})
        else:
            self.tracker.initialize(frame, bbox)
        
        self.is_initialized = True
        # print(f"Tracker başlatıldı. Hedef: {bbox}")

    def update(self, frame):
        """
        Her karede çalışır.
        Return: (success: bool, bbox: list) -> [x, y, w, h]
        """
        if not self.is_initialized or self.tracker is None:
            return False, None

        # AVTrack'in 'track' fonksiyonunu çağırıyoruz
        outputs = self.tracker.track(frame)
        
        # --- ÇIKTIYI TERCÜME ETME (Senin AVTrack yapına göre) ---
        # Genelde 'target_bbox' ve 'best_score' döner demiştik.
        if (outputs is None):
            return False, None
        print(outputs)
        if 'target_bbox' in outputs:
            bbox = outputs['target_bbox']
            score = outputs.get('best_score', 1.0) # Skor yoksa 1.0 varsay
            
            # Eşik Değeri (Threshold) Kontrolü
            if score > 0.4: # %40'ın altındaysa kaybettik sayalım
                # Float gelebilir, int'e çevirip döndürelim
                final_bbox = [int(v) for v in bbox]
                return True, final_bbox
            else:
                return False, None # Güven skoru çok düşük
        
        return False, None

    def clear_initialization(self):
        """
        Tracker'ı sıfırlar.
        """
        self.is_initialized = False
TrackerAdapter()