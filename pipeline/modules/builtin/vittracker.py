import cv2 as cv

class VitTracker:
    def __init__(self):
        # OpenCV'nin yerleşik VIT (Vision Transformer) Tracker'ını oluşturuyoruz.
        # Not: Bunun çalışması için opencv-python sürümünün güncel olması (4.5+) önerilir.
        try:
            net = "/home/furkan/Desktop/CS/altek/pipeline/models/vitTracker.onnx"  # VIT model yolu
            model = cv.dnn.readNet(net)
            self.tracker = cv.TrackerVit_create(model,tracking_score_threshold=0.5)
            print(">> VIT Tracker başarıyla oluşturuldu.")
        except AttributeError:
            print("HATA: OpenCV sürümünüzde VIT Tracker yok. 'pip install opencv-contrib-python' deneyin veya sürümü güncelleyin.")
            self.tracker = None

    def initialize(self, frame, bbox):
        """
        Tracker'ı ilk hedefe kilitler.
        bbox formatı: [x, y, w, h] (int veya float olabilir)
        """
        if self.tracker is None: return
        
        # OpenCV trackerları bazen float bbox sevmez, int'e çevirelim garanti olsun.
        bbox = tuple(map(int, bbox))
        
        # VIT Tracker tek seferlik bir nesne değildir, her initialize'da sıfırlamak gerekebilir
        # Ancak OpenCV'de 'init' metodu genelde resetler. Yine de garanti olması için yeniden oluşturabiliriz.
        # (Performans düşerse bu satırı kaldır, ama doğruluk artar)
        if (not self.tracker):   self.tracker = cv.TrackerVit_create() 
        
        self.tracker.init(frame, bbox)

    def track(self, frame):
        """
        Bir sonraki karedeki konumu tahmin eder.
        Dönüş: [x, y, w, h] veya None (Eğer kaybettiyse)
        """
        if self.tracker is None: return None

        success, bbox = self.tracker.update(frame)
        print("Success:", success, "BBox:", bbox)
        if success:
            # OpenCV tuple (x,y,w,h) döner, biz bunu listeye çevirip int yapalım
            return {"target_bbox": bbox, "best_score": self.tracker.getTrackingScore()}
        else:
            return None