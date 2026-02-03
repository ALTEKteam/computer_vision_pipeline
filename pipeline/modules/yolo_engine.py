import cv2 as cv
import numpy as np
from onnxruntime import InferenceSession

class YoloDetector:
    def __init__(self, model_path, input_shape=(640, 640), conf_thres=0.5, iou_thres=0.4):
        # Modeli bir kere yüklüyoruz (GPU varsa kullanır)
        # Burada ilk olarak CUDAExecutionProvider eklenmesiyle GPU'nun ön planda olması sağlanır.
        self.session = InferenceSession(model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        self.output_shape = input_shape
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

    def preprocess(self, img):
        """
        Görüntüyü modele hazırlar (Letterbox + Blob).
        """
        h, w = img.shape[:2]
        
        # Scaling faktörü (Letterbox mantığı)
        scale = min(self.output_shape[0]/h, self.output_shape[1]/w)
        
        # Yeni boyutlar (Padding öncesi)
        new_unpad = (int(round(w * scale)), int(round(h * scale)))
        
        # Resize
        resized = cv.resize(img, new_unpad, interpolation=cv.INTER_LINEAR)
        
        # Padding hesaplama (Gri alanlar)
        dw = (self.output_shape[1] - new_unpad[0]) / 2
        dh = (self.output_shape[0] - new_unpad[1]) / 2
        
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        
        # Gri çerçeveyi ekle (114, 114, 114)
        img_padded = cv.copyMakeBorder(resized, top, bottom, left, right, cv.BORDER_CONSTANT, value=(114, 114, 114))
        
        # Blob oluştur (Normalize et ve Channel-First formatına çevir)
        blob = cv.dnn.blobFromImage(img_padded, 1/255.0, self.output_shape, swapRB=True, crop=False)
        # --- KRİTİK DÜZELTME BURASI ---
        # Veriyi modelin beklediği formata (Float16) çeviriyoruz
        blob = blob.astype(np.float16)
        # Geri dönüşüm için parametreleri sakla
        return blob, (dw, dh), scale

    def detect(self, frame):
        original_h, original_w = frame.shape[:2]
        
        # 1. Preprocess
        blob, (dw, dh), scale = self.preprocess(frame)
        
        # 2. Inference
        outputs = self.session.run(None, {self.input_name: blob})
        
        # 3. HIZLANDIRILMIŞ POST-PROCESS (Döngüsüz)
        # Çıktı Shape: (1, 85, 25200) -> (85, 25200)
        data = outputs[0].squeeze(axis=0) 
        
        # Güven skorlarını al (4. satır)
        scores = data[4, :]
        
        # --- HIZ ARTIŞI 1: Eşik altındakileri hemen ele (Maskeleme) ---
        # conf_thres'ten büyük olanların indeksini al
        valid_indices = np.where(scores > self.conf_thres)[0]
        
        if len(valid_indices) == 0:
            return None

        # Sadece geçerli kutuları çek (25000 yerine 5-10 tane kalır)
        valid_data = data[:, valid_indices]
        valid_scores = scores[valid_indices]
        
        # Koordinatları al
        # valid_data[0]: center_x, [1]: center_y, [2]: w, [3]: h
        center_xs = valid_data[0, :]
        center_ys = valid_data[1, :]
        ws = valid_data[2, :]
        hs = valid_data[3, :]
        
        # --- HIZ ARTIŞI 2: Toplu Matematik İşlemi ---
        # Tek tek hesaplamak yerine tüm diziyi bir kerede çarp/böl
        lefts = ((center_xs - ws/2) - dw) / scale
        tops = ((center_ys - hs/2) - dh) / scale
        widths = ws / scale
        heights = hs / scale
        
        # OpenCV NMSBoxes int ve list ister
        boxes = np.stack((lefts, tops, widths, heights), axis=1).astype(int).tolist()
        confidences = valid_scores.tolist()
        
        # 4. NMS
        indices = cv.dnn.NMSBoxes(boxes, confidences, self.conf_thres, self.iou_thres)
        
        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = boxes[i]
                
                # Filtreler (Büyük ve Merkezde mi?)
                is_big = w > (original_w * 0.05) or h > (original_h * 0.05)
                
                # Bu kısmı numpy ile yapmadık çünkü genelde 1-2 kutu kalır, sorun olmaz.
                in_center_x = (original_w * 0.15) < x and (x + w) < (original_w * 0.85)
                in_center_y = (original_h * 0.10) < y and (y + h) < (original_h * 0.90)
                
                if is_big and in_center_x and in_center_y:
                    return [x, y, w, h]
        
        return None