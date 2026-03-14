import torch
import numpy as np

from lib.test.tracker.avtrack import AVTrack
from lib.test.tracker.avtrack_trt import AVTrackEngine 

class AVTrackTRT(AVTrack):
    def __init__(self, params, dataset_name,engine_path):
        super(AVTrackTRT, self).__init__(params, dataset_name)
        if hasattr(self, 'network'):
            del self.network
        torch.cuda.empty_cache()
        
        self.trt_engine = AVTrackEngine(engine_path)
        
        # Template (Hedef) görüntüsünü NumPy olarak saklayacağımız değişken
        self.template_np = None
    def initialize(self, image, info: dict):
        super(AVTrackTRT, self).initialize(image, info)
        
        # 2. Orijinal kodun hazırladığı Template Tensörünü alıp NumPy'a çevir
        # AVTrack/OSTrack mimarisinde template genelde 'tensors' objesi içinde saklanır
        template_tensor = self.z_dict1.tensors if hasattr(self.z_dict1, 'tensors') else self.z_dict1
        
        # Tensörü CPU'ya çek ve NumPy array yap (TensorRT böyle istiyor)
        self.template_np = template_tensor.cpu().numpy()