from easydict import EasyDict

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
params = EasyDict()
params.cfg = mixformer_config
params.checkpoint =  '/home/furkan/Desktop/CS/altek/pipeline/models/mixformerv2_small.pth'

params.search_factor = mixformer_config.TEST.SEARCH_FACTOR
params.search_size = mixformer_config.TEST.SEARCH_SIZE
params.template_factor = mixformer_config.TEST.TEMPLATE_FACTOR
params.template_size = mixformer_config.TEST.TEMPLATE_SIZE
params.debug = False
params.save_all_boxes = False