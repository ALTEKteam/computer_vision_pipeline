from easydict import EasyDict
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
# CHANGE HERE
params = EasyDict({'cfg': cfg, 'search_size': 256, 'template_size': 128,
    'checkpoint': '/home/furkan/Desktop/CENG/altek/pipeline/models/AVTrack_model.pth',
    'debug':False,
    'use_visdom':False,
    'save_all_boxes':False,
    'template_factor':2.0,
    'template_size':128,
    'search_factor':4.0,
    'search_size':256})

dataset_name='AVTrack-MD-DeiT'