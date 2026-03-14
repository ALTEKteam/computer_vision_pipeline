"""
AVTrack Pipeline Adapter (without AM head)
"""

import os
import sys
from time import time
import types
import numpy as np
import cv2 as cv
import torch
import onnxruntime as ort

sys.path.append(r'/home/furkan/Desktop/CS/altek/tracking_implementations/AVTrack')
# =============================================================================
# Load of AVTrack model and configuration
# =============================================================================

def _load_avtrack(config_name, checkpoint_path):
    """Loads AVTrack model and the configuration by using config_name parameter"""
    from lib.config.avtrack.config import cfg, update_config_from_file
    from lib.models.avtrack import build_avtrack
    from lib.models.avtrack.utils import combine_tokens, recover_tokens

    # Config
    yaml_path = os.path.join(os.path.dirname(__file__), '..', 
                             'experiments', 'avtrack', f'{config_name}.yaml')
    # If yaml does not exist in the path, search in other locations
    if not os.path.exists(yaml_path):
        for base in sys.path:
            candidate = os.path.join(base, 'experiments', 'avtrack', f'{config_name}.yaml')
            if os.path.exists(candidate):
                yaml_path = candidate
                break

    if os.path.exists(yaml_path):
        update_config_from_file(yaml_path)

    # Model
    model = build_avtrack(cfg, training=False)

    # Checkpoint
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    sd = ckpt.get('net', ckpt.get('state_dict', ckpt.get('model', ckpt))) if isinstance(ckpt, dict) else ckpt
    clean = {k.replace('module.', ''): v for k, v in sd.items()}
    model.load_state_dict(clean, strict=False)

    # Dissmiss attention maps and AM head overwriting backbone forward method
    def fwd_no_am(self, z, x, is_distill=False):
        x = self.patch_embed(x)
        z = self.patch_embed(z)
        z += self.pos_embed_z
        x += self.pos_embed_x
        x = combine_tokens(z, x)
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        lens_z = self.pos_embed_z.shape[1]
        lens_x = self.pos_embed_x.shape[1]
        x = recover_tokens(x, lens_z, lens_x)
        return self.norm(x), {"attn": None, "probs_active": []}

    def fwd_backbone(self, z, x, **kwargs):
        return self.forward_features(z, x, kwargs.get('is_distill', False))

    model.backbone.forward_features = types.MethodType(fwd_no_am, model.backbone)
    model.backbone.forward = types.MethodType(fwd_backbone, model.backbone)

    model.eval()
    return model, cfg


# =============================================================================
# Pre/post processing
# =============================================================================

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _preprocess(image, size):
    img = cv.cvtColor(image, cv.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = cv.resize(img, (size, size), interpolation=cv.INTER_LINEAR)
    img = (img - MEAN) / STD
    return torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).float().pin_memory()

def _crop(frame, cx, cy, wh_area, out_size, factor):
    crop_sz = max(int(np.ceil(np.sqrt(wh_area) * factor)), 1)
    x1, y1 = int(cx - crop_sz / 2), int(cy - crop_sz / 2)
    x2, y2 = x1 + crop_sz, y1 + crop_sz
    H, W = frame.shape[:2]
    pads = [max(0, -y1), max(0, y2 - H), max(0, -x1), max(0, x2 - W)]
    x1c, y1c = max(0, x1), max(0, y1)
    x2c, y2c = min(W, x2), min(H, y2)
    cropped = frame[y1c:y2c, x1c:x2c]
    if any(p > 0 for p in pads):
        avg = np.mean(cropped, axis=(0, 1)).astype(np.uint8).tolist()
        cropped = cv.copyMakeBorder(cropped, *pads, cv.BORDER_CONSTANT, value=avg)
    rf = out_size / crop_sz
    return cv.resize(cropped, (out_size, out_size), interpolation=cv.INTER_LINEAR), rf, crop_sz


# =============================================================================
# AVTrack Tracker
# =============================================================================

class AVTrackTracker:
    def __init__(self, config_name='deit_tiny_patch16_224', checkpoint_path=None,
             onnx_path=None, engine_path=None, device='cuda'):
        """
        Initialize the AVTrackTracker by using given parameters
        in case fof onnx_path or engine_path, checkpoint_path is not required and will be ignored.
         - onnx_path: The path of ONNx Model
         - engine_path: The path of TensorRT engine file (required for using TensorRT, faster than ONNX but only supports CUDA)
         - config_name: The name of the configuration yaml file (without .yaml extension) located in experiments/avtrack/
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.use_ort = onnx_path is not None

        self.use_trt = engine_path is not None
        self.use_ort = onnx_path is not None and not self.use_trt

        if self.use_trt:
            import tensorrt as trt

            self.trt_logger = trt.Logger(trt.Logger.WARNING)
            with open(engine_path, 'rb') as f:
                runtime = trt.Runtime(self.trt_logger)
                self.trt_engine = runtime.deserialize_cuda_engine(f.read())
            self.trt_context = self.trt_engine.create_execution_context()

            self.trt_inputs = {}
            self.trt_outputs = {}
            for i in range(self.trt_engine.num_io_tensors):
                name = self.trt_engine.get_tensor_name(i)
                shape = tuple(self.trt_engine.get_tensor_shape(name))
                dtype = trt.nptype(self.trt_engine.get_tensor_dtype(name))
                tensor = torch.empty(shape, dtype=torch.float32, device='cuda')
                self.trt_context.set_tensor_address(name, tensor.data_ptr())
                mode = self.trt_engine.get_tensor_mode(name)
                if mode == trt.TensorIOMode.INPUT:
                    self.trt_inputs[name] = tensor
                else:
                    self.trt_outputs[name] = tensor

            from lib.config.avtrack.config import cfg, update_config_from_file
            for base in sys.path:
                yp = os.path.join(base, 'experiments', 'avtrack', f'{config_name}.yaml')
                if os.path.exists(yp):
                    update_config_from_file(yp)
                    break
            self.cfg = cfg
            self.model = None
            self.sess = None

            # Warmup
            for _ in range(5):
                self.trt_context.execute_async_v3(stream_handle=torch.cuda.current_stream().cuda_stream)
            torch.cuda.synchronize()
            print(f"[AVTrack] TensorRT modu | {engine_path}")

        elif self.use_ort:
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_options.enable_mem_pattern = True
            sess_options.enable_cpu_mem_arena = True

            self.sess = ort.InferenceSession(
                onnx_path,
                sess_options=sess_options,
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            )
            # Load configuration again (for constraints and hyperparameters)
            from lib.config.avtrack.config import cfg, update_config_from_file
            yaml_path = None
            for base in sys.path:
                candidate = os.path.join(base, 'experiments', 'avtrack', f'{config_name}.yaml')
                if os.path.exists(candidate):
                    yaml_path = candidate
                    break
            if yaml_path:
                update_config_from_file(yaml_path)
            self.cfg = cfg
            self.model = None
        else:
            if checkpoint_path is None:
                raise ValueError("checkpoint_path veya onnx_path gerekli")
            self.model, self.cfg = _load_avtrack(config_name, checkpoint_path)
            self.model.to(self.device)
            self.sess = None

        self.template_size = self.cfg.TEST.TEMPLATE_SIZE
        self.search_size = self.cfg.TEST.SEARCH_SIZE
        self.template_factor = self.cfg.TEST.TEMPLATE_FACTOR
        self.search_factor = self.cfg.TEST.SEARCH_FACTOR

        self.template_tensor = None
        self.template_np = None
        self.cx = self.cy = self.w = self.h = 0
        dummy_t = np.zeros((1, 3, self.template_size, self.template_size), dtype=np.float32)
        dummy_s = np.zeros((1, 3, self.search_size, self.search_size), dtype=np.float32)
        if self.use_ort and self.sess is not None:
            dummy_t = np.zeros((1, 3, self.template_size, self.template_size), dtype=np.float32)
            dummy_s = np.zeros((1, 3, self.search_size, self.search_size), dtype=np.float32)
            for _ in range(5):
                self.sess.run(None, {'template': dummy_t, 'search': dummy_s})
            print("[AVTrack] ORT Warmup tamamlandi")

    '''
    Start tracking by initializing the tracker with the first frame and the initial bounding box.
        - frame: The first video frame (numpy array)
        - info: The initial bounding box, can be either a list/tuple [x, y, w, h] or a dictionary with key 'init_bbox'
    '''
    def initialize(self, frame, info):
        bbox = info['init_bbox'] if isinstance(info, dict) else info
        x, y, w, h = bbox
        self.cx, self.cy = x + w / 2, y + h / 2
        self.w, self.h = w, h

        tmpl, _, _ = _crop(frame, self.cx, self.cy, w * h,
                           self.template_size, self.template_factor)

        if self.use_ort or self.use_trt:
            img = cv.cvtColor(tmpl, cv.COLOR_BGR2RGB).astype(np.float32) / 255.0
            img = cv.resize(img, (self.template_size, self.template_size))
            img = (img - MEAN) / STD
            self.template_np = img.transpose(2, 0, 1)[np.newaxis].astype(np.float32)
        else:
            self.template_tensor = _preprocess(tmpl, self.template_size).to(self.device)
        dummy_s = np.zeros((1, 3, self.search_size, self.search_size), dtype=np.float32)
        self._init_w = self.w
        self._init_h = self.h

    '''
    General tracker method called for each frame. It directs the tracking process according to the current mode.
        - frame: The current video frame (numpy array)
    '''
    def track(self, frame):
        wh_area = self.w * self.h
        search_img, rf, crop_sz = _crop(frame, self.cx, self.cy, wh_area,
                                         self.search_size, self.search_factor)

        if self.use_trt:
            return self._track_trt(search_img, rf)
        elif self.use_ort:
            return self._track_ort(search_img, rf)
        else:
            return self._track_pytorch(search_img, rf)

    '''
    Tracking method using ONNX Runtime. It does preprocess the search image, runs inference, and makes postprocessing the output to update the target bounding box.
        - search_img: The cropped search image centered around the last known target position
        - rf: The resize factor used during cropping, needed to scale the predictions back to the original image coordinates
    '''
    def _track_ort(self, search_img, rf):
        img = cv.cvtColor(search_img, cv.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = cv.resize(img, (self.search_size, self.search_size))
        img = (img - MEAN) / STD
        search_np = img.transpose(2, 0, 1)[np.newaxis].astype(np.float32)
        score_map, size_map, offset_map = self.sess.run(
            None, {'template': self.template_np, 'search': search_np})
        sm = score_map[0, 0]
        my, mx = np.unravel_index(np.argmax(sm), sm.shape)
        pw, ph = size_map[0, 0, my, mx], size_map[0, 1, my, mx]
        ox, oy = offset_map[0, 0, my, mx], offset_map[0, 1, my, mx]
        stride = self.search_size / sm.shape[0]
        max_score = float(sm.max())

        new_w = pw * self.search_size / rf
        new_h = ph * self.search_size / rf

        # Mutlak sinir — ilk bbox'in 5 katini gecemez, 0.2 katinin altina dusmez
        new_w = np.clip(new_w, self._init_w * 0.2, self._init_w * 5)
        new_h = np.clip(new_h, self._init_h * 0.2, self._init_h * 5)

        self.cx += ((mx + ox) * stride - self.search_size / 2) / rf
        self.cy += ((my + oy) * stride - self.search_size / 2) / rf
        self.w = new_w
        self.h = new_h

        return {
            'target_bbox': [self.cx - self.w/2, self.cy - self.h/2, self.w, self.h],
            'best_score': max_score,
        }

    '''
    Tracking method using TensorRT engine. Similar to the ONNX Runtime method but uses the TensorRT execution context for inference..
        - search_img: The cropped search image centered around the last known target position
        - rf: The resize factor used during cropping, needed to scale the predictions back to the original image coordinates
    '''
    def _track_trt(self, search_img, rf):
        img = cv.cvtColor(search_img, cv.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = cv.resize(img, (self.search_size, self.search_size))
        img = (img - MEAN) / STD
        search_t = torch.from_numpy(img.transpose(2, 0, 1)[np.newaxis].astype(np.float32)).cuda()
        tmpl_t = torch.from_numpy(self.template_np).cuda()

        self.trt_inputs['template'].copy_(tmpl_t)
        self.trt_inputs['search'].copy_(search_t)

        self.trt_context.execute_async_v3(stream_handle=torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()

        sm = self.trt_outputs['score_map'].cpu().numpy()[0, 0]
        size_map = self.trt_outputs['size_map'].cpu().numpy()
        offset_map = self.trt_outputs['offset_map'].cpu().numpy()

        my, mx = np.unravel_index(np.argmax(sm), sm.shape)
        pw, ph = size_map[0, 0, my, mx], size_map[0, 1, my, mx]
        ox, oy = offset_map[0, 0, my, mx], offset_map[0, 1, my, mx]
        stride = self.search_size / sm.shape[0]
        max_score = float(sm.max())

        new_w = pw * self.search_size / rf
        new_h = ph * self.search_size / rf
        new_w = np.clip(new_w, self._init_w * 0.2, self._init_w * 5)
        new_h = np.clip(new_h, self._init_h * 0.2, self._init_h * 5)

        self.cx += ((mx + ox) * stride - self.search_size / 2) / rf
        self.cy += ((my + oy) * stride - self.search_size / 2) / rf
        self.w = new_w
        self.h = new_h

        return {
            'target_bbox': [self.cx - self.w/2, self.cy - self.h/2, self.w, self.h],
            'best_score': max_score,
        }

    '''
    Tracking method using PyTorch based model. It preprocesses the search image, runs inference through the PyTorch model, 
    and postprocesses the output to update the target bounding box.
        - search_img: The cropped search image centered around the last known target position
        - rf: The resize factor used during cropping, needed to scale the predictions back to the original image coordinates
    '''
    def _track_pytorch(self, search_img, rf):
        search_tensor = _preprocess(search_img, self.search_size).to(self.device)
        with torch.no_grad():
            x, _ = self.model.backbone(z=self.template_tensor, x=search_tensor)
            feat = x[-1] if isinstance(x, list) else x
            enc_opt = feat[:, -self.model.feat_len_s:]
            opt = enc_opt.unsqueeze(-1).permute(0, 3, 2, 1).contiguous()
            bs, Nq, C, HW = opt.size()
            opt_feat = opt.view(-1, C, self.model.feat_sz_s, self.model.feat_sz_s)
            score_map, bbox_pred, size_map, offset_map = self.model.box_head(opt_feat, None)

        pred = bbox_pred.view(bs, Nq, 4).cpu().numpy()[0, 0]
        max_score = float(score_map.cpu().numpy().max())
        self.cx += (pred[0] * self.search_size - self.search_size / 2) / rf
        self.cy += (pred[1] * self.search_size - self.search_size / 2) / rf
        self.w = pred[2] * self.search_size / rf
        self.h = pred[3] * self.search_size / rf

        return {
            'target_bbox': [self.cx - self.w/2, self.cy - self.h/2, self.w, self.h],
            'best_score': max_score,
        }