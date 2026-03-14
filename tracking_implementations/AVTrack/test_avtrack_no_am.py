"""
AVTrack - AM'siz PC Test Script
================================
Bu script AVTrack modelini Activation Module OLMADAN calistirir.
Amac: AM kaldirildiginda tracking performansinin korunup korunmadigini
PC uzerinde dogrulamak.

Kullanim:
    cd <AVTrack_ROOT>
    python test_avtrack_no_am.py \
        --checkpoint model.pth \
        --video test_video.mp4

    # Orijinal (AM'li) ile yan yana karsilastirma:
    python test_avtrack_no_am.py \
        --checkpoint model.pth \
        --video test_video.mp4 \
        --compare
"""

import os
import sys
import time
import copy
import types
import math
import argparse
import numpy as np
import cv2

import torch
import torch.nn as nn

sys.path.append(os.getcwd())


# =============================================================================
# 1) Config yukleme (EasyDict tabanli)
# =============================================================================

def load_config(config_name='deit_tiny_patch16_224'):
    """AVTrack config'ini yukle. EasyDict + update_config_from_file kullanir."""
    from lib.config.avtrack.config import cfg, update_config_from_file

    yaml_path = os.path.join(os.getcwd(), 'experiments', 'avtrack', f'{config_name}.yaml')
    if os.path.exists(yaml_path):
        update_config_from_file(yaml_path)
        print(f"[OK] Config yuklendi: {yaml_path}")
    else:
        print(f"[UYARI] YAML bulunamadi: {yaml_path}")
        print(f"        Default config kullaniliyor")

    return cfg


# =============================================================================
# 2) Model yukleme
# =============================================================================

def load_model(checkpoint_path, cfg):
    """AVTrack modelini yukle."""
    from lib.models.avtrack import build_avtrack
    model = build_avtrack(cfg, training=False)

    ckpt = torch.load(checkpoint_path, map_location='cpu')

    if isinstance(ckpt, dict):
        state_dict = ckpt.get('net', ckpt.get('state_dict', ckpt.get('model', ckpt)))
    else:
        state_dict = ckpt

    clean_sd = {}
    for k, v in state_dict.items():
        clean_sd[k.replace('module.', '')] = v

    missing, unexpected = model.load_state_dict(clean_sd, strict=False)
    if missing:
        print(f"[UYARI] Eksik anahtar: {len(missing)}")
        for k in missing[:5]:
            print(f"         - {k}")
    if unexpected:
        print(f"[UYARI] Beklenmeyen anahtar: {len(unexpected)}")
        for k in unexpected[:5]:
            print(f"         - {k}")

    print(f"[OK] Checkpoint yuklendi: {checkpoint_path}")
    return model


# =============================================================================
# 3) AM'yi devre disi birakma (monkey-patch)
# =============================================================================

def forward_features_no_am(self, z, x, is_distill=False):
    """
    forward_features'in AM'siz versiyonu.
    active_score_module CAGRILMIYOR, tum bloklar her zaman calisiyor.
    """
    from lib.models.avtrack.utils import combine_tokens, recover_tokens

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

    aux_dict = {"attn": None, "probs_active": []}
    return self.norm(x), aux_dict


def forward_backbone_no_am(self, z, x, tnc_keep_rate=None,
                            return_last_attn=False, is_distill=False):
    x, aux_dict = self.forward_features(z, x, is_distill)
    return x, aux_dict


def patch_model_no_am(model):
    """Backbone forward_features'ini AM'siz versiyonla degistirir."""
    backbone = model.backbone
    backbone.forward_features = types.MethodType(forward_features_no_am, backbone)
    backbone.forward = types.MethodType(forward_backbone_no_am, backbone)
    print("[OK] AM devre disi birakildi (monkey-patch)")


# =============================================================================
# 4) Inference forward
# =============================================================================

def inference_forward(model, template, search):
    """Training-only parametreleri atlayarak temiz inference yapar."""
    model.eval()
    with torch.no_grad():
        x, aux_dict = model.backbone(z=template, x=search)

        feat_last = x
        if isinstance(x, list):
            feat_last = x[-1]

        feat_len_s = model.feat_len_s
        enc_opt = feat_last[:, -feat_len_s:]
        opt = (enc_opt.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()
        bs, Nq, C, HW = opt.size()
        opt_feat = opt.view(-1, C, model.feat_sz_s, model.feat_sz_s)

        score_map_ctr, bbox, size_map, offset_map = model.box_head(opt_feat, None)
        outputs_coord = bbox.view(bs, Nq, 4)

    return {
        'pred_boxes': outputs_coord,
        'score_map': score_map_ctr,
        'size_map': size_map,
        'offset_map': offset_map,
    }


# =============================================================================
# 5) Pre/post processing
# =============================================================================

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess(image, size):
    """BGR -> [1, 3, H, W] normalize tensor"""
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
    img = (img - MEAN) / STD
    return torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).float()


def crop_region(frame, cx, cy, wh_area, output_size, factor):
    """Merkez + alan bazli crop. Ortalama renk ile padding."""
    crop_sz = int(np.ceil(np.sqrt(wh_area) * factor))
    crop_sz = max(crop_sz, 1)

    x1, y1 = int(cx - crop_sz / 2), int(cy - crop_sz / 2)
    x2, y2 = x1 + crop_sz, y1 + crop_sz

    H, W = frame.shape[:2]
    pads = [max(0, -y1), max(0, y2 - H), max(0, -x1), max(0, x2 - W)]

    x1c, y1c = max(0, x1), max(0, y1)
    x2c, y2c = min(W, x2), min(H, y2)
    cropped = frame[y1c:y2c, x1c:x2c]

    if any(p > 0 for p in pads):
        avg = np.mean(cropped, axis=(0, 1)).astype(np.uint8).tolist()
        cropped = cv2.copyMakeBorder(cropped, *pads, cv2.BORDER_CONSTANT, value=avg)

    resize_factor = output_size / crop_sz
    cropped = cv2.resize(cropped, (output_size, output_size), interpolation=cv2.INTER_LINEAR)
    return cropped, resize_factor, crop_sz


def decode_bbox(out, cx, cy, crop_sz, resize_factor, search_size):
    """
    pred_boxes: [cx, cy, w, h] normalized (0-1).
    Bunlari orijinal frame koordinatlarina cevirir.
    """
    pred = out['pred_boxes'].cpu().numpy()[0, 0]  # [4]
    score_map = out['score_map'].cpu().numpy()[0, 0]
    max_score = float(score_map.max())

    pred_cx = pred[0] * search_size
    pred_cy = pred[1] * search_size
    pred_w = pred[2] * search_size
    pred_h = pred[3] * search_size

    cx_new = cx + (pred_cx - search_size / 2) / resize_factor
    cy_new = cy + (pred_cy - search_size / 2) / resize_factor
    w_new = pred_w / resize_factor
    h_new = pred_h / resize_factor

    return cx_new, cy_new, w_new, h_new, max_score


# =============================================================================
# 6) Tracking dongusu
# =============================================================================

def run_tracking(model, video_source, device, cfg, label="", score_thr=0.15):
    template_size = cfg.TEST.TEMPLATE_SIZE
    search_size = cfg.TEST.SEARCH_SIZE
    template_factor = cfg.TEST.TEMPLATE_FACTOR
    search_factor = cfg.TEST.SEARCH_FACTOR

    print(f"  Template: {template_size}, Search: {search_size}")
    print(f"  T_factor: {template_factor}, S_factor: {search_factor}")

    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"[HATA] Video acilamadi: {video_source}")
        return None

    ret, frame = cap.read()
    if not ret:
        cap.release()
        return None

    win_name = f"AVTrack {label}" if label else "AVTrack Test"
    print(f"\n[{label or 'INFO'}] Hedef objeyi secin ve ENTER basin...")
    bbox_sel = cv2.selectROI(win_name, frame, fromCenter=False)
    cv2.destroyWindow(win_name)

    if bbox_sel[2] == 0 or bbox_sel[3] == 0:
        cap.release()
        return None

    x, y, w, h = bbox_sel
    cx, cy = x + w / 2, y + h / 2
    wh_area = w * h

    template_img, _, _ = crop_region(frame, cx, cy, wh_area, template_size, template_factor)
    template_tensor = preprocess(template_img, template_size).to(device)

    model.eval()
    model.to(device)

    results = []
    tracking = True

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t0 = time.time()

        if tracking:
            search_img, resize_factor, crop_sz = crop_region(
                frame, cx, cy, wh_area, search_size, search_factor
            )
            search_tensor = preprocess(search_img, search_size).to(device)
            out = inference_forward(model, template_tensor, search_tensor)
            cx, cy, w, h, score = decode_bbox(
                out, cx, cy, crop_sz, resize_factor, search_size
            )
            wh_area = w * h
            if score < score_thr:
                tracking = False
        else:
            score = 0.0

        fps = 1.0 / (time.time() - t0 + 1e-9)
        results.append((cx, cy, w, h, score, fps))

        display = frame.copy()
        if tracking and score >= score_thr:
            x1, y1 = int(cx - w / 2), int(cy - h / 2)
            x2, y2 = int(cx + w / 2), int(cy + h / 2)
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(display, f"{score:.2f}", (x1, y1 - 8),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            cv2.putText(display, "KAYIP", (50, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        info = f"{label} FPS:{fps:.0f}" if label else f"FPS:{fps:.0f}"
        cv2.putText(display, info, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        cv2.imshow(win_name, display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return results


# =============================================================================
# 7) Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='AVTrack AM-siz PC Testi')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--config', type=str, default='deit_tiny_patch16_224')
    parser.add_argument('--video', type=str, default='0')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--compare', action='store_true',
                        help='Orijinal (AM\'li) ile karsilastir')
    args = parser.parse_args()

    video_src = int(args.video) if args.video.isdigit() else args.video
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    cfg = load_config(args.config)
    model = load_model(args.checkpoint, cfg)

    if args.compare:
        model_noam = copy.deepcopy(model)
        patch_model_no_am(model_noam)

        print("\n" + "=" * 50)
        print("  KARSILASTIRMA MODU")
        print("  1) Orijinal (AM'li)")
        print("  2) AM'siz")
        print("  Ayni hedefi secin!")
        print("=" * 50)

        print("\n--- ORIJINAL ---")
        res_orig = run_tracking(model, video_src, device, cfg, label="ORIJINAL")

        print("\n--- AM'SIZ ---")
        res_noam = run_tracking(model_noam, video_src, device, cfg, label="AM-SIZ")

        if res_orig and res_noam:
            fps_o = np.mean([r[5] for r in res_orig])
            fps_n = np.mean([r[5] for r in res_noam])
            sc_o = np.mean([r[4] for r in res_orig if r[4] > 0]) if any(r[4] > 0 for r in res_orig) else 0
            sc_n = np.mean([r[4] for r in res_noam if r[4] > 0]) if any(r[4] > 0 for r in res_noam) else 0

            print("\n" + "=" * 50)
            print(f"  {'':20s} {'ORIJINAL':>12s} {'AM-SIZ':>12s}")
            print(f"  {'Ort. FPS':20s} {fps_o:>12.1f} {fps_n:>12.1f}")
            print(f"  {'Ort. Score':20s} {sc_o:>12.3f} {sc_n:>12.3f}")
            print(f"  {'Frame':20s} {len(res_orig):>12d} {len(res_noam):>12d}")
            print("=" * 50)
    else:
        patch_model_no_am(model)
        results = run_tracking(model, video_src, device, cfg, label="AM-SIZ")

        if results:
            fps_list = [r[5] for r in results]
            scores = [r[4] for r in results if r[4] > 0]
            print(f"\n[SONUC] Ort. FPS: {np.mean(fps_list):.1f}")
            if scores:
                print(f"[SONUC] Ort. Score: {np.mean(scores):.3f}")
            print(f"[SONUC] Frame: {len(results)}")


if __name__ == '__main__':
    main()