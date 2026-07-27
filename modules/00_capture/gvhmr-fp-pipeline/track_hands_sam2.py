#!/usr/bin/env python3
"""Track hand regions across a whole clip with SAM 2's VIDEO predictor, for
when MediaPipe's per-frame detector fails (e.g. stylized game footage where
hands don't look like real human hands -- see extract_hands.py --no-depth).

You click each hand once (points and/or a box) on one frame; SAM 2 segments
+ tracks it forward AND backward across the whole clip -- no per-frame
hand-shape recognition involved, so it doesn't care that the content is
synthetic. Output: <clip>/hands.npz in the same schema as extract_hands.py
(R stays identity -- extract_hands_hamer.py ignores it and rebuilds
orientation from HaMeR's own 3D joints; box/wrist2d/label/valid are what it
actually needs).

Interactive controls (window on the chosen annotation frame):
    TAB           switch active hand (Left / Right)
    LEFT click    add a POSITIVE point for the active hand
    RIGHT click   add a NEGATIVE point for the active hand
    b             draw/redraw a BOX for the active hand (drag, then ENTER)
    u             undo last point (active hand)
    r             reset active hand's prompts
    s             save prompts and track across the whole video
    q             quit without tracking

Run with the sam2 env python:
    ~/.wbt_deps/miniconda3/envs/sam2/bin/python track_hands_sam2.py --clip-dir ~/wbt_clips/femtoXX
"""

import argparse
import glob
import os
import shutil
import sys

import numpy as np

HANDS = ("L", "R")                                        # slot 0 / 1, matches extract_hands.py label convention
COLORS = {"L": (0, 200, 0), "R": (0, 128, 255)}


def _mask_bbox_center(mask):
    ys, xs = np.where(mask)
    if xs.size == 0:
        return None
    return (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())), \
           (float(xs.mean()), float(ys.mean()))


def main():
    ap = argparse.ArgumentParser(description="Track hands across a clip with SAM 2 (video)")
    ap.add_argument("--clip-dir", required=True, help="capture folder (rgb/ + cam_K.txt)")
    ap.add_argument("--ann-frame", type=int, default=0, help="frame index to prompt (rgb/NNNNNN.png)")
    ap.add_argument("--out", default="", help="output npz (default <clip>/hands.npz)")
    ap.add_argument("--checkpoint", default=os.path.expanduser("~/sam2/checkpoints/sam2.1_hiera_large.pt"))
    ap.add_argument("--config", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    ap.add_argument("--max-disp", type=int, default=1600, help="max display width (clicks map back to full res)")
    ap.add_argument("--keep-jpg", action="store_true", help="keep the re-encoded jpg mirror after saving")
    args = ap.parse_args()

    import cv2
    import torch
    from sam2.build_sam import build_sam2_video_predictor

    rgb = sorted(glob.glob(os.path.join(args.clip_dir, "rgb", "*.png")))
    if not rgb:
        sys.exit(f"[sam2-hands] no rgb/ frames in {args.clip_dir}")
    if not (0 <= args.ann_frame < len(rgb)):
        sys.exit(f"[sam2-hands] --ann-frame out of range (0..{len(rgb) - 1})")
    K = np.loadtxt(os.path.join(args.clip_dir, "cam_K.txt"))
    out_path = args.out or os.path.join(args.clip_dir, "hands.npz")

    # SAM 2's video loader only reads JPEG folders (or decodes mp4 itself, independently of
    # our rgb/ extraction -> frame count/timing could drift). Re-encode rgb/*.png to a jpg
    # mirror so indices line up EXACTLY 1:1 with rgb/, which extract_hands_hamer.py assumes.
    jpg_dir = os.path.join(args.clip_dir, ".sam2_jpg")
    os.makedirs(jpg_dir, exist_ok=True)
    print(f"[sam2-hands] re-encoding {len(rgb)} frames -> {jpg_dir} (jpg mirror for SAM 2) ...")
    for i, p in enumerate(rgb):
        jp = os.path.join(jpg_dir, f"{i:05d}.jpg")
        if not os.path.exists(jp):
            cv2.imwrite(jp, cv2.imread(p), [cv2.IMWRITE_JPEG_QUALITY, 95])

    try:
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[sam2-hands] loading {os.path.basename(args.checkpoint)} on {dev} ...")
        predictor = build_sam2_video_predictor(args.config, args.checkpoint, device=dev)
        # offload frames to CPU: a 1000+ frame clip at the model's 1024x1024 internal res
        # would otherwise blow past GPU VRAM (~12MB/frame resident).
        infer_state = predictor.init_state(video_path=jpg_dir, offload_video_to_cpu=True)

        bgr0 = cv2.imread(rgb[args.ann_frame])
        H, W = bgr0.shape[:2]
        scale = min(1.0, args.max_disp / float(W))

        pts = {h: [] for h in HANDS}      # full-res (x,y)
        labs = {h: [] for h in HANDS}     # 1 pos / 0 neg
        boxes = {h: None for h in HANDS}
        ui = {"active": "L", "dirty": True}

        win = "SAM2 hands (TAB=switch hand, s=track, q=quit)"
        cv2.namedWindow(win)

        def on_mouse(event, x, y, flags, _):
            if event not in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN):
                return
            fx, fy = x / scale, y / scale
            a = ui["active"]
            pts[a].append((fx, fy))
            labs[a].append(1 if event == cv2.EVENT_LBUTTONDOWN else 0)
            ui["dirty"] = True

        cv2.setMouseCallback(win, on_mouse)

        def render():
            vis = bgr0.copy()
            for h in HANDS:
                for (x, y), l in zip(pts[h], labs[h]):
                    col = COLORS[h] if l == 1 else (0, 0, 255)
                    cv2.circle(vis, (int(x), int(y)), 6, col, -1)
                    cv2.circle(vis, (int(x), int(y)), 6, (255, 255, 255), 1)
                if boxes[h] is not None:
                    x0, y0, x1, y1 = boxes[h]
                    cv2.rectangle(vis, (int(x0), int(y0)), (int(x1), int(y1)), COLORS[h], 2)
            txt = (f"active={ui['active']}  L: {len(pts['L'])}pt box={boxes['L'] is not None}   "
                   f"R: {len(pts['R'])}pt box={boxes['R'] is not None}   TAB=switch s=track q=quit")
            cv2.putText(vis, txt, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
            cv2.putText(vis, txt, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 1)
            if scale < 1.0:
                vis = cv2.resize(vis, None, fx=scale, fy=scale)
            return vis

        while True:
            if ui["dirty"]:
                cv2.imshow(win, render())
                ui["dirty"] = False
            k = cv2.waitKey(30)
            if k == -1:
                if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                    sys.exit("[sam2-hands] window closed -- quit without tracking")
                continue
            k &= 0xFF
            ch = chr(k).lower() if 32 <= k < 127 else ""
            ui["dirty"] = True
            if ch == "q" or k == 27:
                sys.exit("[sam2-hands] quit without tracking")
            elif k == 9:                                    # TAB
                ui["active"] = "R" if ui["active"] == "L" else "L"
            elif ch == "u" and pts[ui["active"]]:
                pts[ui["active"]].pop(); labs[ui["active"]].pop()
            elif ch == "r":
                a = ui["active"]; pts[a].clear(); labs[a].clear(); boxes[a] = None
            elif ch == "b":
                r = cv2.selectROI(win, render(), showCrosshair=False)
                x, y, w, h = r
                if w > 3 and h > 3:
                    boxes[ui["active"]] = [x / scale, y / scale, (x + w) / scale, (y + h) / scale]
            elif ch == "s":
                if not any(pts[h] or boxes[h] for h in HANDS):
                    print("[sam2-hands] no prompts on either hand"); continue
                break

        cv2.destroyAllWindows(); cv2.waitKey(1)

        for i, h in enumerate(HANDS):
            if not (pts[h] or boxes[h]):
                continue
            pc = np.array(pts[h], np.float32) if pts[h] else None
            pl = np.array(labs[h], np.int32) if labs[h] else None
            bx = np.array(boxes[h], np.float32) if boxes[h] is not None else None
            predictor.add_new_points_or_box(inference_state=infer_state, frame_idx=args.ann_frame,
                                            obj_id=i, points=pc, labels=pl, box=bx)

        T = len(rgb)
        R_all = np.tile(np.eye(3), (T, 2, 1, 1)).astype(np.float32)
        wrist2d = np.zeros((T, 2, 2), np.float32)
        box_out = np.zeros((T, 2, 4), np.float32)
        label = -np.ones((T, 2), np.int8)
        valid = np.zeros((T, 2), bool)

        def consume(frame_idx, out_obj_ids, mask_logits):
            for oid, logits in zip(out_obj_ids, mask_logits):
                slot = oid
                m = (logits > 0).cpu().numpy()
                if m.ndim == 3:
                    m = m[0]
                r = _mask_bbox_center(m)
                if r is None:
                    continue
                bb, ctr = r
                box_out[frame_idx, slot] = bb
                wrist2d[frame_idx, slot] = ctr
                label[frame_idx, slot] = slot
                valid[frame_idx, slot] = True

        print("[sam2-hands] propagating forward ...")
        for frame_idx, out_obj_ids, mask_logits in predictor.propagate_in_video(infer_state):
            consume(frame_idx, out_obj_ids, mask_logits)
        print("[sam2-hands] propagating backward ...")
        for frame_idx, out_obj_ids, mask_logits in predictor.propagate_in_video(infer_state, reverse=True):
            consume(frame_idx, out_obj_ids, mask_logits)

        np.savez_compressed(out_path, R=R_all, wrist2d=wrist2d, box=box_out, label=label,
                            valid=valid, K=K.astype(np.float32))
        print(f"[sam2-hands] wrote {out_path} | frames with >=1 hand: {int(valid.any(1).sum())}/{T}")
    finally:
        if not args.keep_jpg:
            shutil.rmtree(jpg_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
