#!/usr/bin/env python3
"""Extract per-frame hand orientation (camera frame) from MediaPipe 2D landmarks
+ the metric depth.

MediaPipe gives reliable 2D hand landmarks; its own 3D is weak and PnP on a nearly
planar hand flips. Instead we back-project 4 palm landmarks (wrist, index/middle/
pinky MCP) with the RGB-D depth and build the palm frame with the SAME formula as
the SMPL side (y=fingers, z=normal, x=across) -> a metric, unambiguous, sign-
consistent rotation in the camera frame. Feeds wrist_orient.py.

Saves <out>.npz: R (T,2,3,3) palm->camera, wrist2d (T,2,2), label (T,2) 0=Left/1=Right,
valid (T,2). Optional --save-vis draws the palm axes for a sanity check.

Run in the `mp` env (mediapipe + opencv + numpy); needs depth/ in the clip.

--no-depth: skip the depth-based palm normal, only detect presence (bbox +
wrist2d + label). R stays identity -- fine for extract_hands_hamer.py, which
ignores this file's R and rebuilds orientation from HaMeR's own 3D joints.
Use when there's no metric depth (e.g. a non-Femto video).
"""

import argparse
import glob
import os

import numpy as np

DEFAULT_MODEL = os.path.expanduser("~/wbt_clips/hand_landmarker.task")


def deproject(u, v, depth, fx, fy, cx, cy, win=3):
    """Pixel (u,v) + depth window -> 3D point in the camera frame (meters), or None."""
    H, W = depth.shape
    ui, vi = int(round(u)), int(round(v))
    if not (0 <= ui < W and 0 <= vi < H):
        return None
    patch = depth[max(0, vi - win):vi + win + 1, max(0, ui - win):ui + win + 1]
    vals = patch[patch > 0]
    if vals.size < 3:
        return None
    Z = float(np.median(vals)) / 1000.0                     # mm -> m
    return np.array([(u - cx) / fx * Z, (v - cy) / fy * Z, Z])


def main():
    ap = argparse.ArgumentParser(description="Depth-based MediaPipe hand orientation")
    ap.add_argument("--clip-dir", required=True, help="capture folder (rgb/ + depth/ + cam_K.txt)")
    ap.add_argument("--out", default="", help="output npz (default <clip>/hands.npz)")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="hand_landmarker.task path")
    ap.add_argument("--save-vis", action="store_true", help="write axis-overlay frames to hands_vis/")
    ap.add_argument("--min-conf", type=float, default=0.5)
    ap.add_argument("--no-depth", action="store_true",
                    help="no metric depth: only bbox/wrist2d/label, R stays identity")
    args = ap.parse_args()

    import cv2
    import mediapipe as mp
    from mediapipe.tasks import python as mpp
    from mediapipe.tasks.python import vision
    from PIL import Image

    rgb = sorted(glob.glob(os.path.join(args.clip_dir, "rgb", "*.png")))
    dep = [] if args.no_depth else sorted(glob.glob(os.path.join(args.clip_dir, "depth", "*.png")))
    if not rgb or (not args.no_depth and not dep):
        raise SystemExit(f"[hands] need rgb/ (+ depth/ unless --no-depth) in {args.clip_dir}")
    K = np.loadtxt(os.path.join(args.clip_dir, "cam_K.txt"))
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    T = len(rgb) if args.no_depth else min(len(rgb), len(dep))
    H, W = cv2.imread(rgb[0]).shape[:2]
    out = args.out or os.path.join(args.clip_dir, "hands.npz")
    vis_dir = os.path.join(args.clip_dir, "hands_vis")
    if args.save_vis:
        os.makedirs(vis_dir, exist_ok=True)

    opts = vision.HandLandmarkerOptions(
        base_options=mpp.BaseOptions(model_asset_path=args.model), num_hands=2,
        running_mode=vision.RunningMode.VIDEO,               # temporal tracking -> steadier landmarks
        min_hand_detection_confidence=args.min_conf,
        min_hand_presence_confidence=args.min_conf, min_tracking_confidence=args.min_conf)
    landmarker = vision.HandLandmarker.create_from_options(opts)

    R_all = np.tile(np.eye(3), (T, 2, 1, 1)).astype(np.float32)
    wrist2d = np.zeros((T, 2, 2), np.float32)
    box = np.zeros((T, 2, 4), np.float32)              # hand bbox xyxy (for HaMeR crops)
    label = -np.ones((T, 2), np.int8)
    valid = np.zeros((T, 2), bool)

    def proj(p):
        return (int(fx * p[0] / p[2] + cx), int(fy * p[1] / p[2] + cy))

    for t in range(T):
        img = cv2.imread(rgb[t])
        depth = None if args.no_depth else np.asarray(Image.open(dep[t]))  # uint16 mm, aligned to color
        rgb_img = np.ascontiguousarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        res = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_img),
                                          int(t * 1000 / 30))
        for s in range(min(2, len(res.hand_landmarks))):
            lm = res.hand_landmarks[s]
            allxy = np.array([(l.x * W, l.y * H) for l in lm])
            if args.no_depth:                                  # presence only, R stays identity
                wrist2d[t, s] = (lm[0].x * W, lm[0].y * H)
                box[t, s] = [allxy[:, 0].min(), allxy[:, 1].min(), allxy[:, 0].max(), allxy[:, 1].max()]
                label[t, s] = 0 if res.handedness[s][0].category_name == "Left" else 1
                valid[t, s] = True
                continue
            # palm normal: robust plane fit over ALL depth pixels inside the palm polygon
            poly = np.array([[lm[i].x * W, lm[i].y * H] for i in (0, 1, 5, 9, 13, 17)], np.float32)
            mask = np.zeros((H, W), np.uint8)
            cv2.fillConvexPoly(mask, cv2.convexHull(poly).astype(np.int32), 1)
            ys, xs = np.where((mask > 0) & (depth > 0))
            if xs.size < 30:
                continue
            Zc = depth[ys, xs].astype(np.float64) / 1000.0
            pts = np.stack([(xs - cx) / fx * Zc, (ys - cy) / fy * Zc, Zc], 1)
            c = pts.mean(0); nrm = np.linalg.svd(pts - c)[2][2]
            keep = np.abs((pts - c) @ nrm) < 2 * (np.std((pts - c) @ nrm) + 1e-6)   # drop outliers
            c = pts[keep].mean(0); nrm = np.linalg.svd(pts[keep] - c)[2][2]
            # anatomical frame (4 landmarks) for the sign + fingers direction
            P = [deproject(lm[i].x * W, lm[i].y * H, depth, fx, fy, cx, cy) for i in (0, 5, 9, 17)]
            if any(p is None for p in P):
                continue
            wrist, idx, mid, pky = P
            y = mid - wrist; y /= np.linalg.norm(y) + 1e-9
            if nrm @ np.cross(y, pky - idx) < 0:
                nrm = -nrm                                             # match anatomical sign
            z = nrm
            y = y - (y @ z) * z; y /= np.linalg.norm(y) + 1e-9         # fingers, perp to normal
            R_all[t, s] = np.column_stack([np.cross(y, z), y, z])
            wrist2d[t, s] = (lm[0].x * W, lm[0].y * H)
            box[t, s] = [allxy[:, 0].min(), allxy[:, 1].min(), allxy[:, 0].max(), allxy[:, 1].max()]
            label[t, s] = 0 if res.handedness[s][0].category_name == "Left" else 1
            valid[t, s] = True
            if args.save_vis:
                o = proj(wrist)
                for axis, col in ((x, (0, 0, 255)), (y, (0, 255, 0)), (z, (255, 0, 0))):
                    cv2.line(img, o, proj(wrist + 0.1 * axis), col, 3)
        if args.save_vis:
            cv2.imwrite(os.path.join(vis_dir, f"{t:06d}.png"), img)
        if t % 30 == 0:
            print(f"[hands] frame {t}/{T}  (hands: {int(valid[t].sum())})")

    landmarker.close()
    np.savez_compressed(out, R=R_all, wrist2d=wrist2d, box=box, label=label, valid=valid,
                        K=K.astype(np.float32))
    print(f"[hands] wrote {out}  | frames with >=1 hand: {int(valid.any(1).sum())}/{T}")


if __name__ == "__main__":
    main()
