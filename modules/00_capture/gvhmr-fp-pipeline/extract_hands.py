#!/usr/bin/env python3
"""Extract per-frame hand orientation (in the camera frame) with MediaPipe Hands.

For each frame and each detected hand we solve the hand pose by PnP between
MediaPipe's 3D hand landmarks and their 2D image positions using the real camera
K, so the resulting rotation is directly in the same camera frame as GVHMR's
smpl_params_incam. Feeds wrist_orient.py to correct the SMPL-X wrist.

Saves <out>.npz: R (T,2,3,3) hand->camera, wrist2d (T,2,2), label (T,2) 0=Left/1=Right,
valid (T,2). Slots 0/1 are detection order; wrist_orient matches them to the SMPL
wrists by 2D proximity. Optional --save-vis draws the hand axes for a sanity check.

Uses the MediaPipe Tasks API (mediapipe 0.10.x). Run in the `mp` env.
"""

import argparse
import glob
import os

import numpy as np

DEFAULT_MODEL = os.path.expanduser("~/wbt_clips/hand_landmarker.task")


def main():
    ap = argparse.ArgumentParser(description="MediaPipe hand orientation extractor")
    ap.add_argument("--clip-dir", required=True, help="capture folder (rgb/ + cam_K.txt)")
    ap.add_argument("--out", default="", help="output npz (default <clip>/hands.npz)")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="hand_landmarker.task path")
    ap.add_argument("--save-vis", action="store_true", help="write axis-overlay frames to hands_vis/")
    ap.add_argument("--min-conf", type=float, default=0.5)
    args = ap.parse_args()

    import cv2
    import mediapipe as mp
    from mediapipe.tasks import python as mpp
    from mediapipe.tasks.python import vision

    rgb = sorted(glob.glob(os.path.join(args.clip_dir, "rgb", "*.png")))
    if not rgb:
        raise SystemExit(f"[hands] no frames in {args.clip_dir}/rgb")
    K = np.loadtxt(os.path.join(args.clip_dir, "cam_K.txt"))
    T = len(rgb)
    H, W = cv2.imread(rgb[0]).shape[:2]
    out = args.out or os.path.join(args.clip_dir, "hands.npz")
    vis_dir = os.path.join(args.clip_dir, "hands_vis")
    if args.save_vis:
        os.makedirs(vis_dir, exist_ok=True)

    opts = vision.HandLandmarkerOptions(
        base_options=mpp.BaseOptions(model_asset_path=args.model), num_hands=2,
        running_mode=vision.RunningMode.IMAGE,
        min_hand_detection_confidence=args.min_conf,
        min_hand_presence_confidence=args.min_conf,
        min_tracking_confidence=args.min_conf)
    landmarker = vision.HandLandmarker.create_from_options(opts)

    R_all = np.tile(np.eye(3), (T, 2, 1, 1)).astype(np.float32)
    wrist2d = np.zeros((T, 2, 2), np.float32)
    label = -np.ones((T, 2), np.int8)          # 0 = Left, 1 = Right
    valid = np.zeros((T, 2), bool)
    axes3d = np.float32([[0, 0, 0], [0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]])

    for t, f in enumerate(rgb):
        img = cv2.imread(f)
        rgb_img = np.ascontiguousarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        res = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_img))
        for s in range(min(2, len(res.hand_landmarks))):
            p2d = np.float32([[l.x * W, l.y * H] for l in res.hand_landmarks[s]])
            p3d = np.float32([[l.x, l.y, l.z] for l in res.hand_world_landmarks[s]])
            ok, rvec, tvec = cv2.solvePnP(p3d, p2d, K, None, flags=cv2.SOLVEPNP_SQPNP)
            if not ok:
                continue
            R_all[t, s] = cv2.Rodrigues(rvec)[0]
            wrist2d[t, s] = p2d[0]
            label[t, s] = 0 if res.handedness[s][0].category_name == "Left" else 1
            valid[t, s] = True
            if args.save_vis:
                proj, _ = cv2.projectPoints(axes3d, rvec, tvec, K, None)
                o, x, y, z = proj.reshape(-1, 2).astype(int)
                for end, col in ((x, (0, 0, 255)), (y, (0, 255, 0)), (z, (255, 0, 0))):
                    cv2.line(img, tuple(o), tuple(end), col, 3)
        if args.save_vis:
            cv2.imwrite(os.path.join(vis_dir, f"{t:06d}.png"), img)
        if t % 30 == 0:
            print(f"[hands] frame {t}/{T}  (hands: {int(valid[t].sum())})")

    landmarker.close()
    np.savez_compressed(out, R=R_all, wrist2d=wrist2d, label=label, valid=valid,
                        K=K.astype(np.float32))
    print(f"[hands] wrote {out}  | frames with >=1 hand: {int(valid.any(1).sum())}/{T}")


if __name__ == "__main__":
    main()
