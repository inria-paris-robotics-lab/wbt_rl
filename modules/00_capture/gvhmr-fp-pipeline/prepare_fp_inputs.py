#!/usr/bin/env python3
"""Build the data folder FoundationPose expects from a video, reusing GVHMR's
estimated intrinsics so body and object share the same camera.

Layout produced (as expected by FoundationPose run_demo.py):
    <out>/
        rgb/000000.png ...      color frames
        depth/000000.png ...    metric depth (uint16 mm)   [optional here]
        masks/000000.png        object mask, first frame   [optional here]
        mesh/<object>.obj       the CAD, in METERS
        cam_K.txt               3x3 intrinsics

This script lays out the structure and copies the CAD + intrinsics. Metric depth
(Depth Anything V2 / RGB-D sensor) and the SAM2 mask are optional inputs, passed
via --depth-dir / --mask (see README).
"""

import argparse
import os
import shutil
import sys

import numpy as np


def extract_frames(video, out_dir):
    import cv2
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        sys.exit(f"[prep] cannot open {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        cv2.imwrite(os.path.join(out_dir, f"{i:06d}.png"), frame)
        i += 1
    cap.release()
    print(f"[prep] {i} frames extracted -> {out_dir} (fps~{fps:.3f})")
    return i, fps


def write_intrinsics(path, K=None, width=None, height=None, focal=None):
    """Write cam_K.txt. Prefer GVHMR's K; otherwise a heuristic focal."""
    if K is not None:
        K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    else:
        # heuristic: focal ~= 1.1 * max(W,H), principal point at image center
        f = focal if focal else 1.1 * max(width, height)
        K = np.array([[f, 0, width / 2.0], [0, f, height / 2.0], [0, 0, 1.0]])
        print("[prep] heuristic intrinsics (no GVHMR K given); "
              "a wrong focal degrades the object pose.", file=sys.stderr)
    np.savetxt(path, K, fmt="%.8f")
    print(f"[prep] cam_K.txt written:\n{K}")


def load_gvhmr_K(pt_path):
    import torch
    d = torch.load(pt_path, map_location="cpu", weights_only=False)
    K = d.get("K_fullimg")
    if K is None:
        return None
    K = K.detach().cpu().numpy() if hasattr(K, "detach") else np.asarray(K)
    # K_fullimg can be (T,3,3) or (3,3) depending on version -> take the first
    if K.ndim == 3:
        K = K[0]
    return K


def main():
    ap = argparse.ArgumentParser(description="Prepare FoundationPose inputs")
    ap.add_argument("--video", required=True)
    ap.add_argument("--object-mesh", required=True, help="object CAD, in METERS")
    ap.add_argument("--out", required=True, help="FoundationPose data folder")
    ap.add_argument("--gvhmr-pt", default="", help="hmr4d_results.pt (to reuse K)")
    ap.add_argument("--focal", type=float, default=0.0, help="focal px if no GVHMR K")
    ap.add_argument("--depth-dir", default="", help="ready metric depth folder (optional)")
    ap.add_argument("--mask", default="", help="first-frame object mask png (optional)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    for sub in ("rgb", "depth", "masks", "mesh"):
        os.makedirs(os.path.join(args.out, sub), exist_ok=True)

    n, fps = extract_frames(args.video, os.path.join(args.out, "rgb"))

    import cv2
    h, w = cv2.imread(os.path.join(args.out, "rgb", "000000.png")).shape[:2]

    K = load_gvhmr_K(args.gvhmr_pt) if args.gvhmr_pt else None
    write_intrinsics(os.path.join(args.out, "cam_K.txt"), K=K, width=w, height=h,
                     focal=(args.focal or None))

    shutil.copy(args.object_mesh, os.path.join(args.out, "mesh", os.path.basename(args.object_mesh)))
    print("[prep] CAD copied. Must be in METERS, or scale won't match SMPL.")

    if args.depth_dir:
        for f in sorted(os.listdir(args.depth_dir)):
            shutil.copy(os.path.join(args.depth_dir, f), os.path.join(args.out, "depth", f))
        print(f"[prep] depth copied from {args.depth_dir}")
    else:
        print("[prep] depth/ empty. FoundationPose (RGB-D) needs metric depth:\n"
              "        - RGB-D sensor / iPhone LiDAR, or\n"
              "        - Depth Anything V2 (metric) / UniDepth / Metric3D, uint16 mm.\n"
              "        See README, Depth section.")

    if args.mask:
        shutil.copy(args.mask, os.path.join(args.out, "masks", "000000.png"))
        print("[prep] first-frame mask copied.")
    else:
        print("[prep] masks/ empty. Make the object mask on the first frame "
              "(SAM 2, click the object) -> masks/000000.png. See README.")

    print(f"[prep] FoundationPose folder ready: {args.out}  ({n} frames, {w}x{h}, fps~{fps:.2f})")


if __name__ == "__main__":
    main()
