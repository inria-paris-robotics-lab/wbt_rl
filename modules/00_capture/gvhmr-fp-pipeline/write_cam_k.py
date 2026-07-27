#!/usr/bin/env python3
"""Write cam_K.txt for a clip that has no real camera (e.g. a non-Femto video),
reusing GVHMR's own estimated intrinsics (K_fullimg in hmr4d_results.pt) so
extract_hands.py / extract_hands_hamer.py / wrist_orient.py have something to
match 2D positions against. Not a real calibration, but consistent with the
image GVHMR itself was fit on -- good enough for pixel-space matching.

Run in the `holonew` env (needs torch to read the .pt).
"""

import argparse
import os
import sys

import numpy as np


def main():
    ap = argparse.ArgumentParser(description="Write cam_K.txt from GVHMR's estimated intrinsics")
    ap.add_argument("--pt", required=True, help="hmr4d_results.pt")
    ap.add_argument("--clip-dir", required=True, help="capture folder (rgb/ for W,H; cam_K.txt written here)")
    args = ap.parse_args()

    import torch
    from PIL import Image

    rgb0 = os.path.join(args.clip_dir, "rgb", "000000.png")
    w, h = Image.open(rgb0).size

    d = torch.load(args.pt, map_location="cpu", weights_only=False)
    K = d.get("K_fullimg")
    if K is None:
        sys.exit(f"[cam_k] no K_fullimg in {args.pt}")
    K = K.detach().cpu().numpy() if hasattr(K, "detach") else np.asarray(K)
    if K.ndim == 3:
        K = K[0]

    out = os.path.join(args.clip_dir, "cam_K.txt")
    np.savetxt(out, K, fmt="%.8f")
    print(f"[cam_k] {w}x{h} -> {out}:\n{K}")


if __name__ == "__main__":
    main()
