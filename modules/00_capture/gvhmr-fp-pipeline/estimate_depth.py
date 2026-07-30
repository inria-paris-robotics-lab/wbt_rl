#!/usr/bin/env python3
"""RGB-only fallback: estimate metric per-frame depth with Depth Anything V2
(metric) and write it as FoundationPose expects: uint16 PNG in millimeters
(0 = invalid), same names as the rgb/ frames. Feed the output to
prepare_fp_inputs.py via --depth-dir.

Mono depth is metric but approximate (absolute scale depends on model/domain).
Prefer real RGB-D when available -> much more stable object pose. Pick the
indoor/outdoor variant to match your scene.
"""

import argparse
import os
import sys

import numpy as np


def main():
    ap = argparse.ArgumentParser(description="Metric depth (Depth Anything V2) for FoundationPose")
    ap.add_argument("--rgb-dir", required=True, help="rgb/ frames folder (NNNNNN.png)")
    ap.add_argument("--out-dir", required=True, help="output depth/ folder (uint16 mm)")
    ap.add_argument("--model", default="depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf",
                    help="HF model id; ...-Outdoor-... for outdoor scenes")
    ap.add_argument("--max-mm", type=int, default=65535, help="uint16 clamp (mm)")
    args = ap.parse_args()

    try:
        import torch
        from PIL import Image
        from transformers import pipeline
    except ImportError:
        sys.exit("[depth] needs: pip install torch transformers pillow")

    os.makedirs(args.out_dir, exist_ok=True)
    device = 0 if torch.cuda.is_available() else -1
    pipe = pipeline("depth-estimation", model=args.model, device=device)

    frames = sorted(f for f in os.listdir(args.rgb_dir) if f.lower().endswith((".png", ".jpg", ".jpeg")))
    if not frames:
        sys.exit(f"[depth] no frames in {args.rgb_dir}")

    for i, name in enumerate(frames):
        img = Image.open(os.path.join(args.rgb_dir, name)).convert("RGB")
        pred = pipe(img)["predicted_depth"]           # meters (Tensor or array)
        depth_m = pred.squeeze().cpu().numpy() if hasattr(pred, "cpu") else np.asarray(pred).squeeze()
        depth_mm = np.clip(depth_m * 1000.0, 0, args.max_mm).astype(np.uint16)

        out_name = os.path.splitext(name)[0] + ".png"
        Image.fromarray(depth_mm).save(os.path.join(args.out_dir, out_name))
        if i % 50 == 0:
            print(f"[depth] {i+1}/{len(frames)}  (min={depth_m.min():.2f}m max={depth_m.max():.2f}m)")

    print(f"[depth] OK -> {args.out_dir} ({len(frames)} maps, uint16 mm). "
          "Pass this folder to prepare_fp_inputs.py --depth-dir.")


if __name__ == "__main__":
    main()
