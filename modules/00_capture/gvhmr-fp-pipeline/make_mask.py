#!/usr/bin/env python3
"""Make the first-frame object mask FoundationPose needs, without SAM.

Draw a rectangle around the object on frame 0; GrabCut segments it; the binary
mask (object=255) is written to <clip>/masks/000000.png. FoundationPose only
needs this one frame to initialize, then it tracks the object.

Deps: opencv-python + numpy (base env). Needs a display.
Usage: python make_mask.py --clip-dir ~/wbt_clips/femto02
"""

import argparse
import os
import sys

import numpy as np


def main():
    ap = argparse.ArgumentParser(description="First-frame object mask via GrabCut")
    ap.add_argument("--clip-dir", required=True, help="capture folder (rgb/000000.png -> masks/000000.png)")
    ap.add_argument("--frame", default="rgb/000000.png", help="frame to mask (relative to clip-dir)")
    ap.add_argument("--iters", type=int, default=6, help="GrabCut iterations")
    args = ap.parse_args()

    import cv2

    img_path = os.path.join(args.clip_dir, args.frame)
    img = cv2.imread(img_path)
    if img is None:
        sys.exit(f"[mask] cannot read {img_path}")
    out_dir = os.path.join(args.clip_dir, "masks")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "000000.png")

    while True:
        rect = cv2.selectROI("Draw a box around the OBJECT, then ENTER (c=cancel)", img,
                             showCrosshair=False)
        cv2.destroyWindow("Draw a box around the OBJECT, then ENTER (c=cancel)")
        x, y, w, h = rect
        if w < 5 or h < 5:
            print("[mask] cancelled (no box)"); return

        mask = np.zeros(img.shape[:2], np.uint8)
        bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
        cv2.grabCut(img, mask, (x, y, w, h), bgd, fgd, args.iters, cv2.GC_INIT_WITH_RECT)
        fg = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)

        # keep the largest connected component (drop stray specks)
        n, lab, stats, _ = cv2.connectedComponentsWithStats(fg, 8)
        if n > 1:
            big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            fg = np.where(lab == big, 255, 0).astype(np.uint8)

        overlay = img.copy()
        overlay[fg > 0] = (0, 0, 255)
        vis = cv2.addWeighted(img, 0.6, overlay, 0.4, 0)
        cv2.putText(vis, "s=save  r=redo  q=quit", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.imshow("mask preview", vis)
        k = cv2.waitKey(0) & 0xFF
        cv2.destroyAllWindows()
        if k == ord("s"):
            cv2.imwrite(out_path, fg)
            print(f"[mask] saved {out_path}  ({int((fg > 0).sum())} px)")
            return
        if k == ord("q"):
            print("[mask] quit without saving"); return
        # otherwise loop to redraw


if __name__ == "__main__":
    main()
