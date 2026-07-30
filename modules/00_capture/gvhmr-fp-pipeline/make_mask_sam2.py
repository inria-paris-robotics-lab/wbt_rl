#!/usr/bin/env python3
"""First-frame object mask for FoundationPose, using SAM 2 (much cleaner than the
GrabCut rectangle in make_mask.py). Prompt the object with clicks and/or a box on
frame 0; SAM 2 segments it; the binary mask (object=255) is written to
<clip>/masks/000000.png. FoundationPose only needs this one frame to initialize.

Interactive controls (a window opens on rgb/000000.png):
    LEFT click    add a POSITIVE point (part of the object)
    RIGHT click   add a NEGATIVE point (background to exclude)
    b             draw/redraw a BOX around the object (drag, then ENTER)
    u             undo last point
    r             reset (clear points + box)
    s             save mask -> masks/000000.png
    q             quit without saving
The mask recomputes after every prompt. On save it keeps the largest connected
component and fills interior holes.

Run with the sam2 env python:
    ~/.wbt_deps/miniconda3/envs/sam2/bin/python make_mask_sam2.py --clip-dir ~/wbt_clips/femtoXX
"""

import argparse
import os
import sys

import numpy as np


def _postprocess(mask):
    """Keep the largest connected component and fill interior holes."""
    import cv2
    m = (mask > 0).astype(np.uint8) * 255
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if n > 1:
        big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        m = np.where(lab == big, 255, 0).astype(np.uint8)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        filled = np.zeros_like(m)
        cv2.drawContours(filled, [max(cnts, key=cv2.contourArea)], -1, 255, -1)
        m = filled
    return m


def main():
    ap = argparse.ArgumentParser(description="First-frame object mask via SAM 2")
    ap.add_argument("--clip-dir", required=True, help="capture folder (rgb/000000.png -> masks/000000.png)")
    ap.add_argument("--frame", default="rgb/000000.png", help="frame to mask (relative to clip-dir)")
    ap.add_argument("--checkpoint", default=os.path.expanduser("~/sam2/checkpoints/sam2.1_hiera_large.pt"))
    ap.add_argument("--config", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    ap.add_argument("--max-disp", type=int, default=1600, help="max display width (clicks map back to full res)")
    args = ap.parse_args()

    import cv2
    import torch
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    img_path = os.path.join(args.clip_dir, args.frame)
    bgr = cv2.imread(img_path)
    if bgr is None:
        sys.exit(f"[sam2] cannot read {img_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    H, W = rgb.shape[:2]
    out_dir = os.path.join(args.clip_dir, "masks")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "000000.png")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[sam2] loading {os.path.basename(args.checkpoint)} on {dev} ...")
    model = build_sam2(args.config, args.checkpoint, device=dev)
    pred = SAM2ImagePredictor(model)
    with torch.inference_mode(), torch.autocast(dev, dtype=torch.bfloat16, enabled=(dev == "cuda")):
        pred.set_image(rgb)  # embed once; predict() calls below are fast
    print(f"[sam2] image embedded ({W}x{H}). Click the object; 's' to save, 'q' to quit.")

    pts, labs = [], []          # click prompts (full-res coords) + labels (1 pos / 0 neg)
    box = None                  # [x0,y0,x1,y1] full-res
    mask = np.zeros((H, W), np.uint8)

    scale = min(1.0, args.max_disp / float(W))

    def recompute():
        nonlocal mask
        if not pts and box is None:
            mask = np.zeros((H, W), np.uint8)
            return
        pc = np.array(pts, np.float32) if pts else None
        pl = np.array(labs, np.int32) if labs else None
        bx = np.array(box, np.float32)[None] if box is not None else None
        multi = box is None and len(pts) <= 1     # let SAM pick best mask for a lone click
        with torch.inference_mode(), torch.autocast(dev, dtype=torch.bfloat16, enabled=(dev == "cuda")):
            masks, scores, _ = pred.predict(point_coords=pc, point_labels=pl, box=bx,
                                            multimask_output=multi)
        idx = int(np.argmax(scores)) if multi else 0
        mask = (masks[idx] > 0).astype(np.uint8) * 255

    state = {"dirty": True, "lastkey": "-"}    # dirty = needs redraw; lastkey = HUD debug

    def render():
        vis = bgr.copy()
        red = np.zeros_like(vis); red[..., 2] = 255
        sel = mask > 0
        vis[sel] = (0.55 * vis[sel] + 0.45 * red[sel]).astype(np.uint8)
        if box is not None:
            cv2.rectangle(vis, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 255), 2)
        for (x, y), l in zip(pts, labs):
            cv2.circle(vis, (int(x), int(y)), 6, (0, 200, 0) if l == 1 else (0, 0, 255), -1)
            cv2.circle(vis, (int(x), int(y)), 6, (255, 255, 255), 1)
        px = int(sel.sum())
        for i, txt in enumerate((
                f"L=+pt  R=-pt  b=box  u=undo  r=reset  s=SAVE  q/Esc=quit   [{px} px]",
                f"last key: {state['lastkey']}   (if this doesn't change when you type, the window isn't focused)")):
            y0 = 30 + i * 30
            cv2.putText(vis, txt, (15, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
            cv2.putText(vis, txt, (15, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)
        if scale < 1.0:
            vis = cv2.resize(vis, None, fx=scale, fy=scale)
        return vis

    win = "SAM2 mask"
    cv2.namedWindow(win)

    def on_mouse(event, x, y, flags, _):
        if event not in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN):
            return
        fx, fy = x / scale, y / scale     # back to full-res
        pts.append((fx, fy))
        labs.append(1 if event == cv2.EVENT_LBUTTONDOWN else 0)
        recompute()
        state["dirty"] = True

    cv2.setMouseCallback(win, on_mouse)

    while True:
        if state["dirty"]:
            cv2.imshow(win, render())    # redraw ONLY on change, so Qt can deliver key events
            state["dirty"] = False
        k = cv2.waitKey(30)
        if k == -1:                      # no key this tick
            if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                print("[sam2] window closed — quit without saving")
                break
            continue
        k &= 0xFF
        ch = chr(k).lower() if 32 <= k < 127 else ""   # case-insensitive (Shift/CapsLock safe)
        state["lastkey"] = ch if ch else str(k)
        state["dirty"] = True
        if ch == "q" or k == 27:         # q or Esc
            print("[sam2] quit without saving")
            break
        elif ch == "s":
            if mask.sum() == 0:
                print("[sam2] empty mask — prompt the object first")
                continue
            final = _postprocess(mask)
            cv2.imwrite(out_path, final)
            print(f"[sam2] saved {out_path}  ({int((final > 0).sum())} px)")
            break
        elif ch == "u" and pts:
            pts.pop(); labs.pop(); recompute()
        elif ch == "r":
            pts.clear(); labs.clear(); box = None; recompute()
        elif ch == "b":
            r = cv2.selectROI(win, render(), showCrosshair=False)
            x, y, w, h = r
            if w > 3 and h > 3:
                box = [x / scale, y / scale, (x + w) / scale, (y + h) / scale]
                recompute()

    cv2.destroyAllWindows()
    cv2.waitKey(1)     # flush the Qt close event so the window actually disappears


if __name__ == "__main__":
    main()
