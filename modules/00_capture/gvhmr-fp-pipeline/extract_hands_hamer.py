#!/usr/bin/env python3
"""Hand orientation from HaMeR (robust, RGB-based), fed by MediaPipe hand boxes.

MediaPipe (extract_hands.py) gives reliable hand presence + boxes; HaMeR gives a
robust hand mesh from each crop. We match each detected hand to the nearest GVHMR
wrist (unambiguous L/R + rejects false positives), run HaMeR on the box, and build
the palm frame from HaMeR's MANO joints in the full-image camera frame. Output is
the same hands.npz format so wrist_orient.py is reused unchanged.

Run in the `hamer` env (from the hamer repo dir, needs _DATA + smplx + GVHMR pt).
"""

import argparse
import glob
import os

import numpy as np

HAMER_DIR = os.path.expanduser("~/hamer")
JW = {"L": 20, "R": 21}                                # SMPL-X wrist joint indices


def palm(w, i, m, p):
    y = m - w; y /= np.linalg.norm(y) + 1e-9
    z = np.cross(y, p - i); z /= np.linalg.norm(z) + 1e-9
    return np.column_stack([np.cross(y, z), y, z])


def align_z(d):                                        # rotation mapping +Z to unit vector d
    z = np.array([0, 0, 1.0]); v = np.cross(z, d); c = float(np.dot(z, d))
    if c > 1 - 1e-8:
        return np.eye(3)
    if c < -1 + 1e-8:
        return np.diag([1.0, -1.0, -1.0])
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1 + c)


def main():
    ap = argparse.ArgumentParser(description="HaMeR wrist orientation extractor")
    ap.add_argument("--pt", required=True, help="GVHMR hmr4d_results.pt")
    ap.add_argument("--hands", required=True, help="MediaPipe hands.npz (boxes/presence)")
    ap.add_argument("--clip-dir", required=True, help="capture folder (rgb/ + cam_K.txt)")
    ap.add_argument("--model-dir", required=True, help="parent of smplx/ model folder")
    ap.add_argument("--out", default="", help="output npz (default <clip>/hands_hamer.npz)")
    ap.add_argument("--match-px", type=float, default=90.0)
    args = ap.parse_args()

    import cv2
    import torch
    import smplx
    os.chdir(HAMER_DIR)
    from hamer.models import load_hamer, DEFAULT_CHECKPOINT
    from hamer.datasets.vitdet_dataset import ViTDetDataset

    dev = "cuda"
    model, cfg = load_hamer(DEFAULT_CHECKPOINT)
    model.eval().to(dev)

    K = np.loadtxt(os.path.join(args.clip_dir, "cam_K.txt"))
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    Kinv = np.linalg.inv(K)
    rgb = sorted(glob.glob(os.path.join(args.clip_dir, "rgb", "*.png")))
    md = np.load(args.hands)
    mval, mbox, mw2d = md["valid"], md["box"], md["wrist2d"]

    pt = torch.load(args.pt, map_location="cpu", weights_only=False)
    pi = pt["smpl_params_incam"]
    T = np.asarray(pi["transl"]).shape[0]
    nb = np.asarray(pi["betas"]).shape[1]
    sm = smplx.create(args.model_dir, model_type="smplx", gender="neutral", num_betas=nb,
                      use_pca=False, flat_hand_mean=True, batch_size=T)
    with torch.no_grad():
        J = sm(betas=torch.as_tensor(np.asarray(pi["betas"]), dtype=torch.float32),
               global_orient=torch.as_tensor(np.asarray(pi["global_orient"]), dtype=torch.float32),
               body_pose=torch.as_tensor(np.asarray(pi["body_pose"]), dtype=torch.float32),
               transl=torch.as_tensor(np.asarray(pi["transl"]), dtype=torch.float32)).joints.numpy()

    R_all = np.tile(np.eye(3), (T, 2, 1, 1)).astype(np.float32)
    w2d = np.zeros((T, 2, 2), np.float32)
    valid = np.zeros((T, 2), bool)
    n = {"L": 0, "R": 0}

    for t in range(min(T, len(rgb))):
        sw = {h: J[t, JW[h]] for h in JW}
        sw2d = {h: np.array([fx * p[0] / p[2] + cx, fy * p[1] / p[2] + cy]) for h, p in sw.items()}
        boxes, rights, slots = [], [], []
        for s in range(2):
            if not mval[t, s]:
                continue
            dd = {h: np.linalg.norm(mw2d[t, s] - sw2d[h]) for h in JW}
            h = min(dd, key=dd.get)
            if dd[h] > args.match_px:
                continue
            slot = 0 if h == "L" else 1
            if valid[t, slot]:
                continue
            boxes.append(mbox[t, s]); rights.append(slot); slots.append((h, slot))
        if not boxes:
            continue
        img = cv2.imread(rgb[t])
        ds = ViTDetDataset(cfg, img, np.array(boxes, np.float32), np.array(rights))
        b = next(iter(torch.utils.data.DataLoader(ds, batch_size=len(boxes))))
        b = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()}
        with torch.no_grad():
            out = model(b)
        kp = out["pred_keypoints_3d"].cpu().numpy()          # (n, 21, 3) crop frame
        for i, (h, slot) in enumerate(slots):
            mult = 2 * rights[i] - 1                          # HaMeR mirrors left hands -> un-mirror x
            k = kp[i] * np.array([mult, 1, 1])
            bx = (boxes[i][0] + boxes[i][2]) / 2; by = (boxes[i][1] + boxes[i][3]) / 2
            d = Kinv @ np.array([bx, by, 1.0]); d /= np.linalg.norm(d)
            R_all[t, slot] = align_z(d) @ palm(k[0], k[5], k[9], k[17])
            w2d[t, slot] = sw2d[h]
            valid[t, slot] = True
            n[h] += 1

    out_path = args.out or os.path.join(args.clip_dir, "hands_hamer.npz")
    np.savez_compressed(out_path, R=R_all, wrist2d=w2d,
                        label=np.where(valid, [[0, 1]], -1).astype(np.int8), valid=valid,
                        K=K.astype(np.float32))
    print(f"[hamer] wrist orientation L={n['L']} R={n['R']} / {T} -> {out_path}")


if __name__ == "__main__":
    main()
