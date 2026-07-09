#!/usr/bin/env python3
"""Correct the SMPL-X wrist orientation of a GVHMR result with MediaPipe hands.

GVHMR estimates the body from 2D keypoints, so the forearm roll (palm flat vs
perpendicular) is under-constrained. This overrides the two wrist joints (20/21)
of body_pose so the SMPL hand's palm frame matches the palm frame measured by
extract_hands.py (MediaPipe, same camera frame). Fingers stay neutral (fine for a
fingerless robot like G1); only the wrist orientation is fixed.

Method per frame: FK the GVHMR pose to global joint rotations, match each detected
hand to the nearest projected SMPL wrist (rejects false positives), set
    R_wrist_global = R_palm_mediapipe @ R_palm_rest^T
    R_wrist_local  = R_elbow_global^T @ R_wrist_global
and write it into body_pose. Unmatched frames keep GVHMR; the sequence is smoothed.

Run in an env with smplx + torch + scipy + numpy (holonew).
"""

import argparse
import os

import numpy as np


# SMPL-X joint indices (body 0-21, then hands): wrist, index/middle/pinky MCP per hand
JOINTS = {"L": dict(wrist=20, elbow=18, index=25, middle=28, pinky=31),
          "R": dict(wrist=21, elbow=19, index=40, middle=43, pinky=46)}


def palm_frame(wrist, index, middle, pinky):
    """Orthonormal palm frame: y=fingers (wrist->middle), z=normal, x=across.
    Same convention as extract_hands.py so the two frames are comparable."""
    y = middle - wrist; y = y / (np.linalg.norm(y) + 1e-9)
    z = np.cross(y, pinky - index); z = z / (np.linalg.norm(z) + 1e-9)
    x = np.cross(y, z)
    return np.column_stack([x, y, z])


def main():
    ap = argparse.ArgumentParser(description="MediaPipe -> SMPL-X wrist correction")
    ap.add_argument("--pt", required=True, help="GVHMR hmr4d_results.pt")
    ap.add_argument("--hands", required=True, help="hands.npz from extract_hands.py")
    ap.add_argument("--clip-dir", required=True, help="capture folder (cam_K.txt)")
    ap.add_argument("--model-dir", required=True, help="parent of smplx/ model folder")
    ap.add_argument("--out", required=True, help="output corrected .pt")
    ap.add_argument("--gender", default="neutral")
    ap.add_argument("--match-px", type=float, default=70.0, help="max wrist match distance (px)")
    ap.add_argument("--smooth", type=int, default=9, help="temporal smoothing window (rotations)")
    args = ap.parse_args()

    import torch
    import smplx
    from scipy.spatial.transform import Rotation

    pt = torch.load(args.pt, map_location="cpu", weights_only=False)
    hnd = np.load(args.hands)
    Rp, w2d, lab, val = hnd["R"], hnd["wrist2d"], hnd["label"], hnd["valid"]   # (T,2,..)
    K = np.loadtxt(os.path.join(args.clip_dir, "cam_K.txt"))
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    pi = pt["smpl_params_incam"]
    T = np.asarray(pi["transl"]).shape[0]
    nb = np.asarray(pi["betas"]).shape[1]
    go = np.asarray(pi["global_orient"], np.float64).reshape(T, 3)
    bp = np.asarray(pi["body_pose"], np.float64).reshape(T, 21, 3)
    betas = np.asarray(pi["betas"], np.float64)
    transl = np.asarray(pi["transl"], np.float64).reshape(T, 3)

    model = smplx.create(args.model_dir, model_type="smplx", gender=args.gender,
                         num_betas=nb, use_pca=False, flat_hand_mean=True, batch_size=T)
    parents = model.parents.numpy()

    # rest palm frame per hand (wrist global is identity in rest -> R_palm_rest = R_palm_rel)
    with torch.no_grad():
        rest = model(betas=torch.zeros(T, nb)).joints[0].numpy()
    R_rel = {h: palm_frame(rest[j["wrist"]], rest[j["index"]], rest[j["middle"]], rest[j["pinky"]])
             for h, j in JOINTS.items()}

    # SMPL joints per frame (camera) to project the wrists for matching
    with torch.no_grad():
        out = model(betas=torch.as_tensor(betas, dtype=torch.float32),
                    global_orient=torch.as_tensor(go, dtype=torch.float32),
                    body_pose=torch.as_tensor(bp.reshape(T, -1), dtype=torch.float32),
                    transl=torch.as_tensor(transl, dtype=torch.float32))
    J = out.joints.numpy()                              # (T, Jn, 3) camera
    aa2R = lambda a: Rotation.from_rotvec(a).as_matrix()

    n_fixed = {"L": 0, "R": 0}
    matched = np.zeros((T, 2), bool)
    Rtgt = {h: np.tile(np.eye(3), (T, 1, 1)) for h in JOINTS}      # target palm frame per wrist
    Relb = {h: np.tile(np.eye(3), (T, 1, 1)) for h in JOINTS}      # elbow global, every frame

    def ang(A, B):
        return np.arccos(np.clip((np.trace(A.T @ B) - 1) / 2, -1, 1))

    # pass 1: FK every frame (elbow global for all t) + match hands to the nearest projected wrist
    for t in range(T):
        Rloc = np.stack([aa2R(go[t])] + [aa2R(bp[t, j]) for j in range(21)])
        Rg = np.empty((22, 3, 3)); Rg[0] = Rloc[0]
        for j in range(1, 22):
            Rg[j] = Rg[parents[j]] @ Rloc[j]
        for h, jj in JOINTS.items():
            Relb[h][t] = Rg[parents[jj["wrist"]]]
        sw = {h: J[t, jj["wrist"]] for h, jj in JOINTS.items()}
        sw = {h: np.array([fx * p[0] / p[2] + cx, fy * p[1] / p[2] + cy]) for h, p in sw.items()}
        for s in range(2):
            if not val[t, s]:
                continue
            dd = {h: np.linalg.norm(w2d[t, s] - sw[h]) for h in JOINTS}
            h = min(dd, key=dd.get)
            if dd[h] > args.match_px or matched[t, 0 if h == "L" else 1]:
                continue
            Rtgt[h][t] = Rp[t, s]
            matched[t, 0 if h == "L" else 1] = True

    # pass 2+3: per wrist, stabilize the palm-normal sign, slerp-interpolate the target across
    # detection gaps (kills the flat<->vertical alternation), smooth, then apply over the span.
    from scipy.spatial.transform import Slerp
    flip = np.diag([-1.0, 1.0, -1.0])
    hw = args.smooth // 2
    for h in JOINTS:
        hi = 0 if h == "L" else 1
        idx = np.where(matched[:, hi])[0]
        if idx.size < 2:
            continue
        seq = Rtgt[h][idx]
        for k in range(1, len(idx)):
            if ang(seq[k] @ flip, seq[k - 1]) < ang(seq[k], seq[k - 1]):
                seq[k] = seq[k] @ flip
        span = np.arange(idx[0], idx[-1] + 1)
        full = Slerp(idx, Rotation.from_matrix(seq))(span)
        if args.smooth > 1:
            full = Rotation.from_matrix(np.stack(
                [full[max(0, k - hw):min(len(span), k + hw + 1)].mean().as_matrix()
                 for k in range(len(span))]))
        Rmat = full.as_matrix()
        for k, t in enumerate(span):
            R_wrist_global = Rmat[k] @ R_rel[h].T
            bp[t, JOINTS[h]["wrist"] - 1] = Rotation.from_matrix(Relb[h][t].T @ R_wrist_global).as_rotvec()
            n_fixed[h] += 1
            n_fixed[h] += 1

    # write corrected pose back into incam and global (body_pose is frame-independent)
    for key in ("smpl_params_incam", "smpl_params_global"):
        pt[key]["body_pose"] = torch.as_tensor(bp.reshape(T, -1), dtype=torch.float32)
    torch.save(pt, args.out)
    print(f"[wrist] corrected L={n_fixed['L']} R={n_fixed['R']} frames / {T} -> {args.out}")


if __name__ == "__main__":
    main()
