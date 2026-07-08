#!/usr/bin/env python3
"""Depth-anchor a GVHMR SMPL-X result with the clip's metric RGB-D.

GVHMR is monocular: it fits the 2D reprojection but the absolute depth (metric
scale) is ambiguous, so a person walking toward the camera is reconstructed at a
near-constant distance (the body inflates instead of approaching). We fix this
with the Femto depth: per frame, scale the camera-space body about the camera
origin by s(t) = Z_measured / Z_smpl. That keeps the 2D projection identical,
sets the true metric depth, and (because s cancels the 2D-fit size inflation)
yields a constant real body size.

Output: an npz of display-ready world vertices (up = +Z, feet at z=0) + faces,
for view_smpl_viser.py --verts-npz. Body pose/shape come from GVHMR unchanged;
only the metric placement is corrected.

Run in an env with smplx + torch + numpy + PIL (e.g. holonew).
"""

import argparse
import glob
import os

import numpy as np


def rigid_transform(A, B):
    """R,t (no scale) with B ~= R @ A + t. A,B: (N,3). Recovers GVHMR's cam->world."""
    ca, cb = A.mean(0), B.mean(0)
    H = (A - ca).T @ (B - cb)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return R, cb - R @ ca


def smooth_fill(x, win):
    """Fill NaNs by interpolation, then running median (odd win)."""
    x = x.copy()
    n = len(x)
    idx = np.arange(n)
    good = ~np.isnan(x)
    if good.sum() < 2:
        raise SystemExit("[anchor] too few valid depth frames")
    x = np.interp(idx, idx[good], x[good])
    if win > 1:
        h = win // 2
        x = np.array([np.median(x[max(0, i - h):i + h + 1]) for i in range(n)])
    return x


def main():
    ap = argparse.ArgumentParser(description="Depth-anchor a GVHMR SMPL-X clip")
    ap.add_argument("--pt", required=True, help="GVHMR hmr4d_results.pt")
    ap.add_argument("--clip-dir", required=True, help="capture folder (cam_K.txt + depth/)")
    ap.add_argument("--model-dir", required=True, help="parent of smplx/ model folder")
    ap.add_argument("--out", required=True, help="output verts .npz")
    ap.add_argument("--gender", default="neutral")
    ap.add_argument("--smooth", type=int, default=9, help="temporal median window on s(t)")
    args = ap.parse_args()

    import torch
    import smplx
    from PIL import Image

    pt = torch.load(args.pt, map_location="cpu", weights_only=False)
    K = np.loadtxt(os.path.join(args.clip_dir, "cam_K.txt"))
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    deps = sorted(glob.glob(os.path.join(args.clip_dir, "depth", "*.png")))

    pi, pg = pt["smpl_params_incam"], pt["smpl_params_global"]
    T = np.asarray(pi["transl"]).shape[0]
    nb = np.asarray(pi["betas"]).shape[1]
    if len(deps) != T:
        print(f"[anchor] warning: {T} SMPL frames vs {len(deps)} depth files -> using min")
    T = min(T, len(deps))

    model = smplx.create(args.model_dir, model_type="smplx", gender=args.gender,
                         num_betas=nb, use_pca=False, flat_hand_mean=True, batch_size=T)

    def forward(p):
        with torch.no_grad():
            o = model(betas=torch.as_tensor(np.asarray(p["betas"])[:T], dtype=torch.float32),
                      global_orient=torch.as_tensor(np.asarray(p["global_orient"])[:T], dtype=torch.float32),
                      body_pose=torch.as_tensor(np.asarray(p["body_pose"])[:T], dtype=torch.float32),
                      transl=torch.as_tensor(np.asarray(p["transl"])[:T], dtype=torch.float32))
        return o.vertices.numpy(), o.joints.numpy()

    V_cam, J_cam = forward(pi)      # camera frame
    _, J_wld = forward(pg)          # world frame (for cam->world)
    faces = model.faces.astype(np.int32)
    H, W = np.array(Image.open(deps[0])).shape[:2]

    # per-frame metric scale s(t) = Z_measured / Z_smpl_front
    s = np.full(T, np.nan)
    zg = np.full(T, np.nan)   # gvhmr front depth (for the report)
    zm = np.full(T, np.nan)   # measured front depth
    for t in range(T):
        v = V_cam[t]
        z = v[:, 2]
        u = fx * v[:, 0] / z + cx
        w = fy * v[:, 1] / z + cy
        ok = (z > 0) & (u >= 0) & (u < W) & (w >= 0) & (w < H)
        if ok.sum() < 200:
            continue
        z_front = np.percentile(z[ok], 10)
        dm = np.asarray(Image.open(deps[t]), np.float32) / 1000.0   # mm -> m
        meas = dm[w[ok].astype(int), u[ok].astype(int)]
        meas = meas[meas > 0]
        if meas.size < 200:
            continue
        z_meas = np.median(meas)
        zg[t], zm[t] = z_front, z_meas
        s[t] = np.clip(z_meas / z_front, 0.3, 1.6)

    s = smooth_fill(s, args.smooth)

    # scale about camera -> correct depth & constant size, 2D preserved
    V_corr = s[:, None, None] * V_cam

    # cam -> world via GVHMR's own incam<->global correspondence (body joints)
    nbody = min(22, J_cam.shape[1])
    V_wld = np.empty_like(V_corr)
    for t in range(T):
        R, tt = rigid_transform(J_cam[t, :nbody], J_wld[t, :nbody])
        V_wld[t] = V_corr[t] @ R.T + tt

    # display frame: snap up to gravity axis (+Z), feet on the floor
    up_raw = (J_wld[:, 15] - J_wld[:, 0]).mean(0)
    up = np.zeros(3); up[int(np.argmax(np.abs(up_raw)))] = np.sign(up_raw[int(np.argmax(np.abs(up_raw)))])
    b = np.array([0.0, 0.0, 1.0])
    vv = np.cross(up, b); c = float(np.dot(up, b))
    if c > 1 - 1e-8:
        Rz = np.eye(3)
    elif c < -1 + 1e-8:
        Rz = np.diag([1.0, -1.0, -1.0])
    else:
        vx = np.array([[0, -vv[2], vv[1]], [vv[2], 0, -vv[0]], [-vv[1], vv[0], 0]])
        Rz = np.eye(3) + vx + vx @ vx * (1.0 / (1.0 + c))
    V_wld = np.einsum("ij,tvj->tvi", Rz, V_wld).astype(np.float32)
    V_wld[..., 2] -= V_wld[..., 2].min()

    fps = float(pt.get("mocap_framerate", 30.0)) if isinstance(pt, dict) else 30.0
    np.savez_compressed(args.out, verts=V_wld, faces=faces, fps=np.float32(30.0))

    print(f"[anchor] {T} frames | GVHMR depth {np.nanmin(zg):.2f}-{np.nanmax(zg):.2f} m "
          f"(~static) -> measured {np.nanmin(zm):.2f}-{np.nanmax(zm):.2f} m")
    print(f"[anchor] scale s(t): {s.min():.2f}-{s.max():.2f} | wrote {args.out}")
    print(f"[anchor] view: view_smpl_viser.py --verts-npz {args.out}")


if __name__ == "__main__":
    main()
