#!/usr/bin/env python3
"""Fuse GVHMR (body -> SMPL) and FoundationPose (object -> 6DoF pose) into a
single aligned clip, exported as .npz (raw SMPL params + object pose).

In the camera frame, GVHMR's smpl_params_incam and FoundationPose's object pose
are already in the same frame at metric scale, so they compose directly (default).
With --coord world, the object is lifted to the gravity-aligned world frame using
the camera->world transform, recovered by rigid (Umeyama) alignment of the SMPL
joints incam vs global (needs smplx + SMPL models).

Scale matters: the object CAD must be in meters and the depth fed to FoundationPose
must be metric, otherwise the object drifts or clips through the body.
"""

import argparse
import glob
import os
import sys

import numpy as np


def load_gvhmr(pt_path):
    """Load hmr4d_results.pt, return numpy SMPL params per frame."""
    import torch

    data = torch.load(pt_path, map_location="cpu", weights_only=False)

    def _need(key):
        if key not in data:
            print(f"[fuse] keys in .pt: {list(data.keys())}", file=sys.stderr)
            sys.exit(
                f"[fuse] ERROR: key '{key}' missing from GVHMR .pt.\n"
                "        Adapt the key names to your GVHMR version."
            )
        return data[key]

    def _to_np(d):
        return {k: (v.detach().cpu().numpy() if hasattr(v, "detach") else np.asarray(v))
                for k, v in d.items()}

    out = {
        "incam": _to_np(_need("smpl_params_incam")),
        "global": _to_np(_need("smpl_params_global")),
    }
    # GVHMR intrinsics, used to cross-check consistency with FoundationPose
    out["K_fullimg"] = (data["K_fullimg"].detach().cpu().numpy()
                        if "K_fullimg" in data and hasattr(data["K_fullimg"], "detach")
                        else data.get("K_fullimg"))
    return out


def detect_model_type(smpl_params):
    """Infer smpl vs smplx from body_pose width (69 -> smpl, 63 -> smplx)."""
    bp = np.asarray(smpl_params["body_pose"])
    ndof = bp.reshape(bp.shape[0], -1).shape[1]
    if ndof == 69:
        return "smpl", 23
    if ndof == 63:
        return "smplx", 21
    # some GVHMR versions store rotation matrices instead of axis-angle
    print(f"[fuse] warning: body_pose has {ndof} DoF (neither 69 nor 63); "
          "assuming flat axis-angle.", file=sys.stderr)
    return ("smplx" if ndof < 69 else "smpl"), ndof // 3


def load_foundationpose(poses_dir):
    """Load object->camera poses (dir of NNNNNN.txt 4x4, or a .npy (T,4,4)).

    Returns (poses (T,4,4) float64, valid (T,) bool). Missing/invalid frames
    (NaN, det<=0) are flagged and filled by carry-forward for a continuous clip.
    """
    if poses_dir.endswith(".npy"):
        raw = np.load(poses_dir).astype(np.float64)
        files = [f"{i:06d}" for i in range(len(raw))]
    else:
        files = sorted(glob.glob(os.path.join(poses_dir, "*.txt")))
        if not files:
            sys.exit(f"[fuse] ERROR: no pose .txt in {poses_dir}")
        raw = np.stack([np.loadtxt(f).reshape(4, 4) for f in files]).astype(np.float64)

    T = len(raw)
    valid = np.ones(T, dtype=bool)
    for i in range(T):
        M = raw[i]
        R = M[:3, :3]
        if not np.all(np.isfinite(M)) or abs(np.linalg.det(R)) < 1e-6:
            valid[i] = False

    # carry-forward invalid poses
    last = None
    for i in range(T):
        if valid[i]:
            last = raw[i].copy()
        elif last is not None:
            raw[i] = last
    # if the first frames are invalid, back-fill from the first valid one
    if not valid[0] and valid.any():
        first_valid = np.argmax(valid)
        raw[:first_valid] = raw[first_valid]

    n_bad = int((~valid).sum())
    if n_bad:
        print(f"[fuse] FoundationPose: {n_bad}/{T} frames without a valid pose "
              "(carry-forward filled, see object_valid).", file=sys.stderr)
    return raw, valid


def rigid_transform(A, B):
    """R,t with B ~= R @ A + t. A,B: (N,3)."""
    ca, cb = A.mean(0), B.mean(0)
    H = (A - ca).T @ (B - cb)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    t = cb - R @ ca
    return R, t


def smplx_joints(params, model_type, gender, model_dir, num_betas):
    """Run SMPL(-X) params through the body model, return joints (T,J,3)."""
    try:
        import smplx
        import torch
    except ImportError:
        sys.exit("[fuse] --coord world needs 'smplx' and 'torch' "
                 "(pip install smplx) + SMPL models in --smpl-model-dir.")

    T = np.asarray(params["betas"]).shape[0]
    model = smplx.create(model_dir, model_type=model_type, gender=gender,
                         num_betas=num_betas, use_pca=False, flat_hand_mean=True,
                         batch_size=T)
    kw = dict(
        betas=torch.as_tensor(params["betas"], dtype=torch.float32).reshape(T, num_betas),
        global_orient=torch.as_tensor(params["global_orient"], dtype=torch.float32).reshape(T, 3),
        body_pose=torch.as_tensor(params["body_pose"], dtype=torch.float32).reshape(T, -1),
        transl=torch.as_tensor(params["transl"], dtype=torch.float32).reshape(T, 3),
    )
    with torch.no_grad():
        return model(**kw).joints.cpu().numpy()  # (T, J, 3)


def camera_to_world_transforms(g, model_type, n_body, gender, model_dir, num_betas):
    """Per-frame T_world_cam (4x4) via rigid alignment joints_incam -> joints_global."""
    j_cam = smplx_joints(g["incam"], model_type, gender, model_dir, num_betas)
    j_wld = smplx_joints(g["global"], model_type, gender, model_dir, num_betas)
    T = min(len(j_cam), len(j_wld))
    Ts = np.tile(np.eye(4), (T, 1, 1))
    for i in range(T):
        R, t = rigid_transform(j_cam[i], j_wld[i])
        Ts[i, :3, :3] = R
        Ts[i, :3, 3] = t
    return Ts


def build_poses(smpl_params, model_type):
    """Concatenate [global_orient(3), body_pose] into flat axis-angle (T, 3+ndof)."""
    go = np.asarray(smpl_params["global_orient"]).reshape(len(smpl_params["global_orient"]), 3)
    bp = np.asarray(smpl_params["body_pose"])
    bp = bp.reshape(bp.shape[0], -1)
    return np.concatenate([go, bp], axis=1)


def main():
    ap = argparse.ArgumentParser(description="Fuse GVHMR + FoundationPose -> .npz")
    ap.add_argument("--gvhmr-pt", required=True, help="GVHMR hmr4d_results.pt")
    ap.add_argument("--fp-poses", required=True,
                    help="ob_in_cam/ dir (NNNNNN.txt) or .npy (T,4,4)")
    ap.add_argument("--out", required=True, help="output .npz")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--coord", choices=["camera", "world"], default="camera",
                    help="output frame (camera = robust, no extra deps; "
                         "world = gravity-aligned, needs smplx + models)")
    ap.add_argument("--gender", default="neutral")
    ap.add_argument("--object-mesh", default="", help="CAD path (metadata)")
    ap.add_argument("--smpl-model-dir", default="",
                    help="SMPL/SMPL-X model dir (required if --coord world)")
    args = ap.parse_args()

    g = load_gvhmr(args.gvhmr_pt)
    model_type, n_body = detect_model_type(g["incam"])
    num_betas = np.asarray(g["incam"]["betas"]).reshape(np.asarray(g["incam"]["betas"]).shape[0], -1).shape[1]
    print(f"[fuse] detected model: {model_type} ({n_body} body joints, {num_betas} betas)")

    obj_cam, obj_valid = load_foundationpose(args.fp_poses)

    body = g["incam"] if args.coord == "camera" else g["global"]

    # common length (body vs object): truncate and warn
    T_body = np.asarray(body["transl"]).shape[0]
    T = min(T_body, len(obj_cam))
    if T_body != len(obj_cam):
        print(f"[fuse] length mismatch (body={T_body}, object={len(obj_cam)}) "
              f"-> truncated to {T}. Check frames line up (same t0/fps).",
              file=sys.stderr)

    # object stays in camera frame by default, else lift to world
    obj_pose = obj_cam[:T].copy()
    if args.coord == "world":
        if not args.smpl_model_dir:
            sys.exit("[fuse] --coord world needs --smpl-model-dir.")
        Tw = camera_to_world_transforms(g, model_type, n_body, args.gender,
                                        args.smpl_model_dir, num_betas)[:T]
        obj_pose = np.einsum("tij,tjk->tik", Tw, obj_pose)  # T_world_cam @ T_obj_cam

    poses = build_poses(body, model_type)[:T]
    betas = np.asarray(body["betas"]).reshape(T_body, -1)[:T]
    trans = np.asarray(body["transl"]).reshape(T_body, 3)[:T]

    out = dict(
        # body: raw SMPL, AMASS-style
        poses=poses.astype(np.float32),                 # (T, 3+ndof) axis-angle
        betas=betas.astype(np.float32),                 # (T, num_betas)
        betas_mean=betas.mean(0).astype(np.float32),    # (num_betas,)
        trans=trans.astype(np.float32),                 # (T, 3)
        global_orient=poses[:, :3].astype(np.float32),
        body_pose=poses[:, 3:].astype(np.float32),
        mocap_framerate=np.float32(args.fps),
        gender=args.gender,
        model_type=model_type,
        coord_frame=args.coord,
        # object: 6DoF pose (object -> common frame)
        object_pose=obj_pose.astype(np.float32),        # (T, 4, 4)
        object_R=obj_pose[:, :3, :3].astype(np.float32),
        object_t=obj_pose[:, :3, 3].astype(np.float32),
        object_valid=obj_valid[:T],                     # (T,) False = interpolated
        object_mesh=args.object_mesh,
    )
    if g.get("K_fullimg") is not None:
        out["camera_K"] = np.asarray(g["K_fullimg"]).astype(np.float32)

    np.savez(args.out, **out)
    print(f"[fuse] OK -> {args.out}")
    print(f"       {T} frames | body {model_type} in {args.coord} frame | "
          f"object valid on {int(obj_valid[:T].sum())}/{T} frames")


if __name__ == "__main__":
    main()
