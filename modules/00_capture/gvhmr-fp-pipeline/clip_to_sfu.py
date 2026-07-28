#!/usr/bin/env python3
"""capture .pt (GVHMR/HaMeR SMPL-X) -> HoloV2 SFU npz (global per-joint pos/orient, Z-up).

HoloV2's SfuLoader (prepare/load/datasets/sfu.py) reads exactly 4 keys: betas, gender,
global_joint_orientations (T,22,4 wxyz) and global_joint_positions (T,22,3) -- body only,
already Z-up. It does NOT read an AMASS poses/trans blob (that's the RAW mocap format,
a different, earlier stage in the AMASS toolchain).

GVHMR's own "global" world frame is gravity-aligned but its up axis is whichever of x/y/z
gravity landed on (not necessarily Z) -- the same ambiguity view_smpl_viser.py resolves for
display. We compute that snap rotation once from the head-pelvis direction and apply it to
both joint positions (smplx forward pass) and joint orientations (FK of global_orient+body_pose
down the SMPL-X body tree), landing in the Z-up frame the loader expects. A single constant
shift (not per-frame) grounds the ankles near z=0 without flattening real vertical motion
(jumps, crouches) -- retargeting's own calibration re-anchors the floor anyway.

Run in an env with smplx + torch + scipy (holonew).
"""
import argparse

import numpy as np


def rot_to_z(a):
    """Rotation mapping unit vector a to +Z (same construction as view_smpl_viser.py)."""
    a = a / (np.linalg.norm(a) + 1e-9)
    b = np.array([0.0, 0.0, 1.0])
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if c > 1 - 1e-8:
        return np.eye(3)
    if c < -1 + 1e-8:
        return np.diag([1.0, -1.0, -1.0])
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1 + c)


def main():
    ap = argparse.ArgumentParser(description="GVHMR .pt -> HoloV2 SFU npz")
    ap.add_argument("--pt", required=True, help="capture .pt (wrist/contact-corrected recommended)")
    ap.add_argument("--out", required=True, help="output SFU npz")
    ap.add_argument("--model-dir", required=True, help="parent of smplx/ model folder")
    ap.add_argument("--gender", default="neutral")
    ap.add_argument("--clip", default="", help="fuse clip.npz -> also embed the object in the SFU npz "
                    "(same Z-up snap + floor shift as the body, so it stays glued to the hands)")
    ap.add_argument("--object-mesh", default="", help="override object mesh path (else taken from clip.npz)")
    args = ap.parse_args()

    import smplx
    import torch
    from scipy.spatial.transform import Rotation

    pt = torch.load(args.pt, map_location="cpu", weights_only=False)
    p = pt["smpl_params_global"]
    betas_full = np.asarray(p["betas"], np.float32)
    T = np.asarray(p["transl"]).shape[0]
    nb = betas_full.shape[1]

    go = np.asarray(p["global_orient"], np.float32).reshape(T, 3)
    bp = np.asarray(p["body_pose"], np.float32).reshape(T, 63)
    tr = np.asarray(p["transl"], np.float32).reshape(T, 3)

    model = smplx.create(args.model_dir, model_type="smplx", gender=args.gender, num_betas=nb,
                         use_pca=False, flat_hand_mean=True, batch_size=T)
    with torch.no_grad():
        out = model(betas=torch.as_tensor(betas_full, dtype=torch.float32),
                    global_orient=torch.as_tensor(go, dtype=torch.float32),
                    body_pose=torch.as_tensor(bp, dtype=torch.float32),
                    transl=torch.as_tensor(tr, dtype=torch.float32))
    joints = out.joints.numpy()[:, :22]                            # (T,22,3) native GVHMR world

    # native global per-joint rotation: FK of local axis-angle down the SMPL-X body tree
    parents = model.parents.numpy()[:22]
    aa = np.concatenate([go[:, None, :], bp.reshape(T, 21, 3)], axis=1)   # (T,22,3)
    local_R = Rotation.from_rotvec(aa.reshape(-1, 3)).as_matrix().reshape(T, 22, 3, 3)
    g = np.empty((T, 22, 3, 3))
    g[:, 0] = local_R[:, 0]
    for j in range(1, 22):
        g[:, j] = g[:, parents[j]] @ local_R[:, j]

    # snap GVHMR's gravity axis to +Z (same convention as view_smpl_viser.py)
    raw_up = (joints[:, 15] - joints[:, 0]).mean(0)                 # head - pelvis, averaged
    ax = int(np.argmax(np.abs(raw_up)))
    up = np.zeros(3); up[ax] = np.sign(raw_up[ax])
    R = rot_to_z(up)

    pos = np.einsum("ij,tnj->tni", R, joints)
    rot = np.einsum("ij,tnjk->tnik", R, g)

    # single constant floor shift: median of the per-frame lowest FOOT joint (indices 10/11 = L_Foot/
    # R_Foot in SMPLX_BODY_JOINTS), matching HoloV2's own calibration exactly (prepare/calibration/
    # build.py foot_floor_offset, percentile=50 by default; _foot_indices prefers "foot" over "ankle").
    # Grounding on a different joint/percentile here double-shifts the clip, since HoloV2 always
    # re-derives and re-applies its own floor offset on top of whatever this script already did.
    foot_z = pos[:, [10, 11], 2].min(axis=1)
    floor_shift = float(np.percentile(foot_z, 50))
    pos[..., 2] -= floor_shift

    quat_wxyz = Rotation.from_matrix(rot.reshape(-1, 3, 3)).as_quat().reshape(T, 22, 4)[..., [3, 0, 1, 2]]

    save = dict(
        global_joint_positions=pos.astype(np.float32),
        global_joint_orientations=quat_wxyz.astype(np.float64),
        betas=betas_full[0].astype(np.float32),
        gender=args.gender)

    # optional object: read the world pose from clip.npz (already per-frame aligned to the body
    # + held/smoothed by fuse), then apply the SAME snap rotation R and floor shift as the body,
    # so it lands in the SFU Z-up frame glued to the hands. Stored as (T,7) pos-first wxyz.
    if args.clip:
        c = np.load(args.clip, allow_pickle=True)
        OP = np.asarray(c["object_pose"], np.float64)          # (To,4,4) GVHMR global world
        Tm = min(T, OP.shape[0])
        if Tm != T:
            print(f"[sfu] body {T} vs object {OP.shape[0]} frames -> truncated to {Tm}")
            save["global_joint_positions"] = save["global_joint_positions"][:Tm]
            save["global_joint_orientations"] = save["global_joint_orientations"][:Tm]
        o_pos = np.einsum("ij,tj->ti", R, OP[:Tm, :3, 3])
        o_rot = np.einsum("ij,tjk->tik", R, OP[:Tm, :3, :3])
        o_pos[:, 2] -= floor_shift
        o_quat = Rotation.from_matrix(o_rot).as_quat()[:, [3, 0, 1, 2]]     # wxyz
        obj7 = np.concatenate([o_pos, o_quat], axis=1).astype(np.float32)   # (Tm, 7)
        mesh_path = args.object_mesh or str(c["object_mesh"])
        save["object_poses"] = obj7
        save["object_mesh"] = mesh_path
        save["object_name"] = "object"
        print(f"[sfu] + object embedded: poses {obj7.shape}, mesh={mesh_path}")

    np.savez(args.out, **save)
    print(f"[sfu] wrote {args.out}  T={min(T, save['global_joint_positions'].shape[0])}  betas={nb}")


if __name__ == "__main__":
    main()
