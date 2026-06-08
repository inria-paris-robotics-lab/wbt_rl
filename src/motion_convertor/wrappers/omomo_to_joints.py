#!/usr/bin/env python3
"""
Wrapper: OMOMO sequence dict (pickle) → unified .npz (global_joint_positions + height).

Runs in hsretargeting env (human_body_prior.BodyModel available).
Reads one sequence from the OMOMO pickle by index.

Usage:
    python src/motion_convertor/wrappers/omomo_to_joints.py \
        --pickle <train_diffusion_manip_seq_joints24.p> \
        --index <seq_index> \
        --output <out.npz> \
        --model_dir <smplh_model_dir>
"""
import argparse
import numpy as np
import torch
import joblib
from pathlib import Path

from human_body_prior.body_model.body_model import BodyModel


def run_smplh_fk(root_orient, pose_body, trans, betas, gender, model_dir: str):
    """
    SMPL-H FK via human_body_prior.BodyModel.
    Returns (T, 22, 3) global joint positions, Z-up metres.
    """
    gender = gender.lower()
    bm_path = str(Path(model_dir) / gender / "model.npz")
    T = root_orient.shape[0]
    num_betas = betas.shape[0]

    bm = BodyModel(bm_fname=bm_path, num_betas=num_betas)

    output = bm(
        pose_body=torch.from_numpy(pose_body).float(),
        root_orient=torch.from_numpy(root_orient).float(),
        betas=torch.from_numpy(betas).float()[None].expand(T, -1),
        trans=torch.from_numpy(trans).float(),
    )
    joints = output.Jtr[:, :22, :].detach().cpu().numpy()  # (T, 22, 3)
    return joints


def compute_height_smplh(betas, gender, model_dir: str) -> float:
    """Height from SMPL-H T-pose vertices (max_y - min_y)."""
    gender = gender.lower()
    bm_path = str(Path(model_dir) / gender / "model.npz")
    num_betas = betas.shape[0]

    bm = BodyModel(bm_fname=bm_path, num_betas=num_betas)
    output = bm(
        betas=torch.from_numpy(betas).float()[None],
    )
    verts = output.v[0].detach().cpu().numpy()  # (V, 3)
    return float(verts[:, 1].max() - verts[:, 1].min())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pickle",    required=True)
    parser.add_argument("--index",     required=True)
    parser.add_argument("--output",    required=True)
    parser.add_argument("--model_dir", required=True)
    args = parser.parse_args()

    data = joblib.load(args.pickle)
    # index may be int or string key
    key = args.index
    if key not in data:
        key = int(key)
    seq = data[key]

    root_orient = seq["root_orient"].astype(np.float32)   # (T, 3)
    pose_body   = seq["pose_body"].astype(np.float32)     # (T, 63)
    trans       = seq["trans"].astype(np.float32)         # (T, 3)
    betas       = seq["betas"][0].astype(np.float32)      # (16,)
    gender      = str(seq["gender"])

    joints = run_smplh_fk(root_orient, pose_body, trans, betas, gender, args.model_dir)
    height = compute_height_smplh(betas, gender, args.model_dir)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, global_joint_positions=joints.astype(np.float32), height=np.float32(height))
    print(f"Saved: {args.output}  shape={joints.shape}  height={height:.3f}m")


if __name__ == "__main__":
    main()
