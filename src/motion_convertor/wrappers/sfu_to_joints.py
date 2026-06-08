#!/usr/bin/env python3
"""
Wrapper: SFU/AMASS .npz → unified .npz (global_joint_positions + height).

Calls holosoma's prep_amass_smplx_for_rt.py logic on a single file.
Runs in hsretargeting env (human_body_prior.BodyModel available).

Usage:
    python src/motion_convertor/wrappers/sfu_prep.py --input <file.npz> --output <out.npz> --model_root <models_dir>
"""
import argparse
import sys
import numpy as np
import torch
from pathlib import Path

# holosoma data_utils on path via hsretargeting install
sys.path.insert(0, str(Path(__file__).parents[3] /
    "modules/third_party/holosoma/src/holosoma_retargeting"))

from holosoma_retargeting.data_utils.prep_amass_smplx_for_rt import (
    load_ori_npz_file, run_smplx_model, prep_smplx_model, compute_height,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model_root", required=True)
    args = parser.parse_args()

    data = load_ori_npz_file(args.input, dest_fps=30)
    gender = str(data["gender"])
    betas = torch.from_numpy(data["betas"]).float()[None]   # (1, 16)
    root_trans = torch.from_numpy(data["trans"]).float()[None]   # (1, T, 3)
    aa_rot_rep = torch.from_numpy(
        data["poses"].reshape(-1, 55, 3)[:, :52, :]
    ).float()[None]  # (1, T, 52, 3)

    bm_dict = prep_smplx_model(args.model_root)

    global_joint_positions, _, _ = run_smplx_model(
        root_trans=root_trans,
        aa_rot_rep=aa_rot_rep,
        betas=betas,
        gender=[gender],
        bm_dict=bm_dict,
    )
    joints = global_joint_positions.squeeze(0).detach().cpu().numpy()[:, :22, :]  # (T, 22, 3)

    height = compute_height(bm_dict, betas, gender=[gender])

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, global_joint_positions=joints, height=height)
    print(f"Saved: {args.output}  shape={joints.shape}  height={height:.3f}m")


if __name__ == "__main__":
    main()
