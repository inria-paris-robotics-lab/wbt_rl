#!/usr/bin/env python3
"""
GMR forward kinematics wrapper — runs in `gmr` conda environment.

Loads a GMR output .pkl (root_pos, root_rot xyzw, dof_pos) and a robot XML,
runs KinematicsModel.forward_kinematics(), saves body_pos_w as .npz.

Usage (called via subprocess from motion_convertor):
    python src/motion_convertor/wrappers/gmr_fk.py \
        --pkl_path  <path/to/retargeter_output.pkl> \
        --xml_path  <path/to/robot.xml> \
        --out_path  <path/to/body_pos_w.npz>

Output .npz keys:
    body_pos_w  : (T, num_bodies, 3) float32 — world-frame body positions
    body_rot_w  : (T, num_bodies, 4) float32 — world-frame body quats (xyzw, GMR convention)
    body_names  : list of str
"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

# Ensure GMR is importable (script runs from repo root)
sys.path.insert(0, str(Path(__file__).parents[3] / "modules" / "01_retargeting" / "GMR"))

from general_motion_retargeting.kinematics_model import KinematicsModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkl_path", required=True, help="GMR output .pkl")
    parser.add_argument("--xml_path", required=True, help="Robot MuJoCo .xml")
    parser.add_argument("--out_path", required=True, help="Output .npz path")
    args = parser.parse_args()

    device = torch.device("cpu")

    # Load GMR output
    with open(args.pkl_path, "rb") as f:
        pkl = pickle.load(f)

    root_pos = torch.from_numpy(pkl["root_pos"]).float()   # (T, 3)
    root_rot = torch.from_numpy(pkl["root_rot"]).float()   # (T, 4) xyzw
    dof_pos = torch.from_numpy(pkl["dof_pos"]).float()     # (T, N_dof)

    # Build kinematics model
    km = KinematicsModel(args.xml_path, device)

    # Run FK
    body_pos, body_rot = km.forward_kinematics(root_pos, root_rot, dof_pos)
    # body_pos: (T, num_bodies, 3)
    # body_rot: (T, num_bodies, 4) xyzw

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        out_path,
        body_pos_w=body_pos.numpy().astype(np.float32),
        body_rot_w=body_rot.numpy().astype(np.float32),
        body_names=np.array(km.body_names),
    )
    print(f"[gmr_fk] Saved body_pos_w ({body_pos.shape}) → {out_path}")


if __name__ == "__main__":
    main()
