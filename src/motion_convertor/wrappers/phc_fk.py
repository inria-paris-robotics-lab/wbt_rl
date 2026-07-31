#!/usr/bin/env python3
"""
Wrapper: forward kinematics on a PHC fit result, run in the phc env.

PHC's dump carries dof / root_trans_offset / root_rot but no body positions, so the
unified converter needs FK. Humanoid_Batch is PHC's own FK, driven by the same robot
config the fit used — so the positions are consistent with what was fitted.

Usage:
    python src/motion_convertor/wrappers/phc_fk.py \
        --pkl_path <fit.pkl> --robot_cfg <robot.yaml> --out_path <body_pos.npz>
"""
import argparse
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).parents[3]
_PHC_ROOT = _REPO_ROOT / "modules" / "third_party" / "PHC"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkl_path", required=True)
    parser.add_argument("--robot_cfg", required=True)
    parser.add_argument("--out_path", required=True)
    args = parser.parse_args()

    sys.path.insert(0, str(_PHC_ROOT))
    import joblib
    import torch
    from omegaconf import OmegaConf
    from phc.utils.torch_humanoid_batch import Humanoid_Batch

    cfg = OmegaConf.load(args.robot_cfg)
    OmegaConf.resolve(cfg)
    fk = Humanoid_Batch(cfg)

    all_data = joblib.load(args.pkl_path)
    (data,) = all_data.values()
    pose_aa = torch.from_numpy(np.asarray(data["pose_aa"], np.float32))[None]
    root_trans = torch.from_numpy(
        np.asarray(data["root_trans_offset"], np.float32).reshape(-1, 3))[None]

    out = fk.fk_batch(pose_aa, root_trans)
    body_pos = out["global_translation"][0].detach().cpu().numpy()   # (T, B, 3)

    np.savez(args.out_path, body_pos_w=body_pos.astype(np.float32),
             body_names=np.array(fk.body_names_augment))
    print(f"PHC FK → {args.out_path}: {body_pos.shape}")


if __name__ == "__main__":
    main()
