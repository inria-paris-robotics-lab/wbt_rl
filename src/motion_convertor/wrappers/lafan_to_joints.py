#!/usr/bin/env python3
"""
Wrapper: LAFAN .bvh → joint positions.

Runs in hsretargeting env (lafan1 available as submodule).

Produces either:
  --format yup   → .npy  (T, J, 3) Y-up metres  (holosoma native input)
  --format unified → .npz  global_joint_positions (T,J,3) Z-up metres + height

Usage:
    python src/motion_convertor/wrappers/lafan_to_joints.py \
        --input <file.bvh> --output <out> --format yup|unified
"""
import argparse
import sys
import numpy as np
from pathlib import Path

# lafan1 submodule
sys.path.insert(0, str(Path(__file__).parents[3] / "src/motion_convertor/third_party/lafan1"))

from lafan1 import extract, utils

# BVH depth-first order → LAFAN_DEMO_JOINTS order (holosoma)
_BVH_TO_DEMO = [0, 5, 6, 7, 8, 1, 2, 3, 4, 9, 10, 11, 12, 13, 18, 19, 20, 21, 14, 15, 16, 17]


def bvh_to_positions(bvh_path: str):
    """Returns (T, J, 3) Y-up metres in BVH depth-first joint order."""
    anim = extract.read_bvh(bvh_path)
    _, global_positions = utils.quat_fk(anim.quats, anim.pos, anim.parents)
    return global_positions / 100.0  # cm → m


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--format", required=True, choices=["yup", "unified"])
    args = parser.parse_args()

    positions = bvh_to_positions(args.input)  # (T, J, 3) Y-up, BVH order

    if args.format == "yup":
        # Reorder to LAFAN_DEMO_JOINTS.
        # Negate Z axis: BVH convention is forward=-Z, holosoma expects forward=+Z
        # (after its internal Y→Z-up swap, +Z becomes +Y forward in holosoma's world).
        out = positions[:, _BVH_TO_DEMO, :].astype(np.float32)
        out[:, :, 2] *= -1
        np.save(args.output, out)
        print(f"Saved: {args.output}  shape={out.shape}")

    elif args.format == "unified":
        # Reorder + Y→Z-up: holosoma uses [x,z,y] swap (no negation).
        # Apply same Z-negation so unified format is consistent.
        pos = positions[:, _BVH_TO_DEMO, :].astype(np.float32)  # (T, 22, 3) Y-up
        pos[:, :, 2] *= -1
        joints_zup = np.stack([pos[:, :, 0], pos[:, :, 2], pos[:, :, 1]], axis=-1)  # Z-up

        # Height from head (joint 13) and feet (joints 7, 3)
        head_z  = joints_zup[:, 13, 2].max()
        foot_z  = np.minimum(joints_zup[:, 7, 2], joints_zup[:, 3, 2]).min()
        height  = float(head_z - foot_z)

        np.savez(args.output, global_joint_positions=joints_zup, height=np.float32(height))
        print(f"Saved: {args.output}  shape={joints_zup.shape}  height={height:.3f}m")


if __name__ == "__main__":
    main()
