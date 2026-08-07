#!/usr/bin/env python3
"""
convert_motionx_to_amass.py — repack a Motion-X/Motion-X++ SMPL-X clip (322-dim .npy)
into an AMASS-release-style *_stageii.npz that HoloNew's `amass` loader accepts.

Motion-X's flat 322-dim layout lines up field-for-field with AMASS SMPL-X's
root_orient/pose_body/pose_hand/trans, and Motion-X is already 30 fps (AMASS
loader's resample target), so this is a pure repack — no retargeting logic here.

Usage:
    python scripts/convert_motionx_to_amass.py \\
        --input /path/to/clip.npy [/path/to/other.npy ...] \\
        --out-dir data/00_raw_datasets/MotionX/<subset>
"""
import argparse
from pathlib import Path

import numpy as np


def convert(npy_path: Path, out_path: Path, gender: str = "neutral") -> None:
    motion = np.load(npy_path).astype(np.float64)
    if motion.ndim != 2 or motion.shape[1] != 322:
        raise ValueError(f"{npy_path}: expected (T, 322), got {motion.shape}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        surface_model_type="smplx",
        gender=gender,
        mocap_frame_rate=30.0,
        betas=motion[0, 312:322].astype(np.float32),   # constant per sequence
        root_orient=motion[:, 0:3],
        pose_body=motion[:, 3:66],
        pose_hand=motion[:, 66:156],
        trans=motion[:, 309:312],
    )
    print(f"{npy_path.name} ({motion.shape[0]} frames) -> {out_path}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", nargs="+", required=True, help="Motion-X .npy clip(s)")
    p.add_argument("--out-dir", required=True, help="destination dir for *_stageii.npz")
    p.add_argument("--gender", default="neutral")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    for npy in args.input:
        npy_path = Path(npy)
        out_path = out_dir / f"{npy_path.stem}_stageii.npz"
        convert(npy_path, out_path, args.gender)


if __name__ == "__main__":
    main()
