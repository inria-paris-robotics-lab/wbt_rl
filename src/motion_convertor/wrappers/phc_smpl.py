#!/usr/bin/env python3
"""
Wrapper: PHC motion fit for one sequence, run in the phc env.

fit_smpl_motion.py is hydra-driven and batch-oriented: it globs a directory and names
its output after a key derived from the relative path. This wrapper gives it a
single-file directory and moves the result to the requested path.

Requires the shape fit (data/g1_29dof/shape_optimized_v1.pkl), produced once by
fit_smpl_shape.py — see installers/modules/phc.sh.

Usage:
    python src/motion_convertor/wrappers/phc_smpl.py \
        --smpl_file <input.npz> --save_path <output.pkl>
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

_REPO_ROOT = Path(__file__).parents[3]
_PHC_ROOT = _REPO_ROOT / "modules" / "third_party" / "PHC"
_CFG_DIR = _REPO_ROOT / "cfg" / "01_retargeting" / "phc"
_HUMANOID_TYPE = "g1_29dof"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smpl_file", required=True)
    parser.add_argument("--save_path", required=True)
    args = parser.parse_args()

    shape_pkl = _PHC_ROOT / "data" / _HUMANOID_TYPE / "shape_optimized_v1.pkl"
    if not shape_pkl.exists():
        raise SystemExit(f"Missing shape fit: {shape_pkl}\n"
                         f"Run fit_smpl_shape.py once (see installers/modules/phc.sh).")

    src = Path(args.smpl_file)
    dst = Path(args.save_path)
    dst.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as td:
        amass_root = Path(td) / "amass"
        amass_root.mkdir()
        shutil.copy2(src, amass_root / f"{src.stem}.npz")

        # amass_root/fit_all are not declared in PHC's own config.yaml (struct mode),
        # so a plain override is rejected — the "+" prefix adds them instead.
        subprocess.run(
            [sys.executable, "scripts/data_process/fit_smpl_motion.py",
             "--config-dir", str(_CFG_DIR), "robot=unitree_g1_29dof_fitting",
             f"+amass_root={amass_root}", "+fit_all=True"],
            cwd=_PHC_ROOT, check=True,
            env={**os.environ, "WBT_ROOT": str(_REPO_ROOT)},
        )

        produced = _PHC_ROOT / "data" / _HUMANOID_TYPE / "v1" / "singles" / f"0-{src.stem}.pkl"
        if not produced.exists():
            raise RuntimeError(f"PHC produced nothing at {produced}")
        shutil.move(str(produced), dst)

    print(f"PHC output → {dst}")


if __name__ == "__main__":
    main()
