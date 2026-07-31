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
import time
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

        # Staleness guard: PHC writes to a PERSISTENT path (outside our tempdir), keyed
        # solely on `src.stem`. retarget.py names its input file deterministically per
        # sequence, so that stem repeats on every re-run. If process_motion() skips the
        # entry internally (amass_data is None in load_amass_data, or N < 10 frames),
        # all_data stays empty, main() takes the `else` branch (writes v1/amass_all.pkl
        # instead) and exits 0 — check=True never fires. A pkl left at this exact path by
        # an earlier run of the SAME stem would then be unpacked and reported as THIS
        # run's result, with zero error signal. We delete the pkl BEFORE launching the
        # subprocess (if present), then afterwards verify it exists AND that its mtime
        # postdates the launch (belt and suspenders: a pkl reappearing through some other
        # means with a stale mtime is not enough).
        produced = _PHC_ROOT / "data" / _HUMANOID_TYPE / "v1" / "singles" / f"0-{src.stem}.pkl"
        produced.unlink(missing_ok=True)

        # amass_root/fit_all are not declared in PHC's own config.yaml (struct mode),
        # so a plain override is rejected — the "+" prefix adds them instead.
        t_launch_wall = time.time()
        subprocess.run(
            [sys.executable, "scripts/data_process/fit_smpl_motion.py",
             "--config-dir", str(_CFG_DIR), "robot=unitree_g1_29dof_fitting",
             f"+amass_root={amass_root}", "+fit_all=True"],
            cwd=_PHC_ROOT, check=True,
            env={**os.environ, "WBT_ROOT": str(_REPO_ROOT)},
        )

        if not produced.exists():
            raise RuntimeError(
                f"PHC produced nothing at {produced} — process_motion() likely skipped "
                f"the entry internally (amass_data is None, or N < 10 frames)"
            )
        if produced.stat().st_mtime < t_launch_wall:
            raise RuntimeError(
                f"{produced} exists but predates the subprocess launch: stale pkl from "
                f"an earlier run of the same stem, PHC wrote nothing this time"
            )
        shutil.move(str(produced), dst)

    print(f"PHC output → {dst}")


if __name__ == "__main__":
    main()
