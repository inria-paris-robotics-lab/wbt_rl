#!/usr/bin/env python3
"""Package a retarget_to_qpos.py output into the run directory scripts/train.py expects.

train.py resolves data/01_retargeted_motions/{dataset}_{robot}/{RETARGETER}/{run_id}/, reading
any *_output_raw.npz there (falls back to defaults when config.yaml is absent: task_type=
robot_only, object_name=ground). holosoma_custom's native_output_format (cfg/01_retargeting/
holosoma_custom.yaml) is just qpos (T,nq) float64 + fps (scalar) -- verified against an existing
manual run; extra keys aren't needed so this writes a clean minimal file.

Verified compatible: HoloV2's G1_29dof.urdf and holosoma_custom's own G1_29dof.urdf have the same
actuated-joint order (the only diff is 4 extra fixed/decorative finger-tip frames on the
holosoma_custom side) -- a HoloV2 qpos is a drop-in for this feed.

Base quaternion convention differs though: HoloV2's qpos comes from pinocchio's free-flyer
(q[3:7] = [qx,qy,qz,qw], see HoloV2/src/solve/retract.py), but holosoma_custom's MuJoCo bridge
(data_conversion/convert_data_format_mj.py) reads q[3:7] as a raw slice assuming MuJoCo's own
[qw,qx,qy,qz]. Feeding pinocchio's order straight through silently misreads the rotation (a
standing-upright quat gets read as a ~180 degree flip) -- reordered to wxyz below.
"""
import argparse
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser(description="qpos.npz -> holosoma_custom training run")
    ap.add_argument("--qpos", required=True, help="output of retarget_to_qpos.py")
    ap.add_argument("--name", required=True, help="sequence name -> <name>_output_raw.npz")
    ap.add_argument("--dataset", default="SFU")
    ap.add_argument("--robot", default="G1_29dof")
    ap.add_argument("--repo-root", default=str(Path.home() / "Documents/wbt_rl"))
    ap.add_argument("--run-id", default=None, help="default: run_<name>")
    ap.add_argument("--latest", action="store_true", help="point the 'latest' symlink here")
    args = ap.parse_args()

    d = np.load(args.qpos)
    qpos = np.asarray(d["qpos"], np.float64).copy()
    fps = np.int64(round(float(np.asarray(d["fps"]))))

    # pinocchio [qx,qy,qz,qw] -> holosoma_custom/MuJoCo [qw,qx,qy,qz]
    qpos[:, 3:7] = qpos[:, [6, 3, 4, 5]]

    run_id = args.run_id or f"run_{args.name}"
    run_dir = (Path(args.repo_root) / "data" / "01_retargeted_motions"
              / f"{args.dataset}_{args.robot}" / "HOLOSOMA_CUSTOM" / run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    out = run_dir / f"{args.name}_output_raw.npz"
    np.savez(out, qpos=qpos, fps=fps)
    print(f"[qpos->run] wrote {out}  qpos={qpos.shape} fps={int(fps)}")

    if args.latest:
        link = run_dir.parent / "latest"
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(run_dir.name)
        print(f"[qpos->run] latest -> {run_dir.name}")


if __name__ == "__main__":
    main()
