#!/usr/bin/env python3
"""Run HoloV2's own retargeter (prepare -> targets -> solve) end to end and save the resulting
G1 joint trajectory to an npz. Not GMR, not holosoma -- this drives HoloV2's SQP solver
(src/solve), the same code path src.viz.app --solve uses internally, minus the viewer.

Must run with HoloV2/ as the working directory (URDF + model paths resolve relative to it,
same convention as the viewer / CHEATSHEET.md):

    cd .../HoloNew/HoloV2
    <holonew python> /path/to/retarget_to_qpos.py \\
        --dataset sfu --motion-path <clip>_sfu.npz --model-dir <smplx dir> \\
        --out <clip>_qpos.npz

Full-length solves are slow (roughly a second per frame, cvxpy+pinocchio) -- use --frame-step
to subsample for a quick check, --max-frames to cap it.
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser(description="HoloV2 retarget -> qpos npz")
    ap.add_argument("--dataset", default="sfu", help="loader key (sfu/omomo/hodome/hoim3/pahoi)")
    ap.add_argument("--motion-path", required=True)
    ap.add_argument("--model-dir", required=True, help="SMPL-X model dir (the smplx/ folder itself)")
    ap.add_argument("--dataset-root", default=None, help="release root for objects/betas (omomo/hodome/hoim3)")
    ap.add_argument("--urdf", default="models/g1/g1_29dof.urdf")
    ap.add_argument("--robot-name", default="g1")
    ap.add_argument("--dof", type=int, default=29)
    ap.add_argument("--height", type=float, default=1.3)
    ap.add_argument("--frame-step", type=int, default=1)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--backend", default="proxqp", choices=["proxqp", "cvxpy"],
                    help="proxqp needs `pip install proxsuite`; cvxpy is already in the env")
    # Trust-region radii. HoloV2's stock 0.05/0.10 lets the SQP overshoot the mm-scale contact
    # residuals and settle in a period-2 limit cycle ~5cm off the optimum (seen on femto04/06,
    # frame 0: dv pinned at +/-cap forever, predicted cost 3x below actual). 0.01/0.02 converges
    # to a true fixed point. Pass 0.05/0.10/0.10 to reproduce HoloV2 defaults.
    ap.add_argument("--tr-base-pos", type=float, default=0.01, help="base translation step cap (m/iter)")
    ap.add_argument("--tr-base-rot", type=float, default=0.02, help="base rotation step cap (rad/iter)")
    ap.add_argument("--tr-joints", type=float, default=0.02, help="joint step cap (rad/iter)")
    ap.add_argument("--iters-first", type=int, default=60, help="SQP iterations for frame 0 (cold start)")
    ap.add_argument("--iters", type=int, default=8, help="SQP iterations per warm-started frame")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    sys.path.insert(0, os.getcwd())          # run from HoloV2/: makes "src" importable
    from src.prepare.config import PrepareConfig
    from src.prepare.contracts import RobotSpec, SceneSpec
    from src.prepare.runner import prepare
    from src.targets.pipeline import process_frame
    from src.solve.config import SolveConfig
    from src.solve.runner import solve

    robot = RobotSpec(name=args.robot_name, urdf_path=Path(args.urdf), link_names=("pelvis",),
                      dof=args.dof, height=args.height)
    spec = SceneSpec(dataset=args.dataset, motion_path=Path(args.motion_path), robot=robot,
                     smpl_model_dir=Path(args.model_dir),
                     dataset_root=Path(args.dataset_root) if args.dataset_root else None)

    t0 = time.time()
    grounded, ctx = prepare(spec, PrepareConfig())
    print(f"[retarget] prepared in {time.time() - t0:.1f}s  "
          f"({grounded.n_frames} frames, fps={grounded.fps})")

    shown = list(range(0, grounded.n_frames, max(1, args.frame_step)))
    if args.max_frames is not None:
        shown = shown[:args.max_frames]
    print(f"[retarget] solving {len(shown)}/{grounded.n_frames} frames ...")

    frame_targets = [process_frame(grounded, ctx, robot, f) for f in shown]

    t0 = time.time()
    cfg = SolveConfig(backend=args.backend,
                     tr_base_pos=args.tr_base_pos, tr_base_rot=args.tr_base_rot,
                     tr_joints=args.tr_joints,
                     pen_step=min(0.02, args.tr_base_pos),
                     n_iter_first=args.iters_first, n_iter_per_frame=args.iters)
    traj = solve(grounded, ctx, frame_targets, cfg, robot_name=args.robot_name)
    print(f"[retarget] solved in {time.time() - t0:.1f}s")

    np.savez(args.out, qpos=traj.qpos, object_poses=traj.object_poses,
            frame_indices=np.asarray(shown), fps=grounded.fps / max(1, args.frame_step),
            cost=np.asarray([fi.cost for fi in traj.info]),
            status=np.asarray([fi.status for fi in traj.info], dtype="U16"))
    print(f"[retarget] wrote {args.out}  qpos={traj.qpos.shape}")


if __name__ == "__main__":
    main()
