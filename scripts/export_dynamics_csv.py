#!/usr/bin/env python3
"""export_dynamics_csv.py — dump a stage-05 sidecar to CSV for PlotJuggler.

PlotJuggler already does everything we were fighting uPlot for (real
checkbox-per-curve, zoom, multiple files) -- no reason to keep hand-rolling a
sidebar plot when a real tool for this exists and is now installed
(`snap install plotjuggler`).

Usage:
    modules/third_party/spider/.venv/bin/python scripts/export_dynamics_csv.py \
        --sidecar data/.../{seq}_output_dynamics.npz \
        --out /tmp/femto14.csv

Then in PlotJuggler: Ctrl+O / drag the CSV in, tick whichever curves you want
in the left panel, scrub the timeline at the bottom.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sidecar", required=True, help="{seq}_output_dynamics.npz")
    ap.add_argument("--scene", help="scene_act.xml, to resolve real actuator/joint names (optional)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    d = np.load(args.sidecar, allow_pickle=True)
    tau = d["tau"]
    n_frames, n_act = tau.shape
    frame = np.arange(n_frames)

    names = [f"joint{i}" for i in range(n_act)]
    model = None
    if args.scene:
        model = mujoco.MjModel.from_xml_path(args.scene)
        names = [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or f"joint{i}"
            for i in range(n_act)
        ]

    # Real Unitree G1 hardware torque limits (this MJCF sets none anywhere --
    # neither actuator_forcerange nor jnt_actfrcrange -- see
    # scripts/view_dynamics_contacts.py for the same table and why).
    # Verified against the official Unitree g1_29dof.urdf (<limit effort=...>,
    # modules/04_deployment/unitree_ros2/.../unitree_description/model/g1/) --
    # waist_pitch/waist_roll were previously transcribed as 50 (a stale copy
    # from holosoma_custom's robot.py); the real hardware limit is 35, same as
    # the ankles.
    g1_limit_nm = {
        "hip_pitch": 88.0, "hip_roll": 139.0, "hip_yaw": 88.0, "knee": 139.0,
        "ankle_pitch": 35.0, "ankle_roll": 35.0,
        "waist_yaw": 88.0, "waist_pitch": 35.0, "waist_roll": 35.0,
        "shoulder_pitch": 25.0, "shoulder_roll": 25.0, "shoulder_yaw": 25.0,
        "elbow": 25.0, "wrist_roll": 25.0, "wrist_pitch": 5.0, "wrist_yaw": 5.0,
    }

    def _limit_for(name: str, i: int) -> float:
        if model is not None:
            fr = model.actuator_forcerange[i]
            if abs(fr).max() > 0:
                return float(abs(fr).max())
            jid = model.actuator_trnid[i, 0]
            if model.jnt_actfrclimited[jid] and abs(model.jnt_actfrcrange[jid]).max() > 0:
                return float(abs(model.jnt_actfrcrange[jid]).max())
        for key, val in g1_limit_nm.items():
            if key in name:
                return val
        return 1.0

    columns = {"frame": frame, "time_s": frame / float(d["fps"])}
    for i, name in enumerate(names):
        columns[f"tau__{name}"] = tau[:, i]
        limit = _limit_for(name, i)
        columns[f"limit_pos__{name}"] = np.full(n_frames, limit)
        columns[f"limit_neg__{name}"] = np.full(n_frames, -limit)

    if "contact_mask" in d:
        cmask = d["contact_mask"]
        pairs = d["contact_pairs"] if "contact_pairs" in d else [f"pair{i}" for i in range(cmask.shape[1])]
        columns["n_active_contacts"] = cmask.sum(axis=1)
        for i, p in enumerate(pairs):
            columns[f"contact__{p}"] = cmask[:, i].astype(np.float64)

    if "object_poses" in d:
        op = d["object_poses"]  # (T, N, 7) = qw,qx,qy,qz,x,y,z
        names_obj = d["object_names"] if "object_names" in d else [f"obj{i}" for i in range(op.shape[1])]
        for i, oname in enumerate(names_obj):
            columns[f"{oname}__x"] = op[:, i, 4]
            columns[f"{oname}__y"] = op[:, i, 5]
            columns[f"{oname}__z"] = op[:, i, 6]

    header = list(columns.keys())
    n = len(frame)
    with open(args.out, "w") as f:
        f.write(",".join(header) + "\n")
        for t in range(n):
            f.write(",".join(f"{columns[h][t]:.6f}" for h in header) + "\n")

    print(f"wrote {args.out}  ({n} rows, {len(header)} columns)")
    print("Open in PlotJuggler: File > Load data > pick the CSV, or drag-and-drop it in.")


if __name__ == "__main__":
    main()
