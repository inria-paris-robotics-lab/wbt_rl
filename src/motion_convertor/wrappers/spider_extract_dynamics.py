"""
Turn a finished SPIDER run into the WBT-RL dynamics sidecar.

Runs inside SPIDER's own venv (needs mujoco); invoked through `venv_run()`.

SPIDER saves its MPC rollout as `(n_replan, ctrl_steps, ...)` blocks of
generalised coordinates. This script flattens them back to a per-frame
trajectory, resamples to the reference rate, and recovers the two quantities
SPIDER does NOT persist but which are the whole point of the 05_dynamics stage:

  * tau            joint torque actually applied, read off `data.qfrc_actuator`
                   after replaying each frame through `mj_forward`. The robot
                   actuators are <position kp dampratio>, NOT <motor> -- `ctrl`
                   is a position SETPOINT (radians), not a torque, so it cannot
                   be used directly. qfrc_actuator is what MuJoCo's own PD law
                   (tau = kp*(ctrl-q) - kv*qdot) resolves that setpoint to; it
                   is in true N*m and needs no further conversion. It is still
                   specific to the actuator gains this scene compiles with
                   (SPIDER's stock kp=500, deliberately much stiffer than any
                   real G1 joint -- see cfg/05_dynamics/spider_realgains.yaml
                   for why matching real gains here doesn't work). Treat it as
                   an estimate of the torque the MOTION itself demands (valid
                   because kp=500 tracks the kinematic reference tightly, ~13
                   mrad mean error), not as ground truth for what a real-gain
                   controller would apply -- consume it as a soft one-sided
                   envelope downstream, never as an absolute per-joint target.
  * contact_force  per-contact-pair wrench, recomputed by replaying each frame
                   through `mj_forward` so MuJoCo's contact solver runs again

Contact forces come back in the CONTACT frame: component 0 is the normal, 1-2
are the friction tangents. Magnitudes are only as meaningful as the scene's
`solref`/`solimp` — treat them as relative, not calibrated, unless the contact
model has been tuned for the robot in question.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as Rsp


def _flatten(block: np.ndarray) -> np.ndarray:
    """(n_replan, ctrl_steps, D) -> (n_replan*ctrl_steps, D)."""
    return block.reshape(-1, block.shape[-1])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True, help="SPIDER task dir (holds scene_act.xml)")
    ap.add_argument("--data-id", default="0")
    ap.add_argument("--out", required=True, help="destination {seq}_output_dynamics.npz")
    ap.add_argument(
        "--ref-frames",
        type=int,
        default=0,
        help="resample to this many frames (the retargeted clip length); 0 keeps the raw sim rate",
    )
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    scene = run_dir / "scene_act.xml"
    # spider/config.py names this trajectory_mjwp_act.npz when contact_guidance
    # is on, trajectory_mjwp.npz (no suffix) when it's off -- this script has
    # no reason to care which, so just pick whichever the run actually wrote.
    data_dir = run_dir / str(args.data_id)
    traj = data_dir / "trajectory_mjwp_act.npz"
    if not traj.exists():
        traj = data_dir / "trajectory_mjwp.npz"
    meta = json.loads((run_dir / "wbt_scene_meta.json").read_text())

    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)

    out = np.load(traj)
    qpos = _flatten(out["qpos"])
    qvel = _flatten(out["qvel"])
    ctrl = _flatten(out["ctrl"])

    # SPIDER integrates at sim_dt; the retargeted clip lives at its own fps.
    # Subsample rather than interpolate: these are exact simulated states, and
    # interpolating them would invent physically inconsistent intermediate poses.
    n_raw = qpos.shape[0]
    if args.ref_frames and args.ref_frames > 0 and n_raw >= args.ref_frames:
        step = max(int(round(n_raw / args.ref_frames)), 1)
        idx = np.arange(0, n_raw, step)[: args.ref_frames]
    else:
        idx = np.arange(n_raw)
    qpos, qvel, ctrl = qpos[idx], qvel[idx], ctrl[idx]
    n_frames = qpos.shape[0]

    n_robot_act = int(meta["n_robot_actuators"])

    # qfrc_actuator is indexed by DOF, not by actuator -- map each of the first
    # n_robot_act actuators to its joint's DOF address rather than assuming DOF
    # order matches actuator-declaration order (true here, but this is the
    # correct way to not rely on it).
    robot_act_dofadr = np.array(
        [model.jnt_dofadr[model.actuator_trnid[i, 0]] for i in range(n_robot_act)]
    )

    # --- object poses back to the unified wxyz+xyz convention -----------------
    dynamic_names = list(meta["dynamic_names"])
    n_obj = len(dynamic_names)
    object_poses = np.zeros((n_frames, n_obj, 7), dtype=np.float32)
    n_obj_dof = 6 * n_obj
    for i in range(n_obj):
        s = qpos.shape[1] - n_obj_dof + 6 * i
        pos = qpos[:, s : s + 3]
        quat_xyzw = Rsp.from_euler("XYZ", qpos[:, s + 3 : s + 6]).as_quat()
        object_poses[:, i, :4] = quat_xyzw[:, [3, 0, 1, 2]]  # -> wxyz
        object_poses[:, i, 4:] = pos

    # --- contact forces, pair by pair ----------------------------------------
    pair_names, pair_geoms = [], []
    for pid in range(model.npair):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_PAIR, pid)
        if name is None:
            continue
        g1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(model.pair_geom1[pid]))
        g2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(model.pair_geom2[pid]))
        pair_names.append(name)
        pair_geoms.append(frozenset((g1, g2)))

    n_pairs = len(pair_names)
    contact_force = np.zeros((n_frames, n_pairs, 3), dtype=np.float32)
    contact_torque = np.zeros((n_frames, n_pairs, 3), dtype=np.float32)
    contact_mask = np.zeros((n_frames, n_pairs), dtype=bool)
    geoms_to_pair = {g: k for k, g in enumerate(pair_geoms)}
    tau = np.zeros((n_frames, n_robot_act), dtype=np.float32)

    f6 = np.zeros(6)
    for t in range(n_frames):
        data.qpos[:] = qpos[t]
        data.qvel[:] = qvel[t]
        data.ctrl[:] = ctrl[t]
        mujoco.mj_forward(model, data)
        tau[t] = data.qfrc_actuator[robot_act_dofadr]
        for c in range(data.ncon):
            con = data.contact[c]
            n1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, con.geom1)
            n2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, con.geom2)
            k = geoms_to_pair.get(frozenset((n1, n2)))
            if k is None:
                continue
            mujoco.mj_contactForce(model, data, c, f6)
            # Several contact points may belong to one pair; sum them.
            contact_force[t, k] += f6[:3]
            contact_torque[t, k] += f6[3:]
            contact_mask[t, k] = True

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        qpos=qpos.astype(np.float32),
        qvel=qvel.astype(np.float32),
        tau=tau,
        object_names=np.array(dynamic_names, dtype=object),
        object_poses=object_poses,
        contact_pairs=np.array(pair_names, dtype=object),
        contact_force=contact_force,
        contact_torque=contact_torque,
        contact_mask=contact_mask,
        fps=np.int32(meta["fps"]),
        n_robot_actuators=np.int32(n_robot_act),
    )

    active = contact_mask.mean(axis=0)
    print(f"dynamics sidecar -> {out_path}")
    print(f"  frames={n_frames}  tau={tau.shape}  objects={n_obj}  pairs={n_pairs}")
    print(f"  |tau| mean={np.abs(tau).mean():.3f} max={np.abs(tau).max():.3f} Nm")
    for k, name in enumerate(pair_names):
        if active[k] > 0:
            mag = np.linalg.norm(contact_force[:, k], axis=1)
            print(
                f"  {name:28s} active {active[k]:5.1%}  "
                f"F mean={mag[contact_mask[:, k]].mean():7.2f} max={mag.max():7.2f} N"
            )


if __name__ == "__main__":
    main()
