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
                   for why matching real gains here doesn't work).

                   qfrc_actuator is DOMINATED by that kp=500 rig's own
                   position-chasing term whenever tracking error is nonzero --
                   measured on femto14_box36, ~70-75% of the peak wrist torque
                   was kp*(ctrl-q), not the physical grip. Confirmed by feeding
                   the EXACT rollout's own acceleration into mj_inverse: it
                   reproduces qfrc_actuator almost exactly (a peak of -71.6 vs
                   -63.8 N.m), because inverse dynamics faithfully explains
                   whatever acceleration a state actually had -- including one
                   produced by an overly-stiff controller's overshoot. It does
                   NOT strip the artifact for free; see tau_inverse below for
                   what does.
  * tau_inverse    joint torque from `mj_inverse` on a SMOOTHED version of the
                   SAME rollout (Savitzky-Golay on qvel, `--smooth-window`
                   samples, order 3, differentiated analytically for qacc --
                   far less noise-amplifying than a raw finite difference).
                   Smoothing removes the kp=500 controller's high-frequency
                   correction jerk while keeping the validated motion shape
                   (still holds the box, still reaches the table); mj_inverse
                   then reads off the force that SMOOTH motion requires under
                   the true dynamics (mass matrix, Coriolis, gravity, the same
                   contact forces below) -- independent of any actuator gain.
                   Away from contact transitions this reads as a genuinely
                   small, physically sane number (~6-9 N.m at the wrist on
                   femto14_box36, close to the real G1's 5 N.m limit, instead
                   of tau's 60-70 N.m). AT a contact transition (making or
                   breaking hand<->object contact, detect via a `contact_mask`
                   edge) it spikes for real -- first touch is a genuine
                   impulsive event even on hardware -- so don't smooth or
                   penalize those frames the same way as steady contact.
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
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation as Rsp


def _flatten(block: np.ndarray) -> np.ndarray:
    """(n_replan, ctrl_steps, D) -> (n_replan*ctrl_steps, D)."""
    return block.reshape(-1, block.shape[-1])


def _run_sim_dt(run_dir: Path, data_id: str) -> float | None:
    """SPIDER's real integration step for this run.

    NOT the scene XML's `<option timestep>`: that's the unitree_g1 asset's own
    shipped default (0.02), and SPIDER overrides it with `sim_dt` when it
    builds the Warp model -- it never writes that override back into
    scene_act.xml. Contact solving (`solref`'s "positive" form) and the
    tau_inverse acceleration below both depend on the ACTUAL step, so getting
    this wrong silently mis-scales the contact force we read back off a
    correctly-simulated trajectory. Mirrors scripts/replay_openloop.py's
    identical fix for the same file.
    """
    for name in ("config_act.yaml", "config.yaml"):
        cfg = run_dir / data_id / name
        if cfg.exists():
            for line in cfg.read_text().splitlines():
                if line.startswith("sim_dt:"):
                    return float(line.split(":", 1)[1])
    return None


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
    ap.add_argument(
        "--smooth-window",
        type=int,
        default=21,
        help="Savitzky-Golay window (samples, odd) for tau_inverse's qacc -- "
             "21 @ sim_dt=0.01 (210ms) tested on femto14_box36: removes the "
             "kp=500 jerk while keeping genuine contact-transition spikes",
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
    sim_dt = _run_sim_dt(run_dir, args.data_id)
    if sim_dt is None:
        print(f"  (no sim_dt in {run_dir}/{args.data_id}/config_act.yaml -- "
              f"falling back to the scene XML's {model.opt.timestep}; "
              f"contact forces and tau_inverse may be mis-scaled)")
    else:
        model.opt.timestep = sim_dt
    dt = float(model.opt.timestep)
    data = mujoco.MjData(model)

    out = np.load(traj)
    qpos_raw = _flatten(out["qpos"])
    qvel_raw = _flatten(out["qvel"])
    ctrl_raw = _flatten(out["ctrl"])
    n_raw = qpos_raw.shape[0]

    n_robot_act = int(meta["n_robot_actuators"])

    # qfrc_actuator/qfrc_inverse are indexed by DOF, not by actuator -- map
    # each of the first n_robot_act actuators to its joint's DOF address
    # rather than assuming DOF order matches actuator-declaration order (true
    # here, but this is the correct way to not rely on it).
    robot_act_dofadr = np.array(
        [model.jnt_dofadr[model.actuator_trnid[i, 0]] for i in range(n_robot_act)]
    )

    # tau_inverse's acceleration: Savitzky-Golay on the RAW (un-subsampled)
    # qvel, analytic first derivative for qacc. Needs the dense raw-rate
    # series to smooth meaningfully; done here, before subsampling, on ALL nv
    # columns at once so inter-joint coupling terms (mass matrix, Coriolis)
    # stay consistent when mj_inverse reads them back.
    win = min(args.smooth_window, n_raw - (1 - n_raw % 2))  # must be odd, <= n_raw
    if win % 2 == 0:
        win -= 1
    qacc_smooth = (
        savgol_filter(qvel_raw, win, 3, deriv=1, delta=dt, axis=0)
        if win >= 5 else np.gradient(qvel_raw, dt, axis=0)
    )

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
    contact_force_raw = np.zeros((n_raw, n_pairs, 3), dtype=np.float32)
    contact_torque_raw = np.zeros((n_raw, n_pairs, 3), dtype=np.float32)
    contact_mask_raw = np.zeros((n_raw, n_pairs), dtype=bool)
    geoms_to_pair = {g: k for k, g in enumerate(pair_geoms)}
    tau_raw = np.zeros((n_raw, n_robot_act), dtype=np.float32)
    tau_inverse_raw = np.zeros((n_raw, n_robot_act), dtype=np.float32)

    f6 = np.zeros(6)
    for t in range(n_raw):
        data.qpos[:] = qpos_raw[t]
        data.qvel[:] = qvel_raw[t]
        data.ctrl[:] = ctrl_raw[t]
        mujoco.mj_forward(model, data)  # kp=500's actual control + contacts at this state
        tau_raw[t] = data.qfrc_actuator[robot_act_dofadr]
        for c in range(data.ncon):
            con = data.contact[c]
            n1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, con.geom1)
            n2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, con.geom2)
            k = geoms_to_pair.get(frozenset((n1, n2)))
            if k is None:
                continue
            mujoco.mj_contactForce(model, data, c, f6)
            # Several contact points may belong to one pair; sum them.
            contact_force_raw[t, k] += f6[:3]
            contact_torque_raw[t, k] += f6[3:]
            contact_mask_raw[t, k] = True

        # Same state's mass matrix/bias/contacts (mj_forward above), but the
        # SMOOTHED acceleration instead of whatever kp=500's overshoot
        # produced -- this is what actually removes the PD-chase artifact
        # (feeding mj_inverse the rollout's OWN raw qacc just reconstructs
        # qfrc_actuator, since inverse dynamics faithfully explains whatever
        # acceleration a state had, jerk included).
        data.qacc[:] = qacc_smooth[t]
        mujoco.mj_inverse(model, data)
        tau_inverse_raw[t] = data.qfrc_inverse[robot_act_dofadr]

    # SPIDER integrates at sim_dt; the retargeted clip lives at its own fps.
    # Subsample rather than interpolate: these are exact simulated states, and
    # interpolating them would invent physically inconsistent intermediate
    # poses. Deterministic per-state computations (everything above) give the
    # same values whether subsampled before or after, so doing it after here
    # doesn't change tau/contact_force -- it only lets tau_inverse's qacc see
    # the full raw-rate series for smoothing.
    if args.ref_frames and args.ref_frames > 0 and n_raw >= args.ref_frames:
        step = max(int(round(n_raw / args.ref_frames)), 1)
        idx = np.arange(0, n_raw, step)[: args.ref_frames]
    else:
        idx = np.arange(n_raw)
    qpos, qvel = qpos_raw[idx], qvel_raw[idx]
    tau, tau_inverse = tau_raw[idx], tau_inverse_raw[idx]
    contact_force = contact_force_raw[idx]
    contact_torque = contact_torque_raw[idx]
    contact_mask = contact_mask_raw[idx]
    n_frames = qpos.shape[0]

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

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        qpos=qpos.astype(np.float32),
        qvel=qvel.astype(np.float32),
        tau=tau,
        tau_inverse=tau_inverse,
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
    print(f"  frames={n_frames}  tau={tau.shape}  objects={n_obj}  pairs={n_pairs}  sim_dt={dt}")
    print(f"  |tau|         mean={np.abs(tau).mean():.3f} max={np.abs(tau).max():.3f} Nm")
    print(f"  |tau_inverse| mean={np.abs(tau_inverse).mean():.3f} max={np.abs(tau_inverse).max():.3f} Nm")
    for k, name in enumerate(pair_names):
        if active[k] > 0:
            mag = np.linalg.norm(contact_force[:, k], axis=1)
            print(
                f"  {name:28s} active {active[k]:5.1%}  "
                f"F mean={mag[contact_mask[:, k]].mean():7.2f} max={mag.max():7.2f} N"
            )


if __name__ == "__main__":
    main()
