#!/usr/bin/env python3
"""
replay_openloop.py — does the dynamics pass survive WITHOUT the MPC?

view_dynamics.py writes `qpos` straight into MjData and calls mj_forward: it
shows the states SPIDER reached, but nothing is integrated -- a trajectory that
would fall over on its own looks perfectly fine there.

This script instead replays the CONTROL sequence open loop: set the initial
state once, then feed `ctrl[t]` and let mj_step integrate. No replanning, no
state reset, no virtual assistance. If the box stays in the hands and the robot
stays upright for the whole clip, the retarget is physically self-consistent and
worth training on. If it drifts away from the MPC's own rollout, that gap is
exactly the slack the RL policy will have to make up.

Two things make this a fair test on these scenes:

  * the robot actuators are <position kp=500>, so `ctrl` is a joint POSITION
    target, not a torque -- open loop here means "no replanning", not "no PD".
  * the 6 object actuators (obj0_pos_*/obj0_rot_*) carry kp="0" kv="0", so the
    contact-guidance wrench is inert in the saved scene. The box is held by real
    contact only; its leftover ctrl columns do nothing.

Needs mujoco, which lives in SPIDER's venv, not the wbt_rl conda env:

    modules/third_party/spider/.venv/bin/python scripts/replay_openloop.py \\
        --run data/01_retargeted_motions/dynamics_test/femto14_box36

    # just the numbers, no window:
    ... --headless

    # slow motion, and ghost markers where the KINEMATIC retarget wanted to be:
    ... --speed 0.25 --ghost

`--run` locates the scene and trajectory the way enrich_dynamics.py laid them
out; pass --scene/--traj explicitly for a run with a different layout.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

# Wireframe-ish overlay colours for --ghost (RGBA).
_GHOST_BOX = np.array([0.2, 0.8, 1.0, 0.25], dtype=np.float32)
_GHOST_PELVIS = np.array([1.0, 0.6, 0.1, 0.35], dtype=np.float32)


def _run_sim_dt(scene: Path) -> float | None:
    """SPIDER's integration step for this run.

    It does NOT live in the scene XML: the shipped `<option timestep>` is the
    asset's own default (0.02) and SPIDER overrides it with `sim_dt` when it
    builds the Warp model. Replaying at the XML value would integrate at half
    the rate the controls were optimised for -- which on this clip is the
    difference between standing and falling. Read it back from the run config.
    """
    cfg = scene.parent / "0" / "config_act.yaml"
    if not cfg.exists():
        cfg = scene.parent / "0" / "config.yaml"
    if not cfg.exists():
        return None
    for line in cfg.read_text().splitlines():
        if line.startswith("sim_dt:"):
            return float(line.split(":", 1)[1])
    return None


def _locate(run: Path) -> tuple[Path, Path, Path | None]:
    """Find scene_act.xml + trajectory_mjwp*.npz under an enrich_dynamics output dir."""
    scenes = sorted(run.glob("dynamics/processed/*/*/*/*/scene_act.xml"))
    if not scenes:
        raise SystemExit(
            f"No dynamics/processed/.../scene_act.xml under {run}.\n"
            f"Pass --scene and --traj explicitly, or point --run at an enrich_dynamics --out-dir."
        )
    if len(scenes) > 1:
        raise SystemExit(
            "Several scenes under this run -- pass --scene explicitly:\n  "
            + "\n  ".join(str(s) for s in scenes)
        )
    scene = scenes[0]
    task_dir = scene.parent
    traj = task_dir / "0" / "trajectory_mjwp_act.npz"
    if not traj.exists():
        traj = task_dir / "0" / "trajectory_mjwp.npz"
    if not traj.exists():
        raise SystemExit(f"No trajectory_mjwp*.npz in {task_dir / '0'}")
    kin = task_dir / "0" / "trajectory_kinematic_act.npz"
    return scene, traj, kin if kin.exists() else None


def _flatten(block: np.ndarray) -> np.ndarray:
    """(n_replan, ctrl_steps, D) -> (n_replan*ctrl_steps, D), like the sidecar extractor."""
    return block.reshape(-1, block.shape[-1])


def _object_slice(model: mujoco.MjModel) -> slice | None:
    """qpos columns of the first dynamic object (6 DOF: xyz + XYZ euler), if any."""
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "obj0_pos_x")
    if jid < 0:
        return None
    start = int(model.jnt_qposadr[jid])
    return slice(start, start + 6)


def rollout(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    qpos0: np.ndarray,
    qvel0: np.ndarray,
    ctrl: np.ndarray,
) -> np.ndarray:
    """Integrate the whole control sequence open loop; return the visited qpos."""
    data.qpos[:] = qpos0
    data.qvel[:] = qvel0
    mujoco.mj_forward(model, data)
    out = np.empty((ctrl.shape[0], model.nq))
    for t in range(ctrl.shape[0]):
        data.ctrl[:] = ctrl[t]
        mujoco.mj_step(model, data)
        out[t] = data.qpos
    return out


def report(
    model: mujoco.MjModel,
    qpos_ol: np.ndarray,
    qpos_cl: np.ndarray,
    dt: float,
    fall_height: float,
) -> None:
    """Compare the open-loop rollout against the MPC's own closed-loop states."""
    n = min(len(qpos_ol), len(qpos_cl))
    ol, cl = qpos_ol[:n], qpos_cl[:n]
    obj = _object_slice(model)

    root_err = np.linalg.norm(ol[:, :3] - cl[:, :3], axis=1)
    joint_err = np.abs(ol[:, 7:36] - cl[:, 7:36])
    pelvis_z = ol[:, 2]

    print(f"\nOpen loop vs MPC rollout  ({n} steps, {n * dt:.2f} s @ {1 / dt:.0f} Hz)")
    print(f"  root drift      mean {root_err.mean():6.3f} m   final {root_err[-1]:6.3f} m   max {root_err.max():6.3f} m")
    print(f"  joint error     mean {joint_err.mean():6.3f} rad final {joint_err[-1].mean():6.3f} rad max {joint_err.max():6.3f} rad")
    print(f"  pelvis height   min  {pelvis_z.min():6.3f} m   final {pelvis_z[-1]:6.3f} m")

    if obj is not None:
        obj_err = np.linalg.norm(ol[:, obj][:, :3] - cl[:, obj][:, :3], axis=1)
        obj_z = ol[:, obj][:, 2]
        print(f"  object drift    mean {obj_err.mean():6.3f} m   final {obj_err[-1]:6.3f} m   max {obj_err.max():6.3f} m")
        print(f"  object height   min  {obj_z.min():6.3f} m   max   {obj_z.max():6.3f} m   final {obj_z[-1]:6.3f} m")

    fallen = np.nonzero(pelvis_z < fall_height)[0]
    if len(fallen):
        print(f"\n  ✗ FELL at t={fallen[0] * dt:.2f} s (pelvis below {fall_height} m)")
    else:
        print(f"\n  ✓ stayed upright for the whole clip (pelvis never below {fall_height} m)")

    # Compact drift timeline: one column per ~2% of the clip.
    step = max(n // 50, 1)
    scale = " .:-=+*#%@"
    lo, hi = 0.0, max(root_err.max(), 1e-6)
    bar = "".join(scale[min(int((e - lo) / (hi - lo) * (len(scale) - 1)), len(scale) - 1)]
                  for e in root_err[::step])
    print(f"\n  root drift over time (blank=0, @={hi:.2f} m):\n  {bar}")


def view(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    qpos0: np.ndarray,
    qvel0: np.ndarray,
    ctrl: np.ndarray,
    dt: float,
    speed: float,
    loop: bool,
    ghost: np.ndarray | None,
    ghost_stride: int,
) -> None:
    """Step the same rollout live in the passive viewer, paced to wall clock."""
    import mujoco.viewer

    obj = _object_slice(model)

    def reset() -> None:
        data.qpos[:] = qpos0
        data.qvel[:] = qvel0
        data.ctrl[:] = 0.0
        mujoco.mj_forward(model, data)

    reset()
    with mujoco.viewer.launch_passive(model, data) as viewer:
        print(f"{ctrl.shape[0]} steps @ {1 / dt:.0f} Hz, playback x{speed}. Close the window to stop.")
        t = 0
        while viewer.is_running():
            wall0 = time.perf_counter()
            data.ctrl[:] = ctrl[t]
            mujoco.mj_step(model, data)

            if ghost is not None:
                _draw_ghost(viewer, ghost[min(t // ghost_stride, len(ghost) - 1)], obj)
            viewer.sync()

            t += 1
            if t >= ctrl.shape[0]:
                if not loop:
                    print("End of clip. Close the window to stop.")
                    while viewer.is_running():
                        viewer.sync()
                        time.sleep(0.02)
                    return
                t = 0
                reset()

            sleep = dt / speed - (time.perf_counter() - wall0)
            if sleep > 0:
                time.sleep(sleep)


def _draw_ghost(viewer, ref_qpos: np.ndarray, obj: slice | None) -> None:
    """Overlay where the KINEMATIC retarget wanted the pelvis and the box to be."""
    scn = viewer.user_scn
    scn.ngeom = 0

    def add(gtype, size, pos, mat, rgba):
        if scn.ngeom >= len(scn.geoms):
            return
        mujoco.mjv_initGeom(scn.geoms[scn.ngeom], gtype, size, pos, mat, rgba)
        scn.ngeom += 1

    eye = np.eye(3).flatten()
    add(mujoco.mjtGeom.mjGEOM_SPHERE, np.array([0.06, 0, 0]),
        ref_qpos[:3].copy(), eye, _GHOST_PELVIS)

    if obj is not None:
        pos = ref_qpos[obj][:3].copy()
        mat = np.zeros(9)
        quat = np.zeros(4)
        # Lower case, not "XYZ": spider_build_scene writes these angles with
        # scipy's `as_euler("XYZ")`, which is INTRINSIC, and mju_euler2Quat
        # spells intrinsic in lower case (upper case is extrinsic). The same
        # three letters mean opposite things in the two libraries; reading them
        # back as "XYZ" turns the ghost box by ~100 deg on this clip.
        mujoco.mju_euler2Quat(quat, ref_qpos[obj][3:6].copy(), "xyz")
        mujoco.mju_quat2Mat(mat, quat)
        add(mujoco.mjtGeom.mjGEOM_BOX, np.array([0.18, 0.18, 0.18]), pos, mat, _GHOST_BOX)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", type=Path,
                   help="enrich_dynamics --out-dir; finds scene + trajectory under it")
    p.add_argument("--scene", type=Path, help="scene_act.xml (overrides --run)")
    p.add_argument("--traj", type=Path, help="trajectory_mjwp_act.npz (overrides --run)")
    p.add_argument("--reference", type=Path,
                   help="trajectory_kinematic_act.npz for the --ghost overlay")
    p.add_argument("--headless", action="store_true", help="no window, just the drift report")
    p.add_argument("--speed", type=float, default=1.0, help="playback speed (0.25 = slow motion)")
    p.add_argument("--loop", action="store_true", help="restart from the initial state at the end")
    p.add_argument("--ghost", action="store_true",
                   help="overlay the kinematic reference pelvis + box")
    p.add_argument("--fall-height", type=float, default=0.4,
                   help="pelvis height below which the robot counts as fallen")
    p.add_argument("--sim-dt", type=float,
                   help="integration step (default: the run's sim_dt, NOT the scene XML timestep)")
    args = p.parse_args()

    if args.scene and args.traj:
        scene, traj, kin = args.scene, args.traj, args.reference
    elif args.run:
        scene, traj, kin = _locate(args.run.resolve())
        kin = args.reference or kin
    else:
        p.error("give either --run, or both --scene and --traj")

    print(f"scene      {scene}")
    print(f"trajectory {traj}")

    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)

    sim_dt = args.sim_dt or _run_sim_dt(Path(scene))
    if sim_dt is None:
        print(f"  (no sim_dt in the run config -- falling back to the XML timestep "
              f"{model.opt.timestep}; pass --sim-dt if the replay diverges)", file=sys.stderr)
    else:
        if abs(sim_dt - model.opt.timestep) > 1e-12:
            print(f"timestep   {model.opt.timestep} (scene XML) -> {sim_dt} (run sim_dt)")
        model.opt.timestep = sim_dt
    dt = float(model.opt.timestep)

    raw = np.load(traj)
    qpos_cl = _flatten(raw["qpos"])
    qvel_cl = _flatten(raw["qvel"])
    ctrl = _flatten(raw["ctrl"])
    if ctrl.shape[1] != model.nu:
        raise SystemExit(f"trajectory has {ctrl.shape[1]} controls, scene expects {model.nu}")

    ghost = None
    ghost_stride = 1
    if args.ghost:
        if kin is None or not Path(kin).exists():
            print("  (no kinematic reference found -- skipping the ghost overlay)", file=sys.stderr)
        else:
            ghost = np.load(kin)["qpos"]
            # The kinematic clip lives at ref_dt, the rollout at sim_dt.
            ghost_stride = max(int(round(len(ctrl) / len(ghost))), 1)
            print(f"ghost      {kin}  ({len(ghost)} frames, 1 per {ghost_stride} steps)")

    qpos_ol = rollout(model, data, qpos_cl[0], qvel_cl[0], ctrl)
    report(model, qpos_ol, qpos_cl, dt, args.fall_height)

    if not args.headless:
        view(model, data, qpos_cl[0], qvel_cl[0], ctrl, dt,
             args.speed, args.loop, ghost, ghost_stride)


if __name__ == "__main__":
    main()
