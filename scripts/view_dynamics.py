#!/usr/bin/env python3
"""
view_dynamics.py — replay a stage 05 dynamics run.

Loads the already-computed sidecar (no re-simulation) and plays it back.
Two backends:

  --backend mujoco   native passive viewer (mujoco.viewer), a local window,
                      no server. Closest to "just watch the physics."
  --backend viser     web viewer, like HoloNew's viser_player.py: mesh toggle,
                      timeline scrubber, and a semi-transparent ghost of the
                      original kinematic reference (before the physics pass)
                      so you can see exactly where SPIDER corrected it.

Needs mujoco (+ viser, for that backend), which live in SPIDER's venv, not the
wbt_rl conda env:

    modules/third_party/spider/.venv/bin/python scripts/view_dynamics.py \\
        --scene   data/.../dynamics/processed/.../scene_act.xml \\
        --sidecar data/.../{seq}_output_dynamics.npz \\
        --backend mujoco

    # viser, with the pre-physics reference as a ghost overlay:
    ... --backend viser --reference data/.../dynamics/processed/.../0/trajectory_kinematic_act.npz
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco
import numpy as np


def _load(sidecar: Path) -> tuple[np.ndarray, int]:
    d = np.load(sidecar, allow_pickle=True)
    return d["qpos"], int(d["fps"])


def run_mujoco(scene: Path, qpos: np.ndarray, fps: int, loop: bool) -> None:
    import mujoco.viewer

    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    if qpos.shape[1] != model.nq:
        raise ValueError(f"sidecar qpos has {qpos.shape[1]} dims, scene expects {model.nq}")

    dt = 1.0 / fps
    with mujoco.viewer.launch_passive(model, data) as viewer:
        t = 0
        print(f"{qpos.shape[0]} frames @ {fps}Hz. Close the window to stop.")
        while viewer.is_running():
            step0 = time.perf_counter()
            data.qpos[:] = qpos[t]
            mujoco.mj_forward(model, data)  # recompute contacts/derived state for display
            viewer.sync()
            t += 1
            if t >= qpos.shape[0]:
                if not loop:
                    break
                t = 0
            elapsed = time.perf_counter() - step0
            time.sleep(max(0.0, dt - elapsed))


def run_viser(scene: Path, qpos: np.ndarray, fps: int, reference: Path | None) -> None:
    from spider.viewers import viser_viewer

    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    data_ref = mujoco.MjData(model) if reference else None

    qpos_ref = None
    if reference is not None:
        ref = np.load(reference)["qpos"]
        n = min(len(ref), len(qpos))
        qpos_ref = ref[:n]
        qpos = qpos[:n]

    viser_viewer.init_viser(app_name="wbt-rl-dynamics")
    _, _, body_entity_and_ids = viser_viewer.build_and_log_scene(scene)
    server = viser_viewer._get_server()

    print(f"http://localhost:{server.get_port()}  —  {qpos.shape[0]} frames @ {fps}Hz")
    if qpos_ref is not None:
        print("Reference ghost = pre-physics kinematic trajectory (toggle in the 'Reference' checkbox).")

    dt = 1.0 / fps
    for t in range(qpos.shape[0]):
        data.qpos[:] = qpos[t]
        mujoco.mj_kinematics(model, data)
        if data_ref is not None:
            data_ref.qpos[:] = qpos_ref[t]
            mujoco.mj_kinematics(model, data_ref)
        viser_viewer.log_frame(
            data, sim_time=t * dt, viewer_body_entity_and_ids=body_entity_and_ids,
            data_ref=data_ref, playback_fps=float(fps),
        )
        # log_frame only records history + bumps the slider's max -- it never
        # actually pushes positions to the browser (that's _render_frame, wired
        # to the slider's on_update). Without this, every body sits wherever
        # add_frame() put it initially (world origin) until you touch the
        # slider by hand: everything looks piled up in the floor the whole time.
        viser_viewer._STATE.playback_slider.value = t
        viser_viewer._render_frame(viewer_body_entity_and_ids=body_entity_and_ids)
        time.sleep(dt)

    print("Loaded. Use the Timeline slider / Play buttons in the browser.")
    while True:
        time.sleep(3600)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", required=True, help="scene_act.xml from the run's dynamics/ tree")
    ap.add_argument("--sidecar", required=True, help="{seq}_output_dynamics.npz")
    ap.add_argument("--backend", choices=["mujoco", "viser"], default="mujoco")
    ap.add_argument("--reference", help="viser only: trajectory_kinematic_act.npz for the ghost overlay")
    ap.add_argument("--loop", action="store_true", help="mujoco only: loop playback")
    args = ap.parse_args()

    qpos, fps = _load(Path(args.sidecar))

    if args.backend == "mujoco":
        run_mujoco(Path(args.scene), qpos, fps, args.loop)
    else:
        run_viser(Path(args.scene), qpos, fps, Path(args.reference) if args.reference else None)


if __name__ == "__main__":
    main()
