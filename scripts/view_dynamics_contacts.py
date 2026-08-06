#!/usr/bin/env python3
"""
view_dynamics_contacts.py — replay a stage 05 run with contact markers.

Same spirit as HoloNew's viser tools (viser_player.py / examples/view_contact.py):
a standalone, tyro-driven script meant to be launched directly and hacked, not
called through a wrapper. What this adds over view_dynamics.py's plain
playback is exactly what view_contact.py does for retargeting -- a marker at
every ACTIVE contact, colored by force magnitude (blue = low, red = high) --
plus a PlotJuggler-style torque panel: one line per actuator, normalised to
its own actuatorfrcrange so every joint shares the same +-1 = "at its torque
limit" scale, with a live cursor tracking the current frame.

Contact positions are re-solved live via mj_forward each frame rather than
read from the sidecar, which stores force/torque/mask per pair but not
position -- recomputing is cheap (this is a single CPU mj_forward, not the
GPU batch SPIDER ran) and keeps the sidecar schema untouched.

Needs mujoco + viser + tyro, which live in SPIDER's venv:

    modules/third_party/spider/.venv/bin/python scripts/view_dynamics_contacts.py \\
        --scene   data/.../dynamics/processed/.../scene_act.xml \\
        --sidecar data/.../{seq}_output_dynamics.npz \\
        [--reference data/.../dynamics/processed/.../0/trajectory_kinematic_act.npz]

Once running: drag the Timeline slider (or hit Play) and watch the dots move
with the contacts. "Show contact markers" and "Marker scale" are under the
Contacts folder. Feel free to edit `_color_by_force` / add your own overlays --
this file is meant to be edited, not imported.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
import tyro
from viser import uplot

from spider.viewers import viser_viewer


@dataclass
class Config:
    scene: str
    """scene_act.xml from the run's dynamics/ tree."""
    sidecar: str
    """{seq}_output_dynamics.npz."""
    reference: str | None = None
    """Optional trajectory_kinematic_act.npz for a pre-physics ghost overlay."""
    force_scale: float = 0.006
    """Marker radius (m) per newton of contact force, initial value."""
    min_force_n: float = 1.0
    """Contacts below this magnitude are not drawn (reduces clutter from noise)."""
    force_color_max_n: float = 200.0
    """Force magnitude that maps to full red; scale to your run's own peak."""
    fps: float | None = None
    """Override the sidecar's own fps."""


def _color_by_force(mag: float, vmax: float) -> tuple[int, int, int]:
    t = float(np.clip(mag / max(vmax, 1e-6), 0.0, 1.0))
    return (int(255 * t), 60, int(255 * (1 - t)))  # blue (low) -> red (high)


def main(cfg: Config) -> None:
    model = mujoco.MjModel.from_xml_path(cfg.scene)
    data = mujoco.MjData(model)
    data_ref = mujoco.MjData(model) if cfg.reference else None

    d = np.load(cfg.sidecar)
    qpos = d["qpos"]
    fps = cfg.fps or float(d["fps"])
    tau = d["tau"] if "tau" in d else None

    qpos_ref = None
    if cfg.reference:
        ref = np.load(cfg.reference)["qpos"]
        n = min(len(ref), len(qpos))
        qpos, qpos_ref = qpos[:n], ref[:n]

    viser_viewer.init_viser(app_name="wbt-rl-dynamics-contacts")
    _, _, body_entity_and_ids = viser_viewer.build_and_log_scene(Path(cfg.scene))
    # Reaches into spider's viewer module state directly -- there is no public
    # accessor, and this script is meant to be read and edited, not imported.
    server = viser_viewer._get_server()

    with server.gui.add_folder("Contacts"):
        show_cb = server.gui.add_checkbox("Show contact markers", initial_value=True)
        scale_slider = server.gui.add_slider(
            "Marker scale", min=0.001, max=0.05, step=0.001, initial_value=cfg.force_scale
        )

    torque_plot = None
    if tau is not None:
        n_act = tau.shape[1]
        # This particular MJCF sets no torque limit anywhere -- neither
        # actuator_forcerange nor jnt_actfrcrange (both come back all-zero).
        # So there is genuinely no in-model "threshold" to read: the number
        # below is the real Unitree G1 hardware spec (same values already
        # hardcoded in holosoma_custom's joint <default> classes), used here
        # purely as a reference to plot against, not something MuJoCo enforces
        # in this scene.
        _G1_TORQUE_LIMIT_NM = {
            "hip_pitch": 88.0, "hip_roll": 139.0, "hip_yaw": 88.0, "knee": 139.0,
            "ankle_pitch": 35.0, "ankle_roll": 35.0,
            "waist_yaw": 88.0, "waist_pitch": 50.0, "waist_roll": 50.0,
            "shoulder_pitch": 25.0, "shoulder_roll": 25.0, "shoulder_yaw": 25.0,
            "elbow": 25.0, "wrist_roll": 25.0, "wrist_pitch": 5.0, "wrist_yaw": 5.0,
        }

        def _torque_limit(actuator_name: str, joint_id: int) -> float:
            frcrange = model.actuator_forcerange[
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
            ]
            if abs(frcrange).max() > 0:
                return float(abs(frcrange).max())
            if model.jnt_actfrclimited[joint_id] and abs(model.jnt_actfrcrange[joint_id]).max() > 0:
                return float(abs(model.jnt_actfrcrange[joint_id]).max())
            for key, val in _G1_TORQUE_LIMIT_NM.items():
                if key in actuator_name:
                    return val
            return 1.0  # unrecognised joint name -- last-resort fallback, not a real limit

        limit = np.array([
            _torque_limit(
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or f"act{i}",
                model.actuator_trnid[i, 0],
            )
            for i in range(n_act)
        ])
        tau_norm = tau / limit[None, :]
        frames = np.arange(tau.shape[0], dtype=np.float64)

        names = [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or f"act{i}"
            for i in range(n_act)
        ]
        palette = [
            "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231", "#911eb4",
            "#46f0f0", "#f032e6", "#bcf60c", "#fabebe", "#008080", "#e6beff",
        ]
        data_tuple = (frames,) + tuple(tau_norm[:, i] for i in range(n_act)) + (
            np.ones_like(frames), -np.ones_like(frames),
        )
        series_tuple = (
            uplot.Series(label="frame"),
            *(
                uplot.Series(label=names[i], stroke=palette[i % len(palette)], width=1.3)
                for i in range(n_act)
            ),
            uplot.Series(label="+limit", stroke="#888888", width=1.0, dash=(4, 4)),
            uplot.Series(label="-limit", stroke="#888888", width=1.0, dash=(4, 4)),
        )
        with server.gui.add_folder("Torques (fraction of actuatorfrcrange)"):
            torque_plot = server.gui.add_uplot(
                data=data_tuple,
                series=series_tuple,
                title="tau / actuator limit  (+-1 = at the limit)",
                # uPlot treats the x-series as a Unix timestamp by default --
                # our frame numbers (0..326) read as seconds since 1970, which
                # renders as a garbled date axis stuck near the epoch. This is
                # what "l'abscisse est cassee" was: not a data bug, an axis-type
                # default we never overrode.
                scales={"x": uplot.Scale(time=False)},
                cursor=uplot.Cursor(x=True, y=False),
                legend=uplot.Legend(show=True),
                aspect=1.6,
            )

            # uPlot's own click-to-toggle-series-in-legend isn't wired through
            # viser's wrapper, so "show every joint, let me uncheck the ones I
            # don't want" needs real checkboxes instead: one per limb group,
            # flipping the `show` flag on that group's series and pushing a new
            # `series` tuple to the plot (series/data are the mutable fields).
            groups = {
                "left leg": [i for i, n in enumerate(names) if n.startswith("left_hip") or n.startswith("left_knee") or n.startswith("left_ankle")],
                "right leg": [i for i, n in enumerate(names) if n.startswith("right_hip") or n.startswith("right_knee") or n.startswith("right_ankle")],
                "waist": [i for i, n in enumerate(names) if n.startswith("waist")],
                "left arm": [i for i, n in enumerate(names) if n.startswith("left_shoulder") or n.startswith("left_elbow") or n.startswith("left_wrist")],
                "right arm": [i for i, n in enumerate(names) if n.startswith("right_shoulder") or n.startswith("right_elbow") or n.startswith("right_wrist")],
            }
            group_cb = {}
            for gname, idxs in groups.items():
                group_cb[gname] = server.gui.add_checkbox(gname, initial_value=True)

            def _rebuild_series(_=None) -> None:
                new_series = [series_tuple[0]]
                for i in range(n_act):
                    show = any(i in idxs and group_cb[g].value for g, idxs in groups.items())
                    s = dict(series_tuple[i + 1])
                    s["show"] = show
                    new_series.append(s)
                new_series.append(series_tuple[-2])
                new_series.append(series_tuple[-1])
                torque_plot.series = tuple(new_series)

            for cb in group_cb.values():
                cb.on_update(_rebuild_series)

    contact_handle: dict[str, object] = {"h": None}
    f6 = np.zeros(6)

    def draw_contacts(t: int) -> None:
        if contact_handle["h"] is not None:
            contact_handle["h"].remove()
            contact_handle["h"] = None
        if not show_cb.value:
            return
        data.qpos[:] = qpos[t]
        mujoco.mj_forward(model, data)  # solves contacts at this instant
        pts, colors = [], []
        for c in range(data.ncon):
            mujoco.mj_contactForce(model, data, c, f6)
            mag = float(np.linalg.norm(f6[:3]))
            if mag < cfg.min_force_n:
                continue
            pts.append(data.contact[c].pos.copy())
            colors.append(_color_by_force(mag, cfg.force_color_max_n))
        if pts:
            contact_handle["h"] = server.scene.add_point_cloud(
                "/contacts",
                points=np.asarray(pts, dtype=np.float32),
                colors=np.asarray(colors, dtype=np.uint8),
                point_size=float(scale_slider.value),
            )
        if torque_plot is not None:
            torque_plot.cursor = uplot.Cursor(x=True, y=False, idx=float(t))

    # Fires automatically once you interact with the Timeline slider / Play.
    viser_viewer.register_frame_callback(draw_contacts)

    if data_ref is not None:
        pelvis_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        pel_ref_handle = next(
            h for h, b in viser_viewer._STATE.ref_body_handles if b == pelvis_bid
        )

    dt = 1.0 / fps
    print(f"Loading {qpos.shape[0]} frames @ {fps:.0f}Hz ...")
    for t in range(qpos.shape[0]):
        data.qpos[:] = qpos[t]
        mujoco.mj_kinematics(model, data)
        if data_ref is not None:
            data_ref.qpos[:] = qpos_ref[t]
            mujoco.mj_kinematics(model, data_ref)
        viser_viewer.log_frame(
            data, sim_time=t * dt, viewer_body_entity_and_ids=body_entity_and_ids,
            data_ref=data_ref, playback_fps=fps,
        )
        # log_frame only records history + bumps the slider's max -- it never
        # actually pushes positions to the browser (that's _render_frame, wired
        # to the slider's on_update). Without this, every body sits wherever
        # add_frame() put it initially (world origin) until you touch the
        # slider by hand: everything looks piled up in the floor the whole time.
        viser_viewer._STATE.playback_slider.value = t
        viser_viewer._render_frame(viewer_body_entity_and_ids=body_entity_and_ids)
        draw_contacts(t)
        # Without this the whole 327-frame load finishes in well under a
        # second (each frame is cheap): a browser connecting anytime during
        # that window sees nothing animate and lands on the final frame only.
        time.sleep(dt)

    if data_ref is not None:
        print(f"[debug] reference pelvis world position at the LAST loaded frame: "
              f"{pel_ref_handle.position}  (z should be roughly 0.5-0.9, not ~0)")

    print(f"http://localhost:{server.get_port()}  —  drag Timeline / hit Play. "
          "Contact dots: blue = low force, red = high force.")
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main(tyro.cli(Config))
