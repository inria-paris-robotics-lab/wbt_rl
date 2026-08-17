#!/usr/bin/env python3
"""view_robot_collision.py — static viser inspection of a G1 MJCF's collision setup.

No playback, no sidecar -- just the robot at its default (standing) pose, with
three independent toggles in the "Meshes" / "Point cloud" panels:

  Visual      the real robot meshes (rubber hand, STL links, etc.)
  Collision   every class="collision" geom in its true shape (sphere/capsule/mesh),
              semi-transparent green -- this is what the physics actually collides on
  Point cloud just the CENTERS of the auto-generated collision spheres (the geoms
              named "*_collision_<N>" that scripts/gen_mesh_collision.py added),
              drawn as small dots -- lets you see the raw sample density separately
              from the sphere volumes

Needs mujoco + viser, which live in SPIDER's venv:

    modules/third_party/spider/.venv/bin/python scripts/view_robot_collision.py \\
        --xml modules/third_party/holosoma_custom/src/holosoma/holosoma/data/robots/g1/g1_29dof.xml
"""
from __future__ import annotations

import argparse
import re
import tempfile
import time
from pathlib import Path

import mujoco
import numpy as np

from spider.viewers import viser_viewer

_AUTO_GEOM = re.compile(r"_collision_\d+$")


def _inject_floor(xml_path: Path) -> Path:
    """Contact pairs reference geom2="floor", but the robot-only MJCF never
    defines the plane itself (it's normally supplied by the training scene).
    Same workaround as scripts/view_clip.py -- write a sibling copy with an
    absolute meshdir (so it can live in a tempdir) plus a floor geom."""
    xml = xml_path.read_text()
    meshdir = (xml_path.parent / "meshes").resolve()
    xml = re.sub(r'meshdir="[^"]*"', f'meshdir="{meshdir}"', xml)
    xml = xml.replace(
        "<worldbody>",
        '<worldbody><geom name="floor" type="plane" size="5 5 0.1" rgba="0.3 0.3 0.35 1"/>',
        1,
    )
    tmp = Path(tempfile.mkdtemp()) / xml_path.name
    tmp.write_text(xml)
    return tmp


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--xml", required=True, help="G1 MJCF, e.g. g1_29dof.xml or g1_29dof_halfspherehand.xml")
    args = ap.parse_args()

    src = Path(args.xml).resolve()
    patched = _inject_floor(src)

    viser_viewer.init_viser(app_name="wbt-rl-robot-collision")
    spec, model, body_entity_and_ids = viser_viewer.build_and_log_scene(patched)

    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)
    for handle, body_id in body_entity_and_ids:
        handle.position = data.xpos[body_id]
        handle.wxyz = data.xquat[body_id]

    point_geom_ids = [
        gid
        for gid in range(model.ngeom)
        if _AUTO_GEOM.search(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or "")
    ]
    print(f"{len(point_geom_ids)} auto-generated point-cloud geoms found (gen_mesh_collision.py)")

    server = viser_viewer._get_server()
    with server.gui.add_folder("Point cloud"):
        pc_cb = server.gui.add_checkbox("Show sampled points", initial_value=False)
        pc_size = server.gui.add_slider("Point size", min=0.001, max=0.01, step=0.0005, initial_value=0.003)

    pc_handle = {"h": None}

    def draw_point_cloud() -> None:
        if pc_handle["h"] is not None:
            pc_handle["h"].remove()
            pc_handle["h"] = None
        if not pc_cb.value or not point_geom_ids:
            return
        pts = data.geom_xpos[point_geom_ids]
        pc_handle["h"] = server.scene.add_point_cloud(
            "/collision_point_cloud",
            points=np.asarray(pts, dtype=np.float32),
            colors=np.tile(np.array([255, 210, 0], dtype=np.uint8), (len(pts), 1)),
            point_size=float(pc_size.value),
        )

    pc_cb.on_update(lambda _: draw_point_cloud())
    pc_size.on_update(lambda _: draw_point_cloud())

    print(f"http://localhost:{server.get_port()}  —  {src.name}, standing pose (static, no playback).")
    print('Toggle "Visual" / "Collision" under Meshes, "Show sampled points" under Point cloud.')
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()
