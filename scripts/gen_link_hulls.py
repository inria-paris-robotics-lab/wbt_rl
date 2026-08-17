#!/usr/bin/env python3
"""gen_link_hulls.py — one convex-hull collision mesh per robot link.

Same method already used for the half-sphere hand hull (see
cfg/05_dynamics/scenes/femto14_box36_halfsphere.yaml's header): the plain
convex hull of the link's own visual-mesh vertex cloud, which mathematically
CONTAINS every source vertex -- unlike a CoACD decomposition, which minimizes
a volume/surface deviation cost and can leave real geometry sticking out
through gaps (rejected for the hand for exactly that reason).

Built for SPIDER's stage-05 physics solve, whose stock scene only pairs the
box against the two hand hulls (`contact.robot_geoms` in the scene manifest)
-- every other link has collision disabled (`contype=0` on the g1 default
class, re-enabled only via the explicit <pair> list spider_build_scene.py
writes) so the box can visibly tunnel through the torso/arms/legs during
replay. One hull per link (~22 links, not the ~1082 individual CoACD pieces
SPIDER already ships per-link) keeps the box<->robot pair count SPIDER's CEM
actually has to search at runtime in the same ballpark as the current 2
(hands only), instead of blowing it up ~500x.

Usage (needs trimesh, which lives in SPIDER's venv):
    modules/third_party/spider/.venv/bin/python scripts/gen_link_hulls.py \\
        --mesh-dir modules/third_party/spider/spider/assets/robots/unitree_g1/meshes \\
        --out-dir cfg/05_dynamics/robot_assets/g1_29dof_fullbody_hull/meshes \\
        --link pelvis:pelvis.STL \\
        --link torso_link:torso_link_rev_1_0.STL \\
        ...
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import trimesh


def hull_obj(src: Path) -> tuple[trimesh.Trimesh, dict]:
    mesh = trimesh.load(src, process=False)
    hull = mesh.convex_hull
    # Containment is a mathematical guarantee of convex_hull (every input vertex is
    # inside-or-on the hull by construction), checked anyway since a wrong axis
    # convention or degenerate mesh is a silent way to violate it.
    dist = trimesh.proximity.ProximityQuery(hull).signed_distance(mesh.vertices)
    stats = {
        "src_verts": len(mesh.vertices),
        "hull_verts": len(hull.vertices),
        "hull_faces": len(hull.faces),
        "max_outside_mm": float(max(0.0, -dist.min()) * 1000.0),
    }
    return hull, stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mesh-dir", required=True, type=Path, help="dir containing the source STL/OBJ files")
    ap.add_argument("--out-dir", required=True, type=Path, help="dir to write {link}_hull.obj into")
    ap.add_argument(
        "--link", action="append", required=True, dest="links", metavar="LINK:FILE",
        help="link_name:source_mesh_filename, repeatable",
    )
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"{'link':32s} {'src verts':>10s} {'hull verts':>11s} {'hull faces':>11s} {'max outside':>12s}")
    for entry in args.links:
        link, fname = entry.split(":", 1)
        src = args.mesh_dir / fname
        if not src.exists():
            raise FileNotFoundError(f"{link}: source mesh not found: {src}")
        hull, stats = hull_obj(src)
        out = args.out_dir / f"{link}_hull.obj"
        hull.export(out)
        assert stats["max_outside_mm"] < 1e-6, (
            f"{link}: {stats['max_outside_mm']:.4f} mm outside the hull -- containment guarantee violated"
        )
        print(
            f"{link:32s} {stats['src_verts']:10d} {stats['hull_verts']:11d} "
            f"{stats['hull_faces']:11d} {stats['max_outside_mm']:11.4f}mm  -> {out}"
        )


if __name__ == "__main__":
    main()
