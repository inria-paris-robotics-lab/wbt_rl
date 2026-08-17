#!/usr/bin/env python3
"""patch_fullbody_hull.py — graft one convex-hull collision geom per link onto a
SPIDER robot-assets scene.xml, so a scene manifest can pair the box against them.

Companion to gen_link_hulls.py: that script produces the {link}_hull.obj files,
this one wires them in as a <mesh> asset plus a <geom class="collision"> child of
each named <body>, at the body's own origin (the source visual meshes carry no
pos/quat offset, so neither does the hull). Bodies not present in the source
scene are skipped with a warning rather than failing the whole patch.
"""
from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path


def patch(src: Path, dst: Path, hull_dir: Path, links: list[str]) -> None:
    tree = ET.parse(src)
    root = tree.getroot()
    asset = root.find("asset")
    assert asset is not None, f"{src}: no <asset> block"

    missing = []
    for link in links:
        hull_obj = hull_dir / f"{link}_hull.obj"
        assert hull_obj.exists(), f"missing hull mesh for {link!r}: {hull_obj}"

        body = root.find(f".//body[@name='{link}']")
        if body is None:
            missing.append(link)
            continue

        mesh_name = f"{link}_hull"
        ET.SubElement(asset, "mesh", {"name": mesh_name, "file": str(hull_obj.resolve())})
        # Inserted as the FIRST child of the body (index 0), ahead of any nested
        # <body> (child links) so it sits directly on this link's own geometry
        # regardless of how many children the body has.
        geom = ET.Element(
            "geom", {"name": mesh_name, "class": "collision", "type": "mesh", "mesh": mesh_name}
        )
        body.insert(0, geom)

    if missing:
        print(f"WARNING: {len(missing)} link(s) not found in {src}, skipped: {missing}")

    try:
        ET.indent(root, space="  ")
    except AttributeError:
        pass
    tree.write(dst)
    print(f"patched {len(links) - len(missing)}/{len(links)} link hulls -> {dst}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, type=Path, help="source robot-assets scene.xml")
    ap.add_argument("--dst", required=True, type=Path, help="destination scene.xml")
    ap.add_argument("--hull-dir", required=True, type=Path, help="dir containing {link}_hull.obj files")
    ap.add_argument("--link", action="append", required=True, dest="links", help="body/link name, repeatable")
    args = ap.parse_args()
    args.dst.parent.mkdir(parents=True, exist_ok=True)
    patch(args.src, args.dst, args.hull_dir, args.links)


if __name__ == "__main__":
    main()
