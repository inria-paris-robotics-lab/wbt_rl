#!/usr/bin/env python3
"""gen_realgains_assets.py — a SPIDER robot asset carrying holosoma's TRAINING gains.

SPIDER ships its G1 with `kp=500 dampratio=1 armature=1.0` on every joint. The
policy we are producing data for is trained against holosoma_custom's g1_29dof
gains, which are 5-35x softer (kp 14.25-99.1). A stage-05 run on the stock rig
therefore answers a question nobody asked: what the motion costs on a robot
whose joints are far stiffer than the real one. Gravity sag that would open a
grip at training gains is invisible at kp=500.

This builds a copy of the stock asset with, in one place, the three things that
differ from the training setup:

  * actuator kp/kv, per joint family, read from holosoma_custom's own
    RobotControlConfig -- not copied here, so there is one source of truth
  * joint armature (SPIDER 1.0 -> holosoma 0.001)
  * actuator force range, from the URDF effort limits, mirroring holosoma's
    `clip_torques=True` (omit with --no-torque-cap)

Meshes are NOT copied: they weigh 81 MB, so `meshdir` is rewritten as a path
relative to the output directory instead. The result is portable across
machines, which the hand-committed cfg/05_dynamics/robot_assets/* are not --
they carry absolute /home/<someone>/ mesh paths from whoever generated them.

Usage:
    python scripts/gen_realgains_assets.py
    python scripts/gen_realgains_assets.py --out cfg/05_dynamics/robot_assets/foo

Then point a stage-05 run at it:
    python scripts/enrich_dynamics.py --module spider_realgains \\
        --robot-assets cfg/05_dynamics/robot_assets/g1_29dof_realgains ...
"""
from __future__ import annotations

import argparse
import ast
import os
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

_REPO_ROOT = Path(__file__).parents[1]

_DEFAULT_SOURCE = "modules/third_party/spider/spider/assets/robots/unitree_g1"
_DEFAULT_GAINS = (
    "modules/third_party/holosoma_custom/src/holosoma/holosoma/"
    "config_values/robot.py"
)
_DEFAULT_OUT = "cfg/05_dynamics/robot_assets/g1_29dof_realgains"


def read_gains(gains_py: Path, symbol: str) -> tuple[dict[str, float], dict[str, float]]:
    """Pull `stiffness`/`damping` out of a RobotConfig assignment, by AST.

    Parsed rather than imported: importing holosoma's config package drags in
    its whole dependency tree (and an env we may not be running under), while
    these two dicts are plain literals. Parsing keeps the numbers sourced from
    the training config instead of duplicated into this file.
    """
    tree = ast.parse(gains_py.read_text(), str(gains_py))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(getattr(t, "id", None) == symbol for t in node.targets):
            continue
        for kw in getattr(node.value, "keywords", []):
            if kw.arg != "control":
                continue
            found = {}
            for ckw in kw.value.keywords:
                if ckw.arg in ("stiffness", "damping"):
                    found[ckw.arg] = ast.literal_eval(ckw.value)
            if "stiffness" in found and "damping" in found:
                return found["stiffness"], found["damping"]
            raise ValueError(f"{symbol}.control lacks stiffness/damping literals")
    raise ValueError(f"no `{symbol} = RobotConfig(...)` assignment in {gains_py}")


def read_efforts(urdf: Path) -> dict[str, float]:
    """joint name -> effort limit (Nm), from the URDF the asset ships with."""
    root = ET.parse(urdf).getroot()
    out = {}
    for joint in root.iter("joint"):
        limit = joint.find("limit")
        name = joint.get("name")
        if limit is not None and limit.get("effort") and name:
            out[name] = float(limit.get("effort"))
    return out


def _family(joint_name: str, families: dict[str, float]) -> str | None:
    """Match a joint to its gain family, longest key first.

    `ankle_pitch` must win over any shorter key that also appears in the name,
    so length ordering is load-bearing here, not cosmetic.
    """
    for key in sorted(families, key=len, reverse=True):
        if key in joint_name:
            return key
    return None


def build(source: Path, gains_py: Path, out_dir: Path, symbol: str,
          torque_cap: bool) -> None:
    stiffness, damping = read_gains(gains_py, symbol)
    print(f"gains      {symbol}: {len(stiffness)} families, "
          f"kp {min(stiffness.values()):.2f}-{max(stiffness.values()):.2f}, "
          f"kv {min(damping.values()):.3f}-{max(damping.values()):.3f}")
    if set(stiffness) != set(damping):
        raise ValueError("stiffness and damping cover different joint families")

    scene = ET.parse(source / "scene.xml")
    root = scene.getroot()
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- meshes stay where they are; point at them relatively ----------------
    compiler = root.find("compiler")
    if compiler is None:
        raise ValueError(f"{source}/scene.xml has no <compiler>")
    old_meshdir = compiler.get("meshdir", "")
    # The stock meshdir is relative to the scene.xml's own directory.
    abs_meshes = (source / old_meshdir).resolve()
    if not abs_meshes.is_dir():
        raise FileNotFoundError(f"mesh dir not found: {abs_meshes}")
    compiler.set("meshdir", os.path.relpath(abs_meshes, out_dir.resolve()))
    print(f"meshdir    {compiler.get('meshdir')}")

    # --- gains, written on the <default> classes -----------------------------
    # The stock class tree already names one class per joint family, and those
    # names match holosoma's dict keys exactly, so the gains land in 16 places
    # rather than on all 29 actuators.
    touched = set()
    for dflt in root.iter("default"):
        cls = dflt.get("class")
        if cls not in stiffness:
            continue
        pos = dflt.find("position")
        if pos is None:
            pos = ET.SubElement(dflt, "position")
            pos.set("inheritrange", "1")
        pos.set("kp", f"{stiffness[cls]:.9g}")
        pos.set("kv", f"{damping[cls]:.9g}")
        # kv and dampratio are mutually exclusive: dampratio would re-derive a
        # damping from kp and mass and silently override the value we just set.
        pos.attrib.pop("dampratio", None)
        touched.add(cls)
    missing = set(stiffness) - touched
    if missing:
        raise ValueError(f"no <default class=...> for gain families: {sorted(missing)}")
    print(f"gains      written on {len(touched)} default classes")

    # Parent classes keep kp=500 and would leak it to any actuator that sits
    # directly in them rather than in a leaf. Strip those so a miss is a load
    # error, not a silently stiff joint.
    for dflt in root.iter("default"):
        cls = dflt.get("class")
        if cls in stiffness or cls is None:
            continue
        pos = dflt.find("position")
        if pos is not None and pos.get("kp") is not None and cls != "g1":
            pos.attrib.pop("kp", None)
            pos.attrib.pop("dampratio", None)
            print(f"           cleared inherited kp on parent class {cls!r}")

    # --- armature ------------------------------------------------------------
    armature = _read_asset_armature(gains_py, symbol)
    n_arm = 0
    for dflt in root.iter("default"):
        joint = dflt.find("joint")
        if joint is not None and joint.get("armature") is not None:
            joint.set("armature", f"{armature:g}")
            n_arm += 1
    print(f"armature   {armature:g} on {n_arm} default class(es)")

    # --- torque cap ----------------------------------------------------------
    if torque_cap:
        efforts = read_efforts(source / "g1_custom_collision_29dof.urdf")
        actuators = root.find("actuator")
        n_cap = 0
        for act in actuators.iter("position"):
            joint = act.get("joint")
            if joint and joint in efforts:
                e = efforts[joint]
                act.set("forcerange", f"{-e:g} {e:g}")
                n_cap += 1
        print(f"forcerange set on {n_cap} actuators "
              f"({min(efforts.values()):g}-{max(efforts.values()):g} Nm)")

    ET.indent(scene, space="    ")
    dest = out_dir / "scene.xml"
    scene.write(dest, encoding="unicode", xml_declaration=False)
    dest.write_text(dest.read_text() + "\n")
    print(f"wrote      {dest}")


def _read_asset_armature(gains_py: Path, symbol: str) -> float:
    """RobotAssetConfig(armature=...) for the same symbol."""
    tree = ast.parse(gains_py.read_text(), str(gains_py))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(getattr(t, "id", None) == symbol for t in node.targets):
            continue
        for kw in getattr(node.value, "keywords", []):
            if kw.arg != "asset":
                continue
            for akw in kw.value.keywords:
                if akw.arg == "armature":
                    return float(ast.literal_eval(akw.value))
    raise ValueError(f"no asset.armature for {symbol} in {gains_py}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", type=Path, default=_REPO_ROOT / _DEFAULT_SOURCE,
                   help="stock SPIDER robot asset dir (holds scene.xml + meshes/)")
    p.add_argument("--gains", type=Path, default=_REPO_ROOT / _DEFAULT_GAINS,
                   help="holosoma_custom config_values/robot.py")
    p.add_argument("--symbol", default="g1_29dof",
                   help="RobotConfig assignment to read the gains from")
    p.add_argument("--out", type=Path, default=_REPO_ROOT / _DEFAULT_OUT)
    p.add_argument("--no-torque-cap", action="store_true",
                   help="skip the URDF effort limits (holosoma trains with "
                        "clip_torques=True, so the cap is on by default)")
    args = p.parse_args()

    for path in (args.source / "scene.xml", args.gains):
        if not path.exists():
            raise SystemExit(f"not found: {path}")

    build(args.source, args.gains, args.out, args.symbol, not args.no_torque_cap)


if __name__ == "__main__":
    main()
