"""
Build a SPIDER (05_dynamics) working directory from a retargeted motion.

Runs inside SPIDER's own venv (needs mujoco + scipy); invoked through
`venv_run()` by `_to_dynamics_input/spider.py`. Never imported by the wbt_rl env.

Produces, under `--out-dir`:
    scene.xml / scene_act.xml     robot + N objects, with explicit contact pairs
    task_info.json                ref_dt + contact site ids (read by SPIDER)
    <data_id>/trajectory_kinematic_act.npz   qpos/qvel/ctrl/contact/contact_pos

The robot half of qpos is the retargeter's own `joint_pos (T, 7+nj)` — root
xyz + wxyz quat + joint angles — which is already MuJoCo qpos ordering, so the
mapping is a straight copy once joint names are verified to match.

Object layout (matches the `humanoid_nobj` embodiment added by
installers/patches/spider-humanoid-nobj.patch): every DYNAMIC object appends 6
DOF to the qpos tail as 3 slide + 3 hinge joints, the hinges composing as
INTRINSIC XYZ Euler angles (verified against MuJoCo's own composition, and the
convention SPIDER's `_euler_to_quat_wxyz` already assumes). Static objects are
welded bodies with no DOF and no entry in qpos.
"""

from __future__ import annotations

import argparse
import json
import shutil
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np
import yaml
from scipy.spatial.transform import Rotation as Rsp

# MuJoCo pair friction is 5-dimensional: [slide1, slide2, spin, roll1, roll2].
_DEFAULT_FRICTION = (0.9, 0.9, 0.005, 0.0001, 0.0001)


@dataclass
class ObjectSpec:
    """One object in the scene, resolved from a URDF or from explicit geometry."""

    name: str
    kind: str  # "dynamic" | "static"
    pos_key: str
    quat_key: str
    mesh: Path | None = None
    box_half: tuple[float, float, float] | None = None
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    mass: float | None = None
    friction: tuple[float, ...] = _DEFAULT_FRICTION
    rgba: str = "0.7 0.8 0.9 1"

    @property
    def is_dynamic(self) -> bool:
        return self.kind == "dynamic"


def _urdf_first(root: ET.Element, *paths: str):
    for p in paths:
        el = root.find(p)
        if el is not None:
            return el
    return None


def parse_urdf(urdf_path: Path) -> dict:
    """Extract mass / geometry / friction from a (PyBullet-flavoured) URDF.

    Only the first link is read: these object URDFs are single-body descriptions
    (see e.g. holosoma_custom's objects_box36.urdf). Mesh filenames resolve
    relative to the URDF's own directory.
    """
    root = ET.parse(urdf_path).getroot()
    link = root.find("link")
    if link is None:
        raise ValueError(f"{urdf_path}: no <link> found")

    out: dict = {}

    mass_el = link.find("inertial/mass")
    if mass_el is not None and mass_el.get("value"):
        out["mass"] = float(mass_el.get("value"))

    # Prefer the collision geometry — that is what the physics actually uses.
    geom_parent = _urdf_first(link, "collision/geometry", "visual/geometry")
    if geom_parent is None:
        raise ValueError(f"{urdf_path}: no <collision>/<visual> geometry")

    mesh_el = geom_parent.find("mesh")
    box_el = geom_parent.find("box")
    if mesh_el is not None:
        out["mesh"] = (urdf_path.parent / mesh_el.get("filename")).resolve()
        scale = mesh_el.get("scale")
        if scale:
            out["scale"] = tuple(float(v) for v in scale.split())
    elif box_el is not None:
        size = tuple(float(v) for v in box_el.get("size").split())
        out["box_half"] = tuple(s / 2.0 for s in size)
    else:
        raise ValueError(f"{urdf_path}: geometry is neither <mesh> nor <box>")

    # PyBullet puts friction under <contact>; fall back to <dynamics friction=...>.
    lat = _urdf_first(root, "link/contact/lateral_friction", "contact/lateral_friction")
    roll = _urdf_first(root, "link/contact/rolling_friction", "contact/rolling_friction")
    if lat is None:
        dyn = root.find("dynamics")
        lat_v = float(dyn.get("friction")) if dyn is not None and dyn.get("friction") else None
    else:
        lat_v = float(lat.get("value"))
    if lat_v is not None:
        roll_v = float(roll.get("value")) if roll is not None else _DEFAULT_FRICTION[3]
        out["friction"] = (lat_v, lat_v, _DEFAULT_FRICTION[2], roll_v, roll_v)

    return out


def load_manifest(path: Path, repo_root: Path) -> tuple[list[ObjectSpec], dict]:
    """Load the scene manifest (see cfg/05_dynamics/scenes/README.md)."""
    raw = yaml.safe_load(Path(path).read_text())
    if not isinstance(raw, dict) or "objects" not in raw:
        raise ValueError(f"{path}: expected a mapping with an 'objects' key")

    def _resolve(p: str) -> Path:
        q = Path(p).expanduser()
        return q if q.is_absolute() else (repo_root / q)

    specs: list[ObjectSpec] = []
    for i, entry in enumerate(raw["objects"]):
        name = entry.get("name") or f"obj{i}"
        kind = entry.get("kind", "dynamic")
        if kind not in ("dynamic", "static"):
            raise ValueError(f"object {name!r}: kind must be 'dynamic' or 'static', got {kind!r}")

        spec = ObjectSpec(
            name=name,
            kind=kind,
            pos_key=entry.get("pos_key", "object_pos_w"),
            quat_key=entry.get("quat_key", "object_quat_w"),
        )
        if entry.get("urdf"):
            parsed = parse_urdf(_resolve(entry["urdf"]))
            spec.mesh = parsed.get("mesh")
            spec.box_half = parsed.get("box_half")
            spec.scale = parsed.get("scale", spec.scale)
            spec.mass = parsed.get("mass")
            spec.friction = parsed.get("friction", spec.friction)
        if entry.get("mesh"):  # explicit mesh overrides / stands in for a URDF
            spec.mesh = _resolve(entry["mesh"])
        if entry.get("half_extents"):
            spec.box_half = tuple(float(v) for v in entry["half_extents"])
        if entry.get("mass") is not None:
            spec.mass = float(entry["mass"])
        if entry.get("rgba"):
            spec.rgba = entry["rgba"]
        if spec.mesh is None and spec.box_half is None:
            raise ValueError(
                f"object {name!r}: needs a urdf, a mesh, or half_extents to have any geometry"
            )
        if spec.mesh is not None and not spec.mesh.exists():
            raise FileNotFoundError(f"object {name!r}: mesh not found: {spec.mesh}")
        specs.append(spec)

    contact_cfg = raw.get("contact", {}) or {}
    return specs, contact_cfg


def build_scene(
    src_robot_xml: Path,
    dst_xml: Path,
    specs: list[ObjectSpec],
    static_poses: dict[str, tuple[np.ndarray, np.ndarray]],
    contact_cfg: dict,
) -> None:
    """Write an MJCF with the robot plus every object, and their contact pairs."""
    tree = ET.parse(src_robot_xml)
    root = tree.getroot()

    # The stock MJCF carries a meshdir relative to its position inside SPIDER's
    # asset tree. We emit the scene elsewhere, so rebase it to an absolute path
    # rather than replicating that tree (and its staleness) next to every run.
    compiler = root.find("compiler")
    if compiler is not None and compiler.get("meshdir"):
        compiler.set(
            "meshdir", str((src_robot_xml.parent / compiler.get("meshdir")).resolve())
        )

    worldbody = root.find("worldbody")
    actuator = root.find("actuator")
    asset = root.find("asset")
    if asset is None:
        asset = ET.SubElement(root, "asset")
    contact = root.find("contact")
    if contact is None:
        contact = ET.SubElement(root, "contact")

    object_geom_names: list[str] = []
    dynamic_geom_names: list[str] = []

    for i, spec in enumerate(specs):
        # A <geom> takes 3 friction terms (slide, spin, roll); a <pair> takes 5
        # (slide1, slide2, spin, roll1, roll2). spec.friction is stored in pair
        # form, so drop the duplicated slide/roll entries here.
        geom_friction = (spec.friction[0], spec.friction[2], spec.friction[3])
        geom_attrs: dict[str, str] = {
            "name": f"{spec.name}_geom",
            "rgba": spec.rgba,
            "contype": "1",
            "conaffinity": "1",
            "friction": " ".join(str(f) for f in geom_friction),
        }
        if spec.mesh is not None:
            mesh_name = f"{spec.name}_mesh"
            mesh_attrs = {"name": mesh_name, "file": str(spec.mesh)}
            if spec.scale != (1.0, 1.0, 1.0):
                mesh_attrs["scale"] = " ".join(str(s) for s in spec.scale)
            ET.SubElement(asset, "mesh", mesh_attrs)
            geom_attrs.update({"type": "mesh", "mesh": mesh_name})
        else:
            geom_attrs.update(
                {"type": "box", "size": " ".join(f"{h:.6f}" for h in spec.box_half)}
            )
        if spec.mass is not None:
            geom_attrs["mass"] = f"{spec.mass}"

        if spec.is_dynamic:
            body = ET.SubElement(worldbody, "body", {"name": spec.name, "pos": "0 0 0"})
            # 3 slide + 3 hinge, in this exact order: the patched humanoid_nobj
            # embodiment reads each object as a [pos(3), euler(3)] block, and the
            # hinges compose as intrinsic XYZ.
            for suffix, jtype, axis in (
                ("pos_x", "slide", "1 0 0"),
                ("pos_y", "slide", "0 1 0"),
                ("pos_z", "slide", "0 0 1"),
                ("rot_x", "hinge", "1 0 0"),
                ("rot_y", "hinge", "0 1 0"),
                ("rot_z", "hinge", "0 0 1"),
            ):
                jname = f"obj{i}_{suffix}"
                ET.SubElement(
                    body,
                    "joint",
                    {
                        "name": jname,
                        "type": jtype,
                        "axis": axis,
                        "damping": "0.1",
                        "armature": "0.001",
                    },
                )
                # Guidance actuators MUST rest at zero gain: run_mjwp.py asserts
                # it at startup and drives the anneal schedule itself.
                ET.SubElement(
                    actuator,
                    "position",
                    {"name": jname, "joint": jname, "kp": "0", "kv": "0"},
                )
            ET.SubElement(body, "geom", geom_attrs)
            dynamic_geom_names.append(geom_attrs["name"])
        else:
            pos, quat = static_poses[spec.name]
            body = ET.SubElement(
                worldbody,
                "body",
                {
                    "name": spec.name,
                    "pos": " ".join(f"{v:.6f}" for v in pos),
                    "quat": " ".join(f"{v:.6f}" for v in quat),
                },
            )
            ET.SubElement(body, "geom", geom_attrs)
        object_geom_names.append(geom_attrs["name"])

    # --- contact pairs -------------------------------------------------------
    # This robot MJCF disables contact globally (`contype=0` on the g1 default
    # class) and re-enables it through an explicit <pair> list. Objects added
    # after the fact are therefore INVISIBLE to the robot unless paired here —
    # omitting this silently yields a robot whose hands pass through the object.
    robot_geoms = list(contact_cfg.get("robot_geoms", ["lh", "rh"]))
    pair_friction = " ".join(str(f) for f in contact_cfg.get("friction", _DEFAULT_FRICTION))
    condim = str(contact_cfg.get("condim", 4))
    solref = contact_cfg.get("solref", "0.008 1")

    def _pair(g1: str, g2: str) -> None:
        ET.SubElement(
            contact,
            "pair",
            {
                "name": f"{g1}__{g2}",
                "geom1": g1,
                "geom2": g2,
                "solref": solref,
                "friction": pair_friction,
                "condim": condim,
            },
        )

    for rg in robot_geoms:
        for og in object_geom_names:
            _pair(rg, og)
    if contact_cfg.get("object_object", True):
        for a in range(len(object_geom_names)):
            for b in range(a + 1, len(object_geom_names)):
                _pair(object_geom_names[a], object_geom_names[b])
    if contact_cfg.get("object_floor", True):
        for og in dynamic_geom_names:
            _pair(og, "floor")

    try:
        ET.indent(root, space="  ")
    except AttributeError:
        pass
    tree.write(dst_xml)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--motion", required=True, help="retargeted motion npz (holosoma form B)")
    ap.add_argument("--scene-manifest", required=True)
    ap.add_argument("--robot-assets", required=True, help="SPIDER robot asset dir (unitree_g1)")
    ap.add_argument("--out-dir", required=True, help="SPIDER task dir (scene.xml lives here)")
    ap.add_argument("--data-id", default="0")
    ap.add_argument("--repo-root", required=True)
    ap.add_argument("--fps", type=int, default=0, help="override the npz's own fps")
    args = ap.parse_args()

    repo_root = Path(args.repo_root)
    out_dir = Path(args.out_dir)
    data_dir = out_dir / str(args.data_id)
    data_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(args.motion, allow_pickle=True)
    fps = int(args.fps) if args.fps else int(np.asarray(data["fps"]).ravel()[0])
    dt = 1.0 / fps
    qpos_robot = np.asarray(data["joint_pos"], dtype=np.float64)  # (T, 7+nj)
    n_frames = qpos_robot.shape[0]

    specs, contact_cfg = load_manifest(Path(args.scene_manifest), repo_root)
    dynamic = [s for s in specs if s.is_dynamic]

    # Static poses are read once (they do not move); dynamic ones per frame.
    static_poses: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for spec in specs:
        if spec.is_dynamic:
            continue
        pos = np.asarray(data[spec.pos_key], dtype=np.float64).reshape(-1)[:3]
        quat = np.asarray(data[spec.quat_key], dtype=np.float64).reshape(-1)[:4]
        static_poses[spec.name] = (pos, quat)

    scene_path = out_dir / "scene.xml"
    robot_xml = Path(args.robot_assets).resolve() / "scene.xml"
    build_scene(robot_xml, scene_path, specs, static_poses, contact_cfg)
    # contact_guidance reads scene_act.xml; keep both names pointing at one scene.
    shutil.copy(scene_path, out_dir / "scene_act.xml")

    model = mujoco.MjModel.from_xml_path(str(scene_path))
    mj_data = mujoco.MjData(model)
    n_obj_dof = 6 * len(dynamic)
    expected_nq = qpos_robot.shape[1] + n_obj_dof
    if model.nq != expected_nq:
        raise ValueError(
            f"nq mismatch: model has {model.nq}, motion+objects imply {expected_nq} "
            f"({qpos_robot.shape[1]} robot + {n_obj_dof} object DOF)"
        )

    # Joint-order check: a silent mismatch here corrupts every downstream number.
    motion_joints = [str(n) for n in data["joint_names"]]
    model_joints = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        for j in range(model.njnt)
        if model.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE
    ]
    robot_joints = model_joints[: len(motion_joints)]
    if robot_joints != motion_joints:
        raise ValueError(
            "robot joint order differs between the motion file and the MJCF.\n"
            f"  motion: {motion_joints}\n  mjcf:   {robot_joints}"
        )

    # --- assemble qpos ------------------------------------------------------
    qpos = np.zeros((n_frames, model.nq), dtype=np.float64)
    qpos[:, : qpos_robot.shape[1]] = qpos_robot
    for i, spec in enumerate(dynamic):
        pos = np.asarray(data[spec.pos_key], dtype=np.float64).reshape(n_frames, 3)
        quat_wxyz = np.asarray(data[spec.quat_key], dtype=np.float64).reshape(n_frames, 4)
        euler = Rsp.from_quat(quat_wxyz[:, [1, 2, 3, 0]]).as_euler("XYZ")
        s = qpos_robot.shape[1] + 6 * i
        qpos[:, s : s + 3] = pos
        qpos[:, s + 3 : s + 6] = euler

    qvel = np.zeros((n_frames, model.nv), dtype=np.float64)
    for t in range(1, n_frames):
        mujoco.mj_differentiatePos(model, mj_data.qvel, dt, qpos[t - 1], qpos[t])
        qvel[t] = mj_data.qvel

    # ctrl mirrors the actuator list: robot motors, then object guidance DOFs.
    n_robot_act = model.nu - n_obj_dof
    ctrl = np.zeros((n_frames, model.nu), dtype=np.float64)
    ctrl[:, :n_robot_act] = qpos_robot[:, 7 : 7 + n_robot_act]
    if n_obj_dof:
        ctrl[:, n_robot_act:] = qpos[:, -n_obj_dof:]

    # --- contact channels ---------------------------------------------------
    contact_site_ids, contact_site_names = [], []
    for sid in range(model.nsite):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, sid)
        if name and "contact" in name:
            contact_site_ids.append(sid)
            contact_site_names.append(name)
    if not contact_site_ids:
        raise ValueError(f"{scene_path}: no site whose name contains 'contact'")

    # Ground-truth hand contact when the retargeter provides it; feet fall back
    # to a height threshold. SPIDER only uses this to gate its contact reward.
    gt_hand = np.zeros(n_frames, dtype=bool)
    for key in ("object_ref_contact", "support_ref_contact"):
        if key in data:
            gt_hand |= np.asarray(data[key]).astype(bool).reshape(-1)[:n_frames]

    contact = np.zeros((n_frames, len(contact_site_ids)), dtype=np.float64)
    contact_pos = np.zeros((n_frames, len(contact_site_ids), 3), dtype=np.float64)
    for t in range(n_frames):
        mj_data.qpos[:] = qpos[t]
        mujoco.mj_kinematics(model, mj_data)
        pos = mj_data.site_xpos[contact_site_ids, :]
        contact_pos[t] = pos
        for k, name in enumerate(contact_site_names):
            contact[t, k] = float(pos[k, 2] < 0.02) if "foot" in name else float(gt_hand[t])

    out_npz = data_dir / "trajectory_kinematic_act.npz"
    np.savez(
        out_npz,
        qpos=qpos,
        qvel=qvel,
        ctrl=ctrl,
        contact=contact,
        contact_pos=contact_pos,
    )
    # spider/config.py picks the filename by config.contact_guidance:
    # trajectory_kinematic_act.npz when True, trajectory_kinematic.npz (no
    # suffix) when False. This build step runs before that flag is known (it's
    # resolved later, inside SPIDER's own Hydra config), so write both -- same
    # copy/duplicate pattern already used for scene.xml / scene_act.xml above.
    shutil.copy(out_npz, data_dir / "trajectory_kinematic.npz")
    (out_dir / "task_info.json").write_text(
        json.dumps({"ref_dt": dt, "contact_site_ids": contact_site_ids}, indent=2)
    )

    meta = {
        "num_objects": len(dynamic),
        "dynamic_names": [s.name for s in dynamic],
        "static_names": [s.name for s in specs if not s.is_dynamic],
        "nq": int(model.nq),
        "nv": int(model.nv),
        "nu": int(model.nu),
        "n_robot_actuators": int(n_robot_act),
        "fps": fps,
        "n_frames": int(n_frames),
        "contact_sites": contact_site_names,
    }
    (out_dir / "wbt_scene_meta.json").write_text(json.dumps(meta, indent=2))

    print(f"scene      -> {scene_path}")
    print(f"trajectory -> {out_npz}")
    print(
        f"nq={model.nq} nv={model.nv} nu={model.nu} | "
        f"{len(dynamic)} dynamic + {len(specs) - len(dynamic)} static object(s) | "
        f"T={n_frames} @ {fps}Hz"
    )


if __name__ == "__main__":
    main()
