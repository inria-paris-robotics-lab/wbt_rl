#!/usr/bin/env python3
"""
retarget.py — run a full retargeting job for one (dataset, robot, retargeter) combination.

Usage:
    python scripts/retarget.py \
        --dataset LAFAN \
        --robot G1_29dof \
        --retargeter GMR \
        [--sequences seq1 seq2 ...] \
        [--run-id run_20240301_120000]

Output: data/01_retargeted_motions/{dataset}_{robot}/{retargeter}/run_{timestamp}/
"""
import argparse
import re
import sys
import yaml
from datetime import datetime
from pathlib import Path

_DOF_SUFFIX_RE = re.compile(r"_\d+dof$", re.IGNORECASE)


def _validate_robot(robot: str) -> str:
    """Require explicit dof suffix; normalize to uppercase base + lowercase suffix (G1_29dof)."""
    m = _DOF_SUFFIX_RE.search(robot)
    if not m:
        raise ValueError(
            f"--robot must include an explicit DOF suffix (e.g. G1_29dof or G1_27dof), got {robot!r}"
        )
    return robot[:m.start()].upper() + m.group(0).lower()

_REPO_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

import motion_convertor
from motion_convertor._config import dataset_path, output_path, repo_root, body_model_smplx_path
from motion_convertor._subprocess import load_module_cfg, conda_run
from motion_convertor.formats import validate_format


# ---------------------------------------------------------------------------
# Dataset helpers — driven by cfg/00_datasets/<dataset>.yaml
# ---------------------------------------------------------------------------

def _load_dataset_cfg(dataset: str) -> dict:
    cfg_path = repo_root() / "cfg" / "00_datasets" / f"{dataset.lower()}.yaml"
    return yaml.safe_load(cfg_path.read_text())


def discover_sequences(dataset: str, sequences: list[str] | None) -> list[tuple]:
    ds = dataset.upper()
    ds_cfg = _load_dataset_cfg(ds)
    raw_dir = dataset_path(ds)
    discover = ds_cfg["discover"]

    if discover == "single_files":
        pattern = ds_cfg["file_pattern"]
        if "**" in pattern:
            files = sorted(raw_dir.rglob(pattern))
        else:
            files = sorted(raw_dir.glob(pattern))
        seqs = [(f.stem, f) for f in files]

    elif discover == "pickle_keys":
        p_file = next(raw_dir.glob(ds_cfg["file_pattern"]))
        seqs = [("omomo_train", p_file)]   # expanded later per-sequence

    else:
        raise ValueError(f"Unknown discover strategy {discover!r} for dataset {ds!r}")

    if sequences:
        seqs = [(name, path) for name, path in seqs if name in sequences]
    return seqs


# ---------------------------------------------------------------------------
# Per-sequence retargeting
# ---------------------------------------------------------------------------

def retarget_sequence(
    seq_name: str,
    raw_path: Path,
    run_dir: Path,
    dataset: str,
    robot: str,
    retargeter: str,
    cfg: dict,
    seq_data: dict | None = None,
    task_type: str = "robot_only",
    visualize: bool = False,
) -> None:
    dataset_up = dataset.upper()
    retargeter_lo = retargeter.lower()

    # Resolve formats from module YAML
    input_format = cfg["native_input_format"][dataset_up]
    output_format = cfg["native_output_format"]
    validate_format(input_format)
    validate_format(output_format)

    # Derive file extensions from format names
    input_ext = _ext_from_format(input_format)
    output_ext = _ext_from_format(output_format)

    if cfg.get("input_path_style") == "dir":
        input_dir = run_dir / "input"
        input_dir.mkdir(parents=True, exist_ok=True)
        input_raw_path = input_dir / f"{seq_name}{input_ext}"
    else:
        input_raw_path = run_dir / f"{seq_name}_input_raw{input_ext}"

    input_unified_path = run_dir / f"{seq_name}_input_unified.npz"
    output_raw_path = run_dir / f"{seq_name}_output_raw{output_ext}"
    output_unified_path = run_dir / f"{seq_name}_output_unified.npz"

    # Step a: retargeter native input
    print(f"  [1/4] to_retargeter_input → {input_raw_path.name}")
    kw = {}
    if dataset_up == "OMOMO":
        kw["seq_data"] = seq_data
        kw["task_type"] = task_type
    motion_convertor.to_retargeter_input(dataset_up, retargeter_lo, raw_path, input_raw_path, **kw)

    # Step b: unified input (skipped for OMOMO_NEW)
    if dataset_up != "OMOMO_NEW":
        print(f"  [2/4] to_unified_input    → {input_unified_path.name}")
        kw_uni = {}
        if dataset_up == "OMOMO":
            kw_uni["seq_data"] = seq_data
        motion_convertor.to_unified_input(dataset_up, raw_path, input_unified_path, **kw_uni)
    else:
        print(f"  [2/4] to_unified_input    → skipped (OMOMO_new precomputed)")

    # Resolve the interaction object from the sequence name (OMOMO: sub{N}_{object}_{NNN})
    # and make sure its mesh/URDF exist, else holosoma falls back to its largebox default.
    object_name = None
    if task_type == "object_interaction" and dataset_up in ("OMOMO", "OMOMO_NEW"):
        object_name = _omomo_object_name(seq_name)
        robot_urdf = repo_root() / cfg["robot_config"][robot]["urdf"]
        _ensure_omomo_object_model(retargeter_lo, object_name, robot_urdf)

    # Step c: run retargeter subprocess
    print(f"  [3/4] retargeter subprocess ({cfg['env']})")
    _run_retargeter(
        retargeter_lo, cfg, input_format,
        input_raw_path, output_raw_path, run_dir,
        seq_name, robot, task_type, visualize,
        dataset=dataset_up, object_name=object_name,
    )

    # Retargeters that receive --save_dir (not --output) may name the output file
    # themselves (e.g. holosoma writes {seq_name}_original.npz). Normalize to the
    # expected {seq_name}_output_raw{ext} name so downstream steps are consistent.
    if not output_raw_path.exists():
        candidates = [
            f for f in run_dir.glob(f"{seq_name}*{output_ext}")
            if "_input" not in f.name and "unified" not in f.name
        ]
        if len(candidates) == 1:
            candidates[0].rename(output_raw_path)
            print(f"     renamed {candidates[0].name} → {output_raw_path.name}")
        elif not candidates:
            raise FileNotFoundError(
                f"Retargeter produced no output for {seq_name!r} in {run_dir}"
            )
        else:
            raise RuntimeError(
                f"Ambiguous retargeter output for {seq_name!r}: "
                + ", ".join(f.name for f in candidates)
            )

    # Step d: unified output
    print(f"  [4/4] to_unified_output   → {output_unified_path.name}")
    if dataset_up == "OMOMO_NEW":
        height = 0.0
    else:
        from motion_convertor.unified import load_unified
        height = load_unified(input_unified_path)["height"]
    motion_convertor.to_unified_output(retargeter_lo, output_raw_path, output_unified_path, height)


_EXT_MAP: dict[str, str] = {
    "bvh": ".bvh", "npy": ".npy", "npz": ".npz",
    "pkl": ".pkl", "p": ".p", "pt": ".pt",
}


def _ext_from_format(fmt: str) -> str:
    """Derive file extension from format name (suffix after last underscore)."""
    suffix = fmt.rsplit("_", 1)[-1]
    ext = _EXT_MAP.get(suffix)
    if ext is None:
        raise ValueError(f"Cannot derive file extension from format {fmt!r}")
    return ext


def _run_retargeter(
    retargeter: str,
    cfg: dict,
    input_format: str,
    input_raw_path: Path,
    output_raw_path: Path,
    run_dir: Path,
    seq_name: str,
    robot: str,
    task_type: str = "robot_only",
    visualize: bool = False,
    dataset: str = "",
    object_name: str | None = None,
) -> None:
    env = cfg["env"]
    ep_name = cfg["entry_point_by_input_format"][input_format]
    ep = cfg["entry_points"][ep_name]
    arg_map = ep.get("args", {})
    ep_cwd = repo_root() / ep["cwd"] if "cwd" in ep else repo_root()

    robot_config = cfg.get("robot_config", {})
    robot_cfg = robot_config.get(robot, {})
    if not robot_cfg and robot_config:
        supported = list(robot_config.keys())
        raise ValueError(
            f"Robot {robot!r} not supported by retargeter {retargeter!r}. Supported: {supported}"
        )
    format_args = cfg.get("format_args", {}).get(input_format, {})

    cmd = ep["cmd"]

    # Input
    if "input" in arg_map:
        cmd += f" {arg_map['input']} {input_raw_path}"
    elif "input_dir" in arg_map:
        cmd += f" {arg_map['input_dir']} {input_raw_path.parent}"

    # Output
    if "output" in arg_map:
        cmd += f" {arg_map['output']} {output_raw_path}"
    elif "output_dir" in arg_map:
        cmd += f" {arg_map['output_dir']} {run_dir}"

    # Robot (name style — GMR)
    if "robot" in arg_map and "name" in robot_cfg:
        cmd += f" {arg_map['robot']} {robot_cfg['name']}"

    # Robot (urdf style — holosoma)
    if "robot_urdf" in arg_map and "urdf" in robot_cfg:
        cmd += f" {arg_map['robot_urdf']} {repo_root() / robot_cfg['urdf']}"

    # Robot DOF — extracted from the validated robot name (e.g. G1_27dof → 27)
    # Needed so ROBOT_DOF in RobotConfig matches the loaded URDF (default is 29 for G1)
    if "robot_dof" in arg_map:
        m = _DOF_SUFFIX_RE.search(robot)
        if m:
            dof = int(m.group(0)[1:-3])  # "_27dof" → 27
            cmd += f" {arg_map['robot_dof']} {dof}"

    # Body model path (GMR SMPLX entry points only)
    if "body_model_path" in arg_map and dataset and dataset != "LAFAN":
        cmd += f" {arg_map['body_model_path']} {body_model_smplx_path(dataset).parent}"

    # Holosoma-style args
    if "task_type" in arg_map:
        cmd += f" {arg_map['task_type']} {task_type}"
    if "task_name" in arg_map:
        cmd += f" {arg_map['task_name']} {seq_name}"
    if "object_name" in arg_map and object_name:
        cmd += f" {arg_map['object_name']} {object_name}"
    if "data_format" in arg_map and "data_format" in format_args:
        cmd += f" {arg_map['data_format']} {format_args['data_format']}"

    # Visualize
    can_visualize = visualize and "visualize" in arg_map
    if can_visualize:
        cmd += f" {arg_map['visualize']}"
        if "debug" in arg_map:
            cmd += f" {arg_map['debug']}"

    conda_run(env, cmd, cwd=ep_cwd, interactive=can_visualize)


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Object scale helpers
# ---------------------------------------------------------------------------

def _lookup_omomo_obj_scale(object_name: str) -> float | None:
    """Read the mean obj_scale for a given object from the OMOMO training pickle.

    obj_scale is stored per frame and OMOMO pads some frames with 0.0; those are
    dropped so the mean reflects the real metric size of the object. Sequences are
    matched on the exact object token (sub{N}_{object}_{NNN}) to avoid cross-object
    substring collisions.
    """
    try:
        import joblib
        import numpy as np
        p_file = dataset_path("OMOMO") / "train_diffusion_manip_seq_joints24.p"
        if not p_file.exists():
            return None
        data = joblib.load(p_file)
        scales = []
        for i in data:
            seq_name = data[i].get("seq_name", "")
            if seq_name.split("_")[1:2] != [object_name]:
                continue
            s = np.asarray(data[i]["obj_scale"]).ravel()
            s = s[s > 1e-6]
            if s.size:
                scales.append(float(s.mean()))
        return float(np.mean(scales)) if scales else None
    except Exception:
        return None


# Minimal URDF matching the bundled largebox model (mesh sits next to the URDF, so a
# relative filename resolves for yourdfpy during --visualize).
_OBJECT_URDF_TEMPLATE = """\
<?xml version="1.0" ?>
<robot name="{name}">
  <link name="{name}_link">
    <inertial>
      <mass value="0.1"/>
      <origin xyz="0 0 0"/>
      <inertia ixx="0.002" ixy="0" ixz="0" iyy="0.002" iyz="0" izz="0.002"/>
    </inertial>
    <visual>
      <origin rpy="0 0 0" xyz="0 0 0"/>
      <geometry>
        <mesh filename="{name}.obj" scale="1.0 1.0 1.0"/>
      </geometry>
      <material name="mat">
        <color rgba="0.7 0.8 0.9 0.7"/>
      </material>
    </visual>
    <collision name="{name}">
      <origin rpy="0 0 0" xyz="0 0 0"/>
      <geometry>
        <mesh filename="{name}.obj" scale="1.0 1.0 1.0"/>
      </geometry>
    </collision>
  </link>
</robot>
"""


def _omomo_object_name(seq_name: str) -> str:
    """Object token of an OMOMO sequence name (sub{N}_{object}_{NNN})."""
    parts = seq_name.split("_")
    if len(parts) < 3:
        raise ValueError(f"Cannot derive OMOMO object name from sequence {seq_name!r}")
    return parts[1]


def _write_recentred_scaled_obj(src: Path, dst: Path, scale: float) -> None:
    """Recentre an .obj at its vertex mean and scale it, editing only its 'v' lines.

    Kept pure-text so the retargeting orchestrator needs no mesh library. Vertex normals
    ('vn') keep their direction under a uniform positive scale, so they are left untouched.
    """
    lines = src.read_text().splitlines()
    verts = [[float(x) for x in ln.split()[1:4]] for ln in lines if ln.split()[:1] == ["v"]]
    if not verts:
        raise ValueError(f"No vertices found in {src}")
    n = len(verts)
    cx, cy, cz = (sum(v[k] for v in verts) / n for k in range(3))

    out = []
    for ln in lines:
        if ln.split()[:1] == ["v"]:
            p = ln.split()
            x = (float(p[1]) - cx) * scale
            y = (float(p[2]) - cy) * scale
            z = (float(p[3]) - cz) * scale
            out.append(f"v {x:.6f} {y:.6f} {z:.6f}")
        else:
            out.append(ln)
    dst.write_text("\n".join(out) + "\n")


def _ensure_omomo_object_model(retargeter: str, object_name: str, robot_urdf: Path) -> None:
    """Ensure every per-object holosoma asset exists for an OMOMO interaction object.

    Holosoma resolves the interaction object from three bundled-per-object paths; only
    largebox ships, so any other sequence would silently fall back to the largebox assets:

      - models/{object}/{object}.obj  — surface sampled into the interaction mesh
      - models/{object}/{object}.urdf — object model loaded by viser under --visualize
      - models/g1/{robot}_w_{object}.xml — MuJoCo scene (robot + free-joint object body)

    The .obj reproduces the exact convention of the bundled largebox model: the raw OMOMO
    capture (captured_objects/{object}_cleaned_simplified.obj) recentred at its vertex mean
    and scaled to metric size by obj_scale, so the object poses baked in the .pt line up with
    the mesh surface. The scene XML mirrors the bundled largebox scene, which differs from the
    plain robot XML only by the object mesh asset + body, so a name swap is enough. All are
    generated on the fly (not committed); largebox is left untouched.
    """
    models_dir = repo_root() / "modules" / "01_retargeting" / retargeter / "models" / object_name
    obj_out = models_dir / f"{object_name}.obj"
    urdf_out = models_dir / f"{object_name}.urdf"

    if not (obj_out.exists() and urdf_out.exists()):
        raw_mesh = dataset_path("OMOMO") / "captured_objects" / f"{object_name}_cleaned_simplified.obj"
        if not raw_mesh.exists():
            raise FileNotFoundError(
                f"No captured mesh for object {object_name!r}: {raw_mesh} not found"
            )
        scale = _lookup_omomo_obj_scale(object_name)
        if not scale:
            raise ValueError(f"Could not resolve obj_scale for object {object_name!r}")

        models_dir.mkdir(parents=True, exist_ok=True)
        _write_recentred_scaled_obj(raw_mesh, obj_out, scale)
        urdf_out.write_text(_OBJECT_URDF_TEMPLATE.format(name=object_name))
        print(f"     generated object model → models/{object_name}/ (obj_scale {scale:.4f})")

    # Per-robot MuJoCo scene: holosoma loads {robot}_w_{object}.xml for object_interaction.
    robot_urdf = Path(robot_urdf)
    scene_xml = robot_urdf.with_name(f"{robot_urdf.stem}_w_{object_name}.xml")
    if not scene_xml.exists():
        largebox_scene = robot_urdf.with_name(f"{robot_urdf.stem}_w_largebox.xml")
        if not largebox_scene.exists():
            raise FileNotFoundError(
                f"No largebox scene to derive {scene_xml.name} from: {largebox_scene}"
            )
        scene_xml.write_text(largebox_scene.read_text().replace("largebox", object_name))
        print(f"     generated scene xml → models/g1/{scene_xml.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run a retargeting job.")
    parser.add_argument("--dataset", required=True, help="LAFAN | SFU | OMOMO | OMOMO_new")
    parser.add_argument("--robot", required=True,
                        help="Robot with explicit DOF suffix: G1_29dof | G1_27dof (holosoma_custom only)")
    parser.add_argument("--retargeter", required=True, help="gmr | holosoma | holosoma_custom")
    parser.add_argument("--sequences", nargs="*", help="Subset of sequences (default: all)")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--task-type", default="robot_only",
                        choices=["robot_only", "object_interaction"])
    parser.add_argument("--object-urdf", default=None,
                        help="Path to the object URDF used during retargeting (object_interaction only)."
                             " Saved in config.yaml so train.py can pick it up automatically.")
    parser.add_argument("--visualize", action="store_true")
    args = parser.parse_args()

    dataset = args.dataset.upper()
    robot = _validate_robot(args.robot)
    retargeter = args.retargeter.lower()
    task_type = args.task_type

    cfg = load_module_cfg("01_retargeting", retargeter)

    out_base = output_path("retargeted_motions")
    if dataset == "OMOMO":
        task_suffix = "robot" if task_type == "robot_only" else "object"
        dataset_dir = f"OMOMO_{task_suffix}_{robot}"
    elif dataset == "OMOMO_NEW":
        task_suffix = "robot" if task_type == "robot_only" else "object"
        dataset_dir = f"OMOMO_new_{task_suffix}_{robot}"
    else:
        dataset_dir = f"{dataset}_{robot}"

    run_parent = out_base / dataset_dir / retargeter.upper()
    if args.run_id:
        run_dir = run_parent / args.run_id
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = run_parent / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"Run directory: {run_dir}")

    omomo_data = None
    if dataset == "OMOMO":
        import joblib
        p_file = dataset_path("OMOMO") / "train_diffusion_manip_seq_joints24.p"
        print(f"Loading OMOMO pickle: {p_file}")
        omomo_data = joblib.load(p_file)

    seqs = discover_sequences(dataset, args.sequences)

    if dataset == "OMOMO" and omomo_data is not None:
        expanded = []
        for seq_idx in omomo_data:
            seq_name = omomo_data[seq_idx].get("seq_name", str(seq_idx))
            if args.sequences and seq_name not in args.sequences:
                continue
            expanded.append((seq_name, seq_idx))
        seqs = expanded

    print(f"Processing {len(seqs)} sequences...")

    for i, (seq_name, seq_ref) in enumerate(seqs):
        print(f"\n[{i+1}/{len(seqs)}] {seq_name}")
        if dataset == "OMOMO":
            raw_path = dataset_path("OMOMO") / "train_diffusion_manip_seq_joints24.p"
            seq_data = omomo_data[seq_ref]
        else:
            raw_path = seq_ref
            seq_data = None

        try:
            retarget_sequence(
                seq_name, raw_path, run_dir,
                dataset, robot, retargeter,
                cfg, seq_data=seq_data,
                task_type=task_type,
                visualize=args.visualize,
            )
        except Exception as e:
            print(f"  ERROR: {e}")
            continue

    config_out = run_dir / "config.yaml"
    object_name = "ground"
    object_scale = None
    if task_type == "object_interaction":
        if dataset in ("OMOMO", "OMOMO_NEW") and seqs:
            objs = sorted({_omomo_object_name(name) for name, _ in seqs})
            if len(objs) > 1:
                print(f"  WARNING: run mixes objects {objs}; config.yaml records {objs[0]!r}")
            object_name = objs[0]
        else:
            object_name = "largebox"
        object_scale = _lookup_omomo_obj_scale(object_name)

    with open(config_out, "w") as f:
        yaml.dump({
            "dataset": dataset, "robot": robot, "retargeter": retargeter,
            "task_type": task_type, "object_name": object_name,
            "object_urdf": args.object_urdf,
            "object_scale": object_scale,
            "run_dir": str(run_dir), "sequences": args.sequences,
        }, f)

    latest_link = run_parent / "latest"
    if latest_link.is_symlink():
        latest_link.unlink()
    latest_link.symlink_to(run_dir.name)

    print(f"\nDone. Output: {run_dir}")
    print(f"Latest symlink → {run_dir.name}")


if __name__ == "__main__":
    main()
