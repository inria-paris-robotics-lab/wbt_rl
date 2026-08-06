#!/usr/bin/env python3
"""
enrich_dynamics.py — add physics (torques + contact forces) to a retargeted clip.

Stage 05. Retargeting is kinematic: it says where the robot should be, never what
it would take to get there. This runs the clip through SPIDER's sampling-MPC in
MuJoCo Warp, which replays it under real dynamics — gravity, object mass, friction,
contacts — and reports what the motion actually costs and whether it holds up.

Output is a SIDECAR next to the retargeting run's own files. Nothing about the
unified format changes, and every existing consumer keeps working untouched:

    data/01_retargeted_motions/{dataset}_{robot}/{retargeter}/run_*/
        {seq}_output_dynamics.npz      ← qpos, qvel, tau, contact forces
        dynamics/{seq}/                ← SPIDER scene + raw rollout, kept for replay

Usage:
    python scripts/enrich_dynamics.py \\
        --dataset OMOMO --robot G1_29dof --retargeter holosoma_custom \\
        --scene cfg/05_dynamics/scenes/femto14_box36.yaml \\
        [--sequences femto14 ...] [--retarget-run latest] \\
        [--motion path/to/motion.npz]   # bypass the run lookup entirely
        [--num-samples 1024] [--max-iterations 8]

⚠ SPIDER is CC BY-NC 4.0 (non-commercial). See cfg/05_dynamics/spider.yaml.
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from motion_convertor._config import output_path, repo_root  # noqa: E402
from motion_convertor._subprocess import load_module_cfg, venv_run  # noqa: E402
from motion_convertor.connectors import get_connector  # noqa: E402

_DOF_SUFFIX_RE = re.compile(r"_\d+dof$", re.IGNORECASE)


def _validate_robot(robot: str) -> str:
    m = _DOF_SUFFIX_RE.search(robot)
    if not m:
        raise ValueError(
            f"--robot must carry an explicit DOF suffix (e.g. G1_29dof), got {robot!r}"
        )
    return robot[: m.start()].upper() + m.group(0).lower()


def resolve_retarget_run(dataset: str, robot: str, retargeter: str, run_id: str) -> Path:
    """Mirror scripts/train.py's run resolution so both stages agree on 'latest'."""
    run_parent = output_path("retargeted_motions") / f"{dataset}_{robot}" / retargeter.upper()
    if run_id == "latest":
        link = run_parent / "latest"
        if not link.exists():
            raise FileNotFoundError(f"No 'latest' symlink in {run_parent}")
        return link.resolve()
    run_dir = run_parent / run_id
    if not run_dir.exists():
        raise FileNotFoundError(f"Retarget run not found: {run_dir}")
    return run_dir


def _hydra_tokens(cfg: dict, values: dict) -> str:
    """Render `key=value` tokens for SPIDER's Hydra CLI (not `--flag value`)."""
    arg_map = cfg.get("args", {})
    tokens = list(cfg.get("base_overrides", []))
    for key, value in values.items():
        flag = arg_map.get(key)
        if flag is not None and value is not None:
            tokens.append(f"{flag}={value}")
    return " ".join(tokens)


def enrich_sequence(
    motion_path: Path,
    scene_manifest: Path,
    work_root: Path,
    sidecar_path: Path,
    cfg: dict,
    overrides: dict,
    robot_assets: Path | None = None,
) -> None:
    """Build the SPIDER scene for one clip, run the physics pass, extract the sidecar."""
    root = repo_root()
    task = motion_path.stem
    dataset_name = "wbtrl"
    robot_type = cfg["defaults"]["robot_type"]
    # SPIDER hard-codes this directory shape; see spider/io.get_processed_data_dir.
    task_dir = (
        work_root / "processed" / dataset_name / robot_type / "humanoid_nobj" / task
    )

    print(f"\n[1/3] scene + kinematic trajectory  ({scene_manifest.name})")
    get_connector("holosoma_trainer_npz", "spider_scene_npz")(
        motion_path, task_dir, scene_manifest=scene_manifest, data_id="0",
        robot_assets=robot_assets,
    )

    meta = json.loads((task_dir / "wbt_scene_meta.json").read_text())
    num_objects = int(meta["num_objects"])
    n_frames = int(meta["n_frames"])
    if num_objects == 0:
        print("      no dynamic objects — running robot-only physics")

    print(f"[2/3] physics pass  ({num_objects} dynamic object(s), {n_frames} frames)")
    values = {
        "task": task,
        "data_id": 0,
        "robot_type": robot_type,
        "dataset_name": dataset_name,
        "dataset_dir": work_root,
        "num_objects": num_objects,
        **overrides,
    }
    tokens = _hydra_tokens(cfg, values)
    venv_run(
        str(root / cfg["venv"]),
        f"{cfg['cmd']} {tokens}",
        cwd=root / cfg["module_root"],
    )

    print("[3/3] extracting torques + contact forces")
    get_connector("spider_scene_npz", "spider_dynamics_npz")(
        task_dir, sidecar_path, data_id="0", ref_frames=n_frames
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scene", required=True,
                   help="scene manifest yaml (cfg/05_dynamics/scenes/*.yaml)")
    p.add_argument("--module", default="spider",
                   help="cfg/05_dynamics/{module}.yaml -- e.g. spider_noguidance to A/B a config variant")
    p.add_argument("--motion",
                   help="run on this motion npz directly, skipping the retarget-run lookup")
    p.add_argument("--dataset")
    p.add_argument("--robot")
    p.add_argument("--retargeter")
    p.add_argument("--retarget-run", default="latest")
    p.add_argument("--sequences", nargs="*",
                   help="sequence names to process (default: every trainer input in the run)")
    p.add_argument("--out-dir",
                   help="where to write sidecars (default: alongside the retargeting run)")
    p.add_argument("--robot-assets",
                   help="override SPIDER robot asset dir (default: cfg/05_dynamics/spider.yaml's "
                        "robot_assets) -- e.g. to A/B different actuator gains")
    p.add_argument("--num-samples", type=int)
    p.add_argument("--max-iterations", type=int)
    p.add_argument("--keep-work", action="store_true",
                   help="keep SPIDER's scene + raw rollout (default: kept; use for clarity)")
    args = p.parse_args()

    cfg = load_module_cfg("05_dynamics", args.module)
    scene_manifest = Path(args.scene)
    if not scene_manifest.is_absolute():
        scene_manifest = repo_root() / scene_manifest
    if not scene_manifest.exists():
        raise FileNotFoundError(f"Scene manifest not found: {scene_manifest}")

    venv = repo_root() / cfg["venv"]
    if not (venv / "bin" / "python").exists():
        raise SystemExit(
            f"SPIDER is not installed ({venv} missing).\n"
            f"Run:  ./install.sh spider"
        )

    overrides = {
        "num_samples": args.num_samples or cfg["defaults"]["num_samples"],
        "max_num_iterations": args.max_iterations or cfg["defaults"]["max_num_iterations"],
        "sim_dt": cfg["defaults"]["sim_dt"],
        "ref_dt": cfg["defaults"]["ref_dt"],
        "trace_dt": cfg["defaults"]["trace_dt"],
    }

    # --- resolve which clips to process -------------------------------------
    if args.motion:
        motions = [Path(args.motion).resolve()]
        # Must be absolute: SPIDER's subprocess cd's into the module dir (so
        # Hydra can find examples/config/), so a relative dataset_dir built
        # from this would resolve against THAT cwd, not the repo root.
        out_dir = Path(args.out_dir).resolve() if args.out_dir else motions[0].parent
    else:
        missing = [f for f in ("dataset", "robot", "retargeter") if not getattr(args, f)]
        if missing:
            p.error(f"--motion not given, so these are required: {', '.join('--' + m for m in missing)}")
        robot = _validate_robot(args.robot)
        run_dir = resolve_retarget_run(args.dataset, robot, args.retargeter, args.retarget_run)
        print(f"Retarget run: {run_dir}")
        motions = sorted(run_dir.glob("*_trainer_input.npz"))
        if not motions:
            raise FileNotFoundError(
                f"No *_trainer_input.npz in {run_dir} — run scripts/train.py first "
                f"(stage 05 consumes the trainer-input form, which carries object poses)."
            )
        if args.sequences:
            wanted = set(args.sequences)
            motions = [m for m in motions
                       if m.name.replace("_trainer_input.npz", "") in wanted]
            if not motions:
                raise SystemExit(f"None of {sorted(wanted)} found in {run_dir}")
        out_dir = Path(args.out_dir).resolve() if args.out_dir else run_dir

    out_dir.mkdir(parents=True, exist_ok=True)
    work_root = out_dir / "dynamics"

    robot_assets = Path(args.robot_assets).resolve() if args.robot_assets else None

    print(f"Scene manifest: {scene_manifest}")
    print(f"Sequences:      {[m.stem for m in motions]}")
    print(f"Sidecars ->     {out_dir}")
    if robot_assets:
        print(f"Robot assets:   {robot_assets}  (override)")

    for motion in motions:
        seq = motion.stem.replace("_trainer_input", "")
        sidecar = out_dir / f"{seq}_output_dynamics.npz"
        enrich_sequence(motion, scene_manifest, work_root, sidecar, cfg, overrides, robot_assets)

        d = np.load(sidecar, allow_pickle=True)
        tau = d["tau"]
        print(
            f"\n  ✓ {seq}: {tau.shape[0]} frames, "
            f"|tau| mean={np.abs(tau).mean():.2f} max={np.abs(tau).max():.2f} Nm, "
            f"{len(d['contact_pairs'])} contact pairs"
        )

    print(f"\nDone. {len(motions)} sequence(s) enriched.")


if __name__ == "__main__":
    main()
