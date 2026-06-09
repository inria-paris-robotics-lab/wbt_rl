#!/usr/bin/env python3
"""
train.py — prepare trainer input and launch training from an existing retargeting run.

Usage:
    python scripts/train.py \\
        --dataset LAFAN \\
        --robot G1_29dof \\
        --retargeter holosoma \\
        --trainer holosoma \\
        --simulator isaacsim \\
        [--with-object] \\
        [--retarget-run latest] \\
        [--num-envs 4096] \\
        [--checkpoint path/to/checkpoint.pt]

Output: data/02_policies/{dataset}_{robot_base}/{retargeter}_{trainer}/run_{timestamp}/
"""
import argparse
import re
import sys
import yaml
from datetime import datetime
from pathlib import Path

_REPO_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

import motion_convertor
from motion_convertor._config import output_path, repo_root
from motion_convertor._subprocess import conda_run

_DOF_SUFFIX_RE = re.compile(r"_\d+dof$", re.IGNORECASE)


def _validate_robot(robot: str) -> str:
    """Require explicit dof suffix; normalize to uppercase base + lowercase suffix (G1_29dof)."""
    m = _DOF_SUFFIX_RE.search(robot)
    if not m:
        raise ValueError(
            f"--robot must include an explicit DOF suffix (e.g. G1_29dof or G1_27dof), got {robot!r}"
        )
    return robot[:m.start()].upper() + m.group(0).lower()


def _retarget_dataset_dir(dataset: str, robot: str, retarget_task_type: str) -> str:
    """Return the directory name used by retarget.py for a given dataset/robot combination.

    retarget_task_type mirrors the --task-type argument of retarget.py and controls
    which retarget output directory is used — independently of the training --with-object flag.
    Must stay in sync with the dataset_dir logic in scripts/retarget.py.
    """
    if dataset == "OMOMO":
        suffix = "object" if retarget_task_type == "object_interaction" else "robot"
        return f"OMOMO_{suffix}_{robot}"
    if dataset == "OMOMO_NEW":
        suffix = "robot" if retarget_task_type == "robot_only" else "object"
        return f"OMOMO_new_{suffix}_{robot}"
    return f"{dataset}_{robot}"


def resolve_retarget_run(dataset: str, robot: str, retargeter: str, run_id: str,
                         retarget_task_type: str = "robot_only") -> Path:
    """Resolve a retargeting run directory. robot must be the normalized form (e.g. G1_29dof)."""
    base = output_path("retargeted_motions")
    dataset_dir = _retarget_dataset_dir(dataset, robot, retarget_task_type)
    run_parent = base / dataset_dir / retargeter.upper()

    if run_id == "latest":
        link = run_parent / "latest"
        if not link.exists():
            raise FileNotFoundError(f"No 'latest' symlink in {run_parent}")
        return link.resolve()

    run_dir = run_parent / run_id
    if not run_dir.exists():
        raise FileNotFoundError(f"Retarget run not found: {run_dir}")
    return run_dir


def prepare_trainer_inputs(retarget_run: Path, retargeter: str, trainer: str, robot: str,
                           with_object: bool = True) -> tuple[list[Path], str | None]:
    from motion_convertor._subprocess import load_module_cfg
    from motion_convertor.formats import validate_format

    ret_cfg = load_module_cfg("01_retargeting", retargeter.lower())
    output_format = ret_cfg["native_output_format"]
    validate_format(output_format)

    # Load retargeting run config to get task metadata
    run_cfg_path = retarget_run / "config.yaml"
    if run_cfg_path.exists():
        with open(run_cfg_path) as f:
            run_cfg = yaml.safe_load(f)
    else:
        run_cfg = {}

    task_type = run_cfg.get("task_type", "robot_only")
    run_object_name = run_cfg.get("object_name", "ground")
    run_object_urdf = run_cfg.get("object_urdf", None)
    run_object_scale = run_cfg.get("object_scale", None)

    # When training robot_only on an object_interaction retarget run, strip object data
    # by using "ground" as the object name (MuJoCo bridge omits --has-dynamic-object).
    if not with_object:
        object_name = "ground"
    else:
        object_name = run_object_name

    suffix = output_format.rsplit("_", 1)[-1]
    _EXT_MAP = {"bvh": ".bvh", "npy": ".npy", "npz": ".npz", "pkl": ".pkl", "p": ".p", "pt": ".pt"}
    output_ext = _EXT_MAP[suffix]

    output_raw_files = sorted(retarget_run.glob(f"*_output_raw{output_ext}"))

    # Retargeters that receive --save_dir may name output files themselves
    # (e.g. holosoma writes {seq_name}_original.npz, or generic retargeted.npz).
    # Rename to _output_raw so downstream steps are consistent.
    if not output_raw_files:
        candidates = [
            f for f in retarget_run.glob(f"*{output_ext}")
            if "_input" not in f.name and "unified" not in f.name and "trainer_input" not in f.name
        ]
        for candidate in candidates:
            if "_original" in candidate.name:
                normalized = retarget_run / candidate.name.replace("_original", "_output_raw")
            elif "_output_raw" not in candidate.name:
                normalized = retarget_run / f"{candidate.stem}_output_raw{output_ext}"
            else:
                normalized = candidate

            if normalized != candidate:
                candidate.rename(normalized)
                print(f"  renamed {candidate.name} → {normalized.name}")
        output_raw_files = sorted(retarget_run.glob(f"*_output_raw{output_ext}"))

    trainer_input_paths = []

    # When stripping object data from an object_interaction run, use a distinct suffix
    # so both variants (with/without object) can coexist in the same retarget directory.
    strip_object = task_type == "object_interaction" and not with_object
    trainer_input_suffix = "_robot_only_trainer_input.npz" if strip_object else "_trainer_input.npz"

    for output_raw in output_raw_files:
        seq_name = output_raw.stem.replace("_output_raw", "")
        trainer_input_path = retarget_run / f"{seq_name}{trainer_input_suffix}"

        if trainer_input_path.exists():
            print(f"  [skip] {trainer_input_path.name} already exists")
        else:
            print(f"  to_trainer_input → {trainer_input_path.name} (object: {object_name})")
            motion_convertor.to_trainer_input(
                retargeter.lower(), trainer.lower(),
                output_raw, trainer_input_path,
                robot=robot,
                object_name=object_name,
            )

        trainer_input_paths.append(trainer_input_path)

    return trainer_input_paths, run_object_urdf, run_object_scale


def _resolve_exp_name(cfg: dict, robot: str, simulator: str, algo: str, with_object: bool,
                      with_object_actor: bool = False) -> str:
    """Look up the holosoma exp: subcommand name from robot_exp_map.

    robot_exp_map structure: {robot → {simulator → {algo → {robot_only|with_object|with_object_actor → exp_name}}}}
    Raises ValueError with a clear message when the combination is unsupported.
    """
    robot_exp_map = cfg.get("robot_exp_map", {})
    robot_entry = robot_exp_map.get(robot)
    if robot_entry is None:
        raise ValueError(
            f"Robot {robot!r} not in robot_exp_map for this trainer. "
            f"Supported: {list(robot_exp_map)}"
        )

    sim_entry = robot_entry.get(simulator)
    if sim_entry is None:
        raise ValueError(
            f"Simulator {simulator!r} not supported for {robot!r} with this trainer. "
            f"Supported simulators: {list(robot_entry)}"
        )

    algo_entry = sim_entry.get(algo)
    if algo_entry is None:
        raise ValueError(
            f"Algorithm {algo!r} not supported for {robot!r} / {simulator!r} with this trainer. "
            f"Supported algorithms: {list(sim_entry)}"
        )

    if with_object_actor:
        task_key = "with_object_actor"
    elif with_object:
        task_key = "with_object"
    else:
        task_key = "robot_only"
    exp_name = algo_entry.get(task_key)
    if exp_name is None:
        raise ValueError(
            f"Task type {task_key!r} not found in robot_exp_map for "
            f"{robot!r} / {simulator!r} / {algo!r}"
        )
    return exp_name


def run_training(
    cfg: dict,
    simulator: str,
    trainer_input_paths: list[Path],
    policy_run_dir: Path,
    robot: str,
    algo: str,
    with_object: bool,
    logger_type: str,
    num_envs: int | None,
    checkpoint: str | None,
    object_urdf: str | None = None,
    with_object_actor: bool = False,
    no_video: bool = False,
) -> None:
    """Launch the trainer subprocess."""
    sim_cfg = cfg["simulators"][simulator]
    env = sim_cfg["env"]
    base_cmd = sim_cfg["cmd"]
    extra_args = sim_cfg.get("extra_args", "")
    env_vars = sim_cfg.get("env_vars")
    arg_map = cfg.get("args", {})

    exp_name = _resolve_exp_name(cfg, robot, simulator, algo, with_object, with_object_actor)

    # tyro requires all positional subcommands before any --flag arguments
    cmd = base_cmd
    cmd += f" exp:{exp_name}"
    if extra_args:
        cmd += f" {extra_args}"
    cmd += f" logger:{logger_type}"

    if not trainer_input_paths:
        raise ValueError(f"No trainer sequences found. Ensure retargeting outputs are present.")

    if "motion_dir" in arg_map:
        motion_input = trainer_input_paths[0].parent
    else:
        motion_input = trainer_input_paths[0]
    cmd += f" {arg_map.get('motion_dir', arg_map.get('motion_file'))} {motion_input}"

    cmd += f" {arg_map['output_dir']} {policy_run_dir}"
    if num_envs:
        cmd += f" {arg_map['num_envs']} {num_envs}"
    if checkpoint:
        cmd += f" {arg_map['checkpoint']} {checkpoint}"
    if object_urdf and "object_urdf" in arg_map:
        cmd += f" {arg_map['object_urdf']} {object_urdf}"
    if no_video:
        cmd += " --logger.video.enabled False"

    conda_run(env, cmd, cwd=repo_root(), interactive=True, env_vars=env_vars)


def main():
    parser = argparse.ArgumentParser(description="Run a training job.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument(
        "--robot", required=True,
        help="Robot with explicit DOF suffix: G1_29dof | G1_27dof",
    )
    parser.add_argument("--retargeter", required=True)
    parser.add_argument("--trainer", required=True, help="holosoma | holosoma_custom")
    parser.add_argument(
        "--simulator", default="isaacsim",
        help="Simulator backend — must match a key under 'simulators:' in the trainer YAML"
             " and be listed in robot_exp_map (default: isaacsim)",
    )
    parser.add_argument("--retarget-task-type", default="robot_only",
                        choices=["robot_only", "object_interaction"],
                        help="Which retarget run to use as source (mirrors retarget.py --task-type)."
                             " Independent of --with-object (default: robot_only)")
    parser.add_argument("--with-object", action="store_true",
                        help="Train with object in scene — object pose in critic only (default: robot-only)."
                             " Independent of --retarget-task-type: you can train without object"
                             " on a run retargeted with object, and vice-versa.")
    parser.add_argument("--with-object-actor", action="store_true",
                        help="Train with object in scene — object pose in both actor and critic obs."
                             " Implies --with-object. Only supported for PPO.")
    parser.add_argument("--algo", default="ppo", choices=["ppo", "fast_sac"],
                        help="RL algorithm (default: ppo)")
    parser.add_argument("--logger-type", default="wandb",
                        choices=["wandb", "wandb_offline", "disabled"],
                        help="Trainer logger backend (default: wandb)")
    parser.add_argument("--retarget-run", default="latest",
                        help="Retargeting run ID or 'latest' (default: latest)")
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--checkpoint", default=None, help="Resume from checkpoint")
    parser.add_argument("--object-urdf", default=None,
                        help="Path to a custom object URDF (overrides the one hardcoded in the exp preset)."
                             " Only used when --with-object is set.")
    parser.add_argument("--no-video", action="store_true",
                        help="Disable video recording during training (sets --logger.video.enabled False).")
    args = parser.parse_args()

    dataset = args.dataset.upper()
    robot = _validate_robot(args.robot)
    retargeter = args.retargeter.lower()
    trainer = args.trainer.lower()

    # Load trainer config
    cfg_path = repo_root() / "cfg" / "02_training" / f"{trainer}.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    # Locate retargeting run (path uses base robot name, e.g. G1)
    retarget_run = resolve_retarget_run(dataset, robot, retargeter, args.retarget_run,
                                        retarget_task_type=args.retarget_task_type)
    print(f"Retarget run: {retarget_run}")

    with_object_actor = args.with_object_actor
    with_object = args.with_object or with_object_actor

    # Prepare trainer inputs
    print("Preparing trainer inputs...")
    trainer_inputs, run_object_urdf, run_object_scale = prepare_trainer_inputs(
        retarget_run, retargeter, trainer, robot, with_object=with_object)
    print(f"  {len(trainer_inputs)} sequences ready")

    # object_urdf: CLI flag takes precedence, retarget run config.yaml is the fallback
    object_urdf = args.object_urdf or run_object_urdf
    object_scale = run_object_scale or 1.0

    # If path points to a .obj mesh, generate a URDF from it on the fly
    if object_urdf and Path(object_urdf).suffix.lower() == ".obj":
        from motion_convertor._obj_to_urdf import obj_to_urdf
        object_urdf = str(obj_to_urdf(object_urdf, scale=object_scale))
        print(f"  [obj→urdf] scale={object_scale:.4f} → {object_urdf}")

    # Create policy run directory
    policies_base = output_path("policies")
    policy_parent = policies_base / f"{dataset}_{robot}" / f"{retargeter}_{trainer}"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    policy_run_dir = policy_parent / f"run_{timestamp}"
    policy_run_dir.mkdir(parents=True, exist_ok=True)

    print(f"Policy run: {policy_run_dir}")

    # Launch training
    if with_object_actor:
        task_label = "with_object_actor"
    elif with_object:
        task_label = "with_object"
    else:
        task_label = "robot_only"
    logger_type = args.logger_type.replace("_", "-")
    print(f"Launching {trainer} training ({args.simulator}, {robot}, {args.algo}, {task_label}, logger:{logger_type})...")
    run_training(cfg, args.simulator, trainer_inputs, policy_run_dir, robot,
                 args.algo, with_object, logger_type, args.num_envs, args.checkpoint,
                 object_urdf=object_urdf, with_object_actor=with_object_actor,
                 no_video=args.no_video)

    # Write config snapshot
    with open(policy_run_dir / "config.yaml", "w") as f:
        yaml.dump({
            "dataset": dataset,
            "robot": robot,
            "retargeter": retargeter,
            "trainer": trainer,
            "simulator": args.simulator,
            "algo": args.algo,
            "logger_type": logger_type,
            "retarget_task_type": args.retarget_task_type,
            "with_object": with_object,
            "with_object_actor": with_object_actor,
            "retarget_run": str(retarget_run),
            "num_envs": args.num_envs,
            "object_urdf": object_urdf,
            "no_video": args.no_video,
        }, f)

    # Update latest symlink
    latest_link = policy_parent / "latest"
    if latest_link.is_symlink():
        latest_link.unlink()
    latest_link.symlink_to(policy_run_dir.name)

    print(f"\nDone. Policy output: {policy_run_dir}")


if __name__ == "__main__":
    main()
