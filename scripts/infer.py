#!/usr/bin/env python3
"""
infer.py — run a trained policy via ROS2 (holosoma_custom only).

Usage:
    # local policy run
    python scripts/infer.py \\
        --dataset LAFAN --robot G1_29dof \\
        --retargeter GMR --trainer holosoma_custom \\
        --config inference:g1-29dof-wbt \\
        [--policy-run latest]

    # wandb run
    python scripts/infer.py \\
        --trainer holosoma_custom \\
        --config inference:g1-29dof-wbt \\
        --wandb-run wandb://entity/project/run_id/model.onnx
"""
import argparse
import sys
import yaml
from pathlib import Path

_REPO_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from motion_convertor._config import output_path, repo_root
from motion_convertor._subprocess import conda_run


def resolve_policy_run(dataset: str, robot: str, retargeter: str, trainer: str, run_id: str) -> Path:
    """Resolve a policy run directory (or 'latest' symlink)."""
    base = output_path("policies")
    run_parent = base / f"{dataset}_{robot}" / f"{retargeter}_{trainer}"

    if run_id == "latest":
        link = run_parent / "latest"
        if not link.is_symlink():
            raise FileNotFoundError(f"No 'latest' symlink in {run_parent}")
        if not link.exists():
            raise FileNotFoundError(f"'latest' symlink in {run_parent} points to a missing directory")
        return link.resolve()

    run_dir = run_parent / run_id
    if not run_dir.exists():
        raise FileNotFoundError(f"Policy run not found: {run_dir}")
    return run_dir


def _build_infer_cmd(ep: dict, config: str, model_path: str) -> str:
    """Build the subprocess command string for run_policy.py."""
    arg_map = ep.get("args", {})
    cmd = ep["cmd"].strip()
    cmd += f" {config}"
    cmd += f" {arg_map['model_path']} {model_path}"
    extra = ep.get("extra_args", "")
    if extra:
        cmd += f" {extra}"
    return cmd


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run inference with a trained policy via ROS2.")
    parser.add_argument("--trainer", default="holosoma_custom",
                        help="Inference engine — must match cfg/03_inference/{trainer}.yaml (default: holosoma_custom)")
    parser.add_argument("--config", required=True,
                        help="Tyro inference config subcommand (e.g. inference:g1-29dof-wbt)")
    parser.add_argument("--wandb-run", default=None,
                        help="Wandb model URI (e.g. wandb://entity/project/run_id/model.onnx). "
                             "Mutually exclusive with --dataset/--robot/--retargeter.")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--robot", default=None)
    parser.add_argument("--retargeter", default=None)
    parser.add_argument("--policy-run", default="latest",
                        help="Policy run ID or 'latest' (default: latest). Local mode only.")
    parser.add_argument("-i", "--interactive", action="store_true", default=True,
                        help="Run in interactive mode (stdin inherited) to allow keyboard control (default: True)")
    parser.add_argument("--no-interactive", action="store_false", dest="interactive",
                        help="Disable interactive mode")
    return parser


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Validate mutual exclusion and required-field rules. Calls parser.error() on failure."""
    if args.trainer == "holosoma":
        parser.error("holosoma inference is not supported, use holosoma_custom")

    local_args = [args.dataset, args.robot, args.retargeter]
    has_wandb = args.wandb_run is not None
    has_any_local = any(a is not None for a in local_args)
    has_all_local = all(a is not None for a in local_args)

    if has_wandb and has_any_local:
        parser.error("--wandb-run is mutually exclusive with --dataset/--robot/--retargeter")

    if not has_wandb and not has_all_local:
        parser.error(
            "provide either --wandb-run OR all three of --dataset, --robot, --retargeter"
        )


def main():
    parser = _build_parser()
    args = parser.parse_args()
    _validate_args(parser, args)

    trainer = args.trainer.lower()

    cfg_path = repo_root() / "cfg" / "03_inference" / f"{trainer}.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    ep = cfg["entry_points"]["ros2"]
    env = cfg["env"]

    if args.wandb_run:
        model_path = args.wandb_run
        print(f"Wandb model: {model_path}")
    else:
        dataset = args.dataset.upper()
        robot = args.robot
        retargeter = args.retargeter.lower()

        policy_run = resolve_policy_run(dataset, robot, retargeter, trainer, args.policy_run)
        print(f"Policy run: {policy_run}")

        onnx_files = list(policy_run.glob("*.onnx"))
        pt_files = list(policy_run.glob("*.pt"))
        model_file = onnx_files[0] if onnx_files else (pt_files[0] if pt_files else None)
        if model_file is None:
            raise FileNotFoundError(f"No .onnx or .pt policy file found in {policy_run}")
        model_path = str(model_file)
        print(f"Model: {model_path}")

    cmd = _build_infer_cmd(ep, args.config, model_path)
    prefix = cfg.get("env_prefix")
    print(f"Launching {trainer} (ros2)...")
    conda_run(env, cmd, cwd=repo_root(), interactive=args.interactive, prefix=prefix)


if __name__ == "__main__":
    main()
