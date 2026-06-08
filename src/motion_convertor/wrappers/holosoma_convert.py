#!/usr/bin/env python3
"""
holosoma data conversion wrapper — runs in the `hsretargeting` conda environment.

Thin CLI shim around holosoma's convert_data_format_mj.py.
Reads a retargeter output .npz (containing qpos), runs the MuJoCo simulation
loop via the native holosoma bridge, and writes form B output .npz.

Usage (called via subprocess from motion_convertor):
    python src/motion_convertor/wrappers/holosoma_convert.py \\
        --input_file  <path/to/output_raw.npz> \\
        --output_name <path/to/trainer_input.npz> \\
        --robot_type  g1 \\
        --robot_dof   27 \\
        --input_fps   30 \\
        --output_fps  50 \\
        --retargeter_root modules/01_retargeting/holosoma

This is a pass-through to convert_data_format_mj.py with --once so it runs
one full pass and exits (no interactive viewer loop).
"""
import subprocess
import sys
from pathlib import Path

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--output_name", required=True)
    parser.add_argument("--robot_type", default="g1")
    parser.add_argument("--robot_dof", type=int, default=29)
    parser.add_argument("--input_fps", type=int, default=30)
    parser.add_argument("--output_fps", type=int, default=50)
    parser.add_argument("--object_name", default="ground",
                        help="MuJoCo scene object name (default: ground for robot-only)")
    parser.add_argument("--retargeter_root", required=True,
                        help="Path to the holosoma retargeting root directory (containing data_conversion/)")
    args = parser.parse_args()

    retargeter_root = Path(args.retargeter_root).resolve()
    bridge = retargeter_root / "data_conversion/convert_data_format_mj.py"

    if not bridge.exists():
        print(f"ERROR: Bridge script not found at {bridge}")
        sys.exit(1)

    # convert_data_format_mj.py uses tyro.
    # It has a flat structure but nested configs are prefixed with --robot-config.
    cmd = [
        sys.executable, str(bridge),
        f"--input_file={args.input_file}",
        f"--output_name={args.output_name}",
        f"--robot={args.robot_type}",
        f"--robot-config.robot-type={args.robot_type}",
        f"--robot-config.robot-dof={args.robot_dof}",
        f"--input_fps={args.input_fps}",
        f"--output_fps={args.output_fps}",
        f"--object_name={args.object_name}",
        "--once",
    ]
    if args.object_name != "ground":
        cmd.append("--has-dynamic-object")

    # convert_data_format_mj.py resolves models/ relative to cwd.
    # MuJoCo's passive viewer can segfault on cleanup after viewer.close(), so
    # we treat a non-zero exit as success if the output file was written.
    result = subprocess.run(cmd, cwd=str(retargeter_root))
    output_path = Path(args.output_name)
    if result.returncode != 0 and not output_path.exists():
        sys.exit(result.returncode)


if __name__ == "__main__":
    main()
