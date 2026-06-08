#!/usr/bin/env python3
"""
Wrapper: GMR smplx retargeting with configurable body model path.

smplx_to_robot.py hardcodes SMPLX_FOLDER = assets/body_models relative to the script.
This wrapper injects the correct path by running the script source with a patched global.

Runs in gmr env.

Usage:
    python src/motion_convertor/wrappers/gmr_smplx.py \
        --smplx_file <input.npz> \
        --save_path <output.pkl> \
        --robot unitree_g1 \
        --body_model_path <path/containing/smplx/>
"""
import argparse
import sys
from pathlib import Path

# GMR repo root
_GMR_ROOT = Path(__file__).parents[3] / "modules" / "01_retargeting" / "GMR"
_SMPLX_SCRIPT = _GMR_ROOT / "scripts" / "smplx_to_robot.py"

sys.path.insert(0, str(_GMR_ROOT))
sys.path.insert(0, str(_GMR_ROOT / "scripts"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smplx_file", required=True)
    parser.add_argument("--save_path", required=True)
    parser.add_argument("--robot", required=True)
    parser.add_argument("--body_model_path", required=True,
                        help="Directory containing the smplx/ subfolder with SMPLX_*.pkl files")
    args = parser.parse_args()

    body_model_path = Path(args.body_model_path)

    # Patch sys.argv so smplx_to_robot.py's argparse sees the right args
    sys.argv = [
        str(_SMPLX_SCRIPT),
        "--smplx_file", args.smplx_file,
        "--save_path",  args.save_path,
        "--robot",      args.robot,
    ]

    # Read and compile the script source, then exec with SMPLX_FOLDER pre-injected
    source = _SMPLX_SCRIPT.read_text()
    # Replace the hardcoded SMPLX_FOLDER line with our path
    source = source.replace(
        'SMPLX_FOLDER = HERE / ".." / "assets" / "body_models"',
        f'SMPLX_FOLDER = Path(r"{body_model_path}")',
    )

    script_globals = {
        "__name__": "__main__",
        "__file__": str(_SMPLX_SCRIPT),
        "__builtins__": __builtins__,
        "Path": Path,
    }
    exec(compile(source, str(_SMPLX_SCRIPT), "exec"), script_globals)


if __name__ == "__main__":
    main()
