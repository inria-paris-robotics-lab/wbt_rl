#!/usr/bin/env python3
"""
Wrapper: NMR retargeting, one sequence.

NMR's inference.py CLI takes --src / --output-dir and names the output after the input
stem. WBT-RL wants an explicit output path, so this wrapper runs the inference in a
temp dir and moves the result into place.

Runs in the nmr env.

Usage:
    python src/motion_convertor/wrappers/nmr_smplx.py \
        --smplx_file <input.npz> \
        --save_path <output.npz>
"""
import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

_NMR_ROOT = Path(__file__).parents[3] / "modules" / "01_retargeting" / "NMR"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smplx_file", required=True)
    parser.add_argument("--save_path", required=True)
    args = parser.parse_args()

    src = Path(args.smplx_file)
    dst = Path(args.save_path)
    dst.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as td:
        subprocess.run(
            [sys.executable, "inference.py", "--src", str(src), "--output-dir", td],
            cwd=_NMR_ROOT, check=True,
        )
        produced = list(Path(td).glob("*.npz"))
        if len(produced) != 1:
            raise RuntimeError(f"Expected 1 NMR output, got {len(produced)}: {produced}")
        shutil.move(str(produced[0]), dst)

    print(f"NMR output → {dst}")


if __name__ == "__main__":
    main()
