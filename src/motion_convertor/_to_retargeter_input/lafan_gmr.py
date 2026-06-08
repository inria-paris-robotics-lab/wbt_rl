"""
LAFAN → GMR retargeter input.

GMR reads .bvh natively — no conversion needed.
This module copies the source .bvh to the expected output path.
"""
import shutil
from pathlib import Path


def convert(bvh_path: Path | str, out_path: Path | str) -> None:
    """
    Copy a LAFAN .bvh file to the retargeter input location.

    Parameters
    ----------
    bvh_path : source .bvh file
    out_path : destination path (should end with _input_raw.bvh)
    """
    bvh_path = Path(bvh_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(bvh_path, out_path)
