"""
SFU → GMR retargeter input.

GMR reads SFU AMASS .npz natively (keys already match).
This module copies the source .npz to the expected output path.
"""
import shutil
from pathlib import Path


def convert(npz_path: Path | str, out_path: Path | str) -> None:
    """
    Copy a SFU .npz file to the retargeter input location.

    Parameters
    ----------
    npz_path : source .npz file
    out_path : destination path (should end with _input_raw.npz)
    """
    npz_path = Path(npz_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(npz_path, out_path)
