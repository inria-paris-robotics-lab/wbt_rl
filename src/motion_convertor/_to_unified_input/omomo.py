"""
OMOMO → unified format.

Delegates to src/motion_convertor/wrappers/omomo_to_joints.py (hsretargeting env, human_body_prior).
Output: unified .npz — global_joint_positions (T,22,3) Z-up metres + height.
"""
import json
import tempfile
from pathlib import Path

from .._subprocess import run_entry_point
from .._config import body_model_path


def convert(seq_data: dict, out_path: Path | str) -> None:
    """
    Parameters
    ----------
    seq_data : one entry from the joblib-loaded OMOMO pickle dict
    out_path : destination unified .npz path
    """
    import joblib

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model_dir = body_model_path("OMOMO")

    # Write a temp single-entry pickle so the wrapper can load it by index 0
    with tempfile.NamedTemporaryFile(suffix=".p", delete=False) as f:
        tmp_pickle = f.name
    joblib.dump({0: seq_data}, tmp_pickle)

    try:
        run_entry_point(
            "motion_convertor", "holosoma_prep", "omomo_to_joints",
            args={
                "pickle":    tmp_pickle,
                "index":     "0",
                "output":    str(out_path),
                "model_dir": str(model_dir),
            },
        )
    finally:
        Path(tmp_pickle).unlink(missing_ok=True)
