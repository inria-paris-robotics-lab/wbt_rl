"""
SFU → holosoma retargeter input.

Delegates to src/motion_convertor/wrappers/sfu_to_joints.py (hsretargeting env, human_body_prior).
Produces unified .npz — global_joint_positions (T,22,3) Z-up metres + height.
"""
from pathlib import Path
from .._subprocess import run_entry_point
from .._config import body_model_path


def convert(npz_path: Path | str, out_path: Path | str) -> None:
    npz_path = Path(npz_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model_root = body_model_path("SFU")

    run_entry_point(
        "motion_convertor", "holosoma_prep", "sfu_to_joints",
        args={
            "input":      str(npz_path),
            "output":     str(out_path),
            "model_root": str(model_root),
        },
    )
