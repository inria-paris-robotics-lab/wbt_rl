"""
SFU (AMASS SMPL-X) → unified format.

Delegates to src/motion_convertor/wrappers/sfu_to_joints.py running in the hsretargeting conda env,
which uses holosoma's prep_amass_smplx_for_rt.py logic (human_body_prior.BodyModel).

Output: unified .npz at 30 Hz, global_joint_positions (T,22,3) + height (metres).

Requires: hsretargeting conda env with human_body_prior installed.
"""
from pathlib import Path

from .._subprocess import run_entry_point
from .._config import body_model_path


def convert(npz_path: Path | str, out_path: Path | str) -> None:
    """
    Convert a single SFU .npz sequence to unified format.

    Parameters
    ----------
    npz_path : path to the AMASS .npz file
    out_path : path to write the unified .npz output
    """
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
