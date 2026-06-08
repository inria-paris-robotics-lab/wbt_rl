"""
LAFAN → unified format.

Delegates to src/motion_convertor/wrappers/lafan_to_joints.py (hsretargeting env, lafan1).
Output: unified .npz — global_joint_positions (T,22,3) Z-up metres + height.
"""
from pathlib import Path
from .._subprocess import run_entry_point


def convert(bvh_path: Path | str, out_path: Path | str) -> None:
    bvh_path = Path(bvh_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    run_entry_point(
        "motion_convertor", "holosoma_prep", "lafan_to_joints",
        args={"input": str(bvh_path), "output": str(out_path), "format": "unified"},
    )
