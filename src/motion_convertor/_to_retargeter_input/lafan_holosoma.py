"""
LAFAN → holosoma retargeter input.

Delegates to src/motion_convertor/wrappers/lafan_to_joints.py (hsretargeting env, lafan1).
Produces .npy (T, 22, 3) Y-up metres in LAFAN_DEMO_JOINTS order.
holosoma applies Y→Z-up internally.
"""
from pathlib import Path
from .._subprocess import run_entry_point


def convert(bvh_path: Path | str, out_path: Path | str) -> None:
    bvh_path = Path(bvh_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    run_entry_point(
        "motion_convertor", "holosoma_prep", "lafan_to_joints",
        args={"input": str(bvh_path), "output": str(out_path), "format": "yup"},
    )
