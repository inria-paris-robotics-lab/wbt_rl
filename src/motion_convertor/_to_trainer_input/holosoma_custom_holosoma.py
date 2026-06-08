"""
holosoma_custom retargeter output → holosoma trainer input (form B).

Same logic as holosoma_holosoma.py but runs in the hsretargeting conda env
(Guillaume-Bsst/holosoma_custom ecosystem).
"""
import re
from pathlib import Path

from .._subprocess import conda_run
from .._config import repo_root

_WRAPPER = "src/motion_convertor/wrappers/holosoma_convert.py"
_ENV = "hsretargeting"
_RETARGETER_ROOT = "modules/01_retargeting/holosoma_custom"


def convert(
    output_raw_path: Path | str,
    out_path: Path | str,
    robot: str = "G1_29dof",
    input_fps: int = 30,
    output_fps: int = 50,
    object_name: str = "ground",
    **kwargs,
) -> None:
    output_raw_path = Path(output_raw_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Parse robot name (e.g. G1_27dof -> type=g1, dof=27)
    m = re.match(r"([A-Za-z0-9]+)_(\d+)dof", robot)
    if m:
        robot_type = m.group(1).lower()
        robot_dof = int(m.group(2))
    else:
        robot_type = "g1"
        robot_dof = 29

    wrapper = repo_root() / _WRAPPER
    retargeter_root = repo_root() / _RETARGETER_ROOT

    cmd = (
        f"python {wrapper} "
        f"--input_file {output_raw_path} "
        f"--output_name {out_path} "
        f"--robot_type {robot_type} "
        f"--robot_dof {robot_dof} "
        f"--input_fps {input_fps} "
        f"--output_fps {output_fps} "
        f"--object_name {object_name} "
        f"--retargeter_root {retargeter_root}"
    )
    conda_run(_ENV, cmd, cwd=repo_root())
