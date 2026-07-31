"""
PHC retargeter output → unified format.

PHC's dump has no body positions, so FK runs in the phc env via wrappers/phc_fk.py,
mirroring how the GMR converter shells out to gmr_fk.py.

The mapping targets our g1_29dof link names (the robot PHC was configured to fit),
using the same nearest-proxy convention as the GMR and NMR converters.
"""
import tempfile
import numpy as np
from pathlib import Path

from ..unified import save_unified
from .._subprocess import conda_run
from .._config import repo_root

_SMPLX_TO_PHC = {
    0:  "pelvis",                    1:  "left_hip_pitch_link",
    2:  "right_hip_pitch_link",      3:  "waist_yaw_link",
    4:  "left_knee_link",            5:  "right_knee_link",
    6:  "waist_roll_link",           7:  "left_ankle_pitch_link",
    8:  "right_ankle_pitch_link",    9:  "torso_link",
    10: "left_ankle_roll_link",      11: "right_ankle_roll_link",
    12: "torso_link",                13: "left_shoulder_roll_link",
    14: "right_shoulder_roll_link",  15: "torso_link",
    16: "left_shoulder_roll_link",   17: "right_shoulder_roll_link",
    18: "left_elbow_link",           19: "right_elbow_link",
    20: "left_wrist_yaw_link",       21: "right_wrist_yaw_link",
}

_ROBOT_CFG = "cfg/01_retargeting/phc/robot/unitree_g1_29dof_fitting.yaml"


def convert(pkl_path: Path | str, out_path: Path | str, height: float) -> None:
    """Convert a PHC fit .pkl to the unified format, running FK in the phc env."""
    pkl_path = Path(pkl_path)

    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    wrapper = repo_root() / "src" / "motion_convertor" / "wrappers" / "phc_fk.py"
    cmd = (f"WBT_ROOT={repo_root()} python {wrapper} "
           f"--pkl_path {pkl_path} --robot_cfg {repo_root() / _ROBOT_CFG} "
           f"--out_path {tmp_path}")
    conda_run("phc", cmd, cwd=repo_root())

    fk = np.load(tmp_path, allow_pickle=True)
    body_pos_w = fk["body_pos_w"]
    body_names = [str(n) for n in fk["body_names"]]
    tmp_path.unlink(missing_ok=True)

    name_to_idx = {n: i for i, n in enumerate(body_names)}
    T = body_pos_w.shape[0]
    positions = np.zeros((T, 22, 3), dtype=np.float32)
    for smplx_idx, link in _SMPLX_TO_PHC.items():
        if link not in name_to_idx:
            raise ValueError(f"Link {link!r} absent from PHC FK output: {body_names}")
        positions[:, smplx_idx, :] = body_pos_w[:, name_to_idx[link], :]

    save_unified(out_path, positions, height)
