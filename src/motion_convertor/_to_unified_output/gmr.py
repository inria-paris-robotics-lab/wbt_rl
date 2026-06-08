"""
GMR retargeter output → unified format.

GMR output is a .pkl with:
    root_pos (T,3), root_rot (T,4) xyzw, dof_pos (T,N_dof)

Since GMR doesn't store global body positions, we run FK via a subprocess
calling src/motion_convertor/wrappers/gmr_fk.py in the `gmr` conda environment.
The wrapper writes body_pos_w.npz, which we then load and remap.

GMR G1 body names → SMPL-X 22-joint mapping:
  The G1 mocap XML has 38 bodies. 15 have direct SMPL-X equivalents;
  7 joints use the nearest proxy (all spine/neck/head → pelvis_contour_link,
  L_Collar → left_shoulder_roll_link, R_Collar → right_shoulder_roll_link).
"""
import tempfile
import numpy as np
from pathlib import Path

from ..unified import save_unified
from .._subprocess import conda_run
from .._config import repo_root

# SMPL-X joint index → GMR G1 body name
_SMPLX_TO_GMR = {
    0:  "pelvis_contour_link",       # Pelvis
    1:  "left_hip_pitch_link",       # L_Hip
    2:  "right_hip_pitch_link",      # R_Hip
    3:  "pelvis_contour_link",       # Spine1 → proxy: Pelvis
    4:  "left_knee_link",            # L_Knee
    5:  "right_knee_link",           # R_Knee
    6:  "pelvis_contour_link",       # Spine2 → proxy: Pelvis
    7:  "left_ankle_pitch_link",     # L_Ankle
    8:  "right_ankle_pitch_link",    # R_Ankle
    9:  "pelvis_contour_link",       # Spine3 → proxy: Pelvis
    10: "left_toe_link",             # L_Foot
    11: "right_toe_link",            # R_Foot
    12: "pelvis_contour_link",       # Neck → proxy: Pelvis
    13: "left_shoulder_roll_link",   # L_Collar → proxy: L_Shoulder
    14: "right_shoulder_roll_link",  # R_Collar → proxy: R_Shoulder
    15: "pelvis_contour_link",       # Head → proxy: Pelvis
    16: "left_shoulder_roll_link",   # L_Shoulder
    17: "right_shoulder_roll_link",  # R_Shoulder
    18: "left_elbow_link",           # L_Elbow
    19: "right_elbow_link",          # R_Elbow
    20: "left_rubber_hand",          # L_Wrist
    21: "right_rubber_hand",         # R_Wrist
}

# Default G1 XML path (relative to repo root)
_DEFAULT_XML = "modules/01_retargeting/GMR/assets/unitree_g1/g1_mocap_29dof.xml"


def convert(
    pkl_path: Path | str,
    out_path: Path | str,
    height: float,
    xml_path: Path | str | None = None,
) -> None:
    """
    Convert GMR retargeter output to unified format.

    Runs forward kinematics in the `gmr` conda env via subprocess.

    Parameters
    ----------
    pkl_path : GMR output .pkl
    out_path : destination unified .npz path
    height   : subject height in metres (from input_unified.npz)
    xml_path : robot XML for FK. Defaults to G1 29-DOF.
    """
    pkl_path = Path(pkl_path)
    out_path = Path(out_path)

    if xml_path is None:
        xml_path = repo_root() / _DEFAULT_XML
    xml_path = Path(xml_path)

    # Run FK wrapper in gmr env, save intermediate npz to temp file
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    wrapper = repo_root() / "src" / "motion_convertor" / "wrappers" / "gmr_fk.py"
    cmd = (
        f"python {wrapper} "
        f"--pkl_path {pkl_path} "
        f"--xml_path {xml_path} "
        f"--out_path {tmp_path}"
    )
    conda_run("gmr", cmd, cwd=repo_root())

    # Load FK result
    fk_data = np.load(tmp_path, allow_pickle=True)
    body_pos_w = fk_data["body_pos_w"]    # (T, B, 3)
    body_names = list(fk_data["body_names"])
    tmp_path.unlink(missing_ok=True)

    name_to_idx = {name: i for i, name in enumerate(body_names)}

    T = body_pos_w.shape[0]
    positions = np.zeros((T, 22, 3), dtype=np.float32)

    for smplx_idx, gmr_body in _SMPLX_TO_GMR.items():
        if gmr_body not in name_to_idx:
            raise ValueError(
                f"Body '{gmr_body}' not found in GMR FK output. "
                f"Available: {body_names}"
            )
        b_idx = name_to_idx[gmr_body]
        positions[:, smplx_idx, :] = body_pos_w[:, b_idx, :]

    save_unified(out_path, positions, height)
