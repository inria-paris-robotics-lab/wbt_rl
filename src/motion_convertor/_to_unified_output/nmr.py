"""
NMR retargeter output → unified format.

NMR's bmimic npz already carries body_pos_w (T, 30, 3) in world frame together with
the body names, so unlike GMR this needs no FK subprocess — just a remap.

Note: convert_to_bmimic re-origins the trajectory so the root starts at xy = 0. That is
NMR's own output convention and is preserved here; the trainer path re-bases motions
anyway. The benchmark deliberately bypasses this format for that reason.

NMR G1 body names → SMPL-X 22-joint mapping. 15 have direct equivalents; 7 use the
nearest proxy, following the same convention as the GMR converter.
"""
import numpy as np
from pathlib import Path

from ..unified import save_unified

# SMPL-X joint index → NMR bmimic body name
_SMPLX_TO_NMR = {
    0:  "pelvis",                    # Pelvis
    1:  "left_hip_pitch_link",       # L_Hip
    2:  "right_hip_pitch_link",      # R_Hip
    3:  "waist_yaw_link",            # Spine1
    4:  "left_knee_link",            # L_Knee
    5:  "right_knee_link",           # R_Knee
    6:  "waist_roll_link",           # Spine2
    7:  "left_ankle_pitch_link",     # L_Ankle
    8:  "right_ankle_pitch_link",    # R_Ankle
    9:  "torso_link",                # Spine3
    10: "left_ankle_roll_link",      # L_Foot
    11: "right_ankle_roll_link",     # R_Foot
    12: "torso_link",                # Neck → proxy: Spine3
    13: "left_shoulder_roll_link",   # L_Collar → proxy: L_Shoulder
    14: "right_shoulder_roll_link",  # R_Collar → proxy: R_Shoulder
    15: "torso_link",                # Head → proxy: Spine3
    16: "left_shoulder_roll_link",   # L_Shoulder
    17: "right_shoulder_roll_link",  # R_Shoulder
    18: "left_elbow_link",           # L_Elbow
    19: "right_elbow_link",          # R_Elbow
    20: "left_wrist_yaw_link",       # L_Wrist
    21: "right_wrist_yaw_link",      # R_Wrist
}

# bmimic body order, from NMR's convert_bmimic.G1_BMIMIC_BODY_NAMES
_BMIMIC_BODY_NAMES = [
    "pelvis",               "left_hip_pitch_link",   "right_hip_pitch_link",
    "waist_yaw_link",       "left_hip_roll_link",    "right_hip_roll_link",
    "waist_roll_link",      "left_hip_yaw_link",     "right_hip_yaw_link",
    "torso_link",           "left_knee_link",        "right_knee_link",
    "left_shoulder_pitch_link",  "right_shoulder_pitch_link",
    "left_ankle_pitch_link",     "right_ankle_pitch_link",
    "left_shoulder_roll_link",   "right_shoulder_roll_link",
    "left_ankle_roll_link",      "right_ankle_roll_link",
    "left_shoulder_yaw_link",    "right_shoulder_yaw_link",
    "left_elbow_link",           "right_elbow_link",
    "left_wrist_roll_link",      "right_wrist_roll_link",
    "left_wrist_pitch_link",     "right_wrist_pitch_link",
    "left_wrist_yaw_link",       "right_wrist_yaw_link",
]


def convert(npz_path: Path | str, out_path: Path | str, height: float) -> None:
    """Convert an NMR bmimic .npz to the unified format."""
    data = np.load(Path(npz_path), allow_pickle=True)
    body_pos_w = data["body_pos_w"]                       # (T, 30, 3)
    if body_pos_w.shape[1] != len(_BMIMIC_BODY_NAMES):
        raise ValueError(
            f"Expected {len(_BMIMIC_BODY_NAMES)} bmimic bodies, got {body_pos_w.shape[1]}"
        )
    name_to_idx = {n: i for i, n in enumerate(_BMIMIC_BODY_NAMES)}

    T = body_pos_w.shape[0]
    positions = np.zeros((T, 22, 3), dtype=np.float32)
    for smplx_idx, body in _SMPLX_TO_NMR.items():
        positions[:, smplx_idx, :] = body_pos_w[:, name_to_idx[body], :]

    save_unified(out_path, positions, height)
