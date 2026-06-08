"""
OMOMO → GMR retargeter input.

GMR expects SMPL-X .npz with keys: pose_body (T,63), root_orient (T,3),
trans (T,3), betas (16,), gender, mocap_frame_rate.

The OMOMO raw pickle contains these keys directly (with slight reshaping):
  root_orient (T,3), pose_body (T,63), trans (T,3), betas (1,16), gender.

This module reformats the OMOMO sequence dict as a SMPL-X .npz for GMR.
"""
import numpy as np
from pathlib import Path


def convert(seq_data: dict, out_path: Path | str) -> None:
    """
    Convert a single OMOMO sequence dict to GMR SMPL-X input format.

    Parameters
    ----------
    seq_data : one entry from the joblib-loaded OMOMO pickle dict.
               Keys: root_orient (T,3), pose_body (T,63), trans (T,3),
                     betas (1,16), gender.
    out_path : destination .npz path (should end with _input_raw.npz)
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    root_orient = seq_data["root_orient"].astype(np.float32)    # (T, 3)
    pose_body = seq_data["pose_body"].astype(np.float32)         # (T, 63)
    trans = seq_data["trans"].astype(np.float32)                 # (T, 3)
    betas = seq_data["betas"][0].astype(np.float32)              # (1,16) → (16,)
    gender = str(seq_data["gender"])
    T = root_orient.shape[0]

    np.savez(
        out_path,
        root_orient=root_orient,
        pose_body=pose_body,
        trans=trans,
        betas=betas,
        gender=gender,
        mocap_frame_rate=np.float32(30.0),
        # Optional keys GMR accepts; zero-fill if absent
        pose_hand=np.zeros((T, 90), dtype=np.float32),
        pose_jaw=np.zeros((T, 3), dtype=np.float32),
        pose_eye=np.zeros((T, 6), dtype=np.float32),
    )
