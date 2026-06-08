"""
Unified motion format — save and load helpers.

Schema:
    global_joint_positions : (T, 22, 3) float32  — Z-up, world frame, metres
    height                 : float                — subject height in metres
    object_poses           : (T, 7) float32       — OPTIONAL [qw, qx, qy, qz, x, y, z]

Quaternion convention: wxyz throughout.
22 joints follow SMPL-X body convention (indices 0-21, same as SMPL-H joints24 minus L_Hand/R_Hand).
"""
import numpy as np
from pathlib import Path


def save_unified(
    path: Path | str,
    global_joint_positions: np.ndarray,
    height: float,
    object_poses: np.ndarray | None = None,
) -> None:
    """
    Save a unified motion file.

    Parameters
    ----------
    path : Path or str
        Output .npz path.
    global_joint_positions : (T, 22, 3) float32
        Global joint positions, Z-up, metres.
    height : float
        Subject height in metres.
    object_poses : (T, 7) float32, optional
        Object pose per frame: [qw, qx, qy, qz, x, y, z].
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    arrays = dict(
        global_joint_positions=global_joint_positions.astype(np.float32),
        height=np.float32(height),
    )
    if object_poses is not None:
        arrays["object_poses"] = object_poses.astype(np.float32)

    np.savez(path, **arrays)


def load_unified(path: Path | str) -> dict:
    """
    Load a unified motion file.

    Returns
    -------
    dict with keys:
        global_joint_positions : (T, 22, 3) float32
        height                 : float
        object_poses           : (T, 7) float32 or None
    """
    path = Path(path)
    data = np.load(path)
    return {
        "global_joint_positions": data["global_joint_positions"],
        "height": float(data["height"]),
        "object_poses": data["object_poses"] if "object_poses" in data else None,
    }
