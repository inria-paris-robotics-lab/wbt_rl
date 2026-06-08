"""
holosoma / holosoma_custom retargeter output → unified format.

Input : .npz with keys:
    qpos         (T, 7+dof[+7]) float64  — MuJoCo qpos; last 7 cols are object
                                           free joint when task is object_interaction
    human_joints (T, 52, 3)     float32  — SMPL-H joints, Z-up world frame
    fps          scalar
    cost         scalar

Output: unified .npz with global_joint_positions (T, 22, 3) Z-up metres.

SMPL-H joints 0-21 share the SMPL-X body-joint convention, so we take
human_joints[:, :22, :] directly.

Object pose is extracted from qpos when its width matches a known robot config
(root 7 + robot_dof + object 7). Supported robot_dofs: 27, 29.
MuJoCo free-joint qpos layout: [x, y, z, qw, qx, qy, qz].
Unified object_poses: [qw, qx, qy, qz, x, y, z].
"""
import numpy as np
from pathlib import Path

from ..unified import save_unified

_KNOWN_ROBOT_DOFS = [27, 29]


def convert(npz_path: Path | str, out_path: Path | str, height: float) -> None:
    npz_path = Path(npz_path)
    out_path = Path(out_path)

    data = np.load(npz_path, allow_pickle=True)
    human_joints = data["human_joints"]          # (T, 52, 3)
    positions = human_joints[:, :22, :].astype(np.float32)

    object_poses = None
    qpos = data["qpos"]                          # (T, N)
    n = qpos.shape[1]
    for robot_dof in _KNOWN_ROBOT_DOFS:
        if n == 7 + robot_dof + 7:
            obj = 7 + robot_dof
            object_pos  = qpos[:, obj:obj + 3]       # (T, 3)
            object_quat = qpos[:, obj + 3:obj + 7]   # (T, 4) wxyz
            object_poses = np.hstack([object_quat, object_pos]).astype(np.float32)
            break

    save_unified(out_path, positions, height, object_poses)
