"""
GMR retargeter output → holosoma trainer input (form B).

holosoma trainer form B expects:
    fps          : float (50 Hz)
    joint_pos    : (T, N_dof) float32
    joint_vel    : (T, N_dof) float32
    body_pos_w   : (T, N_bodies, 3) float32
    body_quat_w  : (T, N_bodies, 4) float32 wxyz
    body_lin_vel_w : (T, N_bodies, 3) float32
    body_ang_vel_w : (T, N_bodies, 3) float32
    joint_names  : list of str
    body_names   : list of str

GMR output is 30 Hz; trainer expects 50 Hz → interpolate.
FK runs via gmr_fk.py wrapper subprocess in `gmr` env.
GMR FK returns quaternions in xyzw; convert to wxyz for the output.

Body names in the output match GMR's KinematicsModel body names
(same as to_unified_output/gmr.py).
"""
import tempfile
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation, Slerp

from .._subprocess import conda_run
from .._config import repo_root

_DEFAULT_XML = "modules/01_retargeting/GMR/assets/unitree_g1/g1_mocap_29dof.xml"
_INPUT_FPS = 30.0
_OUTPUT_FPS = 50.0


def _lerp(a: np.ndarray, b: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Linear interpolation along last axis."""
    return a + t * (b - a)


def _interpolate_to_fps(
    arr: np.ndarray,
    src_fps: float,
    dst_fps: float,
    is_quat: bool = False,
) -> np.ndarray:
    """
    Resample array from src_fps to dst_fps via linear (or SLERP for quats).
    arr shape: (T, ...) — interpolated along axis 0.
    """
    T_src = arr.shape[0]
    T_dst = int(round((T_src - 1) * dst_fps / src_fps)) + 1
    src_times = np.linspace(0, 1, T_src)
    dst_times = np.linspace(0, 1, T_dst)

    if not is_quat:
        # Linear interpolation per element
        original_shape = arr.shape
        flat = arr.reshape(T_src, -1)
        out = np.zeros((T_dst, flat.shape[1]), dtype=np.float32)
        for j in range(flat.shape[1]):
            out[:, j] = np.interp(dst_times, src_times, flat[:, j])
        return out.reshape((T_dst,) + original_shape[1:])
    else:
        # SLERP — arr shape (T, ..., 4) xyzw
        leading = arr.shape[1:-1]
        flat = arr.reshape(T_src, -1, 4)
        N = flat.shape[1]
        out = np.zeros((T_dst, N, 4), dtype=np.float32)
        for j in range(N):
            r = Rotation.from_quat(flat[:, j, :])  # xyzw input
            slerp = Slerp(src_times, r)
            out[:, j, :] = slerp(dst_times).as_quat()
        return out.reshape((T_dst,) + leading + (4,))


def convert(
    pkl_path: Path | str,
    out_path: Path | str,
    xml_path: Path | str | None = None,
) -> None:
    """
    Convert GMR retargeter output to holosoma trainer input (form B).

    Parameters
    ----------
    pkl_path : GMR output .pkl
    out_path : destination .npz path
    xml_path : robot XML for FK. Defaults to G1 29-DOF.
    """
    import pickle

    pkl_path = Path(pkl_path)
    out_path = Path(out_path)

    if xml_path is None:
        xml_path = repo_root() / _DEFAULT_XML
    xml_path = Path(xml_path)

    # Load GMR output for joint data
    with open(pkl_path, "rb") as f:
        pkl = pickle.load(f)

    dof_pos = pkl["dof_pos"].astype(np.float32)  # (T, N_dof)

    # Run FK wrapper to get body positions and rotations
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

    fk_data = np.load(tmp_path, allow_pickle=True)
    body_pos_w = fk_data["body_pos_w"]    # (T, B, 3)
    body_rot_w = fk_data["body_rot_w"]    # (T, B, 4) xyzw
    body_names = list(fk_data["body_names"])
    tmp_path.unlink(missing_ok=True)

    T = body_pos_w.shape[0]

    # --- Compute velocities at 30 Hz via finite differences ---
    dt_30 = 1.0 / _INPUT_FPS

    def _finite_diff(arr):
        """Central differences, forward/backward at boundaries."""
        vel = np.zeros_like(arr)
        vel[1:-1] = (arr[2:] - arr[:-2]) / (2 * dt_30)
        vel[0] = (arr[1] - arr[0]) / dt_30
        vel[-1] = (arr[-1] - arr[-2]) / dt_30
        return vel

    joint_vel_30 = _finite_diff(dof_pos)                  # (T, N_dof)
    body_lin_vel_30 = _finite_diff(body_pos_w)             # (T, B, 3)

    # Angular velocity from quaternion finite differences (xyzw → rotvec diff)
    body_ang_vel_30 = np.zeros((T, body_pos_w.shape[1], 3), dtype=np.float32)
    for i in range(1, T - 1):
        for b in range(body_pos_w.shape[1]):
            q1 = Rotation.from_quat(body_rot_w[i - 1, b])
            q2 = Rotation.from_quat(body_rot_w[i + 1, b])
            dq = (q2 * q1.inv())
            body_ang_vel_30[i, b] = dq.as_rotvec() / (2 * dt_30)
    body_ang_vel_30[0] = body_ang_vel_30[1]
    body_ang_vel_30[-1] = body_ang_vel_30[-2]

    # --- Interpolate 30 Hz → 50 Hz ---
    dof_pos_50 = _interpolate_to_fps(dof_pos, _INPUT_FPS, _OUTPUT_FPS)
    joint_vel_50 = _interpolate_to_fps(joint_vel_30, _INPUT_FPS, _OUTPUT_FPS)
    body_pos_50 = _interpolate_to_fps(body_pos_w, _INPUT_FPS, _OUTPUT_FPS)
    body_rot_50_xyzw = _interpolate_to_fps(body_rot_w, _INPUT_FPS, _OUTPUT_FPS, is_quat=True)
    body_lin_vel_50 = _interpolate_to_fps(body_lin_vel_30, _INPUT_FPS, _OUTPUT_FPS)
    body_ang_vel_50 = _interpolate_to_fps(body_ang_vel_30, _INPUT_FPS, _OUTPUT_FPS)

    # Convert quaternions xyzw → wxyz for holosoma
    body_quat_50_wxyz = body_rot_50_xyzw[:, :, [3, 0, 1, 2]].astype(np.float32)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        fps=np.float32(_OUTPUT_FPS),
        joint_pos=dof_pos_50.astype(np.float32),
        joint_vel=joint_vel_50.astype(np.float32),
        body_pos_w=body_pos_50.astype(np.float32),
        body_quat_w=body_quat_50_wxyz,
        body_lin_vel_w=body_lin_vel_50.astype(np.float32),
        body_ang_vel_w=body_ang_vel_50.astype(np.float32),
        joint_names=np.array([]),   # GMR doesn't store joint names in pkl
        body_names=np.array(body_names),
    )
