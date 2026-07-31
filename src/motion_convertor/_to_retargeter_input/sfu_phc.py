"""
SFU raw SMPL-X npz → PHC input npz.

PHC's load_amass_data() reads entry_data['poses'][:, :66], ['trans'], ['betas'],
['gender'] and tests literally for the key 'mocap_framerate' — no underscore, unlike
GMR's 'mocap_frame_rate'. It returns None (silently skipping the motion) if that exact
key is absent, so the spelling below is load-bearing.

The first 22 SMPL-X body joints coincide with SMPL's, so [root_orient, body_pose]
transfers as-is; the two trailing hand joints are zero.
"""
import numpy as np
from pathlib import Path


def convert(src: Path | str, dst: Path | str) -> None:
    data = np.load(Path(src), allow_pickle=True)
    root = np.asarray(data["root_orient"], np.float64).reshape(-1, 3)
    body = np.asarray(data["pose_body"], np.float64).reshape(root.shape[0], -1)[:, :63]
    poses = np.concatenate([root, body, np.zeros((root.shape[0], 6))], axis=1)

    betas = np.zeros(16)
    b = np.asarray(data["betas"], np.float64).reshape(-1)
    betas[: min(16, b.size)] = b[:16]

    fps = float(data["mocap_frame_rate"]) if "mocap_frame_rate" in data else 30.0

    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    np.savez(dst, poses=poses, trans=np.asarray(data["trans"], np.float64),
             betas=betas, gender=data["gender"],
             mocap_framerate=np.array(fps))     # PHC teste CE nom exact
