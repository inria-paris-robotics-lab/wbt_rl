"""
Retargeted motion (+ scene manifest) → SPIDER working directory.

The heavy lifting needs mujoco, which the `wbt_rl` env does not carry, so it
runs in SPIDER's own venv through `wrappers/spider_build_scene.py`.
"""
from pathlib import Path

from .._config import repo_root
from .._subprocess import load_module_cfg, venv_run

_WRAPPER = "src/motion_convertor/wrappers/spider_build_scene.py"


def convert(
    motion_path: Path | str,
    out_dir: Path | str,
    scene_manifest: Path | str,
    robot_assets: Path | str | None = None,
    data_id: str = "0",
    fps: int = 0,
    **kwargs,
) -> None:
    """Build scene.xml + trajectory_kinematic_act.npz for one clip.

    Parameters
    ----------
    motion_path : retargeted motion npz (holosoma trainer form B)
    out_dir     : SPIDER task dir — scene.xml lands here, data under {data_id}/
    scene_manifest : yaml listing the clip's objects (see cfg/05_dynamics/scenes/)
    robot_assets : SPIDER robot asset dir; defaults to the module's unitree_g1
    """
    cfg = load_module_cfg("05_dynamics", "spider")
    root = repo_root()

    if robot_assets is None:
        robot_assets = root / cfg["robot_assets"]

    cmd = (
        f"python {root / _WRAPPER}"
        f" --motion {Path(motion_path)}"
        f" --scene-manifest {Path(scene_manifest)}"
        f" --robot-assets {Path(robot_assets)}"
        f" --out-dir {Path(out_dir)}"
        f" --data-id {data_id}"
        f" --repo-root {root}"
        f" --fps {int(fps)}"
    )
    venv_run(str(root / cfg["venv"]), cmd, cwd=root)
