"""
Finished SPIDER run → WBT-RL dynamics sidecar (`{seq}_output_dynamics.npz`).

Runs in SPIDER's venv (needs mujoco to re-solve contacts) via
`wrappers/spider_extract_dynamics.py`.
"""
from pathlib import Path

from .._config import repo_root
from .._subprocess import load_module_cfg, venv_run

_WRAPPER = "src/motion_convertor/wrappers/spider_extract_dynamics.py"


def convert(
    run_dir: Path | str,
    out_path: Path | str,
    data_id: str = "0",
    ref_frames: int = 0,
    smooth_window: int = 21,
    **kwargs,
) -> None:
    """Extract qpos/qvel/tau(+tau_inverse) + per-pair contact forces from a SPIDER run.

    `ref_frames` resamples the sim-rate rollout back onto the retargeted clip's
    frame count; 0 keeps SPIDER's native rate. `smooth_window` is the
    Savitzky-Golay window (samples) used to smooth qacc for `tau_inverse` --
    see spider_extract_dynamics.py's docstring for why it exists.
    """
    cfg = load_module_cfg("05_dynamics", "spider")
    root = repo_root()

    cmd = (
        f"python {root / _WRAPPER}"
        f" --run-dir {Path(run_dir)}"
        f" --data-id {data_id}"
        f" --out {Path(out_path)}"
        f" --ref-frames {int(ref_frames)}"
        f" --smooth-window {int(smooth_window)}"
    )
    venv_run(str(root / cfg["venv"]), cmd, cwd=root)
