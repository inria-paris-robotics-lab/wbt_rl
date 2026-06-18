#!/usr/bin/env python3
"""
deploy.py — launch deployment components for WBT-RL via tmux.

Usage:
    python scripts/deploy.py --mode SIM --robot g1_27dof
    python scripts/deploy.py --mode REAL --robot g1_27dof
"""
import argparse
import socket
import subprocess
import sys
import yaml
from pathlib import Path

_REPO_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from motion_convertor._config import repo_root

_CONDA_ROOT = Path.home() / ".wbt_deps" / "miniconda3"
_ROBOT_SUBNET = "192.168.123"

_ROBOT_MAP = {
    "g1_27dof": "g1",
    "g1_29dof": "g1",
}

_DDS_MODE = {
    "SIM": "SIMULATION",
    "REAL": "REAL",
}


def _find_robot_interface() -> tuple[str | None, str | None]:
    """Return (ifname, ip) of the first interface on the robot subnet, or (None, None)."""
    import fcntl, struct
    for _, ifname in socket.if_nameindex():
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            ip = socket.inet_ntoa(fcntl.ioctl(
                s.fileno(), 0x8915,
                struct.pack("256s", ifname[:15].encode()),
            )[20:24])
            if ip.startswith(_ROBOT_SUBNET):
                return ifname, ip
        except OSError:
            pass
        finally:
            s.close()
    return None, None


def _check_robot_network():
    ifname, ip = _find_robot_interface()
    if ifname is None:
        print(
            f"ERROR: No network interface found on {_ROBOT_SUBNET}.x subnet.\n"
            "       Connect the robot via Ethernet and configure the interface IP\n"
            f"       (e.g. 192.168.123.222) before launching in REAL mode.",
            file=sys.stderr,
        )
        sys.exit(1)
    print(f"Robot interface: {ifname}  ({ip})")


def _robot_to_ros2(robot: str) -> str:
    key = robot.lower()
    if key not in _ROBOT_MAP:
        raise ValueError(f"Unknown robot: {robot!r}. Supported: {list(_ROBOT_MAP)}")
    return _ROBOT_MAP[key]


def _build_preamble(env: str, cyclonedds_ws: str, dds_mode: str) -> str:
    # Requires bash as the active shell in each tmux pane (uses source <(...) substitution).
    ws_path = repo_root() / cyclonedds_ws
    steps = [
        f"source {_CONDA_ROOT}/etc/profile.d/conda.sh",
        f"conda activate {env}",
        f"source {ws_path}/install/setup.bash",
        f"source <(ros2 run unitree_control_interface autoset_environment_dds.py {dds_mode})",
    ]
    return " && ".join(steps)


def _load_cfg(deployer: str) -> dict:
    cfg_path = repo_root() / "cfg" / "04_deployment" / f"{deployer}.yaml"
    return yaml.safe_load(cfg_path.read_text())


def _build_pane_cmd(ep: dict, robot_ros2: str, preamble: str, g1_dof: int) -> str:
    cmd = ep["cmd"].replace("{repo_root}", str(repo_root()))
    args_map = ep.get("args", {})
    defaults = ep.get("defaults", {})
    arg_parts = []
    for wbt_arg, ros2_prefix in args_map.items():
        if wbt_arg in ("robot", "robot_type"):
            arg_parts.append(f"{ros2_prefix}{robot_ros2}")
        elif wbt_arg == "dof":
            arg_parts.append(f"{ros2_prefix}{g1_dof}")
        elif wbt_arg in defaults:
            arg_parts.append(f"{ros2_prefix}{defaults[wbt_arg]}")
    if arg_parts:
        cmd += " " + " ".join(arg_parts)
    return f"{preamble} && {cmd}"


def _pane_defs(mode: str, cfg: dict, robot_ros2: str, g1_dof: int) -> list[dict]:
    env = cfg["env"]
    cws = cfg["cyclonedds_ws"]
    dds = _DDS_MODE[mode]
    preamble = _build_preamble(env, cws, dds)
    eps = cfg["entry_points"]

    if mode == "SIM":
        return [
            {"name": "sim",      "cmd": _build_pane_cmd(eps["sim"],      robot_ros2, preamble, g1_dof)},
            {"name": "watchdog", "cmd": _build_pane_cmd(eps["watchdog"], robot_ros2, preamble, g1_dof)},
            {"name": "bridge",   "cmd": _build_pane_cmd(eps["bridge"],   robot_ros2, preamble, g1_dof)},
        ]
    return [
        {"name": "shutdown", "cmd": _build_pane_cmd(eps["shutdown_sportsmode"], robot_ros2, preamble, g1_dof)},
        {"name": "watchdog", "cmd": _build_pane_cmd(eps["watchdog"],            robot_ros2, preamble, g1_dof)},
        {"name": "bridge",   "cmd": _build_pane_cmd(eps["bridge"],              robot_ros2, preamble, g1_dof)},
    ]


def _launch_tmux(session_name: str, panes: list[dict]) -> None:
    """Create a 3-pane tmux session and attach.

    Layout:
      ┌─────────────┬─────────────┐
      │  pane[0]    │  pane[1]    │
      ├─────────────┴─────────────┤
      │  pane[2]                  │
      └───────────────────────────┘
    """
    subprocess.run(["tmux", "kill-session", "-t", session_name], capture_output=True)
    subprocess.run(["tmux", "new-session", "-d", "-s", session_name, "-x", "120", "-y", "40"], check=True)

    # 1. Create layout first (Indices: 0=top-left, 1=top-right, 2=bottom)
    # Split top/bottom -> creates 0.0 (top) and 0.1 (bottom)
    subprocess.run(["tmux", "split-window", "-t", f"{session_name}:0.0", "-v", "-l", "12"], check=True)
    # Split top-left/top-right -> creates 0.1 (top-right) and shifts bottom to 0.2
    subprocess.run(["tmux", "split-window", "-t", f"{session_name}:0.0", "-h"], check=True)

    # 2. Send commands to stable indices
    # Sim/Shutdown -> top-left (0.0)
    subprocess.run(["tmux", "send-keys", "-t", f"{session_name}:0.0", panes[0]["cmd"], "Enter"], check=True)
    # Watchdog -> top-right (0.1)
    subprocess.run(["tmux", "send-keys", "-t", f"{session_name}:0.1", panes[1]["cmd"], "Enter"], check=True)
    # Bridge -> bottom (0.2)
    subprocess.run(["tmux", "send-keys", "-t", f"{session_name}:0.2", panes[2]["cmd"], "Enter"], check=True)

    # 3. Select bridge pane (bottom) by default
    subprocess.run(["tmux", "select-pane", "-t", f"{session_name}:0.2"], check=True)

    result = subprocess.run(["tmux", "attach-session", "-t", session_name])
    if result.returncode != 0:
        print(f"tmux session '{session_name}' created. Attach with:  tmux attach -t {session_name}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Launch sim-side components for WBT-RL deployment via tmux."
    )
    parser.add_argument("--mode", required=True, choices=["SIM", "REAL"],
                        help="SIM: simulator+watchdog+bridge. REAL: shutdown+watchdog+bridge.")
    parser.add_argument("--robot", default="g1_27dof",
                        help="Robot variant (default: g1_27dof)")
    parser.add_argument("--g1-dof", type=int, choices=[27, 29], default=27, dest="g1_dof",
                        help="Actuated G1 DOF threaded to watchdog (dof:=) and bridge (--dof). "
                             "27 = waist_roll/pitch locked (mode 6); 29 = actuated (mode 5). Default 27.")
    parser.add_argument("--deployer", default="unitree",
                        help="Deployer config — matches cfg/04_deployment/{deployer}.yaml (default: unitree)")
    return parser


def main():
    if not _CONDA_ROOT.exists():
        raise RuntimeError(
            f"Conda root not found at {_CONDA_ROOT}. Run ./install.sh first."
        )
    parser = _build_parser()
    args = parser.parse_args()

    if args.mode == "REAL":
        _check_robot_network()

    robot_ros2 = _robot_to_ros2(args.robot)
    cfg = _load_cfg(args.deployer)
    panes = _pane_defs(args.mode, cfg, robot_ros2, args.g1_dof)
    session_name = f"wbt-deploy-{args.mode.lower()}"

    print(f"Launching {args.mode} deployment (tmux session: {session_name})")
    for p in panes:
        print(f"  [{p['name']}]")
    _launch_tmux(session_name, panes)


if __name__ == "__main__":
    main()
