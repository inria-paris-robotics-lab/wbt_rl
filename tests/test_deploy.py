import pytest
import sys
from pathlib import Path
from unittest.mock import patch

_REPO_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(_REPO_ROOT / "scripts"))
sys.path.insert(0, str(_REPO_ROOT / "src"))

import deploy


# ---------------------------------------------------------------------------
# _robot_to_ros2
# ---------------------------------------------------------------------------

def test_robot_to_ros2_g1_27dof():
    assert deploy._robot_to_ros2("g1_27dof") == "g1"


def test_robot_to_ros2_case_insensitive():
    assert deploy._robot_to_ros2("G1_27DOF") == "g1"


def test_robot_to_ros2_unknown_raises():
    with pytest.raises(ValueError, match="Unknown robot"):
        deploy._robot_to_ros2("unknown_robot")


# ---------------------------------------------------------------------------
# _build_preamble
# ---------------------------------------------------------------------------

def test_build_preamble_sim_contains_expected_parts():
    result = deploy._build_preamble(
        env="unitree_control_interface",
        cyclonedds_ws="modules/04_deployment/unitree_ros2/cyclonedds_ws",
        dds_mode="SIMULATION",
    )
    assert "conda activate unitree_control_interface" in result
    assert "setup.bash" in result
    assert "autoset_environment_dds.py SIMULATION" in result
    assert "conda.sh" in result


def test_build_preamble_real_uses_real_dds():
    result = deploy._build_preamble(
        env="unitree_control_interface",
        cyclonedds_ws="modules/04_deployment/unitree_ros2/cyclonedds_ws",
        dds_mode="REAL",
    )
    assert "autoset_environment_dds.py REAL" in result


def test_build_preamble_is_single_line():
    result = deploy._build_preamble("env", "ws", "SIMULATION")
    assert "\n" not in result


# ---------------------------------------------------------------------------
# _load_cfg
# ---------------------------------------------------------------------------

def test_load_cfg_unitree_returns_expected_keys():
    cfg = deploy._load_cfg("unitree")
    assert "env" in cfg
    assert "cyclonedds_ws" in cfg
    assert "entry_points" in cfg
    assert cfg["env"] == "unitree_control_interface"


def test_load_cfg_missing_deployer_raises():
    with pytest.raises(FileNotFoundError):
        deploy._load_cfg("nonexistent_deployer")


# ---------------------------------------------------------------------------
# _build_pane_cmd
# ---------------------------------------------------------------------------

_PREAMBLE = "PREAMBLE"

def test_build_pane_cmd_sim_entry_point():
    ep = {
        "cmd": "ros2 launch unitree_simulation launch_sim.launch.py",
        "args": {"robot": "robot:=", "unlock_base": "unlock_base:="},
        "defaults": {"unlock_base": "False"},
    }
    result = deploy._build_pane_cmd(ep, "g1", _PREAMBLE)
    assert result == "PREAMBLE && ros2 launch unitree_simulation launch_sim.launch.py robot:=g1 unlock_base:=False"


def test_build_pane_cmd_watchdog_entry_point():
    ep = {
        "cmd": "ros2 launch unitree_control_interface watchdog.launch.py",
        "args": {"robot_type": "robot_type:="},
    }
    result = deploy._build_pane_cmd(ep, "g1", _PREAMBLE)
    assert result == "PREAMBLE && ros2 launch unitree_control_interface watchdog.launch.py robot_type:=g1"


def test_build_pane_cmd_bridge_substitutes_repo_root():
    ep = {"cmd": "python {repo_root}/src/ros2_bridge/holosoma_inference_custom.py"}
    result = deploy._build_pane_cmd(ep, "g1", _PREAMBLE)
    assert "{repo_root}" not in result
    assert "holosoma_inference_custom.py" in result
    assert result.startswith("PREAMBLE &&")


def test_build_pane_cmd_no_args_no_suffix():
    ep = {"cmd": "ros2 run unitree_control_interface shutdown_sportsmode.py"}
    result = deploy._build_pane_cmd(ep, "g1", _PREAMBLE)
    assert result == "PREAMBLE && ros2 run unitree_control_interface shutdown_sportsmode.py"


# ---------------------------------------------------------------------------
# _pane_defs
# ---------------------------------------------------------------------------

def _make_cfg():
    return deploy._load_cfg("unitree")


def test_pane_defs_sim_returns_three_panes():
    panes = deploy._pane_defs("SIM", _make_cfg(), "g1")
    assert len(panes) == 3
    assert panes[0]["name"] == "sim"
    assert panes[1]["name"] == "watchdog"
    assert panes[2]["name"] == "bridge"


def test_pane_defs_real_returns_three_panes():
    panes = deploy._pane_defs("REAL", _make_cfg(), "g1")
    assert len(panes) == 3
    assert panes[0]["name"] == "shutdown"
    assert panes[1]["name"] == "watchdog"
    assert panes[2]["name"] == "bridge"


def test_pane_defs_sim_cmds_contain_dds_simulation():
    panes = deploy._pane_defs("SIM", _make_cfg(), "g1")
    for p in panes:
        assert "SIMULATION" in p["cmd"]


def test_pane_defs_real_cmds_contain_dds_real():
    panes = deploy._pane_defs("REAL", _make_cfg(), "g1")
    for p in panes:
        assert "autoset_environment_dds.py REAL" in p["cmd"]


def test_pane_defs_sim_contains_robot_param():
    panes = deploy._pane_defs("SIM", _make_cfg(), "g1")
    sim_cmd = panes[0]["cmd"]
    assert "robot:=g1" in sim_cmd
    watchdog_cmd = panes[1]["cmd"]
    assert "robot_type:=g1" in watchdog_cmd


def test_pane_defs_real_watchdog_contains_robot_type():
    panes = deploy._pane_defs("REAL", _make_cfg(), "g1")
    assert "robot_type:=g1" in panes[1]["cmd"]  # watchdog pane


# ---------------------------------------------------------------------------
# _launch_tmux
# ---------------------------------------------------------------------------

from unittest.mock import MagicMock

_PANES = [
    {"name": "sim",      "cmd": "CMD_SIM"},
    {"name": "watchdog", "cmd": "CMD_WATCHDOG"},
    {"name": "bridge",   "cmd": "CMD_BRIDGE"},
]


def test_launch_tmux_kills_existing_session():
    with patch("deploy.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        deploy._launch_tmux("wbt-deploy-sim", _PANES)
    first_call_args = mock_run.call_args_list[0][0][0]
    assert "kill-session" in first_call_args


def test_launch_tmux_creates_new_session():
    with patch("deploy.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        deploy._launch_tmux("wbt-deploy-sim", _PANES)
    all_args = [str(c) for c in mock_run.call_args_list]
    assert any("new-session" in a for a in all_args)
    assert any("-x" in a and "120" in a for a in all_args)
    assert any("-y" in a and "40" in a for a in all_args)


def test_launch_tmux_sends_all_three_pane_commands():
    with patch("deploy.subprocess.run") as mock_run:
        # mock_run needs to return an object with a .stdout for the split-window calls
        mock_ret = MagicMock()
        mock_ret.stdout = "1\n"
        mock_run.return_value = mock_ret
        deploy._launch_tmux("wbt-deploy-sim", _PANES)
    all_args = [str(c) for c in mock_run.call_args_list]
    assert any("CMD_SIM" in a for a in all_args)
    assert any("CMD_WATCHDOG" in a for a in all_args)
    assert any("CMD_BRIDGE" in a for a in all_args)


def test_launch_tmux_attaches_at_end():
    with patch("deploy.subprocess.run") as mock_run:
        mock_ret = MagicMock()
        mock_ret.stdout = "1\n"
        mock_run.return_value = mock_ret
        deploy._launch_tmux("wbt-deploy-sim", _PANES)
    last_call_args = mock_run.call_args_list[-1][0][0]
    assert "attach-session" in last_call_args


def test_launch_tmux_pane_targets_are_correct():
    with patch("deploy.subprocess.run") as mock_run:
        mock_ret = MagicMock()
        mock_ret.stdout = "1" # simplified for test
        mock_run.return_value = mock_ret
        deploy._launch_tmux("wbt-deploy-sim", _PANES)
    calls = mock_run.call_args_list
    send_keys_calls = [c for c in calls if c[0][0][1] == "send-keys"]
    assert len(send_keys_calls) == 3


# ---------------------------------------------------------------------------
# _build_parser / main argument validation
# ---------------------------------------------------------------------------

def test_parser_mode_required():
    parser = deploy._build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_parser_sim_mode_accepted():
    parser = deploy._build_parser()
    args = parser.parse_args(["--mode", "SIM"])
    assert args.mode == "SIM"
    assert args.robot == "g1_27dof"
    assert args.deployer == "unitree"


def test_parser_real_mode_accepted():
    parser = deploy._build_parser()
    args = parser.parse_args(["--mode", "REAL", "--robot", "g1_27dof"])
    assert args.mode == "REAL"
    assert args.robot == "g1_27dof"


def test_parser_invalid_mode_rejected():
    parser = deploy._build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--mode", "INVALID"])
