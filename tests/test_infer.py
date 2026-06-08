import pytest
import sys
from pathlib import Path
from unittest.mock import patch

_REPO_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(_REPO_ROOT / "scripts"))
sys.path.insert(0, str(_REPO_ROOT / "src"))

import infer


# ---------------------------------------------------------------------------
# resolve_policy_run
# ---------------------------------------------------------------------------

def test_resolve_policy_run_latest(tmp_path):
    run_dir = tmp_path / "LAFAN_G1_29dof" / "GMR_holosoma_custom" / "run_20240101_000000"
    run_dir.mkdir(parents=True)
    latest = tmp_path / "LAFAN_G1_29dof" / "GMR_holosoma_custom" / "latest"
    latest.symlink_to("run_20240101_000000")

    with patch("infer.output_path", return_value=tmp_path):
        result = infer.resolve_policy_run("LAFAN", "G1_29dof", "GMR", "holosoma_custom", "latest")
    assert result == run_dir


def test_resolve_policy_run_explicit_id(tmp_path):
    run_dir = tmp_path / "LAFAN_G1_29dof" / "GMR_holosoma_custom" / "run_abc"
    run_dir.mkdir(parents=True)

    with patch("infer.output_path", return_value=tmp_path):
        result = infer.resolve_policy_run("LAFAN", "G1_29dof", "GMR", "holosoma_custom", "run_abc")
    assert result == run_dir


def test_resolve_policy_run_missing_latest(tmp_path):
    (tmp_path / "LAFAN_G1_29dof" / "GMR_holosoma_custom").mkdir(parents=True)

    with patch("infer.output_path", return_value=tmp_path):
        with pytest.raises(FileNotFoundError, match="No 'latest' symlink"):
            infer.resolve_policy_run("LAFAN", "G1_29dof", "GMR", "holosoma_custom", "latest")


def test_resolve_policy_run_missing_explicit(tmp_path):
    (tmp_path / "LAFAN_G1_29dof" / "GMR_holosoma_custom").mkdir(parents=True)

    with patch("infer.output_path", return_value=tmp_path):
        with pytest.raises(FileNotFoundError, match="Policy run not found"):
            infer.resolve_policy_run("LAFAN", "G1_29dof", "GMR", "holosoma_custom", "run_missing")


# ---------------------------------------------------------------------------
# _build_infer_cmd
# ---------------------------------------------------------------------------

_EP = {
    "cmd": "python modules/03_inference/holosoma_custom/run_policy.py",
    "extra_args": "--robot.sdk-type ros2",
    "args": {
        "config": None,
        "model_path": "--task.model-path",
    },
}


def test_build_infer_cmd_local_path():
    cmd = infer._build_infer_cmd(_EP, "inference:g1-29dof-wbt", "/data/02_policies/run/model.onnx")
    assert cmd == (
        "python modules/03_inference/holosoma_custom/run_policy.py"
        " inference:g1-29dof-wbt"
        " --task.model-path /data/02_policies/run/model.onnx"
        " --robot.sdk-type ros2"
    )


def test_build_infer_cmd_wandb_uri():
    cmd = infer._build_infer_cmd(_EP, "inference:g1-29dof-wbt", "wandb://entity/project/abc/model.onnx")
    assert cmd == (
        "python modules/03_inference/holosoma_custom/run_policy.py"
        " inference:g1-29dof-wbt"
        " --task.model-path wandb://entity/project/abc/model.onnx"
        " --robot.sdk-type ros2"
    )


def test_build_infer_cmd_no_extra_args():
    ep = {
        "cmd": "python run_policy.py",
        "args": {"config": None, "model_path": "--task.model-path"},
    }
    cmd = infer._build_infer_cmd(ep, "inference:g1-29dof-wbt", "/model.onnx")
    assert "--robot.sdk-type" not in cmd


# ---------------------------------------------------------------------------
# _validate_args
# ---------------------------------------------------------------------------

def _make_parser():
    return infer._build_parser()


def test_validate_wandb_and_dataset_rejected():
    parser = _make_parser()
    args = parser.parse_args([
        "--trainer", "holosoma_custom",
        "--config", "inference:g1-29dof-wbt",
        "--wandb-run", "wandb://x/y/z/model.onnx",
        "--dataset", "LAFAN",
    ])
    with pytest.raises(SystemExit):
        infer._validate_args(parser, args)


def test_validate_neither_mode_rejected():
    parser = _make_parser()
    args = parser.parse_args([
        "--trainer", "holosoma_custom",
        "--config", "inference:g1-29dof-wbt",
    ])
    with pytest.raises(SystemExit):
        infer._validate_args(parser, args)


def test_validate_local_mode_incomplete_args_rejected():
    parser = _make_parser()
    args = parser.parse_args([
        "--trainer", "holosoma_custom",
        "--config", "inference:g1-29dof-wbt",
        "--dataset", "LAFAN",
        # missing --robot and --retargeter
    ])
    with pytest.raises(SystemExit):
        infer._validate_args(parser, args)


def test_validate_wandb_mode_accepted():
    parser = _make_parser()
    args = parser.parse_args([
        "--trainer", "holosoma_custom",
        "--config", "inference:g1-29dof-wbt",
        "--wandb-run", "wandb://entity/project/abc/model.onnx",
    ])
    infer._validate_args(parser, args)  # must not raise


def test_validate_local_mode_accepted():
    parser = _make_parser()
    args = parser.parse_args([
        "--dataset", "LAFAN",
        "--robot", "G1_29dof",
        "--retargeter", "GMR",
        "--trainer", "holosoma_custom",
        "--config", "inference:g1-29dof-wbt",
    ])
    infer._validate_args(parser, args)  # must not raise


def test_validate_holosoma_trainer_rejected():
    parser = _make_parser()
    args = parser.parse_args([
        "--trainer", "holosoma",
        "--config", "inference:g1-29dof-wbt",
        "--wandb-run", "wandb://x/y/z/model.onnx",
    ])
    with pytest.raises(SystemExit):
        infer._validate_args(parser, args)
