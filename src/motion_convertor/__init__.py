"""
motion_convertor — adapter layer between all pipeline stages.

Public API — 4 flat dispatch functions:

    to_retargeter_input(dataset, retargeter, raw_path, out_path, **kwargs)
    to_unified_input(dataset, raw_path, out_path, **kwargs)
    to_unified_output(retargeter, output_raw_path, out_path, height, **kwargs)
    to_trainer_input(retargeter, trainer, output_raw_path, out_path, **kwargs)

Called by scripts/retarget.py and scripts/train.py — never standalone.
"""
from pathlib import Path

import yaml

from .formats import validate_format
from .connectors import get_connector
from ._subprocess import load_module_cfg


def _dataset_raw_format(dataset: str) -> str:
    """Read raw_format from cfg/00_datasets/<dataset>.yaml."""
    from ._config import repo_root
    cfg_path = repo_root() / "cfg" / "00_datasets" / f"{dataset.lower()}.yaml"
    cfg = yaml.safe_load(cfg_path.read_text())
    fmt = cfg["raw_format"]
    validate_format(fmt)
    return fmt


def to_retargeter_input(
    dataset: str,
    retargeter: str,
    raw_path: Path | str,
    out_path: Path | str,
    **kwargs,
) -> None:
    dataset = dataset.upper()
    retargeter = retargeter.lower()
    raw_path = Path(raw_path)
    out_path = Path(out_path)

    cfg = load_module_cfg("01_retargeting", retargeter)
    src_fmt = _dataset_raw_format(dataset)
    dst_fmt = cfg["native_input_format"][dataset]
    validate_format(dst_fmt)

    connector = get_connector(src_fmt, dst_fmt)
    connector(raw_path, out_path, **kwargs)


def to_unified_input(
    dataset: str,
    raw_path: Path | str,
    out_path: Path | str,
    **kwargs,
) -> None:
    dataset = dataset.upper()
    raw_path = Path(raw_path)
    out_path = Path(out_path)

    src_fmt = _dataset_raw_format(dataset)
    connector = get_connector(src_fmt, "unified_npz")
    connector(raw_path, out_path, **kwargs)


def to_unified_output(
    retargeter: str,
    output_raw_path: Path | str,
    out_path: Path | str,
    height: float,
    **kwargs,
) -> None:
    retargeter = retargeter.lower()
    output_raw_path = Path(output_raw_path)
    out_path = Path(out_path)

    cfg = load_module_cfg("01_retargeting", retargeter)
    src_fmt = cfg["native_output_format"]
    validate_format(src_fmt)

    connector = get_connector(src_fmt, "unified_npz")
    connector(output_raw_path, out_path, height=height, **kwargs)


def to_trainer_input(
    retargeter: str,
    trainer: str,
    output_raw_path: Path | str,
    out_path: Path | str,
    **kwargs,
) -> None:
    retargeter = retargeter.lower()
    trainer = trainer.lower()
    output_raw_path = Path(output_raw_path)
    out_path = Path(out_path)

    ret_cfg = load_module_cfg("01_retargeting", retargeter)
    trn_cfg = load_module_cfg("02_training", trainer)
    src_fmt = ret_cfg["native_output_format"]
    dst_fmt = trn_cfg["native_input_format"]
    validate_format(src_fmt)
    validate_format(dst_fmt)

    connector = get_connector(src_fmt, dst_fmt)
    connector(output_raw_path, out_path, **kwargs)
