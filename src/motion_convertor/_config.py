"""
Loads cfg/00_datasets/data.yaml and exposes helpers to resolve dataset and body model paths.
All paths are relative to the repo root.
"""
import yaml
from pathlib import Path

_REPO_ROOT = Path(__file__).parents[2]
_cfg = yaml.safe_load((_REPO_ROOT / "cfg" / "00_datasets" / "data.yaml").read_text())


def repo_root() -> Path:
    return _REPO_ROOT


def dataset_path(dataset: str) -> Path:
    return _REPO_ROOT / _cfg["raw_datasets"][dataset]["path"]


def body_model_path(dataset: str) -> Path:
    return _REPO_ROOT / _cfg["raw_datasets"][dataset]["body_model"]


def body_model_smplx_path(dataset: str) -> Path:
    return _REPO_ROOT / _cfg["raw_datasets"][dataset]["body_model_smplx"]


def output_path(stage: str) -> Path:
    return _REPO_ROOT / _cfg["output"][stage]
