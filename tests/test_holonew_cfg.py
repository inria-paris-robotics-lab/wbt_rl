"""Le retargeter holonew réutilise le connector holosoma_custom_qpos_npz → holosoma_trainer_npz."""
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from motion_convertor._subprocess import load_module_cfg
from motion_convertor.connectors import get_connector
from motion_convertor.formats import validate_format


def test_holonew_cfg_reutilise_connector_existant():
    ret = load_module_cfg("01_retargeting", "holonew")
    trn = load_module_cfg("02_training", "holosoma_custom")
    assert ret["native_output_format"] == "holosoma_custom_qpos_npz"
    validate_format(ret["native_output_format"])
    conn = get_connector(ret["native_output_format"], trn["native_input_format"])
    assert callable(conn)
