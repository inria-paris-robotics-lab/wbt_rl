import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from motion_convertor.connectors import get_connector, ConnectorNotFoundError

def test_known_pair_returns_callable():
    fn = get_connector("lafan_bvh", "holosoma_npy")
    assert callable(fn)

def test_unknown_pair_raises():
    with pytest.raises(ConnectorNotFoundError, match="No connector"):
        get_connector("lafan_bvh", "policy_onnx")

def test_identity_connector_is_callable():
    fn = get_connector("lafan_bvh", "gmr_bvh")
    assert callable(fn)

def test_all_retargeter_input_pairs_registered():
    pairs = [
        ("lafan_bvh",       "gmr_bvh"),
        ("lafan_bvh",       "holosoma_npy"),
        ("sfu_smplx_npz",   "gmr_smplx_npz"),
        ("sfu_smplx_npz",   "holosoma_smplx_npz"),
        ("omomo_smplh_p",   "gmr_smplx_npz"),
        ("omomo_smplh_p",   "holosoma_smplh_npz"),
        ("omomo_new_pt",    "holosoma_pt"),
    ]
    for src, dst in pairs:
        fn = get_connector(src, dst)
        assert callable(fn), f"Missing connector ({src}, {dst})"

def test_all_unified_output_pairs_registered():
    for src in ["gmr_pkl", "holosoma_qpos_npz", "holosoma_custom_qpos_npz"]:
        fn = get_connector(src, "unified_npz")
        assert callable(fn), f"Missing connector ({src!r}, 'unified_npz')"

def test_all_trainer_input_pairs_registered():
    for src in ["gmr_pkl", "holosoma_qpos_npz", "holosoma_custom_qpos_npz"]:
        fn = get_connector(src, "holosoma_trainer_npz")
        assert callable(fn), f"Missing connector ({src!r}, 'holosoma_trainer_npz')"
