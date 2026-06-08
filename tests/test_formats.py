import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from motion_convertor.formats import validate_format, KNOWN_FORMATS

def test_known_format_passes():
    validate_format("unified_npz")

def test_unknown_format_raises():
    with pytest.raises(ValueError, match="Unknown format"):
        validate_format("not_a_real_format")

def test_known_formats_is_nonempty():
    assert len(KNOWN_FORMATS) > 10

def test_all_raw_dataset_formats_present():
    for fmt in ["lafan_bvh", "sfu_smplx_npz", "omomo_smplh_p", "omomo_new_pt"]:
        assert fmt in KNOWN_FORMATS, f"{fmt} missing from KNOWN_FORMATS"

def test_all_unified_formats_present():
    assert "unified_npz" in KNOWN_FORMATS

def test_holosoma_custom_output_format_registered():
    validate_format("holosoma_custom_qpos_npz")
