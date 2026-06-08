import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import pytest

from motion_convertor._subprocess import _deep_merge, load_module_cfg


@pytest.fixture(autouse=True)
def clear_cfg_cache():
    load_module_cfg.cache_clear()
    yield
    load_module_cfg.cache_clear()


def test_deep_merge_scalar_override():
    base = {"a": 1, "b": 2}
    override = {"b": 99, "c": 3}
    assert _deep_merge(base, override) == {"a": 1, "b": 99, "c": 3}


def test_deep_merge_nested_dict():
    base = {"env": "base_env", "entry_points": {"single": {"cmd": "base.py", "cwd": "base/"}}}
    override = {"entry_points": {"single": {"cwd": "override/"}, "batch": {"cmd": "batch.py"}}}
    result = _deep_merge(base, override)
    assert result["env"] == "base_env"
    assert result["entry_points"]["single"]["cmd"] == "base.py"   # inherited
    assert result["entry_points"]["single"]["cwd"] == "override/" # overridden
    assert result["entry_points"]["batch"]["cmd"] == "batch.py"   # added


def test_deep_merge_list_replaces():
    base = {"items": [1, 2, 3]}
    override = {"items": [4, 5]}
    assert _deep_merge(base, override) == {"items": [4, 5]}


def test_deep_merge_does_not_mutate_base():
    base = {"a": {"x": 1}}
    override = {"a": {"y": 2}}
    _deep_merge(base, override)
    assert "y" not in base["a"]


def test_load_module_cfg_holosoma_custom_inherits_format_args():
    # After holosoma_custom.yaml gains `base: holosoma`, its merged cfg
    # must contain format_args inherited from holosoma.yaml.
    cfg = load_module_cfg("01_retargeting", "holosoma_custom")
    assert "format_args" in cfg
    assert "holosoma_npy" in cfg["format_args"]


def test_load_module_cfg_holosoma_custom_overrides_output_format():
    cfg = load_module_cfg("01_retargeting", "holosoma_custom")
    assert cfg["native_output_format"] == "holosoma_custom_qpos_npz"


def test_load_module_cfg_holosoma_custom_has_g1_27dof():
    # G1_27dof is only in holosoma_custom, not in holosoma base.
    cfg = load_module_cfg("01_retargeting", "holosoma_custom")
    assert "G1_27dof" in cfg["robot_config"]


def test_load_module_cfg_no_base_field_in_result():
    # The `base:` key must be stripped before returning.
    cfg = load_module_cfg("01_retargeting", "holosoma_custom")
    assert "base" not in cfg


def test_load_module_cfg_cached_same_object():
    # lru_cache must return the same dict object on repeated calls — both for
    # plain YAMLs and for merged (base: holosoma_custom) ones.
    cfg1 = load_module_cfg("01_retargeting", "holosoma")
    cfg2 = load_module_cfg("01_retargeting", "holosoma")
    assert cfg1 is cfg2

    cfg3 = load_module_cfg("01_retargeting", "holosoma_custom")
    cfg4 = load_module_cfg("01_retargeting", "holosoma_custom")
    assert cfg3 is cfg4


def test_deep_merge_empty_base():
    assert _deep_merge({}, {"a": 1}) == {"a": 1}


def test_deep_merge_empty_override():
    assert _deep_merge({"a": 1}, {}) == {"a": 1}
