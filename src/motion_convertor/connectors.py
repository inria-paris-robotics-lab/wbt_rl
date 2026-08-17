import shutil
from collections.abc import Callable
from pathlib import Path


class ConnectorNotFoundError(KeyError):
    pass


def _identity(src: Path, dst: Path, **_) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def get_connector(src_fmt: str, dst_fmt: str):
    pair = (src_fmt, dst_fmt)
    if pair not in CONNECTORS:
        raise ConnectorNotFoundError(
            f"No connector for ({src_fmt!r}, {dst_fmt!r}). "
            f"Register it in src/motion_convertor/connectors.py"
        )
    return CONNECTORS[pair]


# Lazy imports inside each function so unused converters don't fail on import
# if their conda env is not active.

def _lafan_to_holosoma(src, dst, **kw):
    from ._to_retargeter_input.lafan_holosoma import convert
    convert(src, dst)

def _sfu_to_holosoma(src, dst, **kw):
    from ._to_retargeter_input.sfu_holosoma import convert
    convert(src, dst)

def _omomo_to_smplx_npz(src, dst, **kw):
    from ._to_retargeter_input.omomo_gmr import convert
    convert(kw["seq_data"], dst)

def _omomo_to_holosoma(src, dst, **kw):
    task_type = kw.get("task_type", "robot_only")
    if task_type == "robot_only":
        from ._to_retargeter_input.omomo_holosoma import convert_robot_only
        convert_robot_only(kw["seq_data"], dst)
    else:
        seq_name = kw["seq_data"].get("seq_name", "unknown")
        from ._to_retargeter_input.omomo_holosoma import convert_object_interaction
        convert_object_interaction(seq_name, src, dst)

def _lafan_to_unified(src, dst, **kw):
    from ._to_unified_input.lafan import convert
    convert(src, dst)

def _sfu_to_unified(src, dst, **kw):
    from ._to_unified_input.sfu import convert
    convert(src, dst)

def _omomo_to_unified(src, dst, **kw):
    from ._to_unified_input.omomo import convert
    convert(kw["seq_data"], dst)

def _gmr_to_unified(src, dst, **kw):
    from ._to_unified_output.gmr import convert
    convert(src, dst, kw["height"], xml_path=kw.get("xml_path"))

def _holosoma_to_unified(src, dst, **kw):
    from ._to_unified_output.holosoma import convert
    convert(src, dst, kw["height"])

def _gmr_to_holosoma_trainer(src, dst, **kw):
    from ._to_trainer_input.gmr_holosoma import convert
    convert(src, dst, xml_path=kw.get("xml_path"))

def _holosoma_to_holosoma_trainer(src, dst, **kw):
    from ._to_trainer_input.holosoma_holosoma import convert
    convert(src, dst,
            robot=kw.get("robot", "G1_29dof"),
            input_fps=kw.get("input_fps", 30),
            output_fps=kw.get("output_fps", 50),
            object_name=kw.get("object_name", "ground"))

def _holosoma_custom_to_unified(src, dst, **kw):
    from ._to_unified_output.holosoma import convert
    convert(src, dst, kw["height"])

def _holosoma_trainer_to_spider_scene(src, dst, **kw):
    # `dst` is the SPIDER task DIRECTORY here, not a single file: the stage emits
    # scene.xml + task_info.json + {data_id}/trajectory_kinematic_act.npz.
    from ._to_dynamics_input.spider import convert
    convert(src, dst,
            scene_manifest=kw["scene_manifest"],
            robot_assets=kw.get("robot_assets"),
            data_id=kw.get("data_id", "0"),
            fps=kw.get("fps", 0))

def _spider_run_to_dynamics(src, dst, **kw):
    # `src` is the finished SPIDER task directory.
    from ._to_dynamics_output.spider import convert
    convert(src, dst,
            data_id=kw.get("data_id", "0"),
            ref_frames=kw.get("ref_frames", 0))

def _holosoma_custom_to_holosoma_trainer(src, dst, **kw):
    from ._to_trainer_input.holosoma_custom_holosoma import convert
    convert(src, dst,
            robot=kw.get("robot", "G1_29dof"),
            input_fps=kw.get("input_fps", 30),
            output_fps=kw.get("output_fps", 50),
            object_name=kw.get("object_name", "ground"))

def _nmr_to_unified(src, dst, **kw):
    from ._to_unified_output.nmr import convert
    convert(src, dst, kw["height"])

def _sfu_to_phc(src, dst, **kw):
    from ._to_retargeter_input.sfu_phc import convert
    convert(src, dst)

def _phc_to_unified(src, dst, **kw):
    from ._to_unified_output.phc import convert
    convert(src, dst, kw["height"])


CONNECTORS: dict[tuple[str, str], Callable[..., None]] = {
    # Raw → retargeter native input
    ("lafan_bvh",               "gmr_bvh"):               _identity,
    ("lafan_bvh",               "holosoma_npy"):           _lafan_to_holosoma,
    ("sfu_smplx_npz",           "gmr_smplx_npz"):          _identity,
    ("sfu_smplx_npz",           "holosoma_smplx_npz"):     _sfu_to_holosoma,
    ("omomo_smplh_p",           "gmr_smplx_npz"):          _omomo_to_smplx_npz,
    ("omomo_smplh_p",           "holosoma_smplh_npz"):     _omomo_to_holosoma,
    ("omomo_new_pt",            "holosoma_pt"):             _identity,
    ("sfu_smplx_npz",           "nmr_smplx_npz"):          _identity,
    ("omomo_smplh_p",           "nmr_smplx_npz"):          _omomo_to_smplx_npz,
    ("sfu_smplx_npz",           "phc_smpl_npz"):           _sfu_to_phc,

    # Raw → unified (to_unified_input path)
    ("lafan_bvh",               "unified_npz"):             _lafan_to_unified,
    ("sfu_smplx_npz",           "unified_npz"):             _sfu_to_unified,
    ("omomo_smplh_p",           "unified_npz"):             _omomo_to_unified,

    # Retargeter native output → unified
    ("gmr_pkl",                 "unified_npz"):             _gmr_to_unified,
    ("holosoma_qpos_npz",       "unified_npz"):             _holosoma_to_unified,
    ("holosoma_custom_qpos_npz", "unified_npz"):             _holosoma_custom_to_unified,
    ("nmr_bmimic_npz",          "unified_npz"):             _nmr_to_unified,
    ("phc_pkl",                 "unified_npz"):             _phc_to_unified,

    # Retargeter native output → trainer native input
    ("gmr_pkl",                 "holosoma_trainer_npz"):    _gmr_to_holosoma_trainer,
    ("holosoma_qpos_npz",       "holosoma_trainer_npz"):    _holosoma_to_holosoma_trainer,
    ("holosoma_custom_qpos_npz", "holosoma_trainer_npz"):    _holosoma_custom_to_holosoma_trainer,

    # Trainer input → dynamics enrichment (stage 05), and back out again
    ("holosoma_trainer_npz",    "spider_scene_npz"):        _holosoma_trainer_to_spider_scene,
    ("spider_scene_npz",        "spider_dynamics_npz"):     _spider_run_to_dynamics,
}
