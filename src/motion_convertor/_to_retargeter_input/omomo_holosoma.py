"""
OMOMO → holosoma retargeter input.

  robot_only         → omomo_to_joints wrapper (hsretargeting env)
  object_interaction → omomo_to_intermimic wrapper (interact env)
"""
import tempfile
from pathlib import Path

from .._subprocess import run_entry_point
from .._config import body_model_path, body_model_smplx_path, dataset_path


def convert_robot_only(seq_data: dict, out_path: Path | str) -> None:
    """
    Convert an OMOMO robot_only sequence to holosoma retargeter input.

    Produces unified .npz — global_joint_positions (T,22,3) Z-up metres + height.

    Parameters
    ----------
    seq_data : one entry from the joblib-loaded OMOMO pickle dict
    out_path : destination .npz path
    """
    import joblib

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model_dir = body_model_path("OMOMO")

    with tempfile.NamedTemporaryFile(suffix=".p", delete=False) as f:
        tmp_pickle = f.name
    joblib.dump({0: seq_data}, tmp_pickle)

    try:
        run_entry_point(
            "motion_convertor", "holosoma_prep", "omomo_to_joints",
            args={
                "pickle":    tmp_pickle,
                "index":     "0",
                "output":    str(out_path),
                "model_dir": str(model_dir),
            },
        )
    finally:
        Path(tmp_pickle).unlink(missing_ok=True)


def convert_object_interaction(
    seq_name: str,
    raw_path: Path | str,
    out_path: Path | str,
) -> None:
    """
    Convert one OMOMO sequence to a holosoma object_interaction .pt tensor.

    Runs the omomo_to_intermimic wrapper (interact env) which executes
    process_omomo + interact2mimic with patched hardcoded paths.

    Parameters
    ----------
    seq_name : sequence name (e.g. "sub3_largebox_003")
    raw_path : path to the OMOMO train pickle (.p file)
    out_path : destination .pt file path
    """
    omomo_data_root = dataset_path("OMOMO")
    smplh_pkl_dir   = body_model_path("OMOMO")
    smplh_npz_root  = body_model_path("OMOMO")
    smplx_path      = body_model_smplx_path("OMOMO")
    object_path     = omomo_data_root / "captured_objects"

    run_entry_point(
        "motion_convertor", "interact", "omomo_to_pt",
        args={
            "seq_name":       seq_name,
            "pickle_file":    str(raw_path),
            "smplh_pkl_dir":  str(smplh_pkl_dir),
            "smplh_npz_root": str(smplh_npz_root),
            "smplx_path":     str(smplx_path),
            "object_path":    str(object_path),
            "output":         str(out_path),
        },
    )
