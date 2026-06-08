#!/usr/bin/env python3
"""
Wrapper: OMOMO raw .p → InterMimic .pt tensor for holosoma object_interaction.

Orchestrates the full 2-step InterAct pipeline for a single sequence:
  Step 1: process_omomo.py — raw .p → sequences/{seq_name}/human.npz + object.npz
  Step 2: interact2mimic.py — sequences/ → .pt tensor

Both scripts have hardcoded relative paths. This wrapper patches them via exec()
and arranges a temp working directory with the expected symlink layout.

Usage:
    python src/motion_convertor/wrappers/omomo_to_intermimic.py \\
        --seq_name sub3_largebox_003 \\
        --pickle_file /path/to/train_diffusion_manip_seq_joints24.p \\
        --smplh_pkl_dir  /path/to/OMOMO/smplx/smplh      \\
        --smplh_npz_root /path/to/OMOMO/smplh             \\
        --smplx_path     /path/to/SFU/models_smplx_v1_1/models/smplx \\
        --object_path    /path/to/OMOMO/data/captured_objects \\
        --output         /path/to/output/sub3_largebox_003.pt

Runs in interact env.
"""
import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

_INTERACT_ROOT = Path(__file__).parents[3] / "src" / "motion_convertor" / "third_party" / "InterAct"
_SIMULATE_DIR = _INTERACT_ROOT / "simulation"

sys.path.insert(0, str(_INTERACT_ROOT))
sys.path.insert(0, str(_SIMULATE_DIR))
for _subdir in ["phc", "poselib", "uhc"]:
    _p = _SIMULATE_DIR / _subdir
    if _p.exists():
        sys.path.insert(0, str(_p))

import pickle
import numpy as np

import pickle
import numpy as np

# --- PATCH DE COMPATIBILITÉ SÉCURISÉ ---

# 1. On définit un Unpickler qui sait traduire "numpy._core" -> "numpy.core"
class NumPy1_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if 'numpy._core' in module:
            module = module.replace('numpy._core', 'numpy.core')
        return super().find_class(module, name)

# 2. On remplace la fonction pickle.load globale
def patched_load(file, **kwargs):
    return NumPy1_Unpickler(file, **kwargs).load()

pickle.load = patched_load

# 3. Patchs pour Isaac Gym (les alias supprimés dans NumPy 1.24+)
if not hasattr(np, "float"): np.float = float
if not hasattr(np, "int"): np.int = int
if not hasattr(np, "bool"): np.bool = bool
# ---------------------------------------

# isaacgym's gymapi requires GPU/driver libraries — inject a lightweight stub
# that exposes only the pure-PyTorch math functions used by interact2mimic.
def _inject_isaacgym_stub() -> None:
    import types, importlib
    if "isaacgym" in sys.modules:
        return
    # Copy torch_utils.py content without the problematic __init__ import chain
    _ISAACGYM_SRC = Path.home() / ".holosoma_deps" / "isaacgym" / "python" / "isaacgym"
    if not _ISAACGYM_SRC.exists():
        raise RuntimeError(f"isaacgym not found at {_ISAACGYM_SRC}")
    pkg = types.ModuleType("isaacgym")
    pkg.__path__ = []
    pkg.__package__ = "isaacgym"
    sys.modules["isaacgym"] = pkg
    # Load torch_utils directly, bypassing __init__ → gymapi → gymdeps
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "isaacgym.torch_utils",
        str(_ISAACGYM_SRC / "torch_utils.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = "isaacgym"
    sys.modules["isaacgym.torch_utils"] = mod
    spec.loader.exec_module(mod)

_inject_isaacgym_stub()


def _inject_smpl_mujoco_stub() -> None:
    """
    uhc.smpllib.smpl_mujoco imports glfw/cv2 (rendering deps) at module level,
    but interact2mimic only needs the constants it re-exports from smpl_parser.
    Inject a lightweight stub that imports directly from smpl_parser.
    """
    import types, importlib.util
    if "uhc.smpllib.smpl_mujoco" in sys.modules:
        return
    # Ensure parent packages exist with correct __path__ for submodule discovery
    if "uhc" not in sys.modules:
        m = types.ModuleType("uhc")
        m.__path__ = [str(_SIMULATE_DIR / "uhc")]
        m.__package__ = "uhc"
        sys.modules["uhc"] = m
    if "uhc.smpllib" not in sys.modules:
        m = types.ModuleType("uhc.smpllib")
        m.__path__ = [str(_SIMULATE_DIR / "uhc" / "smpllib")]
        m.__package__ = "uhc.smpllib"
        sys.modules["uhc.smpllib"] = m

    _PARSER = _SIMULATE_DIR / "uhc" / "smpllib" / "smpl_parser.py"
    spec = importlib.util.spec_from_file_location("uhc.smpllib.smpl_parser", str(_PARSER))
    parser_mod = importlib.util.module_from_spec(spec)
    parser_mod.__package__ = "uhc.smpllib"
    sys.modules["uhc.smpllib.smpl_parser"] = parser_mod
    spec.loader.exec_module(parser_mod)

    stub = types.ModuleType("uhc.smpllib.smpl_mujoco")
    stub.SMPLH_BONE_ORDER_NAMES  = parser_mod.SMPLH_BONE_ORDER_NAMES
    stub.SMPLH_SEGMENT           = parser_mod.SMPLH_SEGMENT
    stub.smplx_vert_segmentation = parser_mod.smplx_vert_segmentation
    stub.smpl_vert_segmentation  = parser_mod.smpl_vert_segmentation
    sys.modules["uhc.smpllib.smpl_mujoco"] = stub


_inject_smpl_mujoco_stub()


def _inject_geom_stub() -> None:
    """uhc.utils.geom imports vtk (large visualization lib) — stub it out."""
    import types
    if "uhc.utils.geom" in sys.modules:
        return
    if "uhc.utils" not in sys.modules:
        m = types.ModuleType("uhc.utils")
        m.__path__ = [str(_SIMULATE_DIR / "uhc" / "utils")]
        m.__package__ = "uhc.utils"
        sys.modules["uhc.utils"] = m
    stub = types.ModuleType("uhc.utils.geom")
    stub.quadric_mesh_decimation = lambda *a, **kw: None
    stub.center_scale_mesh = lambda *a, **kw: None
    sys.modules["uhc.utils.geom"] = stub


_inject_geom_stub()


def _fix_poselib_import() -> None:
    """
    interact2mimic imports 'from poselib.poselib.skeleton...'
    The pip-installed poselib resolves to the inner poselib/ dir, so
    poselib.poselib is not available. Alias it so the import works.
    """
    import importlib
    import poselib as _poselib_inner
    if "poselib.poselib" not in sys.modules:
        sys.modules["poselib.poselib"] = _poselib_inner
        for sub in ["skeleton", "skeleton.skeleton3d", "core"]:
            try:
                mod = importlib.import_module(f"poselib.{sub}")
                sys.modules[f"poselib.poselib.{sub}"] = mod
            except ImportError:
                pass


_fix_poselib_import()


def _setup_model_dir(
    smplh_pkl_dir: Path,
    smplh_npz_root: Path,
    smplx_path: Path,
    tmp_dir: Path,
) -> Path:
    """
    Build models/ with layout expected by interact2mimic.py:
      models/smplh/SMPLH_{MALE,FEMALE,NEUTRAL}.pkl
      models/smplh/{male,female,neutral}/model.npz
      models/smplx/SMPLX_{MALE,FEMALE,NEUTRAL}.{npz,pkl}
    """
    model_dir = tmp_dir / "models"

    smplh_out = model_dir / "smplh"
    smplh_out.mkdir(parents=True, exist_ok=True)

    # .pkl for smplx.create
    for gender in ["MALE", "FEMALE"]:
        src = smplh_pkl_dir / f"SMPLH_{gender}.pkl"
        dst = smplh_out / f"SMPLH_{gender}.pkl"
        if src.exists() and not dst.exists():
            dst.symlink_to(src.resolve())
    neutral_dst = smplh_out / "SMPLH_NEUTRAL.pkl"
    if not neutral_dst.exists():
        neutral_src = smplh_pkl_dir / "SMPLH_NEUTRAL.pkl"
        if neutral_src.exists():
            neutral_dst.symlink_to(neutral_src.resolve())
        else:
            male_src = smplh_pkl_dir / "SMPLH_MALE.pkl"
            if male_src.exists():
                shutil.copy2(male_src, neutral_dst)

    # model.npz subdirs for BodyModel (16-beta)
    for gender in ["male", "female", "neutral"]:
        src_g = smplh_npz_root / gender
        dst_g = smplh_out / gender
        if not dst_g.exists() and src_g.is_dir():
            dst_g.symlink_to(src_g.resolve())

    # smplx
    smplx_out = model_dir / "smplx"
    smplx_out.mkdir(parents=True, exist_ok=True)
    for f in smplx_path.glob("SMPLX_*.npz"):
        lnk = smplx_out / f.name
        if not lnk.exists():
            lnk.symlink_to(f.resolve())
    for f in smplx_path.glob("SMPLX_*.pkl"):
        lnk = smplx_out / f.name
        if not lnk.exists():
            lnk.symlink_to(f.resolve())

    return model_dir


def _run_process_omomo(
    seq_name: str,
    pickle_file: Path,
    smplh_npz_root: Path,
    smplx_path: Path,
    object_path: Path,
    work_dir: Path,
) -> Path:
    """
    Execute process_omomo.py with patched paths for a single sequence.
    Returns the output sequences/{seq_name} directory.
    """
    import joblib

    omomo_dir = work_dir / "data" / "omomo"
    raw_dir = omomo_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Symlink raw pickle; provide empty test pickle
    train_link = raw_dir / "train_diffusion_manip_seq_joints24.p"
    if not train_link.exists():
        train_link.symlink_to(pickle_file.resolve())
    test_link = raw_dir / "test_diffusion_manip_seq_joints24.p"
    if not test_link.exists():
        joblib.dump({}, test_link)

    # Model dirs (shared with _setup_model_dir — guard against already-existing links)
    smplx_dir = work_dir / "models" / "smplx"
    smplx_dir.mkdir(parents=True, exist_ok=True)
    for f in smplx_path.glob("SMPLX_*.npz"):
        lnk = smplx_dir / f.name
        if not lnk.exists():
            lnk.symlink_to(f.resolve())

    smplh_dir = work_dir / "models" / "smplh"
    smplh_dir.mkdir(parents=True, exist_ok=True)
    for gender in ["male", "female", "neutral"]:
        src_g = smplh_npz_root / gender
        dst_g = smplh_dir / gender
        if not dst_g.exists() and src_g.is_dir():
            dst_g.symlink_to(src_g.resolve())

    sequences_path = omomo_dir / "sequences"
    objects_path = omomo_dir / "objects"

    script = (_INTERACT_ROOT / "process" / "process_omomo.py").read_text()
    replacements = {
        "MOTION_PATH = './data/omomo/sequences'":              f"MOTION_PATH = {str(sequences_path)!r}",
        "OBJECT_PATH = './data/omomo/objects'":                f"OBJECT_PATH = {str(objects_path)!r}",
        "MOTION_PATH_RAW = './data/omomo/raw/train_diffusion_manip_seq_joints24.p'": f"MOTION_PATH_RAW = {str(train_link)!r}",
        "MOTION_PATH_RAW_TEST = './data/omomo/raw/test_diffusion_manip_seq_joints24.p'": f"MOTION_PATH_RAW_TEST = {str(test_link)!r}",
        "OBJECT_PATH_RAW = './data/omomo/raw/captured_objects'": f"OBJECT_PATH_RAW = {str(object_path)!r}",
        "SMPLX_PATH = './models/smplx'":                       f"SMPLX_PATH = {str(smplx_dir)!r}",
        "SMPLH_PATH = './models/smplh'":                       f"SMPLH_PATH = {str(smplh_dir)!r}",
    }
    for old, new in replacements.items():
        script = script.replace(old, new)

    # Filter to only the requested sequence
    filter_code = f"""
_target = {seq_name!r}
_filtered = {{k: v for k, v in data_dict.items() if v.get('seq_name', str(k)) == _target}}
if not _filtered:
    raise RuntimeError(f"Sequence {{_target!r}} not found in pickle")
data_dict = _filtered
"""
    script = script.replace(
        "data_dict.update(data_dict_test)\n",
        "data_dict.update(data_dict_test)\n" + filter_code,
    )

    globs = {
        "__name__": "__main__",
        "__file__": str(_INTERACT_ROOT / "process" / "process_omomo.py"),
        "__builtins__": __builtins__,
    }
    exec(compile(script, "process_omomo.py", "exec"), globs)

    out = sequences_path / seq_name
    if not out.exists():
        raise RuntimeError(f"process_omomo produced no output at {out}")
    return out


def _run_canonicalize_human(
    seq_name: str,
    sequences_dir: Path,
    sequences_canonical_dir: Path,
    model_dir: Path,
) -> None:
    """
    Run canonicalize_human.py for a single OMOMO sequence via exec() with patched paths.

    Reads sequences_dir/{seq_name}/human.npz + object.npz (process_omomo output).
    Writes canonicalized sequences_canonical/{seq_name}/human.npz + object.npz.

    Directory layout expected (derived from sequences_dir):
      sequences_dir        = omomo_dir / "sequences"
      omomo_dir / "objects" — scaled .obj meshes written by process_omomo
      model_dir             — SMPLH pkl + npz, SMPLX npz + pkl
    """
    omomo_dir = sequences_dir.parent     # tmp_dir/data/omomo
    data_dir  = omomo_dir.parent         # tmp_dir/data
    tmp_dir   = data_dir.parent          # tmp_dir

    # canonicalize_human.py reads from sequences_seg/ — symlink our output there
    sequences_seg_dir = omomo_dir / "sequences_seg"
    sequences_seg_dir.mkdir(parents=True, exist_ok=True)
    seq_link = sequences_seg_dir / seq_name
    if not seq_link.exists():
        seq_link.symlink_to((sequences_dir / seq_name).resolve())

    sequences_canonical_dir.mkdir(parents=True, exist_ok=True)

    results_dir = tmp_dir / "results"
    results_dir.mkdir(exist_ok=True)

    script = (_INTERACT_ROOT / "process" / "canonicalize_human.py").read_text()
    replacements = {
        "MODEL_PATH = './models'":
            f"MODEL_PATH = {str(model_dir)!r}",
        'results_folder = "./results"':
            f'results_folder = {str(results_dir)!r}',
        "datasets = ['behave', 'intercap', 'omomo', 'grab', 'arctic', 'parahome']":
            "datasets = ['omomo']",
        "data_root = './data'":
            f"data_root = {str(data_dir)!r}",
    }
    for old, new in replacements.items():
        script = script.replace(old, new)

    # Filter inner loop to only our sequence (8-space indent matches the for-dataset block)
    script = script.replace(
        "        data_name = os.listdir(MOTION_PATH)\n",
        f"        data_name = os.listdir(MOTION_PATH)\n"
        f"        data_name = [n for n in data_name if n == {seq_name!r}]\n",
    )

    orig_cwd = os.getcwd()
    os.chdir(str(tmp_dir))
    try:
        globs = {
            "__name__": "__main__",
            "__file__": str(_INTERACT_ROOT / "process" / "canonicalize_human.py"),
            "__builtins__": __builtins__,
        }
        exec(compile(script, "canonicalize_human.py", "exec"), globs)
    finally:
        os.chdir(orig_cwd)

    out = sequences_canonical_dir / seq_name
    if not out.exists():
        raise RuntimeError(f"canonicalize_human produced no output at {out}")


def _run_interact2mimic(
    seq_name: str,
    sequences_canonical_dir: Path,
    model_dir: Path,
    output_pt: Path,
    work_dir: Path,
) -> None:
    """
    Execute interact2mimic.py for a single sequence with patched paths.
    """
    sim_work = work_dir / "simulation"
    sim_work.mkdir(exist_ok=True)

    # interact2mimic reads ../data/omomo/sequences_canonical/ relative to its cwd (sim_work)
    canonical = work_dir / "data" / "omomo" / "sequences_canonical"
    canonical.mkdir(parents=True, exist_ok=True)
    seq_link = canonical / seq_name
    if not seq_link.exists():
        seq_link.symlink_to((sequences_canonical_dir / seq_name).resolve())
    elif seq_link.is_symlink() and seq_link.resolve() != (sequences_canonical_dir / seq_name).resolve():
        seq_link.unlink()
        seq_link.symlink_to((sequences_canonical_dir / seq_name).resolve())

    # Output dirs it creates inside cwd
    (sim_work / "intermimic" / "InterAct" / "omomo").mkdir(parents=True, exist_ok=True)
    (sim_work / "intermimic" / "data" / "assets" / "smplx").mkdir(parents=True, exist_ok=True)
    (sim_work / "intermimic" / "data" / "assets" / "objects").mkdir(parents=True, exist_ok=True)

    # symlink phc/data (read-only assets like XML templates) from _SIMULATE_DIR
    phc_link = sim_work / "phc"
    if not phc_link.exists():
        phc_link.symlink_to((_SIMULATE_DIR / "phc").resolve())

    script = (_SIMULATE_DIR / "interact2mimic.py").read_text()
    script = script.replace(
        'MODEL_PATH = "../models"',
        f'MODEL_PATH = {str(model_dir)!r}',
    )
    # Debug: print key values to diagnose object Z offset
    script = script.replace(
        "        obj_trans_delta = rotation_matrix_x.apply(obj_trans - pelvis.cpu().numpy())\n",
        "        obj_trans_delta = rotation_matrix_x.apply(obj_trans - pelvis.cpu().numpy())\n"
        "        print(f'[DBG] obj_trans[0]={obj_trans[0]}')\n"
        "        print(f'[DBG] pelvis[0]={pelvis.cpu().numpy()[0]}')\n"
        "        print(f'[DBG] root_trans_before_Rx[0]={root_trans[0]}')\n"
        "        print(f'[DBG] obj_minus_pelvis_Y={float(obj_trans[0,1] - pelvis.cpu().numpy()[0,1]):.4f}')\n"
        "        print(f'[DBG] obj_trans_delta[0]={obj_trans_delta[0]}')\n",
    )
    script = script.replace(
        "            obj_new_trans = trans + torch.from_numpy(obj_trans_delta).double()\n",
        "            obj_new_trans = trans + torch.from_numpy(obj_trans_delta).double()\n"
        "            print(f'[DBG] trans_after_floorfix[0]={trans[0].numpy()}')\n"
        "            print(f'[DBG] obj_new_trans[0]={obj_new_trans[0].numpy()}')\n"
        "            print(f'[DBG] relative_Z_frame0={float(obj_new_trans[0,-1] - trans[0,-1]):.4f}')\n",
    )

    orig_cwd = os.getcwd()
    os.chdir(str(sim_work))
    try:
        sys.argv = ["interact2mimic.py", "--dataset_name", "omomo"]
        globs = {
            "__name__": "__main__",
            "__file__": str(_SIMULATE_DIR / "interact2mimic.py"),
            "__builtins__": __builtins__,
        }
        exec(compile(script, "interact2mimic.py", "exec"), globs)
    finally:
        os.chdir(orig_cwd)

    generated = sim_work / "intermimic" / "InterAct" / "omomo" / f"{seq_name}.pt"
    if not generated.exists():
        raise RuntimeError(f"interact2mimic produced no output at {generated}")

    output_pt.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(generated, output_pt)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_name",      required=True, help="Sequence name (e.g. sub3_largebox_003)")
    parser.add_argument("--pickle_file",   required=True, help="OMOMO train .p pickle path")
    parser.add_argument("--smplh_pkl_dir", required=True, help="Dir with SMPLH_MALE.pkl + SMPLH_FEMALE.pkl")
    parser.add_argument("--smplh_npz_root",required=True, help="Dir with male/female/neutral/model.npz subdirs")
    parser.add_argument("--smplx_path",    required=True, help="Dir with SMPLX_*.npz + SMPLX_*.pkl")
    parser.add_argument("--object_path",   required=True, help="Dir with captured_objects (raw mesh .obj files)")
    parser.add_argument("--output",        required=True, help="Output .pt file path")
    args = parser.parse_args()

    pickle_file    = Path(args.pickle_file)
    smplh_pkl_dir  = Path(args.smplh_pkl_dir)
    smplh_npz_root = Path(args.smplh_npz_root)
    smplx_path     = Path(args.smplx_path)
    object_path    = Path(args.object_path)
    output_pt      = Path(args.output)

    with tempfile.TemporaryDirectory(prefix="wbt_intermimic_") as tmp:
        tmp_dir = Path(tmp)

        print(f"[1/4] Setting up model directory...")
        model_dir = _setup_model_dir(smplh_pkl_dir, smplh_npz_root, smplx_path, tmp_dir)

        print(f"[2/4] process_omomo: '{args.seq_name}'...")
        _run_process_omomo(
            args.seq_name, pickle_file, smplh_npz_root,
            smplx_path, object_path, tmp_dir,
        )

        omomo_dir         = tmp_dir / "data" / "omomo"
        sequences_dir     = omomo_dir / "sequences"
        sequences_can_dir = omomo_dir / "sequences_canonical"
        sequences_can_dir.mkdir(parents=True, exist_ok=True)

        print(f"[3/4] canonicalize_human: '{args.seq_name}'...")
        _run_canonicalize_human(
            args.seq_name,
            sequences_dir,
            sequences_can_dir,
            model_dir,
        )

        print(f"[4/4] interact2mimic: '{args.seq_name}'...")
        _run_interact2mimic(args.seq_name, sequences_can_dir, model_dir, output_pt, tmp_dir)

    print(f"Done: {output_pt}")


if __name__ == "__main__":
    main()
