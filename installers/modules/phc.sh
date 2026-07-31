#!/usr/bin/env bash
# PHC (Perpetual Humanoid Control) — we use ONLY its retargeting front-end:
#   scripts/data_process/fit_smpl_shape.py  then  fit_smpl_motion.py
#
# We deliberately do NOT install requirement.txt. It pins the RL training stack
# (rl-games==1.1.4, opencv-python==4.6.0.66, gym), which is irrelevant here and does
# not build on a modern Python. We install the retargeting subset instead.
#
# chumpy IS required, despite looking like legacy baggage: the SMPL v1.1.0 .pkl body
# models unpickle into chumpy objects. It imports numpy.bool/int/float, removed in
# numpy 1.24 — hence the pin below, scoped to this isolated env only.
#
# ONE-TIME SHAPE FIT (not run by this script): src/motion_convertor/wrappers/phc_smpl.py
# and Paper/adapters/phc_headless.py both require modules/third_party/PHC/data/g1_29dof/
# shape_optimized_v1.pkl to exist before either can run. It is produced once by, from the
# PHC root (modules/third_party/PHC), in the `phc` env, with WBT_ROOT set to the repo root:
#
#   WBT_ROOT=<repo_root> python scripts/data_process/fit_smpl_shape.py \
#       --config-dir <repo_root>/cfg/01_retargeting/phc robot=unitree_g1_29dof_fitting
#
# The output is intentionally not committed (lives inside the PHC submodule, keyed to the
# robot config); re-run the command above whenever data/g1_29dof/shape_optimized_v1.pkl is
# missing or the fitting config changes.
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &>/dev/null && pwd)/lib.sh"

install_phc() {
  _header "PHC env (retargeting front-end only)"
  _ensure_conda "$WBT_CONDA_ROOT" "$HOME/.wbt_deps"
  _create_env   "$WBT_CONDA_ROOT" "phc" "3.10"
  local ENV_ROOT="$WBT_CONDA_ROOT/envs/phc"

  [[ "$(uname -s)" == "Linux" ]] && \
    "$WBT_CONDA_ROOT/bin/mamba" install -y --prefix "$ENV_ROOT" \
      -c conda-forge libstdcxx-ng --override-channels

  # chumpy's setup.py does `from pip._internal.req import parse_requirements` at
  # build time, which needs pip importable in the *build* env. uv's isolated build
  # venvs don't include pip, so the build fails with ModuleNotFoundError: pip unless
  # we route this one package's build through the target env (which has pip).
  _uv_pip "$ENV_ROOT" --no-build-isolation-package chumpy "numpy<1.24" chumpy
  _uv_pip "$ENV_ROOT" torch --index-url https://download.pytorch.org/whl/cu118
  _uv_pip "$ENV_ROOT" scipy joblib tqdm easydict lxml mujoco \
                      "hydra-core>=1.3" omegaconf
  _uv_pip "$ENV_ROOT" "git+https://github.com/ZhengyiLuo/smplx.git@master"
  _uv_pip "$ENV_ROOT" "git+https://github.com/ZhengyiLuo/SMPLSim.git@master"

  # Not pulled in by anything above: phc/utils/torch_humanoid_batch.py (imported by
  # both fit_smpl_shape.py and fit_smpl_motion.py) does `import open3d as o3d` at
  # module load time, so both retargeting entry points fail without it.
  _uv_pip "$ENV_ROOT" open3d

  local SMPL_SRC="$REPO_ROOT/data/00_raw_datasets/models/smpl"
  if [[ ! -f "$SMPL_SRC/SMPL_NEUTRAL.pkl" ]]; then
    echo "ERROR: $SMPL_SRC/SMPL_NEUTRAL.pkl missing." >&2
    echo "       Download SMPL v1.1.0 and rename basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl" >&2
    echo "       to SMPL_NEUTRAL.pkl (same for MALE/FEMALE)." >&2
    exit 1
  fi
  mkdir -p "$PHC_DIR/data"
  ln -sfn "$SMPL_SRC" "$PHC_DIR/data/smpl"
  _ok "data/smpl -> $SMPL_SRC"

  _ok "PHC installed"
}

install_phc "$@"
