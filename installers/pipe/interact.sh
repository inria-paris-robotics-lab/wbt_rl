#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &>/dev/null && pwd)/lib.sh"

install_interact() {
  _header "interact env"
  _ensure_conda "$WBT_CONDA_ROOT" "$HOME/.wbt_deps"
  _create_env   "$WBT_CONDA_ROOT" "interact"
  local ENV_ROOT="$WBT_CONDA_ROOT/envs/interact"

  _uv_pip "$ENV_ROOT" torch==2.0.0 --index-url https://download.pytorch.org/whl/cpu
  _uv_pip "$ENV_ROOT" scipy trimesh joblib smplx tqdm numpy==1.23.1 poselib PyYAML \
    mujoco lxml numpy-stl opencv-python-headless
  # --ignore-requires-python: human_body_prior pins an old Python version; uv doesn't support
  # this flag, so fall back to pip for this one package.
  "$ENV_ROOT/bin/python" -m pip install --no-deps --ignore-requires-python \
    "$REPO_ROOT/src/motion_convertor/third_party/human_body_prior"
  # CPU-only pytorch3d prebuilt wheel; optional — no matching wheel for py311 but kept for future
  _uv_pip "$ENV_ROOT" \
    --find-links "https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu117_pyt200/download.html" \
    pytorch3d || true
  # poselib from bundled InterAct submodule (InterMimic dependency)
  local POSELIB="$REPO_ROOT/src/motion_convertor/third_party/InterAct/simulation/poselib"
  [[ -f "$POSELIB/setup.py" ]] && _uv_pip "$ENV_ROOT" --no-deps -e "$POSELIB"

  _ok "interact env installed"
}

install_interact "$@"
