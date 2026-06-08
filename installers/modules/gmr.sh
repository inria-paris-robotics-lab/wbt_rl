#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &>/dev/null && pwd)/lib.sh"

install_gmr() {
  _header "GMR env"
  _ensure_conda "$WBT_CONDA_ROOT" "$HOME/.wbt_deps"
  _create_env   "$WBT_CONDA_ROOT" "gmr"
  local ENV_ROOT="$WBT_CONDA_ROOT/envs/gmr"
  [[ "$(uname -s)" == "Linux" ]] && \
    "$WBT_CONDA_ROOT/bin/mamba" install -y --prefix "$ENV_ROOT" \
      -c conda-forge libstdcxx-ng --override-channels
  _uv_pip "$ENV_ROOT" -e "$GMR_DIR"
  _ok "GMR installed"
}

install_gmr "$@"
