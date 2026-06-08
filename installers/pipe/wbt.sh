#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &>/dev/null && pwd)/lib.sh"

install_wbt() {
  _header "wbt_rl env"
  _ensure_conda "$WBT_CONDA_ROOT" "$HOME/.wbt_deps"
  _create_env   "$WBT_CONDA_ROOT" "wbt_rl"
  _uv_pip "$WBT_CONDA_ROOT/envs/wbt_rl" -e "$REPO_ROOT"
  _ok "wbt_rl installed"
}

install_wbt "$@"
