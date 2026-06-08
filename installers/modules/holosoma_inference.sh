#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &>/dev/null && pwd)/lib.sh"

install_holosoma_inference() {
  # Uses holosoma_custom scripts — hsinference env is shared by holosoma and holosoma_custom
  _header "holosoma_custom — hsinference"
  _holosoma_prep_env hsinference 3.11 swig
  _holosoma_run --sudo hsinference "$HOLOSOMA_SCRIPTS/setup_inference_py311.sh"
  local ENV_ROOT="$HOLOSOMA_CONDA_ROOT/envs/hsinference"
  # Install without [unitree]: unitree_sdk2 wheel is cp310-only; we use the ros2 interface.
  _uv_pip "$ENV_ROOT" \
    -e "$REPO_ROOT/modules/third_party/holosoma_custom/src/holosoma_inference"
  [[ "$(uname -m)" == "aarch64" ]] && _uv_pip "$ENV_ROOT" "pin>=3.8.0"
  _ok "hsinference installed"
}

install_holosoma_inference "$@"
