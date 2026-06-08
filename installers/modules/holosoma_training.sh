#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &>/dev/null && pwd)/lib.sh"

install_holosoma_training() {
  # Uses holosoma_custom scripts — hsmujoco/hsgym/hssim envs are shared by holosoma and holosoma_custom
  _header "holosoma_custom — training envs"
  local modules=() no_warp=""
  for arg in "$@"; do
    [[ "$arg" == "--no-warp" ]] && no_warp="--no-warp" || modules+=("$arg")
  done
  [[ ${#modules[@]} -eq 0 ]] && modules=(mujoco isaacgym isaacsim)

  for mod in "${modules[@]}"; do
    local script="$HOLOSOMA_SCRIPTS/setup_${mod}.sh"
    [[ -f "$script" ]] || { echo "Warning: setup_${mod}.sh not found — skipping" >&2; continue; }
    _header "holosoma_custom — $mod"
    case "$mod" in
      mujoco)   _holosoma_prep_env hsmujoco 3.10
                _holosoma_run           hsmujoco "$script" ${no_warp:+$no_warp} ;;
      isaacgym) _holosoma_prep_env hsgym 3.8
                _holosoma_run           hsgym    "$script" ;;
      isaacsim) _holosoma_prep_env hssim 3.11
                _holosoma_run --sudo OMNI_KIT_ACCEPT_EULA=1 hssim "$script" ;;
      *)        _holosoma_run           hsmujoco "$script" ;;
    esac
  done
}

install_holosoma_training "$@"
