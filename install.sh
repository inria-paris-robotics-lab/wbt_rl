#!/usr/bin/env bash
# =============================================================================
# WBT-RL — installer orchestrator
#
# Usage:
#   ./install.sh [all]
#   ./install.sh wbt | interact
#   ./install.sh gmr
#   ./install.sh holosoma_retargeting
#   ./install.sh holosoma_training [mujoco|isaacgym|isaacsim] [--no-warp]
#   ./install.sh holosoma_inference
#   ./install.sh unitree_control_interface
#
# To add a new module: create installers/pipe/<name>.sh or installers/modules/<name>.sh
# and add one line to installers/MODULES. This file never needs to change.
# Extra arguments are forwarded to the target script automatically.
#
# Note: holosoma_* installers use holosoma_custom scripts to create shared conda envs
# (hsretargeting, hsmujoco, hsgym, hssim, hsinference). These envs serve both the
# upstream holosoma and holosoma_custom trainers — no separate install needed.
# =============================================================================
set -euo pipefail

INSTALLERS="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/installers" &>/dev/null && pwd)"

TARGET="${1:-all}"
shift || true

case "$TARGET" in
  all)
    while IFS= read -r mod || [[ -n "$mod" ]]; do
      [[ -z "$mod" || "$mod" == \#* ]] && continue
      bash "$INSTALLERS/$mod.sh"
    done < "$INSTALLERS/MODULES"
    ;;
  *)
    if   [[ -f "$INSTALLERS/pipe/$TARGET.sh"    ]]; then bash "$INSTALLERS/pipe/$TARGET.sh" "$@"
    elif [[ -f "$INSTALLERS/modules/$TARGET.sh" ]]; then bash "$INSTALLERS/modules/$TARGET.sh" "$@"
    else
      echo "Unknown target: $TARGET" >&2
      echo "Usage: $0 [all|wbt|interact|gmr|holosoma_retargeting|holosoma_training|holosoma_inference|unitree_control_interface|test_pipe_retargeting]" >&2
      exit 1
    fi
    ;;
esac

echo ""
echo "  ~/.wbt_deps/      wbt_rl, gmr, interact, unitree_control_interface"
echo "  ~/.holosoma_deps/    holosoma_custom envs (hs*)"
echo "  ~/.test_pipe_deps/   test_pipe envs (tp*)"
echo "  To activate: source scripts/activate_wbt.sh"
