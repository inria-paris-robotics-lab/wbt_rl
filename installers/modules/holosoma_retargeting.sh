#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &>/dev/null && pwd)/lib.sh"

install_holosoma_retargeting() {
  # Uses holosoma_custom scripts — hsretargeting env is shared by holosoma and holosoma_custom
  _header "holosoma_custom — hsretargeting"
  _holosoma_prep_env hsretargeting 3.11
  _holosoma_run hsretargeting "$HOLOSOMA_SCRIPTS/setup_retargeting.sh"
  _ok "hsretargeting installed"
}

install_holosoma_retargeting "$@"
