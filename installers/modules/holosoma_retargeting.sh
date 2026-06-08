#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &>/dev/null && pwd)/lib.sh"

install_holosoma_retargeting() {
  # Uses holosoma_custom scripts — hsretargeting env is shared by holosoma and holosoma_custom
  _header "holosoma_custom — hsretargeting"
  _holosoma_prep_env hsretargeting 3.11
  _holosoma_run hsretargeting "$HOLOSOMA_SCRIPTS/setup_retargeting.sh"

  # SMPL-H FK for OMOMO/SFU wrappers (src/motion_convertor/wrappers/*_to_joints.py).
  # --no-deps: body_model.py only needs numpy+torch (already installed by setup_retargeting).
  # --ignore-requires-python via real pip: the env's `pip` is shimmed to uv, which lacks
  # that flag, so call python -m pip directly (same pattern as installers/pipe/interact.sh).
  local hsr_root="$HOLOSOMA_CONDA_ROOT/envs/hsretargeting"
  "$hsr_root/bin/python" -m pip install --no-deps --ignore-requires-python \
    "$REPO_ROOT/src/motion_convertor/third_party/human_body_prior"

  _ok "hsretargeting installed"
}

install_holosoma_retargeting "$@"
