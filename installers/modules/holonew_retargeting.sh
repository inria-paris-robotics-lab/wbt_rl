#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &>/dev/null && pwd)/lib.sh"

install_holonew_retargeting() {
  # HoloNew is a copy of holosoma_retargeting; it gets its own isolated env
  # ('holonew' under ~/.holonew_deps) so changes here never touch the shared
  # hsretargeting env. Mirrors installers/modules/holosoma_retargeting.sh.
  _header "HoloNew — holonew"
  _holonew_prep_env holonew 3.11
  _holonew_run holonew "$HOLONEW_SCRIPTS/setup_retargeting.sh"

  # SMPL-H FK for OMOMO/SFU wrappers (src/motion_convertor/wrappers/*_to_joints.py).
  # --no-deps: body_model.py only needs numpy+torch (already installed by setup_retargeting).
  # --ignore-requires-python via real pip: the env's `pip` is shimmed to uv, which lacks
  # that flag, so call python -m pip directly (same pattern as holosoma_retargeting.sh).
  local hn_root="$HOLONEW_CONDA_ROOT/envs/holonew"
  "$hn_root/bin/python" -m pip install --no-deps --ignore-requires-python \
    "$REPO_ROOT/src/motion_convertor/third_party/human_body_prior"

  _ok "holonew installed"
}

install_holonew_retargeting "$@"
