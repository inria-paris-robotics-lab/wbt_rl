#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &>/dev/null && pwd)/lib.sh"

install_test_pipe_retargeting() {
  _header "test_pipe — tpretargeting"
  _test_pipe_prep_env tpretargeting 3.11
  _test_pipe_run tpretargeting "$TEST_PIPE_SCRIPTS/setup_retargeting.sh"
  _ok "tpretargeting installed"
}

install_test_pipe_retargeting "$@"
