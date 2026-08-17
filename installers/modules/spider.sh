#!/usr/bin/env bash
# SPIDER — physics-informed dynamics enrichment (stage 05_dynamics).
#
# Deliberately NOT in installers/MODULES: SPIDER is CC BY-NC 4.0 (non-commercial),
# so a full `./install.sh` must not pull it in silently. Install on demand:
#     ./install.sh spider
#
# Unlike every other module here, SPIDER is not conda-based — upstream ships a
# uv workflow, so this creates a .venv inside the submodule instead of an env
# under ~/.wbt_deps. `venv_run()` in src/motion_convertor/_subprocess.py knows
# how to invoke it.
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &>/dev/null && pwd)/lib.sh"

SPIDER_DIR="$REPO_ROOT/modules/third_party/spider"
SPIDER_PATCH="$REPO_ROOT/installers/patches/spider-humanoid-nobj.patch"

install_spider() {
  _header "SPIDER env (stage 05_dynamics)"

  if [[ ! -f "$SPIDER_DIR/pyproject.toml" ]]; then
    echo "ERROR: $SPIDER_DIR is empty — run: git submodule update --init modules/third_party/spider" >&2
    exit 1
  fi

  # uv is not one of the repo's standing dependencies; fetch it if absent.
  if ! command -v uv &>/dev/null; then
    _header "Bootstrapping uv"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
  fi

  # --- patch -----------------------------------------------------------------
  # Upstream SPIDER only ever drives FIXED-BASE dexterous hands through its
  # contact-guidance path, and models exactly one object. The patch adds a
  # `humanoid_nobj` embodiment: floating-base robot + N objects, each 6 DOF.
  # Applied to the working tree only — the submodule gitlink stays on upstream.
  if git -C "$SPIDER_DIR" apply --reverse --check "$SPIDER_PATCH" 2>/dev/null; then
    _ok "humanoid_nobj patch already applied"
  else
    git -C "$SPIDER_DIR" apply --check "$SPIDER_PATCH" \
      || { echo "ERROR: $SPIDER_PATCH does not apply — did the submodule move?" >&2; exit 1; }
    git -C "$SPIDER_DIR" apply "$SPIDER_PATCH"
    _ok "applied humanoid_nobj patch"
  fi

  # --- deps ------------------------------------------------------------------
  # NOTE: upstream's uv.lock pins every package to Meta's internal CodeArtifact
  # registry, which is unreachable from outside Meta. `uv sync` therefore
  # re-resolves against public PyPI and REWRITES uv.lock — expect the submodule
  # to show that file as modified afterwards. It is expected, not a mistake, and
  # is not part of installers/patches/spider-humanoid-nobj.patch.
  ( cd "$SPIDER_DIR" && uv sync )

  "$SPIDER_DIR/.venv/bin/python" -c "import mujoco, mujoco_warp, warp; print('  mujoco', mujoco.__version__)"
  _ok "SPIDER installed (CC BY-NC 4.0 — non-commercial use only)"
}

install_spider "$@"
