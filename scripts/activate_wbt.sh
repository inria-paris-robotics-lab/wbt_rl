#!/usr/bin/env bash
# =============================================================================
# WBT-RL — ecosystem activation
#
# Source this file (do NOT run it) to activate the wbt ecosystem:
#
#   source scripts/activate_wbt.sh
#
# What it does:
#   - Initializes ~/.wbt_deps/miniconda3 as the active conda
#   - Registers all three ecosystems in envs_dirs so conda env list shows everything:
#       ~/.wbt_deps/miniconda3/envs/         (wbt_rl, gmr)
#       ~/.holosoma_deps/miniconda3/envs/       (upstream: hsretargeting, hsmujoco, hsgym, hssim, hsinference)
#       ~/.holosoma_custom_deps/miniconda3/envs/ (custom fork: same env names)
#   - Activates the wbt_rl conda env
#
# After sourcing, use conda normally:
#   conda activate gmr
#   conda activate hsinference      # upstream
#   conda activate hsinference      # (same name — activate by path if ambiguous)
# =============================================================================

# Guard: must be sourced, not executed
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "ERROR: activate_wbt.sh must be sourced, not executed."
  echo "  Use:  source scripts/activate_wbt.sh"
  exit 1
fi

WORKSPACE_DIR="$HOME/.wbt_deps"
CONDA_ROOT="$WORKSPACE_DIR/miniconda3"

# Initialize wbt conda for this shell session
if [[ ! -f "$CONDA_ROOT/etc/profile.d/conda.sh" ]]; then
  echo "WARNING: wbt miniconda not found at $CONDA_ROOT"
  echo "  Run ./install.sh first."
  return 1
fi

WBT_ENV="$CONDA_ROOT/envs/wbt_rl"

if [[ ! -d "$WBT_ENV" ]]; then
  echo "WARNING: wbt_rl env not found at $WBT_ENV"
  echo "  Run ./install.sh first."
  return 1
fi

# Source wbt miniconda so the conda() function is available for sub-envs
source "$CONDA_ROOT/etc/profile.d/conda.sh"

# Prepend wbt_rl env bin to PATH — takes priority over any active miniforge env
export PATH="$WBT_ENV/bin:$PATH"
export CONDA_PREFIX="$WBT_ENV"
export CONDA_DEFAULT_ENV="wbt_rl"

# Register all three ecosystems in ~/.condarc so `conda env list` shows everything.
# conda config --add is idempotent (deduplicates automatically).
# This is a one-time setup — subsequent sources are no-ops.
for envs_dir in \
    "$HOME/.holosoma_custom_deps/miniconda3/envs" \
    "$HOME/.holosoma_deps/miniconda3/envs" \
    "$CONDA_ROOT/envs"; do
  if [[ -d "$envs_dir" ]]; then
    "$CONDA_ROOT/bin/conda" config --add envs_dirs "$envs_dir" 2>/dev/null || true
  fi
done

echo "WBT-RL ecosystem active"
echo "  ~/.wbt_deps/              wbt_rl, gmr"
echo "  ~/.holosoma_deps/            hsretargeting, hsmujoco, hsgym, hssim, hsinference (upstream)"
echo "  ~/.holosoma_custom_deps/     hscretargeting, hscmujoco, hscgym, hscsim, hscinference (custom)"
echo ""
echo "  conda env list   → shows all envs across all three ecosystems"
