#!/usr/bin/env bash
# NMR (MakeTrackingEasy) — neural SMPL-X -> G1 retargeter.
# Feed-forward net: the checkpoint and the SMPL-X body model are pulled at install
# time on purpose. NMR downloads them lazily on first use, which would otherwise land
# inside the first benchmark run and corrupt its runtime measurement.
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &>/dev/null && pwd)/lib.sh"

install_nmr() {
  _header "NMR env"
  _ensure_conda "$WBT_CONDA_ROOT" "$HOME/.wbt_deps"
  _create_env   "$WBT_CONDA_ROOT" "nmr" "3.10"
  local ENV_ROOT="$WBT_CONDA_ROOT/envs/nmr"

  [[ "$(uname -s)" == "Linux" ]] && \
    "$WBT_CONDA_ROOT/bin/mamba" install -y --prefix "$ENV_ROOT" \
      -c conda-forge libstdcxx-ng --override-channels

  _uv_pip "$ENV_ROOT" torch --index-url https://download.pytorch.org/whl/cu118
  _uv_pip "$ENV_ROOT" -r "$NMR_DIR/requirements.txt"

  _header "NMR weights (≈620 MB, one-off)"
  "$ENV_ROOT/bin/python" - <<PY
from huggingface_hub import hf_hub_download
for f in ("weights/epoch_30.pth", "assets/SMPLX_NEUTRAL.npz"):
    p = hf_hub_download(repo_id="RayZhao/NMR", filename=f, local_dir="$NMR_DIR")
    print("  ✓", p)
PY

  _ok "NMR installed"
}

install_nmr "$@"
