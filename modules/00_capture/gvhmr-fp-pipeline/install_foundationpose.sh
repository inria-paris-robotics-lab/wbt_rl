#!/usr/bin/env bash
# Install FoundationPose (local conda, no sudo).
# nvcc comes from conda-forge (cuda-toolkit); CUDA_HOME = env prefix.
set -uo pipefail
FP=~/FoundationPose
ENV=foundationpose
CUDA_VER=12.4          # must match the torch build (cu124)
log(){ echo "===== $* ====="; }

cd "$FP"

log "1) create conda env ($ENV) from environment.yml"
conda env create -f environment.yml || { echo "FAILED step 1"; exit 1; }

log "2a) cuda-toolkit $CUDA_VER (nvcc) via conda-forge"
conda install -n "$ENV" -y -c conda-forge "cuda-toolkit=$CUDA_VER" || { echo "FAILED step 2a"; exit 1; }

log "2b) PyTorch cu124"
conda run -n "$ENV" python -m pip install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu124 || { echo "FAILED step 2b"; exit 1; }

log "3) PyTorch3D + NVDiffRast (build from source)"
conda run -n "$ENV" bash -c '
  export CUDA_HOME=$CONDA_PREFIX
  export PATH="$CUDA_HOME/bin:$PATH"
  echo "nvcc: $(which nvcc)"; nvcc --version | tail -2
  python -m pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git" &&
  python -m pip install --no-build-isolation "git+https://github.com/NVlabs/nvdiffrast.git"
' || { echo "FAILED step 3"; exit 1; }

log "4a) requirements.txt"
conda run -n "$ENV" python -m pip install -r requirements.txt || { echo "FAILED step 4a"; exit 1; }

log "4b) build mycpp extension"
conda run -n "$ENV" bash -c '
  export CUDA_HOME=$CONDA_PREFIX
  export PATH="$CUDA_HOME/bin:$PATH"
  bash build_all_conda.sh
' || { echo "FAILED step 4b"; exit 1; }

log "FOUNDATIONPOSE INSTALL OK"
