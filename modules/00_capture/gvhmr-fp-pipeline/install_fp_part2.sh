#!/usr/bin/env bash
# FoundationPose install - PART 2 (builds external git code: pytorch3d, nvdiffrast).
# Run this yourself (sandbox needs your go-ahead for third-party code).
# Requires PART1 done (foundationpose env + torch + nvcc OK).
set -uo pipefail
FP=~/FoundationPose; ENV=foundationpose
cd "$FP"

echo "===== 1) PyTorch3D + NVDiffRast (build, ~15-30 min) ====="
conda run -n "$ENV" bash -c '
  export CUDA_HOME=$CONDA_PREFIX
  export PATH="$CUDA_HOME/bin:$PATH"
  echo "nvcc: $(which nvcc)"; nvcc --version | tail -2
  python -m pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git" &&
  python -m pip install --no-build-isolation "git+https://github.com/NVlabs/nvdiffrast.git"
' || { echo "FAILED step 1 (pytorch3d/nvdiffrast)"; exit 1; }

echo "===== 2) requirements.txt ====="
conda run -n "$ENV" python -m pip install -r requirements.txt || { echo "FAILED step 2 (requirements)"; exit 1; }

echo "===== 3) build mycpp extension ====="
conda run -n "$ENV" bash -c '
  export CUDA_HOME=$CONDA_PREFIX
  export PATH="$CUDA_HOME/bin:$PATH"
  bash build_all_conda.sh
' || { echo "FAILED step 3 (mycpp)"; exit 1; }

echo "===== PART2 OK - FoundationPose installed ====="
