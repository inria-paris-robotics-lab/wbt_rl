#!/usr/bin/env bash
# FoundationPose install - PART 1 (network, no external git code):
# conda env + CUDA toolkit (nvcc) + PyTorch cu124, with network retries.
set -uo pipefail
FP=~/FoundationPose; ENV=foundationpose; CUDA_VER=12.4
cd "$FP"

echo "===== 1) conda env ($ENV) - up to 3 tries (flaky network) ====="
ok=0
for i in 1 2 3; do
  echo "--- try $i ---"
  conda env remove -n "$ENV" -y >/dev/null 2>&1 || true
  if conda env create -f environment.yml; then ok=1; break; fi
  echo "try $i failed, retrying..."
done
[ "$ok" = 1 ] || { echo "FAILED step 1 (env)"; exit 1; }

echo "===== 2) cuda-toolkit $CUDA_VER (nvcc) ====="
conda install -n "$ENV" -y -c conda-forge "cuda-toolkit=$CUDA_VER" || { echo "FAILED step 2 (cuda)"; exit 1; }

echo "===== 3) PyTorch cu124 ====="
conda run -n "$ENV" python -m pip install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu124 || { echo "FAILED step 3 (torch)"; exit 1; }

echo "===== check ====="
conda run -n "$ENV" bash -c 'export CUDA_HOME=$CONDA_PREFIX; export PATH="$CUDA_HOME/bin:$PATH"; echo -n "nvcc: "; which nvcc; python -c "import torch; print(\"torch\", torch.__version__, \"cuda\", torch.version.cuda, \"avail\", torch.cuda.is_available())"'
echo "===== PART1 OK ====="
