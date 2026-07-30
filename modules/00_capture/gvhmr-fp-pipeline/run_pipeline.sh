#!/usr/bin/env bash
# video -> raw SMPL (body) + 6DoF pose (object) -> clip.npz
# GVHMR (body) + FoundationPose (object, known CAD) + fuse.py.
# GVHMR and FoundationPose live in separate conda envs (conflicting deps);
# each step runs in its own env via `conda run`. Edit the paths below.
set -euo pipefail

VIDEO="${1:?usage: run_pipeline.sh <video.mp4> <object.obj>}"
OBJECT_MESH="${2:?usage: run_pipeline.sh <video.mp4> <object.obj>}"   # CAD in METERS
FPS=30
COORD=camera                             # camera (robust) | world (gravity-aligned)

GVHMR_DIR="$HOME/GVHMR"
FP_DIR="$HOME/FoundationPose"
GLUE_DIR="$(cd "$(dirname "$0")" && pwd)"

GVHMR_ENV=gvhmr
FP_ENV=foundationpose
SMPL_MODEL_DIR="$GVHMR_DIR/inputs/checkpoints/body_models"   # needed if COORD=world

WORK="$GLUE_DIR/work/$(basename "${VIDEO%.*}")"
mkdir -p "$WORK"

echo "==> [1/4] GVHMR: body -> SMPL"
# -s = static camera; drop it if the camera moves (enables visual odometry)
conda run -n "$GVHMR_ENV" python "$GVHMR_DIR/tools/demo/demo.py" \
    --video="$VIDEO" -s
GVHMR_PT="$GVHMR_DIR/outputs/demo/$(basename "${VIDEO%.*}")/hmr4d_results.pt"
echo "    -> $GVHMR_PT"

echo "==> [2/4] Prepare FoundationPose inputs (frames, GVHMR K, folders)"
FP_DATA="$WORK/fp_data"
conda run -n "$FP_ENV" python "$GLUE_DIR/prepare_fp_inputs.py" \
    --video "$VIDEO" --object-mesh "$OBJECT_MESH" --out "$FP_DATA" \
    --gvhmr-pt "$GVHMR_PT"
# manual steps here (see README) if not done yet:
#   a) first-frame object mask -> $FP_DATA/masks/000000.png  (SAM 2)
#   b) metric depth            -> $FP_DATA/depth/*.png        (RGB-D or Depth Anything V2)
echo "    Fill $FP_DATA/masks/ and $FP_DATA/depth/ before continuing."
read -rp "    Press Enter once masks/ and depth/ are ready... " _

echo "==> [3/4] FoundationPose: object -> 6DoF pose (object->camera)"
conda run -n "$FP_ENV" python "$FP_DIR/run_demo.py" \
    --mesh_file "$FP_DATA/mesh/$(basename "$OBJECT_MESH")" \
    --test_scene_dir "$FP_DATA" \
    --debug_dir "$WORK/fp_out" --debug 1
# poses (object->camera) land in <debug_dir>/ob_in_cam/*.txt
FP_POSES="$WORK/fp_out/ob_in_cam"
echo "    -> $FP_POSES"

echo "==> [4/4] Fuse -> clip.npz (raw SMPL + object pose)"
EXTRA=""
if [ "$COORD" = "world" ]; then EXTRA="--smpl-model-dir $SMPL_MODEL_DIR"; fi
conda run -n "$GVHMR_ENV" python "$GLUE_DIR/fuse.py" \
    --gvhmr-pt "$GVHMR_PT" --fp-poses "$FP_POSES" \
    --out "$WORK/clip.npz" --fps "$FPS" --coord "$COORD" \
    --object-mesh "$OBJECT_MESH" $EXTRA

echo "==> done: $WORK/clip.npz"
