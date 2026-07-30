# SMPL + object clip — GVHMR + FoundationPose

From a monocular video, reconstruct one clip:
- the **body** as raw SMPL params (GVHMR, temporally smoothed, world-grounded);
- the **manipulated object** as per-frame 6DoF pose (FoundationPose, CAD known);
- both in the same frame, exported to `.npz`.

```
                 ┌─ GVHMR ──────────────► SMPL (θ, β, transl) / frame
   Video ────────┤
                 └─ SAM 2 (mask) + depth ─► FoundationPose (+ CAD) ─► object 6DoF / frame
                                                        │
                                              fuse.py (same frame)
                                                        ▼
                                            clip.npz : raw SMPL + object pose
```

## Files

| File | Role |
|---|---|
| `prepare_fp_inputs.py` | video → FoundationPose data folder (frames, `cam_K.txt` reused from GVHMR) |
| `estimate_depth.py` | RGB-only fallback: metric depth via Depth Anything V2 |
| `fuse.py` | core: load GVHMR + FoundationPose, align, export `clip.npz` |
| `run_pipeline.sh` | orchestrates the 4 steps (separate conda envs) |
| `requirements-glue.txt` | glue deps only |

GVHMR and FoundationPose have incompatible dependencies. Keep them in separate
conda envs; each step runs in its own env (`conda run -n ...`).

## Requirements

- NVIDIA GPU (both models are heavy; FoundationPose builds CUDA extensions).
- `conda`, `git`, `ffmpeg`.
- Object CAD **in meters** (`.obj`/`.glb`…). SMPL is metric — a CAD in mm/cm makes
  the object float or clip through the body.

## 1. Install GVHMR (body → SMPL)

```bash
git clone https://github.com/zju3dv/GVHMR ~/GVHMR
cd ~/GVHMR
# follow docs/INSTALL.md: creates conda env `gvhmr` + downloads checkpoints
```

SMPL/SMPL-X body models (free registration, not shipped with the repo):
- SMPL: https://smpl.is.tue.mpg.de
- SMPL-X: https://smpl-x.is.tue.mpg.de

Place them in `~/GVHMR/inputs/checkpoints/body_models/` (see INSTALL.md for layout).

Test:
```bash
conda run -n gvhmr python ~/GVHMR/tools/demo/demo.py --video=video.mp4 -s
# -> ~/GVHMR/outputs/demo/video/hmr4d_results.pt
```
`-s` = static camera (skips visual odometry). Moving camera → drop `-s`.

## 2. Install FoundationPose (object → 6DoF)

```bash
git clone https://github.com/NVlabs/FoundationPose ~/FoundationPose
cd ~/FoundationPose
# follow the README, or use install_fp_part1.sh then install_fp_part2.sh here
```

## 3. Install the glue

```bash
conda run -n gvhmr pip install -r requirements-glue.txt   # numpy, opencv
conda run -n gvhmr pip install smplx                      # only for --coord world
```

## Two manual steps (mask + depth)

FoundationPose is RGB-D model-based. On top of frames and the CAD it needs:

**a) Object mask on the first frame → `fp_data/masks/000000.png`**
Use SAM 2: one click on the object, binary mask (object=255). FoundationPose
initializes the pose on this frame, then tracks the object.

**b) Metric depth → `fp_data/depth/000000.png …` (uint16, mm)**
- With RGB-D (depth camera, iPhone LiDAR): use the real depth aligned to frames.
- RGB only: estimate metric depth (Depth Anything V2 metric, UniDepth, Metric3D),
  saved as uint16 mm, one image per frame, same names as `rgb/`. See `estimate_depth.py`.

Non-metric depth breaks the object scale and pose — this is the weak point of RGB-only.

`prepare_fp_inputs.py` creates empty `masks/` and `depth/` and reminds you to fill
them; you can also pass `--mask` and `--depth-dir` if they are ready.

## Run

Edit the paths/envs at the top of `run_pipeline.sh`, then:
```bash
bash run_pipeline.sh video.mp4 object.obj
```
Steps: GVHMR → FP prep → (pause for mask + depth) → FoundationPose → `fuse.py`.
Output: `work/video/clip.npz`.

### Output frame (`COORD` / `--coord`)
- `camera` (default, robust): body (`smpl_params_incam`) and object are already in
  the camera frame at the same scale → direct compose, no extra deps.
- `world` (gravity-aligned): the object is lifted to world via the camera→world
  transform from rigid (Umeyama) alignment of SMPL joints incam↔global. Needs
  `smplx` + SMPL models (`--smpl-model-dir`).

## `clip.npz` layout

| Key | Shape | Meaning |
|---|---|---|
| `poses` | (T, 3+ndof) | axis-angle `[global_orient(3), body_pose]` — SMPL ndof=69, SMPL-X 63 |
| `betas` | (T, nβ) | shape (per frame) |
| `betas_mean` | (nβ,) | mean shape |
| `trans` | (T, 3) | root translation |
| `global_orient` / `body_pose` | (T,3) / (T,ndof) | same, split |
| `mocap_framerate` | scalar | fps |
| `model_type` | str | `smpl` or `smplx` (auto-detected) |
| `coord_frame` | str | `camera` or `world` |
| `object_pose` | (T, 4, 4) | object → common frame |
| `object_R` / `object_t` | (T,3,3) / (T,3) | object rotation / translation |
| `object_valid` | (T,) bool | `False` = no detection, pose carry-forward filled |
| `camera_K` | (3,3) or (T,3,3) | GVHMR intrinsics |

```python
import numpy as np
d = np.load("clip.npz", allow_pickle=True)
print(d["model_type"], d["coord_frame"], d["poses"].shape, d["object_pose"].shape)
```

## Gotchas

- **Scale**: CAD in meters + metric depth, or body/object won't match. Bug #1.
- **Key names**: a few GVHMR/FoundationPose key and subfolder names depend on the
  version. `fuse.py` prints the available keys and exits cleanly if one differs.
- **SMPL vs SMPL-X**: GVHMR often outputs SMPL-X (`body_pose`=63); `fuse.py` detects
  and adapts. A downstream that wants strict SMPL (72) needs a joint remap.
- **Frame alignment**: GVHMR and FoundationPose must start at the same t0/fps.
  `fuse.py` truncates to the common length and warns on mismatch.
- **Hand/object occlusion**: manipulation hides the object, so FoundationPose may
  drop frames (`object_valid=False`). Temporal smoothing of `object_pose` helps.
- **Moving camera**: drop `-s` from GVHMR; per-frame metric depth handles the object.
