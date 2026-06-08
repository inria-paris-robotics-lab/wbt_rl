# motion_convertor

Passive adapter layer between raw datasets and retargeters. It does not call retargeters — that is `scripts/retarget.py`'s job.

For each (dataset, retargeter) pair it produces two independent outputs:

1. **retargeter input** — native format expected by the retargeter
2. **unified input** — `global_joint_positions (T,22,3)` + `height` + optional `object_poses (T,7)`

These two are always derived directly from the raw dataset (option 2 — retargeter-first). They are independent: computing one does not require the other.

After retargeting, it also converts the retargeter's native output to unified:

3. **unified output** — same unified format, but for the robot motion

---

## Full data flow

The tool exposes **3 distinct functions**, each with a single responsibility:

```
── Role 1 ── called by scripts/retarget.py, BEFORE the retargeter ──

00_raw_datasets/{dataset}/
        ├─── to_retargeter_input()  ──►  {seq}_input_raw.{ext}
        └─── to_unified_input()    ──►  {seq}_input_unified.npz

        [retargeter — called by scripts/retarget.py, not by motion_convertor]
                │
                ▼
        {seq}_output_raw.{ext}

── Role 2 ── called by scripts/retarget.py, AFTER the retargeter ──

        {seq}_output_raw.{ext}
                └─── to_unified_output()  ──►  {seq}_output_unified.npz

── Role 3 ── called by scripts/train.py ──

        {seq}_output_raw.{ext}    (existing, from a prior retarget run)
                └─── to_trainer_input()   ──►  {seq}_trainer_input.npz
```

All files land in `01_retargeted_motions/{dataset}_{robot}/{retargeter}/run_{timestamp}/`.

> Role 3 reads the **raw retargeter output**, not the unified. The unified drops retargeter-specific data — raw→trainer preserves full fidelity.

---

## Unified format

| Key | Shape | Description |
|-----|-------|-------------|
| `global_joint_positions` | `(T, 22, 3)` | World-frame joint positions, Z-up, metres |
| `height` | float | Subject height in metres |
| `object_poses` *(optional)* | `(T, 7)` | `[qw, qx, qy, qz, x, y, z]` |

22 joints, SMPL-X convention; quaternions are wxyz (MuJoCo convention) throughout. **This section is the
canonical reference for the unified format** — other docs link here rather than redefining it.

---

## Architecture

```
src/motion_convertor/
├── __init__.py                   # 4 public dispatch functions
├── connectors.py                 # (src_fmt, dst_fmt) → converter dispatch table
├── formats.py                    # format registry + validate_format()
├── unified.py                    # save_unified / load_unified
├── _config.py                    # loads cfg/00_datasets/data.yaml, exposes repo_root() etc.
├── _subprocess.py                # conda_run(), run_entry_point(), load_module_cfg()
├── _to_unified_input/            # dataset FK → (T,22,3) Z-up
│   ├── lafan.py
│   ├── sfu.py
│   └── omomo.py
├── _to_retargeter_input/         # (dataset, retargeter) native input
│   ├── lafan_gmr.py / lafan_holosoma.py
│   ├── sfu_gmr.py / sfu_holosoma.py
│   └── omomo_gmr.py / omomo_holosoma.py
├── _to_unified_output/           # retargeter output → (T,22,3)
│   ├── gmr.py
│   └── holosoma.py               # also used for holosoma_custom output
├── _to_trainer_input/            # retargeter output → trainer native
│   ├── gmr_holosoma.py
│   ├── holosoma_holosoma.py
│   └── holosoma_custom_holosoma.py
├── wrappers/                     # thin scripts run via subprocess in module envs
│   ├── gmr_fk.py / gmr_smplx.py            # gmr env
│   ├── lafan_to_joints.py / sfu_to_joints.py / holosoma_convert.py  # hsretargeting env
│   └── omomo_to_joints.py / omomo_to_intermimic.py                  # hsretargeting / interact env
└── third_party/                  # git submodules
    ├── InterAct/                 # OMOMO → holosoma object_interaction preprocessing
    ├── lafan1/                   # LAFAN BVH tools (used by hsretargeting wrappers)
    ├── human_body_prior/         # SMPL-H FK (used by hsretargeting wrappers)
    └── smplx/                    # SMPL-X / SMPL-H body models (FK, merge_smplh_mano)
```

Adding a converter = add one function in `connectors.py` and register its `(src_fmt, dst_fmt)` pair in the
`CONNECTORS` table. The `_to_*` folders hold the actual conversion code, grouped by role.
See the repo-root [CONTRIBUTING.md](../../CONTRIBUTING.md) for the full module-authoring checklist.

---

## Conversion matrix

### raw → retargeter input

| Dataset | Retargeter | Conversion | Notes |
|---------|-----------|------------|-------|
| LAFAN | GMR | **none** — pass `.bvh` directly | GMR reads BVH natively, applies Y-up→Z-up internally |
| LAFAN | holosoma | BVH → `.npy (T,23,3)` Y-up, metres | Run BVH FK, keep Y-up (holosoma corrects internally) |
| SFU | GMR | **none** — pass `.npz` directly | GMR reads SMPL-X npz natively |
| SFU | holosoma | SMPL-X FK → unified `.npz` (T,22,3) + height, downsample 120→30 Hz | unified = retargeter input for holosoma |
| OMOMO | GMR | SMPL-H pickle → SMPL-X `.npz` | Remap 24 SMPL-H joints → 22 SMPL-X, drop L_Hand/R_Hand, reformat keys |
| OMOMO | holosoma | SMPL-H FK → unified `.npz` (T,22,3) + height + object_poses (T,7) | unified = retargeter input for holosoma |

### raw → unified input

| Dataset | Conversion | Body model needed |
|---------|-----------|-------------------|
| LAFAN | BVH FK → `(T,22,3)` Z-up + height=1.75 m | none (BVH skeleton) |
| SFU | SMPL-X FK → `(T,22,3)` Z-up + height from betas, downsample 120→30 Hz | SMPL-X (`models_smplx_v1_1/`) |
| OMOMO | SMPL-H FK → `(T,22,3)` Z-up + height + object_poses axis-angle→wxyz quat | SMPL-H (`smplh/`) |

### retargeter output → unified output

| Retargeter | Native output | Conversion |
|-----------|--------------|------------|
| GMR | `.pkl` — `root_pos (T,3)`, `root_rot (T,4)` xyzw, `dof_pos (T,N)` | xyzw→wxyz swap on root_rot, run robot FK to get `global_joint_positions (T,22,3)` |
| holosoma | `.npz` — `body_pos_w (T,B,3)`, `body_quat_w (T,B,4)` wxyz, `joint_pos (T,N)` | Extract the 22 tracked body positions → `global_joint_positions (T,22,3)` |
| holosoma_custom | same `.npz` schema as holosoma | identical conversion — reuses `_to_unified_output/holosoma.py` |

### retargeter output (raw) → trainer input

Input is always the **raw retargeter output**, not the unified.

| Retargeter | Trainer | Conversion | Output |
|-----------|---------|------------|--------|
| holosoma | holosoma | **none** — raw output is already form B (body_pos_w, joint_pos, ..., 50 Hz) | `.npz` passed directly |
| holosoma | holosoma (via native bridge) | Pass raw `qpos (T,36)` output → holosoma runs `convert_data_format_mj.py` internally | form A → form B done inside holosoma |
| GMR | holosoma | `root_pos + root_rot(xyzw) + dof_pos` → run robot FK in MuJoCo → produce `body_pos_w`, `joint_pos`, etc. at 50 Hz | `.npz` form B |
| holosoma_custom | holosoma_custom | resampling + object handling (`_to_trainer_input/holosoma_custom_holosoma.py`) | `.npz` form B |

> For holosoma→holosoma: the raw retargeter output **is** already the trainer input (form B). No conversion needed.

> **holosoma_custom** mirrors holosoma's data formats, so it reuses holosoma's unified-output converter and
> has its own trainer-input converter. **test_pipe** is an experimental sandbox that reuses the holosoma
> formats as well — it has no dedicated converter and is not part of the benchmarking baseline.

### Two connector philosophies (by design)

The conversion graph is **not** uniform, and this is intentional:

- **Retargeting path** (`to_unified_input` / `to_unified_output`) routes everything **through the unified
  format** — each solution only needs converters to/from unified, so adding a retargeter is *O(1)*.
- **Training path** (`to_trainer_input`) uses **direct `(retargeter, trainer)` pairs** (e.g.
  `gmr_pkl → holosoma_trainer_npz`), **not** the unified format — to preserve retargeter-specific data the
  unified format drops. The cost: this path is *O(retargeters × trainers)*. With a single trainer family
  (holosoma) today this is cheap; adding a structurally different trainer means writing one converter per
  retargeter that feeds it.

---

## third_party/

| Submodule | Used for |
|-----------|---------|
| **InterAct** | SMPL-H processing for OMOMO object_interaction → holosoma (via `src/motion_convertor/wrappers/omomo_to_intermimic.py`, `interact` env) |
| **lafan1** | LAFAN BVH parsing utilities (used by `src/motion_convertor/wrappers/lafan_to_joints.py`, `hsretargeting` env) |
| **human_body_prior** | SMPL-H forward kinematics for OMOMO (used by `src/motion_convertor/wrappers/omomo_to_joints.py`, `hsretargeting` env) |
| **smplx** | SMPL-X / SMPL-H body models and FK; `merge_smplh_mano` tooling (see `third_party/TODO.md`) |
