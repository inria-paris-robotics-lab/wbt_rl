# Retargeted Motions

Outputs of the retargeting pipeline. Organized as `{dataset}_{robot}/{retargeter}/run_{timestamp}/`.

The dataset and robot are the top-level context (fixed for a given experiment). The retargeter is what varies between runs, allowing direct comparison on identical data.

Each run is stored in a timestamped subfolder with its config:

```
{dataset}_{robot}/{retargeter}/
├── run_YYYYMMDD_HHMMSS/
│   ├── config.yaml                   # exact parameters used for this run
│   ├── {sequence}_input_raw.{ext}    # native input format for the retargeter
│   ├── {sequence}_input_unified.npz  # input in unified format
│   ├── {sequence}_output_raw.{ext}   # native output format of the retargeter
│   └── {sequence}_output_unified.npz # output in unified format
└── latest -> run_YYYYMMDD_HHMMSS/    # symlink to the most recent run
```

The `_unified` files are produced by `src/motion_convertor/`. See [specs/README.md](../../specs/README.md) for the full format specification of each dataset and retargeter.

---

## Unified format

All `_unified.npz` files follow this schema:

| Key | Shape | Description |
|-----|-------|-------------|
| `global_joint_positions` | `(T, 22, 3)` | Joint positions in world frame (metres), Z-up |
| `height` | `float` | Subject height in metres |
| `object_poses` *(optional)* | `(T, 7)` | `[qw, qx, qy, qz, x, y, z]` — object interaction only |

- 22 joints, SMPL-X convention
- Coordinate system: Z-up, world frame, metres

---

## Structure

```
01_retargeted_motions/
├── assets/                           # scene assets used during retargeting
│   ├── LAFAN/terrains/
│   ├── OMOMO/
│   │   ├── objects/                  # object meshes/URDFs (largebox, smallbox, ...)
│   │   └── terrains/
│   └── SFU/terrains/
│
├── LAFAN_G1/
│   ├── GMR/run_{timestamp}/
│   └── holosoma_retargeting/run_{timestamp}/
├── SFU_G1/
│   ├── GMR/run_{timestamp}/
│   └── holosoma_retargeting/run_{timestamp}/
├── OMOMO_robot_only_G1/
│   ├── GMR/run_{timestamp}/
│   └── holosoma_retargeting/run_{timestamp}/
└── OMOMO_object_interaction_G1/
    └── holosoma_retargeting/run_{timestamp}/
```

Folders are created automatically by `scripts/retarget.py` — no need to create them manually.
