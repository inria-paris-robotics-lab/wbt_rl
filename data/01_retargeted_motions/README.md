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

The `_unified.npz` files are produced by `src/motion_convertor/` and all follow a single schema
(`global_joint_positions (T,22,3)` Z-up + `height` + optional `object_poses (T,7)`). The
[motion_convertor README](../../src/motion_convertor/README.md#unified-format) is the **canonical
reference** for that format.

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
├── LAFAN_G1_29dof/
│   ├── GMR/run_{timestamp}/
│   └── holosoma/run_{timestamp}/
├── SFU_G1_29dof/
│   ├── GMR/run_{timestamp}/
│   └── holosoma_custom/run_{timestamp}/
├── OMOMO_robot_G1_29dof/                # robot_only retargets
│   ├── GMR/run_{timestamp}/
│   └── holosoma/run_{timestamp}/
└── OMOMO_object_G1_27dof/               # object_interaction retargets
    └── holosoma_custom/run_{timestamp}/
```

`{robot}` always carries its DOF suffix (`G1_29dof`, `G1_27dof`) and `{retargeter}` is the value passed to
`--retargeter` (`GMR`, `holosoma`, `holosoma_custom`). OMOMO splits into `OMOMO_robot_*` and `OMOMO_object_*`
sub-trees (see [scripts/README.md](../../scripts/README.md)).

Folders are created automatically by `scripts/retarget.py` — no need to create them manually.
