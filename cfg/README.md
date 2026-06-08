# cfg/

Configuration files for all pipeline stages. Each yaml is the **single point of contact** between WBT-RL and an external module — it describes how to call the module without touching its code.

---

## Structure

```
cfg/
├── 00_datasets/           # data.yaml (paths) + per-dataset format metadata
├── 01_retargeting/        # one yaml per retargeter (gmr, holosoma, holosoma_custom, ...)
├── 02_training/           # one yaml per trainer (holosoma, holosoma_custom, ...)
├── 03_inference/          # one yaml per inference engine
├── 04_deployment/         # one yaml per deployer (unitree, ...)
└── motion_convertor/      # internal WBT-RL preprocessing helpers (not pipeline stages)
```

Numbered directories mirror `modules/` and `data/`: the number is the pipeline stage.
`motion_convertor/` has no number — it wraps WBT-RL's own wrapper scripts, not external modules.

---

## 00_datasets/

- `data.yaml` — centralises all dataset paths and body model locations, read by `src/motion_convertor/_config.py`
- `<dataset>.yaml` — per-dataset metadata (raw format, discover strategy), read by `scripts/retarget.py`

---

## Module yamls (`01_retargeting/`, `02_training/`, `03_inference/`, `04_deployment/`)

Each yaml describes one external module. The format is intentionally **not standardised** — each yaml mirrors the actual structure of the module it wraps.

Common fields (when applicable):

| Field | Description |
|-------|-------------|
| `env` | conda environment name to activate for this module |
| `cmd` | python command to run (relative to repo root) |
| `args` | mapping of WBT-RL argument names → module CLI flags |
| `setup_script` | module's own setup/activation script (if needed) |

Modules with multiple environments (e.g. holosoma training with isaacgym/isaacsim/mjwarp) use a `simulators:` sub-dict — one entry per env.

A yaml may declare `base: <other_module>` to inherit and override another yaml in the same stage directory (used by holosoma_custom to extend holosoma).

### Customisation

These files ship with defaults matching the standard install of each submodule. If you fork a module, rename a script, or use a different conda env name — edit only the relevant yaml. Nothing else in the pipeline needs to change.

### Adding a new module

1. Add the submodule under `modules/third_party/` and create symlinks in `modules/NN_stage/`
2. Create `cfg/NN_stage/<module_name>.yaml` following the pattern of existing yamls
3. Add the corresponding adapter in `src/motion_convertor/` (see `src/motion_convertor/README.md`)

---

## motion_convertor/

Internal preprocessing configs — FK computation, SMPL-H body model conversion, OMOMO → .pt conversion. These run WBT-RL's own wrapper scripts (in `src/motion_convertor/wrappers/`) inside specific conda envs and are not exposed as user-facing pipeline stages.
