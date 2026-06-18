# WBT-RL

A modular benchmarking and integration framework for humanoid robot Whole-Body Tracking (WBT). The goal is not to implement a single pipeline, but to provide the infrastructure to **plug in, compare, and combine** existing solutions at each stage of the WBT pipeline — from motion retargeting to RL training to real-robot deployment.

Each stage is interchangeable. Every run is versioned and traceable. Any two solutions can be compared on the same data under the same conditions.

---

## Philosophy

Most WBT research produces isolated solutions: a retargeter here, a trainer there, an inference stack somewhere else. Getting them to work together requires ad-hoc glue code that makes fair comparison impossible.

WBT-RL solves this by defining:

1. **A unified data format** — a common interface between all stages so any retargeter can feed any trainer
2. **Adapters** (`src/motion_convertor/`) — thin translation layers that convert each solution's native I/O to/from the unified format, without modifying the solutions themselves
3. **Versioned runs** — every run stores its config, inputs, and outputs so results are reproducible and comparable
4. **A modular module registry** (`modules/`) — existing solutions are plugged in as submodules or symlinks, untouched

---

## Pipeline Overview

```
Raw MoCap datasets (LAFAN1, OMOMO, SFU, ...)
        │
        ▼
  00_raw_datasets/
        │
        │  scripts/retarget.py       ← run any retargeter on any dataset
        │  src/motion_convertor/     ← adapter: dataset/retargeter formats ↔ unified format
        ▼
  01_retargeted_motions/             ← {dataset}_{robot}/{retargeter}/run_{timestamp}/
        │                                 native input, unified input
        │                                 native output, unified output
        │                                 config.yaml
        │  scripts/train.py          ← run any trainer on any retargeted motion
        │  src/motion_convertor/     ← adapter: retargeter output → trainer input format
        ▼
  02_policies/                       ← {dataset}_{robot}/{retargeter}_{trainer}/run_{timestamp}/
        │
        │  scripts/infer.py          ← deploy any policy in sim or on real robot
        │                               sdk_type: mujoco | ros2
        ▼
  04_deployment/
        │  unitree_ros2 + unitree_control_interface
        ▼
     Unitree G1 (or other humanoid) — sim or real
```

→ Format details: [data/01_retargeted_motions/README.md](data/01_retargeted_motions/README.md)

---

## Currently Integrated Solutions

### Retargeters

| Retargeter | Source | Datasets |
|------------|--------|---------|
| **GMR** | [YanjieZe/GMR](https://github.com/YanjieZe/GMR) | LAFAN1, SFU |
| **holosoma_retargeting** | [amazon-far/holosoma](https://github.com/amazon-far/holosoma) | LAFAN1, SFU, OMOMO_NEW |
| **holosoma_retargeting_custom** | [Guillaume-Bsst/holosoma_custom](https://github.com/Guillaume-Bsst/holosoma_custom) | LAFAN1, SFU, OMOMO_NEW |
| **HoloNew** ⚠️ | [Guillaume-Bsst/HoloNew](https://github.com/Guillaume-Bsst/HoloNew) | experimental |
| **test_pipe** ⚠️ | [Guillaume-Bsst/test_pipe](https://github.com/Guillaume-Bsst/test_pipe) | experimental |

> **OMOMO (original):** pipeline not yet working end-to-end — 🚧 in progress.
>
> ⚠️ **HoloNew** is a custom fork of holosoma_retargeting with its own isolated conda env (`holonew` under
> `~/.holonew_deps/`). It ships an installer but is **not yet wired into `cfg/`**, so it cannot be used
> through `retarget.py` yet — install on demand with `./install.sh holonew_retargeting`. 🚧 in progress.
>
> ⚠️ **test_pipe** is an experimental sandbox module. It is wired into `cfg/` and ships an installer, but
> is **excluded from `./install.sh`** (install it on demand with `./install.sh test_pipe_retargeting`).
> It reuses the holosoma data formats and is **not part of the benchmarking baseline**.

### Trainers

| Trainer | Source | Algorithms | Simulators |
|---------|--------|-----------|-----------|
| **holosoma** | [amazon-far/holosoma](https://github.com/amazon-far/holosoma) | PPO, FastSAC | IsaacGym, IsaacSim |
| **holosoma_custom** | [Guillaume-Bsst/holosoma_custom](https://github.com/Guillaume-Bsst/holosoma_custom) | PPO, FastSAC | IsaacSim |
| **test_pipe** ⚠️ | [Guillaume-Bsst/test_pipe](https://github.com/Guillaume-Bsst/test_pipe) | experimental | experimental |

### Inference

| Engine | Source | Modes |
|--------|--------|-------|
| **holosoma_inference** | [amazon-far/holosoma](https://github.com/amazon-far/holosoma) | MuJoCo sim-to-sim, Unitree API |
| **holosoma_inference_custom** | [Guillaume-Bsst/holosoma_custom](https://github.com/Guillaume-Bsst/holosoma_custom) | MuJoCo sim-to-sim, Unitree API, **ROS2** |
| **test_pipe** ⚠️ | [Guillaume-Bsst/test_pipe](https://github.com/Guillaume-Bsst/test_pipe) | experimental |

### Deployment

| Bridge | Source | Modes |
|--------|--------|-------|
| **unitree_ros2** | [unitreerobotics/unitree_ros2](https://github.com/unitreerobotics/unitree_ros2) | PyBullet sim-to-sim, sim-to-real via ROS2 |

---

## Repository Structure

```
wbt_rl/
├── data/
│   ├── 00_raw_datasets/       → see data/00_raw_datasets/README.md
│   ├── 01_retargeted_motions/ → see data/01_retargeted_motions/README.md
│   └── 02_policies/
│
├── cfg/                       # module configs — see cfg/README.md
│   ├── 00_datasets/           # data.yaml (paths) + per-dataset raw-format metadata
│   ├── 01_retargeting/        # one yaml per retargeter (gmr, holosoma, holosoma_custom, test_pipe)
│   ├── 02_training/           # one yaml per trainer (holosoma, holosoma_custom, test_pipe)
│   ├── 03_inference/          # one yaml per inference engine
│   ├── 04_deployment/         # one yaml per deployer (unitree)
│   └── motion_convertor/      # internal preprocessing helpers (not pipeline stages)
│
├── scripts/                   → see scripts/README.md
│   ├── retarget.py
│   ├── train.py
│   ├── infer.py
│   ├── deploy.py
│   └── activate_wbt.sh        # source this to activate the ecosystem
│
├── src/
│   ├── motion_convertor/      → see src/motion_convertor/README.md
│   │   └── wrappers/          # thin scripts run inside module envs (subprocess)
│   └── ros2_bridge/           # ROS2 inference bridge node (used by deploy.py)
│
├── install.sh                 # one-shot installer for all envs
├── CONTRIBUTING.md            # how to add a new module
│
└── modules/
    ├── 01_retargeting/
    │   ├── GMR/                            # submodule — YanjieZe/GMR
    │   ├── HoloNew/                        # submodule — Guillaume-Bsst/HoloNew  (experimental, not yet in cfg/)
    │   ├── holosoma_retargeting            # symlink → third_party/holosoma
    │   ├── holosoma_retargeting_custom     # symlink → third_party/holosoma_custom
    │   └── test_pipe                       # symlink → third_party/test_pipe  (experimental)
    ├── 02_training/
    │   ├── holosoma                        # symlink → third_party/holosoma
    │   ├── holosoma_custom                 # symlink → third_party/holosoma_custom
    │   └── test_pipe                       # symlink → third_party/test_pipe  (experimental)
    ├── 03_inference/
    │   ├── holosoma_inference              # symlink → third_party/holosoma
    │   ├── holosoma_inference_custom       # symlink → third_party/holosoma_custom
    │   └── test_pipe                       # symlink → third_party/test_pipe  (experimental)
    ├── 04_deployment/
    │   └── unitree_ros2/                   # submodule — unitreerobotics/unitree_ros2
    └── third_party/
        ├── holosoma/                       # submodule — amazon-far/holosoma
        ├── holosoma_custom/                # submodule — Guillaume-Bsst/holosoma_custom
        └── test_pipe/                      # submodule — Guillaume-Bsst/test_pipe  (experimental)
```

---

## Installation

### 1 — Clone with submodules
```bash
git clone --recurse-submodules https://github.com/inria-paris-robotics-lab/wbt_rl
# or after cloning:
git submodule update --init --recursive
```

### 2 — Install all envs
```bash
./install.sh
```

Four isolated ecosystems, nothing touches your system conda:

| Ecosystem | Location | Envs |
|-----------|----------|------|
| wbt + GMR | `~/.wbt_deps/` | `wbt_rl`, `gmr` |
| holosoma (upstream + custom) | `~/.holosoma_deps/` | `hsretargeting`, `hsmujoco`, `hsgym`, `hssim`, `hsinference` |
| HoloNew ⚠️ | `~/.holonew_deps/` | `holonew` |
| deployment | your system conda | `unitree_control_interface` |

Re-running is safe — already-installed envs are skipped via sentinel files.

**Selective install:**
```bash
./install.sh                                    # install everything (all variants)
./install.sh wbt                                # wbt_rl env only
./install.sh gmr                                # GMR env only
./install.sh interact                           # InterAct env (OMOMO object_interaction)
./install.sh holosoma_retargeting               # hsretargeting env (shared by holosoma + holosoma_custom)
./install.sh holosoma_training                  # hsmujoco + hsgym + hssim envs
./install.sh holosoma_training mujoco           # hsmujoco only
./install.sh holosoma_training isaacsim         # hssim only
./install.sh holosoma_training isaacgym         # hsgym only
./install.sh holosoma_training mujoco --no-warp # mujoco without warp
./install.sh holosoma_inference                 # hsinference env
./install.sh unitree_control_interface          # deployment env (system conda)
# On-demand only (excluded from full install):
./install.sh holonew_retargeting                # HoloNew isolated env
./install.sh test_pipe_retargeting              # test_pipe isolated env
```

### 3 — Install the datasets you want to use
Please follow [data/00_raw_datasets/README.md](data/00_raw_datasets/README.md)

### 4 — Activate the ecosystem
```bash
source scripts/activate_wbt.sh
```

And you can fully use the scripts ! [scripts/README.md](scripts/README.md)

Points your shell to `~/.wbt_deps/miniconda3` and activates `wbt_rl`. Switch to other envs with `conda activate <env>` as usual.