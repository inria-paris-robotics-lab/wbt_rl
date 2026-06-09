# Scripts

Pipeline entry points. Each script orchestrates one stage:
- imports `src/motion_convertor/` directly (same `wbt_rl` conda env)
- calls external modules via **subprocess** in their own conda env, described in `cfg/`
- never modifies submodule code

---

## Execution model

```
scripts/retarget.py        (runs in: wbt_rl env)
        │
        ├── import motion_convertor     (same process, same env)
        │
        └── subprocess.run(             (child process, module's own env)
                conda run -n {env}
                python {cmd} {args}     (read from cfg/01_retargeting/{retargeter}.yaml)
            )
```

The conda env name, command, and argument mapping for each module are declared in `cfg/`. Scripts read the relevant yaml at runtime — adding or swapping a module requires no script changes.

---

## Robot naming

`--robot` always requires an explicit DOF suffix:

| Value | Meaning | Supported by |
|-------|---------|-------------|
| `G1_29dof` | Unitree G1, 29 DOF | GMR, holosoma, holosoma_custom |
| `G1_27dof` | Unitree G1, 27 DOF | holosoma_custom only |
| `H1_29dof` | Unitree H1, 29 DOF | GMR |

Plain `G1` is rejected with an explicit error message.

Output directories embed the full robot name: `data/01_retargeted_motions/LAFAN_G1_29dof/`, `data/02_policies/LAFAN_G1_27dof/`, etc.

---

## retarget.py

Runs a full retargeting job for one (dataset, robot, retargeter) combination.

**CLI:**
```bash
source scripts/activate_wbt.sh
python scripts/retarget.py \
    --dataset LAFAN \
    --robot G1_29dof \
    --retargeter GMR \
    [--sequences seq1 seq2 ...]    # optional, defaults to all sequences
    [--run-id run_20240301_120000] # optional, resumes an existing run
    [--task-type robot_only|object_interaction]  # OMOMO only, default: robot_only
    [--object-urdf /path/to/object.urdf]         # object_interaction only — saved in config.yaml for train.py
    [--visualize]
```

**What it does, in order:**
1. Reads `cfg/00_datasets/data.yaml` → resolves raw dataset path
2. Reads `cfg/01_retargeting/{retargeter}.yaml` → resolves env, cmd, args, robot URDF/name
3. Validates that `--robot` is listed in `robot_config` for the chosen retargeter (error otherwise)
4. For each sequence:
   - `motion_convertor.to_retargeter_input()` → `{seq}_input_raw.{ext}`
   - `motion_convertor.to_unified_input()` → `{seq}_input_unified.npz`
   - subprocess (module env) → runs retargeter → `{seq}_output_raw.{ext}`
   - `motion_convertor.to_unified_output()` → `{seq}_output_unified.npz`
5. Writes `config.yaml` (full CLI args snapshot, includes `object_urdf` when provided)
6. Updates `latest →` symlink

**Output** — `data/01_retargeted_motions/{dataset}_{robot}/{retargeter}/run_{timestamp}/`:
```
{seq}_input_raw.{ext}
{seq}_input_unified.npz
{seq}_output_raw.{ext}
{seq}_output_unified.npz
config.yaml          ← includes object_urdf path if --object-urdf was passed
```

**Supported robot/retargeter combinations:**

| Robot | GMR | holosoma | holosoma_custom |
|-------|-----|----------|-----------------|
| G1_29dof | ✅ | ✅ | ✅ |
| G1_27dof | ❌ | ❌ | ✅ |

**Supported dataset/retargeter combinations:**

| Dataset | GMR | holosoma | holosoma_custom |
|---------|-----|----------|-----------------|
| LAFAN robot_only | ✅ | ✅ | ✅ |
| SFU robot_only | ✅ | ✅ | ✅ |
| OMOMO_NEW robot_only | ❌ | ✅ | ✅ |
| OMOMO_NEW object_interaction | ❌ | ✅ | ✅ |
| OMOMO robot_only | ✅ | ✅ | ✅ |
| OMOMO object_interaction | ❌ | 🚧 WIP | 🚧 WIP |

**Examples:**
```bash
source scripts/activate_wbt.sh

# LAFAN — GMR, single sequence
python scripts/retarget.py \
    --dataset LAFAN \
    --robot G1_29dof \
    --retargeter GMR \
    --sequences walk2_subject1

# SFU — holosoma, single sequence, with visualizer
python scripts/retarget.py \
    --dataset SFU \
    --robot G1_29dof \
    --retargeter holosoma \
    --sequences 0005_2FeetJump001_stageii \
    --visualize

# OMOMO_NEW — holosoma_custom, object_interaction, G1_27dof, custom object URDF
python scripts/retarget.py \
    --dataset OMOMO_NEW \
    --robot G1_27dof \
    --retargeter holosoma_custom \
    --task-type object_interaction \
    --object-urdf /path/to/my_object.urdf \
    --sequences sub3_largebox_003 \
    --visualize
```

---

## train.py

Prepares trainer input and launches training from an existing retargeting run.

**CLI:**
```bash
python scripts/train.py \
    --dataset LAFAN \
    --robot G1_29dof \
    --retargeter holosoma \
    --trainer holosoma \
    --simulator isaacsim \                              # see supported combinations below
    [--algo ppo|fast_sac] \                             # RL algorithm (default: ppo)
    [--logger-type wandb|wandb_offline|disabled] \      # trainer logger (default: wandb)
    [--retarget-task-type robot_only|object_interaction]  # which retarget run to use as source (default: robot_only)
    [--with-object] \                                   # train with object — object pose in critic only (default: robot-only)
    [--with-object-actor] \                             # train with object — object pose in both actor and critic obs (PPO only; implies --with-object)
    [--object-urdf /path/to/object.urdf] \              # override object URDF (auto-resolved from retarget config.yaml if absent)
    [--retarget-run latest] \                           # run ID or 'latest' (default: latest)
    [--num-envs 4096] \
    [--checkpoint path/to/ckpt.pt] \
    [--no-video]                                        # disable video recording during training
```

**`--retarget-task-type` vs `--with-object` / `--with-object-actor`** — these flags are independent:

| `--retarget-task-type` | object flag | Meaning |
|------------------------|-------------|---------|
| `robot_only` | *(absent)* | Source: no-object retarget → train without object |
| `object_interaction` | *(absent)* | Source: with-object retarget → train without object |
| `object_interaction` | `--with-object` | Source: with-object retarget → train with object (critic sees object) |
| `object_interaction` | `--with-object-actor` | Source: with-object retarget → train with object (actor + critic see object) |

**What it does, in order:**
1. Reads `cfg/02_training/{trainer}.yaml` → resolves env, cmd, `robot_exp_map`
2. Validates that `(--robot, --simulator, --algo)` is listed in `robot_exp_map` (error otherwise)
3. Locates retargeting run using `--retarget-task-type` to resolve the source directory:
   - LAFAN/SFU: `data/01_retargeted_motions/{dataset}_{robot}/{retargeter}/`
   - OMOMO robot_only: `OMOMO_robot_{robot}/{retargeter}/`
   - OMOMO object_interaction: `OMOMO_object_{robot}/{retargeter}/`
   - OMOMO_NEW: always `OMOMO_new_object_{robot}/{retargeter}/`
4. Reads `object_urdf` from the retarget run `config.yaml` (set by `retarget.py --object-urdf`); `--object-urdf` on this script takes precedence
5. For each sequence:
   - `motion_convertor.to_trainer_input()` → `{seq}_trainer_input.npz` (with object) or `{seq}_robot_only_trainer_input.npz` (without object, when `--retarget-task-type object_interaction` but no `--with-object`)
6. subprocess (trainer env) → runs training:
   ```
   python train_agent.py  exp:{exp_name}  simulator:{sim}  logger:{logger}  --motion-config...  --logger.base-dir ...
   ```
7. Saves to `data/02_policies/{dataset}_{robot}/{retargeter}_{trainer}/run_{timestamp}/`
8. Updates `latest →` symlink

**Supported robot/trainer/simulator/algo combinations:**

| Robot | Trainer | Simulator | Algo | robot_only | with_object | with_object_actor |
|-------|---------|-----------|------|------------|-------------|-------------------|
| G1_29dof | holosoma | isaacsim | ppo | ✅ | ✅ | ❌ |
| G1_29dof | holosoma | isaacsim | fast_sac | ✅ | ✅ | ❌ |
| G1_29dof | holosoma_custom | isaacsim | ppo | ✅ | ✅ | ✅ |
| G1_29dof | holosoma_custom | isaacsim | fast_sac | ✅ | ✅ | ❌ |
| G1_27dof | holosoma_custom | isaacsim | ppo | ✅ | ✅ | ✅ |
| G1_27dof | holosoma_custom | isaacsim | fast_sac | ✅ | ✅ | ❌ |

> **Note:** WBT training runs on **IsaacSim** for both trainers. isaacgym and mjwarp are not supported for
> WBT (asserted-unsupported at the env level for holosoma; no working WBT presets for holosoma_custom).

**Examples:**
```bash
# LAFAN — holosoma, G1_29dof, isaacsim, PPO, wandb
python scripts/train.py \
    --dataset LAFAN \
    --robot G1_29dof \
    --retargeter holosoma \
    --trainer holosoma \
    --simulator isaacsim \
    --no-video

# LAFAN — holosoma, G1_29dof, isaacsim, Fast-SAC, wandb offline
python scripts/train.py \
    --dataset LAFAN \
    --robot G1_29dof \
    --retargeter holosoma \
    --trainer holosoma \
    --simulator isaacsim \
    --algo fast_sac \
    --logger-type wandb_offline \
    --no-video

# OMOMO_NEW — retargeted with object, trained without object (decoupled)
python scripts/train.py \
    --dataset OMOMO_NEW \
    --robot G1_27dof \
    --retargeter holosoma_custom \
    --trainer holosoma_custom \
    --simulator isaacsim \
    --retarget-task-type object_interaction \
    --no-video
    # no --with-object → robot-only training on object-interaction retargeted data

# OMOMO_NEW — retargeted with custom object, trained with object
# object_urdf is auto-resolved from the retarget run config.yaml (set via retarget.py --object-urdf)
python scripts/train.py \
    --dataset OMOMO_NEW \
    --robot G1_27dof \
    --retargeter holosoma_custom \
    --trainer holosoma_custom \
    --simulator isaacsim \
    --retarget-task-type object_interaction \
    --with-object \
    --algo fast_sac \
    --no-video
    # --object-urdf is optional: auto-filled from retarget run config.yaml

# OMOMO — retargeted with object, trained with object, explicit URDF override
python scripts/train.py \
    --dataset OMOMO \
    --robot G1_27dof \
    --retargeter holosoma_custom \
    --trainer holosoma_custom \
    --simulator isaacsim \
    --retarget-task-type object_interaction \
    --with-object \
    --object-urdf /path/to/my_object.urdf \
    --no-video

# OMOMO_NEW — object pose exposed to actor (PPO only)
python scripts/train.py \
    --dataset OMOMO_NEW \
    --robot G1_29dof \
    --retargeter holosoma_custom \
    --trainer holosoma_custom \
    --simulator isaacsim \
    --retarget-task-type object_interaction \
    --with-object-actor \
    --no-video

# SFU — holosoma_custom, Fast-SAC, resume from checkpoint
python scripts/train.py \
    --dataset SFU \
    --robot G1_29dof \
    --retargeter holosoma_custom \
    --trainer holosoma_custom \
    --simulator isaacsim \
    --algo fast_sac \
    --checkpoint data/02_policies/SFU_G1_29dof/holosoma_custom_holosoma_custom/latest/checkpoint.pt \
    --no-video
```

---

## infer.py

Runs a trained policy via ROS2 (holosoma_custom only).

**CLI:**
```bash
# local policy run
python scripts/infer.py \
    --dataset LAFAN \
    --robot G1_29dof \
    --retargeter GMR \
    --trainer holosoma_custom \
    --config inference:g1-29dof-wbt \
    [--policy-run latest]

# wandb run (no local policy required)
python scripts/infer.py \
    --trainer holosoma_custom \
    --config inference:g1-29dof-wbt \
    --wandb-run wandb://entity/project/run_id/model.onnx
```

**Arguments:**

| Argument | Required | Notes |
|---|---|---|
| `--config` | yes | Tyro inference config (e.g. `inference:g1-29dof-wbt`, `inference:g1-27dof-wbt`) |
| `--trainer` | no | Default: `holosoma_custom`. Must match a file in `cfg/03_inference/`. `holosoma` is explicitly unsupported. |
| `--wandb-run` | wandb mode | Wandb URI passed directly as `--task.model-path` to `run_policy.py`. Mutually exclusive with `--dataset/--robot/--retargeter`. |
| `--dataset` | local mode | |
| `--robot` | local mode | Must include DOF suffix (e.g. `G1_29dof`) |
| `--retargeter` | local mode | |
| `--policy-run` | no | Run ID or `latest` (default: `latest`). Local mode only. |

**What it does:**
1. Reads `cfg/03_inference/{trainer}.yaml`
2. *Local mode:* locates `data/02_policies/{dataset}_{robot}/{retargeter}_{trainer}/{policy-run}/`, picks first `.onnx` or `.pt` file
3. *Wandb mode:* uses `--wandb-run` URI directly as model path
4. Subprocess (`hscinference` env): `python modules/03_inference/holosoma_custom/run_policy.py {config} --task.model-path {model} --robot.sdk-type ros2`

---

## deploy.py

Launches deployment components for WBT-RL via tmux. Orchestrates the simulator, watchdog, and ROS2 bridge.

**CLI:**
```bash
# Simulation mode (Gazebo + Watchdog + Bridge)
python scripts/deploy.py --mode SIM --robot g1_27dof

# Real robot mode (Shutdown sportsmode + Watchdog + Bridge)
python scripts/deploy.py --mode REAL --robot g1_27dof
```

**Arguments:**

| Argument | Required | Notes |
|---|---|---|
| `--mode` | yes | `SIM` (simulation) or `REAL` (physical robot) |
| `--robot` | no | Robot variant. Default: `g1_27dof`. Supported: `g1_27dof` |
| `--deployer` | no | Config name in `cfg/04_deployment/`. Default: `unitree` |

**What it does:**
1. Validates that the conda environment and workspace exist.
2. Reads `cfg/04_deployment/{deployer}.yaml` to resolve commands and environments.
3. Creates a **tmux session** (`wbt-deploy-sim` or `wbt-deploy-real`) with 3 panes:
   - **Pane 0 (top-left):**
     - `SIM`: `ros2 launch unitree_simulation launch_sim.launch.py`
     - `REAL`: `ros2 run unitree_control_interface shutdown_sportsmode.py`
   - **Pane 1 (top-right):** `ros2 launch unitree_control_interface watchdog.launch.py` (Watchdog)
   - **Pane 2 (bottom):** `python src/ros2_bridge/holosoma_inference_custom.py` (ROS2 Bridge)
4. Each pane automatically:
   - Activates the correct conda environment (e.g., `unitree_control_interface`).
   - Sources the ROS2 workspace (`cyclonedds_ws`).
   - Runs `autoset_environment_dds.py` to configure CycloneDDS for the selected mode.

**Layout:**
```
┌─────────────┬─────────────┐
│  sim /      │             │
│  shutdown   │  watchdog   │
├─────────────┴─────────────┤
│  bridge                   │
└───────────────────────────┘
```
