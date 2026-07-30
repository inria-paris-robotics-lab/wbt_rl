# Raw Motion Data

Raw datasets and body models to download before running the pipeline. No processing here — just
acquisition. All paths below are resolved from [`cfg/00_datasets/data.yaml`](../../cfg/00_datasets/data.yaml);
edit that file if your data lives elsewhere.

The folders below sit alongside this README:

| Folder | Contents | Used by |
|--------|----------|---------|
| `LAFAN/`    | LAFAN1 `.bvh` motions | LAFAN |
| `SFU/`      | SFU AMASS `.npz` motions | SFU |
| `CMU/`      | CMU AMASS `.npz` motions | CMU |
| `OMOMO/`    | OMOMO sequences + objects | OMOMO (original) |
| `OMOMO_new/`| InterMimic-processed OMOMO `.pt` | OMOMO_NEW |
| `HODome/`   | HODome SMPL-X human + single-object poses/meshes | HODome |
| `HOI-M3/`   | HOI-M3 "ground" SMPL-X human + object poses/meshes | HOI-M3 |
| `models/`   | **Shared body models** (SMPL-X, SMPL+H, MANO, DMPL) | SFU, OMOMO, HODome, HOI-M3 |

Body models are **centralised in `models/`** and shared across datasets — they are not duplicated under each
dataset folder.

---

## LAFAN/

```
LAFAN/
└── lafan1/
    ├── walk1_subject1.bvh
    ├── dance2_subject4.bvh
    └── ...                       # 77 .bvh sequences
```

1. Download [lafan1.zip](https://github.com/ubisoft/ubisoft-laforge-animation-dataset/blob/master/lafan1/lafan1.zip) (click "View Raw")
2. Extract the `.bvh` files into `LAFAN/lafan1/`

---

## SFU/

```
SFU/
└── SFU/
    ├── 0005/
    │   ├── 0005_Walking001_stageii.npz
    │   └── ...
    ├── 0008/  0012/  0015/  ...  # per-subject AMASS folders
    └── LICENSE.txt
```

1. Follow the [AMASS download instructions](https://amass.is.tue.mpg.de/) and select the **SFU** subset
   (SMPL-X N format) into `SFU/SFU/`
2. SFU's body model is the shared **SMPL-X** under `models/` — see [Body models](#models--shared-body-models) below

---

## CMU/

```
CMU/
└── CMU/
    ├── 01/
    │   ├── 01_01_stageii.npz
    │   └── ...
    ├── 63/  135/  ...          # per-subject AMASS folders
    └── ...
```

1. Follow the [AMASS download instructions](https://amass.is.tue.mpg.de/) and select the **CMU** subset
   (SMPL-X N format) into `CMU/CMU/`
2. Like SFU, CMU is a body-only AMASS subset — same double-nested layout and shared **SMPL-X** body model
   under `models/` (see [Body models](#models--shared-body-models) below); no objects

---

## OMOMO/

Original OMOMO dataset (`object_interaction` pipeline — 🚧 not yet working end-to-end).

```
OMOMO/
├── data/
│   ├── train_diffusion_manip_seq_joints24.p
│   ├── test_diffusion_manip_seq_joints24.p
│   ├── *_window_120_*_joints24.p          # processed windows + stats
│   ├── captured_objects/                  # object meshes (*.obj)
│   ├── object_bps_npy_files_joints24/     # BPS features
│   ├── object_bps_npy_files_for_eval_joints24/
│   └── rest_object_sdf_256_npy_files/     # object SDFs
└── omomo_text_anno/
    └── omomo_text_anno_json_data/         # text annotations (*.json)
```

1. Download the [OMOMO dataset](https://drive.google.com/file/d/1tZVqLB7II0whI-Qjz-z-AU3ponSEyAmm/view) and
   extract `data/` into `OMOMO/data/`
2. OMOMO's body models are the shared **SMPL+H** and **SMPL-X** under `models/` — see
   [Body models](#models--shared-body-models) below (required for `object_interaction` retargeting via the
   InterAct / InterMimic pipeline)

---

## OMOMO_new/

InterMimic-processed OMOMO. The data format differs from the original OMOMO dataset — one `.pt` file per
sequence, no separate body model needed.

```
OMOMO_new/
└── OMOMO_new/
    ├── sub1_suitcase_068.pt
    ├── sub10_clothesstand_000.pt
    └── ...                       # processed .pt sequences
```

1. Download the processed OMOMO data from [this link](https://drive.google.com/file/d/141YoPOd2DlJ4jhU2cpZO5VU5GzV_lm5j/view)
2. Extract so the `.pt` files land in `OMOMO_new/OMOMO_new/`

---

## HODome/

HODome release — single person + single object per sequence (`{subject}_{object}`). Reuses the shared
**SMPL-X** body model under `models/` (nothing extra to download).

```
HODome/
├── smplx/                       # SMPL-X human, one {subject}_{object}.npz (+ _meta.json) per sequence
├── object/                      # per-frame object 6DoF, one {subject}_{object}.npz per sequence
├── scaned_object/               # object meshes, one {object}.tar each
├── calibration_ground/{date}/   # per-date camera params
├── dataset_information.json      # date -> sequence list
└── startframe.json              # per-sequence start frames
```

---

## HOI-M3/

HOI-M3 **"ground"** release — multi-scene human-object interaction (living/bed/office/dining/fitness
rooms). Reuses the shared **SMPL-X** body model under `models/` (nothing extra to download).

```
HOI-M3/
├── mocap_ground/                # 204 {scene}_data{NN}_human.npz  (SMPL-X human)
│                                # 177 {scene}_data{NN}_object.npz (per-frame object 6DoF) — same folder, split by suffix
├── scanned_object/{object}/     # ~86 object meshes (.obj + textures)
├── calibration_ground/{date}/   # per-date camera params (calibration.json), 8 dates
├── dataset_information.json      # capture date -> sequence list
└── object_index.json            # object class list
```

1. Download the HOI-M3 "ground" archives (`calibration_ground*.zip`, `mocap_ground*.zip`, `scanned_object.tar`,
   `dataset_information.json`, `object_index.json`)
2. Extract each archive into `HOI-M3/` so the `calibration_ground/`, `mocap_ground/` and `scanned_object/`
   folders land directly under it, and drop the two `.json` files alongside them

---

## models/ — shared body models

Body models referenced by `cfg/00_datasets/data.yaml` (`body_model`, `body_model_smplx`). Shared across
datasets, so each model is downloaded once.

```
models/
├── models_smplx_v1_1/models/smplx/    # SMPL-X v1.1  — SFU + OMOMO object_interaction
│   ├── SMPLX_{NEUTRAL,MALE,FEMALE}.npz
│   └── SMPLX_{NEUTRAL,MALE,FEMALE}.pkl
├── smplh/                             # Extended SMPL+H — OMOMO
│   ├── {male,female,neutral}/model.npz
│   └── SMPLH_{MALE,FEMALE,NEUTRAL}.pkl  # built from SMPL+H + MANO (see step 4)
├── mano_v1_2/models/                  # MANO hand params — used to build SMPL+H
│   ├── MANO_{LEFT,RIGHT}.pkl
│   └── SMPLH_{female,male}.pkl
└── dmpls/                             # DMPL soft-tissue (AMASS)
    └── {male,female,neutral}/model.npz
```

1. **SMPL-X** — download from [smpl-x.is.tue.mpg.de](https://smpl-x.is.tue.mpg.de/download.php) and extract
   `models_smplx_v1_1.zip` into `models/models_smplx_v1_1/`
2. **Extended SMPL+H** — download from [mano.is.tue.mpg.de](https://mano.is.tue.mpg.de/download.php) and
   extract `smplh.tar.xz` into `models/smplh/`
3. **MANO** — download `mano_v1_2` from [mano.is.tue.mpg.de](https://mano.is.tue.mpg.de/download.php) into
   `models/mano_v1_2/` (and **DMPL** into `models/dmpls/` if needed)
4. The `SMPLH_{MALE,FEMALE,NEUTRAL}.pkl` files in `models/smplh/` are **produced**, not downloaded — merge the
   SMPL+H model with MANO hand parameters following
   [`src/motion_convertor/third_party/TODO.md`](../../src/motion_convertor/third_party/TODO.md)
