# Raw Motion Data

Raw datasets and body models to download before running the pipeline. No processing here — just acquisition.

Three dataset folders sit alongside this README: `LAFAN/`, `OMOMO/`, `SFU/`.

---

## LAFAN/

```
LAFAN/
└── lafan1/
    ├── walk1_subject1.bvh
    ├── dance2_subject4.bvh
    └── ...
```

1. Download [lafan1.zip](https://github.com/ubisoft/ubisoft-laforge-animation-dataset/blob/master/lafan1/lafan1.zip) (click "View Raw")
2. Extract `.bvh` files into `LAFAN/lafan1/`

---

## OMOMO/

```
OMOMO/
├── data/
│   ├── train_diffusion_manip_seq_joints24.p
│   ├── test_diffusion_manip_seq_joints24.p
│   ├── captured_objects/
│   │   ├── largebox_cleaned_simplified.obj
│   │   └── ...
│   └── ...
├── smplh/                         ← Extended SMPL+H (smplh.tar.xz)
│   ├── male/model.npz
│   ├── female/model.npz
│   └── neutral/model.npz
└── smplx/                         ← SMPL-X (smplx.zip) — for object_interaction
    └── smplh/
        ├── SMPLH_MALE.pkl
        └── SMPLH_FEMALE.pkl
```

1. Download the [OMOMO dataset](https://drive.google.com/file/d/1tZVqLB7II0whI-Qjz-z-AU3ponSEyAmm/view) and extract `data/` into `OMOMO/data/`
2. Download **Extended SMPL+H model for AMASS** from [mano.is.tue.mpg.de](https://mano.is.tue.mpg.de/download.php) and extract `smplh.tar.xz` into `OMOMO/smplh/`
3. Download **SMPL-X models** from [smpl-x.is.tue.mpg.de](https://smpl-x.is.tue.mpg.de/download.php) and extract `smplx.zip` into `OMOMO/smplx/` — required for `object_interaction` retargeting (InterAct/InterMimic pipeline)

---

## OMOMO_new/

The holosoma retargeting pipeline uses the processed dataset by InterMimic. The data format differs from the original OMOMO dataset.

1. Download the processed OMOMO data from [this link](https://drive.google.com/file/d/141YoPOd2DlJ4jhU2cpZO5VU5GzV_lm5j/view)
2. Extract the downloaded folder to `/OMOMO_new`

The data in `OMOMO_new/` should be `.pt` files.

---

## SFU/

```
SFU/
├── SFU/
│   ├── 0005/
│   │   ├── neutral_stagei.npz
│   │   ├── 0005_Walking001_stageii.npz
│   │   └── ...
│   ├── 0008/
│   └── ...
└── models_smplx_v1_1/
    └── models/
        └── smplx/
            ├── SMPLX_NEUTRAL.npz
            ├── SMPLX_MALE.npz
            └── SMPLX_FEMALE.npz
```

1. Follow [AMASS download instructions](https://amass.is.tue.mpg.de/) and select the **SFU** subset (SMPL-H format) into `SFU/SFU/`
2. Download [SMPL-X models](https://smpl-x.is.tue.mpg.de/) (SMPL-X N neutral format) and extract `models_smplx_v1_1.zip` into `SFU/models_smplx_v1_1/`
