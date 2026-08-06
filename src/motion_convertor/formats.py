KNOWN_FORMATS: frozenset[str] = frozenset({
    # Raw dataset formats
    "lafan_bvh",
    "sfu_smplx_npz",
    "omomo_smplh_p",
    "omomo_new_pt",
    # Retargeter native input formats
    "gmr_bvh",
    "gmr_smplx_npz",
    "holosoma_npy",
    "holosoma_smplx_npz",
    "holosoma_smplh_npz",
    "holosoma_pt",
    # Retargeter native output formats
    "gmr_pkl",
    "holosoma_qpos_npz",
    "holosoma_custom_qpos_npz",
    # WBT-RL unified format
    "unified_npz",
    # Trainer native input formats
    "holosoma_trainer_npz",
    # Dynamics-enrichment formats (stage 05)
    "spider_scene_npz",      # SPIDER's trajectory_kinematic_act.npz + scene.xml
    "spider_dynamics_npz",   # WBT-RL sidecar: qpos/qvel/tau + contact forces
    # Policy formats
    "policy_onnx",
    "policy_pt",
    # Deployment
    "unitree_ros2_cmd",
})


def validate_format(fmt: str) -> None:
    if fmt not in KNOWN_FORMATS:
        raise ValueError(f"Unknown format {fmt!r}. Known formats: {sorted(KNOWN_FORMATS)}")
