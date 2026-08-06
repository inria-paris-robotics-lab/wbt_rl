#!/usr/bin/env python3
"""
merge_dynamics.py — fold a stage-05 dynamics sidecar into its trainer npz.

Takes the per-contact-pair sidecar produced by enrich_dynamics.py and
aggregates it into the compact, side-resolved signals a reward function
actually wants: per-hand object contact, per-hand grip force, per-foot
contact schedule, per-foot ground reaction force, and the corrected
per-joint torque. Written as new `dyn_*` keys on top of the existing
trainer npz -- nothing already there is touched, so clips that were never
enriched keep working untouched (holosoma_custom's MotionLoader gates every
`dyn_*` field behind a `has_dyn_*` flag, same pattern as `has_gt_witness`).

Usage:
    python scripts/merge_dynamics.py \\
        --motion path/to/clip_trainer_input.npz \\
        --sidecar path/to/clip_output_dynamics.npz \\
        --out path/to/clip_enriched.npz

Pair-name conventions assumed (set by cfg/05_dynamics/scenes/*.yaml + the
scene builder, src/motion_convertor/wrappers/spider_build_scene.py):
    {side}_footN_floor                          foot / ground
    {side}_wrist_yaw_link_cvx_N__{obj}_geom      hand / dynamic object
    {obj}_geom__support_geom                     object / static support
    {obj}_geom__floor                            object / ground
where {obj} is the sidecar's own `object_names[0]` and {side} in
("left", "right"). Only single-dynamic-object scenes are supported.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

_SIDES = ("left", "right")


def _aggregate(contact_mask: np.ndarray, contact_force: np.ndarray,
               pairs: list[str], predicate) -> tuple[np.ndarray, np.ndarray]:
    """OR the mask and sum the force vector across every pair `predicate` selects."""
    idx = [i for i, p in enumerate(pairs) if predicate(p)]
    n_frames = contact_mask.shape[0]
    if not idx:
        return np.zeros(n_frames, dtype=bool), np.zeros(n_frames, dtype=np.float32)
    mask = contact_mask[:, idx].any(axis=1)
    force = np.linalg.norm(contact_force[:, idx].sum(axis=1), axis=-1).astype(np.float32)
    return mask, force


def convert(motion_path: Path, sidecar_path: Path, out_path: Path) -> None:
    motion = dict(np.load(motion_path, allow_pickle=True))
    sidecar = np.load(sidecar_path, allow_pickle=True)

    n_frames = motion["joint_pos"].shape[0]
    if sidecar["qpos"].shape[0] != n_frames:
        raise ValueError(
            f"frame count mismatch: motion has {n_frames}, sidecar has "
            f"{sidecar['qpos'].shape[0]} -- did the sidecar come from a "
            f"different clip / resampling?"
        )

    object_names = list(sidecar["object_names"])
    if len(object_names) != 1:
        raise ValueError(
            f"merge_dynamics.py only supports single-dynamic-object scenes, "
            f"got object_names={object_names}"
        )
    object_geom = f"{object_names[0]}_geom"

    contact_mask = sidecar["contact_mask"]
    contact_force = sidecar["contact_force"]
    pairs = list(sidecar["contact_pairs"])

    obj_contact_lr = np.zeros((n_frames, 2), dtype=bool)
    grip_force_lr = np.zeros((n_frames, 2), dtype=np.float32)
    foot_contact_lr = np.zeros((n_frames, 2), dtype=bool)
    foot_grf_lr = np.zeros((n_frames, 2), dtype=np.float32)
    for k, side in enumerate(_SIDES):
        obj_contact_lr[:, k], grip_force_lr[:, k] = _aggregate(
            contact_mask, contact_force, pairs,
            lambda p, side=side: p.endswith(f"__{object_geom}") and f"{side}_wrist" in p,
        )
        foot_contact_lr[:, k], foot_grf_lr[:, k] = _aggregate(
            contact_mask, contact_force, pairs,
            lambda p, side=side: p.endswith("_floor") and f"{side}_foot" in p,
        )

    box_support_contact, _ = _aggregate(
        contact_mask, contact_force, pairs, lambda p: p == f"{object_geom}__support_geom"
    )
    box_floor_contact, _ = _aggregate(
        contact_mask, contact_force, pairs, lambda p: p == f"{object_geom}__floor"
    )

    tau = sidecar["tau"]
    n_robot_dof = motion["joint_pos"].shape[1] - 7
    if tau.shape[1] != n_robot_dof:
        raise ValueError(
            f"actuator count {tau.shape[1]} != motion joint count {n_robot_dof} -- "
            f"joint ordering between the sidecar and the trainer npz is assumed "
            f"identical (both derive from the same G1 URDF joint declaration "
            f"order, empirically confirmed by the ~13 mrad tracking error between "
            f"them staying low and structured rather than blowing up); this "
            f"mismatch means that assumption broke and dyn_tau would be silently "
            f"misaligned per-joint if forced through anyway."
        )

    motion["dyn_tau"] = tau.astype(np.float32)
    motion["dyn_obj_contact_lr"] = obj_contact_lr
    motion["dyn_grip_force_lr"] = grip_force_lr
    motion["dyn_foot_contact_lr"] = foot_contact_lr
    motion["dyn_foot_grf_lr"] = foot_grf_lr
    motion["dyn_box_support_contact"] = box_support_contact
    motion["dyn_box_floor_contact"] = box_floor_contact

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **motion)

    print(f"merged dynamics -> {out_path}")
    print(f"  dyn_obj_contact_lr   L {obj_contact_lr[:, 0].mean():5.1%}  R {obj_contact_lr[:, 1].mean():5.1%}  "
          f"both {(obj_contact_lr[:, 0] & obj_contact_lr[:, 1]).mean():5.1%}")
    print(f"  dyn_foot_contact_lr  L {foot_contact_lr[:, 0].mean():5.1%}  R {foot_contact_lr[:, 1].mean():5.1%}")
    print(f"  dyn_box_support_contact {box_support_contact.mean():5.1%}  "
          f"dyn_box_floor_contact {box_floor_contact.mean():5.1%}")
    print(f"  dyn_tau |mean|={np.abs(tau).mean():.2f} max={np.abs(tau).max():.2f} Nm")
    if not box_support_contact.any() and not box_floor_contact.any():
        print("  WARNING: zero box/support/floor contact found -- check that "
              "'support_geom' matches this scene's static object name")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--motion", required=True, type=Path, help="retargeted clip trainer npz")
    ap.add_argument("--sidecar", required=True, type=Path, help="{seq}_output_dynamics.npz")
    ap.add_argument("--out", required=True, type=Path, help="destination enriched npz")
    args = ap.parse_args()
    convert(args.motion, args.sidecar, args.out)


if __name__ == "__main__":
    main()
