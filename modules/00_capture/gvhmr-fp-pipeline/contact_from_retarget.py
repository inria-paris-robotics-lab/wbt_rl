#!/usr/bin/env python3
"""Extract per-frame, per-hand hand<->object contact from HoloV2's OWN point-cloud interaction
fields (the "w/d/n" -- witness/distance/normal + active-in-margin flag), instead of a downstream
Euclidean-distance proxy on the retargeted robot.

This runs the `prepare()` stage only (grounds the demo + builds SDFs/correspondence) and calls
`process_frame()` per frame -- NO SQP solve needed, because `robot_interaction` is the DEMO's own
motion (human capture, point-cloud-correspondence-transported onto robot query points), independent
of what the solver will later do with the robot. So this is available immediately after retargeting
prepare, cheap, and reflects genuine captured contact (mesh-to-mesh SDF), not a center-to-center
heuristic.

Output mirrors the `select_grasp_anchor` convention used by holosoma_custom's grasp-settle code
(nearest of the two wrists): per frame, ``contact_active`` = OR over hands, ``contact_dist`` =
min over hands, ``contact_anchor_idx`` = argmin hand (0=left, 1=right, matching
grasp_settle_config.anchor_body_names order). These three arrays are what
`MotionLoader.has_gt_contact` expects in the training NPZ (object_ref_contact /
object_ref_contact_dist / object_ref_anchor_idx).

Also exports ``object_ref_witness_local`` (T,3): the WITNESS point (nearest box-surface point to
the selected hand) expressed in the box's LOCAL frame at that frame (world witness rotated/translated
by the inverse of the box's world pose). Box-local, not world, because it must stay comparable to
the box's CURRENT simulated pose at training time regardless of how the box has moved/rotated --
holosoma_custom's new ``object_surface_contact_error_exp`` reward computes the CURRENT witness the
same way (``box_geometry.box_nearest_and_signed_distance`` on the live sim) and measures the surface
geodesic distance between the two.

Must run with HoloV2/ as the working directory (same convention as retarget_to_qpos.py):

    cd .../HoloNew/HoloV2
    <holonew python> /path/to/contact_from_retarget.py \\
        --dataset amass --motion-path <clip>_built_amass.npz --model-dir <smplx dir> \\
        --hand-links left_wrist_yaw_link,right_wrist_yaw_link \\
        --out <clip>_contact.npz
"""
import argparse
import os
import sys

import numpy as np


def main():
    ap = argparse.ArgumentParser(description="HoloV2 demo contact (w/d/n) -> per-frame npz")
    ap.add_argument("--dataset", default="amass")
    ap.add_argument("--motion-path", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--dataset-root", default=None)
    ap.add_argument("--urdf", default="models/g1/g1_29dof.urdf")
    ap.add_argument("--robot-name", default="g1")
    ap.add_argument("--dof", type=int, default=29)
    ap.add_argument("--height", type=float, default=1.3)
    ap.add_argument("--hand-links", default="left_wrist_yaw_link,right_wrist_yaw_link",
                    help="comma-separated, ORDER MATTERS -- must match grasp_settle_config.anchor_body_names")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    sys.path.insert(0, os.getcwd())
    from pathlib import Path

    from src.prepare.config import PrepareConfig
    from src.prepare.contracts import RobotSpec, SceneSpec
    from src.prepare.runner import prepare
    from src.targets.pipeline import process_frame

    robot = RobotSpec(name=args.robot_name, urdf_path=Path(args.urdf), link_names=("pelvis",),
                      dof=args.dof, height=args.height)
    spec = SceneSpec(dataset=args.dataset, motion_path=Path(args.motion_path), robot=robot,
                     smpl_model_dir=Path(args.model_dir),
                     dataset_root=Path(args.dataset_root) if args.dataset_root else None)

    grounded, ctx = prepare(spec, PrepareConfig())
    print(f"[contact] prepared: {grounded.n_frames} frames, fps={grounded.fps}")

    obj_channel_idxs = [i for i, ch in enumerate(ctx.channels) if ch.object_idx is not None]
    print(f"[contact] channels: {[(i, ch.name, ch.object_idx) for i, ch in enumerate(ctx.channels)]}")
    if not obj_channel_idxs:
        raise SystemExit("no object channel found -- this clip has no object in the scene")
    box_channel = obj_channel_idxs[0]  # single-grasped-object clips (femto14): the box, not the support

    hand_link_order = args.hand_links.split(",")
    link_names = ctx.correspondence.link_names
    hand_point_masks = []
    for name in hand_link_order:
        assert name in link_names, f"{name} not in correspondence.link_names: {link_names}"
        link_id = link_names.index(name)
        mask = np.asarray(ctx.correspondence.link_idx) == link_id
        hand_point_masks.append(mask)
        print(f"[contact] hand '{name}': {mask.sum()} correspondence points")

    T = grounded.n_frames
    n_hands = len(hand_link_order)
    dist_per_hand = np.full((T, n_hands), np.nan)
    active_per_hand = np.zeros((T, n_hands), dtype=bool)
    witness_local_per_hand = np.zeros((T, n_hands, 3))

    for f in range(T):
        targets = process_frame(grounded, ctx, robot, f)
        field = targets.robot_interaction.field  # MultiChannelField (C, P)
        for h, mask in enumerate(hand_point_masks):
            hand_dist = field.distance[box_channel, mask]
            nearest_local_idx = int(np.argmin(hand_dist))
            dist_per_hand[f, h] = float(hand_dist[nearest_local_idx])
            active_per_hand[f, h] = bool(field.active[box_channel, mask].any())
            # witness of the point WITHIN this hand closest to the box (not an average -- the actual
            # nearest contact location). eval_fields transforms probes into the channel's (box-local)
            # frame BEFORE sampling the SDF (see targets/interaction/fields.py: `probe = (pts-pos)@rot`
            # happens first), so `witness` here is ALREADY box-local -- no further transform needed.
            witness_local_per_hand[f, h] = field.witness[box_channel, mask][nearest_local_idx]

    contact_anchor_idx = dist_per_hand.argmin(axis=1).astype(np.int64)
    contact_dist = dist_per_hand.min(axis=1)
    contact_active = active_per_hand[np.arange(T), contact_anchor_idx]
    witness_local = witness_local_per_hand[np.arange(T), contact_anchor_idx]  # (T, 3), box-local

    np.savez(
        args.out,
        object_ref_contact=contact_active,
        object_ref_contact_dist=contact_dist.astype(np.float32),
        object_ref_anchor_idx=contact_anchor_idx,
        object_ref_witness_local=witness_local.astype(np.float32),
        dist_per_hand=dist_per_hand.astype(np.float32),
        active_per_hand=active_per_hand,
        hand_link_order=np.asarray(hand_link_order),
        fps=grounded.fps,
    )
    n_on = int(contact_active.sum())
    print(f"[contact] wrote {args.out}: {n_on}/{T} frames in contact ({100 * n_on / T:.1f}%)")
    d = np.diff(np.concatenate([[0], contact_active.astype(np.int8), [0]]))
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1) - 1
    print(f"[contact] windows: {list(zip(starts.tolist(), ends.tolist()))}")
    print(f"[contact] anchor use: " +
          ", ".join(f"{hand_link_order[h]}={int((contact_anchor_idx == h).sum())}" for h in range(n_hands)))


if __name__ == "__main__":
    main()
