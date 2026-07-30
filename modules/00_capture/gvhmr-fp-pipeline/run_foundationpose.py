#!/usr/bin/env python3
"""Run FoundationPose on a capture folder, downscaled + headless.

Same register(frame 0)+track loop as FoundationPose's run_demo.py, but:
  - downscales via YcbineoatReader(shorter_side=...) so 720p input fits in GPU memory
    (full-res register OOMs on ~12 GB cards),
  - no cv2.imshow (runs in the background); optionally writes pose-overlay frames.

Writes 4x4 object->camera poses to <debug_dir>/ob_in_cam/*.txt (what fuse.py reads).
Run with the foundationpose env python.
"""

import argparse
import os
import sys

import numpy as np

FP_DIR = os.path.expanduser("~/FoundationPose")


def main():
    ap = argparse.ArgumentParser(description="FoundationPose runner (downscaled, headless)")
    ap.add_argument("--mesh_file", required=True)
    ap.add_argument("--test_scene_dir", required=True)
    ap.add_argument("--debug_dir", required=True)
    ap.add_argument("--shorter-side", type=int, default=640,
                    help="downscale so short side = this (640 keeps more depth detail; 12GB VRAM OK. "
                         "Lower to 480 if register() OOMs, raise toward 720 for more precision)")
    ap.add_argument("--est-refine-iter", type=int, default=5)
    ap.add_argument("--track-refine-iter", type=int, default=2)
    ap.add_argument("--save-vis", action="store_true", help="write pose-overlay frames to track_vis/")
    args = ap.parse_args()

    sys.path.insert(0, FP_DIR)
    os.chdir(FP_DIR)
    import cv2
    import trimesh
    import nvdiffrast.torch as dr
    from estimater import (FoundationPose, ScorePredictor, PoseRefinePredictor,
                           set_logging_format, set_seed, draw_posed_3d_box, draw_xyz_axis)
    from datareader import YcbineoatReader
    import logging

    set_logging_format()
    set_seed(0)

    mesh = trimesh.load(args.mesh_file)
    os.system(f"rm -rf {args.debug_dir}/* && mkdir -p {args.debug_dir}/ob_in_cam {args.debug_dir}/track_vis")
    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)

    est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh,
                         scorer=ScorePredictor(), refiner=PoseRefinePredictor(),
                         debug_dir=args.debug_dir, debug=0, glctx=dr.RasterizeCudaContext())
    logging.info("estimator ready")

    reader = YcbineoatReader(video_dir=args.test_scene_dir, shorter_side=args.shorter_side, zfar=np.inf)
    logging.info(f"streaming {len(reader.color_files)} frames at {reader.W}x{reader.H}")

    for i in range(len(reader.color_files)):
        color = reader.get_color(i)
        depth = reader.get_depth(i)
        if i == 0:
            mask = reader.get_mask(0).astype(bool)
            pose = est.register(K=reader.K, rgb=color, depth=depth, ob_mask=mask,
                                iteration=args.est_refine_iter)
        else:
            pose = est.track_one(rgb=color, depth=depth, K=reader.K, iteration=args.track_refine_iter)
        np.savetxt(f"{args.debug_dir}/ob_in_cam/{reader.id_strs[i]}.txt", pose.reshape(4, 4))

        if args.save_vis:
            cp = pose @ np.linalg.inv(to_origin)
            vis = draw_posed_3d_box(reader.K, img=color, ob_in_cam=cp, bbox=bbox)
            vis = draw_xyz_axis(color, ob_in_cam=cp, scale=0.1, K=reader.K, thickness=3,
                                transparency=0, is_input_rgb=True)
            cv2.imwrite(f"{args.debug_dir}/track_vis/{reader.id_strs[i]}.png", vis[..., ::-1])
        if i % 30 == 0:
            logging.info(f"frame {i}")

    logging.info(f"done -> {args.debug_dir}/ob_in_cam")


if __name__ == "__main__":
    main()
