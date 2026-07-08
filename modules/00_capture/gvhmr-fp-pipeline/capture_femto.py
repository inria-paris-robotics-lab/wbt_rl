#!/usr/bin/env python3
"""Capture aligned RGB-D from an Orbbec Femto Mega (Azure-Kinect ToF sensor) into
the folder layout the pipeline expects (FoundationPose data + a video for GVHMR):

    <out>/
        rgb/000000.png ...     color frames (BGR)
        depth/000000.png ...   depth aligned to color, uint16 mm
        masks/                 empty — put the SAM2 first-frame mask here
        mesh/                  empty — put the (scaled) object CAD here
        cam_K.txt              color intrinsics (real sensor K)
        video.mp4              color frames as a video, for GVHMR

A live window shows the color + depth feed. It opens in AIMING mode (nothing is
saved); press SPACE to start recording, q to stop/quit. Depth is aligned to the
color stream (AlignFilter), so depth and rgb share frame/intrinsics — what
FoundationPose needs. cam_K.txt holds the real color intrinsics.

The depth stream is NFOV (640x576): its cone is narrower than the color frame, so
keep the subject centered (edges have no depth). Femto depth is uint16 mm.

Deps: pyorbbecsdk (Orbbec GitHub wheel) + opencv-python + numpy.
"""

import argparse
import os
import sys

import numpy as np


def _find(pl, w, h, fmt, fps):
    """First video profile in `pl` matching (w, h, fmt, fps), or None."""
    for i in range(pl.get_count()):
        p = pl.get_stream_profile_by_index(i).as_video_stream_profile()
        if (p.get_width(), p.get_height(), p.get_fps(),
                str(p.get_format()).split(".")[-1]) == (w, h, fps, fmt):
            return p
    return None


def main():
    ap = argparse.ArgumentParser(description="Capture aligned RGB-D from an Orbbec Femto Mega")
    ap.add_argument("--out", required=True, help="output folder (fp_data)")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--color", default="1280x720", help="color WxH (BGR)")
    ap.add_argument("--depth", default="640x576", help="depth WxH (Y16); 640x576=NFOV, 512x512=WFOV")
    ap.add_argument("--frames", type=int, default=0, help="stop after N recorded frames (0 = until 'q')")
    ap.add_argument("--warmup", type=int, default=30, help="frames dropped for exposure/ToF to settle")
    ap.add_argument("--no-preview", action="store_true", help="headless: record immediately, no window")
    args = ap.parse_args()

    try:
        import cv2
        import pyorbbecsdk as ob
    except ImportError as e:
        sys.exit(f"[femto] missing dep ({e}). Install pyorbbecsdk + opencv-python")

    cw, ch = (int(x) for x in args.color.lower().split("x"))
    dw, dh = (int(x) for x in args.depth.lower().split("x"))

    pipe = ob.Pipeline()
    cfg = ob.Config()
    cprof = (_find(pipe.get_stream_profile_list(ob.OBSensorType.COLOR_SENSOR), cw, ch, "BGR", args.fps)
             or _find(pipe.get_stream_profile_list(ob.OBSensorType.COLOR_SENSOR), 1280, 720, "BGR", 30))
    dprof = (_find(pipe.get_stream_profile_list(ob.OBSensorType.DEPTH_SENSOR), dw, dh, "Y16", args.fps)
             or _find(pipe.get_stream_profile_list(ob.OBSensorType.DEPTH_SENSOR), 640, 576, "Y16", 30))
    if cprof is None or dprof is None:
        sys.exit("[femto] no matching color/depth profile (try --color 1280x720 --depth 640x576)")
    cfg.enable_stream(cprof)
    cfg.enable_stream(dprof)
    try:
        pipe.enable_frame_sync()
    except Exception:
        pass
    pipe.start(cfg)
    align = ob.AlignFilter(ob.OBStreamType.COLOR_STREAM)

    it = cprof.get_intrinsic()
    K = np.array([[it.fx, 0, it.cx], [0, it.fy, it.cy], [0, 0, 1.0]])
    print(f"[femto] color {cprof.get_width()}x{cprof.get_height()} + depth {dprof.get_width()}x{dprof.get_height()} @ {args.fps}")
    print(f"[femto] color intrinsics:\n{K}")

    gui = not args.no_preview
    if gui:
        try:
            cv2.namedWindow("Femto")
        except cv2.error:
            gui = False
            print("[femto] no GUI available -> headless, recording immediately")

    writer = None
    recording = not gui
    n = 0

    def _start_recording():
        for sub in ("rgb", "depth", "masks", "mesh"):
            os.makedirs(os.path.join(args.out, sub), exist_ok=True)
        np.savetxt(os.path.join(args.out, "cam_K.txt"), K, fmt="%.8f")
        w = cv2.VideoWriter(os.path.join(args.out, "video.mp4"),
                            cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (cprof.get_width(), cprof.get_height()))
        print(f"[femto] recording -> {args.out}")
        return w

    if recording:
        writer = _start_recording()

    disp_scale = min(1.0, 1600.0 / (2 * cprof.get_width()))
    print("[femto] AIMING — frame the shot, SPACE to record, q to quit" if (gui and not recording)
          else "[femto] recording — press 'q' to stop")
    warmed = 0
    try:
        while True:
            fs = pipe.wait_for_frames(200)
            if fs is None:
                continue
            out = align.process(fs)
            if out is None:
                continue
            fs = out.as_frame_set() if hasattr(out, "as_frame_set") else out
            if fs is None:
                continue
            c = fs.get_color_frame()
            d = fs.get_depth_frame()
            if not (c and d):
                continue
            if warmed < args.warmup:
                warmed += 1
                continue

            color = np.frombuffer(c.get_data(), np.uint8).reshape(c.get_height(), c.get_width(), 3)
            draw = np.frombuffer(d.get_data(), np.uint16).reshape(d.get_height(), d.get_width())
            depth_mm = np.clip(draw.astype(np.float32) * d.get_depth_scale(), 0, 65535).astype(np.uint16)

            if recording:
                name = f"{n:06d}.png"
                cv2.imwrite(os.path.join(args.out, "rgb", name), color)
                cv2.imwrite(os.path.join(args.out, "depth", name), depth_mm)
                writer.write(color)
                n += 1
                if n % 30 == 0:
                    print(f"[femto] {n} frames")
                if args.frames and n >= args.frames:
                    break

            if gui:
                vis = cv2.applyColorMap(cv2.convertScaleAbs(depth_mm, alpha=0.03), cv2.COLORMAP_JET)
                vis = np.hstack([color, cv2.resize(vis, (color.shape[1], color.shape[0]))])
                if recording:
                    txt, col = f"REC  {n} frames   [q] stop", (0, 0, 255)
                    cv2.circle(vis, (25, 30), 10, (0, 0, 255), -1)
                else:
                    txt, col = "AIMING   [SPACE] record   [q] quit", (0, 220, 0)
                cv2.putText(vis, txt, (45, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4)
                cv2.putText(vis, txt, (45, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, 2)
                if disp_scale < 1.0:
                    vis = cv2.resize(vis, None, fx=disp_scale, fy=disp_scale)
                cv2.imshow("Femto", vis)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if not recording and key in (32, ord("r")):
                    recording, n = True, 0
                    writer = _start_recording()
    except KeyboardInterrupt:
        print("\n[femto] stopped")
    finally:
        if writer is not None:
            writer.release()
        pipe.stop()
        if gui:
            cv2.destroyAllWindows()

    if n == 0:
        print("[femto] quit without recording (nothing saved)")
        return
    print(f"[femto] done: {n} frames -> {args.out}")
    print("       rgb/ depth/ (uint16 mm) cam_K.txt video.mp4")
    print("       next: SAM2 mask -> masks/000000.png ; scaled CAD -> mesh/ ; then GVHMR + FoundationPose")


if __name__ == "__main__":
    main()
