#!/usr/bin/env python3
"""Capture aligned RGB-D from an Intel RealSense D435 into the folder layout the
pipeline expects (FoundationPose data + a video for GVHMR):

    <out>/
        rgb/000000.png ...     color frames (BGR)
        depth/000000.png ...   depth aligned to color, uint16 mm
        masks/                 empty — put the SAM2 first-frame mask here
        mesh/                  empty — put the (scaled) object CAD here
        cam_K.txt              color intrinsics (real sensor K)
        video.mp4              color frames as a video, for GVHMR

A live window shows the color + depth feed. It opens in AIMING mode (nothing is
saved) so you can frame the shot; press SPACE to start recording, q to stop/quit.
Depth is aligned to the color stream (rs.align), so depth and rgb share the same
frame/intrinsics — what FoundationPose needs. cam_K.txt holds the real color
intrinsics (feed this to FoundationPose rather than GVHMR's estimate).

Deps: pip install pyrealsense2 opencv-python numpy
"""

import argparse
import os
import sys

import numpy as np


def main():
    ap = argparse.ArgumentParser(description="Capture aligned RGB-D from a RealSense D435")
    ap.add_argument("--out", required=True, help="output folder (fp_data)")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--color", default="1280x720", help="color WxH")
    ap.add_argument("--depth", default="848x480", help="depth WxH (aligned up to color)")
    ap.add_argument("--frames", type=int, default=0, help="stop after N recorded frames (0 = until 'q')")
    ap.add_argument("--warmup", type=int, default=30, help="frames dropped for auto-exposure to settle")
    ap.add_argument("--no-preview", action="store_true", help="headless: record immediately, no window")
    args = ap.parse_args()

    try:
        import cv2
        import pyrealsense2 as rs
    except ImportError as e:
        sys.exit(f"[d435] missing dep ({e}). pip install pyrealsense2 opencv-python")

    cw, ch = (int(x) for x in args.color.lower().split("x"))
    dw, dh = (int(x) for x in args.depth.lower().split("x"))

    pipe = rs.pipeline()

    def _try(cow, coh, dew, deh, fps):
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, cow, coh, rs.format.bgr8, fps)
        cfg.enable_stream(rs.stream.depth, dew, deh, rs.format.z16, fps)
        return pipe.start(cfg)

    # requested config first, then USB2-safe fallbacks (640x480 @30/15/6)
    attempts = [(cw, ch, dw, dh, args.fps),
                (640, 480, 640, 480, 30),
                (640, 480, 640, 480, 15),
                (640, 480, 640, 480, 6)]
    profile = None
    for (cow, coh, dew, deh, fps) in attempts:
        try:
            profile = _try(cow, coh, dew, deh, fps)
        except RuntimeError:
            continue
        if (cow, coh, dew, deh, fps) != attempts[0]:
            print(f"[d435] requested config unavailable (USB {profile.get_device().get_info(rs.camera_info.usb_type_descriptor)}?) "
                  f"-> fell back to color {cow}x{coh} + depth {dew}x{deh} @ {fps}. "
                  "Use a USB 3.0 port for higher res @30.")
        cw, ch, args.fps = cow, coh, fps
        print(f"[d435] streaming color {cow}x{coh} + depth {dew}x{deh} @ {fps} fps")
        break
    if profile is None:
        sys.exit("[d435] no supported stream config found (check the cable/port and `rs-enumerate-devices`)")

    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()  # m per unit
    align = rs.align(rs.stream.color)  # depth -> color frame
    intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    K = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1.0]])
    print(f"[d435] depth_scale={depth_scale} m/unit | color intrinsics:\n{K}")

    # GUI available?
    gui = not args.no_preview
    if gui:
        try:
            cv2.namedWindow("D435")
        except cv2.error:
            gui = False
            print("[d435] no GUI available -> headless, recording immediately")

    writer = None
    recording = not gui           # headless: start recording right away
    n = 0

    def _start_recording():
        """Create output tree, write cam_K, open the video writer (lazy: only if we record)."""
        for sub in ("rgb", "depth", "masks", "mesh"):
            os.makedirs(os.path.join(args.out, sub), exist_ok=True)
        np.savetxt(os.path.join(args.out, "cam_K.txt"), K, fmt="%.8f")
        w = cv2.VideoWriter(os.path.join(args.out, "video.mp4"),
                            cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (cw, ch))
        print(f"[d435] recording -> {args.out}")
        return w

    if recording:
        writer = _start_recording()

    disp_scale = min(1.0, 1600.0 / (2 * cw))  # keep the side-by-side window reasonable
    print("[d435] warming up..." if not recording else "[d435] recording — press 'q' to stop")
    try:
        for _ in range(args.warmup):
            pipe.wait_for_frames()
        if gui and not recording:
            print("[d435] AIMING — frame the shot, press SPACE to record, q to quit")

        while True:
            frames = align.process(pipe.wait_for_frames())
            c = frames.get_color_frame()
            d = frames.get_depth_frame()
            if not c or not d:
                continue

            color = np.asanyarray(c.get_data())                        # (ch, cw, 3) BGR
            depth_raw = np.asanyarray(d.get_data()).astype(np.float32)  # units
            depth_mm = np.clip(depth_raw * depth_scale * 1000.0, 0, 65535).astype(np.uint16)

            if recording:
                name = f"{n:06d}.png"
                cv2.imwrite(os.path.join(args.out, "rgb", name), color)
                cv2.imwrite(os.path.join(args.out, "depth", name), depth_mm)
                writer.write(color)
                n += 1
                if n % 30 == 0:
                    print(f"[d435] {n} frames")
                if args.frames and n >= args.frames:
                    break

            if gui:
                vis = cv2.applyColorMap(cv2.convertScaleAbs(depth_mm, alpha=0.03), cv2.COLORMAP_JET)
                vis = np.hstack([color, cv2.resize(vis, (cw, ch))])
                if recording:
                    txt, col = f"REC  {n} frames   [q] stop", (0, 0, 255)
                    cv2.circle(vis, (25, 30), 10, (0, 0, 255), -1)
                else:
                    txt, col = "AIMING   [SPACE] record   [q] quit", (0, 220, 0)
                cv2.putText(vis, txt, (45, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4)
                cv2.putText(vis, txt, (45, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, 2)
                if disp_scale < 1.0:
                    vis = cv2.resize(vis, None, fx=disp_scale, fy=disp_scale)
                cv2.imshow("D435", vis)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if not recording and key in (32, ord("r")):  # SPACE or r
                    recording, n = True, 0
                    writer = _start_recording()
    except KeyboardInterrupt:
        print("\n[d435] stopped")
    finally:
        if writer is not None:
            writer.release()
        pipe.stop()
        if gui:
            cv2.destroyAllWindows()

    if n == 0:
        print("[d435] quit without recording (nothing saved)")
        return
    print(f"[d435] done: {n} frames -> {args.out}")
    print("       rgb/ depth/ (uint16 mm) cam_K.txt video.mp4")
    print("       next: SAM2 mask -> masks/000000.png ; scaled CAD -> mesh/ ; then GVHMR + FoundationPose")


if __name__ == "__main__":
    main()
