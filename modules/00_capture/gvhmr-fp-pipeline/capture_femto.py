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


def _set_int(dev, ob, name, val, label):
    pid = getattr(ob.OBPropertyID, name, None)
    if pid is None:
        print(f"[femto]   {label}: property absent on this firmware")
        return
    try:
        dev.set_int_property(pid, int(val))
        print(f"[femto]   {label} = {val}")
    except Exception as e:
        print(f"[femto]   {label}: could not set ({str(e).splitlines()[0][:60]})")


def _set_bool(dev, ob, name, val, label):
    pid = getattr(ob.OBPropertyID, name, None)
    if pid is None:
        print(f"[femto]   {label}: property absent on this firmware")
        return
    try:
        dev.set_bool_property(pid, bool(val))
        print(f"[femto]   {label} = {bool(val)}")
    except Exception as e:
        print(f"[femto]   {label}: could not set ({str(e).splitlines()[0][:60]})")


def _apply_color_controls(dev, ob, args):
    """Push the Jetson-ISP color settings that matter for GVHMR: anti-flicker,
    and either auto-exposure-with-a-ceiling (bounds motion blur) or fixed manual
    exposure. Must run AFTER pipe.start() — most controls aren't writable before."""
    print("[femto] applying Jetson-ISP color controls:")
    # anti-flicker: 0=off 1=50Hz 2=60Hz  (European mains -> 50 Hz)
    _set_int(dev, ob, "OB_PROP_COLOR_POWER_LINE_FREQUENCY_INT", args.powerline, "powerline/anti-flicker")
    if args.exposure is not None:
        # fixed manual exposure: kills auto hunting, most consistent frames
        _set_bool(dev, ob, "OB_PROP_COLOR_AUTO_EXPOSURE_BOOL", False, "auto-exposure")
        _set_int(dev, ob, "OB_PROP_COLOR_EXPOSURE_INT", args.exposure, "exposure (manual)")
        if args.gain is not None:
            _set_int(dev, ob, "OB_PROP_COLOR_GAIN_INT", args.gain, "gain (manual)")
    else:
        # auto-exposure ON to meter the scene; the blur ceiling is enforced in
        # software after warmup (this firmware rejects AE_MAX_EXPOSURE writes).
        _set_bool(dev, ob, "OB_PROP_COLOR_AUTO_EXPOSURE_BOOL", True, "auto-exposure")
        if args.max_exposure:
            print(f"[femto]   AE blur cap = {args.max_exposure} (enforced after {args.warmup}-frame warmup)")
    if args.wb is not None:
        _set_bool(dev, ob, "OB_PROP_COLOR_AUTO_WHITE_BALANCE_BOOL", False, "auto-WB")
        _set_int(dev, ob, "OB_PROP_COLOR_WHITE_BALANCE_INT", args.wb, "white balance (K)")


def _enforce_exposure_ceiling(dev, ob):
    """Read what auto-exposure settled on; if it exceeds args.max_exposure, lock
    the exposure to the cap and raise gain to keep roughly the same brightness.
    Returns after acting once. No-op in manual mode or when auto <= cap.
    Bound arg comes via closure attributes set by main()."""
    args = _enforce_exposure_ceiling.args
    if args.exposure is not None or not args.max_exposure:
        return
    try:
        cur = dev.get_int_property(ob.OBPropertyID.OB_PROP_COLOR_EXPOSURE_INT)
    except Exception:
        return
    if cur <= args.max_exposure:
        print(f"[femto]   auto-exposure settled at {cur} <= cap {args.max_exposure} -> kept on auto")
        return
    try:
        cur_gain = dev.get_int_property(ob.OBPropertyID.OB_PROP_COLOR_GAIN_INT)
    except Exception:
        cur_gain = 60
    # keep exposure*gain roughly constant so clamping the shutter doesn't darken
    new_gain = int(round(cur_gain * cur / args.max_exposure))
    new_gain = max(1, min(240, new_gain))
    if args.max_gain:
        new_gain = min(new_gain, args.max_gain)
    print(f"[femto]   auto-exposure hit {cur} > cap {args.max_exposure} -> clamping to reduce motion blur")
    _set_bool(dev, ob, "OB_PROP_COLOR_AUTO_EXPOSURE_BOOL", False, "auto-exposure")
    _set_int(dev, ob, "OB_PROP_COLOR_EXPOSURE_INT", args.max_exposure, "exposure (clamped to cap)")
    _set_int(dev, ob, "OB_PROP_COLOR_GAIN_INT", new_gain, f"gain ({cur_gain}->{new_gain}) to compensate")


def _apply_depth_controls(dev, ob, args):
    """Turn on the sensor's depth noise/speckle filters (off by default on this
    firmware). Removes flying pixels at object edges that corrupt the point cloud
    FoundationPose registers against. Depth range clipping is done per-frame in
    the loop (MIN/MAX_DEPTH device props are not writable on this firmware)."""
    if not args.depth_filter:
        print("[femto] depth noise filters: OFF (raw depth)")
        return
    print("[femto] enabling depth noise filters:")
    _set_bool(dev, ob, "OB_PROP_DEPTH_SOFT_FILTER_BOOL", True, "depth soft-filter (spatial)")
    _set_bool(dev, ob, "OB_PROP_DEPTH_NOISE_REMOVAL_FILTER_BOOL", True, "depth noise/speckle removal")


def main():
    ap = argparse.ArgumentParser(description="Capture aligned RGB-D from an Orbbec Femto Mega")
    ap.add_argument("--out", required=True, help="output folder (fp_data)")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--color", default="1920x1080", help="color WxH (BGR); 1920x1080@30 = SMPL sweet spot (4K/2.5K cap at 25fps)")
    ap.add_argument("--depth", default="640x576", help="depth WxH (Y16); 640x576=NFOV, 512x512=WFOV")
    ap.add_argument("--frames", type=int, default=0, help="stop after N recorded frames (0 = until 'q')")
    ap.add_argument("--warmup", type=int, default=30, help="frames dropped for exposure/ToF to settle")
    ap.add_argument("--no-preview", action="store_true", help="headless: record immediately, no window")
    # --- Jetson-ISP color controls (best RGB for GVHMR: sharp = low motion blur) ---
    ap.add_argument("--powerline", type=int, default=1, choices=[0, 1, 2],
                    help="anti-flicker: 0=off 1=50Hz(Europe) 2=60Hz")
    ap.add_argument("--max-exposure", type=int, default=300,
                    help="ceiling on AUTO-exposure to bound motion blur in dim light; 0=no cap")
    ap.add_argument("--max-gain", type=int, default=0,
                    help="ceiling on AUTO-gain to bound sensor noise; 0=leave firmware default")
    ap.add_argument("--exposure", type=int, default=None,
                    help="fixed MANUAL exposure (disables auto-exposure; overrides --max-exposure)")
    ap.add_argument("--gain", type=int, default=None, help="manual gain, use with --exposure")
    ap.add_argument("--wb", type=int, default=None,
                    help="fixed white balance in K (disables auto-WB), e.g. 5000; omit = auto")
    # --- depth quality for FoundationPose (cleaner point cloud = better 6DoF) ---
    ap.add_argument("--depth-filter", dest="depth_filter", action="store_true", default=True,
                    help="enable sensor depth noise/speckle filters (default ON; better FoundationPose)")
    ap.add_argument("--no-depth-filter", dest="depth_filter", action="store_false",
                    help="disable depth noise filters (raw depth)")
    ap.add_argument("--depth-min", type=int, default=0,
                    help="zero out depth below this (mm); 0=off — trims near clutter (e.g. 300)")
    ap.add_argument("--depth-max", type=int, default=0,
                    help="zero out depth above this (mm); 0=off — trims background wall/floor (e.g. 2000)")
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
    dev = pipe.get_device()
    _enforce_exposure_ceiling.args = args
    try:
        _apply_color_controls(dev, ob, args)
        _apply_depth_controls(dev, ob, args)
    except Exception as e:
        print(f"[femto] controls skipped: {e}")
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
    ceiling_done = False
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
            if not ceiling_done:
                _enforce_exposure_ceiling(dev, ob)
                ceiling_done = True

            color = np.frombuffer(c.get_data(), np.uint8).reshape(c.get_height(), c.get_width(), 3)
            draw = np.frombuffer(d.get_data(), np.uint16).reshape(d.get_height(), d.get_width())
            depth_mm = np.clip(draw.astype(np.float32) * d.get_depth_scale(), 0, 65535).astype(np.uint16)
            if args.depth_min:
                depth_mm[depth_mm < args.depth_min] = 0
            if args.depth_max:
                depth_mm[depth_mm > args.depth_max] = 0

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
