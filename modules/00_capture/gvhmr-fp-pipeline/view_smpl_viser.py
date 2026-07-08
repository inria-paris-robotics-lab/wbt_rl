#!/usr/bin/env python3
"""View a GVHMR SMPL-X result (hmr4d_results.pt) in a viser 3D viewer.

Loads the SMPL-X params, runs the body model to get per-frame vertices, and serves
an animated mesh with a frame slider + play/pause at http://localhost:<port>.
Self-contained: no dataset format conversion, straight from the GVHMR .pt.

Run in an env with viser + smplx + torch (e.g. holonew):
    python view_smpl_viser.py --pt <hmr4d_results.pt> --model-dir <body_models_parent>
"""

import argparse
import time

import numpy as np


def main():
    ap = argparse.ArgumentParser(description="viser viewer for a GVHMR SMPL-X clip")
    ap.add_argument("--pt", help="GVHMR hmr4d_results.pt (SMPL mode)")
    ap.add_argument("--model-dir",
                    help="parent of the smplx/ model folder (SMPL mode)")
    ap.add_argument("--verts-npz",
                    help="precomputed display-ready verts npz (e.g. from anchor_depth.py)")
    ap.add_argument("--coord", choices=["global", "incam"], default="global",
                    help="which frame to show (global = gravity-aligned world)")
    ap.add_argument("--gender", default="neutral")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--fps", type=float, default=30.0)
    args = ap.parse_args()

    import viser

    if args.verts_npz:
        d = np.load(args.verts_npz)
        verts = d["verts"].astype(np.float32)          # (T, V, 3), already display-ready
        faces = d["faces"].astype(np.int32)
        T = verts.shape[0]
        if "fps" in d:
            args.fps = float(d["fps"])
        print(f"[viser] {T} frames | verts-npz (depth-anchored, display-ready)")
    else:
        if not (args.pt and args.model_dir):
            raise SystemExit("[viser] need --pt and --model-dir (or use --verts-npz)")
        import torch
        import smplx

        data = torch.load(args.pt, map_location="cpu", weights_only=False)
        key = "smpl_params_global" if args.coord == "global" else "smpl_params_incam"
        p = data[key]
        betas = torch.as_tensor(np.asarray(p["betas"]), dtype=torch.float32)
        go = torch.as_tensor(np.asarray(p["global_orient"]), dtype=torch.float32)
        bp = torch.as_tensor(np.asarray(p["body_pose"]), dtype=torch.float32)
        tr = torch.as_tensor(np.asarray(p["transl"]), dtype=torch.float32)
        T, nb = betas.shape[0], betas.shape[1]
        print(f"[viser] {T} frames | betas={nb} | coord={args.coord}")

        model = smplx.create(args.model_dir, model_type="smplx", gender=args.gender,
                             num_betas=nb, use_pca=False, flat_hand_mean=True, batch_size=T)
        with torch.no_grad():
            out = model(betas=betas, global_orient=go, body_pose=bp, transl=tr)
        verts = out.vertices.numpy().astype(np.float32)     # (T, V, 3)
        joints = out.joints.numpy()                         # (T, J, 3)
        faces = model.faces.astype(np.int32)

        # Rotate geometry so the body's vertical axis becomes viser's up (+Z), then drop the
        # feet onto z=0. set_up_direction only moves the camera, not the grid, so the body would
        # otherwise lie flat in the ground plane — we rotate the mesh itself instead.
        def rot_to_z(a):
            a = a / (np.linalg.norm(a) + 1e-9)
            b = np.array([0.0, 0.0, 1.0])
            v = np.cross(a, b)
            c = float(np.dot(a, b))
            if c > 1 - 1e-8:
                return np.eye(3)
            if c < -1 + 1e-8:
                return np.diag([1.0, -1.0, -1.0])
            vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
            return np.eye(3) + vx + vx @ vx * (1.0 / (1.0 + c))

        # GVHMR world is gravity-aligned, so the true up is a pure axis. Snap head-pelvis to
        # the nearest world axis, else bent frames tilt the floor.
        raw_up = (joints[:, 15] - joints[:, 0]).mean(0)    # head - pelvis, averaged
        up = np.zeros(3)
        ax = int(np.argmax(np.abs(raw_up)))
        up[ax] = np.sign(raw_up[ax])
        R = rot_to_z(up)
        verts = np.einsum("ij,tvj->tvi", R, verts).astype(np.float32)
        # per-frame grounding: drop each frame's lowest point to z=0 (removes GVHMR's
        # vertical root drift — the "floats then lands" — without touching pose/size)
        verts[..., 2] -= verts[..., 2].min(axis=1, keepdims=True)

    server = viser.ViserServer(port=args.port)
    cx, cy = float(verts[..., 0].mean()), float(verts[..., 1].mean())
    server.scene.add_grid("/grid", width=6.0, height=6.0, position=(cx, cy, 0.0))

    def draw(i):
        server.scene.add_mesh_simple("/body", verts[i], faces,
                                     color=(210, 210, 210), flat_shading=False, side="double")

    gui_frame = server.gui.add_slider("frame", min=0, max=T - 1, step=1, initial_value=0)
    gui_play = server.gui.add_checkbox("play", True)
    gui_frame.on_update(lambda _: draw(gui_frame.value))
    draw(0)
    print(f"[viser] open http://localhost:{args.port}  (Ctrl-C to stop)")

    dt = 1.0 / args.fps
    while True:
        if gui_play.value:
            gui_frame.value = (gui_frame.value + 1) % T   # triggers on_update -> draw
        time.sleep(dt)


if __name__ == "__main__":
    main()
