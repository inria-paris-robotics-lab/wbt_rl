#!/usr/bin/env python3
"""View a fused clip (body SMPL-X + object 6DoF) from fuse.py's clip.npz in viser.

Runs SMPL-X for the body, applies the per-frame object pose to the object mesh,
and serves both animated in http://localhost:<port>. Body and object are in the
same metric frame (clip made with --coord world), so the hands meet the object.

Run in an env with viser + smplx + torch + trimesh (e.g. holonew):
    python view_clip_viser.py --clip clip.npz --model-dir <body_models_parent>
"""

import argparse
import time

import numpy as np


def main():
    ap = argparse.ArgumentParser(description="viser viewer for a fused body+object clip")
    ap.add_argument("--clip", required=True, help="clip.npz from fuse.py")
    ap.add_argument("--model-dir", required=True, help="parent of the smplx/ model folder")
    ap.add_argument("--object-mesh", default="", help="override object CAD (else read from npz)")
    ap.add_argument("--extra-mesh", default="", help="2e CAD affiché CONCENTRIQUE (ex box36 réelle), en fil "
                    "de fer togglable dans viser -> voir de combien les mains poussent dans la vraie caisse")
    ap.add_argument("--gender", default="neutral")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--fps", type=float, default=30.0)
    args = ap.parse_args()

    import torch
    import smplx
    import trimesh
    import viser

    d = np.load(args.clip, allow_pickle=True)
    betas = torch.as_tensor(np.asarray(d["betas"]), dtype=torch.float32)
    go = torch.as_tensor(np.asarray(d["global_orient"]), dtype=torch.float32)
    bp = torch.as_tensor(np.asarray(d["body_pose"]), dtype=torch.float32)
    tr = torch.as_tensor(np.asarray(d["trans"]), dtype=torch.float32)
    T, nb = betas.shape[0], betas.shape[1]
    model_type = str(d["model_type"])
    print(f"[clip] {T} frames | {model_type} | {nb} betas")

    model = smplx.create(args.model_dir, model_type=model_type, gender=args.gender,
                         num_betas=nb, use_pca=False, flat_hand_mean=True, batch_size=T)
    with torch.no_grad():
        out = model(betas=betas, global_orient=go, body_pose=bp, transl=tr)
    body_v = out.vertices.numpy().astype(np.float32)     # (T, V, 3)
    joints = out.joints.numpy()
    body_f = model.faces.astype(np.int32)

    # object: apply per-frame 4x4 pose (object -> world) to the CAD vertices
    obj_path = args.object_mesh or str(d["object_mesh"])
    mesh = trimesh.load(obj_path, force="mesh", process=False)
    ov = np.asarray(mesh.vertices, np.float32)
    obj_f = np.asarray(mesh.faces, np.int32)
    P = np.asarray(d["object_pose"], np.float32)          # (T, 4, 4)
    obj_v = np.einsum("tij,vj->tvi", P[:, :3, :3], ov) + P[:, None, :3, 3]  # (T, Vo, 3)

    # upright: snap body head-pelvis to nearest axis -> +Z; rotate both meshes
    raw_up = (joints[:, 15] - joints[:, 0]).mean(0)
    up = np.zeros(3); ax = int(np.argmax(np.abs(raw_up))); up[ax] = np.sign(raw_up[ax])
    b = np.array([0.0, 0.0, 1.0]); v = np.cross(up, b); c = float(np.dot(up, b))
    if c > 1 - 1e-8:
        R = np.eye(3)
    elif c < -1 + 1e-8:
        R = np.diag([1.0, -1.0, -1.0])
    else:
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R = np.eye(3) + vx + vx @ vx * (1.0 / (1.0 + c))
    body_v = np.einsum("ij,tvj->tvi", R, body_v).astype(np.float32)
    obj_v = np.einsum("ij,tvj->tvi", R, obj_v).astype(np.float32)

    # body: per-frame grounding on its lowest point (feet on the floor each frame).
    floor_t = body_v[..., 2].min(axis=1)                  # (T,)
    body_v[..., 2] -= floor_t[:, None]
    # object: shift by a SMOOTHED floor, so the body's per-frame foot jitter is not
    # injected into the box (that was a source of the visible tremble).
    k = 15
    floor_s = np.convolve(np.pad(floor_t, k // 2, mode="edge"), np.ones(k) / k, mode="valid")
    obj_v[..., 2] -= floor_s[:, None]
    # object-floor contact: the box rests on the ground -> put its resting level at z=0.
    box_bottom = obj_v[..., 2].min(axis=1)
    resting = box_bottom <= np.percentile(box_bottom, 30)
    rest_shift = float(np.median(box_bottom[resting]))
    obj_v[..., 2] -= rest_shift

    # 2e caisse (ex box36 réelle) : MÊMES pose/rotation/décalages -> CONCENTRIQUE avec l'objet grippé,
    # donc les mains posées sur la 32 apparaissent à l'intérieur de la 36 (= la poussée du grip serré).
    obj36_v = obj36_f = None
    if args.extra_mesh:
        m36 = trimesh.load(args.extra_mesh, force="mesh", process=False)
        v36 = np.asarray(m36.vertices, np.float32)
        obj36_f = np.asarray(m36.faces, np.int32)
        obj36_v = np.einsum("tij,vj->tvi", P[:, :3, :3], v36) + P[:, None, :3, 3]
        obj36_v = np.einsum("ij,tvj->tvi", R, obj36_v).astype(np.float32)
        obj36_v[..., 2] -= floor_s[:, None]
        obj36_v[..., 2] -= rest_shift

    server = viser.ViserServer(port=args.port)
    cx, cy = float(body_v[..., 0].mean()), float(body_v[..., 1].mean())
    server.scene.add_grid("/grid", width=6.0, height=6.0, position=(cx, cy, 0.0))

    gui_show36 = (server.gui.add_checkbox("caisse 36 (réelle, fil de fer)", True)
                  if obj36_v is not None else None)

    def draw(i):
        server.scene.add_mesh_simple("/body", body_v[i], body_f,
                                     color=(210, 210, 210), flat_shading=False, side="double")
        server.scene.add_mesh_simple("/object", obj_v[i], obj_f,
                                     color=(235, 130, 20), flat_shading=True, side="double")
        if obj36_v is not None:
            server.scene.add_mesh_simple("/object36", obj36_v[i], obj36_f,
                                         color=(60, 140, 235), flat_shading=True, side="double",
                                         wireframe=True, visible=gui_show36.value)

    gui_frame = server.gui.add_slider("frame", min=0, max=T - 1, step=1, initial_value=0)
    gui_play = server.gui.add_checkbox("play", True)
    gui_frame.on_update(lambda _: draw(gui_frame.value))
    if gui_show36 is not None:
        gui_show36.on_update(lambda _: draw(gui_frame.value))
    draw(0)
    print(f"[clip] open http://localhost:{args.port}  (Ctrl-C to stop)")

    dt = 1.0 / args.fps
    while True:
        if gui_play.value:
            gui_frame.value = (gui_frame.value + 1) % T
        time.sleep(dt)


if __name__ == "__main__":
    main()
