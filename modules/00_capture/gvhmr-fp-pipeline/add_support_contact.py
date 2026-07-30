#!/usr/bin/env python3
"""Ajoute au npz training la REFERENCE de contact robot<->table (support), symetrique du
contact robot<->box deja present (object_ref_*). Rend la table un objet a part entiere du
training : le robot la voit (obs) et est recompense pour placer sa main correctement pres
d'elle (reward), au lieu de la cuire dans le sol.

Calcul self-contained (trimesh, pas de dependance HoloV2) : pour chaque frame et chaque
poignet (anchor_body_names), distance non signee a la SURFACE du mesh table (dans le repere
LOCAL table, pose statique support_pos_w/quat_w), witness = point de surface le plus proche
(deja local). anchor = poignet le plus proche ; contact = distance < seuil.

Ecrit :
  support_ref_contact (T,) bool, support_ref_contact_dist (T,) f32,
  support_ref_anchor_idx (T,) i64, support_ref_witness_local (T,3) f32,
  support_half_extents (3,) f32  (demi-tailles du mesh -> SDF boite pour la reward courante),
  support_mass (), support_pointcloud (P,3) f32 (surface echantillonnee, obs/viz/futur).

Usage :
  python add_support_contact.py --motion <..._gtcontact.npz> --mesh-dir <dir des .obj>
      [--thr 0.30] [--n-cloud 512] [--out <..._nobj.npz>]
"""
import argparse
import os

import numpy as np
import trimesh


def quat_wxyz_to_R(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--motion", required=True, help="npz training avec support_pos_w/quat_w/mesh + body_pos_w")
    ap.add_argument("--mesh-dir", required=True, help="dossier contenant le mesh table (support_mesh)")
    ap.add_argument("--anchors", default="left_wrist_yaw_link,right_wrist_yaw_link",
                    help="ORDRE = grasp_settle.anchor_body_names (0=left,1=right)")
    ap.add_argument("--thr", type=float, default=0.30, help="seuil de contact main<->table (m)")
    ap.add_argument("--n-cloud", type=int, default=512, help="points de surface echantillonnes")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    d = dict(np.load(args.motion, allow_pickle=True))
    assert "support_pos_w" in d, "pas de support (table) dans ce npz"
    body_names = [str(x) for x in d["body_names"]]
    anchors = args.anchors.split(",")
    a_idx = [body_names.index(a) for a in anchors]  # indices des poignets
    T = d["body_pos_w"].shape[0]

    sup_p = np.asarray(d["support_pos_w"], float)          # (3,)
    sup_q = np.asarray(d["support_quat_w"], float)         # (4,) wxyz
    R = quat_wxyz_to_R(sup_q)                               # table->world
    mesh_name = str(d["support_mesh"])
    mesh = trimesh.load(os.path.join(args.mesh_dir, mesh_name), force="mesh")
    half = (mesh.bounds[1] - mesh.bounds[0]) / 2.0         # demi-extents (mesh centre)

    # poignets monde -> repere LOCAL table
    wrists_w = d["body_pos_w"][:, a_idx, :]                 # (T, H, 3)
    H = wrists_w.shape[1]
    local = np.einsum("ij,thj->thi", R.T, wrists_w - sup_p)  # (T,H,3) repere table
    q = local.reshape(-1, 3)
    closest, dist, _ = mesh.nearest.on_surface(q)          # witness local + distance non signee
    dist = dist.reshape(T, H)
    witness = closest.reshape(T, H, 3)

    anchor = dist.argmin(axis=1).astype(np.int64)          # (T,) poignet le plus proche
    ti = np.arange(T)
    d_sel = dist[ti, anchor]                                # (T,)
    w_sel = witness[ti, anchor]                             # (T,3) local
    contact = d_sel < args.thr

    print(f"[support-contact] anchors={anchors} idx={a_idx}  table half-extents={np.round(half,3).tolist()}")
    print(f"[support-contact] dist main<->table : min={d_sel.min():.3f} p10={np.percentile(d_sel,10):.3f} "
          f"med={np.median(d_sel):.3f} max={d_sel.max():.3f}")
    print(f"[support-contact] seuil {args.thr} -> {int(contact.sum())}/{T} frames en contact "
          f"({100*contact.mean():.1f}%)  fenetre=[{np.where(contact)[0].min() if contact.any() else -1}.."
          f"{np.where(contact)[0].max() if contact.any() else -1}]")

    # nuage de surface (repere LOCAL table) pour obs/viz/futur
    cloud, _ = trimesh.sample.sample_surface(mesh, args.n_cloud)

    d["support_ref_contact"] = contact
    d["support_ref_contact_dist"] = d_sel.astype(np.float32)
    d["support_ref_anchor_idx"] = anchor
    d["support_ref_witness_local"] = w_sel.astype(np.float32)
    d["support_half_extents"] = half.astype(np.float32)
    d["support_mass"] = np.array(50.0, dtype=np.float32)   # table lourde/statique
    d["support_pointcloud"] = np.asarray(cloud, np.float32)

    out = args.out or args.motion.replace(".npz", "_nobj.npz")
    np.savez(out, **d)
    print(f"[support-contact] -> {out}")


if __name__ == "__main__":
    main()
