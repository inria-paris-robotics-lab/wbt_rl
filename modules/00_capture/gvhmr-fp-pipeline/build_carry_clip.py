#!/usr/bin/env python3
"""Construit un clip de portage PROPRE — ÉTAPE 1/2 : la CAISSE.

Garde la LOCOMOTION captée (root/jambes/torse/tête du .pt) et remplace la trajectoire de la caisse
par une trajectoire AUTORÉE, propre :
  - caisse FACE AU SOL (une face à plat sur le sol) à la prise et à la repose,
  - levée -> portée -> reposée, ancrée DEVANT le corps (offset avant + vertical = ceux du portage réel,
    ou --fwd-off/--vert-off), à hauteur constante pendant le portage,
  - orientation UPRIGHT (axe vertical de la caisse = up du monde), yaw = cap du corps -> alignée pile.

Sortie : clip.npz (corps capté + caisse propre) — visualisable par view_clip_viser, exportable par
clip_to_amass. À l'ÉTAPE 2, les bras seront mis en IK symétrique sur cette caisse (les bras captés ne
la grippent pas encore pile). Run env holonew (smplx + torch + trimesh + scipy).
"""
import argparse

import numpy as np


def _phase_frames(fp_dir, T, thr=0.005, smooth=5):
    """(grab, place) depuis la vitesse de la caisse FoundationPose : 1re/dernière frame en mouvement."""
    import glob
    import os
    fps = sorted(glob.glob(os.path.join(fp_dir, "*.txt")))
    t = np.stack([np.loadtxt(f).reshape(4, 4)[:3, 3] for f in fps]).astype(np.float64)[:T]
    v = np.linalg.norm(np.diff(t, axis=0, prepend=t[:1]), axis=1)
    v = np.convolve(v, np.ones(smooth) / smooth, mode="same")
    mv = np.where(v > thr)[0]
    return (int(mv.min()), int(mv.max())) if len(mv) else (T // 3, 2 * T // 3)


def main():
    ap = argparse.ArgumentParser(description="Construire un clip de portage propre (caisse) — étape 1/2")
    ap.add_argument("--pt", required=True, help="GVHMR .pt (locomotion à garder)")
    ap.add_argument("--out", required=True, help="clip.npz de sortie")
    ap.add_argument("--model-dir", required=True, help="parent de smplx/")
    ap.add_argument("--gender", default="neutral")
    ap.add_argument("--box-size", type=float, default=0.36, help="côté de la caisse (m)")
    ap.add_argument("--object-mesh", default="", help="mesh caisse (défaut: ~/wbt_clips/box{size}.obj sinon cube)")
    ap.add_argument("--fp-poses", default="", help="ob_in_cam/ pour auto-détecter la prise/repose + l'ancrage")
    ap.add_argument("--grab", type=int, default=-1, help="frame de prise (sinon auto via --fp-poses)")
    ap.add_argument("--place", type=int, default=-1, help="frame de repose (sinon auto)")
    ap.add_argument("--fwd-off", type=float, default=None, help="offset avant caisse/poitrine (m ; sinon 0.38)")
    ap.add_argument("--vert-off", type=float, default=None, help="offset vertical caisse/poitrine (m ; sinon -0.30)")
    ap.add_argument("--grab-height", type=float, default=-1.0, help="hauteur du BAS de la caisse à la prise (m ; défaut = sol)")
    ap.add_argument("--place-height", type=float, default=-1.0, help="hauteur du BAS de la caisse à la repose (m ; ex 0.745 pour une table)")
    ap.add_argument("--lift-dur", type=int, default=15, help="frames de levée")
    ap.add_argument("--lower-dur", type=int, default=15, help="frames de repose")
    ap.add_argument("--smooth", type=int, default=9, help="lissage de la trajectoire caisse (moyenne glissante)")
    # --- étape 2 : IK des bras (grip symétrique de la caisse) ---
    ap.add_argument("--ik-arms", action="store_true", help="ÉTAPE 2 : mettre les bras en IK pour gripper la caisse")
    ap.add_argument("--out-pt", default="", help="sortie .pt (corps + bras IK) pour clip_to_amass (sinon <out>.pt)")
    ap.add_argument("--wrist-gap", type=float, default=0.06, help="distance poignet-face (m ; paume à plat sur la face)")
    ap.add_argument("--ik-iters", type=int, default=300)
    ap.add_argument("--ik-edge-ramp", type=int, default=8, help="fondu vers les bras captés à la prise/relâche (arc large)")
    ap.add_argument("--settle-frames", type=int, default=8, help="frames où les mains tiennent la caisse POSÉE sur la table avant de la lâcher")
    ap.add_argument("--release-frame", type=int, default=-1, help="frame où le contact doit être NUL (l'arc de relâche s'y termine) ; -1 = auto (pose+settle)")
    args = ap.parse_args()

    import os
    import torch
    import smplx
    import trimesh
    from scipy.spatial.transform import Rotation

    pt = torch.load(args.pt, map_location="cpu", weights_only=False)
    pgl = pt["smpl_params_global"]
    T = np.asarray(pgl["transl"]).shape[0]
    nb = np.asarray(pgl["betas"]).shape[1]
    go = np.asarray(pgl["global_orient"], np.float32).reshape(T, 3)
    bp = np.asarray(pgl["body_pose"], np.float32).reshape(T, 63)
    tr = np.asarray(pgl["transl"], np.float32).reshape(T, 3)
    betas = np.asarray(pgl["betas"], np.float32)

    model = smplx.create(args.model_dir, model_type="smplx", gender=args.gender, num_betas=nb,
                         use_pca=False, flat_hand_mean=True, batch_size=T)
    with torch.no_grad():
        J = model(betas=torch.as_tensor(betas), global_orient=torch.as_tensor(go),
                  body_pose=torch.as_tensor(bp), transl=torch.as_tensor(tr)).joints.numpy()

    # repère : axe UP du monde GVHMR (tête - bassin), avant/latéral du corps par frame
    up_raw = (J[:, 15] - J[:, 0]).mean(0)
    up = np.zeros(3); up[int(np.argmax(np.abs(up_raw)))] = np.sign(up_raw[int(np.argmax(np.abs(up_raw)))])
    chest = (J[:, 16] + J[:, 17]) / 2.0
    lateral = J[:, 17] - J[:, 16]                                   # épaule G -> D
    fwd = np.cross(np.broadcast_to(up, lateral.shape), lateral)
    fwd /= np.linalg.norm(fwd, axis=1, keepdims=True) + 1e-9        # avant du corps (horizontal)
    lat = np.cross(fwd, np.broadcast_to(up, fwd.shape))             # ré-orthogonalisé (horizontal)
    lat /= np.linalg.norm(lat, axis=1, keepdims=True) + 1e-9
    ground = float(np.median(np.einsum("tij,j->ti", J[:, [10, 11]], up).min(axis=1)))  # niveau des pieds
    half = args.box_size / 2.0

    # timing prise/repose
    if args.fp_poses and (args.grab < 0 or args.place < 0):
        grab, place = _phase_frames(args.fp_poses, T)
    else:
        grab = args.grab if args.grab >= 0 else T // 3
        place = args.place if args.place >= 0 else 2 * T // 3
    print(f"[build] up=axe{int(np.argmax(np.abs(up_raw)))} | prise={grab} repose={place} | sol(up)={ground:.2f}")

    # ancrage caisse vs poitrine (offsets avant / vertical)
    fwd_off = args.fwd_off if args.fwd_off is not None else 0.38
    vert_off = args.vert_off if args.vert_off is not None else -0.30

    # centre caisse PENDANT le portage (suit le corps), et positions AU SOL prise/repose
    carry = chest + fwd * fwd_off + up[None] * vert_off            # (T,3)

    grab_lvl = ground + args.grab_height if args.grab_height >= 0 else ground     # hauteur AU-DESSUS du sol
    place_lvl = ground + args.place_height if args.place_height >= 0 else ground

    def to_level(p, level):                                        # bas de la caisse posé à `level`, face à plat
        h = p - (p @ up) * up
        return h + up * (level + half)

    g_grab = to_level(carry[grab], grab_lvl)
    g_place = to_level(carry[place], place_lvl)
    print(f"[build] bas caisse : prise à {grab_lvl:.2f} m, repose à {place_lvl:.2f} m")

    center = np.zeros((T, 3))
    for t in range(T):
        if t < grab:
            center[t] = g_grab
        elif t < grab + args.lift_dur:
            s = (t - grab) / max(1, args.lift_dur)
            center[t] = (1 - s) * g_grab + s * carry[t]
        elif t < place - args.lower_dur:
            center[t] = carry[t]
        elif t < place:
            s = (t - (place - args.lower_dur)) / max(1, args.lower_dur)
            center[t] = (1 - s) * carry[t] + s * g_place
        else:
            center[t] = g_place

    if args.smooth > 1:
        k = np.ones(args.smooth) / args.smooth
        center = np.stack([np.convolve(np.pad(center[:, i], args.smooth // 2, mode="edge"), k, "valid")[:T]
                           for i in range(3)], axis=1)

    # orientation : upright (y_local = up), yaw = cap du corps (x_local = lat, z_local = fwd)
    Rbox = np.stack([lat, np.broadcast_to(up, fwd.shape), fwd], axis=2)   # colonnes [lat, up, fwd]
    # garantir une ROTATION propre (det +1) : [lat, up, fwd] est main gauche (lat×up = -fwd) et
    # scipy.from_matrix (clip_to_amass -> quaternion) refuse une réflexion. Caisse = cube ⇒ inverser
    # la colonne z_local ne change aucune face, rend juste le repère droitier.
    det = np.linalg.det(Rbox)
    Rbox[det < 0, :, 2] *= -1.0
    P = np.tile(np.eye(4), (T, 1, 1)).astype(np.float32)
    P[:, :3, :3] = Rbox
    P[:, :3, 3] = center

    # === ÉTAPE 2 : IK des DEUX bras pour gripper la caisse (faces latérales opposées), symétrique, arc aux bords ===
    if args.ik_arms:
        import sys as _sys
        _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from contact_opt import blend_from_span, build_sdf_grid, query_sdf
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        gh = args.box_size / 2.0
        RH = [21] + list(range(40, 55)); LH = [20] + list(range(25, 40))
        lbs = model.lbs_weights.detach().cpu().numpy()
        rhv = np.where(lbs[:, RH].sum(1) > 0.5)[0]                           # sommets paume DROITE
        lhv = np.where(lbs[:, LH].sum(1) > 0.5)[0]                           # sommets paume GAUCHE
        # maintien jusqu'à ce que la caisse soit POSÉE + settle (relâche après, pas pendant la descente).
        # --release-frame R impose : contact 0 à la frame R (le fondu = l'arc se termine pile à R).
        if args.release_frame >= 0:
            hold_end = max(grab, args.release_frame - args.ik_edge_ramp - 1)  # blend=0 à hold_end+ramp+1 = R
        else:
            hold_end = min(T - 1, place + args.settle_frames)
        blend = blend_from_span(T, grab, hold_end, args.ik_edge_ramp)        # 1 en portage+pose, arc capté aux bords
        rel = int(np.argmax(blend[hold_end:] <= 1e-6)) + hold_end if (blend[hold_end:] <= 1e-6).any() else T
        print(f"[ik] maintien plein jusqu'à {hold_end} ; contact nul (bras captés/arc) dès frame {rel}")
        lo, hi = max(0, grab - 15), min(T, hold_end + args.ik_edge_ramp + 5)
        win = np.arange(lo, hi); Nw = len(win)
        box_mesh = trimesh.creation.box(extents=[args.box_size] * 3)
        sdf_vol, sdf_lo, sdf_hi = build_sdf_grid(box_mesh, 48, 0.15, dev)
        Rbw = torch.as_tensor(P[win, :3, :3], dtype=torch.float32, device=dev)
        tbw = torch.as_tensor(P[win, :3, 3], dtype=torch.float32, device=dev)
        latw = torch.as_tensor(lat[win].astype(np.float32), device=dev)     # +latéral caisse (face droite)
        target_R = tbw + latw * (gh + args.wrist_gap)                       # poignet DROIT visé (face +lat)
        target_L = tbw - latw * (gh + args.wrist_gap)                       # poignet GAUCHE visé (face -lat)
        blend_w = torch.as_tensor(blend[win], dtype=torch.float32, device=dev)
        R_cols = [39, 40, 41, 48, 49, 50, 54, 55, 56, 60, 61, 62]           # bras D : collier/épaule/coude/poignet
        L_cols = [36, 37, 38, 45, 46, 47, 51, 52, 53, 57, 58, 59]           # bras G
        mask = torch.zeros(63, device=dev); mask[R_cols] = 1.0; mask[L_cols] = 1.0
        mir_sign = torch.tensor([1., -1, -1] * 4, device=dev)               # miroir sagittal des rotvecs (par joint)
        Rc = torch.tensor(R_cols, device=dev); Lc = torch.tensor(L_cols, device=dev)
        model_w = smplx.create(args.model_dir, model_type="smplx", gender=args.gender, num_betas=nb,
                               use_pca=False, flat_hand_mean=True, batch_size=Nw).to(dev)
        betas_w = torch.as_tensor(betas[win], device=dev)
        go_w = torch.as_tensor(go[win], device=dev); tr_w = torch.as_tensor(tr[win], device=dev)
        bp0w = torch.as_tensor(bp[win], device=dev)
        delta = torch.zeros((Nw, 63), device=dev, requires_grad=True)
        opt = torch.optim.Adam([delta], lr=0.02)
        bw = blend_w[:, None]

        def contact(verts, hv):                                             # patch paume le plus proche -> face
            hl = torch.einsum("tij,tnj->tni", Rbw.transpose(1, 2), verts[:, hv] - tbw[:, None, :])
            sdf = query_sdf(sdf_vol, sdf_lo, sdf_hi, hl.reshape(-1, 3)).view(Nw, -1)
            k = max(1, int(0.2 * sdf.shape[1]))
            near = torch.topk(sdf, k, dim=1, largest=False).values
            return (blend_w * torch.relu(near + 0.005).pow(2).mean(1)).sum() / (blend_w.sum() + 1e-6)

        for it in range(args.ik_iters):
            opt.zero_grad()
            d_app = delta * mask * bw                                       # fondu vers capté aux bords (arc large)
            out = model_w(betas=betas_w, global_orient=go_w, body_pose=bp0w + d_app, transl=tr_w)
            e_reach = ((blend_w * (out.joints[:, 21] - target_R).pow(2).sum(-1)).sum()
                       + (blend_w * (out.joints[:, 20] - target_L).pow(2).sum(-1)).sum()) / (2 * blend_w.sum() + 1e-6)
            e_contact = contact(out.vertices, rhv) + contact(out.vertices, lhv)
            poseR = bp0w[:, Rc] + d_app[:, Rc]; poseL = bp0w[:, Lc] + d_app[:, Lc]
            e_sym = (bw * (poseL - poseR * mir_sign).pow(2)).sum() / (12 * blend_w.sum() + 1e-6)  # bras symétriques
            e_smooth = (d_app[2:] - 2 * d_app[1:-1] + d_app[:-2]).pow(2).sum(-1).mean() if Nw > 2 else torch.zeros((), device=dev)
            loss = 10 * e_reach + 8 * e_contact + 3.0 * e_sym + 1.0 * d_app.pow(2).sum(-1).mean() + 4.0 * e_smooth
            loss.backward(); opt.step()
            if it % 100 == 0 or it == args.ik_iters - 1:
                print(f"[ik] it {it:3d}  reach~{100*float(e_reach)**0.5:.1f}cm  contact {float(e_contact):.4f}  sym {float(e_sym):.4f}")
        bp = bp.copy()
        bp[win] = (bp0w + delta * mask * bw).detach().cpu().numpy()
        print("[ik] 2 bras IK (faces ±lat) symétriques ; arc capté aux bords (prise + relâche)")

    mesh_path = args.object_mesh or os.path.expanduser(f"~/wbt_clips/box{int(args.box_size*100)}.obj")
    if not os.path.exists(mesh_path):
        mesh_path = os.path.join(os.path.dirname(args.out), "box_built.obj")
        trimesh.creation.box(extents=[args.box_size] * 3).export(mesh_path)

    np.savez(args.out,
             betas=betas, betas_mean=betas.mean(0).astype(np.float32),
             global_orient=go, body_pose=bp, trans=tr,
             poses=np.concatenate([go, bp], axis=1).astype(np.float32),
             mocap_framerate=np.float32(30.0), gender=args.gender, model_type="smplx",
             coord_frame="world",
             object_pose=P, object_R=P[:, :3, :3], object_t=P[:, :3, 3],
             object_valid=np.ones(T, bool), object_mesh=mesh_path)
    if args.ik_arms:
        out_pt = args.out_pt or (os.path.splitext(args.out)[0] + ".pt")
        for key in ("smpl_params_incam", "smpl_params_global"):
            pt[key]["body_pose"] = torch.as_tensor(bp, dtype=torch.float32)
        torch.save(pt, out_pt)
        print(f"[build] wrote {args.out} + {out_pt} (corps + bras IK, caisse {int(args.box_size*100)}) -> clip_to_amass --pt {os.path.basename(out_pt)}")
    else:
        print(f"[build] wrote {args.out}  (caisse propre, mesh={mesh_path}) — bras encore captés (étape 2 = --ik-arms)")


if __name__ == "__main__":
    main()
