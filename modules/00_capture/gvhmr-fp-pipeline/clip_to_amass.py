#!/usr/bin/env python3
"""capture .pt (GVHMR/HaMeR SMPL-X) -> HoloV2 "amass" npz (raw SMPL-X params, Z-up) + objet.

La nouvelle version de HoloV2 a remplacé le loader `sfu` (positions/orientations globales par
joint) par un loader `amass` qui lit le format AMASS SMPL-X BRUT : root_orient, pose_body,
pose_hand, trans, betas(16), mocap_frame_rate, surface_model_type. C'est en fait plus proche de
notre .pt (smpl_params_global = déjà des axis-angle locaux), il suffit de :
  - snapper l'axe gravité de GVHMR sur +Z (même construction que clip_to_sfu / view_smpl_viser),
  - poser les pieds ~z=0 (un shift constant, comme clip_to_sfu ; le retargeting re-calibre le sol),
  - composer ce snap dans root_orient et la position monde de la racine -> trans AMASS,
  - padder les betas 10 -> 16 (GVHMR n'en régresse que 10 ; les 6 hauts = 0, forme inchangée),
  - pose_hand = 0 (le corps GVHMR n'a pas les doigts).

Avec --clip clip.npz, embarque aussi l'objet (object_poses (T,7) pos-first wxyz + object_mesh),
dans le MÊME repère Z-up que le corps, pour que le loader amess le fournisse au retargeter.

Run in an env with smplx + torch + scipy (holonew).
"""
import argparse
import os

import numpy as np


def rot_to_z(a):
    """Rotation mappant le vecteur unitaire a sur +Z (même construction que clip_to_sfu)."""
    a = a / (np.linalg.norm(a) + 1e-9)
    b = np.array([0.0, 0.0, 1.0])
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if c > 1 - 1e-8:
        return np.eye(3)
    if c < -1 + 1e-8:
        return np.diag([1.0, -1.0, -1.0])
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1 + c)


def _stretch_grid(T, factor):
    """Grille de rééchantillonnage temporel : T' = round((T-1)*factor)+1 frames RÉGULIÈRES sur la
    même plage [0, T-1] -> même mouvement mais étiré dans le temps (durée réelle × facteur, ex 1.7 =
    1.7x plus lent). Retourne (i0, i1, w, T') — même convention que resample_motion (interp linéaire
    des indices sources encadrants)."""
    Tp = max(2, int(round((T - 1) * factor)) + 1)
    t = np.linspace(0, T - 1, Tp)
    i0 = np.floor(t).astype(int)
    i1 = np.minimum(i0 + 1, T - 1)
    return i0, i1, (t - i0), Tp


def _warp_grid(T, base_factor, segments):
    """Grille de warp PAR SEGMENTS : facteur local f sur [s,e) (frames SOURCE), base_factor
    ailleurs. Ex : marche resynthétisée (gait_opt) gardée ~vitesse réelle (1.25) pendant que
    prise/dépôt restent ralentis (1.6). Temps de sortie = cumul des facteurs par intervalle,
    inversé par interp monotone -> même convention de retour que _stretch_grid."""
    loc = np.full(T - 1, float(base_factor))
    for (s, e, f) in segments:
        loc[max(0, s):min(T - 1, e)] = f
    ct = np.concatenate([[0.0], np.cumsum(loc)])
    Tp = max(2, int(round(ct[-1])) + 1)
    src = np.interp(np.linspace(0.0, ct[-1], Tp), ct, np.arange(T, dtype=float))
    i0 = np.floor(src).astype(int)
    i1 = np.minimum(i0 + 1, T - 1)
    return i0, i1, (src - i0), Tp


def _stretch_lerp(arr, i0, i1, w):
    """arr (T, C) -> (T', C) par lerp linéaire (positions)."""
    return arr[i0] * (1.0 - w[:, None]) + arr[i1] * w[:, None]


def _stretch_slerp_rotvec_blocks(arr, i0, i1, w):
    """arr (T, 3k) rotvecs empilés par blocs de 3 colonnes (k joints) -> slerp bloc par bloc,
    vectorisé sur toutes les colonnes (même construction que resample_motion de HoloV2)."""
    from scipy.spatial.transform import Rotation
    T, C = arr.shape
    J = C // 3
    a = arr.reshape(T, J, 3)
    r0 = Rotation.from_rotvec(a[i0].reshape(-1, 3))
    r1 = Rotation.from_rotvec(a[i1].reshape(-1, 3))
    rel = (r0.inv() * r1).as_rotvec()
    ww = np.repeat(w, J)[:, None]
    return (r0 * Rotation.from_rotvec(rel * ww)).as_rotvec().reshape(len(i0), C)


def _stretch_slerp_quat_wxyz(arr, i0, i1, w):
    """arr (T,4) quaternion wxyz -> slerp -> (T',4) wxyz."""
    from scipy.spatial.transform import Rotation
    r0 = Rotation.from_quat(arr[i0][:, [1, 2, 3, 0]])          # wxyz -> xyzw (scipy)
    r1 = Rotation.from_quat(arr[i1][:, [1, 2, 3, 0]])
    rel = (r0.inv() * r1).as_rotvec()
    out = (r0 * Rotation.from_rotvec(rel * w[:, None])).as_quat()   # xyzw
    return out[:, [3, 0, 1, 2]]                                  # -> wxyz


def table_mesh(w, d, h, top_t=0.04, leg=0.05):
    """Table 4 pieds, plateau w×d, hauteur h (du sol au DESSUS). Mesh centré sur le centre de sa
    boîte englobante (z ∈ [−h/2, +h/2], dessus du plateau à +h/2), axe X = w, axe Y = d : la pose
    d'injection place +h/2 à la hauteur voulue et applique le yaw. Pieds carrés aux 4 coins (rentrés)."""
    import trimesh
    parts = [trimesh.creation.box(extents=[w, d, top_t]).apply_translation([0, 0, h / 2 - top_t / 2])]
    inset = leg / 2 + 0.02
    leg_h = h - top_t
    for sx in (-1, 1):
        for sy in (-1, 1):
            parts.append(trimesh.creation.box(extents=[leg, leg, leg_h]).apply_translation(
                [sx * (w / 2 - inset), sy * (d / 2 - inset), -h / 2 + leg_h / 2]))
    return trimesh.util.concatenate(parts)


def main():
    ap = argparse.ArgumentParser(description="GVHMR .pt -> HoloV2 amass npz (+ object)")
    ap.add_argument("--pt", required=True, help="capture .pt (wrist/contact-corrected recommended)")
    ap.add_argument("--out", required=True, help="output amass npz")
    ap.add_argument("--model-dir", required=True, help="parent of smplx/ model folder")
    ap.add_argument("--gender", default="neutral")
    ap.add_argument("--clip", default="", help="fuse clip.npz -> also embed the object (same Z-up frame)")
    ap.add_argument("--object-mesh", default="", help="override object mesh path (else from clip.npz)")
    ap.add_argument("--smooth", type=int, default=0, help="lissage Savitzky-Golay (fenêtre impaire, 0=off) du "
                    "corps entier (root_orient/pose_body/trans) -> mouvement retargeté plus lisse")
    ap.add_argument("--ground", default="feet-median", choices=["feet-median", "per-frame"],
                    help="feet-median (défaut) = un seul décalage Z constant (médiane du pied le plus bas sur "
                    "tout le clip) ; per-frame = décalage Z par frame basé sur le point le plus bas du MESH "
                    "entier (mains/genoux/dos compris), pour les mouvements au sol (allongé -> assis -> genoux "
                    "-> debout) où un décalage constant fait flotter ou pénétrer le sol selon la phase")
    ap.add_argument("--ground-smooth", type=int, default=9, help="fenêtre Savitzky-Golay (impaire) du lissage "
                    "temporel du décalage sol en mode per-frame (0=off) ; le décalage lissé est toujours plafonné "
                    "au minimum brut de la frame pour garantir zéro pénétration")
    ap.add_argument("--slow-factor", type=float, default=1.0, help="étire le TEMPS du clip d'un facteur "
                    "(ex 1.7 = 1.7x plus lent) : interp slerp/lerp du corps ET des objets sur une grille "
                    "temporelle dilatée, mocap_frame_rate INCHANGÉ (30) -> corrige une capture dont le fps "
                    "réel était plus bas que le 30 déclaré (mouvement rejoué trop vite)")
    ap.add_argument("--warp", default="", help="facteurs par SEGMENTS 's:e:f[,s:e:f...]' (frames source) ; "
                    "--slow-factor s'applique ailleurs. Ex --slow-factor 1.6 --warp 95:132:1.25 = "
                    "prise/dépôt 1.6x plus lents, marche 1.25x seulement")
    # support statique (2e objet) dérivé de la caisse posée en fin de clip
    ap.add_argument("--support", action="store_true", help="ajoute un support STATIQUE sous la caisse posée")
    ap.add_argument("--support-shape", default="box", choices=["box", "table"],
                    help="box = boîte pleine ; table = plateau 4 pieds (défaut 0.90x0.60) orienté sur le cap robot")
    ap.add_argument("--support-size", default="", help="empreinte WxD du plateau (m), ex 0.9x0.6 ; défaut = table 0.9x0.6 / box caisse+marge")
    ap.add_argument("--support-yaw-offset", type=float, default=0.0, help="rotation en degrés ajoutée au cap (ex 90 pour tourner la table d'un quart de tour)")
    ap.add_argument("--support-margin", type=float, default=0.08, help="marge autour de l'empreinte caisse si --support-size absent (m)")
    ap.add_argument("--support-still", type=float, default=0.003, help="vitesse caisse sous laquelle elle est 'posée' (m/frame)")
    ap.add_argument("--support-rest-frames", type=int, default=15, help="frames de fin agrégées pour la pose de repos")
    args = ap.parse_args()

    import smplx
    import torch
    from scipy.spatial.transform import Rotation

    pt = torch.load(args.pt, map_location="cpu", weights_only=False)
    p = pt["smpl_params_global"]
    betas = np.asarray(p["betas"], np.float32)
    T = np.asarray(p["transl"]).shape[0]
    nb = betas.shape[1]

    go = np.asarray(p["global_orient"], np.float32).reshape(T, 3)
    bp = np.asarray(p["body_pose"], np.float32).reshape(T, 63)
    tr = np.asarray(p["transl"], np.float32).reshape(T, 3)

    model = smplx.create(args.model_dir, model_type="smplx", gender=args.gender, num_betas=nb,
                         use_pca=False, flat_hand_mean=True, batch_size=T)
    with torch.no_grad():
        out = model(betas=torch.as_tensor(betas, dtype=torch.float32),
                    global_orient=torch.as_tensor(go, dtype=torch.float32),
                    body_pose=torch.as_tensor(bp, dtype=torch.float32),
                    transl=torch.as_tensor(tr, dtype=torch.float32))
    joints = out.joints.numpy()[:, :22]                        # (T,22,3) GVHMR native world
    # pelvis rest natif (pose zéro) : le "j0" que le loader amass ajoute à trans pour la racine
    model1 = smplx.create(args.model_dir, model_type="smplx", gender=args.gender, num_betas=nb,
                          use_pca=False, flat_hand_mean=True, batch_size=1)
    with torch.no_grad():
        j0 = model1(betas=torch.as_tensor(betas[:1], dtype=torch.float32)).joints.numpy()[0, 0]  # (3,)

    # snap gravité GVHMR -> +Z (head - pelvis, moyenné)
    raw_up = (joints[:, 15] - joints[:, 0]).mean(0)
    ax = int(np.argmax(np.abs(raw_up)))
    up = np.zeros(3); up[ax] = np.sign(raw_up[ax])
    Rsnap = rot_to_z(up)

    pos_snap = np.einsum("ij,tnj->tni", Rsnap, joints)
    if args.ground == "per-frame":
        # décalage sol PAR FRAME : point le plus bas du mesh entier (mains/genoux/dos au sol
        # comptent, pas seulement les pieds) -> suit le vrai contact au sol frame par frame.
        # Pas de lissage ICI (un savgol sur ce signal peut osciller/déborder près des transitions
        # rapides, ex allongé -> genoux) : on applique la pose brute, lisse le corps ensuite comme
        # d'habitude, puis on RE-PLAQUE exactement (voir plus bas, après le lissage --smooth).
        verts_snap = np.einsum("ij,tnj->tni", Rsnap, out.vertices.numpy())
        floor_shift = verts_snap[:, :, 2].min(axis=1)                   # (T,)
        print(f"[amass] sol per-frame : décalage brut [{floor_shift.min():.3f},{floor_shift.max():.3f}]")
    else:
        # floor shift : médiane du pied le plus bas (joints 10/11) APRÈS snap
        floor_shift = float(np.percentile(pos_snap[:, [10, 11], 2].min(axis=1), 50))

    # root_orient Z-up = Rsnap ∘ global_orient ; trans = pos_racine_Zup - j0
    root_R = Rotation.from_rotvec(go).as_matrix()                       # (T,3,3)
    root_orient = Rotation.from_matrix(np.einsum("ij,tjk->tik", Rsnap, root_R)).as_rotvec()
    root_pos_zup = np.einsum("ij,tj->ti", Rsnap, joints[:, 0])          # pelvis monde Z-up
    root_pos_zup[:, 2] -= floor_shift
    trans = root_pos_zup - j0                                           # convention loader amass

    # lissage temporel du corps ENTIER (avant export) : Savitzky-Golay (ordre 2) sur root_orient,
    # pose_body et trans -> le retargeting suit un mouvement plus doux. Fenêtre bornée à T (impaire).
    if args.smooth and args.smooth > 3:
        from scipy.signal import savgol_filter
        w = args.smooth + (1 - args.smooth % 2)                        # rendre impair
        w = min(w, T if T % 2 == 1 else T - 1)
        if w > 3:
            root_orient = savgol_filter(root_orient, w, 2, axis=0)
            bp = savgol_filter(bp, w, 2, axis=0)
            trans = savgol_filter(trans, w, 2, axis=0)
            print(f"[amass] corps lissé (Savitzky-Golay fenêtre {w})")

    if args.ground == "per-frame":
        # RE-PLAQUAGE exact au sol APRÈS le lissage du corps (le lissage de la pose peut décoller
        # ou enfoncer légèrement le mesh du sol par rapport au calcul fait sur la pose brute,
        # surtout près des transitions rapides) : refait un forward avec la pose lissée et corrige
        # trans_z du résidu exact, sans lissage additionnel (évite tout débordement de savgol).
        with torch.no_grad():
            out2 = model(betas=torch.as_tensor(betas, dtype=torch.float32),
                        global_orient=torch.as_tensor(root_orient, dtype=torch.float32),
                        body_pose=torch.as_tensor(bp, dtype=torch.float32),
                        transl=torch.as_tensor(trans, dtype=torch.float32))
        resid = out2.vertices.numpy()[:, :, 2].min(axis=1)
        trans = trans.copy()
        trans[:, 2] -= resid
        print(f"[amass] sol per-frame : résidu post-lissage [{resid.min():.4f},{resid.max():.4f}] -> re-plaqué")

    betas16 = np.zeros(16, np.float32)
    betas16[:min(nb, 16)] = betas[0, :min(nb, 16)]

    save = dict(
        surface_model_type="smplx",
        mocap_frame_rate=np.float32(30.0),
        gender=args.gender,
        betas=betas16,
        root_orient=root_orient.astype(np.float32),        # (T,3) Z-up
        pose_body=bp.astype(np.float32),                   # (T,63) local, frame-invariant
        pose_hand=np.zeros((T, 90), np.float32),           # pas de doigts (corps GVHMR)
        trans=trans.astype(np.float32),                    # (T,3)
    )

    if args.clip:
        c = np.load(args.clip, allow_pickle=True)
        OP = np.asarray(c["object_pose"], np.float64)      # (To,4,4) GVHMR global world
        Tm = min(T, OP.shape[0])
        if Tm != T:                                        # corps ET objet alignés (le loader exige T==)
            print(f"[amass] body {T} vs object {OP.shape[0]} frames -> tout tronqué à {Tm}")
            for k in ("root_orient", "pose_body", "pose_hand", "trans"):
                save[k] = save[k][:Tm]
        box_mesh = args.object_mesh or str(c["object_mesh"])
        o_pos = np.einsum("ij,tj->ti", Rsnap, OP[:Tm, :3, 3])
        o_rot = np.einsum("ij,tjk->tik", Rsnap, OP[:Tm, :3, :3])
        fs_obj = floor_shift[:Tm] if isinstance(floor_shift, np.ndarray) else floor_shift
        o_pos[:, 2] -= fs_obj
        o_quat = Rotation.from_matrix(o_rot).as_quat()[:, [3, 0, 1, 2]]
        box7 = np.concatenate([o_pos, o_quat], axis=1).astype(np.float32)   # (Tm,7) caisse Z-up

        poses_l, mesh_l, name_l = [box7], [box_mesh], ["box"]

        if args.support:
            import trimesh
            Vbox = np.asarray(trimesh.load(box_mesh, force="mesh", process=False).vertices)
            # phase de repos en fin : la caisse ne bouge plus (elle est posée sur le support)
            spd = np.linalg.norm(np.diff(box7[:, :3], axis=0), axis=1)
            mv = np.where(spd > args.support_still)[0]
            rest0 = int(mv.max()) + 1 if len(mv) else 0
            rest = np.arange(max(rest0, Tm - args.support_rest_frames), Tm)
            if len(rest) == 0:
                rest = np.arange(max(0, Tm - 5), Tm)
            # fond + centre xy de la caisse au repos (repère amass Z-up) -> dessus/centre du support
            Rr = Rotation.from_quat(box7[:, [4, 5, 6, 3]]).as_matrix()      # wxyz -> xyzw
            top_z = float(np.median([(Vbox @ Rr[f].T + box7[f, :3])[:, 2].min() for f in rest]))
            sx, sy = float(np.median(box7[rest, 0])), float(np.median(box7[rest, 1]))
            if args.support_size:
                w, dep = (float(x) for x in args.support_size.lower().split("x"))
            elif args.support_shape == "table":
                w, dep = 0.90, 0.60                                         # défaut table
            else:
                w = float(np.ptp(Vbox[:, 0])) + 2 * args.support_margin     # box = empreinte caisse + marge
                dep = float(np.ptp(Vbox[:, 1])) + 2 * args.support_margin
            h = max(top_z, 0.02)                                           # du sol (z=0) au dessous de la caisse
            # yaw = cap du robot (heading des épaules, joints 16/17) au repos -> table orientée vers lui
            yaw = 0.0
            if args.support_shape == "table":
                shL = np.median(pos_snap[rest, 16], axis=0)
                shR = np.median(pos_snap[rest, 17], axis=0)
                lateral = shR - shL; lateral[2] = 0.0                       # épaule L->R (horizontal)
                fwd = np.cross([0.0, 0.0, 1.0], lateral)                    # avant du corps = up × latéral
                n = float(np.linalg.norm(fwd))
                if n > 1e-6:
                    yaw = float(np.arctan2(fwd[1] / n, fwd[0] / n))
                mesh_obj = table_mesh(w, dep, h)                           # plateau 4 pieds (axe X=w le long de l'avant)
            else:
                mesh_obj = trimesh.creation.box(extents=[w, dep, h])       # boîte pleine centrée
            yaw += np.radians(args.support_yaw_offset)                     # décalage manuel (ex +90°)
            sup_path = os.path.join(os.path.dirname(box_mesh) or ".", "support.obj")
            mesh_obj.export(sup_path)
            quat = Rotation.from_euler("z", yaw).as_quat()[[3, 0, 1, 2]]    # wxyz
            # pose statique : mesh centré -> translation = (sx, sy, h/2), rot = yaw ; dessus du plateau à z=h
            sup7 = np.tile(np.concatenate([[sx, sy, h / 2.0], quat]).astype(np.float32), (Tm, 1))
            poses_l.append(sup7); mesh_l.append(sup_path); name_l.append("support")
            print(f"[amass] + support ({args.support_shape}): dessus z={top_z:.2f} m, centre ({sx:.2f},{sy:.2f}), "
                  f"{w:.2f}x{dep:.2f}x{h:.2f} m, yaw {np.degrees(yaw):.0f}°, repos [{int(rest[0])}:{int(rest[-1])}]")

        save["object_poses"] = np.stack(poses_l, 0).astype(np.float32)     # (N, Tm, 7)
        save["object_meshes"] = np.array(mesh_l)
        save["object_names"] = np.array(name_l)
        print(f"[amass] + {len(poses_l)} objet(s) embarqué(s): {name_l}")

    if args.slow_factor != 1.0 or args.warp:
        Torig = save["trans"].shape[0]
        if args.warp:
            segs = [tuple(float(x) if i == 2 else int(x) for i, x in enumerate(sp.split(":")))
                    for sp in args.warp.split(",")]
            i0, i1, w, Tp = _warp_grid(Torig, args.slow_factor, segs)
            print(f"[amass] warp segments {segs} (base x{args.slow_factor:.2f})")
        else:
            i0, i1, w, Tp = _stretch_grid(Torig, args.slow_factor)
        save["root_orient"] = _stretch_slerp_rotvec_blocks(save["root_orient"], i0, i1, w).astype(np.float32)
        save["pose_body"] = _stretch_slerp_rotvec_blocks(save["pose_body"], i0, i1, w).astype(np.float32)
        save["pose_hand"] = _stretch_slerp_rotvec_blocks(save["pose_hand"], i0, i1, w).astype(np.float32)
        save["trans"] = _stretch_lerp(save["trans"], i0, i1, w).astype(np.float32)
        if "object_poses" in save:
            op = save["object_poses"]                                     # (N, Torig, 7)
            new_op = np.zeros((op.shape[0], Tp, 7), np.float32)
            for n in range(op.shape[0]):
                new_op[n, :, :3] = _stretch_lerp(op[n, :, :3], i0, i1, w)
                new_op[n, :, 3:7] = _stretch_slerp_quat_wxyz(op[n, :, 3:7], i0, i1, w)
            save["object_poses"] = new_op
        print(f"[amass] ralenti x{args.slow_factor:.2f} : {Torig} -> {Tp} frames (mocap_frame_rate={float(save['mocap_frame_rate']):.0f} inchangé)")

    np.savez(args.out, **save)
    print(f"[amass] wrote {args.out}  T={T}  betas16 (10 réels + zéros)  root_orient/pose_body/trans Z-up")


if __name__ == "__main__":
    main()
