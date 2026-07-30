#!/usr/bin/env python3
"""Replaque le SMPL global au sol, frame par frame — pour les mouvements avec contact sol variable
(allongé -> assis -> genoux -> debout) où GVHMR dérive en hauteur absolue et où un simple shift
constant (clip_to_amass.py --ground feet-median) laisse le corps flotter ou pénétrer le sol selon
la phase.

Par défaut, le décalage par frame est basé sur le point le plus bas du MESH ENTIER (--phases
absent). C'est insuffisant quand PLUSIEURS points devraient toucher en même temps (dos+fesses
allongé, fesses assise, genoux à quatre pattes/agenouillé) : le point le plus bas du mesh entier
tombe souvent sur une main ou un pied mal estimé, et la zone qui DEVRAIT toucher (dos, fesses,
genoux) reste décollée. --phases permet de restreindre, PAR SEGMENT DE FRAMES, la zone du corps
utilisée pour le calage (ex : les fesses pendant qu'il est assis, les genoux à quatre pattes) :

    --phases "0:42:back+butt+feet,42:66:butt,66:100:knees,100:140:knees+feet,140:177:feet"

Groupes dispos (rayon autour de joints SMPL-X en pose de repos, topologie fixe donc les indices de
vertex sélectionnés restent valides pose par pose) : back (spine1/2/3), butt (pelvis+hanches),
knees, feet (chevilles+pieds), hands (poignets), all (mesh entier, défaut hors segment couvert).

Exploite le fait que `transl` SMPL-X est un pur décalage uniforme (indépendant de la pose) :
  - détecte l'axe up natif GVHMR (tête - pelvis, moyenné, plus grande composante, comme
    clip_to_amass.py / view_smpl_viser.py),
  - pour chaque frame, translate le long de cet axe pour que le point le plus bas de la zone de
    contact ACTIVE (mesh entier par défaut, ou le(s) groupe(s) --phases de cette frame) touche
    exactement le sol (contact continu supposé : pas de phase aérienne dans ce type de mouvement),
  - lissage optionnel (Savitzky-Golay) de global_orient/body_pose/transl, puis RE-PLAQUAGE exact
    après lissage, avec la MÊME zone de contact (le lissage peut décoller/enfoncer le mesh du sol,
    surtout aux transitions rapides type "pousser sur les mains pour se relever"),
  - ne touche NI smpl_params_incam NI le reste du .pt (K_fullimg, net_outputs) : seul
    smpl_params_global est corrigé, sauvegardé dans le MÊME format .pt (compatible
    view_smpl_viser.py --coord global, clip_to_amass.py --pt, etc.).

Run env holonew (smplx + torch + scipy).
"""
import argparse

import numpy as np

# groupe -> (joints SMPL-X 22-joint concernés, rayon en m autour de leur position en pose de repos)
REGION_JOINTS = {
    "back": ([3, 6, 9], 0.15),          # spine1, spine2, spine3
    "butt": ([0, 1, 2], 0.14),          # pelvis, hanche G/D
    "knees": ([4, 5], 0.11),            # genou G/D
    "feet": ([7, 8, 10, 11], 0.11),     # cheville + pied G/D
    "hands": ([20, 21], 0.09),          # poignet G/D
}

# groupe -> indices body_pose (63,) qui PILOTENT ce groupe, pour la correction de pose secondaire
# --pose-ik (translation seule ne peut pas plaquer deux zones à des hauteurs différentes : il faut
# plier/redresser la chaîne cinématique correspondante). pelvis(0) est la racine (transl/global_orient,
# pas body_pose) : "butt" suit donc la racine directement, pas de DOF body_pose dédié -> à utiliser
# comme groupe PRIMAIRE (translation), jamais comme cible --pose-ik.
REGION_DOF = {
    "back": [6, 7, 8, 15, 16, 17, 24, 25, 26],                                       # spine1/2/3
    "knees": [0, 1, 2, 3, 4, 5, 9, 10, 11, 12, 13, 14],                              # hanches + genoux G/D
    "feet": [0, 1, 2, 3, 4, 5, 9, 10, 11, 12, 13, 14, 18, 19, 20, 21, 22, 23, 27, 28, 29, 30, 31, 32],
    "hands": [45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62],  # épaules+coudes+poignets G/D
}


def parse_phases(spec, T):
    """'s:e:g1+g2,s:e:g...' -> liste (start,end,[groupes]) triée ; None si spec vide."""
    if not spec:
        return None
    segs = []
    for chunk in spec.split(","):
        s, e, g = chunk.split(":")
        segs.append((int(s), min(int(e), T), g.split("+")))
    return segs


def parse_pose_ik(spec):
    """'region:s:e[,region:s:e...]' -> liste (region,start,end) ; [] si spec vide."""
    if not spec:
        return []
    out = []
    for chunk in spec.split(","):
        region, s, e = chunk.split(":")
        out.append((region, int(s), int(e)))
    return out


def smooth_rotvec(rotvec, w):
    """Lisse une séquence (T,3) de rotations axis-angle en passant par le quaternion, PAS le rotvec
    brut : un rotvec peut représenter la même rotation physique par deux vecteurs quasi opposés
    (double-cover / passage proche de +-pi), et un Savitzky-Golay composante-par-composante sur le
    rotvec brut moyenne alors deux orientations quasi opposées -> sortie n'importe quoi pile aux
    rotations rapides (observé : bascule assis -> quatre pattes, ~180° en peu de frames -> le corps
    "saute" debout un instant). Fix : quaternion + continuité de signe (dot consécutif > 0) avant
    lissage, renormalisé après.
    """
    from scipy.spatial.transform import Rotation
    q = Rotation.from_rotvec(rotvec).as_quat()               # (T,4) xyzw
    for t in range(1, len(q)):
        if np.dot(q[t], q[t - 1]) < 0:
            q[t] = -q[t]
    from scipy.signal import savgol_filter
    q_s = savgol_filter(q, w, 2, axis=0)
    q_s /= np.linalg.norm(q_s, axis=1, keepdims=True)
    return Rotation.from_quat(q_s).as_rotvec()


def parse_ranges(spec):
    """'s:e[,s:e...]' -> liste (start,end) ; [] si spec vide."""
    if not spec:
        return []
    out = []
    for chunk in spec.split(","):
        s, e = chunk.split(":")
        out.append((int(s), int(e)))
    return out


def main():
    ap = argparse.ArgumentParser(description="Replaque le SMPL global au sol frame par frame")
    ap.add_argument("--pt", required=True, help=".pt d'entrée (hmr4d_results.pt, smpl_params_global)")
    ap.add_argument("--out", required=True, help=".pt de sortie (même format, smpl_params_global replaqué/lissé)")
    ap.add_argument("--model-dir", required=True, help="parent de smplx/")
    ap.add_argument("--gender", default="neutral")
    ap.add_argument("--smooth", type=int, default=9, help="lissage Savitzky-Golay (fenêtre impaire, 0=off) de "
                    "global_orient/body_pose/transl")
    ap.add_argument("--phases", default="", help="'s:e:groupe[+groupe...][,s:e:groupe...]' zone de contact "
                    "par segment de frames (groupes: back,butt,knees,feet,hands,all). Frames hors segment -> all.")
    ap.add_argument("--pose-ik", default="", help="'region:s:e[,region:s:e...]' correction de POSE (pas "
                    "seulement translation) pour plaquer une 2e zone qui devrait toucher en même temps que "
                    "la zone --phases de ce segment (ex : le dos pendant que les fesses sont déjà au sol par "
                    "translation). Régions dispos : back, knees, feet (PAS butt : suit la racine, pas de DOF "
                    "dédié -> à mettre en primaire --phases). Optimise UNIQUEMENT les DOF body_pose qui "
                    "pilotent cette région, UNIQUEMENT sur ces frames, garde tout le reste intact.")
    ap.add_argument("--ik-iters", type=int, default=300)
    ap.add_argument("--ik-lr", type=float, default=0.03)
    ap.add_argument("--ik-w-target", type=float, default=200.0, help="poids : zone --pose-ik au sol (min proj = 0)")
    ap.add_argument("--ik-w-reg", type=float, default=1.0, help="poids : retour à la pose d'origine")
    ap.add_argument("--ik-w-smooth", type=float, default=20.0, help="poids : lissage temporel (accél angulaire)")
    ap.add_argument("--pen-ik", default="", help="'s:e[,s:e...]' passe(s) IK anti-pénétration GÉNÉRALE (tous "
                    "les DOF body_pose, pas une région nommée) : plie juste ce qu'il faut pour qu'AUCUN point du "
                    "mesh entier ne traverse le sol sur ce segment, sans bouger la translation ni deviner quelle "
                    "zone est fautive (contrairement à --pose-ik). Tourne APRÈS --pose-ik, dessus.")
    ap.add_argument("--pen-ik-iters", type=int, default=400)
    ap.add_argument("--pen-ik-lr", type=float, default=0.02)
    ap.add_argument("--pen-ik-w-pen", type=float, default=800.0, help="poids : pénétration résiduelle (mesh entier)")
    ap.add_argument("--pen-ik-w-reg", type=float, default=1.0, help="poids : retour à la pose d'origine")
    ap.add_argument("--pen-ik-w-smooth", type=float, default=20.0, help="poids : lissage temporel (accél angulaire)")
    args = ap.parse_args()

    import smplx
    import torch
    from scipy.signal import savgol_filter

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
    model1 = smplx.create(args.model_dir, model_type="smplx", gender=args.gender, num_betas=nb,
                          use_pca=False, flat_hand_mean=True, batch_size=1)

    def forward(go_, bp_, tr_):
        with torch.no_grad():
            out = model(betas=torch.as_tensor(betas, dtype=torch.float32),
                        global_orient=torch.as_tensor(go_, dtype=torch.float32),
                        body_pose=torch.as_tensor(bp_, dtype=torch.float32),
                        transl=torch.as_tensor(tr_, dtype=torch.float32))
        return out.joints.numpy()[:, :22], out.vertices.numpy()

    joints, verts = forward(go, bp, tr)
    V = verts.shape[1]

    # axe up natif GVHMR : tête - pelvis, moyenné, plus grande composante
    raw_up = (joints[:, 15] - joints[:, 0]).mean(0)
    ax = int(np.argmax(np.abs(raw_up)))
    sign = float(np.sign(raw_up[ax]))
    up = np.zeros(3, np.float32)
    up[ax] = sign
    print(f"[ground_smpl] axe up détecté : composante {ax} signe {sign:+.0f}")

    # index de vertex par groupe, sélectionnés en pose de REPOS (topologie fixe -> valides à toute
    # frame quelle que soit la pose posée ensuite)
    with torch.no_grad():
        rest = model1(betas=torch.as_tensor(betas[:1], dtype=torch.float32))
    rest_j, rest_v = rest.joints.numpy()[0, :22], rest.vertices.numpy()[0]
    region_idx = {"all": np.arange(V)}
    for name, (jids, radius) in REGION_JOINTS.items():
        mask = np.zeros(V, bool)
        for j in jids:
            mask |= np.linalg.norm(rest_v - rest_j[j], axis=1) < radius
        region_idx[name] = np.where(mask)[0]
        print(f"[ground_smpl] groupe '{name}' : {mask.sum()} vertices")

    phases = parse_phases(args.phases, T)
    if phases:
        frame_idx = [None] * T                    # T listes d'indices de vertex actifs
        covered = np.zeros(T, bool)
        for s, e, groups in phases:
            idx = np.unique(np.concatenate([region_idx[g] for g in groups]))
            for t in range(s, e):
                frame_idx[t] = idx
                covered[t] = True
        for t in range(T):
            if not covered[t]:
                frame_idx[t] = region_idx["all"]
        if not covered.all():
            print(f"[ground_smpl] {(~covered).sum()} frames hors --phases -> mesh entier (all)")
    else:
        frame_idx = [region_idx["all"]] * T

    def lowest_proj(v):                      # (T,V,3), zone active par frame -> (T,) point le plus bas
        return np.array([(v[t, frame_idx[t]] @ up).min() for t in range(T)])

    raw_shift = lowest_proj(verts)            # (T,)
    print(f"[ground_smpl] décalage brut [{raw_shift.min():.3f},{raw_shift.max():.3f}]")
    tr_grounded = tr - raw_shift[:, None] * up[None, :]

    go_out, bp_out, tr_out = go, bp, tr_grounded
    if args.smooth and args.smooth > 3:
        w = args.smooth + (1 - args.smooth % 2)                        # rendre impair
        w = min(w, T if T % 2 == 1 else T - 1)
        if w > 3:
            go_out = smooth_rotvec(go, w)
            bp_out = np.stack([smooth_rotvec(bp[:, 3 * j:3 * j + 3], w) for j in range(21)], axis=1).reshape(T, 63)
            tr_out = savgol_filter(tr_grounded, w, 2, axis=0)
            print(f"[ground_smpl] corps lissé (quaternion+Savitzky-Golay fenêtre {w})")

    # re-plaquage exact après lissage, MÊME zone de contact (le lissage peut décoller/enfoncer
    # localement le mesh du sol)
    _, verts2 = forward(go_out, bp_out, tr_out)
    resid = lowest_proj(verts2)
    tr_out = tr_out - resid[:, None] * up[None, :]
    print(f"[ground_smpl] résidu post-lissage [{resid.min():.4f},{resid.max():.4f}] -> re-plaqué")

    # --pose-ik : plie/redresse la chaîne cinématique d'une 2e zone NOMMÉE (ex : dos) pour qu'elle
    # touche elle aussi le sol, SANS bouger la translation (zone primaire --phases déjà plaquée
    # ci-dessus). --pen-ik : passe GÉNÉRALE (tous les DOF, pas de région à deviner) qui nettoie ce
    # qui reste après --pose-ik.
    pose_ik = parse_pose_ik(args.pose_ik)
    pen_ik = parse_ranges(args.pen_ik)
    if pose_ik or pen_ik:
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        model_ik = model.to(dev)
        betas_t = torch.as_tensor(betas, device=dev)
        go_t = torch.as_tensor(go_out, dtype=torch.float32, device=dev)
        tr_t = torch.as_tensor(tr_out, dtype=torch.float32, device=dev)
        bp_t = torch.as_tensor(bp_out, dtype=torch.float32, device=dev)
        up_t = torch.as_tensor(up, device=dev)

        for region, s, e in pose_ik:
            e = min(e, T)
            dof = REGION_DOF[region]
            vidx = torch.as_tensor(region_idx[region], device=dev)
            var = bp_t[s:e, dof].clone().requires_grad_(True)
            var0 = bp_t[s:e, dof].clone()
            opt = torch.optim.Adam([var], lr=args.ik_lr)
            print(f"[ground_smpl] pose-ik région '{region}' frames {s}:{e} ({len(dof)} DOF, "
                  f"{len(region_idx[region])} vertices)")
            for it in range(args.ik_iters):
                opt.zero_grad()
                bp_full = bp_t.clone()                             # (T,63) : modèle créé batch_size=T
                bp_full[s:e, dof] = var
                out = model_ik(betas=betas_t, global_orient=go_t, body_pose=bp_full, transl=tr_t)
                proj = out.vertices[s:e][:, vidx] @ up_t            # (e-s, len(vidx))
                l_target = (proj.min(dim=1).values ** 2).mean()
                l_reg = ((var - var0) ** 2).mean()
                l_smooth = ((var[2:] - 2 * var[1:-1] + var[:-2]) ** 2).mean() if e - s > 2 else var.new_zeros(())
                loss = args.ik_w_target * l_target + args.ik_w_reg * l_reg + args.ik_w_smooth * l_smooth
                loss.backward()
                opt.step()
                if it % 100 == 0 or it == args.ik_iters - 1:
                    print(f"  [ik {region}] {it:4d} loss {loss.item():.4f} target {l_target.item():.5f} "
                          f"reg {l_reg.item():.4f} smooth {l_smooth.item():.5f}")
            with torch.no_grad():
                bp_t[s:e, dof] = var

        # --pen-ik : anti-pénétration générale, TOUS les DOF body_pose du segment, pas de région
        # à nommer : le hinge-loss ne pénalise QUE ce qui passe sous le sol -> l'optimiseur ne
        # plie que ce qui doit vraiment l'être (la régularisation tire tout le reste vers la pose
        # déjà obtenue par --pose-ik ci-dessus).
        for s, e in pen_ik:
            e = min(e, T)
            var = bp_t[s:e].clone().requires_grad_(True)
            var0 = bp_t[s:e].clone()
            opt = torch.optim.Adam([var], lr=args.pen_ik_lr)
            print(f"[ground_smpl] pen-ik frames {s}:{e} (63 DOF, mesh entier)")
            for it in range(args.pen_ik_iters):
                opt.zero_grad()
                bp_full = bp_t.clone()
                bp_full[s:e] = var
                out = model_ik(betas=betas_t, global_orient=go_t, body_pose=bp_full, transl=tr_t)
                proj = out.vertices[s:e] @ up_t                    # (e-s, V)
                viol = torch.clamp(-proj.min(dim=1).values, min=0.0)   # pénétration (>=0), 0 si déjà au-dessus
                l_pen = (viol ** 2).mean()
                l_reg = ((var - var0) ** 2).mean()
                l_smooth = ((var[2:] - 2 * var[1:-1] + var[:-2]) ** 2).mean() if e - s > 2 else var.new_zeros(())
                loss = args.pen_ik_w_pen * l_pen + args.pen_ik_w_reg * l_reg + args.pen_ik_w_smooth * l_smooth
                loss.backward()
                opt.step()
                if it % 100 == 0 or it == args.pen_ik_iters - 1:
                    print(f"  [pen-ik] {it:4d} loss {loss.item():.4f} pen {l_pen.item():.5f} "
                          f"reg {l_reg.item():.4f} smooth {l_smooth.item():.5f}")
            with torch.no_grad():
                bp_t[s:e] = var

        bp_out = bp_t.detach().cpu().numpy()
        model = model.to("cpu")                 # model_ik.to(dev) mutates model in place

        # re-plaquage final : ni pose-ik ni pen-ik ne devraient bouger la zone PRIMAIRE (chaînes
        # cinématiques indépendantes de la racine), mais on re-vérifie par sécurité (léger blend
        # de skinning possible) sans re-lisser (l'IK a déjà sa propre régularisation/lissage).
        _, verts3 = forward(go_out, bp_out, tr_out)
        resid2 = lowest_proj(verts3)
        tr_out = tr_out - resid2[:, None] * up[None, :]
        print(f"[ground_smpl] résidu post-ik [{resid2.min():.4f},{resid2.max():.4f}] -> re-plaqué")

    # sécurité anti-pénétration GLOBALE (mesh entier) : une zone non ciblée par --phases/--pose-ik
    # (ex les pieds pendant une correction dos/fesses, ou une frame de transition rapide non
    # couverte) peut finir sous le sol -> on soulève juste ces frames-là du minimum nécessaire.
    # Ne retouche PAS les frames déjà propres (mesh min >= 0).
    _, verts_final = forward(go_out, bp_out, tr_out)
    global_min = (verts_final @ up).min(axis=1)
    pen = np.minimum(global_min, 0.0)
    if (pen < -1e-4).any():
        tr_out = tr_out - pen[:, None] * up[None, :]
        print(f"[ground_smpl] sécurité anti-pénétration : {int((pen < -1e-4).sum())} frames relevées "
              f"(max {-pen.min():.3f} m)")
        import os
        if os.environ.get("GROUND_SMPL_DEBUG"):
            bad = np.where(pen < -0.02)[0]
            for t in bad:
                vidx = int((verts_final[t] @ up).argmin())
                print(f"  [dbg] t={t} pen={pen[t]:.3f} argmin_vertex={vidx}")

    p["global_orient"] = torch.as_tensor(go_out.reshape(np.asarray(p["global_orient"]).shape), dtype=torch.float32)
    p["body_pose"] = torch.as_tensor(bp_out.reshape(np.asarray(p["body_pose"]).shape), dtype=torch.float32)
    p["transl"] = torch.as_tensor(tr_out.reshape(np.asarray(p["transl"]).shape), dtype=torch.float32)

    torch.save(pt, args.out)
    print(f"[ground_smpl] écrit {args.out}  T={T}")


if __name__ == "__main__":
    main()
