#!/usr/bin/env python3
"""Nettoyage foot-skate + resynthèse de pas — pendant PIEDS de contact_opt.py (mains).

Constat (femto14, systématique GVHMR) : les orteils ne décollent jamais (>6 cm) alors que la
vidéo montre de vrais pas ; les pieds GLISSENT (~0,2 m/s) pendant toute la marche. La policy
qui track cette réf fait des pas chassés. Ce script répare la réf au niveau SMPL :

  1. segmentation stand/marche sur la vitesse horizontale du pelvis ;
  2. phases STAND : chaque pied est ÉPINGLÉ à sa position médiane (fin du micro-glissé) ;
  3. phases MARCHE : plan de pas alternés le long du chemin du pelvis (longueur/cadence
     paramétrables), swing paraboliques (--swing-height), footholds raccordés aux stands
     adjacents -> zéro skate par construction ;
  4. IK jambes batch (Adam, autograd smplx) : hanches/genoux/chevilles/orteils seulement —
     root, torse, bras, mains, caisse INTACTS ;
  5. sortie .pt (smpl_params_global) + npz synchronisés + labels de stance par pied
     (foot_stance_l/r) pour le QC et d'éventuelles contraintes retarget.

Un stance épinglé reste propre quel que soit le retiming appliqué ensuite (clip_to_amass
--slow-factor / warp par segment). Run env holonew (smplx + torch + numpy + scipy).
"""

import argparse

import numpy as np

# body_pose (63,) : entrée 3*(j-1) pour le joint j de SMPL-X.
LEG_BP = {
    "L": [0, 1, 2, 9, 10, 11, 18, 19, 20, 27, 28, 29],   # left hip(1), knee(4), ankle(7), foot(10)
    "R": [3, 4, 5, 12, 13, 14, 21, 22, 23, 30, 31, 32],  # right hip(2), knee(5), ankle(8), foot(11)
}
J_ANKLE = {"L": 7, "R": 8}
J_TOE = {"L": 10, "R": 11}


def detect_up_axis(J):
    """Axe vertical = plus grande composante moyenne de pelvis->tête (joint 15)."""
    v = (J[:, 15] - J[:, 0]).mean(0)
    return int(np.argmax(np.abs(v))), float(np.sign(v[np.argmax(np.abs(v))]))


def smooth1d(x, w):
    """Moyenne glissante, padding par réflexion (le mode='same' zéro-padde -> fausses vitesses aux bords)."""
    if w <= 1:
        return x
    if x.ndim == 2:
        return np.stack([smooth1d(x[:, i], w) for i in range(x.shape[1])], 1)
    xp = np.pad(x, w // 2, mode="reflect")
    return np.convolve(xp, np.ones(w) / w, mode="valid")[: len(x)]


def segment_walk(speed, thr, min_len, fps):
    """Masque marche booléen : vitesse pelvis > thr, fermé/ouvert morphologiquement, segments courts fusionnés."""
    walk = speed > thr
    # fermeture : combler les trous < 0.2 s
    gap = int(0.2 * fps)
    idx = np.where(walk)[0]
    if len(idx) == 0:
        return walk
    for a, b in zip(idx[:-1], idx[1:]):
        if 1 < b - a <= gap:
            walk[a:b] = True
    # ouverture : retirer les segments < min_len
    out = np.zeros_like(walk)
    s = None
    for t in range(len(walk)):
        if walk[t] and s is None:
            s = t
        if (not walk[t] or t == len(walk) - 1) and s is not None:
            e = t if not walk[t] else t + 1
            if e - s >= min_len:
                out[s:e] = True
            s = None
    return out


def plan_steps(path_xy, seg, feet_start, feet_end, lat_off, step_len, step_dur_f, ds_f, first):
    """Plan de pas alternés sur un segment de marche, indexé sur l'ABSCISSE CURVILIGNE du pelvis.

    Le root est fixé (on ne touche pas la trajectoire captée) : c'est donc la progression du
    pelvis qui dicte OÙ et QUAND poser les pieds — le pied k se pose quand le pelvis atteint
    son foothold (~12 cm derrière), sinon la cible est inatteignable (jambe trop courte) et
    l'IK part en vrille. path_xy : (T,2) chemin pelvis lissé. seg=(s,e). feet_start/end :
    {L,R:(2,)} positions épinglées des stands adjacents. Retourne, par pied,
    [(t_liftoff, t_strike, foothold_from(2,), foothold_to(2,))].
    """
    s, e = seg
    d = np.linalg.norm(np.diff(path_xy[s:e], axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(d)])  # progression pelvis par frame du segment
    dist = float(arc[-1])
    # pas "voyage" le long du chemin (on réserve les ~40 derniers cm aux pas de settle),
    # + 2 pas de SETTLE en fin de segment vers les positions finales : ils ne sont frappés
    # que quand le pelvis est (presque) arrivé -> toujours atteignables, zéro glissé forcé.
    n_travel = max(0, int(np.ceil(max(0.0, dist - 0.4) / step_len)))
    order = [first, "LR".replace(first, "")]
    n_total = n_travel + 2
    # budget temporel : si le segment est court, on raccourcit le swing plutôt que sauter des pas
    swing_f = min(step_dur_f, max(4, (e - s) // n_total - ds_f))
    steps = {"L": [], "R": []}
    cur = {k: np.asarray(feet_start[k], float) for k in "LR"}
    heading = smooth1d(np.gradient(path_xy, axis=0), 9)
    prev_strike = s
    for k in range(n_travel):
        foot = order[k % 2]
        target_arc = dist * (k + 1) / (n_travel + 1)
        t_strike = min(e - 2, s + int(np.searchsorted(arc, max(0.0, target_arc - 0.12))))
        t_idx = s + min(e - 1 - s, int(np.searchsorted(arc, target_arc)))
        c = path_xy[t_idx]
        h = heading[t_idx]
        nh = np.linalg.norm(h)
        h = h / nh if nh > 1e-6 else np.array([1.0, 0.0])
        perp = np.array([-h[1], h[0]])
        sign = 1.0 if foot == "L" else -1.0
        target = c + 0.12 * h + sign * lat_off * perp
        t_lift = max(s, prev_strike + ds_f, t_strike - swing_f)  # jamais deux pieds en l'air
        if t_lift >= t_strike:
            continue
        steps[foot].append((int(t_lift), int(t_strike), cur[foot].copy(), target.copy()))
        cur[foot] = target
        prev_strike = t_strike
    for j, t_strike in enumerate([e - 1 - (swing_f + ds_f), e - 1]):  # settle : avant-dernier puis dernier
        foot = order[(n_travel + j) % 2]
        t_strike = max(prev_strike + ds_f + 1, t_strike)
        t_lift = max(s, prev_strike + ds_f, t_strike - swing_f)
        if t_lift >= t_strike or t_strike > e - 1:
            continue
        target = np.asarray(feet_end[foot], float)
        steps[foot].append((int(t_lift), int(t_strike), cur[foot].copy(), target.copy()))
        cur[foot] = target
        prev_strike = t_strike
    return steps


def main():
    ap = argparse.ArgumentParser(description="Fix foot-skate + resynthèse de pas (SMPL-X, niveau clip construit)")
    ap.add_argument("--pt", required=True, help=".pt d'entrée (smpl_params_global, ex clip_built_box36.pt)")
    ap.add_argument("--out-pt", required=True, help=".pt de sortie")
    ap.add_argument("--npz", default="", help="clip npz apparié (body sync + labels stance) -> --out-npz")
    ap.add_argument("--out-npz", default="", help="npz de sortie (défaut : <npz>_gaitfix.npz)")
    ap.add_argument("--model-dir", required=True, help="parent de smplx/")
    ap.add_argument("--gender", default="neutral")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--walk-thr", type=float, default=0.35, help="vitesse pelvis (m/s) déclenchant une phase de marche")
    ap.add_argument("--min-walk", type=float, default=0.3, help="durée mini (s) d'un segment de marche")
    ap.add_argument("--step-len", type=float, default=0.35, help="longueur de pas cible (m)")
    ap.add_argument("--step-dur", type=float, default=0.30, help="durée de swing (s)")
    ap.add_argument("--double-support", type=float, default=0.10, help="double appui entre pas (s)")
    ap.add_argument("--swing-height", type=float, default=0.09, help="apex du swing (m)")
    ap.add_argument("--lat-scale", type=float, default=1.0, help="échelle de l'écart latéral mesuré en stand")
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--w-target", type=float, default=100.0, help="poids position orteil+cheville")
    ap.add_argument("--w-reg", type=float, default=0.5, help="poids retour à la pose d'origine")
    ap.add_argument("--w-smooth", type=float, default=20.0, help="poids lissage temporel (accélération angulaire)")
    args = ap.parse_args()

    import torch
    import smplx

    pt = torch.load(args.pt, map_location="cpu", weights_only=False)
    p = pt["smpl_params_global"]
    T = np.asarray(p["transl"]).shape[0]
    betas = np.asarray(p["betas"], np.float32)
    go = np.asarray(p["global_orient"], np.float32).reshape(T, 3)
    bp0 = np.asarray(p["body_pose"], np.float32).reshape(T, 63)
    tr = np.asarray(p["transl"], np.float32).reshape(T, 3)

    model = smplx.create(args.model_dir, model_type="smplx", gender=args.gender, num_betas=betas.shape[1],
                         use_pca=False, flat_hand_mean=True, batch_size=T)

    def fk(bp_t):
        return model(betas=torch.as_tensor(betas), global_orient=torch.as_tensor(go),
                     body_pose=bp_t, transl=torch.as_tensor(tr)).joints

    with torch.no_grad():
        J = fk(torch.as_tensor(bp0)).numpy()

    up, up_sign = detect_up_axis(J)
    horiz = [i for i in range(3) if i != up]
    print(f"[gait] {T} frames @ {args.fps:.0f} fps, up axis = {'xyz'[up]}{'+' if up_sign > 0 else '-'}")

    # --- segmentation stand / marche -------------------------------------------------------
    pel = smooth1d(J[:, 0, horiz], 7)
    speed = np.linalg.norm(np.gradient(pel, axis=0), axis=1) * args.fps
    walk = segment_walk(speed, args.walk_thr, int(args.min_walk * args.fps), args.fps)
    segs = []
    s = None
    for t in range(T):
        if walk[t] and s is None:
            s = t
        if (not walk[t] or t == T - 1) and s is not None:
            segs.append((s, t if not walk[t] else t + 1))
            s = None
    print(f"[gait] segments de marche : {segs} (vitesse max {speed.max():.2f} m/s)")

    # --- cibles pieds ----------------------------------------------------------------------
    toe = {k: J[:, J_TOE[k], :] for k in "LR"}
    ank = {k: J[:, J_ANKLE[k], :] for k in "LR"}
    ground = min(float(np.percentile(toe["L"][:, up] * up_sign, 5)),
                 float(np.percentile(toe["R"][:, up] * up_sign, 5)))
    stance_mask = ~walk
    tgt_toe = {k: toe[k].copy() for k in "LR"}
    tgt_ank = {k: ank[k].copy() for k in "LR"}
    stance = {k: np.ones(T, bool) for k in "LR"}
    landing = {k: np.zeros(T, bool) for k in "LR"}  # frames juste après un strike : arrêt net exigé

    # stands : épingler à la position médiane du segment (hauteur = sol)
    for k in "LR":
        s = None
        for t in range(T + 1):
            in_stand = t < T and stance_mask[t]
            if in_stand and s is None:
                s = t
            if not in_stand and s is not None:
                med = np.median(toe[k][s:t], axis=0)
                med[up] = ground * up_sign
                dz = np.median((ank[k][s:t] - toe[k][s:t]), axis=0)
                tgt_toe[k][s:t] = med
                tgt_ank[k][s:t] = med + dz
                s = None

    # marches : plan de pas + trajectoires de swing
    for (s, e) in segs:
        prev_end = s - 1 if s > 0 else 0
        next_start = e if e < T else T - 1
        feet_start = {k: tgt_toe[k][prev_end][horiz] for k in "LR"}
        feet_end = {k: np.median(toe[k][next_start:min(T, next_start + 10)], axis=0)[horiz] for k in "LR"}
        # écart latéral = composante PERPENDICULAIRE au déplacement (pas la norme : un pied
        # devant l'autre gonflerait l'écart) ; premier pied à partir = le pied ARRIÈRE
        disp = pel[min(e - 1, T - 1)] - pel[s]
        nd = np.linalg.norm(disp)
        fwd = disp / nd if nd > 1e-6 else np.array([1.0, 0.0])
        perp = np.array([-fwd[1], fwd[0]])
        rel = toe["L"][prev_end][horiz] - toe["R"][prev_end][horiz]
        lat = max(0.06, 0.5 * abs(float(rel @ perp))) * args.lat_scale
        first = "L" if float((toe["L"][prev_end][horiz] - toe["R"][prev_end][horiz]) @ fwd) <= 0 else "R"
        plan = plan_steps(pel, (s, e), feet_start, feet_end, lat,
                          args.step_len, int(args.step_dur * args.fps), int(args.double_support * args.fps), first)
        for k in "LR":
            # par défaut pendant la marche : pied posé à sa dernière position connue
            cur = tgt_toe[k][prev_end].copy()
            cur[up] = ground * up_sign
            dz = np.median((ank[k] - toe[k])[stance_mask], axis=0)
            for t in range(s, e):
                tgt_toe[k][t] = cur
            for (t0, t1, p_from, p_to) in plan[k]:
                for t in range(t0, min(t1 + 1, T)):
                    u = (t - t0) / max(1, t1 - t0)
                    # translation horizontale FENÊTRÉE sur u∈[0.2, 0.8] : le pied ne bouge
                    # horizontalement que quand il est au-dessus de ~5 cm -> zéro skate aux
                    # bords de swing (décollage/atterrissage verticaux)
                    v = np.clip((u - 0.2) / 0.6, 0.0, 1.0)
                    w = 3 * v**2 - 2 * v**3
                    xy = (1 - w) * p_from + w * p_to
                    z = ground + args.swing_height * np.sin(np.pi * np.clip(u, 0, 1))
                    v3 = np.zeros(3)
                    v3[horiz] = xy
                    v3[up] = z * up_sign
                    tgt_toe[k][t] = v3
                    stance[k][t] = False
                after = np.zeros(3)
                after[horiz] = p_to
                after[up] = ground * up_sign
                for t in range(min(t1 + 1, T), e):
                    tgt_toe[k][t] = after
                landing[k][min(t1, T - 1):min(t1 + 5, T)] = True
            for t in range(s, e):
                tgt_ank[k][t] = tgt_toe[k][t] + dz
        print(f"[gait] seg {s}-{e} : {len(plan['L'])} pas L, {len(plan['R'])} pas R (1er : {first}, lat {lat:.2f} m)")

    # --- IK jambes batch -------------------------------------------------------------------
    import torch as th
    dev = "cuda" if th.cuda.is_available() else "cpu"
    model = model.to(dev)
    idx = LEG_BP["L"] + LEG_BP["R"]
    bp = th.as_tensor(bp0, device=dev)
    leg = bp[:, idx].clone().requires_grad_(True)
    leg0 = bp[:, idx].clone()
    tt = {k: th.as_tensor(tgt_toe[k], dtype=th.float32, device=dev) for k in "LR"}
    ta = {k: th.as_tensor(tgt_ank[k], dtype=th.float32, device=dev) for k in "LR"}
    # les frames de stance comptent triple (le skate se joue là), les atterrissages x9 (arrêt net)
    wf = {k: th.as_tensor(np.where(landing[k], 9.0, np.where(stance[k], 3.0, 1.0)),
                          dtype=th.float32, device=dev) for k in "LR"}
    go_t = th.as_tensor(go, device=dev)
    tr_t = th.as_tensor(tr, device=dev)
    betas_t = th.as_tensor(betas, device=dev)
    opt = th.optim.Adam([leg], lr=args.lr)
    for it in range(args.iters):
        opt.zero_grad()
        bp_full = bp.clone()
        bp_full[:, idx] = leg
        joints = model(betas=betas_t, global_orient=go_t, body_pose=bp_full, transl=tr_t).joints
        l_t = sum((wf[k] * ((joints[:, J_TOE[k]] - tt[k]) ** 2).sum(-1)).mean() +
                  (wf[k] * ((joints[:, J_ANKLE[k]] - ta[k]) ** 2).sum(-1)).mean() for k in "LR")
        l_r = ((leg - leg0) ** 2).mean()
        l_s = ((leg[2:] - 2 * leg[1:-1] + leg[:-2]) ** 2).mean()
        loss = args.w_target * l_t + args.w_reg * l_r + args.w_smooth * l_s
        loss.backward()
        opt.step()
        if it % 100 == 0 or it == args.iters - 1:
            print(f"[ik] {it:4d} loss {loss.item():.4f} target {l_t.item():.5f} reg {l_r.item():.4f} smooth {l_s.item():.5f}")

    bp_new = bp0.copy()
    bp_new[:, idx] = leg.detach().cpu().numpy()

    # --- métriques avant/après -------------------------------------------------------------
    with th.no_grad():
        J2 = model(betas=betas_t, global_orient=go_t, body_pose=th.as_tensor(bp_new, device=dev),
                   transl=tr_t).joints.cpu().numpy()
    for tag, JJ in [("avant", J), ("après", J2)]:
        for k in "LR":
            h = JJ[:, J_TOE[k], up] * up_sign - ground
            hv = np.linalg.norm(np.gradient(JJ[:, J_TOE[k]][:, horiz], axis=0), axis=1) * args.fps
            low = h < 0.05
            skate = (hv[low] > 0.10).sum() if low.any() else 0
            above = h > 0.06
            lifts = int(((above[1:]) & (~above[:-1])).sum())
            print(f"[metrics {tag}] pied {k} : lift max {h.max():.3f} m, lifts>6cm {lifts}, "
                  f"frames skate(bas & >0.1 m/s) {int(skate)}/{int(low.sum())}")

    # --- sorties ---------------------------------------------------------------------------
    p["body_pose"] = bp_new.reshape(np.asarray(p["body_pose"]).shape)
    import torch
    torch.save(pt, args.out_pt)
    print(f"[gait] .pt -> {args.out_pt}")
    if args.npz:
        d = dict(np.load(args.npz, allow_pickle=True))
        d["body_pose"] = bp_new
        if "poses" in d and d["poses"].shape[1] >= 66:
            d["poses"][:, 3:66] = bp_new
        d["foot_stance_l"] = stance["L"]
        d["foot_stance_r"] = stance["R"]
        out_npz = args.out_npz or args.npz.replace(".npz", "_gaitfix.npz")
        np.savez(out_npz, **d)
        print(f"[gait] npz -> {out_npz}")


if __name__ == "__main__":
    main()
