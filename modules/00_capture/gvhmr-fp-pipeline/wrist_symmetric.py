#!/usr/bin/env python3
"""Rendre les DEUX BRAS symétriques (miroir sagittal du corps) — collier, épaule, coude, poignet —
sur une fenêtre optionnelle. Le bras gauche devient le MIROIR EXACT du droit (mode ``r2l``, pour un
gauche mal capté), l'inverse (``l2r``), ou la moyenne des deux (``average``). Résout le cas « le
poignet gauche part dans tous les sens » : les deux bras bougent identiquement en miroir par rapport
au plan sagittal (≈ centre de la caisse portée à deux mains), donc positions ET orientations.

Miroir SMPL standard, purement LOCAL : échange gauche<->droit des rotations d'os + ``rotvec × [1,-1,-1]``
(réflexion de l'axe gauche/droite). Le rest pose SMPL étant symétrique, mirrorer les rotations locales
de la chaîne du bras donne un bras exactement miroir de l'autre. Fondu cosinus aux bords de la fenêtre
(fluidité). body_pose écrit incam+global. Pas besoin de smplx (miroir purement local).

Run in an env with torch + scipy (holonew). À lancer APRÈS contact_opt, AVANT fuse.
"""
import argparse

import numpy as np

# paires (joint gauche, joint droit) de la chaîne du BRAS SMPL-X
ARM_PAIRS = [(13, 14), (16, 17), (18, 19), (20, 21)]   # collier, épaule, coude, poignet
_MIRROR = np.array([1.0, -1.0, -1.0])                  # réflexion de l'axe G/D sur un rotvec LOCAL


def _cols(j):
    """Colonnes body_pose (63) du joint j (les joints 1..21 -> cols (j-1)*3 .. +2)."""
    return [(j - 1) * 3, (j - 1) * 3 + 1, (j - 1) * 3 + 2]


def blend_from_span(T, g0, g1, ramp):
    """1.0 sur [g0, g1], rampes cosinus de longueur `ramp` aux bords, 0 dehors."""
    w = np.zeros(T, np.float32)
    w[g0:g1 + 1] = 1.0
    for k in range(1, ramp + 1):
        v = 0.5 * (1.0 + np.cos(np.pi * k / (ramp + 1)))
        if g0 - k >= 0:
            w[g0 - k] = v
        if g1 + k < T:
            w[g1 + k] = v
    return w


def main():
    ap = argparse.ArgumentParser(description="Symétriser les deux bras (miroir sagittal)")
    ap.add_argument("--pt", required=True, help="GVHMR .pt (contact recommandé)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", default="", help="'a:b' fenêtre à symétriser (défaut: tout le clip)")
    ap.add_argument("--edge-ramp", type=int, default=5, help="frames de fondu cosinus aux bords")
    ap.add_argument("--mode", default="r2l", choices=["average", "r2l", "l2r"],
                    help="r2l = miroir du DROIT vers le gauche (garde le droit — si le gauche est mal capté) ; "
                    "l2r = inverse ; average = miroir-moyenne des deux")
    args = ap.parse_args()

    import torch
    from scipy.spatial.transform import Rotation

    pt = torch.load(args.pt, map_location="cpu", weights_only=False)
    T = np.asarray(pt["smpl_params_global"]["transl"]).shape[0]
    bp = np.asarray(pt["smpl_params_global"]["body_pose"], np.float64).reshape(T, 63).copy()

    w = np.ones(T, np.float32)
    if args.frames:
        a, b = (int(x) for x in args.frames.split(":"))
        w = blend_from_span(T, max(0, a), min(T - 1, b), args.edge_ramp)
        print(f"[sym] miroir bras ({args.mode}) sur [{a}:{b}] (fondu {args.edge_ramp})")
    else:
        print(f"[sym] miroir bras ({args.mode}) sur tout le clip")

    def geo_mean(rv0, rv1):
        """Moyenne géodésique de deux rotvecs (T,3) -> (T,3)."""
        return np.stack([Rotation.from_rotvec(np.stack([rv0[t], rv1[t]])).mean().as_rotvec()
                         for t in range(T)])

    def slerp_rv(orig, new, wt):
        """Interpole (slerp) rotvec orig->new au poids wt (par frame)."""
        from scipy.spatial.transform import Slerp
        out = orig.copy()
        for t in range(len(orig)):
            if wt[t] <= 1e-6:
                continue
            out[t] = (new[t] if wt[t] >= 1 - 1e-6
                      else Slerp([0.0, 1.0], Rotation.from_rotvec([orig[t], new[t]]))(wt[t]).as_rotvec())
        return out

    for jl, jr in ARM_PAIRS:
        cL, cR = _cols(jl), _cols(jr)
        rvL, rvR = bp[:, cL].copy(), bp[:, cR].copy()
        if args.mode == "r2l":
            newR, newL = rvR, rvR * _MIRROR
        elif args.mode == "l2r":
            newL, newR = rvL, rvL * _MIRROR
        else:                                              # average : moyenne dans le repère droit
            avg = geo_mean(rvR, rvL * _MIRROR)
            newR, newL = avg, avg * _MIRROR
        bp[:, cR] = slerp_rv(rvR, newR, w)
        bp[:, cL] = slerp_rv(rvL, newL, w)

    for key in ("smpl_params_incam", "smpl_params_global"):
        pt[key]["body_pose"] = torch.as_tensor(bp, dtype=torch.float32)
    torch.save(pt, args.out)
    print(f"[sym] wrote {args.out}")


if __name__ == "__main__":
    main()
