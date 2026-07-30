#!/usr/bin/env python3
"""Rigid-grasp refinement + optional SDF contact/non-penetration.

Primary layer (always on): grasp timing from OBJECT MOTION (FoundationPose speed), a
constant wrist-in-object offset estimated on the carry frames, a smooth per-frame wrist
target = object_pose(t) . offset, and an arm(+optional spine) IK to reach it. This gives
the gross, smooth, correct-face placement.

Refinement layer (optional, --w-penetration / --w-sdf-contact, set both to 0 to disable
and skip the SDF build entirely): a precomputed signed-distance field of the object CAD
(queried differentiably via trilinear grid_sample in the object-local frame) adds
  - non-penetration: push back only hand vertices deeper than --pen-tol (a few cm of
    penetration stays free, so this term never fights contact),
  - SDF contact: pull the nearest hand patch to --contact-depth INSIDE the surface, so
    contact is GUARANTEED (small controlled penetration) instead of hovering at zero.
Priority baked into the defaults: contact is mandatory, a couple cm of penetration is fine.
Between contact-depth and pen-tol neither term acts -> a stable dead-zone, no in/out
oscillation. Both gated by the carry blend weight; SDF is signed (no NN flicker).

Position only -- HaMeR wrist orientation and finger pose preserved. Root kept. body_pose
is frame-independent -> written to incam+global.

Run in an env with smplx + torch + numpy + trimesh + scipy (holonew).
"""

import argparse
import glob
import os

import numpy as np

JW = {"L": 20, "R": 21}                       # SMPL-X wrist joints
HAND_JOINTS = {"L": [20] + list(range(25, 40)), "R": [21] + list(range(40, 55))}
ARM_BP = {
    "L": [36, 37, 38, 45, 46, 47, 51, 52, 53],   # left_collar(13), left_shoulder(16), left_elbow(18)
    "R": [39, 40, 41, 48, 49, 50, 54, 55, 56],   # right_collar(14), right_shoulder(17), right_elbow(19)
}
WRIST_BP = {"L": [57, 58, 59], "R": [60, 61, 62]}   # wrist joints 20/21 -> palm orientation (--free-wrist)
SPINE_BP = [6, 7, 8, 15, 16, 17, 24, 25, 26]     # spine1(3), spine2(6), spine3(9) -> torso flexion


def blend_from_span(T, g0, g1, ramp):
    """1.0 across [g0, g1], cosine ramps of length `ramp` at the edges, 0 outside."""
    w = np.zeros(T, np.float32)
    w[g0:g1 + 1] = 1.0
    for k in range(1, ramp + 1):
        v = 0.5 * (1.0 + np.cos(np.pi * k / (ramp + 1)))
        if g0 - k >= 0:
            w[g0 - k] = v
        if g1 + k < T:
            w[g1 + k] = v
    return w


def build_sdf_grid(mesh, res, margin, dev):
    """Voxelize the object CAD into a signed-distance grid (our convention: <0 inside),
    once, in object-local frame. Returns (vol[1,1,Z,Y,X], lo[3], hi[3]) for grid_sample."""
    import torch
    from trimesh.proximity import signed_distance
    lo = mesh.bounds[0] - margin
    hi = mesh.bounds[1] + margin
    xs = np.linspace(lo[0], hi[0], res)
    ys = np.linspace(lo[1], hi[1], res)
    zs = np.linspace(lo[2], hi[2], res)
    gz, gy, gx = np.meshgrid(zs, ys, xs, indexing="ij")           # (Z,Y,X)
    pts = np.stack([gx, gy, gz], -1).reshape(-1, 3).astype(np.float32)
    sd = np.asarray(signed_distance(mesh, pts), np.float32)
    c_sd = float(signed_distance(mesh, mesh.centroid.reshape(1, 3))[0])
    sign = -1.0 if c_sd > 0 else 1.0                              # make inside negative
    sd = (sign * sd).reshape(res, res, res)                       # vol[z,y,x]
    vol = torch.as_tensor(sd, dtype=torch.float32, device=dev)[None, None]
    lo_t = torch.as_tensor(lo, dtype=torch.float32, device=dev)
    hi_t = torch.as_tensor(hi, dtype=torch.float32, device=dev)
    return vol, lo_t, hi_t


def query_sdf(vol, lo, hi, pts):
    """pts (M,3) object-local -> (M,) signed distance via trilinear interpolation."""
    import torch
    import torch.nn.functional as F
    n = 2.0 * (pts - lo) / (hi - lo) - 1.0                        # normalized x,y,z
    grid = n.view(1, -1, 1, 1, 3)
    out = F.grid_sample(vol, grid, mode="bilinear", padding_mode="border", align_corners=True)
    return out.view(-1)


def main():
    ap = argparse.ArgumentParser(description="Rigid-grasp SMPL-X refinement (+optional SDF contact)")
    ap.add_argument("--pt", required=True, help="GVHMR .pt (wrist-corrected recommended)")
    ap.add_argument("--fp-poses", required=True, help="FoundationPose ob_in_cam/ dir")
    ap.add_argument("--object-mesh", required=True, help="object CAD (meters)")
    ap.add_argument("--clip-dir", required=True, help="capture folder (cam_K.txt)")
    ap.add_argument("--model-dir", required=True, help="parent of smplx/ model folder")
    ap.add_argument("--out", required=True, help="output refined .pt")
    ap.add_argument("--gender", default="neutral")
    ap.add_argument("--move-th", type=float, default=0.002, help="object speed to count as 'carried' (m/frame)")
    ap.add_argument("--move-smooth", type=int, default=5, help="frames to smooth the object speed")
    ap.add_argument("--participate-th", type=float, default=0.12, help="min hand-object dist to count a hand as grasping (m)")
    ap.add_argument("--carry-frames", default="", help="force carry window 'a:b' (covers a lift/grasp the motion detector misses)")
    ap.add_argument("--repel-frames", default="", help="frames 'a:b,c:d' où on éloigne la main de l'objet "
                    "(barrière de DÉGAGEMENT, sans grasp) — évite que les doigts frôlent/bougent l'objet au "
                    "début/à la fin, tout en gardant l'arc GVHMR ; blend cosinus pour la fluidité")
    ap.add_argument("--repel-clearance", type=float, default=0.04, help="distance mini (m) main<->objet à tenir sur les repel-frames")
    ap.add_argument("--w-repel", type=float, default=5.0, help="poids de la barrière de dégagement (repel-frames)")
    ap.add_argument("--free-wrist", action="store_true", help="also optimize the wrist so the palm flattens onto the object (no-HaMeR path; overrides input wrist orientation)")
    ap.add_argument("--pad", type=int, default=15, help="frames of padding around the carry window")
    ap.add_argument("--edge-ramp", type=int, default=10, help="frames of cosine ramp at carry edges")
    ap.add_argument("--n-obj", type=int, default=800, help="object surface samples (participation test)")
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--w-reach", type=float, default=8.0, help="pull wrist to its object-locked target (gross placement)")
    ap.add_argument("--w-prior", type=float, default=1.0, help="keep the ARM close to GVHMR")
    ap.add_argument("--w-smooth", type=float, default=2.0, help="temporal smoothness")
    ap.add_argument("--free-torso", action="store_true", help="also let the spine flex (needed for low pickups)")
    ap.add_argument("--w-torso-prior", type=float, default=8.0, help="stiffness of the spine (higher = less bend)")
    # --- SDF contact/penetration layer (set both weights to 0 to fully disable) ---
    # priority: CONTACT is mandatory; a few cm of penetration is acceptable.
    ap.add_argument("--w-penetration", type=float, default=5.0, help="penalize penetration DEEPER than --pen-tol (0=off)")
    ap.add_argument("--w-sdf-contact", type=float, default=20.0, help="pull the closest hand patch into contact (0=off)")
    ap.add_argument("--pen-tol", type=float, default=0.02, help="free penetration depth (m); only deeper is pushed back")
    ap.add_argument("--contact-depth", type=float, default=0.008, help="target penetration (m) that counts as contact; guarantees touch")
    ap.add_argument("--contact-frac", type=float, default=0.20, help="fraction of the CLOSEST hand vertices pulled into contact (palm patch, not 1 fingertip)")
    ap.add_argument("--sdf-res", type=int, default=64, help="SDF grid resolution")
    ap.add_argument("--sdf-margin", type=float, default=0.22, help="SDF grid margin around the CAD (m); big enough that the field reaches floated-away hands")
    ap.add_argument("--sdf-tau", type=float, default=0.01, help="soft-min temperature for SDF contact (m)")
    ap.add_argument("--post-smooth-window", type=int, default=9, help="Savitzky-Golay window (odd, 0=disable)")
    ap.add_argument("--post-smooth-poly", type=int, default=2)
    args = ap.parse_args()

    import torch
    import smplx
    import trimesh

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    pt = torch.load(args.pt, map_location="cpu", weights_only=False)
    pi = pt["smpl_params_incam"]
    T = np.asarray(pi["transl"]).shape[0]
    nb = np.asarray(pi["betas"]).shape[1]

    betas = torch.as_tensor(np.asarray(pi["betas"]), dtype=torch.float32, device=dev)
    go = torch.as_tensor(np.asarray(pi["global_orient"], np.float32).reshape(T, 3), device=dev)
    tr = torch.as_tensor(np.asarray(pi["transl"], np.float32).reshape(T, 3), device=dev)
    bp0 = np.asarray(pi["body_pose"], np.float32).reshape(T, 63)

    model = smplx.create(args.model_dir, model_type="smplx", gender=args.gender, num_betas=nb,
                         use_pca=False, flat_hand_mean=True, batch_size=T).to(dev)

    lbs = model.lbs_weights.detach().cpu().numpy()
    hand_vidx = {h: np.where(lbs[:, HAND_JOINTS[h]].sum(1) > 0.5)[0] for h in JW}

    with torch.no_grad():
        out0 = model(betas=betas, global_orient=go, body_pose=torch.as_tensor(bp0, device=dev), transl=tr)
    J0 = out0.joints
    V0 = out0.vertices

    mesh = trimesh.load(args.object_mesh, force="mesh", process=False)
    osamp = mesh.sample(args.n_obj).astype(np.float32)
    fps = sorted(glob.glob(os.path.join(args.fp_poses, "*.txt")))
    P = np.stack([np.loadtxt(f).reshape(4, 4) for f in fps]).astype(np.float32)[:T]
    Robj, tobj = P[:, :3, :3], P[:, :3, 3]
    obj_np = np.einsum("tij,nj->tni", Robj, osamp) + tobj[:, None, :]
    obj = torch.as_tensor(obj_np, device=dev)

    # per-hand min hand-object distance -> grasp detection + participation
    dmin = {}
    with torch.no_grad():
        for h in JW:
            dmin[h] = torch.cdist(V0[:, hand_vidx[h]], obj).amin(dim=(1, 2)).cpu().numpy()
    near_any = np.zeros(T, bool)
    for h in JW:
        near_any |= dmin[h] < args.participate_th

    # object speed (grasp timing + offset frame selection)
    vel = np.linalg.norm(np.diff(tobj, axis=0, prepend=tobj[:1]), axis=1)
    if args.move_smooth > 1:
        vel = np.convolve(vel, np.ones(args.move_smooth) / args.move_smooth, mode="same")
    moving = vel > args.move_th

    # carry window: manual override, else the object-motion span EXTENDED backward/forward
    # over the contiguous grasp (hand still on the object), so the lift and set-down are
    # covered -- the motion detector alone misses them (box barely moves while gripped/lifted).
    if args.carry_frames:
        g0, g1 = (int(x) for x in args.carry_frames.split(":"))
        g0, g1 = max(0, g0), min(T - 1, g1)
        print(f"[rigid] carry window forced to [{g0}:{g1}]")
    else:
        mv_idx = np.where(moving)[0]
        if len(mv_idx) == 0:
            print(f"[rigid] no object motion above {args.move_th} m/frame (max={vel.max():.4f}); unchanged")
            torch.save(pt, args.out); return
        g0, g1 = int(mv_idx.min()), int(mv_idx.max())
        while g0 > 0 and near_any[g0 - 1]:
            g0 -= 1
        while g1 < T - 1 and near_any[g1 + 1]:
            g1 += 1
        print(f"[rigid] carried [{g0}:{g1}]  (motion max {vel.max():.4f} m/frame, extended over grasp)")

    J0_np = J0.detach().cpu().numpy()
    offset, blend = {}, {}
    for h in JW:
        if dmin[h][g0:g1 + 1].min() > args.participate_th:
            print(f"[rigid] hand {h} not grasping (min carry dist {dmin[h][g0:g1+1].min():.3f} m)")
            continue
        wrist_cam = J0_np[:, JW[h]]
        local = np.einsum("tij,tj->ti", Robj.transpose(0, 2, 1), wrist_cam - tobj)
        sel = moving & (dmin[h] < args.participate_th)
        if sel.sum() < 3:
            sel = np.zeros(T, bool); sel[g0:g1 + 1] = True
        offset[h] = np.median(local[sel], axis=0)
        blend[h] = blend_from_span(T, g0, g1, args.edge_ramp)
    grasping = list(offset.keys())
    if not grasping:
        print("[rigid] no hand grasping; unchanged"); torch.save(pt, args.out); return
    print(f"[rigid] grasping hands: {grasping}  free-torso={args.free_torso}")

    # répulsion seule (pénétration) sur des frames hors-grasp (ex début/fin) : pousse la main HORS
    # de l'objet sans la coller (ni reach ni contact) -> corrige un bras mal capté qui traverse,
    # en gardant l'arc GVHMR (prior) et la fluidité (smooth). Blend cosinus pour la douceur.
    repel_blend = np.zeros(T, np.float32)
    for part in filter(None, args.repel_frames.split(",")):
        a, b = (int(x) for x in part.split(":"))
        repel_blend = np.maximum(repel_blend, blend_from_span(T, max(0, a), min(T - 1, b), args.edge_ramp))
    if repel_blend.max() > 0:
        print(f"[rigid] repulsion-seule sur {args.repel_frames} (pousse hors de l'objet, arc GVHMR gardé)")

    active = np.zeros(T, bool); active[g0:g1 + 1] = True; active |= repel_blend > 1e-6
    aidx = np.where(active)[0]
    lo, hi = max(0, int(aidx.min()) - args.pad), min(T, int(aidx.max()) + args.pad + 1)
    win = np.arange(lo, hi); Nw = len(win)
    print(f"[rigid] optimizing window [{lo}:{hi}) ({Nw} frames)")

    model_w = smplx.create(args.model_dir, model_type="smplx", gender=args.gender, num_betas=nb,
                           use_pca=False, flat_hand_mean=True, batch_size=Nw).to(dev)
    betas_w, go_w, tr_w, bp0_w = betas[win], go[win], tr[win], bp0[win]

    Rw = torch.as_tensor(P[win, :3, :3], device=dev)
    tw = torch.as_tensor(P[win, :3, 3], device=dev)
    target = {h: torch.einsum("tij,j->ti", Rw, torch.as_tensor(offset[h], dtype=torch.float32, device=dev)) + tw
              for h in grasping}
    blend_w = {h: torch.as_tensor(blend[h][win], dtype=torch.float32, device=dev) for h in grasping}
    repel_blend_w = torch.as_tensor(repel_blend[win], dtype=torch.float32, device=dev)

    # optional SDF grid (skipped entirely if all SDF terms are off)
    repel_on = bool(repel_blend.max() > 0)
    use_sdf = (args.w_penetration > 0 or args.w_sdf_contact > 0 or repel_on)
    sdf_vol = sdf_lo = sdf_hi = None
    if use_sdf:
        if not mesh.is_watertight:
            print("[rigid] WARNING: object mesh not watertight -> SDF sign may be unreliable "
                  "(disable with --w-penetration 0 --w-sdf-contact 0 if it misbehaves)")
        sdf_vol, sdf_lo, sdf_hi = build_sdf_grid(mesh, args.sdf_res, args.sdf_margin, dev)
        print(f"[rigid] built SDF grid {args.sdf_res}^3  watertight={mesh.is_watertight}")

    arm_cols = sorted({c for h in grasping for c in ARM_BP[h]})
    wrist_cols = sorted({c for h in grasping for c in WRIST_BP[h]}) if args.free_wrist else []
    spine_cols = SPINE_BP if args.free_torso else []
    free_cols = sorted(set(arm_cols) | set(wrist_cols) | set(spine_cols))
    mask = torch.zeros(63, device=dev); mask[free_cols] = 1.0
    prior_w = torch.zeros(63, device=dev)
    prior_w[arm_cols] = args.w_prior
    if wrist_cols:
        prior_w[wrist_cols] = args.w_prior * 0.25   # loose -> the palm can rotate flat onto the face
    if spine_cols:
        prior_w[spine_cols] = args.w_torso_prior

    bp0_w_t = torch.as_tensor(bp0_w, device=dev)
    delta = torch.zeros((Nw, 63), device=dev, requires_grad=True)
    opt = torch.optim.Adam([delta], lr=args.lr)

    for it in range(args.iters):
        opt.zero_grad()
        bp = bp0_w_t + delta * mask
        out = model_w(betas=betas_w, global_orient=go_w, body_pose=bp, transl=tr_w)
        Jw = out.joints
        e_reach = torch.zeros((), device=dev)
        e_pen = torch.zeros((), device=dev)
        e_sdfc = torch.zeros((), device=dev)
        e_repel = torch.zeros((), device=dev)
        for h in grasping:
            a = blend_w[h]
            wp = Jw[:, JW[h], :]
            e_reach = e_reach + (a * (wp - target[h]).pow(2).sum(-1)).sum() / (a.sum() + 1e-6)
            if use_sdf:
                hv = out.vertices[:, hand_vidx[h]]                      # (Nw, nh, 3) camera
                diff = hv - tw[:, None, :]
                hv_local = torch.einsum("tij,tnj->tni", Rw.transpose(1, 2), diff)   # object-local
                sdf = query_sdf(sdf_vol, sdf_lo, sdf_hi, hv_local.reshape(-1, 3)).view(Nw, -1)
                if args.w_penetration > 0:
                    pen = torch.relu(-sdf - args.pen_tol)             # only penetration deeper than the tolerance
                    e_pen = e_pen + (a * pen.pow(2).sum(1)).sum() / (a.sum() + 1e-6)
                if args.w_sdf_contact > 0:
                    # pull the CLOSEST patch of the hand (not a single vertex) to contact_depth
                    # inside -> the palm rests on the face instead of one fingertip poking in.
                    k = max(1, int(args.contact_frac * sdf.shape[1]))
                    near_k = torch.topk(sdf, k, dim=1, largest=False).values   # k smallest sdf = palm side
                    short = torch.relu(near_k + args.contact_depth)
                    e_sdfc = e_sdfc + (a * short.pow(2).mean(1)).sum() / (a.sum() + 1e-6)
                if repel_on:
                    # barrière de DÉGAGEMENT : pousse le point de main le plus proche à >= repel_clearance
                    # de la surface (sdf du plus proche vertex via soft-min) -> pas de frôlement.
                    soft_min = -args.sdf_tau * torch.logsumexp(-sdf / args.sdf_tau, dim=1)   # ~ min sdf (m)
                    clear = torch.relu(args.repel_clearance - soft_min)   # >0 si trop proche (dedans ou <clearance)
                    e_repel = e_repel + (repel_blend_w * clear.pow(2)).sum() / (repel_blend_w.sum() + 1e-6)
        dm = delta * mask
        e_prior = (prior_w * dm.pow(2)).sum(-1).mean()
        e_smooth = (dm[2:] - 2 * dm[1:-1] + dm[:-2]).pow(2).sum(-1).mean() if Nw > 2 else torch.zeros((), device=dev)
        loss = (args.w_reach * e_reach + e_prior + args.w_smooth * e_smooth
                + args.w_penetration * e_pen + args.w_sdf_contact * e_sdfc
                + args.w_repel * e_repel)
        loss.backward()
        opt.step()
        if it % 50 == 0 or it == args.iters - 1:
            rms_cm = 100.0 * float((e_reach / max(len(grasping), 1)).detach()) ** 0.5
            msg = f"[rigid] it {it:3d}  loss {loss.item():.5f}  reach~{rms_cm:.1f}cm  prior {float(e_prior.detach()):.4f}"
            if use_sdf:
                msg += f"  pen {float(e_pen.detach()):.5f}  sdfc {float(e_sdfc.detach()):.5f}"
            print(msg)

    bp_new_w = bp0_w + (delta * mask).detach().cpu().numpy()

    if args.post_smooth_window and args.post_smooth_window > args.post_smooth_poly and Nw >= args.post_smooth_window:
        from scipy.signal import savgol_filter
        bp_new_w[:, free_cols] = savgol_filter(bp_new_w[:, free_cols], args.post_smooth_window,
                                               args.post_smooth_poly, axis=0)

    bp_full = bp0.copy(); bp_full[win] = bp_new_w
    for key in ("smpl_params_incam", "smpl_params_global"):
        pt[key]["body_pose"] = torch.as_tensor(bp_full, dtype=torch.float32)
    torch.save(pt, args.out)
    print(f"[rigid] wrote {args.out}")


if __name__ == "__main__":
    main()