#!/usr/bin/env python3
"""Finalise le npz training holosoma : merge GT-contact + pose du support statique.

Remplace l'étape ad-hoc (non commitée) qui avait produit *_gtcontact(_slow16).npz :

  1. GT-contact : rééchantillonne les champs de contact_from_retarget.py (30 fps, timeline du
     clip amass/qpos) sur la timeline du npz training (50 fps, sortie de
     convert_data_format_mj.py) — nearest pour bool/idx/witness (pas de blend entre faces),
     lerp pour la distance. Clés écrites : object_ref_contact, object_ref_contact_dist,
     object_ref_anchor_idx, object_ref_witness_local.
  2. Support (table) : pose statique depuis object_poses[:, 1] du qpos npz (médiane des frames,
     wxyz) + chemin de mesh -> clés support_pos_w (3,), support_quat_w (4, wxyz),
     support_mesh (str). Consommées par le loader holosoma pour spawner la table comme objet
     de scène (au lieu de la cuire dans le terrain).

Usage :
  python merge_training_npz.py --motion <w_obj.npz> --contact <contact.npz> \
      [--qpos <qpos.npz> --support-mesh <mesh.obj>] [--out <final.npz>]
"""
import argparse

import numpy as np


def main():
    ap = argparse.ArgumentParser(description="Merge GT-contact + support dans le npz training")
    ap.add_argument("--motion", required=True, help="npz training 50 fps (convert_data_format_mj)")
    ap.add_argument("--contact", required=True, help="npz contact 30 fps (contact_from_retarget)")
    ap.add_argument("--qpos", default="", help="qpos npz avec object_poses (T,2,7) -> support = objet 1")
    ap.add_argument("--support-mesh", default="", help="nom du mesh CENTRÉ du support (centré sur le "
                    "centre de sa boîte, comme la box) écrit dans support_mesh -- PAS le mesh world-baked, "
                    "la pose support_pos_w/quat_w le positionne. Ex femto14_support_box36.obj")
    ap.add_argument("--out", default="", help="sortie (défaut : <motion> avec suffixe _gtcontact)")
    args = ap.parse_args()

    d = dict(np.load(args.motion, allow_pickle=True))
    c = np.load(args.contact, allow_pickle=True)
    t50 = d["body_pos_w"].shape[0]
    t30 = c["object_ref_contact"].shape[0]

    # même durée, fps différents : index source par frame de sortie
    src = np.linspace(0.0, t30 - 1, t50)
    near = np.round(src).astype(int)
    i0 = np.floor(src).astype(int)
    i1 = np.minimum(i0 + 1, t30 - 1)
    w = (src - i0)[:, None]

    d["object_ref_contact"] = c["object_ref_contact"][near]
    d["object_ref_anchor_idx"] = c["object_ref_anchor_idx"][near]
    d["object_ref_contact_dist"] = (
        c["object_ref_contact_dist"][i0] * (1 - w[:, 0]) + c["object_ref_contact_dist"][i1] * w[:, 0]
    ).astype(np.float32)
    if "object_ref_witness_local" in c:
        d["object_ref_witness_local"] = c["object_ref_witness_local"][near].astype(np.float32)
    n_contact = int(d["object_ref_contact"].sum())
    print(f"[merge] contact 30fps({t30}) -> 50fps({t50}) : {n_contact} frames en contact "
          f"({100.0 * n_contact / t50:.1f}%)")

    if args.qpos:
        q = np.load(args.qpos, allow_pickle=True)
        op = q["object_poses"]
        assert op.shape[1] >= 2, f"pas de 2e objet (support) dans {args.qpos}: {op.shape}"
        sup = np.median(op[:, 1], axis=0)  # statique -> médiane (robuste au jitter du solve SQP, ~1 cm)
        drift = np.abs(op[:, 1] - sup).max()
        assert drift < 3e-2, f"le support n'est pas statique (drift max {drift:.4f})"
        if drift > 5e-3:
            print(f"[merge] note: jitter support {drift * 100:.1f} cm absorbé par la médiane")
        d["support_pos_w"] = sup[:3].astype(np.float32)
        d["support_quat_w"] = sup[3:7].astype(np.float32)  # wxyz (convention npz)
        if args.support_mesh:
            d["support_mesh"] = np.array(args.support_mesh)
        print(f"[merge] support: pos {np.round(sup[:3], 3)}, quat(wxyz) {np.round(sup[3:7], 3)}"
              + (f", mesh {args.support_mesh}" if args.support_mesh else ""))

    out = args.out or args.motion.replace(".npz", "_gtcontact.npz")
    np.savez(out, **d)
    print(f"[merge] -> {out}")


if __name__ == "__main__":
    main()
