#!/usr/bin/env python
"""Rejoue un clip retargete dans le viewer MuJoCo, caisse et table comprises.

Le clip porte deja tout ce qu'il faut : ``joint_pos`` est le qpos MuJoCo complet (7 racine en
wxyz + 29 articulations), ``object_pos_w``/``object_quat_w`` la pose de la caisse, et
``support_pos_w``/``support_quat_w`` celle de la table. On injecte donc juste les deux corps dans
le XML du robot et on rejoue image par image.

Le HUD affiche le numero de frame et l'etat du flag de contact GT (``object_ref_contact``), qui est
la verite terrain de "le robot porte la caisse a cette frame".

Commandes :  ESPACE pause/reprise · fleches gauche/droite = +/-1 frame en pause · R = retour a 0

    python scripts/view_clip.py <clip.npz> [--fps 50]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
XML = REPO / "modules/third_party/holosoma_custom/src/holosoma/holosoma/data/robots/g1/g1_29dof.xml"


def build_model(box_half: float, support_mesh: Path | None):
    """Injecte la caisse (corps libre) et la table (corps statique) dans le XML du robot."""
    import mujoco

    xml = XML.read_text()

    # Framebuffer hors-ecran assez grand pour un rendu propre (defaut MuJoCo : 640x480), sinon
    # tout rendu offscreen au-dela leve "Image width > framebuffer width". Sans effet sur le
    # viewer interactif.
    xml = xml.replace(
        "<worldbody>", '<visual><global offwidth="1920" offheight="1080"/></visual>\n    <worldbody>', 1
    )

    # La table est rendue avec son VRAI mesh, pas la boite englobante support_half_extents : c'est
    # ce que fait l'entrainement depuis "support table uses the real mesh, not an AABB box". Une
    # AABB donnerait un plateau plus large que la table reelle et fausserait la lecture du depot.
    if support_mesh is not None:
        xml = xml.replace(
            "<worldbody>",
            f'<asset><mesh name="support_mesh" file="{support_mesh.name}"/></asset>\n    <worldbody>',
            1,
        )

    # Le XML du robot declare des paires de contact pieds<->'floor' mais ne definit pas le sol :
    # il est normalement fourni par la scene d'entrainement. On l'ajoute ici, sinon le modele ne
    # compile pas ("geom 'floor' not found in collision 0").
    extra = f"""
        <geom name="floor" type="plane" size="20 20 0.05" rgba="0.3 0.3 0.35 1"/>
        <body name="carry_box" pos="0 0 -5">
            <freejoint name="carry_box_joint"/>
            <geom type="box" size="{box_half} {box_half} {box_half}" rgba="0.85 0.45 0.15 0.9"/>
        </body>
    """
    if support_mesh is not None:
        extra += """
        <body name="support_table" pos="0 0 -5">
            <geom type="mesh" mesh="support_mesh" rgba="0.35 0.45 0.6 0.85"/>
        </body>
        """
    xml = xml.replace("</worldbody>", extra + "\n    </worldbody>")

    # Les chemins de meshes du XML sont relatifs a son dossier ; from_xml_string ne connait pas
    # cette base. On passe donc les fichiers explicitement plutot que d'ecrire un XML temporaire
    # dans l'arborescence de donnees.
    assets = {
        str(p.relative_to(XML.parent)): p.read_bytes()
        for p in XML.parent.rglob("*")
        if p.is_file() and p.suffix.lower() in (".stl", ".obj", ".png")
    }
    if support_mesh is not None:
        assets[support_mesh.name] = support_mesh.read_bytes()
    return mujoco.MjModel.from_xml_string(xml, assets)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("clip", help="chemin du .npz retargete")
    ap.add_argument("--fps", type=float, default=None, help="force la cadence de lecture")
    args = ap.parse_args()

    import mujoco
    import mujoco.viewer

    d = np.load(args.clip, allow_pickle=True)
    qpos_robot = np.asarray(d["joint_pos"], dtype=float)  # (T, 36) = 7 racine wxyz + 29 dof
    n_frames = qpos_robot.shape[0]
    fps = args.fps or float(np.atleast_1d(d["fps"])[0])

    has_obj = "object_pos_w" in d.files
    obj_pos = np.asarray(d["object_pos_w"], float) if has_obj else None
    obj_quat = np.asarray(d["object_quat_w"], float) if has_obj else None
    contact = np.asarray(d["object_ref_contact"], bool) if "object_ref_contact" in d.files else None

    box_half = 0.18
    support_mesh = support_pos = support_quat = None
    if "support_pos_w" in d.files:
        support_pos = np.asarray(d["support_pos_w"], float)
        support_quat = np.asarray(d["support_quat_w"], float)
        name = str(d["support_mesh"]) if "support_mesh" in d.files else ""
        cand = Path(args.clip).parent / name
        if name and cand.exists():
            support_mesh = cand
        else:
            print(f"[!] mesh de table '{name}' introuvable, table non affichee")

    model = build_model(box_half, support_mesh)
    data = mujoco.MjData(model)

    def adr(name: str) -> int:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        return model.jnt_qposadr[jid]

    box_adr = adr("carry_box_joint") if has_obj else None
    if support_mesh is not None:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "support_table")
        model.body_pos[bid] = support_pos
        model.body_quat[bid] = support_quat

    # Resume des plages de portage, imprime avant l'ouverture de la fenetre.
    if contact is not None:
        edges = np.diff(np.concatenate([[0], contact.astype(int), [0]]))
        starts, ends = np.where(edges == 1)[0], np.where(edges == -1)[0] - 1
        print(f"\nclip : {n_frames} frames a {fps:g} fps ({n_frames / fps:.2f} s)")
        print(f"portage sur {int(contact.sum())} frames ({100 * contact.mean():.1f} %)")
        for s, e in zip(starts, ends):
            print(f"  frames {s:4d} -> {e:4d}   ({s / fps:6.2f} s -> {e / fps:6.2f} s,  {e - s + 1} frames)")
    print("\nESPACE pause · fleches +/-1 frame en pause · R retour a 0\n")

    state = {"t": 0, "paused": False}

    def on_key(keycode: int) -> None:
        if keycode == 32:  # espace
            state["paused"] = not state["paused"]
        elif keycode in (262, 263):  # droite / gauche
            state["t"] = int(np.clip(state["t"] + (1 if keycode == 262 else -1), 0, n_frames - 1))
            state["paused"] = True
        elif keycode in (82, 114):  # R
            state["t"] = 0

    with mujoco.viewer.launch_passive(model, data, key_callback=on_key) as viewer:
        last = time.perf_counter()
        while viewer.is_running():
            t = state["t"]
            data.qpos[:36] = qpos_robot[t]
            if has_obj:
                data.qpos[box_adr : box_adr + 3] = obj_pos[t]
                data.qpos[box_adr + 3 : box_adr + 7] = obj_quat[t]
            data.qvel[:] = 0.0
            mujoco.mj_forward(model, data)

            held = "" if contact is None else ("  [PORTE]" if contact[t] else "  [libre]")
            viewer.user_scn.ngeom = 0
            viewer.sync()
            print(f"\rframe {t:4d}/{n_frames - 1}   t={t / fps:6.2f}s{held}   ", end="", flush=True)

            now = time.perf_counter()
            time.sleep(max(0.0, 1.0 / fps - (now - last)))
            last = time.perf_counter()
            if not state["paused"]:
                state["t"] = (t + 1) % n_frames
    print()


if __name__ == "__main__":
    main()
