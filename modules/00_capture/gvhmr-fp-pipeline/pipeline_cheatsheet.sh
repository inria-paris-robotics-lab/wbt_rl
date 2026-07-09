#!/usr/bin/env bash
# =====================================================================
#  WBT pipeline — aide-mémoire complet (vidéo RGB-D -> SMPL corps+mains+objet)
#  Lance les étapes UNE PAR UNE. Change juste CLIP.
# =====================================================================

# ---- Variables (copier en tête de session) --------------------------
CLIP=femto06
C="$HOME/wbt_clips/$CLIP"
PIPE="$HOME/Documents/wbt_rl/modules/00_capture/gvhmr-fp-pipeline"
SMPLX="$HOME/GVHMR/inputs/checkpoints/body_models"
# pythons par env :
PY_BASE=python                                                        # capture Femto (pyorbbecsdk + opencv)
GV="$HOME/.wbt_deps/miniconda3/bin/conda run -n gvhmr --no-capture-output python"
FP="$HOME/.wbt_deps/miniconda3/bin/conda run -n foundationpose --no-capture-output python"
PY_MP="$HOME/.wbt_deps/miniconda3/envs/mp/bin/python"                 # MediaPipe (boîtes mains)
PY_HAMER="$HOME/.wbt_deps/miniconda3/envs/hamer/bin/python"           # HaMeR (orientation mains)
PY_HOLO="$HOME/.holonew_deps/miniconda3/envs/holonew/bin/python"      # smplx + viser (correction + affichage)

# =====================================================================
#  SAUVEGARDE :  $C/{rgb,depth,cam_K.txt,video.mp4,masks,mesh}
#                $C/gvhmr/video/{hmr4d_results.pt, hmr4d_results_wrist.pt, 1_incam.mp4, 2_global.mp4}
#                $C/{hands.npz, hands_hamer.npz, fp_out/ob_in_cam/*.txt, clip.npz}
# =====================================================================

# ------ 1. CAPTURE (Femto Mega, alim DC branchée) --------------------
#   fenêtre AIMING -> ESPACE pour enregistrer -> q pour arrêter
$PY_BASE "$PIPE/capture_femto.py" --out "$C"

# ------ 2. OUVRIR LES MP4 (une fois : sudo apt install -y vlc) --------
vlc "$C/video.mp4"                          # vidéo couleur brute
vlc "$C/gvhmr/video/1_incam.mp4"            # overlay mesh SMPL (après GVHMR)

# ------ 3. GVHMR : corps -> SMPL  (--f_mm 22 = vraie focale Femto) ----
#   si tu re-filmes ce CLIP : d'abord  rm -rf "$C/gvhmr"  (sinon cache périmé)
cd "$HOME/GVHMR" && $GV tools/demo/demo.py \
    --video "$C/video.mp4" -s --f_mm 22 --output_root "$C/gvhmr"

# ------ 4. VISER : corps seul  -> http://localhost:8080  (Ctrl-C stop)
fuser -k 8080/tcp 2>/dev/null
$PY_HOLO "$PIPE/view_smpl_viser.py" \
    --pt "$C/gvhmr/video/hmr4d_results.pt" --model-dir "$SMPLX" --coord global --port 8080

# =====================================================================
#  MAINS (orientation des poignets)  : MediaPipe -> HaMeR -> correction
# =====================================================================
# ------ 5a. MediaPipe : boîtes des mains (env mp) ---------------------
$PY_MP "$PIPE/extract_hands.py" --clip-dir "$C"

# ------ 5b. HaMeR : orientation des mains (env hamer) -----------------
$PY_HAMER "$PIPE/extract_hands_hamer.py" \
    --pt "$C/gvhmr/video/hmr4d_results.pt" --hands "$C/hands.npz" \
    --clip-dir "$C" --model-dir "$SMPLX"

# ------ 5c. Correction du poignet SMPL (env holonew) -----------------
$PY_HOLO "$PIPE/wrist_orient.py" \
    --pt "$C/gvhmr/video/hmr4d_results.pt" --hands "$C/hands_hamer.npz" \
    --clip-dir "$C" --model-dir "$SMPLX" \
    --out "$C/gvhmr/video/hmr4d_results_wrist.pt" --smooth 15

# ------ 5d. VISER : corps + poignets corrigés ------------------------
fuser -k 8080/tcp 2>/dev/null
$PY_HOLO "$PIPE/view_smpl_viser.py" \
    --pt "$C/gvhmr/video/hmr4d_results_wrist.pt" --model-dir "$SMPLX" --coord global --port 8080

# =====================================================================
#  OBJET (si le clip a un objet manipulé)  : masque -> FoundationPose -> fuse
# =====================================================================
# ------ 6a. CAD (cube 36 cm) dans le clip ----------------------------
cp "$HOME/wbt_clips/box36.obj" "$C/mesh/"

# ------ 6b. Masque de l'objet sur la frame 0 (env base, opencv) ------
#   trace un rectangle autour de l'objet -> 's' pour sauver
$PY_BASE "$PIPE/make_mask.py" --clip-dir "$C"

# ------ 6c. FoundationPose : objet -> pose 6DoF (env foundationpose) --
$FP "$PIPE/run_foundationpose.py" \
    --mesh_file "$C/mesh/box36.obj" --test_scene_dir "$C" \
    --debug_dir "$C/fp_out" --shorter-side 480 --save-vis

# ------ 6d. Fuse : corps (poignets corrigés) + objet -> clip.npz -----
$PY_HOLO "$PIPE/fuse.py" \
    --gvhmr-pt "$C/gvhmr/video/hmr4d_results_wrist.pt" \
    --fp-poses "$C/fp_out/ob_in_cam" \
    --out "$C/clip.npz" --fps 30 --coord world \
    --object-mesh "$C/mesh/box36.obj" --smpl-model-dir "$SMPLX" --smooth-object 15

# ------ 6d. Fuse : corps (poignets non corrigés) + objet -> clip.npz -----
$PY_HOLO "$PIPE/fuse.py" \
    --gvhmr-pt "$C/gvhmr/video/hmr4d_results.pt" \
    --fp-poses "$C/fp_out/ob_in_cam" \
    --out "$C/clip.npz" --fps 30 --coord world \
    --object-mesh "$C/mesh/box36.obj" --smpl-model-dir "$SMPLX" --smooth-object 15

# ------ 6e. VISER : corps + objet ensemble --------------------------
fuser -k 8080/tcp 2>/dev/null
$PY_HOLO "$PIPE/view_clip_viser.py" \
    --clip "$C/clip.npz" --model-dir "$SMPLX" --port 8080
