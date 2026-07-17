#!/usr/bin/env bash
# =====================================================================
#  WBT pipeline — aide-mémoire complet (vidéo RGB-D -> SMPL corps+mains+objet -> retargeting G1)
#  Lance les étapes UNE PAR UNE. Change juste CLIP.
# =====================================================================

# ---- Variables (copier en tête de session) --------------------------
CLIP=femto06
C="$HOME/wbt_clips/$CLIP"
PIPE="$HOME/Documents/wbt_rl/modules/00_capture/gvhmr-fp-pipeline"
SMPLX="$HOME/GVHMR/inputs/checkpoints/body_models"                    # parent de smplx/ (convention gvhmr-fp-pipeline)
V2="$HOME/Documents/wbt_rl/modules/01_retargeting/HoloNew/HoloV2"
# pythons par env :
PY_BASE=python                                                        # capture Femto (pyorbbecsdk + opencv)
GV="$HOME/.wbt_deps/miniconda3/bin/conda run -n gvhmr --no-capture-output python"
FP="$HOME/.wbt_deps/miniconda3/bin/conda run -n foundationpose --no-capture-output python"
PY_MP="$HOME/.wbt_deps/miniconda3/envs/mp/bin/python"                 # MediaPipe (boîtes mains)
PY_HAMER="$HOME/.wbt_deps/miniconda3/envs/hamer/bin/python"           # HaMeR (orientation mains)
PY_HOLO="$HOME/.holonew_deps/miniconda3/envs/holonew/bin/python"      # smplx + viser + HoloV2 (correction, fuse, sfu, retarget)
PY_SAM2="$HOME/.wbt_deps/miniconda3/envs/sam2/bin/python"             # SAM 2 (masque objet frame 0, checkpoint ~/sam2)

# =====================================================================
#  SAUVEGARDE :  $C/{rgb,depth,cam_K.txt,video.mp4,masks,mesh}
#                $C/gvhmr/video/{hmr4d_results.pt, hmr4d_results_wrist.pt, hmr4d_results_contact.pt, 1_incam.mp4, 2_global.mp4}
#                $C/{hands.npz, hands_hamer.npz, fp_out/ob_in_cam/*.txt, clip.npz, mesh/support.obj, ${CLIP}_amass.npz, ${CLIP}_qpos.npz}
# =====================================================================

# ------ 1. CAPTURE (Femto Mega, alim DC branchée) --------------------
#   fenêtre AIMING -> ESPACE pour enregistrer -> q pour arrêter
#   défauts SMPL : 1080p@30, anti-flicker 50Hz, auto-expo plafonnée, filtres depth ON.
#   sujet ~2.2 m (corps entier) BIEN ÉCLAIRÉ ; dans l'AIMING tu dois apparaître COLORÉ (pas noir) dans le panneau depth.
#   clip AVEC objet (FoundationPose) : ajoute --depth-min 300 --depth-max 2800  (2800 couvre 2.2 m ; 2000 te couperait).
#   caisse centrée (cône NFOV plus étroit que la couleur). Support/table en fin : pose la caisse et laisse-la
#   IMMOBILE ~15 frames (déclenche la détection du repos qui dérive le support à l'étape 7).
$PY_BASE "$PIPE/capture_femto.py" --out "$C"            # objet : ... --depth-min 300 --depth-max 2800

# ------ 2. OUVRIR LES MP4 (une fois : sudo apt install -y vlc) --------
vlc "$C/video.mp4"                          # vidéo couleur brute
vlc "$C/gvhmr/video/1_incam.mp4"            # overlay mesh SMPL (après GVHMR)

# ------ 3. GVHMR : corps -> SMPL  (--f_mm 22 = vraie focale Femto) ----
#   si tu re-filmes ce CLIP : d'abord  rm -rf "$C/gvhmr"  (sinon cache périmé)
cd "$HOME/GVHMR" && $GV tools/demo/demo.py \
    --video "$C/video.mp4" -s --f_mm 22 --output_root "$C/gvhmr"

# ------ 4. VISER : corps seul (sanity check)  -> http://localhost:8080 (Ctrl-C stop) ---
fuser -k 8080/tcp 2>/dev/null
$PY_HOLO "$PIPE/view_smpl_viser.py" \
    --pt "$C/gvhmr/video/hmr4d_results.pt" --model-dir "$SMPLX" --coord global --port 8080

# =====================================================================
#  MAINS (orientation des poignets, optionnel)  : MediaPipe -> HaMeR -> correction
#  Passe direct à l'étape 7 (amass) avec hmr4d_results.pt si tu n'en as pas besoin.
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
#  OBJET (uniquement si le clip a un objet manipulé) : masque -> FoundationPose ->
#  contact (grasp rigide + SDF) -> fuse -> viser. Clip sans objet : saute direct à 7.
# =====================================================================
# ------ 6a. CAD (cube 36 cm) dans le clip ----------------------------
cp "$HOME/wbt_clips/box36.obj" "$C/mesh/"

# ------ 6b. Masque de l'objet sur la frame 0 (SAM 2, env sam2) -------
#   clique l'objet : G=point +, D=point - ; 'b'=boîte ; 'u'=undo ; 's'=sauver -> masks/000000.png
#   (fallback GrabCut sans SAM :  $PY_BASE "$PIPE/make_mask.py" --clip-dir "$C")
$PY_SAM2 "$PIPE/make_mask_sam2.py" --clip-dir "$C"

# ------ 6c. FoundationPose : objet -> pose 6DoF (env foundationpose) --
$FP "$PIPE/run_foundationpose.py" \
    --mesh_file "$C/mesh/box36.obj" --test_scene_dir "$C" \
    --debug_dir "$C/fp_out" --shorter-side 640 --save-vis

# ------ 6d. (optionnel, OUTIL LOCAL — contact_opt.py n'est PAS versionné dans ce dépôt) --------
#   Optimisation grasp rigide + SDF (patch paume = contact garanti) sur le poignet SMPL, pour un
#   contact plus propre que le suivi HaMeR brut. Script gardé en local (tuning ad hoc), pas commité.
#   Si tu ne l'as pas : saute à 6e avec --gvhmr-pt hmr4d_results_wrist.pt (ou hmr4d_results.pt).
#   --pt : hmr4d_results_wrist.pt (si 5c a tourné) sinon hmr4d_results.pt (= "sans suivi des mains")
#   défauts calés portage (fenêtre auto étendue sur la prise, contact ~100%). Options utiles :
#     --carry-frames 55:158                    ne contraindre QUE cette fenêtre -> bouts LIBRES = arc GVHMR (début/fin)
#     --repel-frames "0:45,168:193" --repel-clearance 0.04   éloigner la main de l'objet aux bouts (évite de
#            frôler/bouger l'objet) ; garder ces fenêtres HORS du fondu carry (edge-ramp 10 -> ~[45:55] et [158:168])
#     --free-wrist    poignet s'aplatit sur la face (chemin sans HaMeR)      --free-torso   flexion du tronc (portage bas)
$PY_HOLO "$PIPE/contact_opt.py" \
    --pt "$C/gvhmr/video/hmr4d_results_wrist.pt" \
    --fp-poses "$C/fp_out/ob_in_cam" --object-mesh "$C/mesh/box36.obj" \
    --clip-dir "$C" --model-dir "$SMPLX" \
    --out "$C/gvhmr/video/hmr4d_results_contact.pt"

# ------ 6e. Fuse : corps + objet -> clip.npz -------------------------
#   cam->monde PAR FRAME (objet collé au corps en monde) ; --object-hold-until -1 = auto
#   (fige la pose monde de l'objet tant qu'il ne bouge pas, avant la prise -> pas de dérive au début)
#   --gvhmr-pt : hmr4d_results_contact.pt (si 6d a tourné) sinon hmr4d_results_wrist.pt / hmr4d_results.pt
$PY_HOLO "$PIPE/fuse.py" \
    --gvhmr-pt "$C/gvhmr/video/hmr4d_results_contact.pt" \
    --fp-poses "$C/fp_out/ob_in_cam" \
    --out "$C/clip.npz" --fps 30 --coord world \
    --object-mesh "$C/mesh/box36.obj" --smpl-model-dir "$SMPLX" --smooth-object 15

# ------ 6f. VISER : corps + objet ensemble --------------------------
fuser -k 8080/tcp 2>/dev/null
$PY_HOLO "$PIPE/view_clip_viser.py" \
    --clip "$C/clip.npz" --model-dir "$SMPLX" --port 8080

# =====================================================================
#  EXPORT amass (nouvelle version HoloV2 : loader 'amass', PLUS 'sfu').
#  Corps seul par défaut ; --clip clip.npz EMBARQUE l'objet (il traverse
#  le retargeter -> object_poses dans le qpos).
#  (ancien : clip_to_sfu.py + --dataset sfu, pour l'ancienne version HoloV2)
# =====================================================================
# ------ 7. Export amass (choisis le .pt le plus abouti dont tu disposes) ----
#   sans objet : hmr4d_results_wrist.pt ou hmr4d_results.pt  (SANS --clip)
#   avec objet : hmr4d_results_contact.pt (6d) + --clip "$C/clip.npz"
#   + support statique sous la caisse posée : --support (dessus/centre dérivés de la caisse au repos en fin).
#     --support-shape table (plateau 4 pieds 0.90x0.60 orienté cap robot) ou box (défaut) ;
#     --support-size 0.9x0.6 (empreinte) ; --support-yaw-offset 90 (tourne la table d'un quart de tour).
#   LOCOMOTION ONLY : exporte CORPS SEUL (enlève --clip et --support ; l'objet n'est pas utilisé au training).
$PY_HOLO "$PIPE/clip_to_amass.py" \
    --pt "$C/gvhmr/video/hmr4d_results_contact.pt" \
    --model-dir "$SMPLX" \
    --clip "$C/clip.npz" \
    --support --support-shape table --support-yaw-offset 90 \
    --out "$C/${CLIP}_amass.npz"

# =====================================================================
#  RETARGETING (HoloV2 : notre retargeter, ni GMR ni holosoma)
#  --model-dir ici = "$SMPLX/smplx" (le dossier smplx/ lui-même, PAS son parent comme au-dessus)
# =====================================================================
# ------ 8a. VISER : voir le G1 résolu -> http://localhost:8080 -------
#   --tr-base-pos/--tr-joints : rayons de confiance SQP réduits (défaut HoloV2 0.05/0.10
#   fait boucler la frame 0 en cycle limite ~5-8cm hors optimum sur les clips sans objet ;
#   voir cette conversation pour le détail). Enlève les 4 flags --tr-*/--solve-iters* pour
#   retrouver le comportement stock HoloV2.
cd "$V2"
fuser -k 8080/tcp 2>/dev/null
$PY_HOLO -m src.viz.app --dataset amass \
    --motion-path "$C/${CLIP}_amass.npz" \
    --model-dir "$SMPLX/smplx" \
    --frame-step 1 --max-frames 400 --solve \
    --tr-base-pos 0.01 --tr-joints 0.02 --solve-iters-first 60 --solve-iters 8

# ------ 8b. Sauver la trajectoire G1 (qpos) sur disque ---------------
#   mêmes petits rayons de confiance par défaut (--tr-base-pos/--tr-base-rot/--tr-joints/
#   --iters-first/--iters overrideables) ; --backend proxqp par défaut (pip install proxsuite
#   sinon --backend cvxpy, déjà dans l'env, juste plus lent)
$PY_HOLO "$PIPE/retarget_to_qpos.py" \
    --dataset amass --motion-path "$C/${CLIP}_amass.npz" \
    --model-dir "$SMPLX/smplx" \
    --out "$C/${CLIP}_qpos.npz"

# =====================================================================
#  TRAINING (wbt_rl : scripts/train.py, retargeter holosoma_custom, robot G1_29dof)
#  Vérifié : le format attendu (holosoma_custom_qpos_npz = qpos + fps) est déjà celui de
#  retarget_to_qpos.py, et l'ordre des joints actionnés de HoloV2 G1_29dof.urdf est identique
#  à celui du G1_29dof.urdf de holosoma_custom (seule diff : 4 frames fixes décoratives bout
#  de doigt côté holosoma_custom, aucun DOF en plus) -- un qpos HoloV2 s'utilise tel quel.
# =====================================================================
WBT_ROOT="$HOME/Documents/wbt_rl"

# ------ 9a. Empaqueter le qpos en run d'entraînement -----------------
#   -> data/01_retargeted_motions/SFU_G1_29dof/HOLOSOMA_CUSTOM/run_${CLIP}/${CLIP}_output_raw.npz
#   (SFU = juste un LABEL de dossier, pas le loader). NB : ne met PAS à jour le symlink 'latest'
#   (ajoute --latest si tu veux) -> lance 9b avec --retarget-run run_${CLIP} EXPLICITE, pas 'latest'.
$PY_HOLO "$PIPE/qpos_to_training_run.py" \
    --qpos "$C/${CLIP}_qpos.npz" \
    --name "$CLIP" --dataset SFU --robot G1_29dof --repo-root "$WBT_ROOT"

# ------ 9b. Lancer le training (env wbt_rl) ---------------------------
#   LOCOMOTION ONLY (robot_only) = PAS de --with-object -> l'objet est retiré, le robot suit juste le mouvement.
#   AVEC objet : ajoute --with-object (et un export amass qui embarque l'objet, étape 7 avec --clip).
#   une fois par shell : cd "$WBT_ROOT" && source scripts/activate_wbt.sh
cd "$WBT_ROOT" && python scripts/train.py \
    --dataset SFU --robot G1_29dof \
    --retargeter holosoma_custom --trainer holosoma_custom \
    --simulator isaacsim --retarget-run "run_${CLIP}" --no-video
