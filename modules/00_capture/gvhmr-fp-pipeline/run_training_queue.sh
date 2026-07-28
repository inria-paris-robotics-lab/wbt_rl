#!/usr/bin/env bash
# Sequential training queue: for each clip in CLIPS, fill in whatever stage is missing
# (SFU export -> HoloV2 retarget -> qpos package) then launch training. Unattended: keeps
# going past a failed/not-ready clip, logs everything, prints a summary at the end.
#
# Launch detached so it survives you logging out (tmux, recommended so you can reattach):
#   tmux new -s training
#   bash run_training_queue.sh
#   # Ctrl-b d to detach ; `tmux attach -t training` to check back in later
# or with nohup instead of tmux:
#   nohup bash run_training_queue.sh > ~/wbt_clips/training_logs/queue.out 2>&1 &
#   disown

set -uo pipefail   # NOT -e: one bad clip must not kill the rest of the queue

# ---- edit this list before you go -----------------------------------
CLIPS=(femto06 femto07 femto08 femto09 femto11)
DATASET=SFU
ROBOT=G1_29dof

# ---- fixed paths/envs -------------------------------------------------
WBT_ROOT="$HOME/Documents/wbt_rl"
PIPE="$WBT_ROOT/modules/00_capture/gvhmr-fp-pipeline"
V2="$WBT_ROOT/modules/01_retargeting/HoloNew/HoloV2"
SMPLX="$HOME/GVHMR/inputs/checkpoints/body_models"
PY_HOLO="$HOME/.holonew_deps/miniconda3/envs/holonew/bin/python"
LOGDIR="$HOME/wbt_clips/training_logs"
mkdir -p "$LOGDIR"

SUMMARY=()

for CLIP in "${CLIPS[@]}"; do
    C="$HOME/wbt_clips/$CLIP"
    LOG="$LOGDIR/${CLIP}.log"
    : > "$LOG"
    echo "===== $CLIP =====" | tee -a "$LOG"

    # pick the best available GVHMR output (contact > wrist-corrected > raw)
    PT=""
    for cand in hmr4d_results_contact.pt hmr4d_results_wrist.pt hmr4d_results.pt; do
        if [ -f "$C/gvhmr/video/$cand" ]; then PT="$C/gvhmr/video/$cand"; break; fi
    done
    if [ -z "$PT" ]; then
        echo "[$CLIP] SKIP: no GVHMR .pt found -- capture/GVHMR not run yet" | tee -a "$LOG"
        SUMMARY+=("$CLIP: SKIPPED (no .pt)")
        continue
    fi
    echo "[$CLIP] using $(basename "$PT")" | tee -a "$LOG"

    SFU="$C/${CLIP}_sfu.npz"
    if [ ! -f "$SFU" ]; then
        echo "[$CLIP] exporting SFU..." | tee -a "$LOG"
        if ! "$PY_HOLO" "$PIPE/clip_to_sfu.py" --pt "$PT" --model-dir "$SMPLX" --out "$SFU" >>"$LOG" 2>&1; then
            echo "[$CLIP] FAILED at SFU export (see $LOG)" | tee -a "$LOG"
            SUMMARY+=("$CLIP: FAILED (sfu export)")
            continue
        fi
    fi

    QPOS="$C/${CLIP}_qpos.npz"
    if [ ! -f "$QPOS" ]; then
        echo "[$CLIP] retargeting to qpos (HoloV2)..." | tee -a "$LOG"
        if ! (cd "$V2" && "$PY_HOLO" "$PIPE/retarget_to_qpos.py" --dataset sfu \
                --motion-path "$SFU" --model-dir "$SMPLX/smplx" --out "$QPOS") >>"$LOG" 2>&1; then
            echo "[$CLIP] FAILED at retarget (see $LOG)" | tee -a "$LOG"
            SUMMARY+=("$CLIP: FAILED (retarget)")
            continue
        fi
    fi

    echo "[$CLIP] packaging training run..." | tee -a "$LOG"
    if ! "$PY_HOLO" "$PIPE/qpos_to_training_run.py" --qpos "$QPOS" --name "$CLIP" \
            --dataset "$DATASET" --robot "$ROBOT" --repo-root "$WBT_ROOT" >>"$LOG" 2>&1; then
        echo "[$CLIP] FAILED at packaging (see $LOG)" | tee -a "$LOG"
        SUMMARY+=("$CLIP: FAILED (package)")
        continue
    fi

    echo "[$CLIP] training (this is the long part -- step counter below, also in $LOG)..." | tee -a "$LOG"
    # piped through tee (not >>) so the step/reward counter is visible live if you're attached
    # to the tmux session, not just buried in the log file. pipefail (set above) keeps the
    # if/else honest about train.py's real exit status despite the pipe.
    if (cd "$WBT_ROOT" && source scripts/activate_wbt.sh >/dev/null && \
            python scripts/train.py --dataset "$DATASET" --robot "$ROBOT" \
                --retargeter holosoma_custom --trainer holosoma_custom \
                --simulator isaacsim --retarget-run "run_${CLIP}" --no-video) 2>&1 | tee -a "$LOG"; then
        echo "[$CLIP] DONE" | tee -a "$LOG"
        SUMMARY+=("$CLIP: OK")
    else
        echo "[$CLIP] FAILED at training (see $LOG)" | tee -a "$LOG"
        SUMMARY+=("$CLIP: FAILED (training)")
    fi
done

echo
echo "===== SUMMARY ====="
printf '%s\n' "${SUMMARY[@]}" | tee "$LOGDIR/summary.txt"
