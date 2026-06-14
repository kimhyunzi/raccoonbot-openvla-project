#!/usr/bin/env bash

RUN_DIR="openvla-runs/openvla-7b+raccoon_pick_place+b16+lr-0.0005+lora-r32+dropout-0.0--raccoon-880ep-multitask-18h-s3000--image_aug"
SNAP_DIR="openvla-runs/checkpoint_snapshots/raccoon-880ep-multitask-18h-s3000"
LOG="logs/finetune_880_multitask_18h_s3000.log"

mkdir -p "$SNAP_DIR"

copy_snapshot () {
  STEP="$1"

  if [ -d "$SNAP_DIR/step${STEP}" ]; then
    echo "[SKIP] step${STEP} already exists"
    return
  fi

  echo "[SNAPSHOT] copying step ${STEP}..."
  rm -rf "$SNAP_DIR/step${STEP}_tmp"
  mkdir -p "$SNAP_DIR/step${STEP}_tmp"

  cp -a "$RUN_DIR/." "$SNAP_DIR/step${STEP}_tmp/"

  rm -rf "$SNAP_DIR/step${STEP}"
  mv "$SNAP_DIR/step${STEP}_tmp" "$SNAP_DIR/step${STEP}"

  echo "[DONE] step${STEP} saved"
  du -sh "$SNAP_DIR/step${STEP}"
}

echo "[WATCH] watching $LOG"
echo "[WATCH] snapshots will be saved under $SNAP_DIR"

while true; do
  for STEP in 15000 18000; do
    if grep -q "Saved Model Checkpoint for Step ${STEP}" "$LOG"; then
      copy_snapshot "$STEP"
    fi
  done

  if grep -q "Max step 18000 reached" "$LOG"; then
    echo "[WATCH] training finished."
    break
  fi

  if ! pgrep -af "raccoon-880ep-multitask-18h-s3000|finetune.py|torchrun" | grep -q "2023741061"; then
    echo "[WATCH] training process not found. stop watching."
    break
  fi

  sleep 60
done
