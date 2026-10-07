#!/usr/bin/env bash
# Two-arm sim data on a GPU pod: collect scripted demos, preview the restyle,
# then export a LeRobot dataset shaped like the real one with part of it restyled.
#
#   bash tools/pod_bimanual.sh                 # all steps
#   STEP=preview bash tools/pod_bimanual.sh    # one step: collect | preview | export
#
# Settings (environment variables):
#   EPISODES=300  WORKERS=16  RESTYLE=0.3  RESTYLE_WORKERS=4
#   WORK=/root/bimanual   (container disk: fast, not kept after the pod stops)
#   OUT=/workspace/bimanual   (network volume: the preview and final dataset)
set -euo pipefail
cd "$(dirname "$0")/.."
EPISODES=${EPISODES:-300}
WORKERS=${WORKERS:-16}
RESTYLE=${RESTYLE:-0.3}
RESTYLE_WORKERS=${RESTYLE_WORKERS:-4}
WORK=${WORK:-/root/bimanual}
OUT=${OUT:-/workspace/bimanual}
STEP=${STEP:-all}
export MUJOCO_GL=${MUJOCO_GL:-egl}
mkdir -p "$WORK" "$OUT"

if [[ $STEP == all || $STEP == collect ]]; then
  python tools/collect_bimanual.py --output "$WORK/episodes" --episodes "$EPISODES" --workers "$WORKERS"
fi
if [[ $STEP == all || $STEP == preview ]]; then
  # Three frames, per camera: sim render | restyled | real frame. Look before exporting.
  python tools/export_bimanual_dataset.py preview --episodes "$WORK/episodes" --output "$OUT/preview.jpg"
  python tools/export_bimanual_dataset.py preview --episodes "$WORK/episodes" --index 1 --seed 1 \
    --output "$OUT/preview_2.jpg"
fi
if [[ $STEP == all || $STEP == export ]]; then
  python tools/export_bimanual_dataset.py export --episodes "$WORK/episodes" \
    --output "$OUT/lerobot_panthera_bimanual_sim_20hz" --rendered "$WORK/rendered" \
    --restyle-fraction "$RESTYLE" --workers "$RESTYLE_WORKERS"
fi
