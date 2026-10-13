#!/usr/bin/env bash
# Two-arm sim data on a GPU pod: collect scripted demos, preview the restyle,
# then export a LeRobot dataset shaped like the real one with part of it restyled.
#
#   bash tools/pod_bimanual.sh                 # all steps
#   STEP=preview bash tools/pod_bimanual.sh    # one step: collect | preview | export
#
# Settings (environment variables):
#   SCENE=cad  EPISODES=300  WORKERS=16  COPIES=3  OPERATOR_SHADOWS=0.5  RESTYLE=0.3  RESTYLE_WORKERS=4
#   SCENE=cad uses Zeb's real CAD parts and place-then-push controller (sim/bimanual_cad.py,
#   ~31 s episodes); SCENE=standin uses the simple stand-in parts (sim/bimanual.py, ~17 s).
#   COPIES renders every demo that many times (copy 0 as estimated, the rest with
#   random lighting, operator shadows and webcam response), so 300 demos -> 900 episodes.
#   WORK=/root/bimanual_$SCENE   (container disk: fast, not kept after the pod stops)
#   OUT=/workspace/bimanual_$SCENE   (network volume: the preview and final dataset)
set -euo pipefail
cd "$(dirname "$0")/.."
SCENE=${SCENE:-cad}
EPISODES=${EPISODES:-300}
WORKERS=${WORKERS:-16}
COPIES=${COPIES:-3}
OPERATOR_SHADOWS=${OPERATOR_SHADOWS:-0.5}
RESTYLE=${RESTYLE:-0.3}
RESTYLE_WORKERS=${RESTYLE_WORKERS:-4}  # each loads its own diffusion models; 8 ran out of memory on an 80 GB A100
WORK=${WORK:-/root/bimanual_$SCENE}
OUT=${OUT:-/workspace/bimanual_$SCENE}
STEP=${STEP:-all}
export MUJOCO_GL=${MUJOCO_GL:-egl}
mkdir -p "$WORK" "$OUT"

if [[ $STEP == all || $STEP == collect ]]; then
  if [[ $SCENE == cad ]]; then COLLECT=tools/collect_bimanual_cad.py; else COLLECT=tools/collect_bimanual.py; fi
  python "$COLLECT" --output "$WORK/episodes" --episodes "$EPISODES" --workers "$WORKERS"
fi
if [[ $STEP == all || $STEP == preview ]]; then
  # Three frames, per camera: estimated lab | augmented | augmented + restyled | real. Look before exporting.
  python tools/export_bimanual_dataset.py preview --episodes "$WORK/episodes" --output "$OUT/preview.jpg"
  python tools/export_bimanual_dataset.py preview --episodes "$WORK/episodes" --index 1 --seed 1 \
    --output "$OUT/preview_2.jpg"
fi
if [[ $STEP == all || $STEP == export ]]; then
  python tools/export_bimanual_dataset.py export --episodes "$WORK/episodes" \
    --output "$OUT/lerobot_panthera_bimanual_sim_20hz" --rendered "$WORK/rendered" \
    --copies "$COPIES" --operator-shadows "$OPERATOR_SHADOWS" \
    --restyle-fraction "$RESTYLE" --workers "$RESTYLE_WORKERS"
fi
