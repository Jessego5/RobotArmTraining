#!/usr/bin/env bash
# Train ACT on scripted demonstrations and score it against the teleop model.
#
#   bash tools/pod_scripted.sh [EPISODES] [STEPS] [KEEP_EVERY]
#
# Generates EPISODES successful scripted demos (tools/scripted_demos.py),
# renders them, builds a LeRobot dataset for that many episodes, trains ACT
# into outputs/act/scripted_<EPISODES>ep_<STEPS> keeping a checkpoint every
# KEEP_EVERY steps, then evaluates every checkpoint alongside the earlier
# models on the same layouts. Runs after tools/pod_run_all.sh; moves
# itself into tmux window "scripted" and logs to /workspace/logs/scripted.log.
set -euo pipefail

EPISODES=${1:-300}
STEPS=${2:-60000}
KEEP_EVERY=${3:-10000}
REPO=/workspace/RobotArmTraining
LOGS=/workspace/logs
export HF_HOME=/workspace/hf MUJOCO_GL=egl

if [ -z "${TMUX:-}" ]; then
    mkdir -p "$LOGS"
    script=$(readlink -f "$0")
    command="bash '$script' $EPISODES $STEPS $KEEP_EVERY 2>&1 | tee $LOGS/scripted.log; exec bash"
    if tmux has-session -t pod 2>/dev/null; then
        tmux new-window -t pod -n scripted "$command"
    else
        tmux new-session -d -s pod -n scripted "$command"
    fi
    echo "Started in tmux window 'scripted'. Log: $LOGS/scripted.log"
    exit 0
fi

cd "$REPO"
git pull --ff-only
PY=.venv-act/bin/python
DATASET=outputs/lerobot/panthera_scripted_${EPISODES}
RUN=outputs/act/scripted_${EPISODES}ep_${STEPS}

[ -f data_scripted/episode_$(printf %03d $((EPISODES - 1)))/data.npz ] || \
    $PY tools/scripted_demos.py --output data_scripted --episodes "$EPISODES"
# Render only the first EPISODES demos, so a smaller rerun uses a subset.
mkdir -p outputs/scripted_input_${EPISODES}
for i in $(seq 0 $((EPISODES - 1))); do
    ln -sfn "$REPO/data_scripted/episode_$(printf %03d "$i")" \
        "outputs/scripted_input_${EPISODES}/episode_$(printf %03d "$i")"
done
$PY teleop/render_vla_dataset.py --input outputs/scripted_input_${EPISODES} \
    --output outputs/rendered_scripted_${EPISODES}
if [ ! -f "$DATASET/cache/decoded_images.uint8.npy" ]; then
    $PY teleop/build_lerobot_dataset.py --input outputs/rendered_scripted_${EPISODES} \
        --output "$DATASET" --force --gripper-state command
    $PY tools/lerobot_image_cache.py "$DATASET"
fi

rm -rf "$RUN"
$PY train_act.py --dataset "$DATASET" --steps "$STEPS" --keep-every "$KEEP_EVERY" \
    --output "$RUN"

baseline=()
for previous in outputs/act/panthera_stack_fixed outputs/act/scripted/checkpoint_060000; do
    [ -f "$previous/config.json" ] && baseline+=("$previous")
done
$PY tools/eval_act.py "$RUN"/checkpoint_* "${baseline[@]}" \
    --episodes 30 --output "outputs/eval/$(basename "$RUN").json"
