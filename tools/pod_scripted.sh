#!/usr/bin/env bash
# Train ACT on scripted demonstrations and score it against the teleop model.
#
#   bash tools/pod_scripted.sh [EPISODES] [STEPS] [KEEP_EVERY] [AUGMENT]
#
# Generates EPISODES successful scripted demos (tools/scripted_demos.py),
# renders them, builds a LeRobot dataset for that many episodes, trains ACT
# into outputs/act/scripted_<EPISODES>ep_<STEPS> keeping a checkpoint every
# KEEP_EVERY steps, then evaluates every checkpoint alongside the earlier
# models on the same layouts. AUGMENT (default 0) is the fraction of demos
# with perturbations, missed-grasp retries and corrective labels; augmented
# sets get their own demo, dataset, run, tmux window and log ("_aug<AUGMENT>").
# Runs after tools/pod_run_all.sh; moves itself into tmux window
# "scripted<tag>" and logs to /workspace/logs/scripted<tag>.log.
set -euo pipefail

EPISODES=${1:-300}
STEPS=${2:-60000}
KEEP_EVERY=${3:-10000}
AUGMENT=${4:-0}
TAG=""
[ "$AUGMENT" = 0 ] || TAG="_aug${AUGMENT}"
REPO=/workspace/RobotArmTraining
LOGS=/workspace/logs
export HF_HOME=/workspace/hf MUJOCO_GL=egl

if [ -z "${TMUX:-}" ]; then
    mkdir -p "$LOGS"
    script=$(readlink -f "$0")
    command="bash '$script' $EPISODES $STEPS $KEEP_EVERY $AUGMENT 2>&1 | tee $LOGS/scripted${TAG}.log; exec bash"
    if tmux has-session -t pod 2>/dev/null; then
        tmux new-window -t pod -n "scripted${TAG}" "$command"
    else
        tmux new-session -d -s pod -n "scripted${TAG}" "$command"
    fi
    echo "Started in tmux window 'scripted${TAG}'. Log: $LOGS/scripted${TAG}.log"
    exit 0
fi

cd "$REPO"
git pull --ff-only
PY=.venv-act/bin/python
DEMOS=data_scripted${TAG}
DATASET=outputs/lerobot/panthera_scripted_${EPISODES}${TAG}
RUN=outputs/act/scripted_${EPISODES}ep_${STEPS}${TAG}
INPUT=outputs/scripted_input_${EPISODES}${TAG}
RENDERED=outputs/rendered_scripted_${EPISODES}${TAG}

[ -f "$DEMOS/episode_$(printf %03d $((EPISODES - 1)))/data.npz" ] || \
    $PY tools/scripted_demos.py --output "$DEMOS" --episodes "$EPISODES" \
        --augment-fraction "$AUGMENT"
# Render only the first EPISODES demos, so a smaller rerun uses a subset.
mkdir -p "$INPUT"
for i in $(seq 0 $((EPISODES - 1))); do
    ln -sfn "$REPO/$DEMOS/episode_$(printf %03d "$i")" "$INPUT/episode_$(printf %03d "$i")"
done
$PY teleop/render_vla_dataset.py --input "$INPUT" --output "$RENDERED"
if [ ! -f "$DATASET/cache/decoded_images.uint8.npy" ]; then
    $PY teleop/build_lerobot_dataset.py --input "$RENDERED" \
        --output "$DATASET" --force --gripper-state command
    $PY tools/lerobot_image_cache.py "$DATASET"
fi

rm -rf "$RUN"
$PY train_act.py --dataset "$DATASET" --steps "$STEPS" --keep-every "$KEEP_EVERY" \
    --output "$RUN"

baseline=()
for previous in outputs/act/panthera_stack_fixed outputs/act/scripted/checkpoint_060000 \
        outputs/act/scripted_${EPISODES}ep_${STEPS}/checkpoint_$(printf %06d "$STEPS"); do
    [ "$previous" = "$RUN/checkpoint_$(printf %06d "$STEPS")" ] && continue
    [ -f "$previous/config.json" ] && baseline+=("$previous")
done
$PY tools/eval_act.py "$RUN"/checkpoint_* "${baseline[@]}" \
    --episodes 30 --output "outputs/eval/$(basename "$RUN").json"
