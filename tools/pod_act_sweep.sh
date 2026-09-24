#!/usr/bin/env bash
# Retrain ACT on the measured finger-opening state, keeping a checkpoint every
# KEEP_EVERY steps, then score every checkpoint (and the previous model as a
# baseline) on the same randomized layouts.
#
#   bash tools/pod_act_sweep.sh [STEPS] [KEEP_EVERY] [EPISODES]
#
# Runs after tools/pod_run_all.sh has set up the pod. Like that script, it
# moves itself into tmux (window "sweep" of session "pod") and logs to
# /workspace/logs/sweep.log. The results table is at the end of the log and in
# outputs/eval/sweep.json.
set -euo pipefail

STEPS=${1:-60000}
KEEP_EVERY=${2:-10000}
EPISODES=${3:-30}
REPO=/workspace/RobotArmTraining
LOGS=/workspace/logs
export HF_HOME=/workspace/hf MUJOCO_GL=egl

if [ -z "${TMUX:-}" ]; then
    mkdir -p "$LOGS"
    script=$(readlink -f "$0")
    command="bash '$script' $STEPS $KEEP_EVERY $EPISODES 2>&1 | tee $LOGS/sweep.log; exec bash"
    if tmux has-session -t pod 2>/dev/null; then
        tmux new-window -t pod -n sweep "$command"
    else
        tmux new-session -d -s pod -n sweep "$command"
    fi
    echo "Started in tmux window 'sweep'. Log: $LOGS/sweep.log"
    exit 0
fi

cd "$REPO"
git pull --ff-only
PY=.venv-act/bin/python

$PY teleop/render_vla_dataset.py
$PY teleop/build_lerobot_dataset.py --force --gripper-state measured
$PY tools/lerobot_image_cache.py outputs/lerobot/panthera_stack

rm -rf outputs/act/sweep
$PY train_act.py --steps "$STEPS" --keep-every "$KEEP_EVERY" --output outputs/act/sweep

baseline=()
[ -f outputs/act/panthera_stack_fixed/config.json ] && baseline=(outputs/act/panthera_stack_fixed)
$PY tools/eval_act.py outputs/act/sweep/checkpoint_* "${baseline[@]}" \
    --episodes "$EPISODES" --output outputs/eval/sweep.json
