#!/usr/bin/env bash
# Set up a fresh GPU pod and train ACT and VLA-Adapter side by side.
#
#   bash tools/pod_run_all.sh [ACT_STEPS] [VLA_STEPS]
#
# Expects a persistent volume at /workspace. The script moves itself into a
# detached tmux session, so it survives an SSH disconnect. Setup runs once in
# that session, then ACT and VLA-Adapter train in their own tmux windows. All
# output is logged under /workspace/logs. Rerunning skips completed setup.
set -euo pipefail

ACT_STEPS=${1:-100000}
VLA_STEPS=${2:-20000}
REPO_URL=https://github.com/Jessego5/RobotArmTraining.git
REPO=/workspace/RobotArmTraining
LOGS=/workspace/logs
export HF_HOME=/workspace/hf UV_CACHE_DIR=/workspace/.uv-cache MUJOCO_GL=egl

if [ "$(df --output=source /workspace | tail -1)" = overlay ]; then
    echo "No network volume is mounted at /workspace; redeploy the pod with one." >&2
    exit 1
fi

if [ -z "${TMUX:-}" ]; then
    command -v tmux >/dev/null || { apt-get update -qq && apt-get install -y -qq tmux; }
    mkdir -p "$LOGS"
    script=$(readlink -f "$0")
    tmux new-session -d -s pod -n setup \
        "bash '$script' $ACT_STEPS $VLA_STEPS 2>&1 | tee $LOGS/setup.log; exec bash"
    echo "Started in tmux session 'pod'. Attach: tmux attach -t pod"
    echo "Logs: $LOGS/{setup,act,vla}.log"
    exit 0
fi

apt-get update -qq
apt-get install -y -qq libegl1 libgl1 libglib2.0-0 rsync
pip install -q uv

[ -d "$REPO/.git" ] || git clone "$REPO_URL" "$REPO"
cd "$REPO"
git pull --ff-only

if [ ! -x .venv-act/bin/python ]; then
    uv venv .venv-act --python 3.10
    uv pip install --python .venv-act/bin/python lerobot==0.4.4 mujoco
    uv pip uninstall --python .venv-act/bin/python opencv-python-headless
    uv pip install --python .venv-act/bin/python opencv-python==4.12.0.88
fi
PY=.venv-act/bin/python
$PY -c "import mujoco; mujoco.Renderer(mujoco.MjModel.from_xml_string('<mujoco/>')); print('EGL ok')"
$PY -c "import torch; assert torch.cuda.is_available(); print('CUDA ok', torch.cuda.get_device_name(0))"

$PY -c "from huggingface_hub import snapshot_download; snapshot_download('FoxNerdSaysMoo/robot-arm-learning-data', repo_type='dataset', allow_patterns=['data/episode_*/data.npz', 'data/episode_*/meta.json'], local_dir='.')"
$PY teleop/render_vla_dataset.py
if [ ! -f outputs/lerobot/panthera_stack/cache/decoded_images.uint8.npy ]; then
    $PY teleop/build_lerobot_dataset.py --force
    $PY tools/lerobot_image_cache.py outputs/lerobot/panthera_stack
fi

tmux new-window -t pod -n act \
    "cd $REPO && $PY train_act.py --steps $ACT_STEPS --output outputs/act/panthera_stack_fixed 2>&1 | tee $LOGS/act.log; exec bash"
tmux new-window -t pod -n vla \
    "cd $REPO && python3 tools/train_vla_pod.py --steps $VLA_STEPS 2>&1 | tee $LOGS/vla.log; exec bash"
echo "Setup done. ACT ($ACT_STEPS steps) and VLA-Adapter ($VLA_STEPS steps) are training."
