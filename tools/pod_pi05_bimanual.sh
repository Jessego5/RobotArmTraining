#!/usr/bin/env bash
# Full-model pi0.5 fine-tune on the two-arm sim data (v1 + v2 exports), on a GPU pod.
#
#   STEP=setup   bash tools/pod_pi05_bimanual.sh   # pinned LeRobot env, tokenizer, pi0.5 base weights
#   STEP=dataset bash tools/pod_pi05_bimanual.sh   # merge v1 + v2 with a demo-level validation holdout
#   STEP=smoke   bash tools/pod_pi05_bimanual.sh   # 4 updates: full-model checks and peak GPU memory
#   STEP=train   bash tools/pod_pi05_bimanual.sh   # the run; rerun with STEP=resume after an interruption
#
# Needs a GPU with >= 85 GB (as the pi0.5 notebook: RTX PRO 6000 96 GB, H100 NVL, H200, B200),
# and an HF token whose account accepted the PaliGemma license
# (https://huggingface.co/google/paligemma-3b-pt-224): export HF_TOKEN=... first.
#
# Settings (environment variables):
#   WORK=/workspace/pi05_bimanual   STEPS=30000  BATCH=8  LR=2.5e-5  CHUNK=50  ACTION_STEPS=10
#   SAVE_FREQ=2000  WORKERS=8  RUN=bimanual_sim_v1v2
#   SOURCES="/workspace/bimanual_cad/lerobot_panthera_bimanual_sim_20hz /workspace/bimanual_cad_v2/lerobot_panthera_bimanual_sim_20hz"
# CHUNK=50 predicts 2.5 s of actions at 20 Hz; ACTION_STEPS=10 replans every 0.5 s.
set -euo pipefail
cd "$(dirname "$0")/.."
REPO=$(pwd)
WORK=${WORK:-/workspace/pi05_bimanual}
STEPS=${STEPS:-30000}
BATCH=${BATCH:-8}
LR=${LR:-2.5e-5}
CHUNK=${CHUNK:-50}
ACTION_STEPS=${ACTION_STEPS:-10}
SAVE_FREQ=${SAVE_FREQ:-2000}
WORKERS=${WORKERS:-8}
RUN=${RUN:-bimanual_sim_v1v2}
SOURCES=${SOURCES:-"/workspace/bimanual_cad/lerobot_panthera_bimanual_sim_20hz /workspace/bimanual_cad_v2/lerobot_panthera_bimanual_sim_20hz"}
STEP=${STEP:?set STEP=setup, dataset, smoke, train or resume}
LEROBOT_COMMIT=e624f3f7f8411ec3a02635d06e79373341e5ef35
LEROBOT=$WORK/lerobot
PY=$LEROBOT/.venv/bin/python
DATA=$WORK/dataset
MODEL=$WORK/pi05_base
mkdir -p "$WORK/tmp" "$WORK/runs" "$WORK/logs"
export UV_CACHE_DIR=$WORK/uv-cache TMPDIR=$WORK/tmp TOKENIZERS_PARALLELISM=false MUJOCO_GL=egl \
       HF_HOME=$WORK/hf-cache HF_LEROBOT_HOME=$WORK/hf-cache/lerobot PYTHONPATH=$REPO CUDA_VISIBLE_DEVICES=0

case $STEP in
setup)
  python -m pip install -q uv huggingface_hub
  if [[ ! -d $LEROBOT/.git ]]; then
    git init "$LEROBOT" && git -C "$LEROBOT" remote add origin https://github.com/huggingface/lerobot.git
  fi
  git -C "$LEROBOT" fetch --depth=1 origin $LEROBOT_COMMIT
  git -C "$LEROBOT" checkout --detach $LEROBOT_COMMIT
  (cd "$LEROBOT" && python -m uv sync --frozen --extra pi --extra training --python 3.12)
  $PY - <<'EOF'
import torch
assert torch.cuda.is_available()
p = torch.cuda.get_device_properties(0)
print(p)
assert p.total_memory / 2**30 >= 85, 'full pi0.5 fine-tuning needs a >= 85 GB GPU (this one is smaller)'
assert torch.cuda.is_bf16_supported()
EOF
  : "${HF_TOKEN:?export HF_TOKEN (an account that accepted the PaliGemma license)}"
  $PY - <<EOF
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer
AutoTokenizer.from_pretrained('google/paligemma-3b-pt-224')
snapshot_download('lerobot/pi05_base', local_dir='$MODEL', allow_patterns=['config.json', 'model.safetensors'])
print('tokenizer and pi0.5 base weights ready')
EOF
  ;;
dataset)
  for s in $SOURCES; do [[ -f $s/COMPLETE ]] || { echo "missing complete export: $s"; exit 1; }; done
  $PY tools/build_pi05_bimanual_dataset.py --sources $SOURCES --output "$DATA"
  ;;
smoke|train)
  [[ -f $DATA/COMPLETE ]] || { echo "build the dataset first: STEP=dataset"; exit 1; }
  EVAL_SPLIT=$($PY -c "import json;print(repr(json.load(open('$DATA/meta/pi05_split.json'))['eval_split']))")
  if [[ $STEP == smoke ]]; then OUT=$WORK/runs/${RUN}_smoke_$(date +%Y%m%d_%H%M%S); N=4; FREQ=4; EXTRA="--save_checkpoint=false"
  else OUT=$WORK/runs/$RUN; N=$STEPS; FREQ=$SAVE_FREQ; EXTRA=""; fi
  $PY -u tools/train_pi05_full.py \
    --dataset.repo_id=local/panthera_bimanual_pi05 --dataset.root="$DATA" --dataset.eval_split="$EVAL_SPLIT" \
    --dataset.use_imagenet_stats=false --dataset.video_backend=pyav \
    --policy.type=pi05 --policy.pretrained_path="$MODEL" \
    --policy.freeze_vision_encoder=false --policy.train_expert_only=false \
    --policy.use_relative_actions=false --policy.empty_cameras=0 \
    --policy.gradient_checkpointing=true --policy.compile_model=false \
    --policy.dtype=bfloat16 --policy.device=cuda --policy.push_to_hub=false \
    '--policy.normalization_mapping={"ACTION":"QUANTILES","STATE":"QUANTILES","VISUAL":"IDENTITY"}' \
    --policy.chunk_size=$CHUNK --policy.n_action_steps=$ACTION_STEPS --policy.optimizer_lr=$LR \
    --policy.scheduler_decay_steps=$STEPS --policy.scheduler_warmup_steps=$(( N / 10 < 1000 ? (N / 10 > 0 ? N / 10 : 1) : 1000 )) \
    --accelerator.gradient_accumulation.steps=1 --accelerator.mixed_precision=bf16 \
    --output_dir="$OUT" --job_name=$RUN --batch_size=$BATCH --num_workers=$WORKERS --prefetch_factor=2 \
    --steps=$N --save_freq=$FREQ --eval_steps=$FREQ --max_eval_samples=128 --env_eval_freq=0 --log_freq=20 \
    --seed=0 --wandb.enable=false $EXTRA 2>&1 | tee "$WORK/logs/$(basename "$OUT").log"
  if [[ $STEP == smoke ]]; then cat "$OUT/full_model_audit.json" "$OUT/peak_memory.json"; fi
  ;;
resume)
  CFG=$WORK/runs/$RUN/checkpoints/last/pretrained_model/train_config.json
  [[ -f $CFG ]] || { echo "no checkpoint to resume at $CFG"; exit 1; }
  $PY -u tools/train_pi05_full.py --config_path="$CFG" --resume=true --steps=$STEPS --dataset.root="$DATA" \
    --wandb.enable=false 2>&1 | tee -a "$WORK/logs/$RUN.log"
  ;;
*) echo "unknown STEP=$STEP"; exit 1 ;;
esac
