#!/usr/bin/env python3
"""Set up and fine-tune VLA-Adapter on a plain Linux GPU machine.

This is the Colab notebook (notebooks/robot_arm_learning_finetune_colab.ipynb)
without Colab: it creates the VLA environment, applies the notebook's
VLA-Adapter patches and EGL check (executed from the notebook itself, so there
is one copy of them), renders and converts the demonstrations, downloads the
base model, and trains.  Every stage is idempotent, so rerunning resumes where
a previous invocation stopped.  Run it with any Python 3 -- it creates its own
Python 3.10 environment for everything else.

    python3 tools/train_vla_pod.py --steps 50          # end-to-end smoke test
    python3 tools/train_vla_pod.py --steps 20000
    python3 tools/train_vla_pod.py --steps 30000 --resume-run-id <RUN_ID>
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent.parent
VLA_DIR = CODE_DIR / "VLA-Adapter"
NOTEBOOK = CODE_DIR / "notebooks" / "robot_arm_learning_finetune_colab.ipynb"
RENDERED_DIR = VLA_DIR / "data/robot_arm_learning_rendered"
RLDS_DIR = VLA_DIR / "data/robot_arm_learning"
MODEL_DIR = VLA_DIR / "pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b"

DATASET_REPO = "FoxNerdSaysMoo/robot-arm-learning-data"
DATASET_REVISION = "main"
VLA_REPO = "https://github.com/OpenHelix-Team/VLA-Adapter.git"
VLA_COMMIT = "23fa0c9c159e2aa04341cdd3e924f44061311060"
MODEL_REPO = "Stanford-ILIAD/prism-qwen25-extra-dinosiglip-224px-0_5b"
INSTRUCTION = "stack the three colored cubes"
SAMPLE_HZ = 10.0
VAL_FRACTION = 0.1
VAL_SPLIT_SEED = 20260920
WANDB_PROJECT = "robot-arm-learning-panthera"


def run(command, **kwargs) -> None:
    print("+", " ".join(map(str, command)), flush=True)
    subprocess.run(list(map(str, command)), check=True, **kwargs)


def notebook_cell(marker: str) -> str:
    """Source of the one notebook code cell containing `marker`."""
    cells = [
        "".join(cell["source"])
        for cell in json.loads(NOTEBOOK.read_text())["cells"]
        if cell["cell_type"] == "code" and marker in "".join(cell["source"])
    ]
    if len(cells) != 1:
        raise SystemExit(f"expected one notebook cell containing {marker!r}, found {len(cells)}")
    return cells[0]


def checkout_vla_adapter() -> None:
    # VLA-Adapter/ may already hold rendered frames from the ACT pipeline, so
    # fetch into it in place rather than cloning into a fresh directory.
    if not (VLA_DIR / ".git").exists():
        VLA_DIR.mkdir(parents=True, exist_ok=True)
        run(["git", "init", "-q"], cwd=VLA_DIR)
        run(["git", "remote", "add", "origin", VLA_REPO], cwd=VLA_DIR)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=VLA_DIR,
                          capture_output=True, text=True).stdout.strip()
    if head != VLA_COMMIT:
        run(["git", "fetch", "-q", "--depth=1", "origin", VLA_COMMIT], cwd=VLA_DIR)
        run(["git", "checkout", "-q", "FETCH_HEAD"], cwd=VLA_DIR)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--save-freq", type=int, default=1_000)
    parser.add_argument("--val-freq", type=int, default=250)
    parser.add_argument("--val-time-limit", type=int, default=60)
    # 8 x 1 keeps the notebook's effective batch of 8 in a quarter of the
    # micro-steps; an 80 GB card has room for it next to an ACT run.
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--gradient-checkpointing",
                        action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--venv", type=Path, default=Path("/workspace/vla-env"))
    parser.add_argument("--output-root", type=Path, default=CODE_DIR / "outputs" / "vla")
    parser.add_argument("--resume-run-id", default="")
    parser.add_argument("--resume-learning-rate", type=float)
    parser.add_argument("--wandb-entity", default="")
    args = parser.parse_args()

    python = str(args.venv / "bin/python")

    # 1. Code and environment (notebook cell 5).
    checkout_vla_adapter()
    if shutil.which("uv") is None:
        run([sys.executable, "-m", "pip", "install", "-q", "uv"])
    run(["uv", "python", "install", "3.10"])
    if not args.venv.exists():
        run(["uv", "venv", args.venv, "--python", "3.10"])
    run(["uv", "pip", "install", "--python", python, "-e", VLA_DIR,
         "tensorflow-metadata==1.13.1", "protobuf==4.25.9",
         "mujoco", "opencv-python-headless", "scipy"])
    run([python, "-c", "import torch, tensorflow as tf, mujoco; "
         "assert torch.cuda.is_available(), 'torch cannot see the GPU'; "
         "print('torch', torch.__version__, 'tensorflow', tf.__version__, "
         "'mujoco', mujoco.__version__)"])

    # 2. The notebook's VLA-Adapter patches and EGL check, run as-is.
    notebook_globals = {
        "__builtins__": __builtins__, "Path": Path, "os": os,
        "subprocess": subprocess, "shutil": shutil,
        "VLA_DIR": VLA_DIR, "PYTHON": python, "run": run,
    }
    exec(notebook_cell("def insert_after(path, anchor"), notebook_globals)
    exec(notebook_cell("NVIDIA_ICD ="), notebook_globals)
    render_env = notebook_globals["render_env"]

    # 3. Demonstrations, observations and RLDS (cells 9 and 12).
    if not any((CODE_DIR / "data").glob("episode_*/data.npz")):
        run([python, "-c",
             "from huggingface_hub import snapshot_download; snapshot_download("
             f"repo_id={DATASET_REPO!r}, repo_type='dataset', revision={DATASET_REVISION!r}, "
             f"local_dir={str(CODE_DIR)!r}, allow_patterns=["
             "'data/episode_*/data.npz', 'data/episode_*/meta.json'])"])
    run([python, CODE_DIR / "teleop/render_vla_dataset.py",
         "--input", CODE_DIR / "data", "--output", RENDERED_DIR,
         "--hz", str(SAMPLE_HZ), "--size", "256"], env=render_env)
    run([python, CODE_DIR / "teleop/build_robot_arm_learning_rlds.py",
         "--rendered-dir", RENDERED_DIR, "--data-dir", RLDS_DIR,
         "--instruction", INSTRUCTION, "--val-fraction", str(VAL_FRACTION),
         "--split-seed", str(VAL_SPLIT_SEED)])

    # 4. Base model (cell 15).
    run([python, "-c",
         "from huggingface_hub import snapshot_download; "
         f"snapshot_download(repo_id={MODEL_REPO!r}, local_dir={str(MODEL_DIR)!r})"])

    # 5. Fine-tune (cell 17).
    args.output_root.mkdir(parents=True, exist_ok=True)
    run_id = args.resume_run_id or (
        "robot-arm-learning-pod-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S"))
    resume_dir = args.output_root / run_id if args.resume_run_id else None
    if resume_dir is not None and not (resume_dir / "lora_adapter").exists():
        raise SystemExit(f"nothing to resume from at {resume_dir}")
    validation_config = {
        "dataset_repo": DATASET_REPO,
        "dataset_revision": DATASET_REVISION,
        "rlds_version": "1.1.0",
        "val_fraction": VAL_FRACTION,
        "split_seed": VAL_SPLIT_SEED,
    }
    validation_config_path = args.output_root / run_id / "validation_split.json"
    if resume_dir is not None:
        if json.loads(validation_config_path.read_text()) != validation_config:
            raise SystemExit("validation split changed since this run began")
    else:
        validation_config_path.parent.mkdir(parents=True, exist_ok=True)
        validation_config_path.write_text(json.dumps(validation_config, indent=2) + "\n")
    print("RUN_ID:", run_id, "(resuming)" if resume_dir else "(new run)", flush=True)

    command = [
        args.venv / "bin/torchrun", "--standalone", "--nnodes", "1", "--nproc-per-node", "1",
        VLA_DIR / "vla-scripts/finetune.py",
        "--vlm_path", MODEL_DIR,
        "--config_file_path", VLA_DIR / "pretrained_models/configs",
        "--data_root_dir", RLDS_DIR,
        "--dataset_name", "robot_arm_learning_panthera",
        "--run_root_dir", args.output_root,
        "--run_id_override", run_id,
        "--use_film", "False",
        "--num_images_in_input", "2",
        "--use_proprio", "True",
        "--use_lora", "True",
        "--use_fz", "False",
        "--use_minivlm", "True",
        "--image_aug", "True",
        "--shuffle_buffer_size", "12000",
        "--use_val_set", "True",
        "--val_freq", str(args.val_freq),
        "--val_time_limit", str(args.val_time_limit),
        "--num_steps_before_decay", str(max(1, int(args.steps * 0.8))),
        "--max_steps", str(args.steps),
        "--save_freq", str(args.save_freq),
        "--save_latest_checkpoint_only", "True",
        "--merge_lora_during_training", "False",
        "--batch_size", str(args.batch_size),
        "--grad_accumulation_steps", str(args.grad_accum),
        "--learning_rate", "2e-4",
        "--lora_rank", "64",
        "--use_pro_version", "True",
        "--use_gradient_checkpointing", str(args.gradient_checkpointing),
        "--balance_z_loss", "True",
        "--lr_warmup_steps", "0",
        "--wandb_project", WANDB_PROJECT,
    ]
    if args.wandb_entity:
        command += ["--wandb_entity", args.wandb_entity]
    if resume_dir is not None:
        command += ["--resume_checkpoint", resume_dir]
        if args.resume_learning_rate is not None:
            command += ["--resume_learning_rate", str(args.resume_learning_rate)]
    train_env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES="0",
        WANDB_MODE="online" if args.wandb_entity else "offline",
        WANDB_RUN_ID=run_id,
        WANDB_RESUME="allow",
        PYTHONPATH=str(VLA_DIR),
        PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
        TF_CPP_MIN_LOG_LEVEL="2",
    )
    run(command, cwd=VLA_DIR, env=train_env)
    print(f"checkpoints: {args.output_root / run_id}", flush=True)


if __name__ == "__main__":
    main()
