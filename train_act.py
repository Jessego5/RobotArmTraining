#!/usr/bin/env python3
"""Train a local LeRobot ACT policy and report held-out imitation loss."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch

from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors

from tools.lerobot_image_cache import CachedLeRobotDataset, default_cache_path


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET = REPO_ROOT / "outputs" / "lerobot" / "panthera_stack"
DEFAULT_OUTPUT = REPO_ROOT / "outputs" / "act" / "panthera_stack"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--steps", type=int, default=1000)
    # Keep the established 1.2M-sample schedule at 100k optimizer steps.
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--action-steps", type=int, default=10)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--val-batches", type=int, default=40)
    parser.add_argument("--log-freq", type=int, default=25)
    parser.add_argument("--seed", type=int, default=20260922)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument(
        "--no-image-cache",
        action="store_true",
        help="ignore a decoded image cache even if one exists",
    )
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    if not 0.0 < args.val_fraction < 1.0:
        raise SystemExit("--val-fraction must be between 0 and 1")
    if args.action_steps > args.chunk_size:
        raise SystemExit("--action-steps cannot exceed --chunk-size")
    seed_everything(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)

    metadata = LeRobotDatasetMetadata("local/panthera_stack", root=args.dataset)
    all_episodes = np.arange(metadata.total_episodes)
    rng = np.random.default_rng(args.seed)
    rng.shuffle(all_episodes)
    n_val = max(1, round(len(all_episodes) * args.val_fraction))
    val_episodes = sorted(all_episodes[:n_val].tolist())
    train_episodes = sorted(all_episodes[n_val:].tolist())

    policy_features = dataset_to_policy_features(metadata.features)
    output_features = {
        key: value for key, value in policy_features.items()
        if value.type is FeatureType.ACTION
    }
    input_features = {
        key: value for key, value in policy_features.items()
        if key not in output_features
    }
    cfg = ACTConfig(
        input_features=input_features,
        output_features=output_features,
        chunk_size=args.chunk_size,
        n_action_steps=args.action_steps,
        device="cuda" if torch.cuda.is_available() else "cpu",
        use_amp=args.amp and torch.cuda.is_available(),
        push_to_hub=False,
    )
    delta_timestamps = {
        "action": [index / metadata.fps for index in cfg.action_delta_indices],
    }
    image_cache = default_cache_path(args.dataset)
    if args.no_image_cache or not image_cache.is_file():
        image_cache = None
    else:
        print(f"using decoded image cache: {image_cache}", flush=True)
    train_dataset = CachedLeRobotDataset(
        "local/panthera_stack",
        root=args.dataset,
        episodes=train_episodes,
        delta_timestamps=delta_timestamps,
        image_cache=image_cache,
    )
    val_dataset = CachedLeRobotDataset(
        "local/panthera_stack",
        root=args.dataset,
        episodes=val_episodes,
        delta_timestamps=delta_timestamps,
        image_cache=image_cache,
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        persistent_workers=args.workers > 0,
        prefetch_factor=args.prefetch_factor if args.workers > 0 else None,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        persistent_workers=args.workers > 0,
        prefetch_factor=args.prefetch_factor if args.workers > 0 else None,
    )

    policy = ACTPolicy(cfg).to(cfg.device)
    preprocessor, postprocessor = make_pre_post_processors(
        cfg, dataset_stats=metadata.stats
    )
    optimizer = torch.optim.AdamW(
        [
            {"params": [p for n, p in policy.named_parameters() if "backbone" not in n]},
            {
                "params": [p for n, p in policy.named_parameters() if "backbone" in n],
                "lr": cfg.optimizer_lr_backbone,
            },
        ],
        lr=cfg.optimizer_lr,
        weight_decay=cfg.optimizer_weight_decay,
    )

    policy.train()
    iterator = iter(train_loader)
    losses: list[float] = []
    started = time.monotonic()
    for step in range(1, args.steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            batch = next(iterator)
        batch = preprocessor(batch)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=cfg.use_amp,
        ):
            loss, output = policy.forward(batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 10.0)
        optimizer.step()
        losses.append(float(loss.detach()))
        if step == 1 or step % args.log_freq == 0 or step == args.steps:
            window = losses[-args.log_freq:]
            rate = step / max(time.monotonic() - started, 1e-6)
            details = " ".join(
                f"{key}={value:.4f}" for key, value in (output or {}).items()
                if isinstance(value, (int, float))
            )
            print(
                f"step={step:05d} loss={np.mean(window):.4f} "
                f"rate={rate:.2f}step/s {details}",
                flush=True,
            )

    # LeRobot 0.4.4's ACT VAE only constructs its posterior while the module is
    # in training mode.  Keep that mode for held-out loss computation (with
    # gradients disabled); inference evaluation is done separately by rollout.
    policy.train()
    val_losses = []
    with torch.inference_mode():
        for batch_i, batch in enumerate(val_loader):
            if batch_i >= args.val_batches:
                break
            batch = preprocessor(batch)
            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=cfg.use_amp,
            ):
                loss, _ = policy.forward(batch)
            val_losses.append(float(loss))

    policy.save_pretrained(args.output)
    preprocessor.save_pretrained(args.output)
    postprocessor.save_pretrained(args.output)
    elapsed = time.monotonic() - started
    report = {
        "steps": args.steps,
        "batch_size": args.batch_size,
        "chunk_size": args.chunk_size,
        "action_steps": args.action_steps,
        "train_episodes": len(train_episodes),
        "val_episodes": len(val_episodes),
        "held_out_episode_indices": val_episodes,
        "initial_train_loss": float(np.mean(losses[: min(25, len(losses))])),
        "final_train_loss": float(np.mean(losses[-min(25, len(losses)) :])),
        "validation_loss": float(np.mean(val_losses)) if val_losses else math.nan,
        "elapsed_seconds": elapsed,
        "steps_per_second": args.steps / elapsed,
        "seed": args.seed,
    }
    (args.output / "experiment.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
