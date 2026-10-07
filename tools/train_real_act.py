#!/usr/bin/env python3
"""Train ACT on the real two-arm gear/carrier/pin demonstrations and judge it offline.

The real dataset (FoxNerdSaysMoo/panthera-gear-carrier-pin-real-20hz) is LeRobot
v3.0: 227 hand-guided episodes at 20 Hz, 14-D state and action (left joints 1-6,
left gripper rad, right joints 1-6, right gripper rad) and three 256x256 cameras.
There is no robot to roll out on, so the policy is judged on held-out episodes:

- open-loop action error in physical units (degrees per joint, gripper radians)
  for the first predicted action and for the whole 1 s chunk, next to the
  trivial "hold the current pose" baseline it has to beat;
- a video per held-out episode: the real overhead frame, then the two-arm sim
  posed where the recording went one chunk later and where the policy said it
  would go (`sim/bimanual.py`, placement estimated).

    python tools/train_real_act.py fetch --root data/real_gear
    python tools/train_real_act.py train --root data/real_gear --output outputs/real_act
    python tools/train_real_act.py evaluate --root data/real_gear --checkpoint outputs/real_act/best
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
REPO_ID = 'FoxNerdSaysMoo/panthera-gear-carrier-pin-real-20hz'
STATE = 'observation.state'
CAMERAS = ('observation.images.overhead', 'observation.images.wrist_port2', 'observation.images.wrist_port3')
JOINT_NAMES = [f'{side} {name}' for side in ('left', 'right')
               for name in (*(f'j{i}' for i in range(1, 7)), 'gripper')]
GRIPPERS = [6, 13]


def fetch(args):
    from huggingface_hub import snapshot_download
    # Only the compact training export; the full-rate archives add ~9 GiB.
    snapshot_download(REPO_ID, repo_type='dataset', local_dir=args.root, allow_patterns=['meta/**', 'data/**'])
    info = json.loads((args.root / 'meta/info.json').read_text())
    print(f"{info['total_episodes']} episodes, {info['total_frames']} frames at {args.root}")


def split(total: int, fraction: float, seed: int) -> tuple[list[int], list[int]]:
    episodes = np.random.default_rng(seed).permutation(total)
    n_val = max(1, round(total * fraction))
    return sorted(episodes[n_val:].tolist()), sorted(episodes[:n_val].tolist())


def dataset(root: Path, episodes: list[int], fps: int, chunk: int):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    return LeRobotDataset(REPO_ID, root=root, episodes=episodes,
                          delta_timestamps={'action': [i / fps for i in range(chunk)]})


def train(args):
    import torch
    from lerobot.configs.types import FeatureType
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    from lerobot.datasets.utils import dataset_to_policy_features
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors

    random.seed(args.seed), np.random.seed(args.seed), torch.manual_seed(args.seed)
    metadata = LeRobotDatasetMetadata(REPO_ID, root=args.root)
    total = min(metadata.total_episodes, args.limit_episodes or metadata.total_episodes)
    train_episodes, val_episodes = split(total, args.val_fraction, args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'split.json').write_text(json.dumps({'train': train_episodes, 'validation': val_episodes}) + '\n')
    features = dataset_to_policy_features(metadata.features)
    # Joint state and the three cameras; velocity and effort are left out.
    inputs = {k: v for k, v in features.items() if k in (STATE, *CAMERAS)}
    outputs = {k: v for k, v in features.items() if v.type is FeatureType.ACTION}
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg = ACTConfig(input_features=inputs, output_features=outputs, chunk_size=args.chunk,
                    n_action_steps=args.action_steps, device=device, use_amp=device == 'cuda', push_to_hub=False)
    policy = ACTPolicy(cfg).to(device)
    # Stats in the export cover all 227 episodes, so the held-out ones leak only
    # their normalization ranges, not their actions.
    preprocessor, postprocessor = make_pre_post_processors(cfg, dataset_stats=metadata.stats)
    optimizer = torch.optim.AdamW(
        [{'params': [p for n, p in policy.named_parameters() if 'backbone' not in n]},
         {'params': [p for n, p in policy.named_parameters() if 'backbone' in n], 'lr': cfg.optimizer_lr_backbone}],
        lr=cfg.optimizer_lr, weight_decay=cfg.optimizer_weight_decay)
    scaler = torch.amp.GradScaler('cuda', enabled=cfg.use_amp)
    start, best = 0, math.inf
    last = args.output / 'last'
    if (last / 'training_state.pt').is_file():
        policy = ACTPolicy.from_pretrained(last, config=cfg).to(device)
        state = torch.load(last / 'training_state.pt', map_location='cpu', weights_only=False)
        optimizer = torch.optim.AdamW(
            [{'params': [p for n, p in policy.named_parameters() if 'backbone' not in n]},
             {'params': [p for n, p in policy.named_parameters() if 'backbone' in n],
              'lr': cfg.optimizer_lr_backbone}], lr=cfg.optimizer_lr, weight_decay=cfg.optimizer_weight_decay)
        optimizer.load_state_dict(state['optimizer'])
        start, best = state['step'], state['best']
        print(f'resumed at step {start}', flush=True)

    loader_args = dict(batch_size=args.batch_size, num_workers=args.workers, pin_memory=device == 'cuda',
                       persistent_workers=args.workers > 0)
    train_loader = torch.utils.data.DataLoader(dataset(args.root, train_episodes, metadata.fps, args.chunk),
                                               shuffle=True, drop_last=True, **loader_args)
    val_data = dataset(args.root, val_episodes, metadata.fps, args.chunk)
    val_data = torch.utils.data.Subset(val_data, np.linspace(0, len(val_data) - 1,
                                                             min(args.val_frames, len(val_data)), dtype=int).tolist())
    val_loader = torch.utils.data.DataLoader(val_data, shuffle=False, **loader_args)
    print(f'train {len(train_episodes)} episodes / {len(train_loader.dataset)} frames, '
          f'validation {len(val_episodes)} episodes; device {device}', flush=True)

    def save(path: Path, step: int):
        path.mkdir(parents=True, exist_ok=True)
        policy.save_pretrained(path)
        preprocessor.save_pretrained(path)
        postprocessor.save_pretrained(path)
        (path / 'deployment.json').write_text(json.dumps(
            {'dataset': REPO_ID, 'fps': metadata.fps, 'chunk': args.chunk, 'action_steps': args.action_steps,
             'step': step, 'split': str((args.output / 'split.json').resolve())}, indent=2) + '\n')
        torch.save({'step': step, 'best': best, 'optimizer': optimizer.state_dict()}, path / '.state.tmp')
        (path / '.state.tmp').replace(path / 'training_state.pt')

    @torch.inference_mode()
    def validate() -> float:
        policy.eval()
        total = count = 0.
        for batch in val_loader:
            batch = preprocessor(batch)
            prediction = policy.predict_action_chunk({k: batch[k] for k in cfg.input_features})
            valid = (~batch['action_is_pad']).unsqueeze(-1).expand_as(prediction)
            total += float((prediction - batch['action']).abs().masked_select(valid).sum())
            count += int(valid.sum())
        policy.train()
        return total / max(count, 1)

    policy.train()
    iterator, losses, started = iter(train_loader), [], time.monotonic()
    for step in range(start + 1, args.steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            batch = next(iterator)
        batch = preprocessor(batch)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=cfg.use_amp):
            loss, _ = policy.forward(batch)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 10.)
        scaler.step(optimizer)
        scaler.update()
        losses.append(float(loss.detach()))
        if step % 100 == 0:
            rate = (step - start) / (time.monotonic() - started)
            print(f'step {step}/{args.steps} loss {np.mean(losses[-100:]):.4f} '
                  f'{rate:.1f} it/s eta {(args.steps - step) / rate / 3600:.1f} h', flush=True)
        if step % args.eval_freq == 0 or step == args.steps:
            val = validate()
            improved = val < best
            best = min(best, val)
            with (args.output / 'validation.jsonl').open('a') as log:
                log.write(json.dumps({'step': step, 'validation_l1': val, 'train_loss': float(np.mean(losses[-100:]))}) + '\n')
            print(f'validation step {step}: normalized L1 {val:.4f}{" (best)" if improved else ""}', flush=True)
            if improved:
                save(args.output / 'best', step)
            save(last, step)
    print(f'done; best validation normalized L1 {best:.4f}; checkpoint {args.output / "best"}', flush=True)


def evaluate(args):
    import torch
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors

    deployment = json.loads((args.checkpoint / 'deployment.json').read_text())
    val_episodes = json.loads(Path(deployment['split']).read_text())['validation']
    if args.episodes:
        val_episodes = val_episodes[:args.episodes]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    policy = ACTPolicy.from_pretrained(args.checkpoint).to(device).eval()
    preprocessor, postprocessor = make_pre_post_processors(policy.config, pretrained_path=str(args.checkpoint))
    chunk = deployment['chunk']
    output = args.output or args.checkpoint / 'offline_eval'
    output.mkdir(parents=True, exist_ok=True)
    errors = {'policy_first': [], 'hold_first': [], 'policy_chunk_end': [], 'hold_chunk_end': []}
    renderer = None
    for episode in val_episodes:
        data = dataset(args.root, [episode], deployment['fps'], chunk)
        loader = torch.utils.data.DataLoader(data, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
        states, targets, predictions, pads, overhead = [], [], [], [], []
        with torch.inference_mode():
            for batch in loader:
                raw_state, raw_action, pad = batch[STATE], batch['action'], batch['action_is_pad']
                images = batch[CAMERAS[0]]
                normalized = preprocessor(batch)
                chunk_prediction = policy.predict_action_chunk({k: normalized[k] for k in policy.config.input_features})
                physical = postprocessor(chunk_prediction.flatten(0, 1)).reshape(chunk_prediction.shape).cpu()
                states.append(raw_state), targets.append(raw_action), predictions.append(physical), pads.append(pad)
                overhead.append((images.permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8))
        state, target, prediction, pad = (torch.cat(x).numpy() for x in (states, targets, predictions, pads))
        last = chunk - 1
        end_valid = ~pad[:, last]
        errors['policy_first'].append(np.abs(prediction[:, 0] - target[:, 0]))
        errors['hold_first'].append(np.abs(state - target[:, 0]))
        errors['policy_chunk_end'].append(np.abs(prediction[end_valid, last] - target[end_valid, last]))
        errors['hold_chunk_end'].append(np.abs(state[end_valid] - target[end_valid, last]))
        np.savez_compressed(output / f'episode_{episode:03d}.npz', state=state, target=target, prediction=prediction, pad=pad)
        if args.videos:
            renderer = renderer or _SimPair()
            renderer.video(output / f'episode_{episode:03d}.mp4', np.concatenate(overhead),
                           target[:, last], prediction[:, last], deployment['fps'])
        print(f'episode {episode}: {len(state)} frames', flush=True)

    def summarize(key):
        values = np.concatenate(errors[key])
        mean = values.mean(0)
        joints = [i for i in range(14) if i not in GRIPPERS]
        return {'joint_mae_deg': float(np.degrees(mean[joints]).mean()),
                'gripper_mae_rad': float(mean[GRIPPERS].mean()),
                'per_dimension': {name: (float(np.degrees(v)) if i not in GRIPPERS else float(v))
                                  for i, (name, v) in enumerate(zip(JOINT_NAMES, mean))}}
    report = {'checkpoint': str(args.checkpoint), 'episodes': val_episodes,
              'chunk_seconds': chunk / deployment['fps'], **{k: summarize(k) for k in errors}}
    (output / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    print(f"{'':>18} {'joints (deg)':>13} {'gripper (rad)':>14}")
    for key in errors:
        print(f"{key:>18} {report[key]['joint_mae_deg']:13.2f} {report[key]['gripper_mae_rad']:14.3f}")
    print(f'wrote {output}')


class _SimPair:
    """Real overhead frame | sim at the recorded pose | sim at the predicted pose."""

    def __init__(self):
        os.environ.setdefault('MUJOCO_GL', 'egl')
        import mujoco
        from sim.bimanual import ARMS, BimanualState, build_model
        self.mujoco, self.arms = mujoco, ARMS
        self.model = build_model()
        self.data = mujoco.MjData(self.model)
        self.poser = BimanualState(self.model)
        self.renderer = mujoco.Renderer(self.model, 256, 256)
        self.options = mujoco.MjvOption()
        self.options.sitegroup[:] = 0
        self.camera = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, 'overhead')

    def render(self, action):
        self.poser.set(self.data, {'left': action[:6], 'right': action[7:13]},
                       {'left': action[6], 'right': action[13]})
        self.mujoco.mj_forward(self.model, self.data)
        self.renderer.update_scene(self.data, self.camera, self.options)
        return self.renderer.render()

    def video(self, path, real, recorded, predicted, fps):
        import cv2
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (768, 256))
        for image, a, b in zip(real, recorded, predicted):
            panels = [image, self.render(a), self.render(b)]
            frame = cv2.cvtColor(np.concatenate(panels, 1), cv2.COLOR_RGB2BGR)
            for x, text in zip((4, 260, 516), ('real overhead', 'sim: recorded +1 chunk', 'sim: policy +1 chunk')):
                cv2.putText(frame, text, (x, 14), cv2.FONT_HERSHEY_SIMPLEX, .4, (0, 255, 255), 1, cv2.LINE_AA)
            writer.write(frame)
        writer.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('fetch')
    p.add_argument('--root', type=Path, default=ROOT / 'data/real_gear')
    p = commands.add_parser('train')
    p.add_argument('--root', type=Path, default=ROOT / 'data/real_gear')
    p.add_argument('--output', type=Path, default=ROOT / 'outputs/real_act')
    p.add_argument('--steps', type=int, default=60000)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--chunk', type=int, default=20, help='predicted actions per chunk (20 = 1 s)')
    p.add_argument('--action-steps', type=int, default=10)
    p.add_argument('--val-fraction', type=float, default=.1)
    p.add_argument('--val-frames', type=int, default=2000)
    p.add_argument('--eval-freq', type=int, default=5000)
    p.add_argument('--seed', type=int, default=20261006)
    p.add_argument('--limit-episodes', type=int, help='use only the first N episodes (smoke tests)')
    p = commands.add_parser('evaluate')
    p.add_argument('--root', type=Path, default=ROOT / 'data/real_gear')
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path)
    p.add_argument('--episodes', type=int, help='first N held-out episodes only')
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--videos', action='store_true', help='also write sim comparison videos')
    args = parser.parse_args()
    {'fetch': fetch, 'train': train, 'evaluate': evaluate}[args.command](args)


if __name__ == '__main__':
    main()
