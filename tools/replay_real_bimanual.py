#!/usr/bin/env python3
"""Replay a real two-arm recording in the simulator next to its camera frames.

Reads one full-rate episode from the real gear/carrier/pin dataset
(`trimmed_full_rate/<id>.tar.gz`, extracted: states.jsonl, frames.jsonl,
images/<camera>/*.jpg). For each overhead camera frame it takes the nearest
recorded state, poses both simulated arms with those joint angles and gripper
angles, and renders the simulated overhead and wrist cameras. Real views are
drawn on the top row and simulated views below, so the placement, camera and
gripper estimates in `sim/bimanual.py` can be judged directly.

The wrist USB ports are not yet mapped to arms, so both real wrist views are
shown beside both simulated wrist views.

    python tools/replay_real_bimanual.py EPISODE_DIR --video replay.mp4 --sheet replay.jpg
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
REAL_CAMERAS = ('overhead', 'wrist_port2', 'wrist_port3')
SIM_CAMERAS = ('overhead', 'left/wrist', 'right/wrist')


def load_episode(path: Path):
    states = [json.loads(line) for line in (path / 'states.jsonl').open()]
    frames = [json.loads(line) for line in (path / 'frames.jsonl').open()]
    state_times = np.array([s['monotonic_ns'] for s in states])
    by_camera = {camera: sorted((f for f in frames if f['camera'] == camera), key=lambda f: f['monotonic_ns'])
                 for camera in REAL_CAMERAS}
    return states, state_times, by_camera


def nearest(times: np.ndarray, t: int) -> int:
    i = int(np.searchsorted(times, t))
    if i == len(times) or (i > 0 and t - times[i - 1] < times[i] - t):
        i -= 1
    return i


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('episode', type=Path, help='extracted full-rate episode directory')
    parser.add_argument('--video', type=Path)
    parser.add_argument('--sheet', type=Path, help='contact sheet of evenly spaced frames')
    parser.add_argument('--sheet-frames', type=int, default=6)
    parser.add_argument('--tile', type=int, nargs=2, default=(320, 240), metavar=('W', 'H'))
    args = parser.parse_args()
    os.environ.setdefault('MUJOCO_GL', 'egl')
    import cv2
    import mujoco
    from sim.bimanual import ARMS, BimanualState, build_model

    model = build_model()
    data = mujoco.MjData(model)
    poser = BimanualState(model)
    renderer = mujoco.Renderer(model, height=480, width=640)
    options = mujoco.MjvOption()
    options.sitegroup[:] = 0  # hide TCP and contact-point markers
    camera_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name) for name in SIM_CAMERAS]
    states, state_times, by_camera = load_episode(args.episode)
    camera_times = {c: np.array([f['monotonic_ns'] for f in by_camera[c]]) for c in REAL_CAMERAS}
    width, height = args.tile

    def label(image, text):
        cv2.rectangle(image, (0, 0), (image.shape[1], 18), (0, 0, 0), -1)
        cv2.putText(image, text, (4, 13), cv2.FONT_HERSHEY_SIMPLEX, .4, (255, 255, 255), 1, cv2.LINE_AA)
        return image

    composites = []
    for frame in by_camera['overhead']:
        t = frame['monotonic_ns']
        state = states[nearest(state_times, t)]
        poser.set(data, {s: state['arms'][s]['position_rad'] for s in ARMS},
                  {s: state['arms'][s]['gripper']['position'] for s in ARMS})
        mujoco.mj_forward(model, data)
        real = []
        for camera in REAL_CAMERAS:
            match = by_camera[camera][nearest(camera_times[camera], t)]
            image = cv2.imread(str(args.episode / match['path']))
            real.append(label(cv2.resize(image, (width, height)), f'real {camera}'))
        sim = []
        for name, camera_id in zip(SIM_CAMERAS, camera_ids):
            renderer.update_scene(data, camera_id, options)
            image = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
            sim.append(label(cv2.resize(image, (width, height)), f'sim {name}'))
        grid = np.concatenate([np.concatenate(real, 1), np.concatenate(sim, 1)], 0)
        cv2.putText(grid, f"t={frame['episode_time_s']:.2f}s", (8, 2 * height - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 255, 255), 1, cv2.LINE_AA)
        composites.append(grid)
    renderer.close()
    if not composites:
        raise SystemExit('no overhead frames in this episode')
    if args.video:
        args.video.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(args.video), cv2.VideoWriter_fourcc(*'mp4v'), 20,
                                 composites[0].shape[1::-1])
        for grid in composites:
            writer.write(grid)
        writer.release()
    if args.sheet:
        picks = np.linspace(0, len(composites) - 1, min(args.sheet_frames, len(composites))).astype(int)
        args.sheet.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.sheet), np.concatenate([composites[i] for i in picks], 0))
    print(f'{len(composites)} frames replayed from {args.episode.name}')


if __name__ == '__main__':
    main()
