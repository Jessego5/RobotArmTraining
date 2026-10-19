#!/usr/bin/env python3
"""Fit the overhead camera pose to real frames by arm-silhouette overlap.

Takes frames spread across the real dataset (FoxNerdSaysMoo/panthera-gear-carrier-
pin-real-20hz), poses both simulated arms from each frame's recorded joint angles
(sim/bimanual.py layout), and searches camera position, aim point, roll and field
of view so the rendered arms overlap the real ones. The real arm mask is
everything on the table that is not wood (hue/saturation) or glare, below the
room band at the top of the crop. The result went into
`OVERHEAD_SETUPS['original']` in sim/bimanual.py.

    python tools/fit_overhead_camera.py --frames 16 --starts 10
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('MUJOCO_GL', 'egl')
import cv2
import mujoco
import numpy as np

from sim.bimanual import OVERHEAD_CROP, OVERHEAD_SETUPS, BimanualState, _look_at, build_model

REAL = 'FoxNerdSaysMoo/panthera-gear-carrier-pin-real-20hz'
ROOM_ROWS = 55  # crop rows above the table edge


def real_frames(count: int):
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, hf_hub_download
    sha = HfApi().dataset_info(REAL).sha
    episodes = pq.read_table(hf_hub_download(REAL, 'meta/episodes/chunk-000/file-000.parquet',
                                             repo_type='dataset', revision=sha)).to_pandas()
    files = sorted(set(zip(episodes['data/chunk_index'], episodes['data/file_index'])))
    frames, states = [], []
    for c, f in files[::max(1, len(files) // max(1, count // 2))][:max(1, count // 2)]:
        table = pq.read_table(hf_hub_download(REAL, f'data/chunk-{c:03d}/file-{f:03d}.parquet', repo_type='dataset',
                                              revision=sha), columns=['observation.images.overhead', 'observation.state'])
        for row in (len(table) // 3, 2 * len(table) // 3):
            frames.append(cv2.imdecode(np.frombuffer(table['observation.images.overhead'][row].as_py()['bytes'],
                                                     np.uint8), 1))
            states.append(np.asarray(table['observation.state'][row].as_py()))
    return frames, states


def robot_mask(image):
    hsv = cv2.cvtColor(cv2.GaussianBlur(image, (5, 5), 0), cv2.COLOR_BGR2HSV).astype(int)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    mask = (~((h >= 8) & (h <= 32) & (s >= 35) & (v >= 90)) & ~(v > 235)).astype(np.uint8)
    mask[:ROOM_ROWS] = 0
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8)).astype(bool)


def main():
    from scipy.optimize import minimize
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--frames', type=int, default=16)
    parser.add_argument('--starts', type=int, default=10)
    parser.add_argument('--fovy', type=float, nargs=2, default=(30., 60.), help='plausible webcam field of view')
    args = parser.parse_args()
    frames, states = real_frames(args.frames)
    real = [robot_mask(f) for f in frames]
    model = build_model(parts=False)
    data = mujoco.MjData(model)
    poser = BimanualState(model)
    cam = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, 'overhead')
    arm = np.array([(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[g]) or '')
                    .startswith(('left/', 'right/')) for g in range(model.ngeom)])
    renderer = mujoco.Renderer(model, 480, 640)
    renderer.enable_segmentation_rendering()
    poses = []
    for st in states:
        poser.set(data, {'left': st[:6], 'right': st[7:13]}, {'left': st[6], 'right': st[13]})
        mujoco.mj_forward(model, data)
        poses.append(data.qpos.copy())
    x0, y0, x1, y1 = OVERHEAD_CROP

    def loss(p):
        px, py, pz, tx, ty, fovy, roll = p
        if not args.fovy[0] <= fovy <= args.fovy[1]:
            return 1. + abs(fovy - np.clip(fovy, *args.fovy))
        model.cam_pos[cam] = [px, py, pz]
        model.cam_quat[cam] = _look_at([px, py, pz], [tx, ty, 0.], roll)
        model.cam_fovy[cam] = fovy
        ious = []
        for qpos, target in zip(poses, real):
            data.qpos[:] = qpos
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, cam)
            seg = renderer.render()
            ids, geom = seg[..., 0], seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)
            mask = np.zeros(ids.shape, bool)
            mask[geom & (ids >= 0)] = arm[ids[geom & (ids >= 0)]]
            mask = mask[y0:y1, x0:x1]
            mask[:ROOM_ROWS] = False
            ious.append((mask & target).sum() / max((mask | target).sum(), 1))
        return 1 - float(np.mean(ious))

    rng = np.random.default_rng(0)
    starts = [[*s['pos'], s['target'][0], s['target'][1], np.clip(s['fovy'], *args.fovy), s.get('roll', 0.)]
              for s in OVERHEAD_SETUPS.values()]
    best = None
    for k in range(args.starts):
        p0 = np.array(starts[k % len(starts)]) + (rng.normal(0, [.15, .05, .15, .2, .05, 5, 3]) if k >= len(starts) else 0)
        result = minimize(loss, p0, method='Nelder-Mead', options=dict(maxiter=250, xatol=1e-3, fatol=1e-4))
        print(f'start {k}: IoU {1 - result.fun:.3f} {np.round(result.x, 3).tolist()}', flush=True)
        if best is None or result.fun < best.fun:
            best = result
    for name, s in zip(OVERHEAD_SETUPS, starts):
        print(f'{name}: IoU {1 - loss(np.array(s)):.3f}')
    px, py, pz, tx, ty, fovy, roll = best.x
    print(f"best IoU {1 - best.fun:.3f}: dict(pos=({px:.3f}, {py:.3f}, {pz:.3f}), target=({tx:.3f}, {ty:.3f}, 0.), "
          f"fovy={fovy:.1f}, roll={roll:.1f})")


if __name__ == '__main__':
    main()
