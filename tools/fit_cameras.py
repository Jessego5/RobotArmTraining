#!/usr/bin/env python3
"""Fit the three real cameras to the demonstrations by matching arm outlines.

Every real frame records both arms' joint angles, so the CAD arms (sim/bimanual_cad.py)
can be posed exactly as they were. For a candidate camera, the posed arms are
rendered as a mask and compared with the real frame's object mask (Depth Anything
plus the depth/colour rules of tools/augment_real_dataset.py); the camera is
searched until the two agree across many frames from many episodes.

- overhead: position, aim, roll, field of view, principal point and radial
  distortion of the webcam, plus where the right base stands relative to the left
  (the base layout was estimated, not measured);
- wrist_port2 / wrist_port3: each camera's mount on its arm (offset and rotation)
  and its equidistant fisheye lens (focal length, centre).

The real mask also holds the parts and the operator's hands, so the score asks
that rendered arm pixels land on real objects (precision) and that the rendered
outline follows a real outline (chamfer), with overlap (IoU) as a smaller term.
Frames from held-out episodes report the result.

    python tools/fit_cameras.py --root outputs/real_gear --output outputs/camera_fit.json
    python tools/fit_cameras.py --cameras overhead --episodes 16 --per-episode 3
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if sys.platform.startswith('linux'):
    os.environ.setdefault('MUJOCO_GL', 'egl')
import cv2
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from sim.bimanual import OVERHEAD_CROP, OVERHEAD_SETUPS, _look_at, rad_to_opening

SIM_CAMERAS = {'overhead': 'overhead', 'wrist_port2': 'left_wrist', 'wrist_port3': 'right_wrist'}
SIZE = 256                # real frames are 256 x 256
TRUNCATE = 8.             # px: outline distances beyond this count as a plain miss
WRIST_RENDER, WRIST_RENDER_FOV = 720, 150.
WRIST_LIMIT = np.radians(74)  # the real wrist lens shows dark corners beyond about this angle


# ---------------------------------------------------------------- real side --

def samples(root: Path, episodes: list[int], per_episode: int):
    """Frames (per camera) and 14-D states, spread through each episode."""
    from tools.augment_real_dataset import decode, episode_rows
    frames = {c: [] for c in SIM_CAMERAS}
    states = []
    for episode in episodes:
        numeric, images = episode_rows(root, episode)
        n = len(numeric['observation.state'])
        for i in np.linspace(0, n - 1, per_episode + 2)[1:-1].astype(int):
            states.append(np.asarray(numeric['observation.state'][i], np.float64))
            for c in SIM_CAMERAS:
                frames[c].append(decode([images[c][i]])[0])
    return {c: np.stack(f) for c, f in frames.items()}, np.stack(states)


def targets(frames_rgb, depth_model):
    """Real object masks, distance to their outlines, and the usable (lit) region."""
    from tools.augment_real_dataset import keep_mask
    disparity = depth_model(frames_rgb)
    out = []
    for frame, d in zip(frames_rgb, disparity):
        keep = keep_mask(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), d)
        edge = (keep ^ cv2.erode(keep.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)).astype(np.uint8)
        distance = cv2.distanceTransform(1 - edge, cv2.DIST_L2, 5)
        lit = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)[..., 2] > 12
        out.append((keep, np.minimum(distance, TRUNCATE), lit))
    return out


# ---------------------------------------------------------------- sim side ---

class Scene:
    """The CAD two-arm scene posed from real states, rendering robot masks per camera."""

    def __init__(self):
        from sim.bimanual_cad import build_scene
        from sim.panthera_env import ARM_JOINTS
        self.model = mujoco.MjModel.from_xml_path(str(build_scene('original')))
        m = self.model
        self.data = mujoco.MjData(m)
        body = lambda name: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
        roots = {body('link1'), body('right_base')}
        robot_bodies = {b for b in range(m.nbody) if m.body_rootid[b] in roots} - {body('right_base')}
        self.robot = np.array([m.geom_bodyid[g] in robot_bodies and m.geom_group[g] < 3 for g in range(m.ngeom)])
        qadr = lambda name: m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)]
        self.joints = {side: [qadr(prefix + j) for j in ARM_JOINTS]
                       for side, prefix in (('left', ''), ('right', 'right_'))}
        self.fingers = {side: (qadr(prefix + 'L_finger_joint'), qadr(prefix + 'R_finger_joint'))
                        for side, prefix in (('left', ''), ('right', 'right_'))}
        self.right_base = body('right_base')
        self.right_base_pos = m.body_pos[self.right_base].copy()
        self.right_base_quat = m.body_quat[self.right_base].copy()
        self.cam = {c: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, s) for c, s in SIM_CAMERAS.items()}
        self.wrist_pos = {c: m.cam_pos[self.cam[c]].copy() for c in ('wrist_port2', 'wrist_port3')}
        self.wrist_quat = {c: m.cam_quat[self.cam[c]].copy() for c in ('wrist_port2', 'wrist_port3')}
        self.overhead = mujoco.Renderer(m, 480, 640)
        self.wrist = mujoco.Renderer(m, WRIST_RENDER, WRIST_RENDER)
        for r in (self.overhead, self.wrist):
            r.enable_segmentation_rendering()
        self.options = mujoco.MjvOption()
        self.options.geomgroup[:] = [1, 1, 1, 0, 0, 0]
        self.options.sitegroup[:] = 0

    def pose(self, state):
        d = self.data
        mujoco.mj_resetDataKeyframe(self.model, d, 0)
        for side, offset in (('left', 0), ('right', 7)):
            d.qpos[self.joints[side]] = state[offset:offset + 6]
            opening = float(rad_to_opening(state[offset + 6]))
            left_finger, right_finger = self.fingers[side]
            d.qpos[left_finger], d.qpos[right_finger] = opening, -opening
        mujoco.mj_kinematics(self.model, d)
        mujoco.mj_camlight(self.model, d)

    def set_right_base(self, dx, dy, dyaw_deg):
        m = self.model
        m.body_pos[self.right_base] = self.right_base_pos + [dx, dy, 0]
        yaw = Rotation.from_euler('z', dyaw_deg, degrees=True)
        w, x, y, z = self.right_base_quat
        q = (yaw * Rotation.from_quat([x, y, z, w])).as_quat()
        m.body_quat[self.right_base] = [q[3], q[0], q[1], q[2]]

    def segment(self, renderer, camera):
        renderer.update_scene(self.data, self.cam[camera], self.options)
        seg = renderer.render()
        ids, is_geom = seg[..., 0], seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)
        mask = np.zeros(ids.shape, bool)
        hit = is_geom & (ids >= 0)
        mask[hit] = self.robot[ids[hit]]
        return mask


# ---------------------------------------------------------------- cameras ----

OVERHEAD_NAMES = ('px', 'py', 'pz', 'tx', 'ty', 'roll', 'fovy', 'cx', 'cy', 'k1', 'base_dx', 'base_dy', 'base_dyaw')
WRIST_NAMES = ('dx', 'dy', 'dz', 'rx', 'ry', 'rz', 'f', 'cx', 'cy')


def overhead_start():
    s = OVERHEAD_SETUPS['original']
    return np.array([*s['pos'], s['target'][0], s['target'][1], s.get('roll', 0.), s['fovy'], 0, 0, 0, 0, 0, 0])


def wrist_start():
    # Equidistant lens of tools/export_bimanual_dataset.py: 62 deg at the middle of each edge.
    return np.array([0, 0, 0, 0, 0, 0, (SIZE / 2) / np.radians(62.), (SIZE - 1) / 2, (SIZE - 1) / 2])


def overhead_masks(scene, p, states):
    px, py, pz, tx, ty, roll, fovy, cx, cy, k1, bdx, bdy, byaw = p
    m, cam = scene.model, scene.cam['overhead']
    lift = np.array([0, 0, scene_table_z()])
    m.cam_pos[cam] = np.add([px, py, pz], lift)
    m.cam_quat[cam] = _look_at(np.add([px, py, pz], lift), np.add([tx, ty, 0.], lift), roll)
    render_fovy = min(fovy + 12, 100)
    m.cam_fovy[cam] = render_fovy
    scene.set_right_base(bdx, bdy, byaw)
    # Output pixel (distorted, full 640x480 webcam) -> pinhole render pixel.
    x0, y0, x1, y1 = OVERHEAD_CROP
    v, u = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    f = 240 / np.tan(np.radians(fovy / 2))
    xn, yn = (u - 319.5 - cx) / f, (v - 239.5 - cy) / f
    scale = 1 + k1 * (xn ** 2 + yn ** 2)
    fr = 240 / np.tan(np.radians(render_fovy / 2))
    map_x = (319.5 + xn * scale * fr).astype(np.float32)
    map_y = (239.5 + yn * scale * fr).astype(np.float32)
    out = []
    for state in states:
        scene.pose(state)
        mask = scene.segment(scene.overhead, 'overhead').astype(np.uint8)
        out.append(cv2.remap(mask, map_x, map_y, cv2.INTER_NEAREST, borderValue=0).astype(bool))
    return out


def wrist_masks(scene, camera, p, states):
    dx, dy, dz, rx, ry, rz, f, cx, cy = p
    m, cam = scene.model, scene.cam[camera]
    m.cam_pos[cam] = scene.wrist_pos[camera] + [dx, dy, dz]
    w, x, y, z = scene.wrist_quat[camera]
    q = (Rotation.from_quat([x, y, z, w]) * Rotation.from_rotvec([rx, ry, rz], degrees=True)).as_quat()
    m.cam_quat[cam] = [q[3], q[0], q[1], q[2]]
    m.cam_fovy[cam] = WRIST_RENDER_FOV
    v, u = np.mgrid[0:SIZE, 0:SIZE].astype(np.float32)
    du, dv = u - cx, v - cy
    radius = np.hypot(du, dv)
    theta = np.minimum(radius / f, np.radians(89))
    fs = (WRIST_RENDER / 2) / np.tan(np.radians(WRIST_RENDER_FOV / 2))
    scale = np.where(radius > 0, fs * np.tan(theta) / np.maximum(radius, 1e-6), 0)
    map_x = (du * scale + (WRIST_RENDER - 1) / 2).astype(np.float32)
    map_y = (dv * scale + (WRIST_RENDER - 1) / 2).astype(np.float32)
    inside = theta <= WRIST_LIMIT
    out = []
    for state in states:
        scene.pose(state)
        mask = scene.segment(scene.wrist, camera).astype(np.uint8)
        out.append(cv2.remap(mask, map_x, map_y, cv2.INTER_NEAREST, borderValue=0).astype(bool) & inside)
    return out


def scene_table_z():
    from sim.actuator_env import TABLE_Z
    return TABLE_Z


def score(rendered, real):
    """Lower is better. Also returns mean IoU, precision and outline distance (px)."""
    losses, ious, precisions, distances = [], [], [], []
    for mask, (keep, distance, lit) in zip(rendered, real):
        mask = mask & lit
        if mask.sum() < 30:
            losses.append(2.); ious.append(0.); precisions.append(0.); distances.append(TRUNCATE)
            continue
        precision = (mask & keep).sum() / mask.sum()
        iou = (mask & keep).sum() / max((mask | (keep & lit)).sum(), 1)
        outline = mask ^ cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        chamfer = float(distance[outline].mean())
        losses.append((1 - precision) + chamfer / TRUNCATE + .5 * (1 - iou))
        ious.append(iou); precisions.append(precision); distances.append(chamfer)
    return float(np.mean(losses)), dict(iou=float(np.mean(ious)), precision=float(np.mean(precisions)),
                                        outline_px=float(np.mean(distances)))


# ---------------------------------------------------------------- search -----

def search(loss, start, stages, steps, restarts, rng, log):
    """Nelder-Mead over growing parameter subsets (pose first, then lens and layout)."""
    best_p, best = start.copy(), loss(start)
    log(f'  start: loss {best:.4f}')
    for names, free, spread in stages:
        for k in range(restarts):
            p0 = best_p.copy()
            if k:
                p0[free] += rng.normal(0, spread)
            sub = lambda x: loss(np.where(np.isin(np.arange(len(p0)), free), _scatter(p0, free, x), p0))
            from scipy.optimize import minimize
            result = minimize(sub, p0[free], method='Nelder-Mead',
                              options=dict(maxiter=steps, xatol=1e-4, fatol=1e-5, adaptive=True))
            if result.fun < best:
                best, best_p = float(result.fun), _scatter(p0, free, result.x)
            log(f'  {names} restart {k}: loss {result.fun:.4f} (best {best:.4f})')
    return best_p


def _scatter(p, free, x):
    out = p.copy()
    out[free] = x
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--root', type=Path, default=ROOT / 'outputs/real_gear', help='local copy of the real dataset')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/camera_fit.json')
    parser.add_argument('--cameras', nargs='+', default=list(SIM_CAMERAS), choices=list(SIM_CAMERAS))
    parser.add_argument('--episodes', type=int, default=12, help='episodes to fit on')
    parser.add_argument('--held-out', type=int, default=6, help='other episodes to report on')
    parser.add_argument('--per-episode', type=int, default=3)
    parser.add_argument('--steps', type=int, default=300, help='Nelder-Mead iterations per stage')
    parser.add_argument('--restarts', type=int, default=3)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    from tools.augment_real_dataset import Depth
    info = json.loads((args.root / 'meta/info.json').read_text())
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(info['total_episodes'])
    fit_episodes = sorted(order[:args.episodes].tolist())
    held_episodes = sorted(order[args.episodes:args.episodes + args.held_out].tolist())
    started = time.time()
    log = lambda text: print(f'[{time.time() - started:6.0f}s] {text}', flush=True)
    depth = Depth()
    fit_frames, fit_states = samples(args.root, fit_episodes, args.per_episode)
    held_frames, held_states = samples(args.root, held_episodes, args.per_episode)
    log(f'{len(fit_states)} fit frames from episodes {fit_episodes}; {len(held_states)} held-out frames')
    scene = Scene()
    results = json.loads(args.output.read_text()) if args.output.exists() else {}
    for camera in args.cameras:
        fit_real, held_real = targets(fit_frames[camera], depth), targets(held_frames[camera], depth)
        log(f'{camera}: real masks ready')
        if camera == 'overhead':
            render = lambda p, states: overhead_masks(scene, p, states)
            start, names = overhead_start(), OVERHEAD_NAMES
            stages = [('pose', [0, 1, 2, 3, 4, 5, 6], [.08, .04, .08, .15, .04, 2, 3]),
                      ('pose+lens+layout', list(range(13)), [.03, .02, .03, .05, .02, 1, 1, 6, 6, .05, .02, .02, 1])]
        else:
            render = lambda p, states, c=camera: wrist_masks(scene, c, p, states)
            start, names = wrist_start(), WRIST_NAMES
            stages = [('mount', [0, 1, 2, 3, 4, 5], [.01, .01, .01, 5, 5, 5]),
                      ('mount+lens', list(range(9)), [.005, .005, .005, 3, 3, 3, 8, 6, 6])]
        loss = lambda p: score(render(p, fit_states), fit_real)[0]
        before = score(render(start, held_states), held_real)[1]
        log(f'{camera}: held-out before {before}')
        best = search(loss, start, stages, args.steps, args.restarts, rng, log)
        after = score(render(best, held_states), held_real)[1]
        log(f'{camera}: held-out after {after}')
        results[camera] = dict(params=dict(zip(names, map(float, best))), held_out_before=before,
                               held_out_after=after, fit_episodes=fit_episodes, held_out_episodes=held_episodes)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2) + '\n')
        # Overlay for a visual check: green = rendered arm outline on held-out frames.
        rows = []
        for frame, mask in list(zip(held_frames[camera], render(best, held_states)))[:6]:
            shown = frame.copy()
            outline = mask ^ cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
            shown[outline] = (0, 255, 0)
            old = render(start, [held_states[len(rows)]])[0]
            old_outline = old ^ cv2.erode(old.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
            before_shown = frame.copy()
            before_shown[old_outline] = (255, 0, 255)
            rows.append(np.concatenate([before_shown, shown], 1))
        path = args.output.with_name(f'{args.output.stem}_{camera}.jpg')
        cv2.imwrite(str(path), cv2.cvtColor(np.concatenate(rows, 0), cv2.COLOR_RGB2BGR))
        log(f'{camera}: wrote {path} (left: current camera, magenta; right: fitted, green)')


if __name__ == '__main__':
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    main()
