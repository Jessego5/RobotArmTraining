#!/usr/bin/env python3
"""Fit how the real operators held each part, from the wrist cameras.

The replay (tools/replay_real_in_sim.py) needs each part's pose in the gripper
while held. Zeb's controller grips the carrier differently from the real
operators (they held it deeper, face-on to the right wrist camera), so the grips
are fitted to the real frames: on frames where a part is held, the arms and the
held part are rendered through the fitted wrist camera of the holding arm, and the
part's pose in the grip_site frame is searched until the rendered part overlaps
the real part. The real part is the real objects (the depth/colour mask) minus
the rendered arms, which line up to a few pixels; overlap (IoU) penalises a part
of the wrong size or place. The orientation search starts from all 24 right-angle
rotations at depths from the fingertips back into the palm.

    python tools/fit_grasps.py --parts large_carrier gear_3 pin_2
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if sys.platform.startswith('linux'):
    os.environ.setdefault('MUJOCO_GL', 'egl')
import cv2
import mujoco
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

from tools.fit_cameras import WRIST_RENDER, Scene, targets
from tools.replay_real_in_sim import GRASP, RealViews, phases, transform

# part -> (holding arm, wrist camera that sees it, phase)
HELD = {'large_carrier': ('right', 'wrist_port3', 'carrier_held'),
        'gear_3': ('left', 'wrist_port2', 'gear_held'),
        'pin_2': ('left', 'wrist_port2', 'pin_held')}


def right_angle_rotations():
    """The 24 rotations of a cube, as 3x3 matrices."""
    out = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            r = np.zeros((3, 3))
            for i, (j, s) in enumerate(zip(perm, signs)):
                r[i, j] = s
            if np.linalg.det(r) > 0:
                out.append(r)
    return out


class GripRenderer:
    """Masks of both arms plus one held part through a fitted wrist camera."""

    def __init__(self, scene: Scene, views: RealViews, part: str, camera: str, side: str):
        self.scene, self.part, self.camera = scene, part, camera
        m = scene.model
        views.place_cameras()
        body = m.body(part).id
        self.part_geoms = np.array([m.geom_bodyid[g] == body and m.geom_group[g] < 3 for g in range(m.ngeom)])
        self.site = m.site('grip_site' if side == 'left' else 'right_grip_site').id
        self.adr = m.jnt_qposadr[m.body_jntadr[body]]
        self.renderer = mujoco.Renderer(m, WRIST_RENDER, WRIST_RENDER)
        self.renderer.enable_segmentation_rendering()
        self.options = mujoco.MjvOption()
        self.options.geomgroup[:] = [1, 1, 1, 0, 0, 0]
        self.options.sitegroup[:] = 0
        map_x, map_y, inside = views.maps[camera]
        self.map_x, self.map_y, self.inside = map_x, map_y, inside

    def masks(self, states, grasp):
        """Per frame: (robot mask, part mask), with occlusion between them."""
        m, d = self.scene.model, self.scene.data
        out = []
        for state in states:
            self.scene.pose(state)
            grip = transform(d.site_xpos[self.site], d.site_xmat[self.site].reshape(3, 3))
            pose = grip @ grasp
            q = np.zeros(4)
            mujoco.mju_mat2Quat(q, pose[:3, :3].flatten())
            d.qpos[self.adr:self.adr + 3], d.qpos[self.adr + 3:self.adr + 7] = pose[:3, 3], q
            mujoco.mj_kinematics(m, d)
            mujoco.mj_camlight(m, d)
            self.renderer.update_scene(d, self.scene.cam[self.camera], self.options)
            seg = self.renderer.render()
            ids, is_geom = seg[..., 0], seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)
            hit = is_geom & (ids >= 0)
            pair = []
            for lookup in (self.scene.robot, self.part_geoms):
                mask = np.zeros(ids.shape, np.uint8)
                mask[hit] = lookup[ids[hit]]
                pair.append(cv2.remap(mask, self.map_x, self.map_y, cv2.INTER_NEAREST, borderValue=0).astype(bool)
                            & self.inside)
            out.append(tuple(pair))
        return out


def part_targets(renderer, states, real, grow=3):
    """The real part per frame: real objects minus the (grown) rendered arms."""
    kernel = np.ones((2 * grow + 1,) * 2, np.uint8)
    robots = [robot for robot, _ in renderer.masks(states, np.eye(4))]
    return [keep & ~cv2.dilate(robot.astype(np.uint8), kernel).astype(bool) for robot, (keep, _, _) in zip(robots, real)]


def part_score(rendered, targets):
    """1 - mean IoU of the rendered part with the real part (frames where either shows)."""
    ious = [(p & t).sum() / max((p | t).sum(), 1) for (_, p), t in zip(rendered, targets) if (p | t).any()]
    return 1 - float(np.mean(ious)) if ious else 1.


def grasp_from(params, base):
    """params: offset (mm) and rotation vector (deg) applied in the grip frame on top of `base`."""
    t = np.eye(4)
    t[:3, :3] = Rotation.from_rotvec(params[3:6], degrees=True).as_matrix()
    t[:3, 3] = np.asarray(params[:3]) / 1000
    return t @ base


def samples(root, episodes, part, per_episode):
    from tools.augment_real_dataset import decode, episode_rows
    side, camera, phase = HELD[part]
    frames, states = [], []
    for episode in episodes:
        numeric, images = episode_rows(root, episode)
        st = numeric['observation.state'].astype(np.float64)
        segment = next((s for s in phases(st)[side] if s['phase'] == phase), None)
        if segment is None:
            continue
        a, b = segment['start'], segment['stop']
        for i in np.linspace(a + (b - a) * .15, b - 1 - (b - a) * .15, per_episode).astype(int):
            frames.append(decode([images[camera][i]])[0])
            states.append(st[i])
    return np.stack(frames), np.stack(states)


def main():
    from tools.augment_real_dataset import Depth
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--root', type=Path, default=ROOT / 'outputs/real_gear')
    parser.add_argument('--parts', nargs='+', default=list(HELD), choices=list(HELD))
    parser.add_argument('--episodes', type=int, default=10)
    parser.add_argument('--per-episode', type=int, default=2)
    parser.add_argument('--keep', type=int, default=3, help='orientations refined after the coarse pass')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/grasp_fit.json')
    args = parser.parse_args()
    info = json.loads((args.root / 'meta/info.json').read_text())
    order = np.random.default_rng(0).permutation(info['total_episodes'])
    fit_eps, held_eps = sorted(order[:args.episodes].tolist()), sorted(order[args.episodes:args.episodes + 5].tolist())
    scene = Scene()
    views = RealViews(scene)
    depth = Depth()
    results = json.loads(args.output.read_text()) if args.output.exists() else {}
    for part in args.parts:
        side, camera, _ = HELD[part]
        frames, states = samples(args.root, fit_eps, part, args.per_episode)
        held_frames, held_states = samples(args.root, held_eps, part, args.per_episode)
        real, held_real = targets(frames, depth), targets(held_frames, depth)
        renderer = GripRenderer(scene, views, part, camera, side)
        target, held_target = part_targets(renderer, states, real), part_targets(renderer, held_states, held_real)
        loss = lambda grasp, st=states, t=target: part_score(renderer.masks(st, grasp), t)
        zeb = transform(*GRASP[part])
        # Coarse: every right-angle orientation of the part about the grip point, at a few depths
        # along the fingers, plus Zeb's grip.
        candidates = [zeb]
        for r in right_angle_rotations():
            for depth_mm in (-105, -90, -75, -60, -45, -30, -15, 0, 15):
                t = np.eye(4)
                t[:3, :3], t[:3, 3] = r @ zeb[:3, :3], zeb[:3, 3] + np.array([depth_mm / 1000, 0, 0])
                candidates.append(t)
        coarse = sorted(((loss(c), k) for k, c in enumerate(candidates)), key=lambda x: x[0])
        print(f'{part}: Zeb grip loss {loss(zeb):.3f}; best coarse {coarse[0][0]:.3f}', flush=True)
        best = None
        for value, k in coarse[:args.keep]:
            base = candidates[k]
            result = minimize(lambda x: loss(grasp_from(x, base)), np.zeros(6), method='Nelder-Mead',
                              options=dict(maxiter=250, initial_simplex=np.vstack(
                                  [np.zeros(6), np.diag([6, 6, 6, 15, 15, 15])]), xatol=.2, fatol=1e-4))
            print(f'  start {k}: {value:.3f} -> {result.fun:.3f}', flush=True)
            if best is None or result.fun < best[0]:
                best = (result.fun, grasp_from(result.x, base))
        grasp = best[1]
        before = 1 - part_score(renderer.masks(held_states, zeb), held_target)
        after = 1 - part_score(renderer.masks(held_states, grasp), held_target)
        print(f'  held-out part overlap (IoU): Zeb grip {before:.3f}, fitted {after:.3f}', flush=True)
        results[part] = dict(offset=grasp[:3, 3].tolist(), rotation=grasp[:3, :3].tolist(),
                             held_out_zeb=before, held_out_fitted=after, fit_episodes=fit_eps)
        args.output.write_text(json.dumps(results, indent=2) + '\n')
        rows = []
        for frame, state in list(zip(held_frames, held_states))[:4]:
            row = [frame]
            for g, colour in ((zeb, (255, 0, 255)), (grasp, (0, 255, 0))):
                mask = renderer.masks([state], g)[0][1]
                edge = mask ^ cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
                shown = frame.copy()
                shown[edge] = colour
                row.append(shown)
            rows.append(np.concatenate(row, 1))
        path = args.output.with_name(f'grasp_fit_{part}.jpg')
        cv2.imwrite(str(path), cv2.cvtColor(np.concatenate(rows, 0), cv2.COLOR_RGB2BGR))
        print(f'  wrote {path} (real | Zeb grip, magenta | fitted grip, green)', flush=True)


if __name__ == '__main__':
    main()
