#!/usr/bin/env python3
"""Sim recreations of the real demonstrations, with small variations in the free motion.

Each real demonstration is recreated in the CAD sim from its recorded joint
angles (tools/replay_real_in_sim.py) and rendered through the cameras fitted to
the real ones (sim/real_cameras.json). Variations add smooth detours to the
grippers' paths while both arms move freely, away from every grasp, release and
insertion: each detour (a few cm and degrees, its own random direction per arm)
rises and falls back to zero inside its window, so the arms are on the recorded
path whenever they touch something. The replays showed contact is only
repeatable to millimetres, so the contacts are left exactly as recorded and only
the free motion varies. The detoured gripper poses go through IK, so the joint
states and actions stay consistent.

Parts follow the phases from the gripper openings: each rests flat on the table
where its gripper closes on it, moves with the gripper while held (the carrier at
the grip measured from 68 real gear releases, the gear and pin at Zeb's top-down
grip), and the gear and pin ride on the carrier after release.

Output: a LeRobot dataset in the real dataset's format (14-D state, 20 Hz, the
three cameras), with plain sim images (restyle them with the real-data tools
afterwards). `meta/provenance.json` records each episode's source demo and
detours.

    python tools/sim_variations.py --episodes 0 1 2 --variations 2 --output outputs/sim_variations
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if sys.platform.startswith('linux'):
    os.environ.setdefault('MUJOCO_GL', 'egl')
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

FPS = 20
SIDES = {'left': (0, 'grip_site'), 'right': (7, 'right_grip_site')}
HOLDS = {'large_carrier': ('right', 'carrier_held'), 'gear_3': ('left', 'gear_held'), 'pin_2': ('left', 'pin_held')}


# ---------------------------------------------------------------- windows ---

def free_windows(segments, count, margin_s=.6, min_s=1.5):
    """Frame ranges where neither arm is within `margin_s` of a grasp or release."""
    busy = np.zeros(count, bool)
    pad = round(margin_s * FPS)
    for side in segments.values():
        for s in side:
            for edge in (s['start'], s['stop']):
                busy[max(0, edge - pad):edge + pad] = True
    busy[:pad] = busy[-pad:] = True  # leave the very start and end as recorded
    from tools.replay_real_in_sim import runs
    return [(int(a), int(b)) for a, b in runs(~busy) if b - a >= min_s * FPS]


def bump(n):
    """Smooth 0 -> 1 -> 0 over n frames (zero slope at both ends)."""
    return np.sin(np.linspace(0, np.pi, n)) ** 2


# ---------------------------------------------------------------- kinematics

class Kinematics:
    """FK and damped least-squares IK of both arms' grip sites on the CAD scene."""

    def __init__(self, scene):
        from sim.panthera_env import ARM_JOINTS
        self.scene, m = scene, scene.model
        self.d = mujoco.MjData(m)
        self.site = {s: m.site(name).id for s, (_, name) in SIDES.items()}
        prefix = {'left': '', 'right': 'right_'}
        self.qadr = {s: np.array([m.jnt_qposadr[m.joint(prefix[s] + j).id] for j in ARM_JOINTS]) for s in SIDES}
        self.dof = {s: np.array([m.jnt_dofadr[m.joint(prefix[s] + j).id] for j in ARM_JOINTS]) for s in SIDES}
        self.range = {s: np.array([m.jnt_range[m.joint(prefix[s] + j).id] for j in ARM_JOINTS]) for s in SIDES}

    def set(self, state):
        d = self.d
        mujoco.mj_resetDataKeyframe(self.scene.model, d, 0)
        for s, (off, _) in SIDES.items():
            d.qpos[self.qadr[s]] = state[off:off + 6]
        mujoco.mj_kinematics(self.scene.model, d)

    def grip(self, side):
        from tools.replay_real_in_sim import transform
        i = self.site[side]
        return transform(self.d.site_xpos[i], self.d.site_xmat[i].reshape(3, 3))

    def ik(self, side, target, q0, iters=60):
        m, d = self.scene.model, self.d
        d.qpos[self.qadr[side]] = q0
        site = self.site[side]
        jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        for _ in range(iters):
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            pos_err = target[:3, 3] - d.site_xpos[site]
            rot_err = Rotation.from_matrix(target[:3, :3] @ d.site_xmat[site].reshape(3, 3).T).as_rotvec()
            if np.linalg.norm(pos_err) < 2e-4 and np.linalg.norm(rot_err) < 2e-3:
                break
            mujoco.mj_jacSite(m, d, jacp, jacr, site)
            J = np.r_[jacp[:, self.dof[side]], .5 * jacr[:, self.dof[side]]]
            dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), np.r_[pos_err, .5 * rot_err])
            r = self.range[side]
            d.qpos[self.qadr[side]] = np.clip(d.qpos[self.qadr[side]] + dq, r[:, 0], r[:, 1])
        mujoco.mj_kinematics(m, d)
        err = np.linalg.norm(target[:3, 3] - d.site_xpos[site])
        return d.qpos[self.qadr[side]].copy(), err


def vary(kin, states, windows, rng, pos_mm=15., rot_deg=4.):
    """States with a smooth detour per arm in each window (IK from the recorded joints)."""
    out = states.copy()
    detours = []
    for a, b in windows:
        shape = bump(b - a)
        for side, (off, _) in SIDES.items():
            shift = np.clip(rng.normal(0, pos_mm / 1000, 3), -2 * pos_mm / 1000, 2 * pos_mm / 1000)
            turn = np.clip(rng.normal(0, np.radians(rot_deg), 3), -np.radians(2 * rot_deg), np.radians(2 * rot_deg))
            worst = 0.
            for k, i in enumerate(range(a, b)):
                kin.set(states[i])
                grip = kin.grip(side)
                target = grip.copy()
                target[:3, 3] += shape[k] * shift
                target[:3, :3] = Rotation.from_rotvec(shape[k] * turn).as_matrix() @ grip[:3, :3]
                q, err = kin.ik(side, target, out[i - 1, off:off + 6] if i > a else states[i, off:off + 6])
                out[i, off:off + 6] = q
                worst = max(worst, err)
            detours.append(dict(side=side, start=int(a), stop=int(b), shift_mm=(shift * 1000).round(1).tolist(),
                                turn_deg=np.degrees(turn).round(1).tolist(), worst_ik_mm=round(worst * 1000, 2)))
    return out, detours


# ---------------------------------------------------------------- parts -----

def part_tracks(kin, states, segments, grips):
    """World pose per frame of the carrier, gear and pin (see the module docstring)."""
    from sim.actuator_env import TABLE_Z
    from tools.replay_real_in_sim import transform
    count = len(states)
    grip = {s: [] for s in SIDES}
    for st in states:
        kin.set(st)
        for s in SIDES:
            grip[s].append(kin.grip(s))
    holds = {}
    for name, (side, phase) in HOLDS.items():
        g = next((s for s in segments[side] if s['phase'] == phase), None)
        if g is None:
            return None
        holds[name] = (side, g['start'], g['stop'] if name != 'large_carrier' else count)
    tracks = {}
    for name in ('large_carrier', 'gear_3', 'pin_2'):
        side, a, b = holds[name]
        held = grip[side][a] @ grips[name]
        yaw = Rotation.from_matrix(held[:3, :3]).as_euler('zyx')[0]
        rest = transform([held[0, 3], held[1, 3], TABLE_Z + .0002], Rotation.from_euler('z', yaw).as_matrix())
        track = [rest if i < a else grip[side][min(i, b - 1)] @ grips[name] for i in range(count)]
        if name != 'large_carrier':  # rides on the carrier after release
            carrier = tracks['large_carrier']
            on = np.linalg.inv(carrier[b - 1]) @ track[b - 1]
            for i in range(b, count):
                track[i] = carrier[i] @ on
        tracks[name] = np.stack(track)
    return tracks


# ---------------------------------------------------------------- job -------

def render_episode(root, dest, episode, seed, variation, pos_mm, rot_deg):
    """One output episode: the recreated (variation 0) or varied demonstration, rendered."""
    global scene, views, kin, grips
    from tools.augment_real_dataset import episode_rows, write_folder
    from tools.replay_real_in_sim import CAMERAS, RealViews, apply_parts, phases, transform
    if 'scene' not in globals():
        from tools.fit_cameras import Scene
        from tools.insertion_env import CARRIER_GRIP
        from tools.physics_replay import grasps
        scene = Scene()
        views = RealViews(scene)
        views.place_cameras()
        m = scene.model
        m.body_pos[m.body('small_carrier').id] = [-1., 0., .05]  # Zeb's fixture is not in the real task
        m.geom_pos[m.geom('fixture').id, :2] = [-1., 0.]
        kin = Kinematics(scene)
        grips = grasps(ROOT / 'none')
        grips['large_carrier'] = CARRIER_GRIP
    shutil.rmtree(dest, ignore_errors=True)
    dest.mkdir(parents=True)
    numeric, _ = episode_rows(root, episode)
    states = numeric['observation.state'].astype(np.float64)
    segments = phases(states)
    windows = free_windows(segments, len(states))
    rng = np.random.default_rng([seed, episode, variation])
    varied, detours = (states, []) if variation == 0 else vary(kin, states, windows, rng, pos_mm, rot_deg)
    tracks = part_tracks(kin, varied, segments, grips)
    images = {c: [] for c in CAMERAS}
    for i, st in enumerate(varied):
        out = views.render(st, tracks, i, None) if tracks else views.render(st)
        for c in CAMERAS:
            images[c].append(out[c])
    new = {k: v.copy() for k, v in numeric.items()}
    new['observation.state'] = varied.astype(np.float32)
    new['action'] = np.r_[varied[1:], varied[-1:]].astype(np.float32)
    new['observation.velocity'] = (np.gradient(varied, axis=0) * FPS).astype(np.float32)
    write_folder(root, dest, new, images, dict(source_episode=episode, variation=variation, windows=windows,
                                              detours=detours, parts=tracks is not None))


def main():
    from tools.augment_real_dataset import run_jobs, write_dataset
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--root', type=Path, default=ROOT / 'outputs/real_gear', help='local copy of the real dataset')
    parser.add_argument('--episodes', type=int, nargs='*', help='source demos (default: all)')
    parser.add_argument('--variations', type=int, default=2, help='varied copies per demo (plus the plain recreation)')
    parser.add_argument('--no-recreation', dest='recreation', action='store_false')
    parser.add_argument('--pos-mm', type=float, default=15.)
    parser.add_argument('--rot-deg', type=float, default=4.)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--work', type=Path, help='per-episode folders (default: OUTPUT.work)')
    parser.add_argument('--workers', type=int, default=6)
    args = parser.parse_args()
    info = json.loads((args.root / 'meta/info.json').read_text())
    episodes = args.episodes if args.episodes else list(range(info['total_episodes']))
    work = args.work or args.output.with_name(args.output.name + '.work')
    items = [(e, v) for e in episodes for v in range(0 if args.recreation else 1, args.variations + 1)]
    dests = [work / f'episode_{e:04d}_v{v}' for e, v in items]
    todo = [k for k, d in enumerate(dests) if not (d / 'done.json').exists()]
    print(f'{len(items)} episodes ({len(items) - len(todo)} ready); rendering {len(todo)}', flush=True)
    run_jobs(render_episode, [(args.root, dests[k], items[k][0], args.seed, items[k][1], args.pos_mm, args.rot_deg)
                              for k in todo], args.workers)
    provenance = dict(generator='tools/sim_variations.py', source='real demonstrations (recorded joint angles)',
                      cameras='sim/real_cameras.json', pos_mm=args.pos_mm, rot_deg=args.rot_deg, seed=args.seed,
                      images='plain CAD sim renders')
    total = write_dataset(args.root, args.output, dests, provenance, keep_jpegs=True)
    print(f'complete: {len(items)} episodes, {total} frames at {args.output}')


if __name__ == '__main__':
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    main()
