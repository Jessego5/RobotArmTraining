#!/usr/bin/env python3
"""Collect scripted two-arm gear/carrier/pin demonstrations in the bimanual sim.

Mirrors the real recordings (FoxNerdSaysMoo/panthera-gear-carrier-pin-real-20hz):
the right arm takes the carrier and holds it up at the assembly point, the left
arm sets the gear on it, then takes the pin from its stand and pushes it through
the gear into the carrier. Like the real cuts, an episode ends with both
grippers still supporting the assembly. Both arms move at once where the real
operator would (each fetches its first part in parallel).

Every episode is physics-validated (gear seated over the hole, pin inside the
carrier, both parts still held) and saved at the real 20 Hz with the full sim
state, the 14-D state in real units (radians; gripper 0 closed .. 2 open),
velocities and actuator efforts. Parts are stand-ins (see sim/bimanual.py).

    python tools/collect_bimanual.py --output data/bimanual --episodes 200 --workers 8
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import mujoco
import numpy as np

from sim.bimanual import (ARMS, CARRIER_HALF, FINGER_OPEN, GEAR_HALF, OVERHEAD_SETUPS, PIN_HALF, PIN_RADIUS,
                          BimanualSim, build_model)

FPS = 20
TASK = 'put the gear in the carrier and put a pin in it'
# Taken from the real recordings: median joint postures (elbow low, so the arms
# stay out of the overhead view) and, by forward kinematics, the hands' approach --
# each from its own side about 65 degrees off straight ahead, both meeting near
# (0.25, 0, 0.03) at the table. The sim grippers are bulkier than the real fin-ray
# ones, so the right hand holds the carrier by its near edge with a shallower tilt
# (30 vs 44 degrees) and the left comes down steeper (75), which keeps the hands
# apart over the hole. A hand keeps one tilt while holding a part: re-tilting
# would tip the part with it.
REAL_POSTURE = {'left': np.array([-.71, 1.60, .72, .08, .28, .37]),
                'right': np.array([.65, 1.58, .55, .19, -.37, -.35])}
RIGHT_YAW, RIGHT_PITCH, RIGHT_EDGE = math.radians(65), 30., .012
LEFT_YAW, LEFT_PITCH = math.radians(-65), 75.


class DemoFailure(RuntimeError):
    pass


def grasp_quat(yaw: float, pitch_deg: float) -> np.ndarray:
    """Approach yaw then downward pitch; the jaws close along the rotated y-axis."""
    c, s = math.cos(yaw), math.sin(yaw)
    rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.]])
    p = math.radians(pitch_deg)
    c, s = math.cos(p), math.sin(p)
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, (rz @ np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])).ravel())
    return quat


def face_yaw(part_quat: np.ndarray, nominal: float) -> float:
    """Approach yaw nearest `nominal` that keeps the jaws square to a part face."""
    w, x, y, z = part_quat
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return nominal + ((yaw - nominal + math.pi / 4) % (math.pi / 2) - math.pi / 4)


class Planner:
    def __init__(self, seed: int):
        self.seed = seed
        self.rng = rng = np.random.default_rng(seed)
        self.sim = sim = BimanualSim(build_model())
        stand = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, 'pin_stand')
        stand_xy = rng.uniform([.20, .14], [.27, .20])
        sim.model.body_pos[stand, :2] = stand_xy
        sim.place_part('pin', stand_xy)
        sim.place_part('gear', rng.uniform([.26, .07], [.34, .13]), rng.uniform(-np.pi, np.pi))
        # The carrier is assembled resting on the table, still held by the right hand.
        self.assembly = np.r_[rng.uniform([.23, -.04], [.29, .01]), CARRIER_HALF[1]]
        carrier_yaw = RIGHT_YAW + rng.uniform(-.15, .15)
        sim.place_part('carrier', self.assembly[:2], carrier_yaw)
        self.speed = float(rng.uniform(.14, .22))  # real episodes average ~12 s
        self.clearance = float(rng.uniform(.05, .08))
        self.overhead_setup = str(rng.choice(sorted(OVERHEAD_SETUPS)))
        self.rows, self.stages, self.events = [], [], set()
        self.ticks = round(1 / FPS / sim.model.opt.timestep)
        # Start near the real postures, so IK stays on the real (elbow-low) branch.
        self.target, self.quat, self.grip, self.qctrl = {}, {}, {}, {}
        for side in ARMS:
            q = REAL_POSTURE[side] + rng.normal(0, .06, 6)
            sim.data.qpos[sim.arms[side].arm_qadr] = q
            grip = float(rng.uniform(.4, 1.)) * FINGER_OPEN
            sim.data.qpos[sim.finger_qadr[side]] = grip, -grip
            sim.set_command(side, q, grip, immediate=True)
            self.grip[side], self.qctrl[side] = grip, q
        # The right hand starts around the carrier: an edge grip that leaves the top
        # free, held about as high as the real right hand (~2 cm).
        quat = grasp_quat(carrier_yaw, RIGHT_PITCH)
        approach = np.array([math.cos(carrier_yaw), math.sin(carrier_yaw)])
        tip = self.assembly + np.r_[-approach * RIGHT_EDGE, .006]
        q, pe, re = sim.arms['right'].ik(tip, quat, q_init=REAL_POSTURE['right'], iters=300, max_joint_step=None)
        if pe > 1e-3 or re > 1e-2:
            raise DemoFailure('right: carrier grasp IK')
        opening = CARRIER_HALF[0] + .004
        sim.data.qpos[sim.arms['right'].arm_qadr] = q
        sim.data.qpos[sim.finger_qadr['right']] = opening, -opening
        sim.set_command('right', q, opening, immediate=True)
        self.grip['right'], self.qctrl['right'] = opening, q
        mujoco.mj_forward(sim.model, sim.data)
        for side in ARMS:
            self.target[side], self.quat[side] = sim.ee_pose(side)
        self.record_from = None

    # ---- per-arm motion scripts are generators yielding once per tick ----
    def move(self, side, name, target=None, quat=None, grip=None, duration=None, settle=6):
        self.stages.append(dict(arm=side, name=name, start=len(self.rows)))
        start, qstart, gstart = self.target[side].copy(), self.quat[side].copy(), self.grip[side]
        target = start if target is None else np.asarray(target, float)
        quat = qstart if quat is None else np.asarray(quat, float)
        grip = gstart if grip is None else grip
        if np.dot(quat, qstart) < 0:
            quat = -quat
        angle = 2 * math.acos(min(1., abs(float(np.dot(quat, qstart)))))
        duration = duration or max(.4, 1.5 * np.linalg.norm(target - start) / self.speed, angle / .7)
        steps = math.ceil(duration * FPS)
        for i in range(1, steps + 1):
            u = i / steps
            u = u * u * (3 - 2 * u)
            q = qstart * (1 - u) + quat * u
            self.target[side], self.quat[side] = start * (1 - u) + target * u, q / np.linalg.norm(q)
            self.grip[side] = gstart * (1 - u) + grip * u
            yield
        for _ in range(settle):  # let the servos finish before a contact change
            yield
        error = np.linalg.norm(self.sim.ee_pose(side)[0] - target)
        if error > .012:
            raise DemoFailure(f'{side} {name}: tracking error {error:.4f}')

    def wait(self, event):
        while event not in self.events:
            yield

    def pick(self, side, part, quat, tip, lift):
        above = tip + [0, 0, self.clearance]
        yield from self.move(side, f'{part}_approach', above, quat, FINGER_OPEN)
        yield from self.move(side, f'{part}_descend', tip)
        yield from self.move(side, f'{part}_close', grip=0., duration=.5)
        if part not in self.sim.pinched(side):
            raise DemoFailure(f'{side}: no two-pad grasp on {part}')
        before = self.sim.part_pose(part)[0][2]
        yield from self.move(side, f'{part}_lift', tip + [0, 0, lift])
        if self.sim.part_pose(part)[0][2] < before + lift / 2:
            raise DemoFailure(f'{side}: {part} slipped during the lift')

    def right_script(self):
        """Close on the carrier, which starts between the open jaws, then hold it.

        Real episodes begin with the carrier already in the right hand, so this
        grasp happens before recording starts.
        """
        sim = self.sim
        yield from self.move('right', 'carrier_close', grip=0., duration=.6)
        if 'carrier' not in sim.pinched('right'):
            raise DemoFailure('right: no two-pad grasp on carrier')
        self.events.add('carrier_ready')
        self.record_from = len(self.rows)
        yield from self.wait('done')

    def left_script(self):
        sim = self.sim
        yield from self.wait('carrier_ready')
        position, orientation = sim.part_pose('gear')
        quat = grasp_quat(face_yaw(orientation, LEFT_YAW), LEFT_PITCH)
        yield from self.pick('left', 'gear', quat, position + [0, 0, .001], .06)
        # Gear centre goes on the carrier hole; carry the measured held offset.
        offset = sim.ee_pose('left')[0] - sim.part_pose('gear')[0]
        seat = sim.site('carrier_hole') + [0, 0, GEAR_HALF[1] + .001] + offset
        yield from self.move('left', 'gear_transfer', seat + [0, 0, .04])
        yield from self.move('left', 'gear_place', seat)
        yield from self.move('left', 'gear_release', grip=FINGER_OPEN, duration=.5)
        yield from self.move('left', 'gear_retreat', seat + [0, 0, .06])
        gear, hole = sim.site('gear_bore'), sim.site('carrier_hole')
        if np.linalg.norm(gear[:2] - hole[:2]) > .002:
            raise DemoFailure(f'gear off the hole by {np.linalg.norm(gear[:2] - hole[:2]) * 1000:.1f} mm')

        # Pin: out of its stand, over the bore, then down into the carrier.
        pin_quat = grasp_quat(LEFT_YAW, LEFT_PITCH)
        pin = sim.part_pose('pin')[0]
        yield from self.pick('left', 'pin', pin_quat, pin + [0, 0, PIN_HALF - .004], .07)
        offset = sim.ee_pose('left')[0] - sim.site('pin_tip')
        hole = sim.site('carrier_hole')
        yield from self.move('left', 'pin_transfer', hole + [0, 0, .035] + offset)
        yield from self.move('left', 'pin_align', hole + [0, 0, .012] + offset, duration=.6)
        depth = 2 * CARRIER_HALF[1] - .006  # down to just above the carrier floor
        yield from self.move('left', 'pin_insert', hole - [0, 0, depth - .002] + offset, duration=1.2)
        yield from self.move('left', 'hold', duration=.5)
        self.events.add('done')

    def tick(self):
        sim = self.sim
        for side in ARMS:
            q, _, _ = sim.arms[side].ik(self.target[side], self.quat[side], q_init=self.qctrl[side], iters=35,
                                         max_joint_step=.08, posture_gain=0.)
            self.qctrl[side] = q
            sim.set_command(side, q, self.grip[side])
        sim.step(self.ticks)
        if not np.isfinite(sim.data.qpos).all():
            raise DemoFailure('nonfinite simulator state')
        self.rows.append(dict(qpos=sim.data.qpos.copy(), state=sim.state_rad(), velocity=sim.velocity_rad(),
                              effort=sim.effort(), command=sim.data.ctrl.copy()))

    def run(self):
        scripts = {'left': self.left_script(), 'right': self.right_script()}
        for _ in range(int(FPS * .3)):
            self.tick()
        while scripts:
            for side in list(scripts):
                try:
                    next(scripts[side])
                except StopIteration:
                    del scripts[side]
            if len(self.rows) > FPS * 90:
                raise DemoFailure('timeout')
            self.tick()
        return self.validate()

    def validate(self):
        sim = self.sim
        hole, gear, tip = sim.site('carrier_hole'), sim.site('gear_bore'), sim.site('pin_tip')
        metrics = dict(gear_offset_mm=float(np.linalg.norm(gear[:2] - hole[:2]) * 1000),
                       pin_offset_mm=float(np.linalg.norm(tip[:2] - hole[:2]) * 1000),
                       pin_depth_mm=float((hole[2] - tip[2]) * 1000),
                       right_holds=sorted(sim.pinched('right')), left_holds=sorted(sim.pinched('left')),
                       right_opening_mm=sim.opening('right') * 1000,
                       carrier_moved_mm=float(np.linalg.norm(sim.part_pose('carrier')[0][:2] - self.assembly[:2]) * 1000))
        if metrics['pin_depth_mm'] < 8 or metrics['pin_offset_mm'] > 3:
            raise DemoFailure(f"pin not inserted: {metrics}")
        # The pin may slide down through the left fingers once seated; it is still in the hole.
        # The edge grip's pad contacts flicker, so "held" means the jaws are still closed
        # to the carrier's width and the carrier has stayed put.
        clamped = abs(metrics['right_opening_mm'] - CARRIER_HALF[0] * 1000) < 2.5
        if metrics['gear_offset_mm'] > 3 or not clamped or metrics['carrier_moved_mm'] > 6:
            raise DemoFailure(f"assembly not held: {metrics}")
        return metrics


def attempt(seed):
    try:
        planner = Planner(seed)
        metrics = planner.run()
    except DemoFailure as error:
        return None, dict(seed=seed, failure=str(error))
    start = planner.record_from
    arrays = {k: np.stack([row[k] for row in planner.rows[start:]]) for k in planner.rows[0]}
    planner.stages = [dict(st, start=st['start'] - start) for st in planner.stages if st['start'] >= start]
    return arrays, dict(seed=seed, metrics=metrics, stages=planner.stages, speed_m_s=planner.speed,
                        clearance_m=planner.clearance, assembly=planner.assembly.tolist(),
                        overhead_setup=planner.overhead_setup)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output', type=Path, default=ROOT / 'data/bimanual')
    parser.add_argument('--episodes', type=int, default=200)
    parser.add_argument('--seed', type=int, default=20261006)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--max-attempts', type=int, default=5000)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    existing = sorted(args.output.glob('episode_*/data.npz'))
    log = args.output / 'attempts.jsonl'
    tried = [json.loads(line)['seed'] for line in log.read_text().splitlines()] if log.exists() else []
    seed, count, started = max([args.seed - 1] + tried) + 1, len(existing), time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        while count < args.episodes and seed - args.seed < args.max_attempts:
            seeds = range(seed, seed + args.workers)
            for arrays, result in pool.map(attempt, seeds):
                if arrays is not None and count < args.episodes:
                    name = f'episode_{count:04d}'
                    temporary = args.output / f'.{name}.tmp'
                    shutil.rmtree(temporary, ignore_errors=True)
                    temporary.mkdir()
                    np.savez_compressed(temporary / 'data.npz', **arrays)
                    (temporary / 'meta.json').write_text(json.dumps(dict(
                        task=TASK, fps=FPS, scene='sim/bimanual.py', frames=len(arrays['state']),
                        state='left j1-6, left gripper rad, right j1-6, right gripper rad (real dataset order)',
                        **result), indent=2) + '\n')
                    temporary.rename(args.output / name)
                    result['episode'] = name
                    count += 1
                with log.open('a') as stream:
                    stream.write(json.dumps({k: v for k, v in result.items() if k != 'stages'}) + '\n')
                print(json.dumps(dict(accepted=count, **{k: v for k, v in result.items() if k != 'stages'})), flush=True)
            seed += len(seeds)
            (args.output / 'status.json').write_text(json.dumps(dict(
                accepted=count, requested=args.episodes, attempts=seed - args.seed,
                elapsed_s=round(time.time() - started), complete=count >= args.episodes), indent=2) + '\n')
    if count < args.episodes:
        raise SystemExit(f'only {count}/{args.episodes} accepted')


if __name__ == '__main__':
    main()
