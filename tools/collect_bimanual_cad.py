#!/usr/bin/env python3
"""Collect Zeb's CAD place-then-push assembly in the real-matched two-arm layout.

Runs `tools/bimanual_actuator_ik.py`'s controller (the right arm holds the
large carrier on its fixture; the left half-inserts gear 3, pushes it home,
flattens it, then guides pin 2 through gear and carrier and presses it) on
`sim/bimanual_cad.py`. An episode is kept when the carrier is held in place,
the gear is seated and the pin passes through both bores. Full pin depth is
recorded but not required: Zeb's press often stops a few millimetres short,
and the real recordings also end with the assembly held rather than certified.

Episodes are saved in the format `tools/export_bimanual_dataset.py` reads, at
20 Hz and in the real dataset's units (radians; grippers 0 closed .. 2 open),
starting once the carrier is held on the fixture, as the real episodes do.

Zeb's pace (4 cm/s, 0.6 rad/s joint cap) makes an assembly ~130 s long; the real
demonstrations average ~12 s. `--speed 4` (default) moves 4x faster with the
real arms' joint speeds (to 1.2 rad/s) in free motion, keeping his gentle
bounds during contact pushes: ~31 s per episode.

    python tools/collect_bimanual_cad.py --output data/bimanual_cad --episodes 200 --workers 16
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MUJOCO_GL', 'egl')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np

from sim.bimanual import OVERHEAD_SETUPS, opening_to_rad
from sim.bimanual_cad import LEFT_CLEAR, ParallelCADSim
from sim.panthera_env import mat_to_quat
from tools.bimanual_actuator_ik import BimanualTrial, LeftTaskView
from tools.try_actuator_assembly import gear_grasp, gear_placement, pin_placement

FPS = 20
TASK = 'put the gear in the carrier and put a pin in it'
START_STAGE = 'carrier_supported_hold'


class ParallelTrial(BimanualTrial):
    """Zeb's trial with the layout's clearance point, per-tick velocity and effort, and a speed-up.

    Zeb's controller moves at 4 cm/s with long settles and force-limited pushes,
    so an assembly takes ~130 s; the real demonstrations take ~12 s. `speed`
    scales free-motion speed and shortens settles and pushes.
    """

    def __init__(self, *args, speed=1., **kw):
        super().__init__(*args, **kw)
        self.velocities, self.efforts = [], []
        self.speed = speed

    def move(self, stage, pos, quat=None, grip=None, duration=None):
        import math
        from tools.bimanual_actuator_ik import smoothstep
        k = self.speed
        self.stage = stage
        pos = np.asarray(pos, dtype=float)
        quat = self.quat.copy() if quat is None else np.asarray(quat, dtype=float)
        grip = self.grip if grip is None else grip
        p0, q0, g0 = self.target.copy(), self.quat.copy(), self.grip
        if q0 @ quat < 0:
            quat = -quat
        angle = 2 * np.arccos(np.clip(q0 @ quat, -1, 1))
        duration = max((duration or 0.) / k, 1. / k, 1.875 * np.linalg.norm(pos - p0) / (.04 * k),
                       1.875 * angle / (.6 * k))
        steps = math.ceil(duration * 20)
        self.stages.append(dict(name=stage, arm=self.arm, frame=len(self.frames), contact_mode=self.contact_mode,
                                duration_s=steps / 20, target_position=pos.tolist(),
                                target_quaternion=quat.tolist()))
        for i in range(1, steps + 1):
            u = smoothstep(i / steps)
            if angle < 1e-8:
                q = q0.copy()
            else:
                half = angle / 2
                q = (np.sin((1 - u) * half) * q0 + np.sin(u * half) * quat) / np.sin(half)
            self.tick(p0 + u * (pos - p0), q, g0 + u * (grip - g0))
        for _ in range(max(4, round(12 / k))):
            self.tick(pos, quat, grip)
        pe, re, tracking, _ = self.tracking[-1]
        self.stages[-1].update(ik_position_error_mm=pe * 1000, tracking_position_error_mm=tracking * 1000)
        if pe > .003 or re > .04 or tracking > .008:
            raise RuntimeError(f'{stage}: unresolved motion ({pe*1000:.2f} mm IK, {tracking*1000:.2f} mm tracking)')

    def push(self, stage, end, direction, tangent, travel, duration=5., search_radius=.0003):
        super().push(stage, end, direction, tangent, travel, max(1.5, duration / self.speed), search_radius)

    def tick(self, pos, quat, grip):
        """Zeb's tick with joint-speed caps from the real arms during free motion.

        Zeb bounds commands to 0.6 rad/s and 2 rad/s^2 throughout. The real
        recordings reach 1-1.5 rad/s (99th percentile), so free motion allows
        1.2 rad/s and 5 rad/s^2; contact pushes keep his gentler bounds.
        """
        s = self.sim if not isinstance(self.sim, LeftTaskView) else self.sim.parent
        self.poses[self.arm] = (np.asarray(pos).copy(), np.asarray(quat).copy())
        self.grips[self.arm] = grip
        errors = {}
        vmax, amax = (.6, .1) if self.contact_mode else (1.2, .25)
        for side, a in s.arms.items():
            p, q = self.poses[side]
            desired, pe, re = a.ik(p, q, q_init=self.ideal[side], iters=80, max_joint_step=.03 * vmax / .6,
                                  posture_gain=0, min_damping=.001,
                                  pos_tol=.00002 if self.contact_mode else .00015, rot_tol=.001)
            if not (side == self.arm and self.contact_mode):
                self.bias[side] = np.clip(self.bias[side] + .15 * (self.ideal[side] - a.q), -.012, .012)
            self.ideal[side] = desired.copy()
            desired = np.clip(desired + self.bias[side], *a.arm_range.T)
            velocity = np.clip((desired - self.commands[side]) / .05, -vmax, vmax)
            velocity = np.clip(velocity, self.velocity[side] - amax, self.velocity[side] + amax)
            self.commands[side] = np.clip(self.commands[side] + .05 * velocity, *a.arm_range.T)
            self.velocity[side] = velocity
            a.set_arm_ctrl(self.commands[side])
            a.set_gripper(self.grips[side])
            errors[side] = (pe, re)
        s.step(50)
        self.target, self.quat, self.grip = np.asarray(pos).copy(), np.asarray(quat).copy(), grip
        self.frames.append(s.data.qpos.copy())
        self.controls.append(s.data.ctrl.copy())
        err = s.arms[self.arm]._ee_error(pos, quat)
        self.tracking.append([*errors[self.arm], np.linalg.norm(err[:3]), np.linalg.norm(err[3:])])
        self.states.append(s.state()); self.actions.append(s.action())
        self.stage_ids.append(len(self.stages) - 1)
        if not np.isfinite(s.data.qpos).all():
            raise RuntimeError('Nonfinite physics state')
        velocity, effort = [], []
        for arm in s.arms.values():
            finger = s.data.qvel[arm.finger_dofadr[0]]
            velocity.append(np.r_[s.data.qvel[arm.arm_dofadr], finger * 2 / .04])
            effort.append(np.r_[s.data.actuator_force[arm.act_ids], s.data.actuator_force[arm.grip_act]])
        self.velocities.append(np.concatenate(velocity))
        self.efforts.append(np.concatenate(effort))

    def run(self):
        result = dict(seed=self.sim.seed, success=False)
        try:
            self.select('left')
            _, rotation, _ = gear_grasp(self.sim, 'gear_3')
            self.move('left_clear_support_workspace', LEFT_CLEAR, mat_to_quat(rotation), 1.)
            self.select('right')
            result['carrier'] = self.hold_carrier()
            self.select('left')
            result['gear'] = self.place_push_gear()
            if not result['gear']['success']:
                raise RuntimeError('gear_not_seated_after_push')
            original = self.sim
            self.sim = LeftTaskView(original)
            try:
                result['pin'] = self.pin_trial('pin_2', allow_partial_entry=True)
            finally:
                self.sim = original
            self.move('final_supported_hold', self.target, duration=2.)
        except RuntimeError as error:
            result['failure'] = str(error)
        return result


def evaluate(sim) -> dict:
    gear = gear_placement(sim, 'gear_3')
    pin = pin_placement(sim, 'pin_2')
    target, quat = sim.assembled_pose('large_carrier')
    bid = sim.model.body('large_carrier').id
    carrier_error = float(np.linalg.norm(sim.data.xpos[bid] - target))
    engaged = bool(pin['through_carrier'] and pin['through_gear']
                   and (pin['carrier_bore_error_mm'] or 9) < .25 and (pin['gear_bore_error_mm'] or 9) < .25)
    return dict(gear_seated=gear['success'], gear=gear, pin_engaged=engaged, pin_fully_seated=pin['success'],
                pin=pin, carrier_error_mm=carrier_error * 1000,
                accepted=bool(gear['success'] and engaged and carrier_error < .0015))


def attempt(seed, speed=1.):
    sim = ParallelCADSim(seed)
    with tempfile.TemporaryDirectory() as scratch:
        trial = ParallelTrial(sim, Path(scratch), speed=speed)
        result = trial.run()
    metrics = evaluate(sim)
    report = dict(seed=seed, speed=speed, failure=result.get('failure'), **metrics)
    if not metrics['accepted']:
        return None, report
    start = next(st['frame'] for st in trial.stages if st['name'] == START_STAGE)
    qpos = np.asarray(trial.frames)[start:]
    state = np.asarray(trial.states)[start:].copy()
    state[:, [6, 13]] = opening_to_rad(state[:, [6, 13]])  # metres -> real radians
    arrays = dict(qpos=qpos, state=state.astype(np.float32),
                  velocity=np.asarray(trial.velocities)[start:].astype(np.float32),
                  effort=np.asarray(trial.efforts)[start:].astype(np.float32),
                  # Commanded joints and gripper openings (m): a closed gripper stalled on a part
                  # is commanded shut while its measured opening stays at the part's width.
                  command=np.asarray(trial.actions)[start:].astype(np.float32))
    stages = [dict(name=st['name'], arm=st['arm'], start=st['frame'] - start)
              for st in trial.stages if st['frame'] >= start]
    rng = np.random.default_rng(seed)
    report.update(stages=stages, overhead_setup=str(rng.choice(sorted(OVERHEAD_SETUPS))),
                  fixture=sim.fixture_pose())
    return arrays, report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output', type=Path, default=ROOT / 'data/bimanual_cad')
    parser.add_argument('--episodes', type=int, default=200)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--max-attempts', type=int, default=5000)
    parser.add_argument('--speed', type=float, default=4.,
                        help="x Zeb's free-motion pace; 4 kept 8/8 test seeds (7 fully pressed) at ~31 s")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    count = len(list(args.output.glob('episode_*/data.npz')))
    log = args.output / 'attempts.jsonl'
    tried = [json.loads(line)['seed'] for line in log.read_text().splitlines()] if log.exists() else []
    seed, started = max([args.seed - 1] + tried) + 1, time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        while count < args.episodes and seed - args.seed < args.max_attempts:
            seeds = range(seed, seed + args.workers)
            for arrays, report in pool.map(partial(attempt, speed=args.speed), seeds):
                if arrays is not None and count < args.episodes:
                    name = f'episode_{count:04d}'
                    temporary = args.output / f'.{name}.tmp'
                    shutil.rmtree(temporary, ignore_errors=True)
                    temporary.mkdir()
                    np.savez_compressed(temporary / 'data.npz', **arrays)
                    (temporary / 'meta.json').write_text(json.dumps(dict(
                        task=TASK, fps=FPS, scene='sim/bimanual_cad.py', frames=len(arrays['state']),
                        state='left j1-6, left gripper rad, right j1-6, right gripper rad (real dataset order)',
                        **report), indent=2, default=float) + '\n')
                    temporary.rename(args.output / name)
                    report['episode'] = name
                    count += 1
                summary = {k: report.get(k) for k in ('seed', 'episode', 'accepted', 'gear_seated', 'pin_engaged',
                                                      'pin_fully_seated', 'failure')}
                with log.open('a') as stream:
                    stream.write(json.dumps(summary) + '\n')
                print(json.dumps(dict(kept=count, **summary)), flush=True)
            seed += len(seeds)
            (args.output / 'status.json').write_text(json.dumps(dict(
                accepted=count, requested=args.episodes, attempts=seed - args.seed,
                elapsed_s=round(time.time() - started), complete=count >= args.episodes), indent=2) + '\n')
    if count < args.episodes:
        raise SystemExit(f'only {count}/{args.episodes} accepted')


if __name__ == '__main__':
    main()
