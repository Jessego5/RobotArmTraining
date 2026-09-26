#!/usr/bin/env python3
"""Collect physics-validated scripted stacks in the native teleop episode format.

``--waits state`` ends each move once the arm and fingers have settled, instead
of after fixed ticks, and skips zero-length moves.  Timed waits leave the robot
motionless while the label switches from "hold" to "move" at an arbitrary tick,
so near-identical observations carry different futures.

``--augment-fraction`` perturbs that share of episodes DART-style: a smooth
random offset on the executed target, sideways first grasp attempts, and
re-grasps after a miss.  Every row also records ``ctrl_label``: the clean plan
solved from the same controller state, so exported actions correct the offset
instead of copying it.  The gripper label only closes or opens once the grip
site is within ``ALIGN_TOL`` of the planned point.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from functools import partial
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import mujoco
import numpy as np
from sim.panthera_env import GRIPPER_OPEN, PantheraSim, mat_to_quat
from sim.stack_task import stack_metrics, ordered_two_stack_metrics, CUBE_EDGE
from teleop.dataset_contract import PhysicsClock, file_hash
from teleop.episode import Episode


class DemoFailure(RuntimeError):
    pass


@dataclass(frozen=True)
class Augment:
    noise_pos: float = 0.      # m, stationary std of the executed-target offset
    noise_yaw: float = 0.      # rad, about world z
    noise_tau: float = .5      # s, Ornstein-Uhlenbeck correlation time
    miss_prob: float = 0.      # chance a grasp's first attempt starts sideways
    miss_offset: float = 0.    # m, largest sideways offset
    max_attempts: int = 1      # grasp attempts per block

    @property
    def active(self) -> bool:
        return self != Augment()


def yaw_quat(angle: float) -> np.ndarray:
    return np.array([np.cos(angle / 2), 0., 0., np.sin(angle / 2)])


def grasp_rotation(sim, index):
    """Face-aligned jaws, with a downward approach that stays in the IK workspace."""
    rotation = sim.data.xmat[sim.object_bodies[index]].reshape(3, 3)
    yaw = math.atan2(rotation[1, 0], rotation[0, 0])
    # Equivalent cube faces; choose the yaw closest to the robot's forward axis.
    yaw = (yaw + np.pi / 4) % (np.pi / 2) - np.pi / 4
    pitch = np.deg2rad(55)
    c, s = np.cos(yaw), np.sin(yaw)
    rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.]])
    c, s = np.cos(pitch), np.sin(pitch)
    return rz @ np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


class Planner:
    ALIGN_TOL = .015  # m, horizontal; the jaw opens ~8 cm around a 4.5 cm cube

    def __init__(self, seed: int, blocks: int = 3, arm_start: str = "random",
                 waits: str = "timed", augment: Augment | None = None):
        self.blocks = blocks
        self.arm_start = arm_start
        self.waits = waits
        self.augment = augment or Augment()
        # Separate streams keep unaugmented episodes identical to before.
        self.noise_rng = np.random.default_rng([seed, 1])
        self.noise = np.zeros(4)   # x, y, z, yaw
        self.bias = np.zeros(3)    # sideways grasp offset
        self.grip_hold = None      # (planned grip-site position, previous grip)
        self.quiet = False         # fade the noise out, e.g. while placing
        self.retries = 0
        self.scene = ROOT / "sim/panthera" / ("scene_two_blocks.xml" if blocks == 2 else "scene.xml")
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.sim = PantheraSim(self.scene)
        self.sim.reset(rng=self.rng)
        self.clock = PhysicsClock(30, self.sim.dt)
        self.episode = Episode()
        self.stages = []
        self.speed = float(self.rng.uniform(.10, .16))
        self.clearance = float(self.rng.uniform(.06, .08))
        self.target, self.quat = self.sim.ee_pose()
        self.grip = 1.
        self.qctrl = self.sim.q

    def _advance_noise(self):
        a = self.augment
        if a.noise_pos <= 0 and a.noise_yaw <= 0:
            return
        if self.quiet:
            self.noise *= .85  # ~0.2 s to fade
            return
        dt = 1 / 30
        # Weaker vertical noise: pressing into the table or a cube cannot be tracked.
        sigma = np.array([a.noise_pos, a.noise_pos, .4 * a.noise_pos, a.noise_yaw])
        self.noise += (-self.noise * dt / a.noise_tau
                       + sigma * np.sqrt(2 * dt / a.noise_tau) * self.noise_rng.standard_normal(4))

    def executed(self, target, quat):
        """The perturbed target actually commanded for a planned pose."""
        if not (self.noise.any() or self.bias.any()):
            return np.asarray(target), np.asarray(quat)
        executed_quat = np.zeros(4)
        mujoco.mju_mulQuat(executed_quat, yaw_quat(self.noise[3]), np.asarray(quat, float))
        return np.array(target) + self.noise[:3] + self.bias, executed_quat

    def tick(self, target, quat, grip):
        sim = self.sim
        self._advance_noise()
        perturbed = self.noise.any() or self.bias.any()
        executed_target, executed_quat = self.executed(target, quat)
        previous = self.qctrl
        q, pe, re = sim.ik(executed_target, executed_quat, q_init=previous, iters=35,
                           max_joint_step=.08, posture_gain=0.)
        # The label is the clean plan solved from the same controller state.
        label_q = sim.ik(target, quat, q_init=previous, iters=35, max_joint_step=.08,
                         posture_gain=0.)[0] if perturbed else q
        label_grip = grip
        if self.grip_hold is not None:
            planned, before = self.grip_hold
            if np.linalg.norm((sim.ee_pos() - planned)[:2]) > self.ALIGN_TOL:
                label_grip = before  # re-align before operating the jaws
        self.qctrl = q
        sim.set_arm_ctrl(q)
        sim.set_gripper(grip)
        ticks = self.clock.next_steps()
        sim.step(ticks)
        if not np.isfinite(sim.data.qpos).all():
            raise DemoFailure('nonfinite simulator state')
        pos, rot = sim.object_poses()
        self.episode.add(dict(t=sim.data.time, sim_time=sim.data.time,
            physics_steps=ticks, finger_q=sim.data.qpos[sim.finger_qadr].copy(),
            finger_dq=sim.data.qvel[sim.finger_dofadr].copy(),
            q=sim.q, dq=sim.dq, ctrl=sim.data.ctrl.copy(),
            ee_pos=sim.ee_pos(), ee_quat=sim.ee_quat(), obj_pos=pos, obj_quat=rot,
            target_pos=np.array(target), target_quat=np.array(quat), gripper=grip,
            ik_pos_err=pe, ik_rot_err=re,
            ctrl_label=np.r_[label_q, np.clip(label_grip, 0, 1) * GRIPPER_OPEN]))
        self.target, self.quat, self.grip = np.array(target), np.array(quat), grip

    def move(self, name, target, quat=None, grip=None, duration=None):
        target = np.array(target)
        quat = self.quat.copy() if quat is None else np.array(quat)
        grip = self.grip if grip is None else grip
        start, qstart, gstart = self.target.copy(), self.quat.copy(), self.grip
        if np.dot(quat, qstart) < 0:
            quat = -quat
        angle = 2 * np.arccos(np.clip(np.dot(quat, qstart), -1, 1))
        if (self.waits == 'state' and grip == gstart and angle < 1e-3
                and np.linalg.norm(target - start) < 1e-3):
            return  # a zero-length move would only be a timed pause
        self.stages.append(dict(name=name, start=len(self.episode)))
        duration = duration or max(.4, 1.5 * np.linalg.norm(target-start)/self.speed, angle/.7)
        for i in range(1, math.ceil(duration*30)+1):
            u = i / math.ceil(duration*30)
            u = u*u*(3-2*u)
            q = qstart*(1-u)+quat*u
            q /= np.linalg.norm(q)
            self.tick(start*(1-u)+target*u, q, gstart*(1-u)+grip*u)
        # Let the position servos finish tracking before a contact transition.
        if self.waits == 'state':
            # Observable end condition instead of a fixed count; capped at 1 s.
            for _ in range(30):
                if self.settled(target, quat):
                    break
                self.tick(target, quat, grip)
        else:
            for _ in range(9):
                self.tick(target, quat, grip)
        error = np.linalg.norm(self.sim.ee_pos() - self.executed(target, quat)[0])
        # Perturbed targets can press into contacts; allow for the noise scale.
        if error > .012 + 3 * self.augment.noise_pos:
            raise DemoFailure(f'{name}: tracking error {error:.4f}')

    def settled(self, target, quat) -> bool:
        sim = self.sim
        return (np.linalg.norm(sim.ee_pos() - self.executed(target, quat)[0]) < .003
                and np.abs(sim.dq).max() < .05
                and np.abs(sim.data.qvel[sim.finger_dofadr]).max() < .005)

    def hold(self, name, seconds, grip=None):
        self.move(name, self.target, grip=grip, duration=seconds)

    def operate(self, name, seconds, grip, gated=True):
        """Close or open the jaws.

        When `gated`, labels keep the jaws as they were until the grip site is
        within ALIGN_TOL of the plan: grasp or release only when aligned.
        Re-opening after a miss is the correct action from anywhere.
        """
        self.grip_hold = (self.target.copy(), self.grip) if gated else None
        try:
            self.hold(name, seconds, grip)
        finally:
            self.grip_hold = None

    def run(self):
        sim = self.sim
        # Fixed starts do not depend on object yaw or the episode RNG.
        for _ in range(1 if self.arm_start == 'fixed' else 100):
            if self.arm_start == 'fixed':
                start = np.array([.38, 0., .24])
                pitch = np.deg2rad(55)
                c, s = np.cos(pitch), np.sin(pitch)
                r = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
            else:
                start = self.rng.uniform([.30,-.16,.18], [.46,.16,.28])
                r = grasp_rotation(sim, 0)
            quat = mat_to_quat(r)
            q, pe, re = sim.ik(start, quat, max_joint_step=None, iters=150)
            if pe < .001 and re < .01:
                break
        else:
            raise DemoFailure('initial IK')
        self.initial_arm_q = q.tolist()
        self.initial_target_pos = start.tolist()
        self.initial_target_quat = quat.tolist()
        sim.data.qpos[sim.arm_qadr] = q
        sim.data.qvel[:] = 0
        sim.data.qpos[sim.finger_qadr] = [.04,-.04]
        sim.set_arm_ctrl(q); sim.set_gripper(1)
        mujoco.mj_forward(sim.model, sim.data)
        self.target, self.quat, self.qctrl = sim.ee_pos(), sim.ee_quat(), q
        self.hold('settle', .3)
        # Consistent color order avoids an ambiguous multimodal imitation target.
        # Red on green, then blue on red.
        pairs = ((0, 1),) if self.blocks == 2 else ((0, 1), (2, 0))
        for level, (block, support) in enumerate(pairs, start=1):
            for attempt in range(self.augment.max_attempts):
                # Re-plan from the cube's current pose; a miss can nudge it.
                r = grasp_rotation(sim, block)
                quat = mat_to_quat(r)
                offset = .018*r[:,0]
                obj = sim.object_poses()[0][block]
                grasp = obj + offset
                above = grasp.copy(); above[2] += self.clearance
                if attempt == 0:
                    # Lift before lateral travel so the open fingers clear other cubes.
                    safe = self.target.copy(); safe[2] = max(safe[2], above[2]+.015)
                    self.move(f'{level}_clear', safe)
                self.move(f'{level}_approach', above, quat, 1.)
                if attempt == 0 and self.noise_rng.random() < self.augment.miss_prob:
                    heading = self.noise_rng.uniform(0, 2*np.pi)
                    self.bias = (self.augment.miss_offset * self.noise_rng.uniform(.6, 1.)
                                 * np.array([np.cos(heading), np.sin(heading), 0.]))
                self.move(f'{level}_descend', grasp)
                self.operate(f'{level}_close', .5, 0.)
                self.bias = np.zeros(3)
                if sim.data.eq_active[sim._grasp_eq[block]]:
                    break
                if attempt + 1 == self.augment.max_attempts:
                    raise DemoFailure(f'{level}: no bilateral grasp')
                self.retries += 1
                self.operate(f'{level}_reopen', .3, 1., gated=False)
                self.move(f'{level}_backoff', above)
            self.move(f'{level}_lift', above)
            if sim.object_poses()[0][block,2] < obj[2]+.04:
                raise DemoFailure(f'{level}: failed lift')
            # Measured held offset accounts for contact seating and servo lag.
            held_offset = sim.ee_pos()-sim.object_poses()[0][block]
            dest = sim.object_poses()[0][support] + [0,0,CUBE_EDGE+.014]
            place = dest + held_offset
            transit = place.copy(); transit[2] += self.clearance
            lift = self.target.copy(); lift[2] = transit[2]
            self.move(f'{level}_raise', lift)
            self.move(f'{level}_transfer', transit)
            # Acceptance needs a stack aligned within 12 mm, so the
            # perturbation fades while placing; labels are the plan throughout.
            self.quiet = True
            self.move(f'{level}_place', place)
            self.operate(f'{level}_release', .5, 1.)
            self.move(f'{level}_retreat', transit)
            self.quiet = False
        self.stages.append(dict(name='validate', start=len(self.episode)))
        for _ in range(30):
            self.tick(self.target, self.quat, 1.)
            metrics = (ordered_two_stack_metrics(sim.object_poses()[0]) if self.blocks == 2
                       else stack_metrics(sim.object_poses()[0]))
            if not metrics['two_stack' if self.blocks == 2 else 'three_stack'] or any(sim.data.eq_active[e] for e in sim._grasp_eq):
                raise DemoFailure('released stack did not remain stable for one second')
            p = sim.object_poses()[0][[1,0] if self.blocks == 2 else [1,0,2]]
            if np.max(np.linalg.norm(np.diff(p[:,:2],axis=0),axis=1)) > .012:
                raise DemoFailure('stack alignment exceeds 12 mm')
        return metrics


def attempt(seed, blocks=3, arm_start="random", waits="timed", augment=None,
            augment_fraction=0.):
    # A per-seed draw, so the choice does not depend on scheduling or failures.
    augmented = bool(augment_fraction) and np.random.default_rng([seed, 2]).random() < augment_fraction
    planner = Planner(seed, blocks=blocks, arm_start=arm_start, waits=waits,
                      augment=augment if augmented else None)
    try:
        metrics = planner.run()
        return planner.episode, dict(seed=seed, stages=planner.stages, metrics=metrics,
            speed_m_s=planner.speed, approach_clearance_m=planner.clearance,
            initial_arm_q=planner.initial_arm_q, initial_target_pos=planner.initial_target_pos,
            initial_target_quat=planner.initial_target_quat, arm_start=arm_start, blocks=blocks,
            waits=waits, augmented=augmented, grasp_retries=planner.retries,
            **(dict(augment=asdict(augment)) if augmented else {}))
    except DemoFailure as error:
        return None, dict(seed=seed, failure=str(error), augmented=augmented)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'data/scripted_stack')
    parser.add_argument('--blocks', type=int, choices=(2, 3), default=3)
    parser.add_argument('--arm-start', choices=('fixed', 'random'), default='random')
    parser.add_argument('--episodes', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=230923)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--max-attempts', type=int, default=10000)
    parser.add_argument('--min-free-gb', type=float, default=5.)
    parser.add_argument('--waits', choices=('timed', 'state'), default='timed',
                        help='state: end moves when the arm has settled, skip zero-length moves')
    parser.add_argument('--augment-fraction', type=float, default=0.,
                        help='share of episodes with perturbations, re-grasps and corrective labels')
    parser.add_argument('--noise-pos', type=float, default=.008, help='m, executed-target noise std')
    parser.add_argument('--noise-yaw-deg', type=float, default=4.)
    parser.add_argument('--miss-prob', type=float, default=.4)
    parser.add_argument('--miss-offset', type=float, default=.025, help='m')
    parser.add_argument('--grasp-attempts', type=int, default=3)
    args = parser.parse_args()
    if min(args.episodes,args.workers,args.max_attempts,args.grasp_attempts) < 1:
        parser.error('episodes, workers, max-attempts and grasp-attempts must be positive')
    if not 0 <= args.augment_fraction <= 1:
        parser.error('augment-fraction must be between 0 and 1')
    augment = Augment(noise_pos=args.noise_pos, noise_yaw=np.deg2rad(args.noise_yaw_deg),
                      miss_prob=args.miss_prob, miss_offset=args.miss_offset,
                      max_attempts=args.grasp_attempts)
    args.output.mkdir(parents=True, exist_ok=True)
    signature = file_hash(Path(__file__))
    contract = dict(generator_sha256=signature, seed=args.seed, control_hz=30,
        blocks=args.blocks, arm_start=args.arm_start, waits=args.waits,
        augment_fraction=args.augment_fraction,
        **(dict(augment=asdict(augment)) if args.augment_fraction else {}),
        task_metrics_sha256=file_hash(ROOT/"sim/stack_task.py"),
        scene_sha256={p.name:file_hash(p) for p in sorted((ROOT/'sim/panthera').glob('*.xml'))},
        simulator_sha256=file_hash(ROOT/'sim/panthera_env.py'))
    contract_path = args.output/'collection.json'
    if contract_path.exists() and json.loads(contract_path.read_text()) != contract:
        raise SystemExit('Collection code/settings changed; use a new output directory.')
    if not contract_path.exists() and list(args.output.glob('episode_*')):
        raise SystemExit('Output contains episodes without a collection contract; use a new directory.')
    contract_path.write_text(json.dumps(contract, indent=2)+'\n')
    existing = sorted(args.output.glob('episode_*/meta.json'))
    for i, path in enumerate(existing):
        if path.parent.name != f'episode_{i:04d}' or not (path.parent/'data.npz').exists():
            raise SystemExit('Incomplete or noncontiguous collection; inspect before resuming.')
    count = len(existing)
    log = args.output/'attempts.jsonl'
    prior = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
    seed = max([args.seed-1]+[item['seed'] for item in prior]+
               [json.loads(p.read_text())['seed'] for p in existing])+1
    started = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        while count < args.episodes and seed-args.seed < args.max_attempts:
            if shutil.disk_usage(args.output).free < args.min_free_gb*1024**3:
                raise RuntimeError('Low disk space; collector stopped safely. Free space and resume.')
            seeds = range(seed, seed+min(args.workers, args.episodes-count, args.max_attempts-(seed-args.seed)))
            for episode, result in pool.map(partial(attempt, blocks=args.blocks, arm_start=args.arm_start,
                                                    waits=args.waits, augment=augment,
                                                    augment_fraction=args.augment_fraction), seeds):
                if episode is not None:
                    name = f'episode_{count:04d}'
                    meta = dict(robot='Panthera-HT (HighTorque) 6-DoF + parallel gripper',
                        scene='sim/panthera/scene_two_blocks.xml' if args.blocks == 2 else 'sim/panthera/scene.xml',
                        control_mode='scripted_ik',
                        task='stack the red cube on the green cube' if args.blocks == 2 else 'stack the three colored cubes',
                        objects=['cube_red','cube_green'] if args.blocks == 2 else ['cube_red','cube_green','cube_blue'],
                        arm_joints=[f'joint{i}' for i in range(1,7)], gripper_open_m=.04,
                        frames={'ee_*/target_*':'robot base frame','quat':'(w, x, y, z)'},
                        recording={'control_hz':30,'clock':'simulation','row_alignment':'post_action'},
                        generator_sha256=signature, **result)
                    temporary = args.output/f'.{name}.tmp'
                    episode.save(temporary, meta, 30, False)
                    temporary.rename(args.output/name)
                    result['episode'] = name
                    count += 1
                with log.open('a') as stream:
                    stream.write(json.dumps(result)+'\n')
                print(json.dumps(dict(accepted=count, requested=args.episodes, **result)), flush=True)
            seed += len(seeds)
            status = dict(accepted=count, requested=args.episodes, attempts=seed-args.seed,
                next_seed=seed, elapsed_s=time.time()-started,
                free_gb=shutil.disk_usage(args.output).free/1024**3, complete=count>=args.episodes)
            temp = args.output/'status.tmp'; temp.write_text(json.dumps(status,indent=2)+'\n')
            temp.replace(args.output/'status.json')
    if count < args.episodes:
        raise SystemExit(f'Only {count}/{args.episodes} accepted before max-attempts')


if __name__ == '__main__':
    main()
