#!/usr/bin/env python3
"""Expert-in-the-loop data collection for ACT (DAgger with the scripted planner).

ACT runs in the simulator exactly as in ``rollout_act.py``. Privileged checks
watch for the onset of a failure; the scripted planner then takes over that same
simulation (``Planner.finish``), recovers into a demonstrated state and
completes the stack the way the demonstrations do. Only the expert's ticks are
recorded, in the native episode format, so the corrections can be exported
together with the base demonstrations and ACT retrained from scratch.

Triggers, checked every control tick:

- ``misaligned_close``: the gripper command starts closing while the grip site
  is not at the planned grasp point of the cube that is due next.
- ``misaligned_release``: the jaws start opening while the held cube is not
  over its support.
- ``wrong_block``: a cube other than the next one is grasped.
- ``dropped``: a lifted cube is lost without completing its pair.
- ``stall``: no progress for a while: no new pair stacked, and the grip site
  (or the held cube) has not come 1 cm closer to its current goal.

Collection stops after ``--interventions`` successful corrections, so rounds
are compared on the same amount of correction data.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'sim'), str(ROOT / 'teleop')]
from teleop.dataset_contract import DEFAULT_DATASET, PhysicsClock, file_hash, resolve_control_hz

TABLE_CUBE_Z = .0725  # cube centre resting on the table


class FailureMonitor:
    """Detect the onset of a failure from privileged simulator state."""

    def __init__(self, probe, hz: float, stall_seconds: float, misalign: float):
        self.probe, self.sim = probe, probe.sim
        self.stall_ticks = round(stall_seconds * hz)
        self.misalign = misalign
        self.grip = 1.
        self.best = (-1, np.inf)  # (stage, distance to the stage's goal)
        self.since_progress = 0
        self.was_aloft = False
        self.lost = None  # (ticks since a lifted cube was released, pairs then)

    def update(self, grip_command: float) -> str | None:
        from tools.collect_scripted import grasp_rotation
        sim, probe = self.sim, self.probe
        positions = sim.object_poses()[0]
        done = probe.completed_pairs()
        held = np.flatnonzero(sim.grasp_flags())
        previous_grip, self.grip = self.grip, grip_command
        if done == len(probe.pairs):
            return None
        block, support = probe.pairs[done]
        if any(h != block for h in held):
            return 'wrong_block'

        aloft = block in held and positions[block, 2] > TABLE_CUBE_Z + .02
        stage = 2 * done + int(aloft)
        if block in held:
            goal = np.linalg.norm(positions[block] - positions[support] - [0, 0, .045])
        else:
            rotation = grasp_rotation(sim, block)
            goal = np.linalg.norm(sim.ee_pos() - positions[block] - .018 * rotation[:, 0])
        best_stage, best_goal = self.best
        if stage > best_stage or (stage == best_stage and goal < best_goal - .01):
            self.best, self.since_progress = (stage, goal), 0
        else:
            self.since_progress += 1
            if self.since_progress >= self.stall_ticks:
                return 'stall'

        if self.lost is not None:
            ticks, done_then = self.lost
            if done > done_then or block in held:
                self.lost = None
            elif ticks >= 15:  # 0.5 s without the pair completing
                return 'dropped'
            else:
                self.lost = (ticks + 1, done_then)
        if self.was_aloft and block not in held:
            self.lost = (0, done)
        self.was_aloft = aloft

        closing = previous_grip >= .5 > grip_command
        opening = previous_grip < .5 <= grip_command
        if closing and block not in held:
            rotation = grasp_rotation(sim, block)
            grasp = positions[block] + .018 * rotation[:, 0]
            offset = sim.ee_pos() - grasp
            if np.linalg.norm(offset[:2]) > self.misalign or abs(offset[2]) > .025:
                return 'misaligned_close'
        if opening and block in held:
            above = positions[block] - positions[support]
            if np.linalg.norm(above[:2]) > .012 or not .032 < above[2] < .070:
                return 'misaligned_release'
        return None


def load_policy(args):
    """Load an ACT checkpoint with the same inference settings as rollout_act.py."""
    import torch
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors

    checkpoint = args.checkpoint.expanduser().resolve()
    inference_path = checkpoint / 'inference.json'
    saved = json.loads(inference_path.read_text()) if inference_path.is_file() else {}
    config = PreTrainedConfig.from_pretrained(checkpoint)
    action_steps = args.action_steps if args.action_steps is not None else saved.get('action_steps')
    if action_steps is not None:
        config.n_action_steps = action_steps
    temporal = saved.get('temporal_ensemble', True) if args.action_steps is None else False
    if temporal:
        config.n_action_steps = 1
        config.temporal_ensemble_coeff = .01
    else:
        config.temporal_ensemble_coeff = None
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    policy = ACTPolicy.from_pretrained(checkpoint, config=config).to(device)
    policy.eval()
    pre, post = make_pre_post_processors(policy.config, pretrained_path=str(checkpoint))
    contract_path = checkpoint / 'deployment.json'
    contract = json.loads(contract_path.read_text()) if contract_path.is_file() else {}
    if contract.get('action_representation', 'absolute') != 'absolute':
        raise SystemExit('DAgger collection supports absolute-action checkpoints only')
    metadata = LeRobotDatasetMetadata('local/panthera_stack', root=args.dataset)
    hz = resolve_control_hz(checkpoint, None, metadata.fps)
    if hz != 30:
        raise SystemExit('The scripted expert records at 30 Hz; use a 30 Hz checkpoint')
    return policy, pre, post, device, contract, hz, dict(temporal_ensemble=temporal,
                                                         action_steps=config.n_action_steps)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, default=DEFAULT_DATASET,
                        help="the checkpoint's training dataset (for its frame rate)")
    parser.add_argument('--output', type=Path, required=True, help='new directory of correction episodes')
    parser.add_argument('--interventions', type=int, default=100, help='successful corrections to collect')
    parser.add_argument('--max-rollouts', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=20262001, help='first scene seed; keep apart from evaluation')
    parser.add_argument('--seconds', type=float, default=45., help='policy time limit per rollout')
    parser.add_argument('--stall-seconds', type=float, default=6.)
    parser.add_argument('--misalign', type=float, default=.015, help='m, horizontal grasp tolerance')
    parser.add_argument('--action-steps', type=int)
    parser.add_argument('--mujoco-gl', default='egl', choices=('egl', 'glfw', 'osmesa', 'cgl'))
    args = parser.parse_args()
    os.environ.setdefault('MUJOCO_GL', args.mujoco_gl)
    import mujoco
    import torch
    from lerobot.policies.utils import prepare_observation_for_inference
    from sim.dynamics import CONTACT_DYNAMICS
    from sim.panthera_env import PantheraSim
    from teleop.keyboard import DEFAULT_ARM_START_RANGE, randomize_arm_start
    from teleop.render_vla_dataset import shoulder_camera, wrist_camera
    from tools.act_scene import reset_fixed_arm, rollout_metrics
    from tools.collect_scripted import DemoFailure, Planner

    if args.output.exists() and any(args.output.glob('episode_*')):
        raise SystemExit('Output already holds episodes; use a new directory per round.')
    args.output.mkdir(parents=True, exist_ok=True)
    policy, pre, post, device, contract, hz, inference = load_policy(args)
    environment = contract.get('environment', {})
    scene = ROOT / environment.get('scene', 'sim/panthera/scene.xml')
    sim = PantheraSim(scene, dynamics=CONTACT_DYNAMICS)
    blocks = len(sim.object_names)
    renderers = [mujoco.Renderer(sim.model, height=256, width=256) for _ in range(2)]
    cameras = [shoulder_camera(sim.model), wrist_camera(sim.model)]
    task = environment.get('task', 'stack the three colored cubes')
    hold_ticks = max(1, round(environment.get('success_hold_seconds', .5) * hz))
    log = (args.output / 'rollouts.jsonl').open('a')
    signature = file_hash(ROOT / 'tools/collect_scripted.py')
    saved = rollouts = 0
    started = time.time()
    counts: dict[str, int] = {}
    try:
        while saved < args.interventions and rollouts < args.max_rollouts:
            seed = args.seed + rollouts
            rollouts += 1
            rng = np.random.default_rng(seed)
            sim.reset(randomize=True, rng=rng)
            if environment.get('arm_start') == 'fixed':
                reset_fixed_arm(sim, environment)
            else:
                randomize_arm_start(sim, DEFAULT_ARM_START_RANGE, rng=rng)
            policy.reset()
            clock = PhysicsClock(hz, sim.dt)
            monitor = FailureMonitor(Planner(seed, blocks=blocks, sim=sim), hz, args.stall_seconds, args.misalign)
            trigger, stable, tick = None, 0, 0
            while tick < round(args.seconds * hz):
                images = []
                for renderer, camera in zip(renderers, cameras):
                    renderer.update_scene(sim.data, camera)
                    images.append(renderer.render().copy())
                state = np.concatenate([sim.q, [float(sim.data.ctrl[sim.grip_act])]]).astype(np.float32)
                observation = pre(prepare_observation_for_inference(
                    {'observation.state': state, 'observation.images.shoulder': images[0],
                     'observation.images.wrist': images[1]},
                    device, task=task, robot_type='panthera_ht_sim'))
                with torch.inference_mode():
                    action = post(policy.select_action(observation))
                action = np.asarray(action.detach().cpu(), dtype=np.float64).reshape(-1)
                grip = float(np.clip(action[6] / .04, 0., 1.))
                sim.set_arm_ctrl(np.clip(action[:6], sim.arm_range[:, 0], sim.arm_range[:, 1]))
                sim.set_gripper(grip)
                sim.step(clock.next_steps())
                tick += 1
                metrics = rollout_metrics(sim.object_poses()[0])
                stable = stable + 1 if metrics['task_stack'] and not sim.grasped else 0
                if stable >= hold_ticks:
                    break
                trigger = monitor.update(grip)
                if trigger:
                    break
            record = dict(seed=seed, policy_ticks=tick)
            if stable >= hold_ticks:
                record.update(outcome='policy_success')
            else:
                trigger = trigger or 'timeout'
                expert = Planner(seed, blocks=blocks, sim=sim)
                try:
                    expert.finish()
                except DemoFailure as error:
                    record.update(outcome='expert_failed', trigger=trigger, failure=str(error))
                else:
                    name = f'episode_{saved:04d}'
                    meta = dict(robot='Panthera-HT (HighTorque) 6-DoF + parallel gripper',
                        scene=str(scene.relative_to(ROOT)), control_mode='dagger_expert', task=task,
                        objects=list(sim.object_names), arm_joints=[f'joint{i}' for i in range(1, 7)],
                        gripper_open_m=.04, frames={'ee_*/target_*': 'robot base frame', 'quat': '(w, x, y, z)'},
                        recording={'control_hz': 30, 'clock': 'simulation', 'row_alignment': 'post_action'},
                        generator_sha256=signature, seed=seed, blocks=blocks, stages=expert.stages,
                        dagger=dict(trigger=trigger, policy_ticks=tick, checkpoint=str(args.checkpoint.resolve()),
                                    inference=inference, misalign_m=args.misalign,
                                    stall_seconds=args.stall_seconds))
                    temporary = args.output / f'.{name}.tmp'
                    expert.episode.save(temporary, meta, 30, False)
                    temporary.rename(args.output / name)
                    saved += 1
                    record.update(outcome='corrected', trigger=trigger, episode=name,
                                  expert_ticks=len(expert.episode))
            counts[record['outcome']] = counts.get(record['outcome'], 0) + 1
            log.write(json.dumps(record) + '\n'); log.flush()
            print(json.dumps(dict(rollout=rollouts, saved=saved, **record)), flush=True)
    finally:
        log.close()
        for renderer in renderers:
            renderer.close()
    status = dict(complete=True, interventions=saved, rollouts=rollouts, outcomes=counts,
                  policy_success_rate=counts.get('policy_success', 0) / max(rollouts, 1),
                  checkpoint=str(args.checkpoint.resolve()), seed=args.seed,
                  elapsed_s=time.time() - started)
    (args.output / 'status.json').write_text(json.dumps(status, indent=2) + '\n')
    print(json.dumps(status, indent=2))


if __name__ == '__main__':
    main()
