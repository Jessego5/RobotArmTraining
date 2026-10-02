#!/usr/bin/env python3
"""Human-in-the-loop corrections: watch ACT, take over from the keyboard.

ACT drives the arm at 30 Hz from the same shoulder/wrist renders it was trained
on. When it is about to fail, press h: control passes to the keyboard at the
current pose and recording starts immediately, so the correction begins in the
state the policy reached. Recover into a state like the demonstrations, then
finish the job the way they do. e saves the correction and hands control back
to ACT in the same scene (intervene again whenever needed); x discards it and
hands back; r starts a new scene.

Corrections are saved in the native teleop format at exactly 30 Hz of simulated
time, so `export_scripted_dataset.py` and `dagger_rounds.py --mode human`
accept them unchanged. Only the human segments are recorded.

Controls (world frame, as in teleop/keyboard.py while you have control):
    h        take over from ACT          e / x    save / discard, back to ACT
    w s a d  forward/back, left/right    space / shift   up / down
    u j  i k  n m   pitch, yaw, roll     o / l    open / close gripper
    [ ]      slower / faster             1-4 / g  focus a view / all views
    r        new scene                   q        quit

    python tools/hil_keyboard.py --checkpoint outputs/hil/round_1_act/best \\
        --out data/human_corrections
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
os.environ.setdefault('MUJOCO_GL', 'glfw')  # the window's context draws the display

import glfw  # noqa: E402
import mujoco  # noqa: E402

from keyboard import (DEFAULT_ARM_START_RANGE, DEFAULT_REGION, GRIP, ROTATE, TRANSLATE,  # noqa: E402
                      VIEW_KEYS, VIEWS, Input, axis_angle_quat, quadrants, randomize_arm_start,
                      set_view, tiles, _add_sphere)
from panthera_env import PantheraSim, quat_mul  # noqa: E402
from episode import Episode, next_episode_dir  # noqa: E402
from teleop.dataset_contract import PhysicsClock  # noqa: E402

HZ = 30
TICK = 1 / HZ


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, help="checkpoint's training dataset; only its frame rate is read")
    parser.add_argument('--out', type=Path, default=ROOT / 'data/human_corrections')
    parser.add_argument('--seed', type=int, default=20263001, help='first scene; r advances it')
    parser.add_argument('--action-steps', type=int)
    parser.add_argument('--speed', type=float, default=.18, help='m/s while a move key is held')
    parser.add_argument('--rot-speed', type=float, default=1.2, help='rad/s while a rotate key is held')
    parser.add_argument('--grip-speed', type=float, default=2.5, help='full travel per second')
    parser.add_argument('--width', type=int, default=1100)
    parser.add_argument('--height', type=int, default=825)
    args = parser.parse_args()

    import torch
    from lerobot.policies.utils import prepare_observation_for_inference
    from teleop.render_vla_dataset import shoulder_camera, wrist_camera
    from tools.act_scene import reset_fixed_arm, rollout_metrics
    from tools.dagger_act import load_policy

    policy, pre, post, device, contract, hz, inference = load_policy(args)
    environment = contract.get('environment', {})
    task = environment.get('task', 'stack the three colored cubes')
    sim = PantheraSim(ROOT / environment.get('scene', 'sim/panthera/scene.xml'))
    lo, hi = np.array(DEFAULT_REGION[0::2]), np.array(DEFAULT_REGION[1::2])

    if not glfw.init():
        raise SystemExit('could not initialise GLFW -- is there a display?')
    window = glfw.create_window(args.width, args.height, 'ACT  |  h to take over', None, None)
    if not window:
        raise SystemExit('could not open a window')
    glfw.make_context_current(window)
    glfw.swap_interval(1)
    # The policy sees the training renders, from their own offscreen contexts.
    policy_renderers = [mujoco.Renderer(sim.model, height=256, width=256) for _ in range(2)]
    policy_cameras = [shoulder_camera(sim.model), wrist_camera(sim.model)]
    glfw.make_context_current(window)
    cams = {}
    for name in VIEWS:
        cam = mujoco.MjvCamera()
        if set_view(cam, sim.model, name):
            cams[name] = cam
    views, focus = tuple(cams), ''
    opt, scene = mujoco.MjvOption(), mujoco.MjvScene(sim.model, maxgeom=2000)
    ctx = mujoco.MjrContext(sim.model, mujoco.mjtFontScale.mjFONTSCALE_150.value)
    inp = Input(window)
    inp.model, inp.scene, inp.cams = sim.model, scene, cams

    meta = dict(robot='Panthera-HT (HighTorque) 6-DoF + parallel gripper',
                scene=str(sim.scene_path.relative_to(ROOT)), control_mode='human_correction',
                simulation_dynamics=sim.dynamics, task=task, objects=list(sim.object_names),
                arm_joints=[f'joint{i}' for i in range(1, 7)], gripper_open_m=.04,
                frames={'ee_*/target_*': 'robot base frame', 'quat': '(w, x, y, z)'},
                recording={'control_hz': HZ, 'clock': 'simulation', 'row_alignment': 'post_action'},
                policy=dict(checkpoint=str(args.checkpoint.resolve()), inference=inference))

    seed = args.seed
    state = {}

    def new_scene():
        rng = np.random.default_rng(seed)
        sim.reset(randomize=True, rng=rng)
        if environment.get('arm_start') == 'fixed':
            reset_fixed_arm(sim, environment)
        else:
            randomize_arm_start(sim, DEFAULT_ARM_START_RANGE, rng=rng)
        hand_back()

    def hand_back():
        policy.reset()
        state.update(mode='policy', clock=PhysicsClock(HZ, sim.dt), stable=0, episode=None)

    def take_over():
        tp, tq = (p.copy() for p in sim.ee_pose())
        state.update(mode='human', tp=tp, tq=tq, q_cmd=sim.data.ctrl[:6].copy(),
                     grip=float(np.clip(sim.data.ctrl[sim.grip_act] / .04, 0, 1)),
                     episode=Episode(simulation_dynamics=sim.dynamics), taken_at=sim.data.time)

    def policy_tick():
        images = []
        for renderer, camera in zip(policy_renderers, policy_cameras):
            renderer.update_scene(sim.data, camera)
            images.append(renderer.render().copy())
        glfw.make_context_current(window)
        observation = np.concatenate([sim.q, [float(sim.data.ctrl[sim.grip_act])]]).astype(np.float32)
        observation = pre(prepare_observation_for_inference(
            {'observation.state': observation, 'observation.images.shoulder': images[0],
             'observation.images.wrist': images[1]}, device, task=task, robot_type='panthera_ht_sim'))
        with torch.inference_mode():
            action = np.asarray(post(policy.select_action(observation)).detach().cpu(),
                                dtype=np.float64).reshape(-1)
        sim.set_arm_ctrl(np.clip(action[:6], sim.arm_range[:, 0], sim.arm_range[:, 1]))
        sim.set_gripper(float(np.clip(action[6] / .04, 0, 1)))
        sim.step(state['clock'].next_steps())
        metrics = rollout_metrics(sim.object_poses()[0])
        state['stable'] = state['stable'] + 1 if metrics['task_stack'] and not sim.grasped else 0

    def human_tick(gain):
        v = np.zeros(3)
        for key, (axis, sign) in TRANSLATE.items():
            if inp.held(key):
                v[axis] += sign
        state['tp'] = np.clip(state['tp'] + v * args.speed * gain * TICK, lo, hi)
        tq = state['tq']
        for key, (axis, sign) in ROTATE.items():
            if inp.held(key):
                tq = quat_mul(axis_angle_quat(axis, sign * args.rot_speed * gain * TICK), tq)
        mujoco.mju_normalize4(tq)
        state['tq'] = tq
        for key, sign in GRIP.items():
            if inp.held(key):
                state['grip'] = float(np.clip(state['grip'] + sign * args.grip_speed * TICK, 0, 1))
        q_cmd, ik_p, ik_r = sim.ik(state['tp'], tq, q_init=state['q_cmd'])
        state['q_cmd'] = q_cmd
        sim.set_arm_ctrl(q_cmd)
        sim.set_gripper(state['grip'])
        steps = state['clock'].next_steps()
        sim.step(steps)
        ee_p, ee_q = sim.ee_pose()
        obj_p, obj_q = sim.object_poses()
        state['episode'].add({
            't': float(sim.data.time - state['taken_at']), 'sim_time': float(sim.data.time),
            'physics_steps': steps, 'finger_q': sim.data.qpos[sim.finger_qadr].copy(),
            'finger_dq': sim.data.qvel[sim.finger_dofadr].copy(),
            'q': sim.q, 'dq': sim.dq, 'ctrl': sim.data.ctrl.copy(), 'ee_pos': ee_p, 'ee_quat': ee_q,
            'obj_pos': obj_p, 'obj_quat': obj_q, 'target_pos': state['tp'].copy(),
            'target_quat': np.asarray(tq).copy(), 'gripper': state['grip'],
            'ik_pos_err': ik_p, 'ik_rot_err': ik_r})

    new_scene()
    saved, gain, behind, last = 0, 1., 0., time.time()
    print(__doc__.split('Controls')[1])
    try:
        while not glfw.window_should_close(window):
            glfw.poll_events()
            now = time.time()
            behind = min(behind + now - last, 3 * TICK)  # never try to catch up more than 3 ticks
            last = now
            for key in inp.drain():
                if key == glfw.KEY_Q:
                    glfw.set_window_should_close(window, True)
                elif key == glfw.KEY_H and state['mode'] == 'policy':
                    take_over()
                    print('you have control -- recording; e saves, x discards')
                elif key in (glfw.KEY_E, glfw.KEY_X) and state['mode'] == 'human':
                    episode = state['episode']
                    if key == glfw.KEY_E and len(episode) >= HZ:
                        out = next_episode_dir(args.out)
                        episode.save(out, dict(meta, seed=seed, taken_over_at_s=state['taken_at']), HZ, False)
                        saved += 1
                        print(f'saved correction {saved} -> {out.name}; ACT has control again')
                    else:
                        print('discarded (corrections shorter than 1 s are not saved); ACT has control')
                    hand_back()
                elif key == glfw.KEY_R:
                    seed += 1
                    new_scene()
                    print(f'new scene (seed {seed})')
                elif key == glfw.KEY_LEFT_BRACKET:
                    gain = max(.1, gain / 1.25)
                elif key == glfw.KEY_RIGHT_BRACKET:
                    gain = min(4., gain * 1.25)
                elif key == glfw.KEY_G:
                    focus = ''
                elif key in VIEW_KEYS:
                    want = VIEWS[VIEW_KEYS[key]]
                    focus = '' if want == focus or want not in cams else want

            if state['mode'] == 'policy':
                if behind >= TICK and state['stable'] < HZ // 2:
                    behind -= TICK
                    policy_tick()  # one per frame: inference sets the pace
                    if state['stable'] == HZ // 2:
                        print('ACT completed the stack -- r for a new scene')
            else:
                while behind >= TICK:
                    behind -= TICK
                    human_tick(gain)

            w, h = glfw.get_framebuffer_size(window)
            viewport = mujoco.MjrRect(0, 0, w, h)
            layout = tiles(views, focus, w, h)
            inp.tiles = layout
            for name, rect in layout:
                mujoco.mjv_updateScene(sim.model, sim.data, opt, None, cams[name],
                                       mujoco.mjtCatBit.mjCAT_ALL.value, scene)
                if state['mode'] == 'human' and name != 'wrist':
                    _add_sphere(scene, state['tp'], .018, (1., .5, 0., .85))
                mujoco.mjr_render(rect, scene, ctx)
                if not focus:
                    mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL.value,
                                       mujoco.mjtGridPos.mjGRID_TOPLEFT.value, rect, name, '', ctx)
            if not focus:
                for rect in quadrants(w, h)[len(layout):]:
                    mujoco.mjr_rectangle(rect, 0, 0, 0, 1)
            if state['mode'] == 'policy':
                status = ('ACT done -- r: new scene' if state['stable'] >= HZ // 2
                          else 'ACT driving -- h: take over')
            else:
                status = f"YOU have control  REC {len(state['episode']) / HZ:.1f}s  -- e: save  x: discard"
            status += f'   seed {seed}   saved {saved}   speed x{gain:.2f}'
            mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL.value, mujoco.mjtGridPos.mjGRID_BOTTOMLEFT.value,
                               viewport, status, '', ctx)
            glfw.swap_buffers(window)
    finally:
        for renderer in policy_renderers:
            renderer.close()
        glfw.terminate()
    if saved:
        args.out.mkdir(parents=True, exist_ok=True)
        status_path = args.out / 'status.json'
        prior = json.loads(status_path.read_text()) if status_path.exists() else {}
        status_path.write_text(json.dumps(dict(complete=True, interventions=len(list(args.out.glob('episode_*'))),
                                               mode='human', checkpoint=str(args.checkpoint.resolve()),
                                               sessions=prior.get('sessions', 0) + 1), indent=2) + '\n')
    print(f'{saved} corrections saved to {args.out}')


if __name__ == '__main__':
    main()
