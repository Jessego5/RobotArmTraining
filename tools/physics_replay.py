#!/usr/bin/env python3
"""Drive the CAD sim's physics with real demonstrations' recorded commands.

The question this answers: if the sim arms follow exactly what the real arms
were commanded to do, does the assembly happen in the sim too? It decides
whether the sim can be trusted to check generated demonstrations (MimicGen-style
variations, RL on the insertions).

What the real demonstrations do (seen in the wrist cameras): the carrier lies
flat on the table, face up; the right arm picks it up from above and holds it in
the air for the rest of the demonstration. The gear lies flat and the pin
stands upright on the table; the left arm picks each up from above, inserts it
into the held carrier, releases it, and pushes it home with the closed gripper.
(Zeb's sim instead stands the carrier on a fixture; that fixture is moved out of
the way here.)

Per demonstration:
- the scene is sim/bimanual_cad.py (Zeb's CAD parts and contact-seating
  geometry), with the right base where the overhead camera solve puts it;
- each part starts flat on the table under the point where its gripper later
  closes on it (phases from tools/replay_real_in_sim.py), with the heading of
  its grip;
- while a gripper holds a part, a soft weld ties the part to a handle that
  follows the gripper with the part's grip (tools/fit_grasps.py output when it
  exists, else Zeb's), so the part moves with the hand but contacts still push
  back; gripping by finger friction is not simulated;
- each recorded action (the next joint angles and gripper openings, 20 Hz) is
  sent to the position actuators and the physics runs for 50 ms;
- at the end the gear and pin are compared with their assembled poses relative
  to the carrier (the sim's assembly layout), with the sim's tolerances.

    python tools/physics_replay.py --episodes 12 60 95 --output outputs/physics_replay.json
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
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from sim.actuator_env import TABLE_Z
from sim.bimanual import rad_to_opening
from sim.panthera_env import ARM_JOINTS
from tools.replay_real_in_sim import GRASP, phases, transform

FPS = 20
PARTS = ('large_carrier', 'gear_3', 'pin_2')
HOLDS = {'large_carrier': ('right', 'carrier_held'), 'gear_3': ('left', 'gear_held'), 'pin_2': ('left', 'pin_held')}


def grasps(path: Path):
    out = {name: transform(*g) for name, g in GRASP.items()}
    if path.exists():
        for name, g in json.loads(path.read_text()).items():
            out[name] = transform(g['offset'], g['rotation'])
    return out


def pose_of(m, d, body):
    return transform(d.xpos[body], d.xmat[body].reshape(3, 3))


def quat(rotation):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(rotation, float).flatten())
    return q


class PhysicsReplay:
    def __init__(self, cameras: Path, exclude_left=True):
        """exclude_left: the gear and pin never collide with the left gripper (replay); the
        insertion environment instead switches that on at release, to push with the fingertips."""
        from sim.bimanual_cad import build_scene
        spec = mujoco.MjSpec.from_file(str(build_scene('original')))
        for name in PARTS:  # a mocap handle per part and a weld to it, off until the part is held
            spec.worldbody.add_body(name=f'hold_{name}', mocap=True)
            spec.add_equality(type=mujoco.mjtEq.mjEQ_WELD, name1=name, name2=f'hold_{name}',
                              objtype=mujoco.mjtObj.mjOBJ_BODY, active=False,
                              data=[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1], solref=[.004, 1])  # a firm grip
        # A held part does not collide with the gripper holding it: the real fin-ray fingers
        # deform around it, and rigid sim fingers overlapping it would jam it.
        pairs = (('large_carrier', 'right_'), ('gear_3', ''), ('pin_2', '')) if exclude_left else \
            (('large_carrier', 'right_'),)
        for part, prefix in pairs:
            for link in ('link6', 'L_finger', 'R_finger'):
                spec.add_exclude(bodyname1=part, bodyname2=prefix + link)
        self.model = m = spec.compile()
        self.data = d = mujoco.MjData(m)
        # The right base where the clicked-fingertip overhead solve puts it.
        dx, dy, dyaw = list(json.loads(cameras.read_text())['overhead']['params'].values())[10:13]
        base = m.body('right_base').id
        m.body_pos[base] += [dx, dy, 0]
        w, x, y, z = m.body_quat[base]
        q = (Rotation.from_euler('z', dyaw, degrees=True) * Rotation.from_quat([x, y, z, w])).as_quat()
        m.body_quat[base] = [q[3], q[0], q[1], q[2]]
        # Zeb's fixture is not part of the real task.
        m.body_pos[m.body('small_carrier').id] = [-1., 0., TABLE_Z]
        m.geom_pos[m.geom('fixture').id, :2] = [-1., 0.]
        qadr = lambda name: m.jnt_qposadr[m.joint(name).id]
        sides = (('left', ''), ('right', 'right_'))
        self.joints = {s: [qadr(p + j) for j in ARM_JOINTS] for s, p in sides}
        self.fingers = {s: (qadr(p + 'L_finger_joint'), qadr(p + 'R_finger_joint')) for s, p in sides}
        self.arm_act = {s: [m.actuator(p + j).id for j in ARM_JOINTS] for s, p in sides}
        self.grip_act = {s: m.actuator(p + 'gripper').id for s, p in sides}
        self.sites = {'left': m.site('grip_site').id, 'right': m.site('right_grip_site').id}
        self.body = {n: m.body(n).id for n in PARTS}
        self.mocap = {n: m.body_mocapid[m.body(f'hold_{n}').id] for n in PARTS}
        self.weld = {n: next(e for e in range(m.neq) if m.eq_obj1id[e] == self.body[n]
                             and m.eq_type[e] == mujoco.mjtEq.mjEQ_WELD) for n in PARTS}
        self.free = {n: (m.jnt_qposadr[m.body_jntadr[self.body[n]]], m.jnt_dofadr[m.body_jntadr[self.body[n]]])
                     for n in PARTS}
        self.steps = round(1 / FPS / m.opt.timestep)

    def set_arms(self, state, ctrl_only=False):
        d = self.data
        for side, off in (('left', 0), ('right', 7)):
            opening = float(rad_to_opening(state[off + 6]))
            d.ctrl[self.arm_act[side]] = state[off:off + 6]
            d.ctrl[self.grip_act[side]] = opening
            if not ctrl_only:
                d.qpos[self.joints[side]] = state[off:off + 6]
                left, right = self.fingers[side]
                d.qpos[left], d.qpos[right] = opening, -opening

    def grip(self, side):
        i = self.sites[side]
        return transform(self.data.site_xpos[i], self.data.site_xmat[i].reshape(3, 3))

    def place(self, name, pose):
        a, v = self.free[name]
        self.data.qpos[a:a + 3], self.data.qpos[a + 3:a + 7] = pose[:3, 3], quat(pose[:3, :3])
        self.data.qvel[v:v + 6] = 0

    def run(self, states, actions, grips, targets, record=None):
        """record: a list that receives qpos after every frame (for rendering)."""
        m, d = self.model, self.data
        mujoco.mj_resetDataKeyframe(m, d, 0)
        d.eq_active[:] = 0
        segments = phases(states)
        holds = {}
        for name, (side, phase) in HOLDS.items():
            g = next((s for s in segments[side] if s['phase'] == phase), None)
            if g is None:
                return dict(skipped=f'no {phase} phase')
            holds[name] = (side, g['start'], g['stop'] if name != 'large_carrier' else len(actions) + 1)
        # Rest poses: flat on the table under the grasp point, with the grip's heading.
        for name, (side, start, _) in holds.items():
            self.set_arms(states[start])
            mujoco.mj_kinematics(m, d)
            held = self.grip(side) @ grips[name]
            yaw = Rotation.from_matrix(held[:3, :3]).as_euler('zyx')[0]
            rest = transform([held[0, 3], held[1, 3], TABLE_Z + .0003], Rotation.from_euler('z', yaw).as_matrix())
            self.place(name, rest)
        self.set_arms(states[0])
        mujoco.mj_forward(m, d)
        for _ in range(200):  # let the parts settle before the arms move
            mujoco.mj_step(m, d)
        events = []
        for i, action in enumerate(actions):
            self.set_arms(action, ctrl_only=True)
            for _ in range(self.steps):
                for name, (side, start, stop) in holds.items():
                    on = start <= i < stop
                    if on:
                        target = self.grip(side) @ grips[name]
                        d.mocap_pos[self.mocap[name]] = target[:3, 3]
                        d.mocap_quat[self.mocap[name]] = quat(target[:3, :3])
                    if bool(d.eq_active[self.weld[name]]) != on:
                        d.eq_active[self.weld[name]] = on
                        events.append(f"{i / FPS:.1f}s {'grab' if on else 'release'} {name}")
                mujoco.mj_step(m, d)
            if record is not None:
                record.append(d.qpos.copy())
        carrier = pose_of(m, d, self.body['large_carrier'])
        result = dict(events=events)
        for name in ('gear_3', 'pin_2'):
            rel = np.linalg.inv(carrier) @ pose_of(m, d, self.body[name])
            err = rel[:3, 3] - targets[name]
            tilt = float(np.degrees(np.arccos(np.clip(rel[2, 2], -1, 1))))
            result[name] = dict(lateral_mm=float(np.linalg.norm(err[:2]) * 1000), axial_mm=float(err[2] * 1000),
                                tilt_deg=tilt)
        g, p = result['gear_3'], result['pin_2']
        result['gear_seated'] = bool(g['lateral_mm'] < 1.5 and -3 < g['axial_mm'] < 1 and g['tilt_deg'] < 4)
        result['pin_seated'] = bool(p['lateral_mm'] < 1. and -3 < p['axial_mm'] < 3 and p['tilt_deg'] < 4)
        result['accepted'] = result['gear_seated'] and result['pin_seated']
        return result


def assembled_offsets():
    """Gear and pin origins in the carrier's frame when assembled (the CAD manifest)."""
    from sim.bimanual_cad import ParallelCADSim
    parts = ParallelCADSim(seed=0, overhead_setup='original').manifest['parts']
    base = np.array(parts['large_carrier']['assembled_origin'])
    return {n: np.array(parts[n]['assembled_origin']) - base for n in ('gear_3', 'pin_2')}


def main():
    from tools.augment_real_dataset import episode_rows
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--root', type=Path, default=ROOT / 'outputs/real_gear')
    parser.add_argument('--episodes', type=int, nargs='+', default=[12, 60, 95])
    parser.add_argument('--grasps', type=Path, default=ROOT / 'outputs/grasp_fit.json')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/physics_replay.json')
    args = parser.parse_args()
    replay = PhysicsReplay(ROOT / 'sim/real_cameras.json')
    grips, targets = grasps(args.grasps), assembled_offsets()
    results = {}
    for episode in args.episodes:
        numeric, _ = episode_rows(args.root, episode)
        started = time.time()
        r = replay.run(numeric['observation.state'].astype(np.float64), numeric['action'].astype(np.float64),
                       grips, targets)
        results[episode] = r
        if 'skipped' in r:
            print(f'demo {episode}: skipped ({r["skipped"]})', flush=True)
            continue
        g, p = r['gear_3'], r['pin_2']
        print(f"demo {episode}: gear {'seated' if r['gear_seated'] else 'not seated'} "
              f"({g['lateral_mm']:.1f} mm off, {g['axial_mm']:+.1f} mm axial, tilt {g['tilt_deg']:.0f} deg); "
              f"pin {'seated' if r['pin_seated'] else 'not seated'} ({p['lateral_mm']:.1f} mm off, "
              f"{p['axial_mm']:+.1f} mm axial, tilt {p['tilt_deg']:.0f} deg)  [{time.time() - started:.0f} s]", flush=True)
    args.output.write_text(json.dumps(results, indent=2) + '\n')
    done = [r for r in results.values() if 'skipped' not in r]
    print(f"{sum(r['accepted'] for r in done)} of {len(done)} replayed demos complete the assembly in the sim; "
          f"gear seated in {sum(r['gear_seated'] for r in done)}, pin in {sum(r['pin_seated'] for r in done)}")


if __name__ == '__main__':
    main()
