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
- each held part lets go where its real gripper opened (from October 30 the
  right arm also sets the finished carrier down);
- the gear and pin are compared with their assembled poses relative to the
  carrier (the sim's assembly layout), with the sim's tolerances: as each is
  released, and finally as the carrier is released (or at the end).

--compliant stands in for the operator's feel: the real arms were hand-guided and
the parts eased in by touch, which a replay of recorded joint angles cannot do.
A held part's weld goes soft once it is near its slot, so contact with the carrier
steers it; the right gripper's fingers yield (soften_right_fingers); at release the left fingers start touching the part, so the recorded
push afterwards can drive it home; and a press fit (as in tools/insertion_env.py)
holds a released part that sits in its slot's mouth and lets it move only inward:
the gear edgewise through the rim, the pin down through the face.

    python tools/physics_replay.py --episodes 12 60 95 --output outputs/physics_replay.json
    python tools/physics_replay.py --root outputs/zeb_real_oct31 --cameras sim/real_cameras_oct30.json \
        --grasps outputs/grasp_fit_oct30.json --episodes 5 71 160 --output outputs/physics_replay_oct30.json
    python tools/physics_replay.py --root outputs/zeb_real_oct31 --cameras sim/real_cameras_oct30.json \
        --grasps sim/real_grips_oct30.json --compliant --episodes 5 71 160
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
FIRM, SOFT = [.004, 1], [.02, 1]  # weld solref: a firm grip; soft enough for contacts to steer the part
NEAR = .02  # m from its slot within which a held part's weld goes soft (--compliant)


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
    # --compliant collision bits. The scene's: 1 arms, 2 part pieces (affinity 3), 4 whole-part
    # hulls that only the table touches, 8 the table. The left gripper gets a type of its own
    # and still touches the arms, the parts and the table; a held part gets one that the
    # carrier and the other part touch but the left gripper does not.
    LEFT_FINGERS, LEFT_AFFINITY, HELD = 16, 1 | 2 | 8, 32

    def __init__(self, cameras: Path, exclude_left=True, compliant=False):
        """exclude_left: the gear and pin never collide with the left gripper (replay); the
        insertion environment instead switches that on at release, to push with the fingertips.
        compliant: soft holds near the slots, fingertip pushes after release and press fits
        (see the module docstring); implies exclude_left=False."""
        self.compliant = compliant
        exclude_left = exclude_left and not compliant
        from sim.bimanual_cad import build_scene
        spec = mujoco.MjSpec.from_file(str(build_scene('original')))
        for name in PARTS:  # a mocap handle per part and a weld to it, off until the part is held
            spec.worldbody.add_body(name=f'hold_{name}', mocap=True)
            spec.add_equality(type=mujoco.mjtEq.mjEQ_WELD, name1=name, name2=f'hold_{name}',
                              objtype=mujoco.mjtObj.mjOBJ_BODY, active=False,
                              data=[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1], solref=FIRM)
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
                             and m.eq_obj2id[e] == m.body(f'hold_{n}').id) for n in PARTS}
        self.free = {n: (m.jnt_qposadr[m.body_jntadr[self.body[n]]], m.jnt_dofadr[m.body_jntadr[self.body[n]]])
                     for n in PARTS}
        self.steps = round(1 / FPS / m.opt.timestep)
        if compliant:
            # A held gear or pin passes through the left fingers but hits the carrier; released,
            # the closed fingertips can push it (the bits above).
            self.set_left_fingers()
            self.part_geoms = {n: self.pieces(n) for n in ('gear_3', 'pin_2')}
            self.soften_right_fingers()

    def set_left_fingers(self):
        m = self.model
        left = {m.body(b).id for b in ('link6', 'L_finger', 'R_finger')}
        geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] in left and m.geom_contype[g]]
        m.geom_contype[geoms], m.geom_conaffinity[geoms] = self.LEFT_FINGERS, self.LEFT_AFFINITY

    def pieces(self, name):
        """A part's collision pieces (type 2, affinity 3), without its `_table_contact` hull: that is
        the convex hull of the whole part (contype 4, affinity 8, the table only), a solid disc
        over the gear's bore."""
        m = self.model
        return [g for g in range(m.ngeom) if m.geom_bodyid[g] == self.body[name] and m.geom_contype[g] == 2]

    def soften_right_fingers(self):
        """The right gripper's fin-ray fingers bend around what they hold: in the real wrist
        videos the gear slides into the carrier's rim right beside one. Its rigid sim fingers
        would block it, so they touch only the table (and the carrier never, as before)."""
        m = self.model
        right = {m.body(b).id for b in ('right_link6', 'right_L_finger', 'right_R_finger')}
        for g in range(m.ngeom):
            if m.geom_bodyid[g] in right and m.geom_contype[g]:
                m.geom_contype[g], m.geom_conaffinity[g] = 0, 8  # the table is type 8

    def set_held(self, name, held):
        """--compliant: a held part passes through the left fingers; a released one does not."""
        self.model.geom_contype[self.part_geoms[name]] = self.HELD if held else 2

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

    def slot_error(self, name, targets, carrier=None):
        """A part's offset from its assembled place in the carrier (now, or in a given carrier pose)."""
        m, d = self.model, self.data
        carrier = pose_of(m, d, self.body['large_carrier']) if carrier is None else carrier
        return _slot_error(carrier, pose_of(m, d, self.body[name]), targets[name])

    def press_fit(self, name, target, entry, fit):
        """Once a released part sits in its slot's mouth (lined up to 1 mm and 5 deg, at most
        8 mm out along `entry`) a soft weld holds it there, keeping its spin, and lets it move
        only inward: each push that takes it deeper becomes the new hold point. fit maps the
        part to its hold depth (mm), None until caught."""
        m, d = self.model, self.data
        slot = np.eye(4)
        slot[:3, 3] = target
        frame = pose_of(m, d, self.body['large_carrier']) @ slot
        rel = np.linalg.inv(frame) @ pose_of(m, d, self.body[name])
        pos = rel[:3, 3] * 1000
        depth = pos @ entry
        if fit[name] is None:
            tilt = np.degrees(np.arccos(np.clip(rel[2, 2], -1, 1)))
            if depth > 8 or np.linalg.norm(pos - depth * entry) > 1 or tilt > 5:
                return
            fit[name] = depth
            m.eq_solref[self.weld[name]] = SOFT
            d.eq_active[self.weld[name]] = 1
        fit[name] = max(0., min(fit[name], depth))
        hold = np.eye(4)
        hold[:3, :3] = Rotation.from_euler('z', np.arctan2(rel[1, 0], rel[0, 0])).as_matrix()
        hold[:3, 3] = fit[name] / 1000 * entry
        pose = frame @ hold
        d.mocap_pos[self.mocap[name]], d.mocap_quat[self.mocap[name]] = pose[:3, 3], quat(pose[:3, :3])

    def run(self, states, actions, grips, targets, record=None):
        """record: a list that receives qpos after every frame (for rendering)."""
        m, d = self.model, self.data
        mujoco.mj_resetDataKeyframe(m, d, 0)
        d.eq_active[list(self.weld.values())] = 0  # the part holds only: the fingers are coupled by equalities too
        m.eq_solref[list(self.weld.values())] = FIRM
        # Which way each part goes in: the gear edgewise through the rim, the pin down the face.
        outward = np.r_[targets['gear_3'][:2], 0.] / np.linalg.norm(targets['gear_3'][:2])
        entry = {'gear_3': outward, 'pin_2': np.array([0., 0., 1.])}
        fit = {}
        if self.compliant:
            for name in self.part_geoms:
                self.set_held(name, True)
        segments = phases(states)
        holds = {}
        for name, (side, phase) in HOLDS.items():
            g = next((s for s in segments[side] if s['phase'] == phase), None)
            if g is None:
                return dict(skipped=f'no {phase} phase')
            holds[name] = (side, g['start'], g['stop'])
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
        events, at_release, carrier = [], {}, None
        holding = {name: False for name in holds}
        for i, action in enumerate(actions):
            self.set_arms(action, ctrl_only=True)
            for _ in range(self.steps):
                for name, (side, start, stop) in holds.items():
                    on = start <= i < stop
                    if on:
                        target = self.grip(side) @ grips[name]
                        d.mocap_pos[self.mocap[name]] = target[:3, 3]
                        d.mocap_quat[self.mocap[name]] = quat(target[:3, :3])
                        if self.compliant and name in targets:
                            rel = np.linalg.inv(pose_of(m, d, self.body['large_carrier'])) @ target
                            near = np.linalg.norm(rel[:3, 3] - targets[name]) < NEAR
                            m.eq_solref[self.weld[name]] = SOFT if near else FIRM
                    if holding[name] != on:
                        if not on and name == 'large_carrier':
                            carrier = pose_of(m, d, self.body[name])  # scored as the hand lets go
                        elif not on:
                            at_release[name] = self.slot_error(name, targets)
                            if self.compliant:
                                self.set_held(name, False)
                                fit[name] = None
                        holding[name] = on
                        d.eq_active[self.weld[name]] = on
                        events.append(f"{i / FPS:.1f}s {'grab' if on else 'release'} {name}")
                    elif name in fit:
                        self.press_fit(name, targets[name], entry[name], fit)
                mujoco.mj_step(m, d)
            if record is not None:
                record.append(d.qpos.copy())
        if carrier is None:  # still held at the end
            carrier = pose_of(m, d, self.body['large_carrier'])
        result = dict(events=events, at_release=at_release, press_fit={n: f is not None for n, f in fit.items()})
        for name in ('gear_3', 'pin_2'):
            result[name] = self.slot_error(name, targets, carrier)
        g, p = result['gear_3'], result['pin_2']
        result['gear_seated'] = bool(g['lateral_mm'] < 1.5 and -3 < g['axial_mm'] < 1 and g['tilt_deg'] < 4)
        result['pin_seated'] = bool(p['lateral_mm'] < 1. and -3 < p['axial_mm'] < 3 and p['tilt_deg'] < 4)
        result['accepted'] = result['gear_seated'] and result['pin_seated']
        return result


def _slot_error(carrier, part, target):
    rel = np.linalg.inv(carrier) @ part
    err = rel[:3, 3] - target
    return dict(lateral_mm=float(np.linalg.norm(err[:2]) * 1000), axial_mm=float(err[2] * 1000),
                tilt_deg=float(np.degrees(np.arccos(np.clip(rel[2, 2], -1, 1)))))


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
    parser.add_argument('--cameras', type=Path, default=ROOT / 'sim/real_cameras.json',
                        help='fitted cameras (their overhead solve places the right base)')
    parser.add_argument('--compliant', action='store_true',
                        help='soft holds near the slots, fingertip pushes and press fits (module docstring)')
    args = parser.parse_args()
    replay = PhysicsReplay(args.cameras, compliant=args.compliant)
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
        for name, e in r['at_release'].items():
            print(f"  at its release, {name} was {e['lateral_mm']:.1f} mm off its place, {e['axial_mm']:+.1f} mm axial, "
                  f"tilt {e['tilt_deg']:.0f} deg", flush=True)
    args.output.write_text(json.dumps(results, indent=2) + '\n')
    done = [r for r in results.values() if 'skipped' not in r]
    print(f"{sum(r['accepted'] for r in done)} of {len(done)} replayed demos complete the assembly in the sim; "
          f"gear seated in {sum(r['gear_seated'] for r in done)}, pin in {sum(r['pin_seated'] for r in done)}")
    for name in ('gear_3', 'pin_2'):
        errors = [r['at_release'][name]['lateral_mm'] for r in done if name in r['at_release']]
        if errors:
            print(f"{name} at release: median {np.median(errors):.1f} mm off its place (of {len(errors)})")


if __name__ == '__main__':
    main()
