#!/usr/bin/env python3
"""Whole October 30-31 demonstrations in physics, with the insertions closed-loop.

Open-loop, the recorded arm motion does not assemble the parts in the sim
(tools/physics_replay.py): even calibrated (sim/real_calibration_oct30.json) the hand
is a few millimetres from where the real one was, and the gear and pin fit with under
half a millimetre to spare, so they hit the carrier's rim. Here the recording drives
both arms through everything else (picking up the carrier, the gear and the pin, carrying,
setting the finished carrier down) and the insertion environment's controllers
(tools/insertion_env.py scripted and scripted_pin) do the last part of each insertion,
as the operators did by eye:

- the gear: from a moment shortly before the recorded release (the first of GEAR_LEADS
  seconds at which the controller seats it), welded to the hand as in the replay;
- the pin: likewise (PIN_LEADS), held only by the fingers' friction from the moment they
  close on it (PinStageEnv grasp='friction');
- if none of those seats the part, from just before the held part first touches the
  carrier (the recording can slide it along the carrier's face, shoving it along the fingers).

The holds come from the gripper readings (tools/replay_real_in_sim.py phases), with split
readings of one hold joined (HybridReplay.__init__).

After each insertion the recording takes over again from just after its release,
blended over 0.75 s from where the controller left the arms. The right arm holds its
recorded pose while a controller works. Everything is one simulation: the calibrated
arms, the machined carrier (tools/make_real_carrier.py) and the table's friction for the
finger pads on it (as PhysicsReplay compliant=True). A gear the controller cannot seat
is left free.

Each demonstration gives a video (sim views above the real ones; the real views hold
still while a controller works) and a JSON of the handovers and final slot errors.

    python tools/hybrid_physics_replay.py --episodes 5 218 --output outputs/physics_examples
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

import tools.insertion_env as I
import tools.replay_real_in_sim as R
from sim.actuator_env import TABLE_Z
from tools.augment_real_dataset import decode, episode_rows
from tools.physics_replay import FIRM, quat

GEAR_LEADS = (.75, .5, 1., .25, .4, 2.)  # s before the recorded release, in the order tried
PIN_LEADS = (2., 1.5, 1.)
BLEND = 15  # frames over which the recording takes the arms back from a controller
HOLD_GAP = 10  # frames: gripper-reading holds closer than this are one hold
TOUCH_LEAD = 2  # frames before the held part first touches the carrier: the last handover tried
ORDER = ('overhead', 'wrist_port3', 'wrist_port2')  # overhead, left and right wrist (from October 30)
NAMES = {'overhead': 'overhead', 'wrist_port3': 'left wrist', 'wrist_port2': 'right wrist'}
RECORDING = 'operator recording (physics)'


class HybridReplay:
    def __init__(self, root: Path, episode: int, cameras: Path, calibration: Path):
        self.root, self.episode = root, episode
        numeric, self.images = episode_rows(root, episode)
        self.states = numeric['observation.state'].astype(np.float64)
        self.actions = numeric['action'].astype(np.float64)
        segments = R.phases(self.states)  # from the recorded gripper readings, before the joint offsets
        first = lambda side, phase: next(s for s in segments[side] if s['phase'] == phase)
        # The gripper reading splits some holds: a gear regripped in place reads as several gear
        # holds, and a pin often reads in the pin's band only while the fingers close, then fully
        # closed (as on nothing) until its release, before the empty close that presses it home.
        # Pieces less than HOLD_GAP apart are one hold. (A gear-width reading just before a pin
        # hold is the hand approaching the pin half closed, not part of the hold.)
        left = segments['left']

        def extend(i, phases):
            hold = dict(left[i])
            for s in left[i + 1:]:
                if s['start'] - hold['stop'] > HOLD_GAP or s['phase'] not in phases:
                    break
                if s['phase'] == 'closed_empty' and not any(t['phase'] == 'closed_empty' for t in left[left.index(s) + 1:]):
                    break  # the last empty close is the press
                hold['stop'] = s['stop']
            return hold
        self.gear_hold = extend(next(i for i, s in enumerate(left) if s['phase'] == 'gear_held'), ('gear_held',))
        self.pin_hold = extend(next(i for i, s in enumerate(left) if s['phase'] == 'pin_held'), ('pin_held', 'closed_empty'))
        kw = dict(noise_mm=0, noise_deg=0, cameras=cameras, calibration=calibration, carrier='real')
        self.gear_env = G = I.GearInsertionEnv(root, [episode], **kw)
        self.pin_env = P = I.PinStageEnv(root, [episode], **kw)
        # One simulation for both stages: the pin stage drives the gear stage's model, with
        # the pin's stiff contacts (PinStageEnv.__init__) copied over.
        P.replay, P.m, P.d, P.ik_data = G.replay, G.m, G.d, G.ik_data
        self.r, self.m, self.d = r, m, d = G.replay, G.m, G.d
        pin = G.part_geoms['pin_2']
        m.geom_solref[pin], m.geom_solimp[pin], m.geom_solmix[pin] = [.003, 1], [.95, .99, .0005, .5, 2], 1e3
        table = m.geom('table_top').id
        m.geom_priority[table], m.geom_friction[table, 0] = 5, .4
        self.grips = {'large_carrier': G.base_carrier_grip, 'gear_3': G.part_grip, 'pin_2': P.part_grip}
        self.holds = {'large_carrier': ('right', first('right', 'carrier_held')), 'gear_3': ('left', self.gear_hold),
                      'pin_2': ('left', self.pin_hold)}
        self.frames = []  # (qpos, recording frame shown beside it, caption)
        self.flags = dict(gear_done=False, pin_done=False, pin_steady=None, gear_seat=False)

    # -------------------------------------------------------------- the recording --
    def start(self):
        r, m, d = self.r, self.m, self.d
        mujoco.mj_resetDataKeyframe(m, d, 0)
        d.eq_active[list(r.weld.values())] = 0  # the part holds only: the fingers are coupled by equalities too
        m.eq_solref[list(r.weld.values())] = FIRM
        for name, (side, seg) in self.holds.items():  # rest poses, as PhysicsReplay.run
            r.set_arms(self.states[seg['start']] + r.offsets)
            mujoco.mj_kinematics(m, d)
            held = r.grip(side) @ self.grips[name]
            yaw = Rotation.from_matrix(held[:3, :3]).as_euler('zyx')[0]
            r.place(name, R.transform([held[0, 3], held[1, 3], TABLE_Z + .0003], Rotation.from_euler('z', yaw).as_matrix()))
        self.gear_env.set_contact('gear_3', held=True)
        self.gear_env.set_contact('pin_2', held=False)
        r.set_arms(self.states[0] + r.offsets)
        mujoco.mj_forward(m, d)
        for _ in range(200):  # let the parts settle before the arms move
            mujoco.mj_step(m, d)

    def play(self, a, b, blend_from=None):
        """Recorded frames a..b-1. The carrier and gear are welded to their hands while held;
        the pin is steadied only while the fingers close on it, then held by friction."""
        r, d, flags = self.r, self.d, self.flags
        for i in range(a, b):
            ctrl = self.actions[i] + r.offsets
            if blend_from is not None and i - a < BLEND:
                w = (i - a + 1) / BLEND
                ctrl = (1 - w) * blend_from + w * ctrl
            r.set_arms(ctrl, ctrl_only=True)
            for name, (side, seg) in self.holds.items():
                if name == 'gear_3' and flags['gear_done'] or name == 'pin_2' and flags['pin_done']:
                    continue
                if name == 'pin_2':
                    if i == seg['start']:
                        r.place(name, r.grip(side) @ self.grips[name])
                        d.eq_active[r.weld[name]], flags['pin_steady'] = 1, i
                    if flags['pin_steady'] is not None and i == flags['pin_steady'] + 6:
                        d.eq_active[r.weld[name]] = 0
                    continue
                d.eq_active[r.weld[name]] = seg['start'] <= i < seg['stop']
            for _ in range(r.steps):
                for name, (side, _) in self.holds.items():
                    following = not (name == 'gear_3' and flags['gear_done'] or name == 'pin_2' and flags['pin_done'])
                    if d.eq_active[r.weld[name]] and following:
                        t = r.grip(side) @ self.grips[name]
                        d.mocap_pos[r.mocap[name]], d.mocap_quat[r.mocap[name]] = t[:3, 3], quat(t[:3, :3])
                if flags['gear_seat']:
                    self.pin_env.each_step()  # the seated gear rides in the carrier
                if flags['pin_done'] and self.pin_env.fit_mm is not None:
                    self.pin_env.press_fit()
                mujoco.mj_step(self.m, d)
            self.frames.append((d.qpos.copy(), i, RECORDING))

    # ------------------------------------------------------------- the controllers --
    def stage(self, env, controller, seg, leads, start, label):
        """Play the recording from `start` to each candidate handover, then let the controller
        insert from the first (in `leads` order) at which it seats the part."""
        m, d = self.m, self.d
        handovers = {lead: seg['stop'] - 1 - round(lead * 20) for lead in leads}
        handovers = {lead: h for lead, h in handovers.items() if h > seg['start'] + 6 and h >= start}
        if not handovers:  # a hold shorter than every lead: hand over once the fingers have closed
            h = max(seg['start'] + 7, start)
            handovers = {round((seg['stop'] - 1 - h) / 20, 2): h}
            leads = list(handovers)
        spec = mujoco.mjtState.mjSTATE_INTEGRATION

        def snapshot():
            state = np.empty(mujoco.mj_stateSize(m, spec))
            mujoco.mj_getState(m, d, state, spec)
            return state, len(self.frames), dict(self.flags)

        def attempt(lead, h, saved):
            state, count, flags = saved
            mujoco.mj_setState(m, d, state, spec)
            mujoco.mj_forward(m, d)
            del self.frames[count:]
            self.flags = dict(flags)
            if env is self.gear_env:
                m.eq_solref[self.r.weld['gear_3']] = FIRM
                env.set_contact('gear_3', held=True)
            env.carrier_grip = self.grips['large_carrier']
            env.state = self.actions[h] + self.r.offsets
            env.q = env.state[:6].copy()
            env.released, env.t, env.closed, env.fit_mm = False, 0, False, None
            step = env.step

            def recorded(action):
                out = step(action)
                self.frames.append((d.qpos.copy(), h, label))
                return out
            env.step = recorded
            seated, final = controller(env)
            env.step = step
            return dict(lead_s=lead, seated=bool(seated), final_mm=np.round(final, 2).tolist())

        first, saved, at = snapshot(), {}, start
        for lead, h in sorted(handovers.items(), key=lambda kv: kv[1]):
            self.play(at, h)
            at = h
            saved[lead] = snapshot()
        result = None
        for lead in [lead for lead in leads if lead in handovers]:
            result = attempt(lead, handovers[lead], saved[lead])
            if result['seated']:
                return result
        # The recording can push the held part into the carrier before every handover (the
        # operator sliding it along the face), shoving it along the fingers. Last try: hand
        # over just before the part first touches the carrier.
        mujoco.mj_setState(m, d, first[0], spec)
        mujoco.mj_forward(m, d)
        del self.frames[first[1]:]
        self.flags = dict(first[2])
        part = np.array(env.part_geoms[env.PART])
        carrier = self.r.body['large_carrier']
        recent = []
        for i in range(start, seg['stop']):
            recent = (recent + [(i, snapshot())])[-TOUCH_LEAD - 1:]
            self.play(i, i + 1)
            g1, g2 = d.contact.geom1[:d.ncon], d.contact.geom2[:d.ncon]
            touching = (np.isin(g1, part) & (m.geom_bodyid[g2] == carrier)) | (np.isin(g2, part) & (m.geom_bodyid[g1] == carrier))
            if i >= seg['start'] + 7 and touching.any():
                h, before = recent[0]
                if h > seg['start'] + 6 and h not in handovers.values():
                    result = dict(attempt(round((seg['stop'] - 1 - h) / 20, 2), h, before), touch=True)
                    return result
                break
        # No better handover: the video ends with the last try's state, as before.
        if result is not None:
            lead = [lead for lead in leads if lead in handovers][-1]
            result = attempt(lead, handovers[lead], saved[lead])
        return result

    def run(self):
        G, P, r, m, d = self.gear_env, self.pin_env, self.r, self.m, self.d
        self.start()
        report = dict(gear=self.stage(G, I.scripted, self.gear_hold, GEAR_LEADS, 0,
                                      'controller inserts the gear (physics)'))
        self.flags['gear_done'] = True
        after = np.r_[G.q, G.state[6:]]
        after[6] = 1.2  # the gripper as the controller left it: open
        # A seated gear rides in the carrier from here (the pin stage's soft hold at its
        # seat); one the controller could not seat is left free.
        seated = bool(report['gear'] and report['gear']['seated'])
        m.eq_solref[r.weld['gear_3']] = [.02, 1]
        d.eq_active[r.weld['gear_3']] = seated
        self.flags['gear_seat'] = seated
        G.set_contact('gear_3', held=False)
        resume = self.gear_hold['stop'] + 20
        first_pin = min(self.pin_hold['stop'] - 1 - round(lead * 20) for lead in PIN_LEADS)
        until = max(resume, min(first_pin, self.pin_hold['start'] + 7))
        self.play(resume, until, blend_from=after)
        report['pin'] = self.stage(P, I.scripted_pin, self.pin_hold, PIN_LEADS, until,
                                   'controller inserts the pin, held by finger friction (physics)')
        self.flags['pin_done'] = True
        self.play(self.pin_hold['stop'] + 20, len(self.actions), blend_from=np.r_[P.q, P.state[6:]])
        return report

    # ------------------------------------------------------------------ the video --
    def write(self, path: Path, cameras: Path, calibration: Path):
        scene = R.Scene(R.camera_setup(cameras), 'real')
        views = R.RealViews(scene, cameras)
        views.place_cameras()
        scene.set_right_base(*json.loads(calibration.read_text())['right_base'])
        real = {c: decode(self.images[c]) for c in ORDER}
        label = lambda im, text: cv2.putText(im, text, (4, 250), 0, .4, (255, 255, 255), 1, cv2.LINE_AA)
        video = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), 20, (256 * 3, 256 * 2 + 22))
        for qpos, i, caption in self.frames:
            scene.data.qpos[:] = qpos
            mujoco.mj_kinematics(scene.model, scene.data)
            mujoco.mj_camlight(scene.model, scene.data)
            sim, rec = [], []
            for c in ORDER:
                renderer = views.overhead if c == 'overhead' else views.wrist
                renderer.update_scene(scene.data, scene.cam[c], views.options)
                map_x, map_y, inside = views.maps[c]
                image = cv2.remap(renderer.render(), map_x, map_y, cv2.INTER_LINEAR, borderValue=0)
                if inside is not None:
                    image[~inside] = 0
                label(image, 'sim ' + NAMES[c])
                sim.append(image)
                image = real[c][i].copy()
                label(image, 'real ' + NAMES[c])
                rec.append(image)
            bar = np.zeros((22, 256 * 3, 3), np.uint8)
            cv2.putText(bar, f'demo {self.episode} | {caption}', (6, 15), 0, .42, (255, 255, 255), 1, cv2.LINE_AA)
            frame = np.concatenate([bar, np.concatenate(sim, 1), np.concatenate(rec, 1)], 0)
            video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        video.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--root', type=Path, default=ROOT / 'outputs/zeb_real_oct31', help='local copy of the real dataset')
    parser.add_argument('--episodes', type=int, nargs='+', default=[218])
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/physics_hybrid')
    parser.add_argument('--cameras', type=Path, default=ROOT / 'sim/real_cameras_oct30.json')
    parser.add_argument('--calibration', type=Path, default=ROOT / 'sim/real_calibration_oct30.json')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for episode in args.episodes:
        hybrid = HybridReplay(args.root, episode, args.cameras, args.calibration)
        report = hybrid.run()
        hybrid.write(args.output / f'hybrid_demo{episode:03d}.mp4', args.cameras, args.calibration)
        (args.output / f'hybrid_demo{episode:03d}.json').write_text(json.dumps(report, indent=2) + '\n')
        print(f"demo {episode}: gear {report['gear']}, pin {report['pin']}", flush=True)


if __name__ == '__main__':
    main()
