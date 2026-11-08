#!/usr/bin/env python3
"""Gear insertion from real pre-insertion poses, in the CAD sim: the RL starting point.

Replaying real demonstrations through the sim's physics (tools/physics_replay.py)
showed the hands' relative pose at insertion is repeatable to only about 4-5 mm
and 10 degrees, against a 1.5 mm seating tolerance: the real operators corrected
by feel and by eye. This environment isolates that last step so a policy can
learn the correction.

Reset: a real demonstration's arm poses two seconds before it released the gear
(phases from tools/replay_real_in_sim.py).

The gear seats as in the real wrist videos: slid edgewise into a slot in the
carrier's rim, released half in, pushed home with the closed fingertips. The
real slot holds a half-inserted gear by friction; the sim's is looser, so a
modelled press fit holds the released gear in the slot's mouth and lets it move
only inward. The right arm holds the carrier at the
grip measured from the gear releases of 68 demonstrations, plus random error
(default 4 mm, 6 degrees, the scatter measured between each demonstration's
gear and pin releases); the left arm holds the gear (Zeb's top-down grip). Holds
are soft welds to handles that follow the grippers, so contacts push back. The right
gripper's fingers yield (PhysicsReplay.soften_right_fingers): the gear goes in beside one.

The gear seats sideways into a slot in the carrier's rim (as in Zeb's controller),
not from above.

--part pin runs the second half (PinStageEnv): the gear seated, the pin in the left
hand from a demonstration's pose before the pin's release, pushed down through the
carrier's upper plate and the gear's bore (0.1 mm clearance, no chamfer), released
part way in and pressed home with the closed fingertips (scripted_pin).

Step: a small move of the left gripper (up to 2 mm and 2 degrees, in the
carrier's frame) through damped least-squares IK, and a release flag; 50 ms of
physics. After release the gear is free and the episode runs a short settle.

Observation (privileged, for now): left arm joints, gear position relative to its
slot in the carrier (mm), its tilt from the slot axis (spin about its own axis
does not matter), whether it is released.
Reward: minus the slot error, plus 10 when seated (the sim's tolerances).

    python tools/insertion_env.py --episodes 20     # scripted baseline success rate
    python tools/insertion_env.py --root outputs/zeb_real_oct31 --cameras sim/real_cameras_oct30.json \
        --grasps sim/real_grips_oct30.json --demo-ids 5 18 71    # from the October 30-31 demos
    python tools/insertion_env.py --part pin ...                  # the pin, the gear already seated
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from sim.actuator_env import TABLE_Z
from tools.physics_replay import PhysicsReplay, assembled_offsets, grasps, pose_of, quat
from tools.replay_real_in_sim import phases, transform

# The carrier in the right grip_site frame, measured from 68 demonstrations' gear
# releases (gear in its slot at release): mean position and mean rotation.
MOUTH_MM = 17.  # the pin's tip is in the carrier's upper plate this far above its seat
FLOOR = .012  # m above the table that fingertips and a held gear's centre stay above
CARRIER_GRIP = transform([-.0279, -.0054, .0214], [[-.88, .47, -.11], [-.45, -.72, .52], [.17, .51, .85]])


class GearInsertionEnv:
    PART, HOLD_PHASE = 'gear_3', 'gear_held'  # the part the left hand inserts, and its hold in the demos
    CATCH_MM = 8.  # how far out of its slot the press fit catches a released part

    def __init__(self, root=ROOT / 'outputs/real_gear', episodes=None, noise_mm=4., noise_deg=6., seed=0,
                 lead_s=2., cameras=ROOT / 'sim/real_cameras.json', grips=None):
        """grips: a grasp file (tools/physics_replay.py format) for the carrier and gear; by
        default CARRIER_GRIP and Zeb's gear grip."""
        from tools.augment_real_dataset import episode_rows
        self.replay = r = PhysicsReplay(Path(cameras), exclude_left=False)
        self.m, self.d = r.model, r.data
        self.rng = np.random.default_rng(seed)
        self.noise_mm, self.noise_deg = noise_mm, noise_deg
        held = grasps(Path(grips) if grips else ROOT / 'none')
        self.part_grip = held[self.PART]
        self.base_carrier_grip = held['large_carrier'] if grips else CARRIER_GRIP
        self.slot = np.eye(4)
        self.slot[:3, 3] = assembled_offsets()[self.PART]
        info = json.loads((Path(root) / 'meta/info.json').read_text())
        self.starts = []
        for e in (episodes if episodes is not None else range(info['total_episodes'])):
            numeric, _ = episode_rows(Path(root), e)
            st = numeric['observation.state'].astype(np.float64)
            g = next((s for s in phases(st)['left'] if s['phase'] == self.HOLD_PHASE), None)
            if g is not None and g['stop'] - 1 - round(lead_s * 20) > g['start']:
                self.starts.append(st[g['stop'] - 1 - round(lead_s * 20)])
        m = self.m
        self.left_dofs = np.array([m.jnt_dofadr[m.joint(j).id] for j in
                                   ('joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6')])
        self.left_q = np.array(r.joints['left'])
        self.range = np.array([m.jnt_range[m.joint(j).id] for j in
                               ('joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6')])
        self.ik_data = mujoco.MjData(m)
        self.outward = self.slot[:3, 3].copy()
        self.outward[2] = 0
        self.outward /= np.linalg.norm(self.outward)
        self.entry = self.outward  # the way the part goes in, reversed (slot frame)
        # Collision bits as in PhysicsReplay --compliant: a held part passes through the left
        # fingers but hits the carrier and the other part; released, the fingertips can push it.
        r.set_left_fingers()
        r.soften_right_fingers()
        self.part_geoms = {n: r.pieces(n) for n in ('gear_3', 'pin_2')}

    # ------------------------------------------------------------ helpers --
    def carrier(self):
        return pose_of(self.m, self.d, self.replay.body['large_carrier'])

    def part(self):
        return pose_of(self.m, self.d, self.replay.body[self.PART])

    def slot_error(self):
        """The part's pose relative to its slot: (position mm, rotation vector rad)."""
        rel = np.linalg.inv(self.carrier() @ self.slot) @ self.part()
        return rel[:3, 3] * 1000, Rotation.from_matrix(rel[:3, :3]).as_rotvec()

    def tilt_error(self):
        """The part's tilt from the slot axis (rad); spin about its own axis does not matter."""
        rel = np.linalg.inv(self.carrier() @ self.slot) @ self.part()
        axis = rel[:3, 2]
        return np.cross([0., 0., 1.], axis)  # rotation vector (small-angle) that would tilt it upright

    def seated(self):
        rel = np.linalg.inv(self.carrier()) @ self.part()
        err = (rel[:3, 3] - self.slot[:3, 3]) * 1000
        tilt = np.degrees(np.arccos(np.clip(rel[2, 2], -1, 1)))
        return bool(np.linalg.norm(err[:2]) < 1.5 and -3 < err[2] < 1 and tilt < 4)

    def ik(self, target, q, iters=30):
        """Left arm joints putting the left grip_site at `target` (damped least squares)."""
        m, d = self.m, self.ik_data
        d.qpos[:] = self.d.qpos
        d.qpos[self.left_q] = q
        site = self.replay.sites['left']
        jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        for _ in range(iters):
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            pos_err = target[:3, 3] - d.site_xpos[site]
            rot_err = Rotation.from_matrix(target[:3, :3] @ d.site_xmat[site].reshape(3, 3).T).as_rotvec()
            err = np.r_[pos_err, .5 * rot_err]
            if np.linalg.norm(pos_err) < 5e-5 and np.linalg.norm(rot_err) < 1e-3:
                break
            mujoco.mj_jacSite(m, d, jacp, jacr, site)
            J = np.r_[jacp[:, self.left_dofs], .5 * jacr[:, self.left_dofs]]
            dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), err)
            d.qpos[self.left_q] = np.clip(d.qpos[self.left_q] + dq, self.range[:, 0], self.range[:, 1])
        return d.qpos[self.left_q].copy()

    def commanded_grip(self):
        m, d = self.m, self.ik_data
        d.qpos[:] = self.d.qpos
        d.qpos[self.left_q] = self.q
        mujoco.mj_kinematics(m, d)
        site = self.replay.sites['left']
        return transform(d.site_xpos[site], d.site_xmat[site].reshape(3, 3))

    # ------------------------------------------------------------ env API --
    def reset(self):
        r, m, d = self.replay, self.m, self.d
        mujoco.mj_resetDataKeyframe(m, d, 0)
        d.eq_active[list(r.weld.values())] = 0  # the part holds only: the fingers are coupled by equalities too
        state = self.starts[self.rng.integers(len(self.starts))]
        r.set_arms(state)
        mujoco.mj_kinematics(m, d)
        noise = transform(self.rng.normal(0, self.noise_mm / 1000 / np.sqrt(3), 3),
                          Rotation.from_rotvec(self.rng.normal(0, np.radians(self.noise_deg) / np.sqrt(3), 3)).as_matrix())
        self.carrier_grip = self.base_carrier_grip @ noise
        for name, side, grip in (('large_carrier', 'right', self.carrier_grip), (self.PART, 'left', self.part_grip)):
            target = r.grip(side) @ grip
            r.place(name, target)
            d.mocap_pos[r.mocap[name]], d.mocap_quat[r.mocap[name]] = target[:3, 3], quat(target[:3, :3])
            d.eq_active[r.weld[name]] = 1
        self.place_others()
        self.q = state[:6].copy()
        self.state = state
        self.released, self.t, self.closed = False, 0, False
        self.fit_mm = None  # depth (mm outside the slot) the press fit holds the gear at, once caught
        self.m.eq_solref[r.weld[self.PART]] = [.004, 1]
        self.set_contact(self.PART, held=True)
        mujoco.mj_forward(m, d)
        return self.observation()

    def place_others(self):
        self.replay.place('pin_2', transform([.45, .25, .0503], np.eye(3)))  # out of the way

    def each_step(self):
        """Called every physics step, before the holds follow the grippers."""

    def set_contact(self, name, held):
        self.m.geom_contype[self.part_geoms[name]] = self.replay.HELD if held else 2

    def press_fit(self):
        """The real slot holds a half-inserted gear by friction; the sim's slot is looser. Once
        the released gear sits in the slot's mouth (lined up to 1 mm and 5 deg, at most 8 mm
        out) a soft weld holds it there, keeping its spin, and lets it move only inward: each
        push that takes it deeper becomes the new hold point."""
        r, d = self.replay, self.d
        rel = np.linalg.inv(self.carrier() @ self.slot) @ self.part()
        pos = rel[:3, 3] * 1000
        radial = pos @ self.entry
        off_axis = np.linalg.norm(pos - radial * self.entry)
        tilt = np.degrees(np.arccos(np.clip(rel[2, 2], -1, 1)))
        if self.fit_mm is None:
            if radial > self.CATCH_MM or off_axis > 1 or tilt > 5:
                return
            self.fit_mm = radial
            self.m.eq_solref[r.weld[self.PART]] = [.02, 1]  # soft enough for a fingertip push
            d.eq_active[r.weld[self.PART]] = 1
        self.fit_mm = max(0., min(self.fit_mm, radial))
        spin = np.arctan2(rel[1, 0], rel[0, 0])
        hold = np.eye(4)
        hold[:3, :3] = Rotation.from_euler('z', spin).as_matrix()
        hold[:3, 3] = self.fit_mm / 1000 * self.entry
        target = self.carrier() @ self.slot @ hold
        d.mocap_pos[r.mocap[self.PART]], d.mocap_quat[r.mocap[self.PART]] = target[:3, 3], quat(target[:3, :3])

    def grip_in_slot(self):
        """Left fingertip point (grip_site) in the slot frame, mm."""
        rel = np.linalg.inv(self.carrier() @ self.slot) @ self.replay.grip('left')
        return rel[:3, 3] * 1000

    def observation(self):
        pos, _ = self.slot_error()
        return np.r_[self.d.qpos[self.left_q], pos, self.tilt_error(), float(self.released)]

    def step(self, action):
        """action: [dx, dy, dz (mm, carrier frame), droll, dpitch, dyaw (deg), release (>0),
        close the gripper after release (>0)]. The hand keeps moving after the release, so it
        can push the gear home with its closed fingertips."""
        r, m, d = self.replay, self.m, self.d
        move = np.clip(action[:3], -2, 2) / 1000
        # Keep the fingertips and a held gear off the table: with the carrier held tilted, moves
        # along its plane can point down. Trim the downward part of a move that would take the
        # commanded fingertips or gear below the floor (the sideways part still happens).
        carrier_rot = self.carrier()[:3, :3]
        world = carrier_rot @ move
        tip = self.commanded_grip()
        low = min(tip[2, 3], (tip @ self.part_grip)[2, 3] if not self.released else np.inf)
        floor = TABLE_Z + FLOOR
        if low + world[2] < floor:
            world[2] = min(max(world[2], floor - low), .002)
            move = carrier_rot.T @ world
        turn = np.radians(np.clip(action[3:6], -2, 2))
        if True:
            carrier = self.carrier()
            # Move from where the hand was *commanded*, not where it is: the joint position
            # control sags by about 1 mrad under gravity, so building on the measured pose
            # would swallow sub-millimetre moves; building on the command integrates them.
            grip = self.commanded_grip()
            target = grip.copy()
            # A blocked hand must not keep stepping its command into what blocks it: the joint
            # controllers' force would build until something gives (the carrier flies). Once the
            # command leads the hand by 3 mm, moves that would add to the lead are dropped.
            world = carrier[:3, :3] @ move
            lead = grip[:3, 3] - r.grip('left')[:3, 3]
            if np.linalg.norm(lead) > .003 and world @ lead > 0:
                world -= (world @ lead) / (lead @ lead) * lead
            target[:3, 3] += world
            target[:3, :3] = carrier[:3, :3] @ Rotation.from_rotvec(turn).as_matrix() @ carrier[:3, :3].T @ grip[:3, :3]
            # At most 0.03 rad per joint per step: an unreachable target must not fling the gear.
            q_new = self.q + np.clip(self.ik(target, self.q) - self.q, -.03, .03)
            # Backstop: a rotation or IK step must not sink the fingertips or gear well below the floor.
            q_old, self.q = self.q, q_new
            tip = self.commanded_grip()
            low = min(tip[2, 3], (tip @ self.part_grip)[2, 3] if not self.released else np.inf)
            if low < TABLE_Z + FLOOR - .003:
                self.q = q_old
            if action[6] > 0 and not self.released:
                self.released, self.released_t = True, self.t
                d.eq_active[r.weld[self.PART]] = 0
                self.set_contact(self.PART, held=True)  # a part held by friction too (PinStageEnv)
            # The released part keeps passing through the fingers until they have opened (0.4 s):
            # the rigid sim fingers overlap what they held, and would push it out sideways.
            if self.released and self.t == self.released_t + 8:
                self.set_contact(self.PART, held=False)
            self.closed = self.released and len(action) > 7 and action[7] > 0
        ctrl = self.state.copy()
        ctrl[:6] = self.q
        if self.released:
            ctrl[6] = 0. if self.closed else 1.2  # open after release; closed to push
        r.set_arms(ctrl, ctrl_only=True)
        for _ in range(r.steps):
            if self.released:
                self.press_fit()
            self.each_step()
            for name, side, grip in (('large_carrier', 'right', self.carrier_grip), (self.PART, 'left', self.part_grip)):
                if name == self.PART and self.released:
                    continue
                if d.eq_active[r.weld[name]]:
                    target = r.grip(side) @ grip
                    d.mocap_pos[r.mocap[name]], d.mocap_quat[r.mocap[name]] = target[:3, 3], quat(target[:3, :3])
            mujoco.mj_step(m, d)
        self.t += 1
        pos, _ = self.slot_error()
        seated = self.seated()
        reward = -np.linalg.norm(pos) / 10 - np.linalg.norm(self.tilt_error()) + (10. if seated else 0.)
        return self.observation(), reward, seated, dict(slot_error_mm=pos, seated=seated)



class PinStageEnv(GearInsertionEnv):
    """The second half of the insertion: the gear already seated in the carrier's rim, the pin
    in the left hand (Zeb's grip, near its top). Reset: a demonstration's arm poses `lead_s`
    before it released the pin. The pin goes down the carrier's face, through the upper plate
    and the gear's bore (8.00 mm pin, 8.20 mm bore, no chamfer: CAD collision pieces), and is
    released part way in and pressed home with the closed fingertips. The gear slides in its
    slot until the pin holds it, so a press fit holds it at its seat here; the released pin
    gets the same press fit along the bore (the sim has no stop below it, the real hole's
    friction holds it). Seated: Zeb's limits, the pin within 0.75 mm of its depth, 3 degrees,
    and its centreline in the bore (0.3 mm, about the clearance).

    grasp='friction' (default): the fingers hold the pin, squeezing it (the real gripper reads
    fully closed on it) with nothing else attached: at reset a hold steadies the pin only
    while the pads close on it. Along recorded pin holds this slips under 2.5 mm and turns
    under 1 degree. The pin keeps its grip file's axis, with the point 12.7 mm from its tip
    (where Zeb's grip holds it) just inside the fingertips, between the pads.
    grasp='weld': the pin is welded to the hand, as before."""
    PART, HOLD_PHASE = 'pin_2', 'pin_held'
    CATCH_MM = 13.  # in the upper plate's hole (the fingertips meet the face 11 mm out)
    PAD_POINT = np.array([-.003, 0., 0.])  # grip frame: just inside the fingertips, mid-pad
    GRIP_UP = .0127  # the fingers hold the pin this far from its tip

    def __init__(self, *args, grasp='friction', **kwargs):
        super().__init__(*args, **kwargs)
        self.grasp = grasp
        self.entry = np.array([0., 0., 1.])
        if grasp == 'friction':
            # The pads (grip frame x -45..0 mm, z -10..10 mm) squeeze along y: keep the pin's axis,
            # in the x-z plane, and put the held point between them.
            axis = self.part_grip[:3, 2].copy()
            axis[1] = 0
            axis /= np.linalg.norm(axis)
            side = np.array([0., 1., 0.])
            rot = np.c_[np.cross(side, axis), side, axis]
            self.part_grip = transform(self.PAD_POINT - self.GRIP_UP * axis, rot)
        # The scene's contacts (solref 0.02) let a 7 g pin sink millimetres into the bore's walls
        # under a fingertip push, so the 0.1 mm clearance would mean nothing and the pin would
        # tilt and wedge. The pin's own contacts are stiff, and win the mix with what they touch.
        pin = self.part_geoms['pin_2']
        self.m.geom_solref[pin] = [.003, 1]
        self.m.geom_solimp[pin] = [.95, .99, .0005, .5, 2]
        self.m.geom_solmix[pin] = 1e3
        # The hold stays firm (a weld's solref scales with the part's mass: for the 7 g pin the
        # firm one is about 0.4 N/mm, near the fin-ray fingers' 1 N/mm in pin_insertion_env.py;
        # the soft one would let gravity swing it by degrees).
        self.gear_seat = np.eye(4)
        self.gear_seat[:3, 3] = assembled_offsets()['gear_3']

    def place_others(self):
        r, d = self.replay, self.d
        target = r.grip('right') @ self.carrier_grip @ self.gear_seat
        r.place('gear_3', target)
        d.mocap_pos[r.mocap['gear_3']], d.mocap_quat[r.mocap['gear_3']] = target[:3, 3], quat(target[:3, :3])
        self.m.eq_solref[r.weld['gear_3']] = [.02, 1]
        d.eq_active[r.weld['gear_3']] = 1
        self.set_contact('gear_3', held=False)

    def reset(self):
        obs = super().reset()
        if self.grasp != 'friction':
            return obs
        r, m, d = self.replay, self.m, self.d
        # The pin keeps its own collision bits, so the pads touch it; the weld set up by the reset
        # steadies it (as the table it was picked from would) while the fingers close, then goes.
        self.set_contact('pin_2', held=False)
        left, right = r.fingers['left']
        d.qpos[left], d.qpos[right] = .0047, -.0047  # just open around the 8 mm pin
        ctrl = self.state.copy()
        ctrl[:6] = self.q
        r.set_arms(ctrl, ctrl_only=True)
        for k in range(12):
            if k == 6:
                d.eq_active[r.weld['pin_2']] = 0
            for _ in range(r.steps):
                self.each_step()
                target = r.grip('right') @ self.carrier_grip
                d.mocap_pos[r.mocap['large_carrier']], d.mocap_quat[r.mocap['large_carrier']] = target[:3, 3], quat(target[:3, :3])
                mujoco.mj_step(m, d)
        return self.observation()

    def each_step(self):
        r, d = self.replay, self.d
        target = self.carrier() @ self.gear_seat
        d.mocap_pos[r.mocap['gear_3']], d.mocap_quat[r.mocap['gear_3']] = target[:3, 3], quat(target[:3, :3])

    def seated(self):
        pos, _ = self.slot_error()
        tilt = np.degrees(np.linalg.norm(self.tilt_error()))
        return bool(np.linalg.norm(pos[:2]) < .3 and abs(pos[2]) < .75 and tilt < 3)


def scripted(env, max_steps=400, standoff_mm=15., half_mm=6., gear_radius_mm=12., tries=3, clear_mm=22.):
    """Privileged baseline doing what the real operators do (seen in the wrist videos):
    slide the gear edgewise into the slot in the carrier's rim, release it half in (the
    slot holds it), back off, close the fingers, and push it home with the fingertips.
    A push that stalls short of flush backs off and comes in again (up to `tries` pushes).
    The approach goes around the carrier, not through it: out past the rim (`clear_mm`
    beyond the slot) while out of the carrier's plane, into the plane, then in sideways."""
    outward = env.slot[:3, 3].copy()
    outward[2] = 0
    outward /= np.linalg.norm(outward)
    stage, count, pushes = 'align', 0, 0
    for _ in range(max_steps):
        pos, _ = env.slot_error()
        rot = env.tilt_error()
        radial = pos @ outward
        off_axis = pos - radial * outward
        tilt = np.degrees(np.linalg.norm(rot))
        turn = -np.degrees(rot) * .5
        release, close = -1., -1.
        if stage == 'align':
            lined_up = np.linalg.norm(off_axis) < .4 and tilt < 1.5
            height = pos[2]  # out of the carrier's plane (slot frame z)
            sideways = off_axis - height * np.array([0., 0., 1.])
            if abs(height) > 3 and radial < clear_mm:
                # Out of the plane and over the carrier: move out past its rim first, at this height.
                move = (clear_mm + 2 - radial) * .7 * outward - sideways * .5
            elif abs(height) > 3:
                # Clear of the rim: come into the carrier's plane.
                move = -height * .7 * np.array([0., 0., 1.]) - sideways * .5 - (radial - clear_mm - 2) * .3 * outward
            elif radial < standoff_mm - 2 and not lined_up:
                move = (standoff_mm - radial) * .7 * outward * (np.linalg.norm(off_axis) > 1.5) - off_axis * .7
            else:
                move = -off_axis * .7 - (radial - standoff_mm) * .3 * outward
            if lined_up and abs(radial - standoff_mm) < 1.5:
                stage = 'insert'
        elif env.released and np.linalg.norm(pos) > 60:
            break  # the gear is gone
        elif stage == 'insert':
            move = -off_axis * .7 - min(radial - half_mm, 1.5) * outward
            if radial <= half_mm + .3:
                stage, release, count = 'open', 1., 0
        else:
            release = 1.
            turn = np.zeros(3)
            tip = env.grip_in_slot()
            if stage == 'open':  # let the fingers open around the gear before moving
                goal = tip
                count += 1
                if count > 6:
                    stage = 'lift'
                    rel = np.linalg.inv(env.carrier() @ env.slot) @ env.replay.grip('left')
                    lift_goal = tip - 20 * rel[:3, 0]  # back along the fingers, away from the gear
            elif stage == 'lift':
                goal = lift_goal
                if np.linalg.norm(tip - goal) < 1.5:
                    stage = 'back_off'
            elif stage == 'back_off':  # out past the gear's rim
                goal = pos + (gear_radius_mm + 12) * outward
                close = -1.
                if np.linalg.norm(tip - goal) < 1.5:
                    stage, count = 'close', 0
            elif stage == 'close':
                goal, close = tip, 1.
                count += 1
                if count > 8:
                    # Aim just outside the gear's rim where it is now. A goal that followed the gear
                    # would chase it inward past its seat if the fingertips touched it early.
                    stage, approach_goal, approach_radial = 'approach', pos + (gear_radius_mm + 3) * outward, radial
            elif stage == 'approach':  # fingertips just outside the gear's outer rim
                goal, close = approach_goal, 1.
                touched = radial < approach_radial - .3  # already moving it: this is the push
                if np.linalg.norm(tip - goal) < 1. or touched:
                    stage, count, last_radial = 'push', 0, radial
                    pushes += 1
            elif stage == 'push':  # push inward until the gear is flush
                goal, close = tip - 1. * outward, 1.
                count += 1
                stalled = count > 12 and radial > .3 and radial > last_radial - .05
                last_radial = min(last_radial, radial) if count > 1 else radial
                if radial < .3:
                    stage, count = 'retreat', 0
                elif (stalled or count > 40) and pushes < tries:
                    stage = 'back_off'  # come in again from outside the rim
                elif count > 40:
                    stage, count = 'retreat', 0
            else:  # retreat
                goal, close = tip + 1.5 * outward, 1.
                count += 1
                if count > 6:
                    break
            move = (goal - tip) * .8
        env.step(np.r_[move, turn, release, close])
    for _ in range(10):
        env.step(np.r_[np.zeros(6), 1., 1.])
    return env.seated(), env.slot_error()[0]



def scripted_pin(env, max_steps=400, clear_mm=24., release_mm=12., length_mm=17.88, tries=3):
    """Privileged baseline for the pin: rise until the pin's tip clears the carrier's face
    (`clear_mm` above its seat), line it up over the hole, lower it straight in, release it
    `release_mm` above its seat (the bore holds it), lift, close the fingers and press its
    head down with the fingertips. A lowering that stalls (the tip on the face, not in the
    hole) lifts and lines up again, up to `tries` times; so does a press that stalls."""
    stage, count, tried, last = 'align', 0, 0, None
    for _ in range(max_steps):
        pos, _ = env.slot_error()
        rot = env.tilt_error()
        lateral = np.r_[pos[:2], 0.]
        tilt = np.degrees(np.linalg.norm(rot))
        # Once the tip is in the hole the bore lines the pin up; turning the hand then only levers it.
        turn = -np.degrees(rot) * .5 * (pos[2] > MOUTH_MM)
        release, close = -1., -1.
        up = np.array([0., 0., 1.])
        if env.released and np.linalg.norm(pos) > 60:
            break  # the pin is gone
        if stage == 'align':
            if pos[2] < clear_mm - 2 and np.linalg.norm(lateral) > .3:
                move = (clear_mm - pos[2]) * .7 * up  # up and clear of the face first
            else:
                move = -lateral * .7 + (clear_mm - pos[2]) * .3 * up * (pos[2] < clear_mm - 2)
            if np.linalg.norm(lateral) < .1 and tilt < .5:
                stage, count, last = 'lower', 0, pos[2]
        elif stage == 'lower':  # slowly through the hole's mouth: it has no chamfer
            speed = .3 if MOUTH_MM - 1 < pos[2] < MOUTH_MM + 2 else 1.
            move = -lateral * .7 - min(pos[2] - release_mm, speed) * up
            count += 1
            if pos[2] <= release_mm + .3:
                stage, release, count = 'open', 1., 0
            elif count % 10 == 0:
                if pos[2] > last - .2 and pos[2] < env.CATCH_MM - .5:
                    stage, release, count = 'open', 1., 0  # the fingertips are on the face; the bore holds it
                elif pos[2] > last - .2 and tried < tries:  # stuck on the face
                    stage, tried = 'align', tried + 1
                last = pos[2]
        else:
            release = 1.
            turn = np.zeros(3)
            tip = env.grip_in_slot()
            head = pos + length_mm * up
            if stage in ('approach', 'lift_again') and pos[2] < .3:
                stage, count = 'retreat', 0  # already home
            if stage == 'open':  # let the fingers open wide of the pin before moving
                goal = tip
                count += 1
                if count > 12:
                    stage, lift_goal = 'lift', tip + 12 * up
            elif stage == 'lift':  # slowly, straight up the pin: a drag would tilt it in its short hold
                goal = tip + np.clip(lift_goal - tip, -.6, .6)
                if np.linalg.norm(tip - lift_goal) < 1.5:
                    stage, count = 'close', 0
            elif stage == 'close':
                goal, close = tip, 1.
                count += 1
                if count > 8:
                    stage, count = 'approach', 0
            elif stage == 'approach':  # fingertips just over the pin's head
                goal, close = head + 3 * up, 1.
                count += 1
                if np.linalg.norm(tip - goal) < 1. or count > 30:
                    stage, count, last = 'press', 0, pos[2]
                    tried += 1
            elif stage == 'press':
                goal, close = tip - 1. * up, 1.
                count += 1
                stalled = count > 12 and pos[2] > last - .05
                last = min(last, pos[2]) if count > 1 else pos[2]
                if pos[2] < .3 or ((stalled or count > 40) and tried >= tries + 1):
                    stage, count = 'retreat', 0
                elif stalled or count > 40:
                    stage = 'lift_again'
            elif stage == 'lift_again':
                goal, close = head + 8 * up, 1.
                if np.linalg.norm(tip - goal) < 1.5:
                    stage, count = 'approach', 0
            else:  # retreat
                goal, close = tip + 1.5 * up, 1.
                count += 1
                if count > 6:
                    break
            move = (goal - tip) * .8
        env.step(np.r_[move, turn, release, close])
    for _ in range(10):
        env.step(np.r_[np.zeros(6), 1., 1.])
    return env.seated(), env.slot_error()[0]


def _evaluate(args):
    seed, demos, count, noise_mm, noise_deg, root, cameras, grips, part = args
    env_class, controller = (PinStageEnv, scripted_pin) if part == 'pin' else (GearInsertionEnv, scripted)
    env = env_class(root, demos, noise_mm=noise_mm, noise_deg=noise_deg, seed=seed, cameras=cameras, grips=grips)
    out = []
    for _ in range(count):
        env.reset()
        start = float(np.linalg.norm(env.slot_error()[0]))
        seated, final = controller(env)
        out.append((start, bool(seated), np.round(final, 1).tolist()))
    return out


def main():
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--episodes', type=int, default=40)
    parser.add_argument('--part', choices=('gear', 'pin'), default='gear',
                        help='pin: the second half (PinStageEnv), the gear already seated')
    parser.add_argument('--demos', type=int, default=60, help='real demonstrations to draw starts from')
    parser.add_argument('--demo-ids', type=int, nargs='+', help='these demonstrations instead of the first --demos')
    parser.add_argument('--root', type=Path, default=ROOT / 'outputs/real_gear', help='local copy of the real dataset')
    parser.add_argument('--cameras', type=Path, default=ROOT / 'sim/real_cameras.json',
                        help='fitted cameras (their overhead solve places the right base)')
    parser.add_argument('--grasps', type=Path, help='carrier and gear grips (default: CARRIER_GRIP, Zeb\'s gear grip)')
    parser.add_argument('--noise-mm', type=float, default=4.)
    parser.add_argument('--noise-deg', type=float, default=6.)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    per = -(-args.episodes // args.workers)
    demos = args.demo_ids or list(range(args.demos))
    jobs = [(seed, demos, per, args.noise_mm, args.noise_deg, args.root, args.cameras, args.grasps, args.part)
            for seed in range(args.workers)]
    with ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        results = [r for chunk in pool.map(_evaluate, jobs) for r in chunk]
    for k, (start, seated, final) in enumerate(results):
        print(f'episode {k}: start {start:.1f} mm from slot -> {"seated" if seated else "not seated"} (final {final} mm)')
    wins = sum(seated for _, seated, _ in results)
    print(f'scripted {args.part} baseline: {wins} of {len(results)} seated ({100 * wins / len(results):.0f}%)')


if __name__ == '__main__':
    main()
