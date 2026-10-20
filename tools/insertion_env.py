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
are soft welds to handles that follow the grippers, so contacts push back.

The gear seats sideways into a slot in the carrier's rim (as in Zeb's controller),
not from above.

Step: a small move of the left gripper (up to 2 mm and 2 degrees, in the
carrier's frame) through damped least-squares IK, and a release flag; 50 ms of
physics. After release the gear is free and the episode runs a short settle.

Observation (privileged, for now): left arm joints, gear position relative to its
slot in the carrier (mm), its tilt from the slot axis (spin about its own axis
does not matter), whether it is released.
Reward: minus the slot error, plus 10 when seated (the sim's tolerances).

    python tools/insertion_env.py --episodes 20     # scripted baseline success rate
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
FLOOR = .012  # m above the table that fingertips and a held gear's centre stay above
CARRIER_GRIP = transform([-.0279, -.0054, .0214], [[-.88, .47, -.11], [-.45, -.72, .52], [.17, .51, .85]])


class GearInsertionEnv:
    def __init__(self, root=ROOT / 'outputs/real_gear', episodes=None, noise_mm=4., noise_deg=6., seed=0,
                 lead_s=2.):
        from tools.augment_real_dataset import episode_rows
        self.replay = r = PhysicsReplay(ROOT / 'sim/real_cameras.json', exclude_left=False)
        self.m, self.d = r.model, r.data
        self.rng = np.random.default_rng(seed)
        self.noise_mm, self.noise_deg = noise_mm, noise_deg
        self.gear_grip = grasps(ROOT / 'none')['gear_3']
        self.slot = np.eye(4)
        self.slot[:3, 3] = assembled_offsets()['gear_3']
        info = json.loads((Path(root) / 'meta/info.json').read_text())
        self.starts = []
        for e in (episodes if episodes is not None else range(info['total_episodes'])):
            numeric, _ = episode_rows(Path(root), e)
            st = numeric['observation.state'].astype(np.float64)
            g = next((s for s in phases(st)['left'] if s['phase'] == 'gear_held'), None)
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
        # Collision bits: the left gripper's geoms get type 2, so a held gear (type 4, affinity 1)
        # passes through the fingers but still hits the carrier; once released the gear goes back
        # to type 1 with affinity 1|2 and the closed fingertips can push it.
        left = {m.body(b).id for b in ('link6', 'L_finger', 'R_finger')}
        self.left_geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] in left and m.geom_contype[g]]
        m.geom_contype[self.left_geoms] = 2
        gear = r.body['gear_3']
        self.gear_geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] == gear and (m.geom_contype[g] or m.geom_conaffinity[g])]

    # ------------------------------------------------------------ helpers --
    def carrier(self):
        return pose_of(self.m, self.d, self.replay.body['large_carrier'])

    def gear(self):
        return pose_of(self.m, self.d, self.replay.body['gear_3'])

    def slot_error(self):
        """Gear pose relative to its slot: (position mm, rotation vector rad)."""
        rel = np.linalg.inv(self.carrier() @ self.slot) @ self.gear()
        return rel[:3, 3] * 1000, Rotation.from_matrix(rel[:3, :3]).as_rotvec()

    def tilt_error(self):
        """Gear tilt from the slot axis (rad); spin about the gear's own axis does not matter."""
        rel = np.linalg.inv(self.carrier() @ self.slot) @ self.gear()
        axis = rel[:3, 2]
        return np.cross([0., 0., 1.], axis)  # rotation vector (small-angle) that would tilt it upright

    def seated(self):
        rel = np.linalg.inv(self.carrier()) @ self.gear()
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
        d.eq_active[:] = 0
        state = self.starts[self.rng.integers(len(self.starts))]
        r.set_arms(state)
        mujoco.mj_kinematics(m, d)
        noise = transform(self.rng.normal(0, self.noise_mm / 1000 / np.sqrt(3), 3),
                          Rotation.from_rotvec(self.rng.normal(0, np.radians(self.noise_deg) / np.sqrt(3), 3)).as_matrix())
        self.carrier_grip = CARRIER_GRIP @ noise
        for name, side, grip in (('large_carrier', 'right', self.carrier_grip), ('gear_3', 'left', self.gear_grip)):
            target = r.grip(side) @ grip
            r.place(name, target)
            d.mocap_pos[r.mocap[name]], d.mocap_quat[r.mocap[name]] = target[:3, 3], quat(target[:3, :3])
            d.eq_active[r.weld[name]] = 1
        r.place('pin_2', transform([.45, .25, .0503], np.eye(3)))  # out of the way
        self.q = state[:6].copy()
        self.state = state
        self.released, self.t, self.closed = False, 0, False
        self.fit_mm = None  # depth (mm outside the slot) the press fit holds the gear at, once caught
        self.m.eq_solref[r.weld['gear_3']] = [.004, 1]
        self.set_gear_contact(held=True)
        mujoco.mj_forward(m, d)
        return self.observation()

    def set_gear_contact(self, held):
        m = self.m
        m.geom_contype[self.gear_geoms] = 4 if held else 1
        m.geom_conaffinity[self.gear_geoms] = 1 if held else 3

    def press_fit(self):
        """The real slot holds a half-inserted gear by friction; the sim's slot is looser. Once
        the released gear sits in the slot's mouth (lined up to 1 mm and 5 deg, at most 8 mm
        out) a soft weld holds it there, keeping its spin, and lets it move only inward: each
        push that takes it deeper becomes the new hold point."""
        r, d = self.replay, self.d
        rel = np.linalg.inv(self.carrier() @ self.slot) @ self.gear()
        pos = rel[:3, 3] * 1000
        radial = pos @ self.outward
        off_axis = np.linalg.norm(pos - radial * self.outward)
        tilt = np.degrees(np.arccos(np.clip(rel[2, 2], -1, 1)))
        if self.fit_mm is None:
            if radial > 8 or off_axis > 1 or tilt > 5:
                return
            self.fit_mm = radial
            self.m.eq_solref[r.weld['gear_3']] = [.02, 1]  # soft enough for a fingertip push
            d.eq_active[r.weld['gear_3']] = 1
        self.fit_mm = max(0., min(self.fit_mm, radial))
        spin = np.arctan2(rel[1, 0], rel[0, 0])
        hold = np.eye(4)
        hold[:3, :3] = Rotation.from_euler('z', spin).as_matrix()
        hold[:3, 3] = self.fit_mm / 1000 * self.outward
        target = self.carrier() @ self.slot @ hold
        d.mocap_pos[r.mocap['gear_3']], d.mocap_quat[r.mocap['gear_3']] = target[:3, 3], quat(target[:3, :3])

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
        low = min(tip[2, 3], (tip @ self.gear_grip)[2, 3] if not self.released else np.inf)
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
            target[:3, 3] += carrier[:3, :3] @ move
            target[:3, :3] = carrier[:3, :3] @ Rotation.from_rotvec(turn).as_matrix() @ carrier[:3, :3].T @ grip[:3, :3]
            # At most 0.03 rad per joint per step: an unreachable target must not fling the gear.
            q_new = self.q + np.clip(self.ik(target, self.q) - self.q, -.03, .03)
            # Backstop: a rotation or IK step must not sink the fingertips or gear well below the floor.
            q_old, self.q = self.q, q_new
            tip = self.commanded_grip()
            low = min(tip[2, 3], (tip @ self.gear_grip)[2, 3] if not self.released else np.inf)
            if low < TABLE_Z + FLOOR - .003:
                self.q = q_old
            if action[6] > 0 and not self.released:
                self.released = True
                d.eq_active[r.weld['gear_3']] = 0
                self.set_gear_contact(held=False)
            self.closed = self.released and len(action) > 7 and action[7] > 0
        ctrl = self.state.copy()
        ctrl[:6] = self.q
        if self.released:
            ctrl[6] = 0. if self.closed else 1.2  # open after release; closed to push
        r.set_arms(ctrl, ctrl_only=True)
        for _ in range(r.steps):
            if self.released:
                self.press_fit()
            for name, side, grip in (('large_carrier', 'right', self.carrier_grip), ('gear_3', 'left', self.gear_grip)):
                if name == 'gear_3' and self.released:
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
                    stage = 'approach'
            elif stage == 'approach':  # fingertips just outside the gear's outer rim
                goal, close = pos + (gear_radius_mm + 3) * outward, 1.
                if np.linalg.norm(tip - goal) < 1.:
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


def _evaluate(args):
    seed, demos, count, noise_mm, noise_deg = args
    env = GearInsertionEnv(episodes=range(demos), noise_mm=noise_mm, noise_deg=noise_deg, seed=seed)
    out = []
    for _ in range(count):
        env.reset()
        start = float(np.linalg.norm(env.slot_error()[0]))
        seated, final = scripted(env)
        out.append((start, bool(seated), np.round(final, 1).tolist()))
    return out


def main():
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--episodes', type=int, default=40)
    parser.add_argument('--demos', type=int, default=60, help='real demonstrations to draw starts from')
    parser.add_argument('--noise-mm', type=float, default=4.)
    parser.add_argument('--noise-deg', type=float, default=6.)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    per = -(-args.episodes // args.workers)
    jobs = [(seed, args.demos, per, args.noise_mm, args.noise_deg) for seed in range(args.workers)]
    with ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        results = [r for chunk in pool.map(_evaluate, jobs) for r in chunk]
    for k, (start, seated, final) in enumerate(results):
        print(f'episode {k}: start {start:.1f} mm from slot -> {"seated" if seated else "not seated"} (final {final} mm)')
    wins = sum(seated for _, seated, _ in results)
    print(f'scripted baseline: {wins} of {len(results)} seated ({100 * wins / len(results):.0f}%)')


if __name__ == '__main__':
    main()
