#!/usr/bin/env python3
"""Gear insertion from real pre-insertion poses, in the CAD sim: the RL starting point.

Replaying real demonstrations through the sim's physics (tools/physics_replay.py)
showed the hands' relative pose at insertion is repeatable to only about 4-5 mm
and 10 degrees, against a 1.5 mm seating tolerance: the real operators corrected
by feel and by eye. This environment isolates that last step so a policy can
learn the correction.

Reset: a real demonstration's arm poses two seconds before it released the gear
(phases from tools/replay_real_in_sim.py). The right arm holds the carrier at the
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

from tools.physics_replay import PhysicsReplay, assembled_offsets, grasps, pose_of, quat
from tools.replay_real_in_sim import phases, transform

# The carrier in the right grip_site frame, measured from 68 demonstrations' gear
# releases (gear in its slot at release): mean position and mean rotation.
CARRIER_GRIP = transform([-.0279, -.0054, .0214], [[-.88, .47, -.11], [-.45, -.72, .52], [.17, .51, .85]])


class GearInsertionEnv:
    def __init__(self, root=ROOT / 'outputs/real_gear', episodes=None, noise_mm=4., noise_deg=6., seed=0,
                 lead_s=2.):
        from tools.augment_real_dataset import episode_rows
        self.replay = r = PhysicsReplay(ROOT / 'sim/real_cameras.json')
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
        self.released, self.t = False, 0
        mujoco.mj_forward(m, d)
        return self.observation()

    def observation(self):
        pos, _ = self.slot_error()
        return np.r_[self.d.qpos[self.left_q], pos, self.tilt_error(), float(self.released)]

    def step(self, action):
        """action: [dx, dy, dz (mm, carrier frame), droll, dpitch, dyaw (deg), release (>0)]."""
        r, m, d = self.replay, self.m, self.d
        move = np.clip(action[:3], -2, 2) / 1000
        turn = np.radians(np.clip(action[3:6], -2, 2))
        if not self.released:
            carrier = self.carrier()
            # Move from where the hand was *commanded*, not where it is: the joint position
            # control sags by about 1 mrad under gravity, so building on the measured pose
            # would swallow sub-millimetre moves; building on the command integrates them.
            grip = self.commanded_grip()
            target = grip.copy()
            target[:3, 3] += carrier[:3, :3] @ move
            target[:3, :3] = carrier[:3, :3] @ Rotation.from_rotvec(turn).as_matrix() @ carrier[:3, :3].T @ grip[:3, :3]
            # At most 0.03 rad per joint per step: an unreachable target must not fling the gear.
            self.q = self.q + np.clip(self.ik(target, self.q) - self.q, -.03, .03)
            if action[6] > 0:
                self.released = True
                d.eq_active[r.weld['gear_3']] = 0
        ctrl = self.state.copy()
        ctrl[:6] = self.q
        if self.released:
            ctrl[6] = 1.2  # open the left gripper
        r.set_arms(ctrl, ctrl_only=True)
        for _ in range(r.steps):
            for name, side, grip in (('large_carrier', 'right', self.carrier_grip), ('gear_3', 'left', self.gear_grip)):
                if d.eq_active[r.weld[name]]:
                    target = r.grip(side) @ grip
                    d.mocap_pos[r.mocap[name]], d.mocap_quat[r.mocap[name]] = target[:3, 3], quat(target[:3, :3])
            mujoco.mj_step(m, d)
        self.t += 1
        pos, _ = self.slot_error()
        seated = self.seated()
        reward = -np.linalg.norm(pos) / 10 - np.linalg.norm(self.tilt_error()) + (10. if seated else 0.)
        return self.observation(), reward, seated, dict(slot_error_mm=pos, seated=seated)


def scripted(env, max_steps=160, standoff_mm=15.):
    """Privileged baseline, the way Zeb's controller seats the gear: the slot is in the
    carrier's rim, so line the gear up a standoff outside the slot in the carrier's plane
    (radially), slide it in, release, let it settle."""
    outward = env.slot[:3, 3].copy()
    outward[2] = 0
    outward /= np.linalg.norm(outward)
    for _ in range(max_steps):
        pos, _ = env.slot_error()
        rot = env.tilt_error()
        if env.released:
            break
        radial = pos @ outward  # mm outside the slot along the rim's outward direction
        off_axis = pos - radial * outward  # sideways and height error
        tilt = np.degrees(np.linalg.norm(rot))
        lined_up = np.linalg.norm(off_axis) < .4 and tilt < 1.5
        if not lined_up and radial < standoff_mm - 2:
            goal = standoff_mm * outward  # back out to the standoff before correcting
            move = -(pos - goal) * .7
            move -= (move @ outward) * outward * (np.linalg.norm(off_axis) > 1.5)  # out first, then across
            move += max(0., standoff_mm - radial) * .7 * outward
        elif not lined_up:
            move = -off_axis * .7 - (radial - standoff_mm) * .3 * outward
        else:
            move = -off_axis * .7 - min(radial, 1.5) * outward  # slide in, 1.5 mm per step
        turn = -np.degrees(rot) * .5
        release = lined_up and radial < .8
        env.step(np.r_[move, turn, 1. if release else -1.])
    for _ in range(10):
        _, _, seated, _ = env.step(np.zeros(7))
    return env.seated(), env.slot_error()[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--episodes', type=int, default=20)
    parser.add_argument('--demos', type=int, default=30, help='real demonstrations to draw starts from')
    parser.add_argument('--noise-mm', type=float, default=4.)
    parser.add_argument('--noise-deg', type=float, default=6.)
    args = parser.parse_args()
    env = GearInsertionEnv(episodes=range(args.demos), noise_mm=args.noise_mm, noise_deg=args.noise_deg)
    print(f'{len(env.starts)} real starting poses')
    wins = 0
    for k in range(args.episodes):
        env.reset()
        start = np.linalg.norm(env.slot_error()[0])
        seated, final = scripted(env)
        wins += seated
        print(f'episode {k}: start {start:.1f} mm from slot -> {"seated" if seated else "not seated"} '
              f'(final {np.round(final, 1)} mm)', flush=True)
    print(f'scripted baseline: {wins} of {args.episodes} seated')


if __name__ == '__main__':
    main()
