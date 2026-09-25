#!/usr/bin/env python3
"""Generate scripted three-cube stacking demonstrations.

A privileged scripted expert (it reads the cube poses) stacks the cubes in a
fixed order -- red at the bottom, then green, then blue -- from the same
randomized cube layouts and arm starts as teleoperation.  It mirrors how the
human demonstrations grasp: the jaw pitched 30 degrees below horizontal and
turned to a cube face.  Motions are straight Cartesian segments at teleop
speed, solved by the same IK every 1/30 s.

Episodes are written in the teleoperation format (data/episode_NNN/data.npz
and meta.json) so the render -> dataset -> train pipeline is unchanged.  Only
episodes that end in a stable three-cube stack are kept.

    python tools/scripted_demos.py --output data_scripted --episodes 300
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(REPO_ROOT / "sim"), str(REPO_ROOT / "teleop")]

from episode import Episode  # noqa: E402
from keyboard import DEFAULT_ARM_START_RANGE, randomize_arm_start  # noqa: E402
from panthera_env import GRIPPER_OPEN, PantheraSim, mat_to_quat  # noqa: E402
from stack_task import CUBE_EDGE, stack_metrics  # noqa: E402

ORDER = ("cube_red", "cube_green", "cube_blue")  # bottom to top
PITCH = np.deg2rad(30.0)
CONTROL_HZ = 30.0


def yaw_of(quat: np.ndarray) -> float:
    w, x, y, z = quat
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def grasp_quat(yaw: float) -> np.ndarray:
    cz, sz, cp, sp = np.cos(yaw), np.sin(yaw), np.cos(PITCH), np.sin(PITCH)
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    return mat_to_quat(rz @ ry)


def face_yaw(cube_quat: np.ndarray, position: np.ndarray) -> float:
    """Cube-face yaw (mod 90 degrees) closest to pointing away from the base."""
    heading = np.arctan2(position[1], position[0])
    base = yaw_of(cube_quat)
    options = base + np.pi / 2 * np.arange(-4, 5)
    return float(options[np.argmin(np.abs(np.angle(np.exp(1j * (options - heading)))))])


class Expert:
    def __init__(self, sim: PantheraSim, rng: np.random.Generator, speed: float):
        self.sim, self.rng = sim, rng
        self.speed = speed * rng.uniform(0.85, 1.15)
        self.q = sim.q.copy()
        self.grip = 1.0
        self.t = 0.0
        self.episode = Episode()
        self.physics_steps = max(1, round(1.0 / CONTROL_HZ / sim.dt))
        self.target_pos, self.target_quat = sim.ee_pose()
        self.yaw = yaw_of(self.target_quat)
        self.ik_err = (0.0, 0.0)

    def tick(self) -> None:
        sim = self.sim
        self.q, pos_err, rot_err = sim.ik(self.target_pos, self.target_quat, q_init=self.q)
        self.ik_err = (pos_err, rot_err)
        sim.set_arm_ctrl(self.q)
        sim.set_gripper(self.grip)
        sim.step(self.physics_steps)
        self.t += 1.0 / CONTROL_HZ
        ee_p, ee_q = sim.ee_pose()
        obj_p, obj_q = sim.object_poses()
        self.episode.add({
            "t": self.t, "q": sim.q, "dq": sim.dq, "ctrl": sim.data.ctrl.copy(),
            "ee_pos": ee_p, "ee_quat": ee_q, "obj_pos": obj_p, "obj_quat": obj_q,
            "target_pos": np.asarray(self.target_pos), "target_quat": np.asarray(self.target_quat),
            "gripper": self.grip, "ik_pos_err": pos_err, "ik_rot_err": rot_err,
        })

    def move(self, pos: np.ndarray, yaw: float | None = None, speed: float | None = None) -> None:
        """Straight line to `pos` while turning to `yaw`."""
        start, start_yaw = np.asarray(self.target_pos, float), self.yaw
        end_yaw = start_yaw if yaw is None else start_yaw + np.angle(np.exp(1j * (yaw - start_yaw)))
        duration = max(np.linalg.norm(pos - start) / (speed or self.speed),
                       abs(end_yaw - start_yaw) / 1.2, 0.2)
        steps = max(1, int(np.ceil(duration * CONTROL_HZ)))
        for i in range(1, steps + 1):
            a = 0.5 - 0.5 * np.cos(np.pi * i / steps)  # ease in and out
            self.yaw = start_yaw + a * (end_yaw - start_yaw)
            self.target_pos = start + a * (pos - start)
            self.target_quat = grasp_quat(self.yaw)
            self.tick()

    def gripper(self, opening: float, settle: float = 0.25) -> None:
        while abs(self.grip - opening) > 1e-6:
            self.grip = float(np.clip(opening, self.grip - 2.5 / CONTROL_HZ,
                                      self.grip + 2.5 / CONTROL_HZ))
            self.tick()
        for _ in range(int(settle * CONTROL_HZ)):
            self.tick()


def run_episode(seed: int, speed: float) -> tuple[Episode, bool]:
    rng = np.random.default_rng(seed)
    sim = PantheraSim()
    sim.reset(randomize=True, rng=rng)
    randomize_arm_start(sim, DEFAULT_ARM_START_RANGE, rng=rng)
    expert = Expert(sim, rng, speed)
    index = [sim.object_names.index(name) for name in ORDER]
    hover = 0.08 + rng.uniform(-0.01, 0.02)

    for moved, below in zip(index[1:], index[:-1]):
        positions, quats = sim.object_poses()
        cube = positions[moved]
        yaw = face_yaw(quats[moved], cube)
        expert.move(cube + [0, 0, hover], yaw)
        expert.move(cube + [0, 0, 0.005], speed=0.10)
        expert.gripper(0.0)
        if not any(eid >= 0 and sim.data.eq_active[eid] for eid in sim._grasp_eq):
            return expert.episode, False
        held_offset = sim.ee_pos() - sim.object_poses()[0][moved]
        expert.move(sim.ee_pos() + [0, 0, hover])

        positions, _ = sim.object_poses()
        seat = positions[below] + [0, 0, CUBE_EDGE + 0.004] + held_offset
        expert.move(seat + [0, 0, hover])
        expert.move(seat, speed=0.08)
        expert.gripper(1.0)
        expert.move(seat + [0, 0, hover * 0.6])

    for _ in range(int(0.5 * CONTROL_HZ)):
        expert.tick()
    return expert.episode, bool(stack_metrics(sim.object_poses()[0])["three_stack"])


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "data_scripted")
    parser.add_argument("--episodes", type=int, default=300, help="successful episodes to keep")
    parser.add_argument("--first-seed", type=int, default=0)
    parser.add_argument("--speed", type=float, default=0.18, help="Cartesian speed, m/s")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    meta = {
        "robot": "Panthera-HT (HighTorque) 6-DoF + parallel gripper",
        "scene": "sim/panthera/scene.xml",
        "control_mode": "scripted",
        "stack_order": list(ORDER),
        "arm_joints": [f"joint{i}" for i in range(1, 7)],
        "gripper_open_m": GRIPPER_OPEN,
        "arm_start_range_m": list(DEFAULT_ARM_START_RANGE),
        "objects": list(ORDER),
        "frames": {"ee_*/target_*": "robot base frame", "quat": "(w, x, y, z)"},
    }
    kept = tried = 0
    seed = args.first_seed
    while kept < args.episodes:
        episode, success = run_episode(seed, args.speed)
        tried += 1
        if success:
            episode.save(args.output / f"episode_{kept:03d}", dict(meta, seed=seed),
                         CONTROL_HZ, save_video=False)
            kept += 1
        seed += 1
    print(f"kept {kept} of {tried} scripted episodes ({kept / tried:.0%} success)")


if __name__ == "__main__":
    main()
