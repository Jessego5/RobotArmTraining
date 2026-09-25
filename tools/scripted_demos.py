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
from dataclasses import dataclass
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


@dataclass
class Augment:
    """Off-nominal states with corrective labels (DART-style), per episode.

    `noise_pos`/`noise_yaw` drive a smooth Ornstein-Uhlenbeck offset added to
    the *executed* target only.  `miss_prob` shifts a grasp attempt sideways
    by up to `miss_offset`, so the gripper can close beside the cube; the
    expert then opens, re-aligns and retries.  Labels always come from the
    clean plan, so they correct the offset instead of copying it.
    """

    noise_pos: float = 0.0     # m, stationary standard deviation
    noise_yaw: float = 0.0     # rad
    noise_tau: float = 0.5     # s, correlation time
    miss_prob: float = 0.0
    miss_offset: float = 0.0   # m


class Expert:
    # The clean expert only operates the gripper within this horizontal
    # distance of the planned point; the jaw opens ~8 cm around a 4.5 cm cube.
    ALIGN_TOL = 0.015  # m

    def __init__(self, sim: PantheraSim, rng: np.random.Generator, speed: float,
                 augment: Augment, noise_rng: np.random.Generator):
        self.sim, self.rng = sim, rng
        self.speed = speed * rng.uniform(0.85, 1.15)
        self.q = sim.q.copy()
        self.grip = 1.0
        self.label_grip = 1.0
        self.gripper_hold: tuple[np.ndarray, float] | None = None  # (position, previous grip)
        self.t = 0.0
        self.episode = Episode()
        self.labels: list[np.ndarray] = []
        self.physics_steps = max(1, round(1.0 / CONTROL_HZ / sim.dt))
        self.target_pos, self.target_quat = sim.ee_pose()
        self.yaw = yaw_of(self.target_quat)
        self.augment, self.noise_rng = augment, noise_rng
        self.noise = np.zeros(4)   # x, y, z, yaw
        self.bias = np.zeros(3)    # deliberate grasp offset
        self.retries = 0

    def _advance_noise(self) -> None:
        a = self.augment
        if a.noise_pos <= 0 and a.noise_yaw <= 0:
            return
        dt = 1.0 / CONTROL_HZ
        sigma = np.array([a.noise_pos] * 3 + [a.noise_yaw])
        self.noise += (-self.noise * dt / a.noise_tau
                       + sigma * np.sqrt(2 * dt / a.noise_tau) * self.noise_rng.standard_normal(4))

    def tick(self) -> None:
        sim = self.sim
        self._advance_noise()
        perturbed = np.any(self.noise) or np.any(self.bias)
        executed_pos = self.target_pos + self.noise[:3] + self.bias
        executed_quat = grasp_quat(self.yaw + self.noise[3]) if perturbed else self.target_quat
        # The label is the clean plan solved from where the arm actually is.
        if perturbed:
            label_q, _, _ = sim.ik(self.target_pos, self.target_quat, q_init=sim.q)
        self.q, pos_err, rot_err = sim.ik(executed_pos, executed_quat, q_init=self.q)
        if not perturbed:
            label_q = self.q
        label_grip = self.grip
        if self.gripper_hold is not None:
            hold_pos, previous = self.gripper_hold
            if np.linalg.norm((sim.ee_pos() - hold_pos)[:2]) > self.ALIGN_TOL:
                label_grip = previous  # re-align before closing or opening
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
        self.labels.append(np.append(label_q, label_grip * GRIPPER_OPEN).astype(np.float32))

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
        self.gripper_hold = (np.asarray(self.target_pos, float).copy(), self.grip)
        while abs(self.grip - opening) > 1e-6:
            self.grip = float(np.clip(opening, self.grip - 2.5 / CONTROL_HZ,
                                      self.grip + 2.5 / CONTROL_HZ))
            self.tick()
        for _ in range(int(settle * CONTROL_HZ)):
            self.tick()
        self.gripper_hold = None

    def grasped(self) -> bool:
        return any(eid >= 0 and self.sim.data.eq_active[eid] for eid in self.sim._grasp_eq)

    def pick(self, index: int, hover: float, attempts: int = 3) -> bool:
        """Grasp cube `index`, retrying from above after a miss."""
        for attempt in range(attempts):
            positions, quats = self.sim.object_poses()
            cube = positions[index]
            yaw = face_yaw(quats[index], cube)
            self.move(cube + [0, 0, hover], yaw)
            if attempt == 0 and self.noise_rng.random() < self.augment.miss_prob:
                angle = self.noise_rng.uniform(0, 2 * np.pi)
                self.bias = self.augment.miss_offset * self.noise_rng.uniform(0.6, 1.0) \
                    * np.array([np.cos(angle), np.sin(angle), 0.0])
            self.move(cube + [0, 0, 0.005], speed=0.10)
            self.gripper(0.0)
            self.bias = np.zeros(3)
            if self.grasped():
                return True
            self.retries += 1
            self.gripper(1.0, settle=0.1)
            self.move(self.sim.ee_pos() + [0, 0, hover])
        return False


def run_episode(seed: int, speed: float, augment: Augment | None = None) -> tuple[Expert, bool]:
    rng = np.random.default_rng(seed)
    sim = PantheraSim()
    sim.reset(randomize=True, rng=rng)
    randomize_arm_start(sim, DEFAULT_ARM_START_RANGE, rng=rng)
    # Augmentation draws from its own stream, so unaugmented episodes are
    # identical to those generated before augmentation existed.
    augment = augment or Augment()
    expert = Expert(sim, rng, speed, augment, np.random.default_rng([seed, 1]))
    index = [sim.object_names.index(name) for name in ORDER]
    hover = 0.08 + rng.uniform(-0.01, 0.02)
    augmented = augment != Augment()

    for moved, below in zip(index[1:], index[:-1]):
        if not expert.pick(moved, hover, attempts=3 if augmented else 1):
            return expert, False
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
    return expert, bool(stack_metrics(sim.object_poses()[0])["three_stack"])


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "data_scripted")
    parser.add_argument("--episodes", type=int, default=300, help="successful episodes to keep")
    parser.add_argument("--first-seed", type=int, default=0)
    parser.add_argument("--speed", type=float, default=0.18, help="Cartesian speed, m/s")
    parser.add_argument("--augment-fraction", type=float, default=0.0,
                        help="fraction of episodes with off-nominal perturbations and "
                             "corrective labels (0 keeps every episode clean)")
    parser.add_argument("--noise-pos", type=float, default=0.008, help="m, executed-target noise")
    parser.add_argument("--noise-yaw-deg", type=float, default=4.0)
    parser.add_argument("--miss-prob", type=float, default=0.4,
                        help="chance a grasp's first attempt is shifted sideways")
    parser.add_argument("--miss-offset", type=float, default=0.025, help="m")
    args = parser.parse_args()
    augment = Augment(noise_pos=args.noise_pos, noise_yaw=np.deg2rad(args.noise_yaw_deg),
                      miss_prob=args.miss_prob, miss_offset=args.miss_offset)

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
    kept = tried = augmented = retried = 0
    seed = args.first_seed
    while kept < args.episodes:
        # A per-seed draw, so the choice does not depend on earlier failures.
        use_augment = np.random.default_rng([seed, 2]).random() < args.augment_fraction
        expert, success = run_episode(seed, args.speed, augment if use_augment else None)
        tried += 1
        if success:
            out = args.output / f"episode_{kept:03d}"
            episode_meta = dict(meta, seed=seed, augmented=use_augment,
                                grasp_retries=expert.retries)
            if use_augment:
                episode_meta["augment"] = vars(augment)
            expert.episode.save(out, episode_meta, CONTROL_HZ, save_video=False)
            # Corrective labels ride along in data.npz; builders prefer them to ctrl.
            with np.load(out / "data.npz") as data:
                arrays = dict(data)
            arrays["ctrl_label"] = np.stack(expert.labels)
            np.savez_compressed(out / "data.npz", **arrays)
            kept += 1
            augmented += use_augment
            retried += expert.retries > 0
        seed += 1
    print(f"kept {kept} of {tried} scripted episodes ({kept / tried:.0%} success); "
          f"{augmented} augmented, {retried} with grasp retries")


if __name__ == "__main__":
    main()
