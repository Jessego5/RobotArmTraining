#!/usr/bin/env python3
"""Replay recorded demonstrations open-loop to bound what a policy can reach.

Each episode starts from its recorded initial state, and its recorded joint
and gripper targets are executed the way ACT actions are (sampled at --hz,
optionally limited to --max-joint-step radians per control step).  The
three-cube success rate is an upper bound for a policy that imitated the
demonstrations perfectly under the same execution settings.

    python tools/replay_actions.py data --episodes 40 --hz 10 20 --max-joint-step 0.15 0.3 0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(REPO_ROOT / "sim"), str(REPO_ROOT / "teleop")]

from panthera_env import GRIPPER_OPEN, PantheraSim  # noqa: E402
from render_vla_dataset import sample_indices  # noqa: E402
from stack_task import stack_metrics  # noqa: E402


def replay(sim: PantheraSim, data: dict, indices: np.ndarray, max_step: float) -> bool:
    first = indices[0]
    sim.data.qpos[:] = sim.model.qpos0
    sim.data.qvel[:] = 0.0
    sim.data.qpos[sim.arm_qadr] = data["q"][first]
    sim.set_object_poses(data["obj_pos"][first], data["obj_quat"][first])
    sim.data.ctrl[:] = data["ctrl"][first]
    sim.data.qpos[sim.finger_qadr] = data["ctrl"][first, 6]
    sim.data.qpos[sim.finger_qadr + 1] = -data["ctrl"][first, 6]  # mirrored right finger
    sim._release_grasps()
    mujoco.mj_forward(sim.model, sim.data)
    t = data["t"]
    for a, b in zip(indices[:-1], indices[1:]):
        target = data["ctrl"][b]
        q = np.clip(target[:6], sim.arm_range[:, 0], sim.arm_range[:, 1])
        if max_step > 0:
            q = np.clip(q, sim.q - max_step, sim.q + max_step)
        sim.set_arm_ctrl(q)
        sim.set_gripper(target[6] / GRIPPER_OPEN)
        sim.step(max(1, round((t[b] - t[a]) / sim.dt)))
    sim.step(50)  # let the stack settle
    return bool(stack_metrics(sim.object_poses()[0])["three_stack"])


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("data", type=Path, help="directory of episode_*/data.npz")
    parser.add_argument("--episodes", type=int, default=40)
    parser.add_argument("--hz", type=float, nargs="+", default=[10.0])
    parser.add_argument("--max-joint-step", type=float, nargs="+", default=[0.15, 0.3, 0.0],
                        help="per-step joint limits to compare; 0 means unlimited")
    args = parser.parse_args()

    episodes = sorted(args.data.glob("episode_*/data.npz"))[:args.episodes]
    if not episodes:
        raise SystemExit(f"no episodes in {args.data}")
    demos = [dict(np.load(path)) for path in episodes]
    sim = PantheraSim()
    full = sum(replay(sim, d, np.arange(len(d["t"])), 0.0) for d in demos)
    print(f"{len(demos)} demos, recorded rate unlimited: {full}/{len(demos)} three-stacks")
    for hz in args.hz:
        for max_step in args.max_joint_step:
            ok = sum(replay(sim, d, sample_indices(d["t"], hz), max_step) for d in demos)
            limit = f"{max_step:g} rad" if max_step > 0 else "none"
            print(f"{hz:>4g} Hz, joint-step limit {limit:>8}: {ok}/{len(demos)}")


if __name__ == "__main__":
    main()
