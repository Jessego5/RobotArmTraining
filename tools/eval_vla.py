#!/usr/bin/env python3
"""Score a VLA-Adapter checkpoint on randomized three-cube stacking episodes.

Uses the same environment, seeds, and success test as tools/eval_act.py, so
VLA-Adapter and ACT numbers are directly comparable.  Model loading and
prediction come from rollout.py; each predicted end-effector delta is
integrated into a commanded pose, solved by IK, and executed as a joint
target with the environment's per-step joint limit.  Run it with the VLA
environment's Python:

    /workspace/vla-env/bin/python tools/eval_vla.py \\
        --checkpoint outputs/vla/<run_id> --episodes 30
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(REPO_ROOT), str(REPO_ROOT / "sim"), str(REPO_ROOT / "teleop")]

import rollout  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--vla-dir", type=Path, default=rollout.default_vla_dir())
    parser.add_argument("--cache-dir", type=Path,
                        default=Path.home() / ".cache" / "robot-arm-learning")
    parser.add_argument("--base-model", type=Path)
    parser.add_argument("--instruction", default="stack the three colored cubes")
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--first-seed", type=int, default=1000)
    parser.add_argument("--episode-steps", type=int, default=300)
    parser.add_argument("--open-loop", type=int, default=8, choices=range(1, 9), metavar="1..8")
    parser.add_argument("--hz", type=float, default=10.0)
    parser.add_argument("--max-joint-step", type=float, default=0.3)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    os.environ.setdefault("MUJOCO_GL", "egl")
    import mujoco

    from panthera_env import GRIPPER_OPEN
    from train_act_rl import PantheraStackEnv

    vla_dir = rollout.ensure_vla_checkout(args.vla_dir)
    rollout.require_ml_dependencies(vla_dir)
    checkpoint = rollout.materialize_checkpoint(args.checkpoint, args.cache_dir.expanduser())
    local_base = vla_dir / "pretrained_models" / "prism-qwen25-extra-dinosiglip-224px-0_5b"
    base_override = args.base_model
    if base_override is None and (local_base / rollout.BASE_CHECKPOINT).is_file():
        base_override = local_base
    base_dir = rollout.ensure_base_model(args.cache_dir.expanduser(), base_override)
    policy = rollout.load_policy(vla_dir, base_dir, checkpoint, args.device)

    env = PantheraStackEnv(args.first_seed, args.episode_steps, args.hz, args.max_joint_step)
    episodes = []
    try:
        for seed in range(args.first_seed, args.first_seed + args.episodes):
            env.rng = np.random.default_rng(seed)
            env.reset()
            sim = env.sim
            target_pos, target_quat = sim.ee_pose()
            queue: list[np.ndarray] = []
            while True:
                _state, shoulder, wrist = env.observation()
                if not queue:
                    chunk = rollout.predict_actions(
                        policy, shoulder, wrist, rollout.current_state(sim), args.instruction)
                    queue.extend(chunk[:args.open_loop])
                action = np.asarray(queue.pop(0), dtype=np.float64)
                target_pos = target_pos + action[:3]
                next_quat = np.zeros(4)
                mujoco.mju_mulQuat(next_quat, rollout.rotvec_quat(action[3:6]), target_quat)
                mujoco.mju_normalize4(next_quat)
                target_quat = next_quat
                q_target, _, _ = sim.ik(target_pos, target_quat, q_init=sim.q,
                                        max_joint_step=None)
                gripper_m = float(np.clip(action[6], 0.0, 1.0)) * GRIPPER_OPEN
                _reward, done, _info, finished = env.step(np.append(q_target, gripper_m))
                if done:
                    break
            episodes.append({"seed": seed, **vars(finished)})
            print(f"seed {seed}: stage {finished.max_stage}, "
                  f"{'stacked' if finished.success else 'not stacked'}", flush=True)
    finally:
        env.close()

    n = len(episodes)
    summary = {
        "episodes": n,
        "success": sum(e["success"] for e in episodes) / n,
        "two_stack": sum(e["two_stack"] for e in episodes) / n,
        "grasped": sum(e["max_stage"] >= 1 for e in episodes) / n,
        "mean_grasps": float(np.mean([e["grasp_events"] for e in episodes])),
    }
    print(f"\n{'checkpoint':<24}{'success':>9}{'2-stack':>9}{'grasped':>9}{'grasps':>8}")
    print(f"{args.checkpoint.name[:23]:<24}{summary['success']:>9.0%}{summary['two_stack']:>9.0%}"
          f"{summary['grasped']:>9.0%}{summary['mean_grasps']:>8.1f}")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(
            {"args": vars(args), "summary": summary, "episodes": episodes},
            indent=2, default=str) + "\n")


if __name__ == "__main__":
    main()
