#!/usr/bin/env python3
"""Score ACT checkpoints on randomized three-cube stacking episodes.

Each checkpoint runs the same seeded layouts in the RL environment
(train_act_rl.PantheraStackEnv), so results are comparable across
checkpoints.  Episodes end on success (a three-cube stack held for five
ticks) or after --episode-steps.  Work is split over --workers processes.

    python tools/eval_act.py outputs/act/sweep/checkpoint_* --episodes 30
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "sim"))

MODES = ("queue", "first", "te")
TASK = "stack the three colored cubes"


def load_policy(checkpoint: Path, mode: str, te_coeff: float, device):
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors

    config = PreTrainedConfig.from_pretrained(checkpoint)
    config.device = str(device)
    if mode == "te":
        config.n_action_steps = 1
        config.temporal_ensemble_coeff = te_coeff
    elif mode == "first":
        config.n_action_steps = 1
        config.temporal_ensemble_coeff = None
    else:  # "queue": the checkpoint's own open-loop action chunk
        config.temporal_ensemble_coeff = None
    policy = ACTPolicy.from_pretrained(checkpoint, config=config).to(device)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config, pretrained_path=str(checkpoint))
    return policy, preprocessor, postprocessor


def run_task(task: dict) -> list[dict]:
    """Run one checkpoint on a chunk of seeds; one call per worker task."""
    os.environ.setdefault("MUJOCO_GL", "egl")
    import torch
    from lerobot.policies.utils import prepare_observation_for_inference

    from act_state import gripper_state_name
    from train_act_rl import PantheraStackEnv

    torch.set_num_threads(2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = Path(task["checkpoint"])
    policy, preprocessor, postprocessor = load_policy(
        checkpoint, task["mode"], task["te_coeff"], device)
    env = PantheraStackEnv(task["seeds"][0], task["episode_steps"], task["hz"],
                           task["max_joint_step"], gripper_state_name(checkpoint))
    results = []
    try:
        for seed in task["seeds"]:
            env.rng = np.random.default_rng(seed)
            env.reset()
            policy.reset()
            while True:
                state, shoulder, wrist = env.observation()
                observation = prepare_observation_for_inference(
                    {"observation.state": state,
                     "observation.images.shoulder": shoulder,
                     "observation.images.wrist": wrist},
                    device, task=TASK, robot_type="panthera_ht_sim")
                with torch.inference_mode():
                    action = postprocessor(policy.select_action(preprocessor(observation)))
                action = np.asarray(action.detach().cpu(), dtype=np.float32).reshape(-1)
                _reward, done, _info, finished = env.step(action)
                if done:
                    break
            results.append({"checkpoint": str(checkpoint), "seed": seed, **vars(finished)})
    finally:
        env.close()
    return results


def summarize(episodes: list[dict]) -> dict:
    n = len(episodes)
    stage = np.array([e["max_stage"] for e in episodes])
    return {
        "episodes": n,
        "success": sum(e["success"] for e in episodes) / n,
        "two_stack": sum(e["two_stack"] for e in episodes) / n,
        "grasped": float((stage >= 1).mean()),
        "mean_grasps": float(np.mean([e["grasp_events"] for e in episodes])),
        "mean_steps": float(np.mean([e["length"] for e in episodes])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoints", type=Path, nargs="+")
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--first-seed", type=int, default=1000)
    parser.add_argument("--episode-steps", type=int, default=300)
    parser.add_argument("--mode", choices=MODES, default="queue",
                        help="queue: execute each predicted chunk open-loop (the checkpoint's "
                             "n_action_steps); first: re-plan every step; te: temporal ensemble")
    parser.add_argument("--te-coeff", type=float, default=0.01)
    parser.add_argument("--hz", type=float, default=10.0)
    parser.add_argument("--max-joint-step", type=float, default=0.3)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--chunk", type=int, default=5, help="episodes per worker task")
    parser.add_argument("--output", type=Path, help="write per-episode results and summary JSON")
    args = parser.parse_args()

    checkpoints = [c.resolve() for c in args.checkpoints if (c / "config.json").is_file()]
    if not checkpoints:
        raise SystemExit("no checkpoint directories (with config.json) given")
    seeds = list(range(args.first_seed, args.first_seed + args.episodes))
    tasks = [
        {"checkpoint": str(c), "seeds": seeds[i:i + args.chunk], "mode": args.mode,
         "te_coeff": args.te_coeff, "episode_steps": args.episode_steps,
         "hz": args.hz, "max_joint_step": args.max_joint_step}
        for c in checkpoints for i in range(0, len(seeds), args.chunk)
    ]
    episodes: list[dict] = []
    with mp.get_context("spawn").Pool(min(args.workers, len(tasks))) as pool:
        for done, chunk in enumerate(pool.imap_unordered(run_task, tasks), 1):
            episodes.extend(chunk)
            print(f"[{done}/{len(tasks)}] {Path(chunk[0]['checkpoint']).name}: "
                  f"{sum(e['success'] for e in chunk)}/{len(chunk)} stacked", flush=True)

    summary = {}
    print(f"\n{'checkpoint':<24}{'success':>9}{'2-stack':>9}{'grasped':>9}{'grasps':>8}{'steps':>7}")
    for c in checkpoints:
        s = summarize([e for e in episodes if e["checkpoint"] == str(c)])
        summary[str(c)] = s
        print(f"{c.name:<24}{s['success']:>9.0%}{s['two_stack']:>9.0%}{s['grasped']:>9.0%}"
              f"{s['mean_grasps']:>8.1f}{s['mean_steps']:>7.0f}")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(
            {"args": vars(args), "summary": summary, "episodes": episodes},
            indent=2, default=str) + "\n")
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
