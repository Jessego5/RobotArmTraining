#!/usr/bin/env python3
"""Roll out a trained LeRobot ACT checkpoint in the Panthera simulation."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from sim.stack_task import stack_metrics as task_stack_metrics


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = REPO_ROOT / "outputs" / "act" / "panthera_stack"
DEFAULT_DATASET = REPO_ROOT / "outputs" / "lerobot" / "panthera_stack"


def reexec_in_act_venv() -> None:
    python = REPO_ROOT / ".venv-act" / "bin" / "python"
    if not python.is_file() or Path(sys.prefix).resolve() == python.parents[1].resolve():
        return
    if os.environ.get("ROBOT_ARM_ACT_REEXEC") == "1":
        return
    environment = os.environ.copy()
    environment["ROBOT_ARM_ACT_REEXEC"] = "1"
    os.execve(str(python), [str(python), str(Path(__file__).resolve()), *sys.argv[1:]], environment)


def stack_metrics(positions: np.ndarray) -> dict:
    metrics = task_stack_metrics(positions)
    return {
        "object_positions": positions.tolist(),
        "horizontal_spread_m": metrics["horizontal_spread_m"],
        "vertical_gaps_m": metrics["vertical_gaps_m"],
        "max_object_height_m": metrics["max_height_m"],
        "two_stacked": metrics["two_stack"],
        "stacked": metrics["three_stack"],
    }


def main() -> None:
    reexec_in_act_venv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--steps",
        type=int,
        default=0,
        help="control steps before stopping; 0 (default) runs until q",
    )
    parser.add_argument("--hz", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-joint-step", type=float, default=0.3)
    parser.add_argument(
        "--temporal-ensemble",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=("blend overlapping ACT chunks; defaults on for imitation checkpoints "
              "and off for first-action RL checkpoints"),
    )
    parser.add_argument(
        "--temporal-ensemble-coeff",
        type=float,
        default=0.01,
        help="exponential weight coefficient for temporal ensembling (default: 0.01)",
    )
    parser.add_argument("--video", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--no-realtime", action="store_true")
    parser.add_argument("--mujoco-gl", default="egl", choices=("egl", "glfw", "osmesa"))
    args = parser.parse_args()
    if args.steps < 0 or args.hz <= 0 or args.max_joint_step <= 0:
        parser.error("--steps must be nonnegative; --hz and --max-joint-step must be positive")
    if args.temporal_ensemble_coeff < 0:
        parser.error("--temporal-ensemble-coeff must be nonnegative")
    if args.no_display and args.steps == 0:
        parser.error("--no-display requires a positive --steps value")

    os.environ.setdefault("MUJOCO_GL", args.mujoco_gl)
    import cv2
    import mujoco
    import torch

    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.utils import prepare_observation_for_inference

    sys.path.insert(0, str(REPO_ROOT / "sim"))
    sys.path.insert(0, str(REPO_ROOT / "teleop"))
    from act_state import gripper_state_name, robot_state
    from keyboard import DEFAULT_ARM_START_RANGE, randomize_arm_start
    from panthera_env import PantheraSim
    from render_vla_dataset import shoulder_camera, wrist_camera

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = args.checkpoint.expanduser().resolve()
    is_rl_checkpoint = (checkpoint / "rl_state.pt").is_file()
    temporal_ensemble = (
        not is_rl_checkpoint if args.temporal_ensemble is None else args.temporal_ensemble
    )
    policy_config = PreTrainedConfig.from_pretrained(checkpoint)
    if temporal_ensemble:
        # Query a new chunk every step and blend its overlapping predictions.
        # Without this, ACT executes n_action_steps open-loop and can jump when
        # it replaces the exhausted action queue with an independently
        # predicted chunk.
        policy_config.n_action_steps = 1
        policy_config.temporal_ensemble_coeff = args.temporal_ensemble_coeff
    elif is_rl_checkpoint:
        # PPO trains the first decoder query and requests a new action each tick.
        policy_config.n_action_steps = 1
        policy_config.temporal_ensemble_coeff = None
    policy = ACTPolicy.from_pretrained(checkpoint, config=policy_config).to(device)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config, pretrained_path=str(checkpoint)
    )
    metadata = LeRobotDatasetMetadata("local/panthera_stack", root=args.dataset)
    gripper_state = gripper_state_name(checkpoint)

    sim = PantheraSim()

    def reset_scene(seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        sim.reset(randomize=True, rng=rng)
        randomize_arm_start(sim, DEFAULT_ARM_START_RANGE, rng=rng)
        policy.reset()
        positions, _ = sim.object_poses()
        return positions

    initial_positions = reset_scene(args.seed)
    renderers = (
        mujoco.Renderer(sim.model, height=256, width=256),
        mujoco.Renderer(sim.model, height=256, width=256),
    )
    cameras = (shoulder_camera(sim.model), wrist_camera(sim.model))
    physics_steps = max(1, round((1.0 / args.hz) / sim.dt))
    writer = None
    if args.video:
        args.video.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(args.video), cv2.VideoWriter_fourcc(*"mp4v"), args.hz, (512, 256)
        )
        if not writer.isOpened():
            raise SystemExit(f"could not open video output {args.video}")

    started = time.monotonic()
    step = 0
    total_steps = 0
    reset_count = 0
    current_seed = args.seed
    limit = f"{args.steps} control steps" if args.steps else "until you quit"
    inference_mode = (
        f"temporal ensemble {args.temporal_ensemble_coeff:g}"
        if temporal_ensemble
        else f"{policy.config.n_action_steps}-step action queue"
    )
    print(
        f"Rolling out ACT {limit} at {args.hz:g} Hz "
        f"(seed {args.seed}; {inference_mode}; r resets, q quits)",
        flush=True,
    )
    try:
        while not args.steps or total_steps < args.steps:
            tick = time.monotonic()
            images = []
            for renderer, camera in zip(renderers, cameras):
                renderer.update_scene(sim.data, camera)
                images.append(renderer.render().copy())
            state = robot_state(sim, gripper_state)
            observation = prepare_observation_for_inference(
                {
                    "observation.state": state,
                    "observation.images.shoulder": images[0],
                    "observation.images.wrist": images[1],
                },
                device,
                task="stack the three colored cubes",
                robot_type="panthera_ht_sim",
            )
            observation = preprocessor(observation)
            with torch.inference_mode():
                action = postprocessor(policy.select_action(observation))
            action = np.asarray(action.detach().cpu(), dtype=np.float64).reshape(-1)
            if action.shape != (7,) or not np.isfinite(action).all():
                raise RuntimeError(f"invalid ACT action: {action}")
            q_target = np.clip(action[:6], sim.arm_range[:, 0], sim.arm_range[:, 1])
            q_target = np.clip(
                q_target,
                sim.q - args.max_joint_step,
                sim.q + args.max_joint_step,
            )
            sim.set_arm_ctrl(q_target)
            sim.set_gripper(float(np.clip(action[6] / 0.04, 0.0, 1.0)))
            sim.step(physics_steps)

            frame = cv2.cvtColor(np.concatenate(images, axis=1), cv2.COLOR_RGB2BGR)
            cv2.rectangle(frame, (0, 0), (512, 43), (0, 0, 0), -1)
            cv2.putText(
                frame,
                f"ACT step {step:04d}  grip {action[6]:.3f}m",
                (8, 17),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                "[r] random reset  [q] quit",
                (8, 36),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (90, 240, 90),
                1,
                cv2.LINE_AA,
            )
            if writer is not None:
                writer.write(frame)
            key = -1
            if not args.no_display:
                cv2.imshow("Panthera ACT rollout - shoulder | wrist", frame)
                key = cv2.waitKey(1) & 0xFF
            total_steps += 1
            if key == ord("q"):
                break
            if key == ord("r"):
                reset_count += 1
                current_seed = args.seed + reset_count
                initial_positions = reset_scene(current_seed)
                step = 0
                print(f"Simulation reset (seed {current_seed})", flush=True)
                continue
            if not args.no_realtime:
                remaining = 1.0 / args.hz - (time.monotonic() - tick)
                if remaining > 0:
                    time.sleep(remaining)
            if step % 10 == 0:
                print(
                    f"step={step:03d} q={sim.q.round(2)} grip={action[6]:.3f}",
                    flush=True,
                )
            step += 1
    finally:
        if writer is not None:
            writer.release()
        for renderer in renderers:
            renderer.close()
        if not args.no_display:
            cv2.destroyAllWindows()

    final_positions, _ = sim.object_poses()
    report = {
        "seed": args.seed,
        "final_seed": current_seed,
        "resets": reset_count,
        "steps": total_steps,
        "elapsed_seconds": time.monotonic() - started,
        "initial": stack_metrics(initial_positions),
        "final": stack_metrics(final_positions),
    }
    print(json.dumps(report, indent=2), flush=True)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
