"""Convert rendered Panthera demonstrations to a local LeRobot dataset.

The ACT policy observes shoulder and wrist RGB images plus the six arm joint
positions and the measured finger opening (``--gripper-state command`` keeps
the older commanded opening).  Its action is the *next* sampled joint target
and gripper command.  The one-sample shift matters because each source row was
recorded after applying that row's control; using the unshifted control would
teach a nearly identity state-to-action mapping.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from lerobot.datasets.lerobot_dataset import LeRobotDataset


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "sim"))

from act_state import ARM_JOINT_NAMES, FINGER_OPENING, GRIPPER_COMMAND  # noqa: E402

DEFAULT_INPUT = REPO_ROOT / "VLA-Adapter" / "data" / "robot_arm_learning_rendered"
DEFAULT_OUTPUT = REPO_ROOT / "outputs" / "lerobot" / "panthera_stack"
ACTION_NAMES = ARM_JOINT_NAMES + [GRIPPER_COMMAND]


def features(height: int, width: int, state_names: list[str]) -> dict:
    return {
        "observation.state": {
            "dtype": "float32",
            "shape": (7,),
            "names": state_names,
        },
        "observation.images.shoulder": {
            "dtype": "image",
            "shape": (height, width, 3),
            "names": ["height", "width", "channels"],
        },
        "observation.images.wrist": {
            "dtype": "image",
            "shape": (height, width, 3),
            "names": ["height", "width", "channels"],
        },
        "action": {
            "dtype": "float32",
            "shape": (7,),
            "names": ACTION_NAMES,
        },
    }


def episode_paths(root: Path) -> list[Path]:
    return sorted(
        path for path in root.glob("episode_*")
        if (path / "trajectory.npz").is_file()
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--repo-id", default="local/panthera_stack")
    parser.add_argument("--task", default="stack the three colored cubes")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--gripper-state", choices=("measured", "command"), default="measured",
                        help="observe the measured finger opening or the gripper command")
    args = parser.parse_args()
    measured = args.gripper_state == "measured"
    state_names = ARM_JOINT_NAMES + [FINGER_OPENING if measured else GRIPPER_COMMAND]

    manifest = json.loads((args.input / "manifest.json").read_text())
    fps = int(round(float(manifest["sample_hz"])))
    height, width = map(int, manifest["image_size"])
    episodes = episode_paths(args.input)
    if len(episodes) != int(manifest["num_episodes"]):
        raise SystemExit(
            f"manifest says {manifest['num_episodes']} episodes but found {len(episodes)}"
        )
    if args.output.exists():
        if not args.force:
            raise SystemExit(f"output already exists: {args.output} (use --force to rebuild)")
        shutil.rmtree(args.output)

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        root=args.output,
        fps=fps,
        robot_type="panthera_ht_sim",
        features=features(height, width, state_names),
        use_videos=False,
        image_writer_threads=8,
        metadata_buffer_size=20,
    )
    try:
        for ep_i, episode in enumerate(episodes):
            with np.load(episode / "trajectory.npz") as data:
                q = np.asarray(data["q"], dtype=np.float32)
                ctrl = np.asarray(data["ctrl"], dtype=np.float32)
                if measured and "finger_opening" not in data.files:
                    raise SystemExit(f"{episode}: no finger_opening; re-render with "
                                     "teleop/render_vla_dataset.py")
                gripper = (np.asarray(data["finger_opening"], dtype=np.float32)
                           if measured else ctrl[:, 6])
            if len(q) < 2 or ctrl.shape != (len(q), 7):
                raise ValueError(f"{episode}: unexpected q/ctrl shapes {q.shape}/{ctrl.shape}")

            state = np.concatenate([q, gripper[:, None]], axis=1)
            action = np.concatenate([ctrl[1:], ctrl[-1:]], axis=0)
            for frame_i in range(len(q)):
                shoulder_path = episode / "shoulder" / f"{frame_i:05d}.jpg"
                wrist_path = episode / "wrist" / f"{frame_i:05d}.jpg"
                if not shoulder_path.is_file() or not wrist_path.is_file():
                    raise FileNotFoundError(f"missing rendered images in {episode}")
                with Image.open(shoulder_path) as image:
                    shoulder = np.asarray(image.convert("RGB"))
                with Image.open(wrist_path) as image:
                    wrist = np.asarray(image.convert("RGB"))
                dataset.add_frame({
                    "observation.state": state[frame_i],
                    "observation.images.shoulder": shoulder,
                    "observation.images.wrist": wrist,
                    "action": action[frame_i],
                    "task": args.task,
                })
            dataset.save_episode()
            print(
                f"[{ep_i + 1:03d}/{len(episodes):03d}] {episode.name}: {len(q)} frames",
                flush=True,
            )
    finally:
        dataset.finalize()
        dataset.stop_image_writer()

    print(f"wrote {len(episodes)} episodes / {manifest['num_frames']} frames to {args.output}")


if __name__ == "__main__":
    main()
