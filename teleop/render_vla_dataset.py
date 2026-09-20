"""Render recorded RobotArmLearning episodes for two-camera VLA training.

The exported observation pair is deliberately the same pair used during
teleoperation: the fixed over-the-shoulder view (``shoulder``) and the camera
mounted on link6 (``wrist``).  Frames are sampled by wall-clock time rather
than by row number because the recorder's frame rate varies slightly.

This is the rendering half of the pipeline.  The companion
``VLA-Adapter/scripts/build_robot_arm_learning_rlds.py`` turns this directory into a
TFDS/RLDS dataset.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "sim"))
os.environ.setdefault("MUJOCO_GL", "egl")

import cv2  # noqa: E402
import mujoco  # noqa: E402

from panthera_env import PantheraSim  # noqa: E402

DEFAULT_INPUT = REPO_ROOT / "data"
DEFAULT_OUTPUT = REPO_ROOT / "VLA-Adapter" / "data" / "robot_arm_learning_rendered"
SHOULDER_AZIMUTH = 14.0
SHOULDER_ELEVATION = -34.0
SHOULDER_DISTANCE = 1.20
SHOULDER_LOOKAT = (0.44, 0.0, 0.12)


def shoulder_camera(model: mujoco.MjModel) -> mujoco.MjvCamera:
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, cam)
    cam.azimuth = SHOULDER_AZIMUTH
    cam.elevation = SHOULDER_ELEVATION
    cam.distance = SHOULDER_DISTANCE
    cam.lookat[:] = SHOULDER_LOOKAT
    return cam


def wrist_camera(model: mujoco.MjModel) -> mujoco.MjvCamera:
    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist")
    if camera_id < 0:
        raise SystemExit("scene has no 'wrist' camera; run `python sim/make_mjcf.py`")
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
    cam.fixedcamid = camera_id
    return cam


def sample_indices(t: np.ndarray, hz: float) -> np.ndarray:
    """Nearest recorded sample at or before each uniform sample time."""
    relative = np.asarray(t, dtype=np.float64) - float(t[0])
    sample_t = np.arange(0.0, relative[-1] + 1e-9, 1.0 / hz)
    indices = np.searchsorted(relative, sample_t, side="right") - 1
    indices = np.clip(indices, 0, len(relative) - 1)
    return np.unique(indices)


def render_episode(path: Path, output: Path, sim: PantheraSim,
                   renderers: tuple[mujoco.Renderer, mujoco.Renderer],
                   cameras: tuple[mujoco.MjvCamera, mujoco.MjvCamera],
                   hz: float, quality: int) -> int:
    with np.load(path / "data.npz") as source:
        required = {"t", "q", "ctrl", "ee_pos", "ee_quat", "obj_pos", "obj_quat", "gripper"}
        missing = required - set(source.files)
        if missing:
            raise ValueError(f"{path}: missing fields {sorted(missing)}")
        indices = sample_indices(source["t"], hz)
        arrays = {key: np.asarray(source[key])[indices].copy() for key in required}

    # The final row has no demonstrated successor and therefore no action.
    if len(indices) < 2:
        raise ValueError(f"{path}: fewer than two samples at {hz:g} Hz")

    temp = Path(tempfile.mkdtemp(prefix=f".{path.name}-", dir=output.parent))
    try:
        (temp / "shoulder").mkdir()
        (temp / "wrist").mkdir()
        for frame_i in range(len(indices)):
            sim.data.qpos[:] = sim.model.qpos0
            sim.data.qvel[:] = 0.0
            sim.data.ctrl[:] = 0.0
            sim.data.qpos[sim.arm_qadr] = arrays["q"][frame_i]
            nctrl = min(sim.model.nu, arrays["ctrl"].shape[1])
            sim.data.ctrl[:nctrl] = arrays["ctrl"][frame_i, :nctrl]
            sim.set_object_poses(arrays["obj_pos"][frame_i], arrays["obj_quat"][frame_i])
            mujoco.mj_forward(sim.model, sim.data)

            for name, renderer, camera in zip(("shoulder", "wrist"), renderers, cameras):
                renderer.update_scene(sim.data, camera)
                bgr = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
                destination = temp / name / f"{frame_i:05d}.jpg"
                if not cv2.imwrite(str(destination), bgr,
                                   [cv2.IMWRITE_JPEG_QUALITY, quality]):
                    raise OSError(f"could not write {destination}")

        np.savez_compressed(
            temp / "trajectory.npz",
            source_indices=indices,
            sample_hz=np.float32(hz),
            q=arrays["q"].astype(np.float32),
            ctrl=arrays["ctrl"].astype(np.float32),
            ee_pos=arrays["ee_pos"].astype(np.float32),
            ee_quat=arrays["ee_quat"].astype(np.float32),
            gripper=arrays["gripper"].astype(np.float32),
        )
        (temp / "source.json").write_text(json.dumps({
            "episode": path.name,
            "source": str(path.resolve()),
            "frames": int(len(indices)),
            "sample_hz": hz,
            "cameras": ["shoulder", "wrist"],
        }, indent=2) + "\n")
        if output.exists():
            shutil.rmtree(output)
        temp.rename(output)
    except BaseException:
        shutil.rmtree(temp, ignore_errors=True)
        raise
    return len(indices)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--hz", type=float, default=10.0)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--jpeg-quality", type=int, default=92)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    episodes = sorted(p for p in args.input.glob("episode_*")
                      if (p / "data.npz").is_file())
    if not episodes:
        raise SystemExit(f"no episodes found in {args.input}")
    args.output.mkdir(parents=True, exist_ok=True)

    sim = PantheraSim()
    renderers = (
        mujoco.Renderer(sim.model, height=args.size, width=args.size),
        mujoco.Renderer(sim.model, height=args.size, width=args.size),
    )
    cameras = (shoulder_camera(sim.model), wrist_camera(sim.model))
    total = 0
    try:
        for number, episode in enumerate(episodes, 1):
            destination = args.output / episode.name
            complete = (destination / "trajectory.npz").is_file()
            if complete and not args.force:
                with np.load(destination / "trajectory.npz") as cached:
                    count = len(cached["q"])
                print(f"[{number:03d}/{len(episodes):03d}] {episode.name}: cached ({count} frames)", flush=True)
            else:
                count = render_episode(episode, destination, sim, renderers,
                                       cameras, args.hz, args.jpeg_quality)
                print(f"[{number:03d}/{len(episodes):03d}] {episode.name}: {count} frames", flush=True)
            total += count
    finally:
        for renderer in renderers:
            renderer.close()

    manifest = {
        "format": "robot-arm-learning-vla-render-v1",
        "episodes": [p.name for p in episodes],
        "num_episodes": len(episodes),
        "num_frames": total,
        "sample_hz": args.hz,
        "image_size": [args.size, args.size],
        "cameras": {"primary": "shoulder", "wrist": "wrist"},
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"rendered {len(episodes)} episodes / {total} observations to {args.output}")


if __name__ == "__main__":
    main()
