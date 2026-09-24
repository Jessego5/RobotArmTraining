"""The seven-value robot state ACT policies observe.

The six arm joints are followed by one gripper value.  Early datasets used the
gripper *command*, which cannot tell a closed grasp on a cube from fingers
closed on nothing; current datasets use the measured finger opening.  Each
checkpoint records which one it was trained on, so older checkpoints still
receive the state they expect.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ARM_JOINT_NAMES = [f"joint{i}" for i in range(1, 7)]
GRIPPER_COMMAND = "gripper_open_m"
FINGER_OPENING = "finger_opening_m"
STATE_FILE = "policy_state.json"


def write_state_names(checkpoint: Path, names: list[str]) -> None:
    (Path(checkpoint) / STATE_FILE).write_text(
        json.dumps({"state_names": list(names)}, indent=2) + "\n")


def gripper_state_name(checkpoint: Path) -> str:
    """The gripper entry a checkpoint was trained on (legacy: the command)."""
    path = Path(checkpoint) / STATE_FILE
    if not path.is_file():
        return GRIPPER_COMMAND
    name = json.loads(path.read_text())["state_names"][-1]
    if name not in (GRIPPER_COMMAND, FINGER_OPENING):
        raise ValueError(f"{path}: unknown gripper state {name!r}")
    return name


def robot_state(sim, gripper: str) -> np.ndarray:
    """Arm joints plus the chosen gripper value, as float32."""
    if gripper == FINGER_OPENING:
        value = sim.finger_opening
    else:
        value = float(sim.data.ctrl[sim.grip_act])
    return np.concatenate([sim.q, [value]]).astype(np.float32)
