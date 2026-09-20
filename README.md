# RobotArmLearning

Keyboard teleoperation and demonstration recording for a simulated HighTorque
Panthera-HT 6-DoF arm in MuJoCo.

## Setup

```bash
pip install -r requirements.txt
python teleop/keyboard.py
```

The keyboard drives a commanded end-effector pose. In the default world-frame
mode, movement is relative to the robot base:

```text
w / s   forward / back (+x / -x)      u / j   pitch
a / d   left / right   (+y / -y)      i / k   yaw
SPACE   up                             n / m   roll
SHIFT   down                           o / l   gripper open / close
[ / ]   slower / faster               c       re-centre target
r       reset arm                     e       save take
x       discard take                  q       quit
1-4     focus a view                  g       show all four views
f       toggle Minecraft-style mouse look
```

Recording begins automatically when a movement or gripper command is made.
Press `e` to save the current take under `data/episode_NNN/`. The default
60-second limit can be changed with `--max-duration`, and `--no-video` records
only arrays.

Run `python teleop/keyboard.py --help` for speed, workspace, window, view, and
mouse-look options.

## Views and mouse-look

The window shows four views:

| Key | View | Purpose |
| --- | --- | --- |
| `1` | shoulder | fixed overview behind the robot |
| `2` | wrist | first-person alignment with the jaws |
| `3` | overhead | table position and depth |
| `4` | chase | overview following the wrist |

Press a number again or `g` to return to the grid. Drag and scroll orbit the
movable shoulder and chase views.

With `f` or `--minecraft`, the mouse aims the gripper and WASD moves on its
level heading. SPACE and SHIFT remain world-up and world-down; mouse buttons
close and open the gripper. ESC releases the cursor before quitting.

## Episode format

Each saved episode contains:

- `data.npz`: time, joint state, controls, achieved and target end-effector
  poses, object poses, gripper command, and IK residuals.
- `meta.json`: robot, scene, recording settings, object-column names, and
  frame conventions.
- `sim.mp4`: the displayed view or four-view mosaic, unless `--no-video` was
  used.

Replay a take:

```bash
python teleop/replay.py data/episode_000
```

Render several episodes into a grid video:

```bash
python teleop/grid_replay.py
```

Render shoulder/wrist observations for the VLA dataset pipeline:

```bash
python teleop/render_vla_dataset.py
```

## Model rollout

`rollout.py` runs the LoRA VLA-Adapter checkpoint produced by the training
notebook in the same two-camera MuJoCo environment. It accepts an extracted
checkpoint, the notebook's `.tar.gz`, or a Google Drive `.zip`; with no
`--checkpoint` it uses the newest `robot-arm-learning*` artifact in
`~/Downloads`. Archive extraction omits the optimizer state, which is not
needed for inference.

The script automatically restarts itself in the project's existing
`~/venvs/vla-adapter` environment, so it can be launched with plain `python`.
Install the simulation dependencies into that environment once:

```bash
~/venvs/vla-adapter/bin/python -m pip install -r requirements.txt
```

Then run a real-time rollout. It continues until you press `q`; press `r` to
reset the arm and start with a newly randomized cube layout:

```bash
python rollout.py
```

The first run downloads the 2.5 GiB base model and may also populate the
Hugging Face cache with its Qwen/DINO/SigLIP backbones. Useful options:

```bash
# Save a headless 20-second rollout using a particular artifact.
python rollout.py \
  --checkpoint ~/Downloads/robot-arm-learning-colab-*.zip \
  --steps 200 --no-display --video rollouts/test.mp4

# Re-query every control step instead of executing all 8 predicted actions.
python rollout.py --open-loop 1

# Run a finite 60-second interactive rollout instead of running indefinitely.
python rollout.py --steps 600
```

## Simulation

The model has six revolute joints, a parallel gripper, a table, and three free
cubes. `PantheraSim` provides reset, stepping, object-pose access, and damped
least-squares IK with joint-limit handling, a home-posture nullspace bias, and
a per-call joint-motion limit.

The commanded target is allowed to enter unreachable parts of its configured
box so failures remain visible: the orange target separates from the gripper
and the HUD reports the IK residual. Press `c` to place the target back on the
arm.

Model utilities:

```bash
python sim/prep_meshes.py
python sim/make_mjcf.py
python sim/panthera_env.py
python -m mujoco.viewer --mjcf sim/panthera/scene.xml
```

## Layout

```text
sim/      prep_meshes.py  make_mjcf.py  panthera_env.py  panthera/
teleop/   keyboard.py  episode.py  replay.py  grid_replay.py
          render_vla_dataset.py
data/     episode_NNN/{data.npz, meta.json, sim.mp4}
```
