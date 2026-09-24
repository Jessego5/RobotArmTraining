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
r       reset arm near home           e       save take
x       discard take                  q       quit
1-4     focus a view                  g       show all four views
f       toggle Minecraft-style mouse look
side-scroll   roll the gripper (right / left = + / -)
```

Recording begins automatically when a movement or gripper command is made.
Press `e` to save the current take under `data/episode_NNN/`. The default
60-second limit can be changed with `--max-duration`, and `--no-video` records
only arrays.

The arm starts at a random reachable position spanning 10–40 cm in height, and
`r` samples a new position. Use `--arm-start-range X0 X1 Y0 Y1 Z0 Z1` to tune
the distribution. Run `python teleop/keyboard.py --help` for speed, workspace,
window, view, and mouse-look options.

## Views and mouse-look

The window shows four views:

| Key | View | Purpose |
| --- | --- | --- |
| `1` | shoulder | fixed overview behind the robot |
| `2` | wrist | first-person alignment with the jaws |
| `3` | overhead | table position and depth |
| `4` | chase | overview following the wrist |

The interactive wrist view is roll-stabilized, so rolling the gripper does not
spin the operator's view. This affects only teleoperation: the wrist images
rendered for model training retain the camera's physical roll.

Press a number again or `g` to return to the grid. Drag and vertical-scroll to
orbit and zoom the movable shoulder and chase views. Horizontal side-scrolling
rolls the gripper; it also works while Minecraft-style mouse look is active.

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

## LeRobot ACT experiment

The same demonstrations can be converted to LeRobot 0.4.4 and used to train
an ACT policy locally. The converter uses both shoulder and wrist images,
seven-dimensional joint/gripper state, and the next 10 Hz joint target as the
action:

```bash
uv venv .venv-act --python 3.10
uv pip install --python .venv-act/bin/python lerobot==0.4.4 mujoco
uv pip uninstall --python .venv-act/bin/python opencv-python-headless
uv pip install --python .venv-act/bin/python opencv-python==4.12.0.88
.venv-act/bin/python teleop/build_lerobot_dataset.py
.venv-act/bin/python tools/lerobot_image_cache.py outputs/lerobot/panthera_stack
.venv-act/bin/python train_act.py
python rollout_act.py --steps 150 --no-display --no-realtime --video outputs/act/rollout.mp4
```

Run an ACT checkpoint interactively until you quit. Press `r` to randomize the
cube layout and arm start, or `q` to close the rollout:

```bash
python rollout_act.py --checkpoint outputs/act/panthera_stack_full
```

The observed gripper value is the measured finger opening, so the policy can
tell a closed grasp from fingers closed on nothing. Each checkpoint records its
state in `policy_state.json`; checkpoints without it receive the older gripper
command. `--keep-every N` keeps a separate checkpoint every N steps, and
`tools/eval_act.py` scores checkpoints on the same seeded layouts:

```bash
.venv-act/bin/python train_act.py --steps 60000 --keep-every 10000 --output outputs/act/sweep
.venv-act/bin/python tools/eval_act.py outputs/act/sweep/checkpoint_* --episodes 30
```

On a prepared GPU pod, `bash tools/pod_act_sweep.sh` rebuilds the dataset and
runs both steps.

Temporal ensembling is enabled by default to smooth transitions between ACT
action chunks. Pass `--no-temporal-ensemble` to compare against the checkpoint's
original 10-step open-loop action queue.

## RL fine-tuning ACT for three-block stacking

`train_act_rl.py` continues from the imitation checkpoint with PPO. It freezes
ACT's visual backbone and transformer, caches the first decoder feature during
rollout, and updates the shared ACT action head. A privileged state critic is
used only while training; the exported policy still consumes the same shoulder
image, wrist image, and seven-dimensional robot state as before.

The shaped reward pays for *changes* in reaching, grasping, lifting, a supported
two-cube pair, and a table-supported three-cube chain. A three-cube stack must
remain valid for five control ticks before the episode succeeds. This avoids
the common failure mode where a policy earns reward indefinitely by hovering
near a block. The last cube also has to be released. Training prints rolling
exploratory grasp/two-stack/three-stack rates, height, throughput, and PPO
diagnostics. Every 50 updates it separately evaluates the deterministic policy
over 10 episodes. It writes all stats to JSONL and saves resumable ACT
checkpoints every ten updates:

```bash
python train_act_rl.py \
  --checkpoint outputs/act/panthera_stack_full \
  --output outputs/act_rl/panthera_stack \
  --num-envs 12 --env-workers 4

# Resume the most recent checkpoint shown in latest.json.
python train_act_rl.py \
  --resume outputs/act_rl/panthera_stack/checkpoint_000100 \
  --output outputs/act_rl/panthera_stack \
  --num-envs 12 --env-workers 4

# RL checkpoints use first-action receding-horizon inference by default.
python rollout_act.py \
  --checkpoint outputs/act_rl/panthera_stack/checkpoint_000100
```

`--env-workers` runs independent MuJoCo/EGL worker processes and exchanges
camera frames, actions, rewards, and critic state through shared memory. Four
workers is the default; each owns an even share of `--num-envs`. Pass
`--env-workers 1` for the original serial loop. Other useful overrides are
`--rollout-steps`, `--episode-steps`, `--checkpoint-freq`, and `--updates`.

The simulator backend is MuJoCo, not MJX: the task's compliant-pad grasp assist
toggles equality constraints from contact state, and both ACT cameras must
still be rendered by MuJoCo. Moving physics alone to MJX would not preserve
those task dynamics or remove the rendering bottleneck.

`tools/benchmark_act_batch.py` measures both GPU-only and end-to-end training
throughput. On the RTX 4060 Laptop GPU used for this experiment, batch 12 was
the fastest sustained end-to-end size; larger batches used the GPU more
efficiently but lost that gain while decoding the embedded camera images.
The optional decoded-image cache is about 10 GiB for this dataset. It preserves
the exact RGB pixels, is memory-mapped rather than loaded into RAM, and lets
training bypass PNG decoding and the large embedded Parquet image columns.
Training discovers the default cache automatically; pass `--no-image-cache`
to compare against the original path.

## VLA-Adapter on a Linux GPU machine

`tools/train_vla_pod.py` runs the Colab notebook's pipeline without Colab:
it creates a Python 3.10 environment, applies the notebook's VLA-Adapter
patches, renders and converts the demonstrations, downloads the base model,
and fine-tunes. Each stage is idempotent, so rerunning resumes setup, and
`--resume-run-id` continues training from the last checkpoint:

```bash
python3 tools/train_vla_pod.py --steps 50       # end-to-end smoke test
python3 tools/train_vla_pod.py --steps 20000
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
