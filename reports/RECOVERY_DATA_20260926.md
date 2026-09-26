# Recovery data for the scripted collector — 26 September 2026

Two opt-in additions to `tools/collect_scripted.py` target the failures in
`ACT_INVESTIGATION_20260923.md` and `ACT_IK_ROOT_CAUSE_20260923.md`: timing-dependent
labels, and no demonstrations of recovering from a missed grasp. Defaults are
unchanged: with neither flag, five regenerated episodes are **byte-identical** to
the current collector's output in every recorded array and stage record.

## 1. State-driven waits (`--waits state`)

Every move ended with nine fixed settle ticks, the episode began with a 0.3 s hold,
and zero-length moves still lasted at least 0.4 s. While the arm is motionless the
label switches from "hold" to "move" at an arbitrary tick, so near-identical
observations carry different futures. With `--waits state` a move ends once the grip
site is within 3 mm of its target and arm and finger velocities are small (capped at
1 s), and zero-length moves are skipped. The gripper ramps themselves are unchanged;
their progress is observable through the commanded opening in the state.

Measured on the same five seeds (230923+), 30 Hz frames, grouping states within
0.2 mrad / 0.2 mm and counting groups whose next 30 actions differ by more than
0.02 rad:

| | Timed (current) | State-driven |
| --- | ---: | ---: |
| Frames in contradictory groups | 23.7% | 0.72% |
| Worst future disagreement | 1.02 rad | 0.62 rad |
| Frames with a motionless arm | 1,160 / 3,974 | 313 / 2,923 |
| Episode length | 24–33 s | 17–27 s |

The six rejected seeds in this range fail identically under both modes (tracking
errors on hard layouts), so acceptance is unaffected.

## 2. Recovery augmentation (`--augment-fraction`)

A per-seed draw selects that share of episodes. In them:

- A smooth Ornstein–Uhlenbeck offset (default 8 mm horizontal, 3.2 mm vertical,
  4° yaw, 0.5 s correlation) is added to the **executed** target. It fades during
  place, release and retreat so accepted stacks still meet the 12 mm alignment rule.
- With probability 0.4, a block's first grasp attempt is shifted 15–25 mm sideways.
- A failed grasp re-opens, backs off and re-plans from the cube's current pose,
  up to three attempts.
- Each row records `ctrl_label`: the clean plan solved by the same IK from the same
  controller state. Arm labels therefore steer back toward the plan. The gripper
  label only closes (or releases) once the grip site is within 15 mm horizontally
  of the planned point; re-opening after a miss is labelled as opening.
- The end-of-move tracking check allows for the noise (12 mm + 3 × noise std).

On 20 accepted episodes with `--waits state --augment-fraction 1`:

| Measure | Value |
| --- | ---: |
| Acceptance | 20/27 (74%; the current collector reported 71.3%) |
| Episodes containing a missed grasp and re-grasp | 7/20 |
| Contradictory frames (as above, labels as futures) | 0.66% |
| Ticks where the gripper label withholds the executed jaw motion | 3.7–4.5% |
| Arm label vs executed command, median / p95 | 0.06 / 0.12 rad |

In clean episodes `ctrl_label` equals `ctrl` exactly. A rendered re-grasp episode was
inspected frame by frame: the shifted first close shuts beside the cube while its
label stays open, the re-open is labelled open, and the second approach centres the
cube and grasps it.

## Export and validation

- `tools/export_scripted_dataset.py` exports `ctrl_label` as the action when present,
  resampled on the rendered 30 Hz grid; states remain the executed ones. Provenance
  records `action_source`. A mixed three-episode export confirmed actions equal the
  labels, and equal the executed controls for the clean episode.
- `tools/validate_scripted_dataset.py` reports augmented episodes separately and does
  not fail on them. Open-loop replay is not bit-exact even for clean demos (joints
  differ from the first tick, cubes drift more than 1 mm during the first descent);
  near a deliberate miss that drift can change the outcome. 15–17 of 20 augmented
  episodes replayed to a stable stack. Clean episodes must still all pass.
- The renderer is unchanged, so existing rendered datasets stay valid.

## π0.5 image augmentation

`IMAGE_AUGMENTATION` in `tools/build_pi05_notebook.py` (default `False`) enables the
pinned LeRobot's training-time image transforms without its `affine` entry:
brightness, contrast, saturation, hue and sharpness, up to three per frame. A random
shift or rotation of the gripper-mounted wrist view would change the apparent
cube-to-jaw offset while the action label stays fixed. The transform list is passed
as JSON because the parser does not accept nested dictionary keys; the equivalent
arguments were checked with LeRobot 0.4.4's parser and transform code. The notebook
regenerates byte-identically when the flag is off (with the local `release.json`).

## Tests

Two tests were added to `tests/test_scripted_collection.py`: state waits remove the
pauses without changing clean labels, and an augmented re-grasp episode labels the
shifted close as open and the re-open as open. Of the full suite, 47 pass and 1 is
skipped; the one failure needs `gdown`, which the local environment lacks.

## Not yet done

- No policy has been trained on these data; the effect on success is untested.
- `tools/publish_pi05_dataset.py` asserts the published dataset's exact counts and an
  audit that compares actions with executed controls. It needs generalizing before a
  labelled dataset can be released.
- VLA-Adapter's RLDS builder derives actions from executed end-effector motion and
  would ignore the labels.

## Usage

```bash
OPENBLAS_NUM_THREADS=1 .venv-act/bin/python tools/collect_scripted.py \
  --output data/scripted_stack_recovery --episodes 1000 --workers 6 \
  --waits state --augment-fraction 0.5
OPENBLAS_NUM_THREADS=1 .venv-act/bin/python tools/validate_scripted_dataset.py \
  --input data/scripted_stack_recovery
OPENBLAS_NUM_THREADS=1 .venv-act/bin/python tools/export_scripted_dataset.py \
  --input data/scripted_stack_recovery \
  --rendered outputs/scripted_stack_recovery_rendered_30hz \
  --output outputs/lerobot/panthera_scripted_stack_recovery_30hz --workers 6
```
