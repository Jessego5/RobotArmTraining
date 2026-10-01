# Expert-in-the-loop ACT collection (DAgger) — 1 October 2026

Adapts human-in-the-loop correction rounds (DAgger; see Aurel Arnold, "Human-in-the-loop
data collection", where interventions took a real-robot ACT policy from 25% to 72% in two
rounds while extra demonstrations plateaued at 38%) to this simulator, with the scripted
IK planner as the intervening expert.

The investigation reports identify the gap this targets: the policy's own positioning
errors take it into states the demonstrations never visit (missed alignment, closing
early), and no demonstration shows a recovery from them. Corrections recorded from the
states the policy actually reaches supply exactly that.

No policy has been trained with these tools yet; this report covers the pipeline and its
validation, not a success-rate result.

## How a round works

`tools/dagger_rounds.py` runs, resumably, for each round r:

1. Export the contact-v2 base demonstrations plus all corrections so far
   (`export_scripted_dataset.py`) and train ACT from scratch (`train_act.py`).
2. Evaluate on fixed scenes (`evaluate_act.py`, seeds from 20261001).
3. Collect `--interventions` corrections with that policy (`tools/dagger_act.py`, seeds
   from 20262001 + 100000·r, never overlapping evaluation).

`dagger_act.py` runs ACT exactly as `rollout_act.py` does (checkpoint inference settings,
deployment environment, contact-v2 dynamics) and checks privileged state every tick:

| Trigger | Fires when |
| --- | --- |
| `misaligned_close` | the gripper command crosses below 50% while the grip site is >15 mm horizontally or >25 mm vertically from the due cube's planned grasp point |
| `misaligned_release` | the command crosses above 50% while the held cube is not within 12 mm over its support |
| `wrong_block` | any cube other than the due one is grasped |
| `dropped` | a lifted cube is released and its pair does not complete within 0.5 s |
| `stall` | for 6 s, no new stage and the grip site (or held cube) never gets 1 cm closer to its goal |

On a trigger, `Planner.finish()` takes over that simulation. Following the post's
recovery rule it first returns to a demonstrated state — keeps carrying the cube that is
due, otherwise opens the jaws and rises — then completes the whole remaining task the way
`run()` does, including the one-second stable-stack acceptance. Only the expert's ticks
are saved, in the native episode format with `control_mode: dagger_expert` and the
trigger, policy ticks, checkpoint and inference settings. Policy actions are never
trained on. A fixed number of saved corrections per round keeps rounds comparable.

## Control arm: the same rounds without the policy

As in the post's comparison with plain demonstrations, `--mode demos` runs identical
rounds but adds the same number of fresh scripted demonstrations from new scenes instead
of policy corrections; `--reuse-round0` makes both arms share one round-0 dataset, policy
and evaluation. An improvement in the correction arm over this arm is then attributable to
where the data was collected, not to having more of it. Arms are matched on episodes;
corrections start mid-task and are shorter, so the control arm receives more frames, which
the results table reports (`added_frames`).

## Changes to existing code

`Planner.run()` was split into `pick`, `place` and `validate` so `finish()` can reuse
them, and `Planner` accepts an existing `sim`. Four regenerated episodes are
byte-identical to the previous collector in every array and stage record.

## Validation

- Takeover from mid-task states built with the planner (12 seeds each): from a fresh
  scene it succeeds on the same 5 layouts `run()` accepts; carrying a cube mid-transfer
  6/12; red already stacked 3/8; after closing 3 cm beside the cube 5/11, always starting
  with `recover_open`. Every failure is the collector's usual arm-tracking rejection on
  hard layouts, and failed takeovers are discarded.
- End to end with a deliberately undertrained ACT: three rollouts gave two saved
  corrections (both after a stall; each opened the half-closed jaws, rose and completed
  the stack) and one discarded takeover. Both corrections pass
  `validate_scripted_dataset.py`, and exporting 4 base demos plus the 2 corrections
  produced one contact-v2 dataset.
- `tests/test_dagger.py`: takeover while carrying and after a missed grasp; the monitor
  fires on a sideways close but not an aligned one, on an out-of-order grasp, and on a
  stall.

## Running it

```bash
OPENBLAS_NUM_THREADS=1 MUJOCO_GL=egl .venv-act/bin/python tools/dagger_rounds.py \
  --work outputs/dagger --rounds 3 --interventions 100 --train-steps 60000
# control arm, sharing round 0
OPENBLAS_NUM_THREADS=1 MUJOCO_GL=egl .venv-act/bin/python tools/dagger_rounds.py \
  --work outputs/dagger_demos --mode demos --reuse-round0 outputs/dagger \
  --base outputs/dagger/base --rounds 3 --interventions 100 --train-steps 60000
```

The base set (1,000 contact-v2 demonstrations) is collected into `WORK/base` unless
`--base` points at an existing contact-v2 collection; the published dataset is weld-v1
and cannot be mixed with corrections. Each round re-exports the base demonstrations
(~22 GiB) and training adds an Arrow cache of similar size; earlier rounds' datasets are
deleted unless `--keep-datasets`. Pass trainer options with `--train-arg=--lr=...`.

## Open questions for the first real run

- Trigger thresholds are reasoned, not tuned. Watch the trigger mix in
  `corrections_*/rollouts.jsonl`; a policy dominated by `stall` corrections may need a
  shorter stall or earlier triggers.
- Corrections run to the end of the task. The post found that finishing the whole subtask
  avoids teaching the policy to "correct" successes; whether 100 corrections per round
  is the right ratio to 1,000 base demos is untested.
