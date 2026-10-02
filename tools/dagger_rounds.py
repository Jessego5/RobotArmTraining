#!/usr/bin/env python3
"""Run expert-in-the-loop (DAgger) rounds for ACT end to end.

Round r trains ACT from scratch on the base demonstrations plus every
correction gathered so far, evaluates it on fixed held-out scenes, then lets
the scripted expert collect ``--interventions`` new corrections from that
policy's failures (``tools/dagger_act.py``). Each step is skipped when its
output already exists, so an interrupted run resumes where it stopped.

    python tools/dagger_rounds.py --work outputs/dagger --rounds 3 \\
        --interventions 100 --train-steps 60000

Evaluation seeds start at ``--eval-seed`` and collection seeds well away from
them, so no evaluated scene is ever corrected by the expert.

``--mode demos`` is the control arm: each round adds the same number of fresh
scripted demonstrations from new scenes instead of policy corrections, with
identical training and evaluation. ``--reuse-round0 OTHER_WORK`` shares the
other arm's round-0 dataset, policy and evaluation so both start identically:

    python tools/dagger_rounds.py --work outputs/dagger --rounds 3 ...
    python tools/dagger_rounds.py --work outputs/dagger_demos --mode demos \
        --reuse-round0 outputs/dagger --rounds 3 ...
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def run(command: list, log: Path, **env) -> None:
    print('+', ' '.join(map(str, command)), flush=True)
    with log.open('a') as stream:
        subprocess.run(list(map(str, command)), cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                       check=True, env={**os.environ, **{k: str(v) for k, v in env.items()}})


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--base', type=Path, help='contact-v2 base demos; collected into WORK/base if absent')
    parser.add_argument('--base-episodes', type=int, default=1000)
    parser.add_argument('--rounds', type=int, default=3, help='correction rounds after the base policy')
    parser.add_argument('--interventions', type=int, default=100, help='corrections collected per round')
    parser.add_argument('--train-steps', type=int, default=60000)
    parser.add_argument('--train-arg', action='append', default=[], help='extra train_act.py argument (repeatable)')
    parser.add_argument('--eval-episodes', type=int, default=50)
    parser.add_argument('--eval-seconds', type=float, default=45.)
    parser.add_argument('--eval-seed', type=int, default=20261001)
    parser.add_argument('--collect-seed', type=int, default=20262001)
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--keep-datasets', action='store_true', help='keep every round\'s exported dataset')
    parser.add_argument('--mode', choices=('corrections', 'demos', 'human'), default='corrections',
                        help='demos: fresh scripted demonstrations instead (control arm); '
                             'human: keyboard corrections from tools/hil_keyboard.py')
    parser.add_argument('--reuse-round0', type=Path, help="another arm's work directory to share round 0 with")
    args = parser.parse_args()
    work = args.work.resolve()
    work.mkdir(parents=True, exist_ok=True)
    log = work / 'dagger_rounds.log'

    base = (args.base or work / 'base').resolve()
    status = base / 'status.json'
    if not (status.exists() and json.loads(status.read_text()).get('complete')):
        run([PY, 'tools/collect_scripted.py', '--output', base, '--episodes', args.base_episodes,
             '--workers', args.workers], log)

    if args.reuse_round0:
        other = args.reuse_round0.resolve()
        for name in ('round_0_episodes', 'round_0_act', 'round_0_eval'):
            if not (other / name).exists():
                raise SystemExit(f'{other / name} is missing; finish round 0 there first')
            if not (work / name).exists():
                (work / name).symlink_to(other / name)
        # The dataset may already be deleted there; it is only re-exported if needed.
        if (other / 'round_0_dataset' / 'COMPLETE').exists() and not (work / 'round_0_dataset').exists():
            (work / 'round_0_dataset').symlink_to(other / 'round_0_dataset')

    corrections = []
    results = []
    previous_dataset = None
    for round_index in range(args.rounds + 1):
        tag = f'round_{round_index}'
        source = work / f'{tag}_episodes'
        if not source.exists():
            source.mkdir()
            episodes = sorted(base.glob('episode_*')) + [p for c in corrections for p in sorted(c.glob('episode_*'))]
            for i, episode in enumerate(episodes):
                (source / f'episode_{i:04d}').symlink_to(episode)
        dataset = work / f'{tag}_dataset'
        checkpoint = work / f'{tag}_act'
        evaluation = work / f'{tag}_eval'
        collected = work / f'corrections_{round_index + 1}'
        collecting = (args.mode == 'corrections' and round_index < args.rounds
                      and not (collected / 'status.json').exists())
        needs_dataset = (collecting or not (evaluation / 'summary.json').exists()
                         or not (checkpoint / 'best' / 'model.safetensors').exists())
        if needs_dataset and not (dataset / 'COMPLETE').exists():
            shutil.rmtree(dataset, ignore_errors=True)
            shutil.rmtree(work / f'{tag}_rendered', ignore_errors=True)
            run([PY, 'tools/export_scripted_dataset.py', '--input', source,
                 '--rendered', work / f'{tag}_rendered', '--output', dataset, '--workers', args.workers], log)
        if not (checkpoint / 'best' / 'model.safetensors').exists():
            # Training materializes an Arrow copy of the dataset; keep it per round
            # so it is deleted with the round's dataset instead of accumulating.
            run([PY, 'train_act.py', '--dataset', dataset, '--output', checkpoint,
                 '--steps', args.train_steps, *args.train_arg], log,
                HF_DATASETS_CACHE=work / f'{tag}_arrow_cache')
        if not (evaluation / 'summary.json').exists():
            run([PY, 'tools/evaluate_act.py', '--checkpoint', checkpoint / 'best', '--dataset', dataset,
                 '--episodes', args.eval_episodes, '--seconds', args.eval_seconds,
                 '--seed', args.eval_seed, '--output', evaluation], log)
        rates = json.loads((evaluation / 'summary.json').read_text())['rates']
        frames = sum(json.loads(m.read_text())['n_steps'] for c in corrections for m in c.glob('episode_*/meta.json'))
        results.append(dict(round=round_index, mode=args.mode,
                            added_episodes=sum(len(list(c.glob('episode_*'))) for c in corrections),
                            added_frames=frames,
                            **{k: rates[k] for k in ('success', 'two_stacked', 'sustained_pickup', 'grasped')}))
        print(json.dumps(results[-1]), flush=True)
        if previous_dataset is not None and not args.keep_datasets and not previous_dataset.is_symlink():
            shutil.rmtree(previous_dataset, ignore_errors=True)
            shutil.rmtree(previous_dataset.with_name(previous_dataset.name.replace('_dataset', '_arrow_cache')),
                          ignore_errors=True)
        previous_dataset = dataset
        if round_index == args.rounds:
            break
        done = collected / 'status.json'
        if not (done.exists() and json.loads(done.read_text()).get('complete')):
            if args.mode != 'human':
                shutil.rmtree(collected, ignore_errors=True)
            if args.mode == 'human':
                print(f"\nRound {round_index} is trained. Collect corrections with this policy, e.g.\n"
                      f"  python tools/hil_keyboard.py --checkpoint {checkpoint / 'best'} --out {collected}\n"
                      f"(on a machine with a display; copy the checkpoint there and the folder back),\n"
                      f"then rerun this command to continue.", flush=True)
                return
            if args.mode == 'demos':
                run([PY, 'tools/collect_scripted.py', '--output', collected,
                     '--episodes', args.interventions, '--workers', args.workers,
                     '--seed', args.collect_seed + 100000 * round_index], log)
            else:
                run([PY, 'tools/dagger_act.py', '--checkpoint', checkpoint / 'best', '--dataset', dataset,
                     '--output', collected, '--interventions', args.interventions,
                     '--seed', args.collect_seed + 100000 * round_index], log)
        corrections.append(collected)

    (work / 'results.json').write_text(json.dumps(results, indent=2) + '\n')
    print(f"\n{args.mode}: {'round':>5} {'added eps':>9} {'frames':>8} {'success':>8} {'2-stack':>8} {'pickup':>8} {'grasped':>8}")
    for r in results:
        print(f"{'':>{len(args.mode) + 1}} {r['round']:>5} {r['added_episodes']:>9} {r['added_frames']:>8} "
              f"{r['success']:>8.0%} {r['two_stacked']:>8.0%} {r['sustained_pickup']:>8.0%} {r['grasped']:>8.0%}")


if __name__ == '__main__':
    main()
