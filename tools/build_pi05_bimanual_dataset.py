#!/usr/bin/env python3
"""Merge two-arm sim exports into one pi0.5 training set with a demo-level holdout.

The v1 and v2 exports render the same scripted demonstrations with different
looks, so a split by episode would validate on motions seen in training. This
holds out a share of *demonstrations*: every rendering of a held-out demo, in
every source export, goes to validation. Validation episodes are placed last,
because the pinned LeRobot trainer (`--dataset.eval_split`) holds out the last
ceil(N * eval_split) episodes; `meta/pi05_split.json` records the exact split and
the eval_split to pass, and `tools/train_pi05_full.py` checks it.

    python tools/build_pi05_bimanual_dataset.py \\
        --sources /workspace/bimanual_cad/lerobot_panthera_bimanual_sim_20hz \\
                  /workspace/bimanual_cad_v2/lerobot_panthera_bimanual_sim_20hz \\
        --output /workspace/pi05_bimanual/dataset
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
REPO_ID = 'local/panthera_bimanual_pi05'


def demo_of(name: str) -> str:
    """'episode_0012_c1' -> 'episode_0012' (the scripted demo behind a rendering)."""
    return name.rsplit('_c', 1)[0]


def main():
    from lerobot.datasets.dataset_tools import merge_datasets, split_dataset
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--sources', type=Path, nargs='+', required=True, help='complete export roots')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--validation-demos', type=float, default=.1, help='share of demonstrations held out')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f'{args.output} exists; choose a fresh path')
    sources = []
    for root in args.sources:
        if not (root / 'COMPLETE').exists():
            raise SystemExit(f'{root} is not a complete export')
        provenance = json.loads((root / 'meta/provenance.json').read_text())
        names = list(provenance['episodes'])  # in episode-index order
        dataset = LeRobotDataset(REPO_ID + '_src', root=root)
        if len(names) != dataset.meta.total_episodes:
            raise SystemExit(f'{root}: provenance lists {len(names)} episodes, dataset has '
                             f'{dataset.meta.total_episodes}')
        sources.append((root, provenance, names, dataset))
    for key in ('fps', 'features'):
        values = {json.dumps(getattr(s[3].meta, key) if key == 'fps' else
                             {k: v['shape'] for k, v in s[3].meta.features.items()}) for s in sources}
        if len(values) != 1:
            raise SystemExit(f'sources differ in {key}')

    demos = sorted({demo_of(n) for _, _, names, _ in sources for n in names})
    order = np.random.default_rng(args.seed).permutation(len(demos))
    held_out = {demos[i] for i in order[:max(1, math.ceil(len(demos) * args.validation_demos))]}
    scratch = args.output.parent / (args.output.name + '.parts')
    shutil.rmtree(scratch, ignore_errors=True)
    train_parts, validation_parts, counts = [], [], []
    for k, (root, _, names, dataset) in enumerate(sources):
        validation = [i for i, n in enumerate(names) if demo_of(n) in held_out]
        train = [i for i, n in enumerate(names) if demo_of(n) not in held_out]
        parts = split_dataset(dataset, {'train': train, 'validation': validation}, output_dir=scratch / f'source{k}')
        train_parts.append(parts['train'])
        validation_parts.append(parts['validation'])
        counts.append(dict(source=str(root), train=len(train), validation=len(validation)))
        print(f'{root}: {len(train)} train, {len(validation)} validation episodes', flush=True)
    merged = merge_datasets(train_parts + validation_parts, output_repo_id=REPO_ID, output_dir=args.output)
    total = merged.meta.total_episodes
    n_validation = sum(c['validation'] for c in counts)
    eval_split = n_validation / total
    if math.ceil(total * eval_split) != n_validation:
        raise SystemExit('eval_split does not reproduce the validation count exactly')
    split = dict(train=list(range(total - n_validation)), validation=list(range(total - n_validation, total)),
                 eval_split=eval_split, validation_demos=sorted(held_out), seed=args.seed, sources=counts)
    (args.output / 'meta/pi05_split.json').write_text(json.dumps(split, indent=2) + '\n')
    tasks = sorted(set(merged.meta.tasks.index)) if hasattr(merged.meta.tasks, 'index') else None
    provenance = dict(simulation_dynamics='contact-v2', merged_from=[str(s[0]) for s in sources],
                      source_provenance={str(s[0]): {k: v for k, v in s[1].items() if k != 'episodes'}
                                         for s in sources},
                      tasks=tasks, split='meta/pi05_split.json')
    (args.output / 'meta/provenance.json').write_text(json.dumps(provenance, indent=2, default=str) + '\n')
    (args.output / 'COMPLETE').write_text(f'{total} episodes, {merged.meta.total_frames} frames; '
                                          f'last {n_validation} are validation\n')
    shutil.rmtree(scratch, ignore_errors=True)
    print(f'{total} episodes ({total - n_validation} train, {n_validation} validation; '
          f'--dataset.eval_split={eval_split!r}), {merged.meta.total_frames} frames at {args.output}', flush=True)


if __name__ == '__main__':
    main()
