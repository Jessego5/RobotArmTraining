#!/usr/bin/env python3
"""Stage and publish any exported scripted dataset for the pi0.5 notebook.

``publish_pi05_dataset.py`` releases the original 1,000-demo weld-v1 dataset and
checks its exact counts. This publishes other exports of the same task -- a new
collection, contact-v2 physics, a domain-randomized or generatively restyled
variant -- with what the notebook needs:

- ``data/**`` and ``meta/**`` (hard-linked, so the export is not duplicated),
- ``assets/pi05_runtime.zip``: the current simulator, built and validated by
  ``bundle_pi05_runtime.py`` against this dataset's provenance, so pi0.5 is
  evaluated in the physics it was trained on,
- ``upload_manifest.json``: size and SHA-256 of every file plus the episode and
  frame counts the notebook checks,
- a dataset card generated from the provenance, and ``COMPLETE`` once the
  remote copy is verified.

    python tools/publish_dataset_variant.py --source outputs/lerobot/panthera_restyled \\
        --episodes data/scripted_contact --repo-id USER/panthera-restyled-30hz --private --push
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tools.publish_pi05_dataset import CAMERAS, digest  # noqa: E402


def validate(root: Path) -> dict:
    """Counts, shapes and every numeric row, without decoding images."""
    import numpy as np
    import pyarrow.parquet as pq

    if not (root / 'COMPLETE').exists():
        raise SystemExit(f'{root} is not a complete export')
    info = json.loads((root / 'meta/info.json').read_text())
    assert info['fps'] == 30, info['fps']
    for key in CAMERAS:
        assert info['features'][key]['shape'] == [256, 256, 3], key
    for key in ('observation.state', 'action'):
        assert info['features'][key]['shape'] == [7], key
    episodes = info['total_episodes']
    rows, counts = 0, np.zeros(episodes, dtype=np.int64)
    for path in sorted((root / 'data').rglob('*.parquet')):
        table = pq.read_table(path, columns=['index', 'episode_index', 'action', 'observation.state'])
        assert np.array_equal(table['index'].to_numpy(), np.arange(rows, rows + len(table))), path
        index = table['episode_index'].to_numpy()
        assert np.all((index >= 0) & (index < episodes)), path
        counts += np.bincount(index, minlength=episodes)
        for key in ('action', 'observation.state'):
            values = np.asarray(table[key].to_pylist())
            assert values.shape == (len(table), 7) and np.isfinite(values).all(), (path, key)
        rows += len(table)
    assert rows == info['total_frames'] and np.all(counts > 0), (rows, info['total_frames'])
    return info


def card(info: dict, provenance: dict, repo_id: str) -> str:
    restyle = provenance.get('restyle')
    randomized = provenance.get('domain_randomization')
    variant = []
    if restyle:
        s = restyle['settings']
        variant.append(
            f"- **Generative restyle:** {restyle['fraction']:.0%} of episodes (seeded per episode) repainted by "
            f"Stable Diffusion 1.5 img2img with depth and object-outline ControlNets and LCM-LoRA "
            f"({s['steps']} steps, strength {s['strength']}, guidance {s['guidance']}, {s['size']} px). Geometry "
            f"comes from the simulator's depth and segmentation; cube pixels are pasted back exactly, so cube "
            f"positions and colours are unchanged. Per-episode prompt, outline recall and flicker are recorded "
            f"beside the rendered trajectories as `restyle.json`.")
    if randomized:
        variant.append('- **Domain randomization:** per-episode table, floor and arm appearance, lighting and '
                       'small camera offsets; cube hues preserved.')
    if not variant:
        variant.append('- Plain simulator renders (no appearance changes).')
    return f"""---
tags:
- lerobot
- robotics
- mujoco
- imitation-learning
- pi05
pretty_name: Panthera three-block stacking ({provenance.get('simulation_dynamics', 'unknown')}, 30 Hz)
---

# Panthera three-block stacking

{info['total_episodes']:,} scripted IK demonstrations, {info['total_frames']:,} training frames at 30 Hz,
simulation dynamics **{provenance.get('simulation_dynamics', 'unknown')}**. Task: red onto green, then blue
onto red; stored prompt `stack the three colored cubes`. LeRobot v3.0 format (exported with LeRobot
0.4.4) with both 256x256 RGB cameras embedded as JPEG in Parquet.

{chr(10).join(variant)}

| Feature | Contract |
| --- | --- |
| observation.state | Six measured joint angles in radians, then the gripper opening command in metres |
| action | Six absolute joint targets in radians, then absolute gripper opening command in metres |
| Timing | Observation at t predicts control at t + 1/30 s; actions are already shifted |
| Camera order | Shoulder, then wrist |

`assets/pi05_runtime.zip` holds the simulator these episodes were generated in, so the pi0.5 notebook
evaluates in matching physics. `meta/provenance.json` keeps source and renderer hashes. Demonstration
checks are not learned-policy success rates.

Source project: https://github.com/zebulontaylor/RobotArmTraining
Dataset: https://huggingface.co/datasets/{repo_id}
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--source', type=Path, required=True, help='complete LeRobot export')
    parser.add_argument('--episodes', type=Path, required=True, help='native episodes it was exported from')
    parser.add_argument('--repo-id', required=True, help='e.g. USER/panthera-restyled-30hz')
    parser.add_argument('--stage', type=Path, help='default: outputs/hf/<repo name>')
    parser.add_argument('--private', action='store_true')
    parser.add_argument('--push', action='store_true', help='upload; without it, only validate and stage')
    args = parser.parse_args()
    if args.repo_id.count('/') != 1 or not all(args.repo_id.split('/')):
        parser.error(f'--repo-id must be USERNAME/NAME, got {args.repo_id!r}')
    source = args.source.resolve()
    stage = (args.stage or ROOT / 'outputs/hf' / args.repo_id.split('/')[-1]).resolve()

    info = validate(source)
    stage.mkdir(parents=True, exist_ok=True)
    files = [Path('meta/info.json'), Path('meta/stats.json'), Path('meta/tasks.parquet'),
             *sorted(p.relative_to(source) for p in (source / 'meta/episodes').rglob('*.parquet')),
             *sorted(p.relative_to(source) for p in (source / 'data').rglob('*.parquet'))]
    for relative in files:
        dest = stage / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            os.link(source / relative, dest)
        elif not os.path.samefile(source / relative, dest):
            raise SystemExit(f'Refusing to replace unrelated staged file {dest}')
    provenance = json.loads((source / 'meta/provenance.json').read_text())
    provenance['rendered_root'] = 'rendered'
    for name, entry in provenance['sources'].items():
        entry['source'] = f'episodes/{name}'  # no machine-specific paths
    (stage / 'meta/provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    (stage / 'README.md').write_text(card(info, provenance, args.repo_id))
    runtime = stage / 'assets/pi05_runtime.zip'
    subprocess.run([sys.executable, str(ROOT / 'tools/bundle_pi05_runtime.py'), '--dataset', str(source),
                    '--episodes', str(args.episodes.resolve()), '--output', str(runtime)], check=True)
    paths = sorted(p for p in stage.rglob('*') if p.is_file() and '.cache' not in p.parts
                   and p.name not in {'upload_manifest.json', 'COMPLETE'})
    manifest = {'repo_id': args.repo_id, 'episodes': info['total_episodes'], 'frames': info['total_frames'],
                'simulation_dynamics': provenance.get('simulation_dynamics'),
                'files': {str(p.relative_to(stage)): {'size': p.stat().st_size, 'sha256': digest(p)}
                          for p in paths}}
    (stage / 'upload_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f"Validated/staged {len(manifest['files'])} files, "
          f"{sum(v['size'] for v in manifest['files'].values()) / 2**30:.2f} GiB at {stage}", flush=True)
    if not args.push:
        return

    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(args.repo_id, repo_type='dataset', private=args.private, exist_ok=True)
    api.upload_large_folder(repo_id=args.repo_id, repo_type='dataset', folder_path=stage,
                            allow_patterns=[*manifest['files'], 'upload_manifest.json'],
                            num_workers=4, print_report_every=60)
    remote = {f.rfilename: f for f in api.dataset_info(args.repo_id, files_metadata=True).siblings}
    for path, record in manifest['files'].items():
        f = remote.get(path)
        if f is None or f.size != record['size']:
            raise SystemExit(f'Incomplete remote file: {path}')
        if f.lfs is not None and f.lfs.sha256 != record['sha256']:
            raise SystemExit(f'Remote hash mismatch: {path}')
    commit = api.upload_file(
        repo_id=args.repo_id, repo_type='dataset', path_in_repo='COMPLETE',
        path_or_fileobj=f"{info['total_episodes']} episodes, {info['total_frames']} frames; remote files verified\n".encode(),
        commit_message='Mark verified dataset complete')
    result = dict(repo_id=args.repo_id, revision=commit.oid, url=f'https://huggingface.co/datasets/{args.repo_id}',
                  episodes=info['total_episodes'], frames=info['total_frames'])
    (stage.parent / f"{stage.name}_upload.json").write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
