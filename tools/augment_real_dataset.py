#!/usr/bin/env python3
"""Augment the real two-arm demonstrations by restyling their table and background.

The motions are real and cannot be edited (the frames show the actual poses),
so this only changes how the scene looks. Per frame:

- a monocular depth model (Depth Anything V2) estimates depth;
- the table is modelled as a smooth surface fitted to the depth of wood-coloured
  pixels; everything that rises above it (arms, grippers, fingers, carrier, gear,
  pin, operator hands), plus anything not wood-coloured that is not farther than
  the table, is KEPT and copied back exactly; glare and small flat patches are not;
- the rest (table and room) is repainted by the structure-locked diffusion
  restyle of tools/sim2real_augment.py, guided by that depth and the image's own
  edges, with prompts for other tables, rooms and lighting.

Output is a LeRobot dataset in the real dataset's exact format with every
original episode plus a restyled copy of a share of them (states, actions,
velocities and efforts copied unchanged). Rendering resumes like the sim export.

    python tools/augment_real_dataset.py preview --output outputs/real_aug_preview.jpg
    python tools/augment_real_dataset.py export --output /workspace/real_aug/lerobot \\
        --work /workspace/real_aug/rendered --fraction 0.5 --workers 4
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np

REAL = 'FoxNerdSaysMoo/panthera-gear-carrier-pin-real-20hz'
CAMERAS = ('overhead', 'wrist_port2', 'wrist_port3')
NUMERIC = ('observation.state', 'action', 'observation.velocity', 'observation.effort')
DEPTH_MODEL = 'depth-anything/Depth-Anything-V2-Small-hf'
SCENES = ('a light maple butcher-block table in a university lab', 'a dark walnut wooden desk in a home office',
          'a white laminate workbench in a bright clean room', 'a grey steel workbench in a machine shop',
          'a black rubber mat on a workshop table', 'a pale plywood table in a cluttered garage',
          'a green cutting mat on a lab bench', 'a polished concrete counter in a warehouse')
LIGHTS = ('bright fluorescent ceiling light', 'warm tungsten lamp light, soft shadows',
          'cool overcast daylight from a window', 'harsh afternoon sunlight, long shadows',
          'dim evening light, a single desk lamp', 'even studio lighting')
ROBOT = ('two white and grey aluminium robot arms with white 3D printed fin-ray gripper fingers and black brackets, '
         'assembling a small black gear and a steel pin into a silver machined carrier')
NEGATIVE = ('cartoon, 3d render, cgi, illustration, painting, text, watermark, extra robot arms, extra hands, '
            'distorted hands, extra gears')


def prompts():
    return tuple(f'a webcam photo of {ROBOT} on {scene}, {light}' for scene in SCENES for light in LIGHTS)


# ---------------------------------------------------------------- masks -----

def wood(image_bgr):
    hsv = cv2.cvtColor(cv2.GaussianBlur(image_bgr, (5, 5), 0), cv2.COLOR_BGR2HSV).astype(int)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    return (h >= 8) & (h <= 32) & (s >= 35) & (v >= 90)


def table_colour(image_bgr):
    """Table-coloured pixels: the wood hue, learned per frame so pale, washed-out wood still counts."""
    lab = cv2.cvtColor(cv2.GaussianBlur(image_bgr, (5, 5), 0), cv2.COLOR_BGR2LAB).astype(np.float32)
    a, b = lab[..., 1] - 128, lab[..., 2] - 128
    hue, chroma = np.degrees(np.arctan2(b, a)), np.hypot(a, b)
    seed = wood(image_bgr)
    if seed.sum() < 200:
        return seed
    table_hue = np.median(hue[seed])
    return seed | ((np.abs((hue - table_hue + 180) % 360 - 180) < 22) & (chroma > 5) & (lab[..., 0] > 60))


def table_surface(disparity, table, grid=64, sigma=4.):
    """Smooth disparity of the table, filled in under objects (follows fisheye and perspective)."""
    height, width = disparity.shape
    d = cv2.resize(disparity, (grid, grid), interpolation=cv2.INTER_AREA)
    w = cv2.resize(table.astype(np.float32), (grid, grid), interpolation=cv2.INTER_AREA)
    w = (w > .6).astype(np.float32)
    surface = d.copy()
    for s in (sigma, sigma * 2, sigma * 4):  # normalised convolution, widening to bridge large objects
        num = cv2.GaussianBlur(d * w, (0, 0), s)
        den = cv2.GaussianBlur(w, (0, 0), s)
        estimate = num / np.maximum(den, 1e-4)
        surface = np.where((w > 0) | (den > .05), estimate, surface) if s == sigma else \
            np.where(den > .02, np.where(w > 0, surface, estimate), surface)
    return cv2.resize(surface, (width, height), interpolation=cv2.INTER_LINEAR)


def keep_mask(image_bgr, disparity, margin=.06):
    """Pixels to copy back unchanged: everything rising above the table, or not table-coloured and not farther."""
    lo, hi = np.percentile(disparity, [1, 99])
    disparity = (disparity - lo) / max(hi - lo, 1e-6)
    hsv = cv2.cvtColor(cv2.GaussianBlur(image_bgr, (5, 5), 0), cv2.COLOR_BGR2HSV)
    bright = cv2.dilate(((hsv[..., 2] > 205) & (hsv[..., 1] < 60)).astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    table = table_colour(image_bgr)
    if table.sum() < 200:  # no visible table: keep everything rather than guess
        return np.ones(disparity.shape, bool)
    surface = table_surface(disparity, table)
    residual = disparity - surface
    # Glare lies flat on the table; bright white things that rise from it (fingers) are objects.
    glare = bright & (residual < margin)
    # The room and walls sit farther than the table (lower disparity), so they are repainted.
    keep = ((residual > margin) | (~table & ~glare & (residual > -.03))).astype(np.uint8)
    keep = cv2.morphologyEx(keep, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    keep = cv2.morphologyEx(keep, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(keep)
    for i in range(1, count):  # small patches that do not rise from the table: grain or glare edges
        part = labels == i
        if stats[i, 4] < 250 and np.median(residual[part]) < margin:
            keep[part] = 0
    return cv2.dilate(keep, np.ones((3, 3), np.uint8)).astype(bool)


class Depth:
    def __init__(self):
        import torch
        from transformers import pipeline
        device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        self.pipe = pipeline('depth-estimation', model=DEPTH_MODEL, device=device)

    def __call__(self, frames_rgb):
        from PIL import Image
        out = []
        for result, frame in zip(self.pipe([Image.fromarray(f) for f in frames_rgb], batch_size=16), frames_rgb):
            d = np.asarray(result['depth'], np.float32)
            out.append(cv2.resize(d, (frame.shape[1], frame.shape[0])))
        return np.stack(out)


def controls(frames_rgb, disparity):
    """Depth control (near = bright, scaled per episode so it does not pulse) and image edges."""
    lo, hi = np.percentile(disparity, [1, 99])
    depth = np.clip((disparity - lo) / max(hi - lo, 1e-6), 0, 1)
    depth_images = np.repeat((depth * 255).astype(np.uint8)[..., None], 3, -1)
    edges = [cv2.Canny(cv2.cvtColor(f, cv2.COLOR_RGB2GRAY), 80, 160) for f in frames_rgb]
    return depth_images, np.repeat(np.stack(edges)[..., None], 3, -1)


# ---------------------------------------------------------------- data ------

def fetch(root: Path, revision: str | None):
    from huggingface_hub import HfApi, snapshot_download
    revision = revision or HfApi().dataset_info(REAL).sha
    snapshot_download(REAL, repo_type='dataset', revision=revision, local_dir=root,
                      allow_patterns=['data/**', 'meta/**'])
    return revision


def episode_rows(root: Path, episode: int):
    """Numeric arrays and JPEG bytes per camera for one real episode."""
    import pyarrow.parquet as pq
    meta = pq.read_table(next(root.glob('meta/episodes/*/*.parquet'))).to_pandas()
    row = meta[meta['episode_index'] == episode].iloc[0]
    path = root / f"data/chunk-{int(row['data/chunk_index']):03d}/file-{int(row['data/file_index']):03d}.parquet"
    table = pq.read_table(path).to_pandas()
    table = table[table['episode_index'] == episode].sort_values('frame_index')
    numeric = {k: np.stack(table[k].to_numpy()).astype(np.float32) for k in NUMERIC}
    images = {c: [cell['bytes'] for cell in table[f'observation.images.{c}']] for c in CAMERAS}
    return numeric, images


def real_features(root: Path) -> dict:
    """The real dataset's own feature schema, minus LeRobot's bookkeeping columns."""
    info = json.loads((root / 'meta/info.json').read_text())
    skip = {'timestamp', 'frame_index', 'episode_index', 'index', 'task_index'}
    return {k: {kk: (tuple(vv) if kk == 'shape' else vv) for kk, vv in v.items() if kk in ('dtype', 'shape', 'names')}
            for k, v in info['features'].items() if k not in skip}


def save_prepared_episode(dataset, image_stats):
    """Save an episode whose image stats were computed in the worker; batch the JPEG embedding."""
    from unittest.mock import patch
    import lerobot.datasets.lerobot_dataset as lerobot_io
    from datasets.table import embed_table_storage
    from lerobot.datasets.compute_stats import compute_episode_stats

    def prepared_stats(buffer, schema):
        numeric = {k: v for k, v in schema.items() if v['dtype'] != 'image'}
        return {**compute_episode_stats({k: buffer[k] for k in numeric}, numeric), **image_stats}

    def embed_batched(table):
        fmt = table.format
        return table.with_format('arrow').map(embed_table_storage, batched=True, batch_size=256).with_format(**fmt)

    with patch.object(lerobot_io, 'compute_episode_stats', prepared_stats), \
            patch.object(lerobot_io, 'embed_images', embed_batched):
        dataset.save_episode()


def decode(jpegs):
    return np.stack([cv2.cvtColor(cv2.imdecode(np.frombuffer(b, np.uint8), 1), cv2.COLOR_BGR2RGB) for b in jpegs])


# ---------------------------------------------------------------- jobs ------

def restyle_settings(args):
    from tools.sim2real_augment import Settings
    return Settings(batch=args.batch, steps=args.steps, strength=args.strength, guidance=args.guidance,
                    size=args.size, depth_scale=args.depth_scale, canny_scale=args.canny_scale, ip_scale=0.,
                    prompts=prompts(), negative=NEGATIVE)


def render_job(root, dest, episode, restyle_seed, settings):
    """Write one output episode's JPEGs (original bytes, or restyled) and its image stats."""
    import pickle
    from lerobot.datasets.compute_stats import compute_episode_stats
    global depth_model, restyler
    shutil.rmtree(dest, ignore_errors=True)
    dest.mkdir(parents=True)
    numeric, images = episode_rows(root, episode)
    count = len(numeric['action'])
    report = dict(source_episode=episode, frames=count, restyled=restyle_seed is not None)
    for camera in CAMERAS:
        (dest / camera).mkdir()
        if restyle_seed is None:
            for i, jpeg in enumerate(images[camera]):
                (dest / camera / f'{i:05d}.jpg').write_bytes(jpeg)
            continue
        if 'depth_model' not in globals():
            from tools.sim2real_augment import Restyler
            depth_model, restyler = Depth(), Restyler(settings)
        frames = decode(images[camera])
        disparity = depth_model(frames)
        keep = np.stack([keep_mask(cv2.cvtColor(f, cv2.COLOR_RGB2BGR), d) for f, d in zip(frames, disparity)])
        restyled = restyler.episode(frames, None, None, keep, restyle_seed, camera, ~keep,
                                    controls=controls(frames, disparity))
        report[f'{camera}_kept_share'] = float(keep.mean())
        for i, frame in enumerate(restyled):
            cv2.imwrite(str(dest / camera / f'{i:05d}.jpg'), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
    if restyle_seed is not None:
        report.update(restyle_seed=restyle_seed, prompt=settings.prompts[restyle_seed % len(settings.prompts)],
                      restyle_settings={k: v for k, v in asdict(settings).items() if k != 'prompts'})
    image_features = {k: v for k, v in real_features(root).items() if v['dtype'] == 'image'}
    buffer = {key: [str(dest / key.rsplit('.', 1)[-1] / f'{j:05d}.jpg') for j in range(count)]
              for key in image_features}
    np.savez(dest / 'numeric.npz', **numeric)
    (dest / 'image_stats.pkl').write_bytes(pickle.dumps(compute_episode_stats(buffer, image_features)))
    (dest / 'done.json').write_text(json.dumps(report) + '\n')


def export(args):
    import pickle
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    revision = fetch(args.root, args.revision)
    info = json.loads((args.root / 'meta/info.json').read_text())
    episodes = list(range(info['total_episodes']))[:args.limit or None]
    rng = np.random.default_rng(args.seed)
    restyled = sorted(rng.choice(episodes, size=round(len(episodes) * args.fraction), replace=False).tolist()) \
        if args.fraction else []
    items = [(e, None) for e in episodes] if args.include_originals else []
    items += [(e, args.seed * 100003 + e) for e in restyled]
    settings = restyle_settings(args)
    args.work.mkdir(parents=True, exist_ok=True)
    contract = dict(source=REAL, revision=revision, items=items,
                    restyle={k: v for k, v in asdict(settings).items() if k != 'prompts'})
    contract_path = args.work / 'settings.json'
    if contract_path.exists() and json.loads(contract_path.read_text()) != json.loads(json.dumps(contract)):
        raise SystemExit(f'{args.work} was rendered with other settings; use a fresh --work path')
    contract_path.write_text(json.dumps(contract, indent=2) + '\n')
    dests = [args.work / (f'episode_{e:04d}' + ('' if s is None else '_restyled')) for e, s in items]
    todo = [i for i, d in enumerate(dests) if not (d / 'done.json').exists()]
    print(f'{len(items) - len(todo)} of {len(items)} episodes ready; rendering {len(todo)} '
          f'({len(restyled)} restyled copies)', flush=True)
    started = time.time()
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        futures = {pool.submit(render_job, args.root, dests[i], items[i][0], items[i][1], settings): i for i in todo}
        for k, future in enumerate(as_completed(futures), 1):
            future.result()
            elapsed = time.time() - started
            print(json.dumps(dict(rendered=k, of=len(todo), elapsed_s=round(elapsed),
                                  eta_h=round(elapsed / k * (len(todo) - k) / 3600, 2))), flush=True)
    shutil.rmtree(args.output, ignore_errors=True)
    task = json.loads(json.dumps(__import__('pyarrow.parquet', fromlist=['x']).read_table(
        args.root / 'meta/tasks.parquet').to_pandas().index.tolist()))[0]
    dataset = LeRobotDataset.create(repo_id='local/panthera_real_augmented', root=args.output, fps=info['fps'],
                                    robot_type=info.get('robot_type', 'panthera_ht_bimanual_real'),
                                    features=real_features(args.root), use_videos=False, image_writer_threads=0,
                                    metadata_buffer_size=20)
    reports, total = {}, 0
    try:
        for (episode, seed), dest in zip(items, dests):
            report = json.loads((dest / 'done.json').read_text())
            reports[dest.name] = report
            with np.load(dest / 'numeric.npz') as numeric:
                n = len(numeric['action'])
                buffer = dataset.episode_buffer
                buffer.update(size=n, task=[task] * n, timestamp=np.arange(n) / info['fps'], frame_index=np.arange(n))
                for key in NUMERIC:
                    buffer[key] = numeric[key]
            for camera in CAMERAS:
                buffer[f'observation.images.{camera}'] = [str(dest / camera / f'{j:05d}.jpg') for j in range(n)]
            save_prepared_episode(dataset, pickle.loads((dest / 'image_stats.pkl').read_bytes()))
            total += n
    finally:
        dataset.finalize()
        dataset.stop_image_writer()
    provenance = dict(source=REAL, revision=revision, generator='tools/augment_real_dataset.py',
                      method='table/background restyle; robot, parts and hands copied back from the real frames',
                      depth_model=DEPTH_MODEL, include_originals=args.include_originals,
                      restyled_source_episodes=restyled, episodes=reports,
                      restyle={k: v for k, v in asdict(settings).items() if k != 'prompts'})
    (args.output / 'meta/provenance.json').write_text(json.dumps(provenance, indent=2, default=str) + '\n')
    (args.output / 'COMPLETE').write_text(f'{len(items)} episodes, {total} frames\n')
    if not args.keep_jpegs:
        for dest in dests:
            for camera in CAMERAS:
                shutil.rmtree(dest / camera, ignore_errors=True)
    print(f'complete: {len(items)} episodes ({len(restyled)} restyled), {total} frames at {args.output}')


def preview(args):
    """A few frames per camera: original | kept pixels | restyled."""
    revision = fetch(args.root, args.revision)
    from tools.sim2real_augment import Restyler
    settings = restyle_settings(args)
    depth_model = Depth()
    restyler = None if args.no_restyle else Restyler(settings)
    from dataclasses import replace
    rows = []
    for k, episode in enumerate(args.episodes):
        _, images = episode_rows(args.root, episode)
        for camera in CAMERAS:
            pick = np.linspace(0, len(images[camera]) - 1, args.frames + 2)[1:-1].astype(int)
            frames = decode([images[camera][i] for i in pick])
            disparity = depth_model(frames)
            keep = np.stack([keep_mask(cv2.cvtColor(f, cv2.COLOR_RGB2BGR), d) for f, d in zip(frames, disparity)])
            shown = frames.copy()
            shown[~keep] = (shown[~keep] * .3 + np.array([0, 80, 255]) * .4).astype(np.uint8)
            columns = [frames, shown]
            if restyler is not None:
                restyler.settings = replace(settings, carry_static=False)
                columns.append(restyler.episode(frames, None, None, keep, args.seed + k, camera, ~keep,
                                                controls=controls(frames, disparity)))
            for i in range(len(frames)):
                rows.append(np.concatenate([c[i] for c in columns], 1))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output), cv2.cvtColor(np.concatenate(rows, 0), cv2.COLOR_RGB2BGR))
    print(f'wrote {args.output} (revision {revision[:8]}): columns original, kept (blue = repainted)'
          + ('' if restyler is None else ', restyled'))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument('--root', type=Path, default=ROOT / 'data/real_gear', help='local copy of the real dataset')
    common.add_argument('--revision', help='dataset revision (default: latest)')
    common.add_argument('--seed', type=int, default=0)
    common.add_argument('--batch', type=int, default=8)
    common.add_argument('--steps', type=int, default=4)
    common.add_argument('--size', type=int, default=384, help='diffusion resolution (frames are 256 px)')
    common.add_argument('--strength', type=float, default=.85)
    common.add_argument('--guidance', type=float, default=1.)
    common.add_argument('--depth-scale', type=float, default=.7)
    common.add_argument('--canny-scale', type=float, default=.5)
    p = sub.add_parser('export', parents=[common])
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--work', type=Path, required=True, help='per-episode render folders (resumable)')
    p.add_argument('--fraction', type=float, default=.5, help='share of episodes that get a restyled copy')
    p.add_argument('--no-originals', dest='include_originals', action='store_false')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--limit', type=int)
    p.add_argument('--keep-jpegs', action='store_true')
    p = sub.add_parser('preview', parents=[common])
    p.add_argument('--output', type=Path, default=ROOT / 'outputs/real_aug_preview.jpg')
    p.add_argument('--episodes', type=int, nargs='+', default=[0, 101])
    p.add_argument('--frames', type=int, default=2)
    p.add_argument('--no-restyle', action='store_true', help='masks only (no diffusion models)')
    args = parser.parse_args()
    {'export': export, 'preview': preview}[args.command](args)


if __name__ == '__main__':
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    main()
