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
          'a green cutting mat on a lab bench', 'a polished concrete counter in a warehouse',
          'a blue anti-static mat on an electronics bench', 'a white marble countertop in a kitchen',
          'a red-painted metal table in a factory', 'a cork board surface in a classroom',
          'a glossy black desk in a dark office', 'a stainless steel table in a clean lab',
          'a scratched old wooden workbench in a basement', 'a beige plastic folding table in a hallway',
          'a dark grey felt mat on an office desk', 'a yellow-orange workbench in a makerspace')
LIGHTS = ('bright fluorescent ceiling light', 'warm tungsten lamp light, soft shadows',
          'cool overcast daylight from a window', 'harsh afternoon sunlight, long shadows',
          'dim evening light, a single desk lamp', 'even studio lighting', 'blue-tinted LED panel light',
          'golden hour sunlight through blinds, striped shadows', 'low light, mostly in shadow',
          'strong side light from a window, high contrast')
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
    bright = cv2.dilate(((hsv[..., 2] > 235) & (hsv[..., 1] < 45)).astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
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
    lightness = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32)
    for i in range(1, count):  # small patches that do not rise from the table: grain or glare edges...
        part = labels == i
        if stats[i, 4] < 250 and np.median(residual[part]) < margin:
            ring = cv2.dilate(part.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool) & ~part
            # ...unless they stand out from the table around them, like the pin lying on it.
            if abs(lightness[part].mean() - lightness[ring].mean()) < 25:
                keep[part] = 0
    keep = cv2.dilate(keep, np.ones((3, 3), np.uint8))
    # Small parts (the pin, the loose gear) get a wider margin: a restyle bleeding a few
    # pixels into them changes what they look like.
    count, labels, stats, _ = cv2.connectedComponentsWithStats(keep)
    small = np.isin(labels, [i for i in range(1, count) if stats[i, 4] < 600])
    return (keep.astype(bool) | cv2.dilate(small.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool))


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


def controls(frames_rgb, disparity, keep=None):
    """Depth control (near = bright, scaled per episode so it does not pulse) and image edges.

    With `keep`, edges are limited to the kept objects and their outlines, so the table
    grain and the room do not pin the repaint to the original surface.
    """
    lo, hi = np.percentile(disparity, [1, 99])
    depth = np.clip((disparity - lo) / max(hi - lo, 1e-6), 0, 1)
    depth_images = np.repeat((depth * 255).astype(np.uint8)[..., None], 3, -1)
    edges = []
    for i, f in enumerate(frames_rgb):
        e = cv2.Canny(cv2.cvtColor(f, cv2.COLOR_RGB2GRAY), 80, 160)
        if keep is not None:
            near = cv2.dilate(keep[i].astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
            e = np.where(near, e, 0).astype(np.uint8)
        edges.append(e)
    return depth_images, np.repeat(np.stack(edges)[..., None], 3, -1)


def sample_lighting(seed: int) -> dict:
    """One episode's whole-frame lighting change, robot included."""
    rng = np.random.default_rng(seed)
    cast = rng.choice(['neutral', 'warm', 'cool', 'green', 'blue', 'neutral'])
    tint = {'neutral': [1, 1, 1], 'warm': [1.18, 1.0, .78], 'cool': [.85, .97, 1.18],
            'green': [.9, 1.12, .9], 'blue': [.8, .92, 1.25]}[cast]
    return dict(gain=float(rng.uniform(.55, 1.35)), gamma=float(rng.uniform(.8, 1.3)),
                tint=(np.array(tint) * rng.uniform(.95, 1.05, 3)).tolist(), cast=str(cast),
                light_azimuth=float(rng.uniform(0, 360)), light_elevation=float(rng.uniform(15, 70)),
                shading=float(rng.uniform(.15, .55)),
                gradient=float(rng.uniform(0, .45)), gradient_angle=float(rng.uniform(0, 360)),
                vignette=float(rng.uniform(0, .45)), shadows=int(rng.integers(0, 3)),
                shadow_seed=int(rng.integers(2**31)), cast_shadow=float(rng.uniform(.3, .6)))


def relight(frames_rgb, disparity, look: dict, keep=None):
    """Apply one episode's lighting to every frame: directional shading from the estimated
    depth, shadows cast by the kept objects onto the table away from the light, a light
    falloff across the frame, colour cast, exposure and drifting soft shadows."""
    count, height, width = disparity.shape
    az, el = np.radians(look['light_azimuth']), np.radians(look['light_elevation'])
    light = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    y, x = np.mgrid[0:height, 0:width].astype(np.float32)
    ga = np.radians(look['gradient_angle'])
    ramp = ((x - width / 2) * np.cos(ga) + (y - height / 2) * np.sin(ga)) / (width / 2)
    gradient = 1 + look['gradient'] * np.clip(ramp, -1, 1)
    radius = np.hypot((x - width / 2) / (width / 2), (y - height / 2) / (height / 2)) / np.sqrt(2)
    vignette = 1 - look['vignette'] * radius ** 2
    rng = np.random.default_rng(look['shadow_seed'])
    blobs = []
    for _ in range(look['shadows']):
        keys = max(2, count // 60)
        points = rng.uniform(-.1, 1.1, (keys, 2)) * [width, height]
        path = np.stack([np.interp(np.linspace(0, keys - 1, count), np.arange(keys), points[:, k]) for k in range(2)], 1)
        blobs.append((path, rng.uniform([30, 12], [110, 45]), rng.uniform(0, 180), rng.uniform(.3, .6)))
    lo, hi = np.percentile(disparity, [1, 99])
    out = np.empty_like(frames_rgb)
    for i, (frame, d) in enumerate(zip(frames_rgb, disparity)):
        d = cv2.GaussianBlur(((d - lo) / max(hi - lo, 1e-6)).astype(np.float32), (0, 0), 5)
        gx, gy = cv2.Sobel(d, cv2.CV_32F, 1, 0, ksize=5), cv2.Sobel(d, cv2.CV_32F, 0, 1, ksize=5)
        normal = np.dstack([-gx * 4, -gy * 4, np.ones_like(d)])
        normal /= np.linalg.norm(normal, axis=-1, keepdims=True)
        lambert = np.clip(normal @ light, 0, 1)
        shade = 1 - look['shading'] + look['shading'] * 1.25 * lambert
        shadow = np.zeros((height, width), np.float32)
        for path, axes, angle, darkness in blobs:
            cv2.ellipse(shadow, tuple(int(v) for v in path[i]), tuple(int(a) for a in axes), angle, 0, 360, darkness, -1)
        if blobs:
            shadow = cv2.GaussianBlur(shadow, (0, 0), 18)
        if keep is not None and look.get('cast_shadow'):
            # The repaint removed the real shadows on the table: cast the arms, parts and hands
            # onto it, away from the new light, longer when the light is low.
            reach = min(60., 18. / max(np.tan(el), .25))
            shift = np.float32([[1, 0, -np.cos(az) * reach], [0, 1, np.sin(az) * reach]])
            caster = cv2.warpAffine(keep[i].astype(np.float32), shift, (width, height))
            cast = cv2.GaussianBlur(caster, (0, 0), 3 + reach / 8) * look['cast_shadow']
            shadow = np.maximum(shadow, np.where(keep[i], 0, cast))
        factor = (shade * gradient * vignette * (1 - np.clip(shadow, 0, .7)))[..., None] * look['gain']
        image = np.clip(frame.astype(np.float32) / 255 * factor * np.array(look['tint']), 0, 1) ** look['gamma']
        out[i] = (image * 255).astype(np.uint8)
    return out


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


def render_job(root, dest, episode, restyle_seed, settings, args_relight=True):
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
                                    controls=controls(frames, disparity, keep))
        if args_relight:
            look = sample_lighting(restyle_seed)  # same lighting for all three cameras of the episode
            restyled = relight(restyled, disparity, look, keep)
            report['lighting'] = look
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
                    restyle={k: v for k, v in asdict(settings).items() if k != 'prompts'}, relight=args.relight)
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
        futures = {pool.submit(render_job, args.root, dests[i], items[i][0], items[i][1], settings, args.relight): i
                   for i in todo}
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
                restyled = restyler.episode(frames, None, None, keep, args.seed + k, camera, ~keep,
                                            controls=controls(frames, disparity, keep))
                columns += [restyled, relight(restyled, disparity, sample_lighting(args.seed + k), keep)]
            for i in range(len(frames)):
                rows.append(np.concatenate([c[i] for c in columns], 1))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output), cv2.cvtColor(np.concatenate(rows, 0), cv2.COLOR_RGB2BGR))
    print(f'wrote {args.output} (revision {revision[:8]}): columns original, kept (blue = repainted)'
          + ('' if restyler is None else ', restyled, restyled + relit'))


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
    common.add_argument('--strength', type=float, default=.95)
    common.add_argument('--guidance', type=float, default=1.)
    common.add_argument('--depth-scale', type=float, default=.6)
    common.add_argument('--canny-scale', type=float, default=.6)
    p = sub.add_parser('export', parents=[common])
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--work', type=Path, required=True, help='per-episode render folders (resumable)')
    p.add_argument('--fraction', type=float, default=.5, help='share of episodes that get a restyled copy')
    p.add_argument('--no-originals', dest='include_originals', action='store_false')
    p.add_argument('--no-relight', dest='relight', action='store_false',
                   help='skip the whole-frame lighting change on restyled copies')
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
