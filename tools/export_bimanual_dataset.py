#!/usr/bin/env python3
"""Render scripted two-arm episodes as a LeRobot dataset shaped like the real one.

Output matches FoxNerdSaysMoo/panthera-gear-carrier-pin-real-20hz feature for
feature, so sim and real can be mixed: 20 Hz, 14-D `observation.state`
(radians; grippers 0 closed .. 2 open), `action` = the state at the next tick
(the real dataset's convention), `observation.velocity`, `observation.effort`
and three 256x256 cameras:

- `overhead`: a 640x480 render cropped to the real [176, 224, 432, 480] window,
  from one of the two real camera setups (before/after the Oct 2 move), with a
  small per-episode pose jitter;
- `wrist_port2` (left) and `wrist_port3` (right): wide renders warped to an
  equidistant fisheye with dark corners, like the real wrist lenses.

`--restyle-fraction` repaints that share of episodes with the structure-locked
diffusion restyle (tools/sim2real_augment.py), steered toward the real look by
real frames from the same camera through an IP-Adapter; part and finger pixels
are pasted back from the render. Check the look first with `preview`:

    python tools/export_bimanual_dataset.py preview --episodes data/bimanual --output outputs/bimanual_preview.jpg
    python tools/export_bimanual_dataset.py export --episodes data/bimanual \\
        --output outputs/lerobot/panthera_bimanual_sim_20hz --restyle-fraction 0.3
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('MUJOCO_GL', 'egl')
import cv2
import mujoco
import numpy as np

from sim.bimanual import ARMS, OVERHEAD_CROP, OVERHEAD_SETUPS, PARTS, build_model

FPS = 20
CAMERAS = {'overhead': 'overhead', 'wrist_port2': 'left/wrist', 'wrist_port3': 'right/wrist'}
REAL_DATASET = 'FoxNerdSaysMoo/panthera-gear-carrier-pin-real-20hz'
JOINTS = [f'{side}_{name}' for side in ARMS for name in (*(f'joint{i}' for i in range(1, 7)), 'gripper')]
WRIST_FOV = 150.        # pinhole render, degrees, before the fisheye warp
FISHEYE_EDGE = 62.      # half-angle (deg) at the middle of each image edge
FISHEYE_LIMIT = 74.     # beyond this the real lens shows dark corners
REAL_LOOK = ('two white and grey aluminium robot arms with white 3D printed fin-ray gripper fingers and black '
             'metal brackets, assembling a small black gear and a steel pin into a silver machined carrier')
PROMPTS = (
    f'a webcam photo of {REAL_LOOK}, light maple butcher-block table with glossy reflections, university lab',
    f'a wide-angle photo of {REAL_LOOK}, wooden workbench with bright ceiling light glare, cluttered lab behind',
    f'a fisheye camera photo of {REAL_LOOK}, light wooden table, fluorescent lighting, slight motion blur',
    f'a photo of {REAL_LOOK} on a pale wood table, overhead lights, realistic, slightly noisy webcam image',
)
NEGATIVE = ('cartoon, 3d render, cgi, illustration, painting, text, watermark, wooden robot, orange robot, '
            'extra robot arms, extra gears, coloured objects, cubes, blocks')


def features():
    vector = lambda: {'dtype': 'float32', 'shape': (14,), 'names': JOINTS}
    image = {'dtype': 'image', 'shape': (256, 256, 3), 'names': ['height', 'width', 'channels']}
    return {'observation.state': vector(), 'action': vector(), 'observation.velocity': vector(),
            'observation.effort': vector(), **{f'observation.images.{c}': dict(image) for c in CAMERAS}}


def fisheye_maps(source: int, out: int):
    """Equidistant fisheye lookup into a square pinhole render of WRIST_FOV."""
    f_out = (out / 2) / np.radians(FISHEYE_EDGE)
    f_src = (source / 2) / np.tan(np.radians(WRIST_FOV / 2))
    v, u = np.mgrid[0:out, 0:out].astype(np.float32) - (out - 1) / 2
    radius = np.hypot(u, v)
    theta = radius / f_out
    scale = np.where(radius > 0, f_src * np.tan(np.minimum(theta, np.radians(89))) / np.maximum(radius, 1e-6), 0)
    map_x = (u * scale + (source - 1) / 2).astype(np.float32)
    map_y = (v * scale + (source - 1) / 2).astype(np.float32)
    return map_x, map_y, theta <= np.radians(FISHEYE_LIMIT)


class Views:
    """The three real camera views of a model, with matching depth and segmentation."""

    def __init__(self, overhead_setup: str, jitter_seed: int | None = None):
        self.model = build_model(overhead=OVERHEAD_SETUPS[overhead_setup])
        self.data = mujoco.MjData(self.model)
        m = self.model
        if jitter_seed is not None:  # the real cameras were not calibrated or fixed exactly
            rng = np.random.default_rng(jitter_seed)
            cam = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, 'overhead')
            m.cam_pos[cam] += rng.uniform(-.02, .02, 3)
            m.cam_fovy[cam] += rng.uniform(-3, 3)
        self.cameras = {name: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, sim) for name, sim in CAMERAS.items()}
        for name in ('wrist_port2', 'wrist_port3'):
            m.cam_fovy[self.cameras[name]] = WRIST_FOV
        self.overhead = mujoco.Renderer(m, 480, 640)
        self.wrist = mujoco.Renderer(m, 960, 960)
        self.maps = fisheye_maps(960, 480)
        self.options = mujoco.MjvOption()
        self.options.sitegroup[:] = 0
        body_names = {b: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or '' for b in range(m.nbody)}
        kept = {b for b, n in body_names.items() if n in PARTS or n.endswith(('/L_finger', '/R_finger'))}
        self.kept_geoms = np.array([g for g in range(m.ngeom) if m.geom_bodyid[g] in kept])
        self.background_geoms = np.array([g for g in range(m.ngeom) if body_names[m.geom_bodyid[g]] == 'table'
                                          or mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) == 'floor'])

    def set(self, qpos):
        self.data.qpos[:] = qpos
        mujoco.mj_forward(self.model, self.data)

    def _shape(self, name, image, nearest=False):
        if name == 'overhead':
            x0, y0, x1, y1 = OVERHEAD_CROP
            return image[y0:y1, x0:x1]
        map_x, map_y, inside = self.maps
        warped = cv2.remap(image, map_x, map_y, cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=-1 if nearest else 0)
        warped[~inside] = -1 if nearest else 0
        return cv2.resize(warped, (256, 256), interpolation=cv2.INTER_NEAREST if nearest else cv2.INTER_AREA)

    def render(self, name, maps=False):
        renderer = self.overhead if name == 'overhead' else self.wrist
        renderer.update_scene(self.data, self.cameras[name], self.options)
        rgb = self._shape(name, renderer.render())
        if not maps:
            return rgb
        renderer.enable_depth_rendering()
        depth = self._shape(name, renderer.render().copy(), nearest=True)
        renderer.disable_depth_rendering()
        renderer.enable_segmentation_rendering()
        seg = renderer.render().copy()
        renderer.disable_segmentation_rendering()
        ids = np.where(seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM), seg[..., 0], -1).astype(np.int32)
        ids = self._shape(name, ids, nearest=True)
        depth = np.where(ids < 0, 5., depth).astype(np.float32)
        return rgb, depth, ids


def real_style_bank(files: int = 3, per_camera: int = 24, cache: Path = ROOT / 'outputs/real_style_bank'):
    """Real frames per camera (from the first `files` real data files, ~80 MB each) to steer the restyle."""
    paths = {c: sorted((cache / c).glob('*.jpg')) for c in CAMERAS}
    if not all(len(p) >= per_camera for p in paths.values()):
        import pyarrow.parquet as pq
        from huggingface_hub import hf_hub_download
        tables = [pq.read_table(hf_hub_download(REAL_DATASET, f'data/chunk-000/file-{i:03d}.parquet',
                                                repo_type='dataset'),
                                columns=[f'observation.images.{c}' for c in CAMERAS]) for i in range(files)]
        rng = np.random.default_rng(0)
        for camera in CAMERAS:
            (cache / camera).mkdir(parents=True, exist_ok=True)
            for k in range(per_camera):
                table = tables[k % len(tables)]
                row = int(rng.integers(len(table)))
                (cache / camera / f'{k:03d}.jpg').write_bytes(
                    table[f'observation.images.{camera}'][row].as_py()['bytes'])
        paths = {c: sorted((cache / c).glob('*.jpg')) for c in CAMERAS}
    return {c: [cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB) for p in ps] for c, ps in paths.items()}


def restyle_settings(args):
    from tools.sim2real_augment import Settings
    return Settings(batch=args.restyle_batch, steps=args.restyle_steps, strength=args.restyle_strength,
                    guidance=args.restyle_guidance, depth_scale=args.restyle_depth, canny_scale=args.restyle_canny,
                    ip_scale=args.restyle_ip, prompts=PROMPTS, negative=NEGATIVE)


def render_episode(source: Path, dest: Path, restyle_seed=None, settings=None):
    """Write JPEGs per camera; optionally restyle them. Returns (frames, report)."""
    meta = json.loads((source / 'meta.json').read_text())
    with np.load(source / 'data.npz') as data:
        qpos = data['qpos']
    views = Views(meta['overhead_setup'], jitter_seed=meta['seed'])
    report = dict(overhead_setup=meta['overhead_setup'])
    global restyler, bank
    for camera in CAMERAS:
        rgb, depth, ids = [], [], []
        for q in qpos:
            views.set(q)
            if restyle_seed is None:
                rgb.append(views.render(camera))
            else:
                r, d, g = views.render(camera, maps=True)
                rgb.append(r), depth.append(d), ids.append(g)
        rgb = np.stack(rgb)
        if restyle_seed is not None:
            from tools.sim2real_augment import Restyler, boundary_recall, flicker
            if 'restyler' not in globals():
                restyler = Restyler(settings, ip_adapter=settings.ip_scale > 0)
                bank = real_style_bank()
            ids = np.stack(ids)
            keep = np.isin(ids, views.kept_geoms)
            background = (ids < 0) | np.isin(ids, views.background_geoms)
            reference = bank[camera][restyle_seed % len(bank[camera])]
            restyled = restyler.episode(rgb, np.stack(depth), ids, keep, restyle_seed,
                                        'overhead' if camera == 'overhead' else 'wrist', background, reference)
            sample = np.linspace(0, len(rgb) - 1, min(10, len(rgb))).astype(int)
            report[camera] = dict(outline_recall=float(np.mean([boundary_recall(ids[i], restyled[i]) for i in sample])),
                                  flicker=flicker(restyled, rgb))
            rgb = restyled
        (dest / camera).mkdir(parents=True, exist_ok=True)
        for i, frame in enumerate(rgb):
            cv2.imwrite(str(dest / camera / f'{i:05d}.jpg'), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
    if restyle_seed is not None:
        report.update(seed=restyle_seed, prompt=settings.prompts[restyle_seed % len(settings.prompts)])
        (dest / 'restyle.json').write_text(json.dumps(report, indent=2) + '\n')
    return len(qpos), report


def render_job(source, dest, restyle_seed, settings):
    from lerobot.datasets.compute_stats import compute_episode_stats
    count, report = render_episode(source, dest, restyle_seed, settings)
    image_features = {k: v for k, v in features().items() if v['dtype'] == 'image'}
    buffer = {key: [str(dest / key.rsplit('.', 1)[-1] / f'{j:05d}.jpg') for j in range(count - 1)]
              for key in image_features}
    return count, compute_episode_stats(buffer, image_features), report


def export(args):
    from dataclasses import asdict
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from tools.export_scripted_dataset import save_prepared_episode

    episodes = sorted(p.parent for p in args.episodes.glob('episode_*/data.npz'))
    if args.limit:
        episodes = episodes[:args.limit]
    if not episodes:
        raise SystemExit('no episodes')
    if args.output.exists() or args.rendered.exists():
        raise SystemExit('output or rendered directory exists; use fresh paths')
    settings = restyle_settings(args)
    dataset = LeRobotDataset.create(repo_id='local/panthera_bimanual_sim', root=args.output, fps=FPS,
                                    robot_type='panthera_ht_bimanual_sim', features=features(), use_videos=False,
                                    image_writer_threads=0, metadata_buffer_size=20)
    pool = ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn'))
    pending, restyled, total, started = {}, {}, 0, time.time()

    def submit(i):
        seed = (args.restyle_seed * 100003 + i
                if np.random.default_rng([args.restyle_seed, i]).random() < args.restyle_fraction else None)
        pending[i] = pool.submit(render_job, episodes[i], args.rendered / episodes[i].name, seed, settings)

    for i in range(min(len(episodes), args.workers * 2)):
        submit(i)
    try:
        for i, source in enumerate(episodes):
            count, image_stats, report = pending.pop(i).result()
            if 'seed' in report:
                restyled[source.name] = report
            dest = args.rendered / source.name
            meta = json.loads((source / 'meta.json').read_text())
            with np.load(source / 'data.npz') as data:
                state, velocity, effort = data['state'], data['velocity'], data['effort']
            n = count - 1  # the last observation has no next-state action
            buffer = dataset.episode_buffer
            buffer.update(size=n, task=[meta['task']] * n, timestamp=np.arange(n) / FPS, frame_index=np.arange(n))
            buffer['observation.state'] = state[:-1].astype(np.float32)
            buffer['action'] = state[1:].astype(np.float32)
            buffer['observation.velocity'] = velocity[:-1].astype(np.float32)
            buffer['observation.effort'] = effort[:-1].astype(np.float32)
            for camera in CAMERAS:
                buffer[f'observation.images.{camera}'] = [str(dest / camera / f'{j:05d}.jpg') for j in range(n)]
            save_prepared_episode(dataset, image_stats)
            if not args.keep_jpegs:
                for camera in CAMERAS:
                    shutil.rmtree(dest / camera)
            total += n
            print(json.dumps(dict(episodes=i + 1, of=len(episodes), frames=total, restyled=len(restyled),
                                  elapsed_s=round(time.time() - started))), flush=True)
            if i + args.workers * 2 < len(episodes):
                submit(i + args.workers * 2)
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
        dataset.finalize()
        dataset.stop_image_writer()
    provenance = dict(source='sim', generator='tools/collect_bimanual.py', exporter='tools/export_bimanual_dataset.py',
                      fps=FPS, action_alignment='next 20 Hz state, as in the real dataset',
                      gripper_units='radians, linear placeholder map from sim finger opening (0 closed, 2 open)',
                      effort_units='arm Nm; gripper entry is sim finger force in N',
                      cameras={c: s for c, s in CAMERAS.items()}, episodes=[e.name for e in episodes])
    if args.restyle_fraction:
        provenance['restyle'] = dict(fraction=args.restyle_fraction, seed=args.restyle_seed,
                                     settings={k: v for k, v in asdict(settings).items() if k not in ('prompts',)},
                                     prompts=list(PROMPTS), episodes=restyled)
    (args.output / 'meta/provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    (args.output / 'COMPLETE').write_text(f'{len(episodes)} episodes, {total} frames\n')


def preview(args):
    """One episode, three spread frames: per camera the render, the restyle and a real frame."""
    from dataclasses import replace
    episode = sorted(p.parent for p in args.episodes.glob('episode_*/data.npz'))[args.index]
    meta = json.loads((episode / 'meta.json').read_text())
    with np.load(episode / 'data.npz') as data:
        frames = np.linspace(0, len(data['qpos']) - 1, args.frames + 2)[1:-1].astype(int)
        qpos = data['qpos'][frames]
    views = Views(meta['overhead_setup'], jitter_seed=meta['seed'])
    bank = real_style_bank()
    from tools.sim2real_augment import Restyler
    settings = restyle_settings(args)
    restyler = Restyler(settings, ip_adapter=settings.ip_scale > 0) if not args.no_restyle else None
    rows = []
    for camera in CAMERAS:
        maps = [views.set(q) or views.render(camera, maps=True) for q in qpos]
        rgb, depth, ids = (np.stack(x) for x in zip(*maps))
        out = [rgb]
        if restyler is not None:
            keep = np.isin(ids, views.kept_geoms)
            background = (ids < 0) | np.isin(ids, views.background_geoms)
            restyler.settings = replace(settings)
            out.append(restyler.episode(rgb, depth, ids, keep, args.seed,
                                        'overhead' if camera == 'overhead' else 'wrist', background,
                                        bank[camera][args.seed % len(bank[camera])]))
        out.append(np.stack([bank[camera][(args.seed + k) % len(bank[camera])] for k in range(len(rgb))]))
        for i in range(len(rgb)):
            rows.append(np.concatenate([o[i] for o in out], 1))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output), cv2.cvtColor(np.concatenate(rows, 0), cv2.COLOR_RGB2BGR))
    print(f"wrote {args.output}: rows overhead, wrist_port2 (left), wrist_port3 (right); columns sim"
          f"{'' if restyler is None else ', restyled'}, real")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument('--episodes', type=Path, default=ROOT / 'data/bimanual')
    common.add_argument('--restyle-batch', type=int, default=8)
    common.add_argument('--restyle-steps', type=int, default=8)
    common.add_argument('--restyle-strength', type=float, default=.9)
    common.add_argument('--restyle-guidance', type=float, default=1.5)
    common.add_argument('--restyle-depth', type=float, default=.75, help='depth ControlNet scale')
    common.add_argument('--restyle-canny', type=float, default=.4, help='outline ControlNet scale')
    common.add_argument('--restyle-ip', type=float, default=.8, help='real-frame IP-Adapter scale (0 = off)')
    p = sub.add_parser('export', parents=[common])
    p.add_argument('--output', type=Path, default=ROOT / 'outputs/lerobot/panthera_bimanual_sim_20hz')
    p.add_argument('--rendered', type=Path, default=ROOT / 'outputs/bimanual_rendered')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--limit', type=int, help='first N episodes only')
    p.add_argument('--restyle-fraction', type=float, default=0.)
    p.add_argument('--restyle-seed', type=int, default=0)
    p.add_argument('--keep-jpegs', action='store_true')
    p = sub.add_parser('preview', parents=[common])
    p.add_argument('--output', type=Path, default=ROOT / 'outputs/bimanual_preview.jpg')
    p.add_argument('--index', type=int, default=0, help='which episode')
    p.add_argument('--frames', type=int, default=3)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--no-restyle', action='store_true', help='sim and real only (no GPU needed)')
    args = parser.parse_args()
    {'export': export, 'preview': preview}[args.command](args)


if __name__ == '__main__':
    main()
