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

Lighting starts from the real lab: two ceiling fixtures whose positions were
estimated from the real glare spots (sim/bimanual.py), a dimmer one over the
arms, and a varnished table. `--copies N` renders every demonstration N times:
copy 0 as estimated, the others with augmented looks, each fixed per episode:

- lighting: fixture positions, brightness, warmth, shadow softness, fixtures
  switched off, ambient level, table tone and gloss;
- operator shadows (`--operator-shadows` share): soft dark shapes drifting over
  the table and parts, as the hand-guiding operator's arms cast in real frames;
- webcam response: exposure, white balance, gamma, sensor noise, slight blur.

`--restyle-fraction` repaints that share of rendered episodes with the
structure-locked diffusion restyle (tools/sim2real_augment.py), steered toward
the real look by real frames from the same camera through an IP-Adapter. The
restyle starts from a render without shadows or highlights, and the sim's own
shadows and glare are multiplied back in afterwards (lit render / unlit render),
so they fall exactly where the scene's lights put them. Part and finger pixels
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

from sim.bimanual import ARMS, LIGHTS, OVERHEAD_CROP, OVERHEAD_SETUPS, PARTS, build_model

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

    def __init__(self, overhead_setup: str, jitter_seed: int | None = None, appearance: dict | None = None):
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
        if appearance:
            apply_appearance(m, appearance)
        self.light_specular = m.light_specular.copy()
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

    def render(self, name, maps=False, unlit=False):
        """RGB (and depth/ids); `unlit` drops shadows and specular highlights."""
        renderer = self.overhead if name == 'overhead' else self.wrist
        if unlit:
            self.model.light_specular[:] = 0
        renderer.update_scene(self.data, self.cameras[name], self.options)
        if unlit:
            renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
        rgb = self._shape(name, renderer.render())
        self.model.light_specular[:] = self.light_specular
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


def sample_appearance(rng) -> dict:
    """One episode's lighting and table look around the estimated real lab."""
    lights = {}
    warmth = rng.uniform(-1, 1)  # one room: cool fluorescent .. warm bulbs
    off = rng.choice(['ceiling_left', 'ceiling_right', None, None, None, None])  # a front fixture out, sometimes
    for name, light in LIGHTS.items():
        tint = np.array([1 + .08 * warmth, 1., 1 - .10 * warmth]) * rng.uniform(.97, 1.03, 3)
        lights[name] = dict(
            offset=(rng.normal(0, [.5, .4, .15])).tolist(),
            scale=float(rng.uniform(.75, 1.3)), tint=tint.tolist(),
            softness=float(rng.uniform(.02, .45)),
            on=bool(name != off and (name != 'ceiling_over' or rng.random() < .6)))
    return dict(lights=lights, ambient=float(rng.uniform(.15, .35)),
                table_tone=float(rng.uniform(.85, 1.12)), table_hue=float(rng.uniform(-.04, .04)),
                table_gloss=float(rng.uniform(.2, .85)))


def apply_appearance(model, look: dict) -> None:
    for name, light in look['lights'].items():
        i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_LIGHT, name)
        model.light_pos[i] += light['offset']
        target = np.array([.35, 0., 0.]) - model.light_pos[i]
        model.light_dir[i] = target / np.linalg.norm(target)
        model.light_diffuse[i] *= np.array(light['tint']) * light['scale']
        model.light_specular[i] *= light['scale']
        model.light_bulbradius[i] = light['softness']
        model.light_active[i] = light['on']
    model.vis.headlight.ambient[:] = look['ambient']
    table = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MATERIAL, 'table')
    rgba = model.mat_rgba[table, :3] * look['table_tone'] + np.array([1, 0, -1]) * look['table_hue']
    model.mat_rgba[table, :3] = np.clip(rgba, 0, 1)
    model.mat_specular[table] = look['table_gloss']


def sample_camera_response(rng) -> dict:
    """Webcam exposure, white balance, gamma, noise and blur for one camera of one episode."""
    return dict(gain=float(rng.uniform(.8, 1.2)), white=rng.uniform(.92, 1.08, 3).tolist(),
                gamma=float(rng.uniform(.88, 1.12)), noise=float(rng.uniform(0, 5)),
                blur=float(rng.uniform(.4, 1.1)) if rng.random() < .3 else 0.)


def camera_response(frame: np.ndarray, response: dict, rng) -> np.ndarray:
    image = frame.astype(np.float32) / 255
    image = np.clip(image * response['gain'] * np.array(response['white']), 0, 1) ** response['gamma']
    if response['blur']:
        image = cv2.GaussianBlur(image, (0, 0), response['blur'])
    image = image * 255 + rng.normal(0, response['noise'], image.shape)
    return np.clip(image, 0, 255).astype(np.uint8)


class OperatorShadows:
    """Soft dark shapes drifting smoothly over a camera's view, like the operator's arms in real frames."""

    def __init__(self, rng, frames: int, camera: str):
        self.blobs = []
        for _ in range(int(rng.integers(1, 3))):
            keys = max(2, frames // 60)
            # The operator stands behind the arms: shadows enter from the bottom of the overhead view.
            lo, hi = ((.1, .55), (.9, 1.1)) if camera == 'overhead' else ((-.1, -.1), (1.1, 1.1))
            points = rng.uniform(lo, hi, (keys, 2)) * 256
            self.blobs.append(dict(
                path=np.stack([np.interp(np.linspace(0, keys - 1, frames), np.arange(keys), points[:, k])
                               for k in range(2)], 1),
                axes=rng.uniform([40, 15], [120, 45]), angle=float(rng.uniform(0, 180)),
                spin=float(rng.uniform(-.2, .2)), darkness=float(rng.uniform(.35, .65)),
                blur=float(rng.uniform(10, 25))))

    def apply(self, frame: np.ndarray, i: int) -> np.ndarray:
        mask = np.zeros(frame.shape[:2], np.float32)
        for blob in self.blobs:
            centre = tuple(int(v) for v in blob['path'][i])
            cv2.ellipse(mask, centre, tuple(int(a) for a in blob['axes']), blob['angle'] + blob['spin'] * i,
                        0, 360, blob['darkness'], -1)
        mask = cv2.GaussianBlur(mask, (0, 0), self.blobs[0]['blur'])
        return (frame * (1 - np.clip(mask, 0, .7))[..., None]).astype(np.uint8)


def transfer_shading(restyled: np.ndarray, lit: np.ndarray, unlit: np.ndarray) -> np.ndarray:
    """Multiply the sim's shadows and highlights (lit / unlit luminance) into a restyled frame."""
    luminance = lambda x: x.astype(np.float32).mean(-1) + 4
    ratio = np.clip(luminance(lit) / luminance(unlit), .25, 2.5)[..., None]
    return np.clip(restyled * ratio, 0, 255).astype(np.uint8)


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


def episode_look(meta: dict, copy: int, operator_share: float) -> dict:
    """Per-(episode, copy) augmentation; copy 0 is the estimated real lab with no extras."""
    rng = np.random.default_rng([meta['seed'], copy])
    if copy == 0:
        return dict(copy=0)
    return dict(copy=copy, appearance=sample_appearance(rng),
                response={c: sample_camera_response(rng) for c in CAMERAS},
                operator_shadows=bool(rng.random() < operator_share), noise_seed=int(rng.integers(2**31)))


def render_episode(source: Path, dest: Path, look: dict, restyle_seed=None, settings=None):
    """Write JPEGs per camera, augmented and optionally restyled. Returns (frames, report)."""
    meta = json.loads((source / 'meta.json').read_text())
    with np.load(source / 'data.npz') as data:
        qpos = data['qpos']
    views = Views(meta['overhead_setup'], jitter_seed=meta['seed'] + 7919 * look['copy'],
                  appearance=look.get('appearance'))
    report = dict(source=source.name, overhead_setup=meta['overhead_setup'], **look)
    global restyler, bank
    for camera in CAMERAS:
        lit, unlit, depth, ids = [], [], [], []
        for q in qpos:
            views.set(q)
            if restyle_seed is None:
                lit.append(views.render(camera))
            else:
                r, d, g = views.render(camera, maps=True)
                lit.append(r), depth.append(d), ids.append(g)
                unlit.append(views.render(camera, unlit=True))
        rgb = np.stack(lit)
        if restyle_seed is not None:
            from tools.sim2real_augment import Restyler, boundary_recall, flicker
            if 'restyler' not in globals():
                restyler = Restyler(settings, ip_adapter=settings.ip_scale > 0)
                bank = real_style_bank()
            ids, unlit = np.stack(ids), np.stack(unlit)
            keep = np.isin(ids, views.kept_geoms)
            background = (ids < 0) | np.isin(ids, views.background_geoms)
            reference = bank[camera][restyle_seed % len(bank[camera])]
            # Restyle the unlit render, then put the scene's shadows and glare back exactly.
            restyled = restyler.episode(unlit, np.stack(depth), ids, keep, restyle_seed,
                                        'overhead' if camera == 'overhead' else 'wrist', background, reference)
            restyled = np.stack([transfer_shading(r, a, b) for r, a, b in zip(restyled, rgb, unlit)])
            sample = np.linspace(0, len(rgb) - 1, min(10, len(rgb))).astype(int)
            report[f'{camera}_restyle'] = dict(
                outline_recall=float(np.mean([boundary_recall(ids[i], restyled[i]) for i in sample])),
                flicker=flicker(restyled, rgb))
            rgb = restyled
        if look['copy']:
            rng = np.random.default_rng([look['noise_seed'], list(CAMERAS).index(camera)])
            shadows = OperatorShadows(rng, len(rgb), camera) if look['operator_shadows'] else None
            rgb = np.stack([camera_response(shadows.apply(f, i) if shadows else f, look['response'][camera], rng)
                            for i, f in enumerate(rgb)])
        (dest / camera).mkdir(parents=True, exist_ok=True)
        for i, frame in enumerate(rgb):
            cv2.imwrite(str(dest / camera / f'{i:05d}.jpg'), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
    if restyle_seed is not None:
        report.update(restyle_seed=restyle_seed, prompt=settings.prompts[restyle_seed % len(settings.prompts)])
    (dest / 'look.json').write_text(json.dumps(report, indent=2) + '\n')
    return len(qpos), report


def render_job(source, dest, look, restyle_seed, settings):
    """Render one (episode, copy) into `dest`; `done.json` marks it finished for resuming."""
    import pickle
    from lerobot.datasets.compute_stats import compute_episode_stats
    shutil.rmtree(dest, ignore_errors=True)  # a half-rendered folder from an interrupted run
    dest.mkdir(parents=True)
    count, report = render_episode(source, dest, look, restyle_seed, settings)
    image_features = {k: v for k, v in features().items() if v['dtype'] == 'image'}
    buffer = {key: [str(dest / key.rsplit('.', 1)[-1] / f'{j:05d}.jpg') for j in range(count - 1)]
              for key in image_features}
    (dest / 'image_stats.pkl').write_bytes(pickle.dumps(compute_episode_stats(buffer, image_features)))
    (dest / 'done.json').write_text(json.dumps(dict(count=count, report=report), default=str) + '\n')


def export(args):
    """Render every (episode, copy) to JPEG folders, then write the dataset in one pass.

    Rendering (and restyling) is the slow part and resumes: rerunning the same
    command skips folders already marked done. The dataset itself is rebuilt from
    the rendered folders in minutes, so an interrupted write is simply redone.
    """
    import pickle
    from dataclasses import asdict
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from tools.export_scripted_dataset import save_prepared_episode

    episodes = sorted(p.parent for p in args.episodes.glob('episode_*/data.npz'))
    if args.limit:
        episodes = episodes[:args.limit]
    if not episodes:
        raise SystemExit('no episodes')
    items = [(episode, copy) for copy in range(args.copies) for episode in episodes]
    if (args.output / 'COMPLETE').exists():
        raise SystemExit(f'{args.output} is already complete')
    settings = restyle_settings(args)
    contract = dict(episodes=str(args.episodes.resolve()), count=len(items), copies=args.copies,
                    operator_shadows=args.operator_shadows, restyle_fraction=args.restyle_fraction,
                    restyle_seed=args.restyle_seed,
                    restyle={k: v for k, v in asdict(settings).items() if k != 'prompts'})
    args.rendered.mkdir(parents=True, exist_ok=True)
    contract_path = args.rendered / 'settings.json'
    if contract_path.exists() and json.loads(contract_path.read_text()) != json.loads(json.dumps(contract)):
        raise SystemExit(f'{args.rendered} was rendered with other settings; use a fresh --rendered path')
    contract_path.write_text(json.dumps(contract, indent=2) + '\n')

    dests = [args.rendered / f'{source.name}_c{copy}' for source, copy in items]
    todo = [i for i, dest in enumerate(dests) if not (dest / 'done.json').exists()]
    print(f'{len(items) - len(todo)} of {len(items)} episodes already rendered; rendering {len(todo)}', flush=True)
    started = time.time()
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        futures = {}
        for i in todo:
            source, copy = items[i]
            seed = (args.restyle_seed * 100003 + i
                    if np.random.default_rng([args.restyle_seed, i]).random() < args.restyle_fraction else None)
            look = episode_look(json.loads((source / 'meta.json').read_text()), copy, args.operator_shadows)
            futures[pool.submit(render_job, source, dests[i], look, seed, settings)] = i
        from concurrent.futures import as_completed
        for k, future in enumerate(as_completed(futures), 1):
            future.result()
            elapsed = time.time() - started
            print(json.dumps(dict(rendered=k, of=len(todo), elapsed_s=round(elapsed),
                                  eta_h=round(elapsed / k * (len(todo) - k) / 3600, 2))), flush=True)

    # Write the dataset from the rendered folders; a partial earlier write is discarded.
    shutil.rmtree(args.output, ignore_errors=True)
    dataset = LeRobotDataset.create(repo_id='local/panthera_bimanual_sim', root=args.output, fps=FPS,
                                    robot_type='panthera_ht_bimanual_sim', features=features(), use_videos=False,
                                    image_writer_threads=0, metadata_buffer_size=20)
    looks, total = {}, 0
    try:
        for i, ((source, copy), dest) in enumerate(zip(items, dests)):
            done = json.loads((dest / 'done.json').read_text())
            looks[dest.name] = done['report']
            meta = json.loads((source / 'meta.json').read_text())
            with np.load(source / 'data.npz') as data:
                state, velocity, effort = data['state'], data['velocity'], data['effort']
            n = done['count'] - 1  # the last observation has no next-state action
            buffer = dataset.episode_buffer
            buffer.update(size=n, task=[meta['task']] * n, timestamp=np.arange(n) / FPS, frame_index=np.arange(n))
            buffer['observation.state'] = state[:-1].astype(np.float32)
            buffer['action'] = state[1:].astype(np.float32)
            buffer['observation.velocity'] = velocity[:-1].astype(np.float32)
            buffer['observation.effort'] = effort[:-1].astype(np.float32)
            for camera in CAMERAS:
                buffer[f'observation.images.{camera}'] = [str(dest / camera / f'{j:05d}.jpg') for j in range(n)]
            save_prepared_episode(dataset, pickle.loads((dest / 'image_stats.pkl').read_bytes()))
            total += n
            if (i + 1) % 25 == 0 or i + 1 == len(items):
                print(json.dumps(dict(written=i + 1, of=len(items), frames=total)), flush=True)
    finally:
        dataset.finalize()
        dataset.stop_image_writer()
    provenance = dict(source='sim', generator='tools/collect_bimanual.py', exporter='tools/export_bimanual_dataset.py',
                      fps=FPS, action_alignment='next 20 Hz state, as in the real dataset',
                      gripper_units='radians, linear placeholder map from sim finger opening (0 closed, 2 open)',
                      effort_units='arm Nm; gripper entry is sim finger force in N',
                      cameras={c: s for c, s in CAMERAS.items()}, copies=args.copies,
                      operator_shadow_share=args.operator_shadows,
                      lights={k: dict(v, pos=v['pos'].tolist()) for k, v in LIGHTS.items()},
                      episodes=looks)
    if args.restyle_fraction:
        provenance['restyle'] = dict(fraction=args.restyle_fraction, seed=args.restyle_seed,
                                     settings={k: v for k, v in asdict(settings).items() if k not in ('prompts',)},
                                     prompts=list(PROMPTS), shading='sim shadows/highlights multiplied back in')
    (args.output / 'meta/provenance.json').write_text(json.dumps(provenance, indent=2, default=str) + '\n')
    (args.output / 'COMPLETE').write_text(f'{len(items)} episodes, {total} frames\n')
    if not args.keep_jpegs:
        for dest in dests:
            for camera in CAMERAS:
                shutil.rmtree(dest / camera, ignore_errors=True)
    print(f'complete: {len(items)} episodes, {total} frames at {args.output}', flush=True)


def preview(args):
    """One episode, spread frames; per camera: estimated lab | augmented look | restyled | real."""
    from dataclasses import replace
    episode = sorted(p.parent for p in args.episodes.glob('episode_*/data.npz'))[args.index]
    meta = json.loads((episode / 'meta.json').read_text())
    with np.load(episode / 'data.npz') as data:
        frames = np.linspace(0, len(data['qpos']) - 1, args.frames + 2)[1:-1].astype(int)
        qpos = data['qpos'][frames]
    look = episode_look(meta, 1 + args.seed, 1.)
    base = Views(meta['overhead_setup'], jitter_seed=meta['seed'])
    varied = Views(meta['overhead_setup'], jitter_seed=meta['seed'] + 7919, appearance=look['appearance'])
    bank = real_style_bank()
    settings = restyle_settings(args)
    restyler = None
    if not args.no_restyle:
        from tools.sim2real_augment import Restyler
        restyler = Restyler(settings, ip_adapter=settings.ip_scale > 0)
        # Preview frames are seconds apart, so carrying the still background over
        # from the previous frame (the export's anti-flicker step) would stitch them.
        restyler.settings = replace(settings, carry_static=False)
    rows = []
    for k, camera in enumerate(CAMERAS):
        plain = np.stack([base.set(q) or base.render(camera) for q in qpos])
        maps = [varied.set(q) or (*varied.render(camera, maps=True), varied.render(camera, unlit=True)) for q in qpos]
        lit, depth, ids, unlit = (np.stack(x) for x in zip(*maps))
        rng = np.random.default_rng([look['noise_seed'], k])
        shadows = OperatorShadows(rng, len(lit) * 30, camera)
        finish = lambda images: np.stack([camera_response(shadows.apply(f, 30 * i), look['response'][camera], rng)
                                          for i, f in enumerate(images)])
        out = [plain, finish(lit)]
        if restyler is not None:
            keep = np.isin(ids, varied.kept_geoms)
            background = (ids < 0) | np.isin(ids, varied.background_geoms)
            restyled = restyler.episode(unlit, depth, ids, keep, args.seed, 'overhead' if camera == 'overhead'
                                        else 'wrist', background, bank[camera][args.seed % len(bank[camera])])
            out.append(finish(np.stack([transfer_shading(r, a, b) for r, a, b in zip(restyled, lit, unlit)])))
        out.append(np.stack([bank[camera][(args.seed + i) % len(bank[camera])] for i in range(len(lit))]))
        for i in range(len(lit)):
            rows.append(np.concatenate([o[i] for o in out], 1))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output), cv2.cvtColor(np.concatenate(rows, 0), cv2.COLOR_RGB2BGR))
    print(f"wrote {args.output}: rows overhead, wrist_port2 (left), wrist_port3 (right); columns estimated lab, "
          f"augmented{'' if restyler is None else ', augmented + restyled'}, real")


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
    p.add_argument('--copies', type=int, default=1, help='renders per demo; copies after the first are augmented')
    p.add_argument('--operator-shadows', type=float, default=.5, help='share of augmented copies with operator shadows')
    p.add_argument('--restyle-fraction', type=float, default=0.)
    p.add_argument('--restyle-seed', type=int, default=0)
    p.add_argument('--keep-jpegs', action='store_true', help='keep the rendered JPEGs after writing the dataset')
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
