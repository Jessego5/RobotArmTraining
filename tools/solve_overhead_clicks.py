#!/usr/bin/env python3
"""Solve the overhead camera from clicked gripper fingertips.

The click page (outputs/overhead_clicks/, published as an artifact) shows real
overhead frames; on each, the two fingertips of the left arm and of the right
arm were clicked. Every frame's joint angles give the 3D position of those
fingertips (the CAD arms of sim/bimanual_cad.py posed from the recorded state),
so the camera follows from a least-squares fit of the projected fingertips to
the clicks: position, aim, roll, field of view, principal point, radial
distortion, and where the right base stands relative to the left. Which of an
arm's two fingers a click belongs to is decided by whichever assignment fits.

The camera model matches tools/fit_cameras.py (`overhead_masks`), so the result
drops into its overlays and score.

    python tools/solve_overhead_clicks.py --clicks outputs/overhead_clicks/clicks.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import mujoco
import numpy as np
from scipy.optimize import least_squares

from sim.bimanual import OVERHEAD_CROP, _look_at
from tools.fit_cameras import OVERHEAD_NAMES, Scene, overhead_masks, overhead_start, scene_table_z, score, targets

TIPS = ('L_finger_tip', 'R_finger_tip', 'right_L_finger_tip', 'right_R_finger_tip')


def load_clicks(path: Path) -> dict:
    """{frame id: [point | None] * 4} from the page's copy box or an ArtifactData export."""
    raw = json.loads(path.read_text())
    if isinstance(raw, list):  # documents: [{doc_id/id, data: {points}}] or [{id, points}]
        raw = {d.get('doc_id', d.get('id')): (d.get('data') or d)['points'] for d in raw}
    return {k: [None if p is None else (float(p['x']), float(p['y'])) for p in v] for k, v in raw.items()}


class Projector:
    def __init__(self, scene: Scene, states):
        self.scene, self.states = scene, states
        m = scene.model
        self.sites = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, n) for n in TIPS]
        self.cam = scene.cam['overhead']

    def tips(self, p):
        """World positions of the four fingertips per frame (the right base moves with p)."""
        self.scene.set_right_base(*p[10:13])
        out = []
        for state in self.states:
            self.scene.pose(state)
            out.append(np.array([self.scene.data.site_xpos[s] for s in self.sites]))
        return np.stack(out)

    def project(self, p, world):
        """Crop pixel coordinates (pixel centres at integers) of world points, shape (..., 2)."""
        px, py, pz, tx, ty, roll, fovy, cx, cy, k1 = p[:10]
        lift = np.array([0, 0, scene_table_z()])
        pos = np.add([px, py, pz], lift)
        w, x, y, z = _look_at(pos, np.add([tx, ty, 0.], lift), roll)
        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, np.array([w, x, y, z]))
        rot = rot.reshape(3, 3)  # camera axes as columns: x right, y up, looking along -z
        local = (world - pos) @ rot
        depth = -local[..., 2]
        xu, yu = local[..., 0] / depth, -local[..., 1] / depth
        xd, yd = xu.copy(), yu.copy()
        for _ in range(20):  # invert xu = xd * (1 + k1 * rd^2)
            scale = 1 + k1 * (xd ** 2 + yd ** 2)
            xd, yd = xu / scale, yu / scale
        f = 240 / np.tan(np.radians(fovy / 2))
        u = 319.5 + cx + xd * f - OVERHEAD_CROP[0]
        v = 239.5 + cy + yd * f - OVERHEAD_CROP[1]
        return np.stack([u, v], -1)


# Gentle pull toward a centred, undistorted lens and the estimated base layout, in
# pixel-equivalents: 10 px of principal-point shift, 0.1 of k1, 2 cm or 2 deg of base
# offset each cost as much as a 1 px click error. Few clicks cannot pull these far.
PRIOR = np.array([10, 10, .1, .02, .02, 2.])
LOW = [-np.inf] * 7 + [-25, -25, -.4, -.05, -.05, -5]
HIGH = [np.inf] * 7 + [25, 25, .4, .05, .05, 5]


def residuals(p, projector, clicks, pairs=True):
    """Pixel errors of every clicked fingertip; each arm's two clicks take the better finger order."""
    pix = projector.project(p, projector.tips(p))
    out = []
    for frame, observed in zip(pix, clicks):
        for arm in (0, 2):
            a, b = observed[arm], observed[arm + 1]
            options = [((a, frame[arm]), (b, frame[arm + 1])), ((a, frame[arm + 1]), (b, frame[arm]))]
            best = None
            for option in options if pairs else options[:1]:
                err = [np.subtract(o, q) for o, q in option if o is not None]
                total = sum(float(e @ e) for e in err)
                if best is None or total < best[0]:
                    best = (total, err)
            for e in best[1]:
                out.extend(e)
            # Keep the residual vector a fixed length whichever points were clicked.
            out.extend([0.] * (4 - 2 * len(best[1])))
    return np.r_[out, p[7:] / PRIOR]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--clicks', type=Path, required=True)
    parser.add_argument('--frames', type=Path, default=ROOT / 'outputs/overhead_clicks/frames.json')
    parser.add_argument('--root', type=Path, default=ROOT / 'outputs/real_gear')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/camera_fit_overhead_clicks.json')
    parser.add_argument('--held-out', type=int, default=6, help='episodes for the outline check')
    args = parser.parse_args()
    frames = json.loads(args.frames.read_text())
    clicks = load_clicks(args.clicks)
    used = [f for f in frames if f['id'] in clicks and any(p is not None for p in clicks[f['id']])]
    states = np.array([f['state'] for f in used])
    observed = [[None if p is None else np.array(p) - .5 for p in clicks[f['id']]] for f in used]
    count = sum(p is not None for o in observed for p in o)
    print(f'{len(used)} frames, {count} clicked fingertips')
    scene = Scene()
    projector = Projector(scene, states)
    start = overhead_start()
    fun = lambda p: residuals(p, projector, observed)
    rms = lambda p: float(np.sqrt(np.sum(fun(p)[:-len(PRIOR)] ** 2) / count))
    print(f'current camera: RMS {rms(start):.1f} px')
    # Pose and zoom first from several starts, then everything.
    rng = np.random.default_rng(0)
    pose = [0, 1, 2, 3, 4, 5, 6]
    best = None
    for k in range(12):
        p0 = start.copy()
        if k:
            p0[pose] += rng.normal(0, [.2, .08, .2, .3, .08, 4, 6])
        sub = lambda x, p0=p0: fun(np.r_[x, p0[7:]])
        r = least_squares(sub, p0[pose], loss='soft_l1', f_scale=3., x_scale=[.1, .1, .1, .1, .1, 1, 1])
        p = np.r_[r.x, p0[7:]]
        if best is None or rms(p) < rms(best):
            best = p
    print(f'pose and zoom: RMS {rms(best):.1f} px')
    scales = [.05, .05, .05, .05, .05, 1, 1, 5, 5, .05, .02, .02, 1]
    r = least_squares(fun, best, loss='soft_l1', f_scale=3., x_scale=scales, bounds=(LOW, HIGH))
    best = r.x
    click_errors = lambda p: fun(p)[:-len(PRIOR)]
    errors = np.linalg.norm(click_errors(best).reshape(-1, 2), axis=1)
    errors = errors[errors > 0]
    print(f'all parameters: RMS {rms(best):.1f} px, median {np.median(errors):.1f} px, '
          f'90th percentile {np.percentile(errors, 90):.1f} px')
    print(dict(zip(OVERHEAD_NAMES, np.round(best, 4).tolist())))
    # Per-frame errors flag a mis-click.
    per = np.linalg.norm(click_errors(best).reshape(len(used), 4, 2), axis=2)
    for f, e in zip(used, per):
        if e.max() > 3 * max(np.median(errors), 1):
            print(f"  check {f['id']} (demo {f['episode']} frame {f['frame']}): errors {np.round(e, 1).tolist()} px")
    # Outline check on held-out episodes (not used for the clicks).
    from tools.augment_real_dataset import Depth
    from tools.fit_cameras import samples
    info = json.loads((args.root / 'meta/info.json').read_text())
    clicked = {f['episode'] for f in frames}
    held = [e for e in np.random.default_rng(1).permutation(info['total_episodes']).tolist() if e not in clicked]
    held = sorted(held[:args.held_out])
    held_frames, held_states = samples(args.root, held, 2)
    real = targets(held_frames['overhead'], Depth())
    before = score(overhead_masks(scene, start, held_states), real)[1]
    after = score(overhead_masks(scene, best, held_states), real)[1]
    print(f'held-out outlines: before {before}\n                   after  {after}')
    result = dict(params=dict(zip(OVERHEAD_NAMES, map(float, best))), rms_px=rms(best),
                  median_px=float(np.median(errors)), clicked_points=count, frames=[f['id'] for f in used],
                  held_out_before=before, held_out_after=after, held_out_episodes=held)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    rows = []
    for frame, state in list(zip(held_frames['overhead'], held_states))[:6]:
        row = []
        for p, colour in ((start, (255, 0, 255)), (best, (0, 255, 0))):
            mask = overhead_masks(scene, p, [state])[0]
            outline = mask ^ cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
            shown = frame.copy()
            shown[outline] = colour
            row.append(shown)
        rows.append(np.concatenate(row, 1))
    image = args.output.with_suffix('.jpg')
    cv2.imwrite(str(image), cv2.cvtColor(np.concatenate(rows, 0), cv2.COLOR_RGB2BGR))
    print(f'wrote {args.output} and {image} (left: current camera, magenta; right: solved, green)')


if __name__ == '__main__':
    main()
