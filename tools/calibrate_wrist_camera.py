#!/usr/bin/env python3
"""Calibrate a wrist camera from a checkerboard: its fisheye lens and where it sits on the arm.

At the robot (about 10 minutes per camera):

1. Print a checkerboard, glue it to something flat and stiff, and measure one
   square after printing. Tape it flat on the table; do not move it again.
2. Record one ordinary teleop episode: move the arm through about 20 different
   poses with the board in view of that arm's wrist camera (near and far,
   tilted, the board in different parts of the image including near the edges)
   and hold each pose still for about a second.

Then:

    python tools/calibrate_wrist_camera.py --dataset data/calib_left --episode 0 \\
        --camera wrist_port2 --cols 9 --rows 6 --square-mm 25 --output outputs/wrist_port2.json

`--cols/--rows` count the board's *inner* corners. `--dataset` is a local LeRobot
dataset in the real format (14-D state, images per camera); `--captures` takes a
folder of `NAME.jpg` + `NAME.json` (`{"state": [14 values]}` or `{"joints": [6]}`)
instead. wrist_port2 is the left arm (state 0-5), wrist_port3 the right (7-12).

What it does: picks moments where the arm is still and the whole board is seen,
finds the corners (OpenCV), solves the fisheye lens (`cv2.fisheye.calibrate`),
gets the board's pose in each view, and solves the camera's fixed pose on
`link6` of sim/panthera/panthera.xml (`cv2.calibrateHandEye`), with each view's
gripper pose from the recorded joints. The board never moved, so its position
recomputed from every view must agree: that scatter (mm, deg) and the corners
reprojected through the whole chain (px) are the quality report.

    python tools/calibrate_wrist_camera.py --selftest    # synthetic check of the maths
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ARM_XML = ROOT / 'sim/panthera/panthera.xml'
CAMERA_SIDE = {'wrist_port2': 0, 'wrist_port3': 7}  # offset of that arm's six joints in the 14-D state


# ---------------------------------------------------------------- inputs ----

def from_dataset(root: Path, episode: int, camera: str):
    """(images, joints) for one episode of a local LeRobot dataset."""
    import pandas as pd
    import pyarrow.parquet as pq
    meta = pd.concat([pq.read_table(f).to_pandas() for f in sorted(root.glob('meta/episodes/*/*.parquet'))])
    row = meta[meta['episode_index'] == episode].iloc[0]
    path = root / f"data/chunk-{int(row['data/chunk_index']):03d}/file-{int(row['data/file_index']):03d}.parquet"
    table = pq.read_table(path, columns=['episode_index', 'frame_index', 'observation.state',
                                         f'observation.images.{camera}']).to_pandas()
    table = table[table['episode_index'] == episode].sort_values('frame_index')
    off = CAMERA_SIDE[camera]
    joints = np.stack(table['observation.state'].to_numpy())[:, off:off + 6].astype(np.float64)
    images = [cell['bytes'] for cell in table[f'observation.images.{camera}']]
    return images, joints


def from_captures(folder: Path, camera: str):
    images, joints = [], []
    for image in sorted(glob.glob(str(folder / '*.jpg')) + glob.glob(str(folder / '*.png'))):
        meta = json.loads(Path(image).with_suffix('.json').read_text())
        q = meta['joints'] if 'joints' in meta else meta['state'][CAMERA_SIDE[camera]:CAMERA_SIDE[camera] + 6]
        images.append(Path(image).read_bytes())
        joints.append(q)
    return images, np.array(joints, np.float64)


def still_frames(joints, window=3, tol=.004):
    """Frames where no joint moved more than `tol` rad within +-`window` frames."""
    still = []
    for i in range(window, len(joints) - window):
        if np.abs(joints[i - window:i + window + 1] - joints[i]).max() < tol:
            still.append(i)
    return still


def find_board(image_bytes, cols, rows):
    gray = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
    found, corners = cv2.findChessboardCornersSB(gray, (cols, rows), flags=cv2.CALIB_CB_EXHAUSTIVE |
                                                 cv2.CALIB_CB_ACCURACY)
    return (corners.reshape(-1, 2).astype(np.float64) if found else None), gray.shape[::-1]


def board_points(cols, rows, square_m):
    grid = np.zeros((rows * cols, 3))
    grid[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_m
    return grid


# ---------------------------------------------------------------- kinematics

class Arm:
    """Forward kinematics of one Panthera arm (sim/panthera/panthera.xml), in its base frame."""

    def __init__(self, xml=ARM_XML):
        import mujoco
        self.mujoco = mujoco
        self.model = mujoco.MjModel.from_xml_path(str(xml))
        self.data = mujoco.MjData(self.model)
        self.qadr = [self.model.jnt_qposadr[self.model.joint(f'joint{i}').id] for i in range(1, 7)]
        self.link6 = self.model.body('link6').id
        self.cam = self.model.camera('wrist').id

    def link6_pose(self, q):
        self.data.qpos[:] = 0
        self.data.qpos[self.qadr] = q
        self.mujoco.mj_kinematics(self.model, self.data)
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = self.data.xmat[self.link6].reshape(3, 3), self.data.xpos[self.link6]
        return T

    def model_camera_in_link6(self):
        """The CAD model's own wrist camera, as an OpenCV camera (x right, y down, z forward)."""
        T = np.eye(4)
        T[:3, 3] = self.model.cam_pos[self.cam]
        R = np.zeros(9)
        self.mujoco.mju_quat2Mat(R, self.model.cam_quat[self.cam])
        T[:3, :3] = R.reshape(3, 3) @ np.diag([1, -1, -1])  # MuJoCo cameras look along -z with y up
        return T


def rigid(R, t):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, np.ravel(t)
    return T


def rotation_deg(R):
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


# ---------------------------------------------------------------- solve -----

def solve(views, objp, image_size, arm: Arm):
    """views: [(corners Nx2, joints 6)]. Returns the calibration and its quality report."""
    obj = [objp.reshape(1, -1, 3).astype(np.float64) for _ in views]
    img = [c.reshape(1, -1, 2).astype(np.float64) for c, _ in views]
    K, D = np.eye(3), np.zeros((4, 1))
    flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC | cv2.fisheye.CALIB_FIX_SKEW
    rms, K, D, rvecs, tvecs = cv2.fisheye.calibrate(obj, img, image_size, K, D, flags=flags,
                                                     criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                                                               200, 1e-9))
    board_in_cam = [rigid(cv2.Rodrigues(r)[0], t) for r, t in zip(rvecs, tvecs)]
    link6_in_base = [arm.link6_pose(q) for _, q in views]
    best = None
    for name in ('PARK', 'TSAI', 'HORAUD', 'ANDREFF', 'DANIILIDIS'):
        R, t = cv2.calibrateHandEye([T[:3, :3] for T in link6_in_base], [T[:3, 3] for T in link6_in_base],
                                    [T[:3, :3] for T in board_in_cam], [T[:3, 3] for T in board_in_cam],
                                    method=getattr(cv2, f'CALIB_HAND_EYE_{name}'))
        cam_in_link6 = rigid(R, t)
        boards = [L @ cam_in_link6 @ B for L, B in zip(link6_in_base, board_in_cam)]
        spread = float(np.median(np.linalg.norm(np.stack([b[:3, 3] for b in boards]) - np.median(
            np.stack([b[:3, 3] for b in boards]), 0), axis=1)))
        if best is None or spread < best[0]:
            best = (spread, name, cam_in_link6, boards)
    _, method, cam_in_link6, boards = best
    # The board's single pose in the arm base (rotation averaged) and the corners reprojected
    # through the whole chain: arm FK -> camera mount -> lens.
    from scipy.spatial.transform import Rotation
    board = rigid(Rotation.from_matrix(np.stack([b[:3, :3] for b in boards])).mean().as_matrix(),
                  np.median(np.stack([b[:3, 3] for b in boards]), 0))
    errors = []
    for (corners, _), L in zip(views, link6_in_base):
        cam = np.linalg.inv(L @ cam_in_link6) @ board
        projected, _ = cv2.fisheye.projectPoints(objp.reshape(1, -1, 3), cv2.Rodrigues(cam[:3, :3])[0],
                                                 cam[:3, 3], K, D)
        errors.append(float(np.sqrt(np.mean(np.sum((projected.reshape(-1, 2) - corners) ** 2, 1)))))
    positions = np.stack([b[:3, 3] for b in boards])
    report = dict(
        views=len(views), lens_rms_px=float(rms), hand_eye_method=method,
        chain_rms_px_median=float(np.median(errors)), chain_rms_px_worst=float(np.max(errors)),
        board_spread_mm_median=float(np.median(np.linalg.norm(positions - board[:3, 3], axis=1)) * 1000),
        board_spread_deg_median=float(np.median([rotation_deg(board[:3, :3].T @ b[:3, :3]) for b in boards])))
    model = arm.model_camera_in_link6()
    report.update(vs_cad_camera_mm=float(np.linalg.norm(cam_in_link6[:3, 3] - model[:3, 3]) * 1000),
                  vs_cad_camera_deg=rotation_deg(model[:3, :3].T @ cam_in_link6[:3, :3]))
    calibration = dict(model='opencv_fisheye', image_size=list(image_size), K=K.tolist(), D=D.ravel().tolist(),
                       camera_in_link6=cam_in_link6.tolist(), board_in_arm_base=board.tolist(),
                       frame_note='camera axes are OpenCV (x right, y down, z forward); link6 and the arm base '
                                  'are sim/panthera/panthera.xml frames')
    return calibration, report, errors


# ---------------------------------------------------------------- self-test -

def selftest(seed=0):
    """Synthetic arm poses, a known lens and mount, a fixed board: the solver must recover them."""
    rng = np.random.default_rng(seed)
    arm = Arm()
    size = (640, 480)
    K = np.array([[230., 0, 322], [0, 228, 236], [0, 0, 1]])
    D = np.array([[.06], [-.02], [.004], [-.001]])
    mount = arm.model_camera_in_link6() @ rigid(cv2.Rodrigues(np.radians([3., -4, 2]))[0], [.004, -.006, .003])
    objp = board_points(9, 6, .025)
    q0 = np.array([-.3, 1.7, 1.0, 0., .6, 0.])  # reaching forward and down over the table
    cam0 = arm.link6_pose(q0) @ mount
    # The board lies flat on the table 18 cm in front of that camera, centred in its view.
    centre = cam0[:3, 3] + cam0[:3, 2] * .18
    board = rigid(np.eye(3), [centre[0] - .1, centre[1] - .0625, centre[2]])
    views = []
    while len(views) < 22:
        q = q0 + rng.normal(0, [.12, .1, .12, .25, .2, .4])
        cam = np.linalg.inv(arm.link6_pose(q) @ mount) @ board
        if cam[2, 3] <= .05:
            continue
        pts, _ = cv2.fisheye.projectPoints(objp.reshape(1, -1, 3), cv2.Rodrigues(cam[:3, :3])[0], cam[:3, 3], K, D)
        pts = pts.reshape(-1, 2)
        if not ((pts > 5).all() and (pts[:, 0] < size[0] - 5).all() and (pts[:, 1] < size[1] - 5).all()):
            continue
        views.append((pts + rng.normal(0, .3, pts.shape), q))
    calibration, report, _ = solve(views, objp, size, arm)
    got = np.array(calibration['camera_in_link6'])
    err_mm = np.linalg.norm(got[:3, 3] - mount[:3, 3]) * 1000
    err_deg = rotation_deg(mount[:3, :3].T @ got[:3, :3])
    focal = np.abs(np.array(calibration['K'])[[0, 1], [0, 1]] - K[[0, 1], [0, 1]]).max()
    print(json.dumps(report, indent=1))
    print(f'recovered mount within {err_mm:.2f} mm and {err_deg:.2f} deg; focal length within {focal:.2f} px')
    ok = err_mm < 1.5 and err_deg < .5 and focal < 2
    print('SELFTEST', 'PASSED' if ok else 'FAILED')
    return ok


# ---------------------------------------------------------------- main ------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset', type=Path, help='local LeRobot dataset with the calibration episode')
    parser.add_argument('--episode', type=int, default=0)
    parser.add_argument('--captures', type=Path, help='folder of NAME.jpg + NAME.json instead of a dataset')
    parser.add_argument('--camera', choices=sorted(CAMERA_SIDE), default='wrist_port2')
    parser.add_argument('--cols', type=int, default=9, help='inner corners along the board')
    parser.add_argument('--rows', type=int, default=6, help='inner corners across the board')
    parser.add_argument('--square-mm', type=float, default=25.)
    parser.add_argument('--max-views', type=int, default=40)
    parser.add_argument('--min-pose-change', type=float, default=.08, help='rad between chosen views')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/wrist_calibration.json')
    parser.add_argument('--selftest', action='store_true')
    args = parser.parse_args()
    if args.selftest:
        sys.exit(0 if selftest() else 1)
    if not args.dataset and not args.captures:
        parser.error('give --dataset or --captures')
    images, joints = from_captures(args.captures, args.camera) if args.captures else \
        from_dataset(args.dataset, args.episode, args.camera)
    candidates = still_frames(joints) if not args.captures else list(range(len(images)))
    print(f'{len(images)} frames, {len(candidates)} with the arm still')
    objp = board_points(args.cols, args.rows, args.square_mm / 1000)
    views, size, chosen = [], None, []
    for i in candidates:
        if chosen and min(np.abs(joints[i] - joints[j]).max() for j in chosen) < args.min_pose_change:
            continue  # same pose as a view already taken
        corners, size = find_board(images[i], args.cols, args.rows)
        if corners is None:
            continue
        views.append((corners, joints[i]))
        chosen.append(i)
        if len(views) >= args.max_views:
            break
    print(f'{len(views)} distinct still views with the whole board found (frames {chosen})')
    if len(views) < 8:
        raise SystemExit('need at least 8 (aim for 15-20): more poses, held still, with the whole board in view')
    calibration, report, errors = solve(views, objp, size, Arm())
    calibration.update(camera=args.camera, board=dict(cols=args.cols, rows=args.rows, square_mm=args.square_mm),
                       frames=chosen, report=report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(calibration, indent=2) + '\n')
    print(json.dumps(report, indent=1))
    worst = sorted(zip(errors, chosen), reverse=True)[:3]
    print('worst views (px, frame):', [(round(e, 2), f) for e, f in worst])
    good = report['chain_rms_px_median'] < 3 and report['board_spread_mm_median'] < 3
    print(f"wrote {args.output} ({'looks good' if good else 'check it: large errors; remove bad views or record more poses'})")


if __name__ == '__main__':
    main()
