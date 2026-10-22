#!/usr/bin/env python3
"""Calibrate the fixed overhead camera: its lens, and its pose in the robot's frame from a sheet of AprilTags.

At the robot:

1. Lens (once per webcam, unless its zoom or resolution changes): save about 15
   full-resolution overhead images of the printed checkerboard used for the wrist
   cameras, held at different places, distances and tilts, including near the
   image edges, in one folder.
2. Print the tag sheet (`--make-sheet sheet.png`: a 3 x 2 grid of tag36h11 tags,
   ids 0-5, with its outer corners numbered 1-4 and a 50 mm scale bar) at 100% on
   A3, check the scale bar, measure one tag's black square and the gap between
   tags, and tape it flat on the table, face up, where the overhead camera sees
   it. Save one full-resolution overhead image. It may lie outside the 256 px crop
   the policy is trained on.
3. Tie the sheet to the robot: record one ordinary teleop episode in which the left
   arm, gripper closed, touches the sheet's 4 numbered outer corners in order 1, 2,
   3, 4, holding each touch still for about 2 seconds. The same with the right arm
   in another episode also measures where the right base sits relative to the left.

Why a sheet and not one tag: a 10 cm tag about a metre away is ~40 px wide in the
overhead image, and the tilt of 4 corners that close together is uncertain by
degrees, which the metre to the camera turns into centimetres. Six tags spread
over 36 cm give 24 corners across the image. The sheet lies flat, face up, so
only its position and turn on the table are fitted to the touches; a rectangle
looks the same turned half round, so the touch order is what ties the touches to
the detected tags, and a wrong order is refused.

    python tools/calibrate_overhead_camera.py --lens-images calib/overhead_board \\
        --cols 9 --rows 6 --square-mm 25 --sheet-image calib/sheet.jpg --tag-mm 100 --gap-mm 30 \\
        --touch-dataset data/calib_touch --left-episode 0 --right-episode 1 \\
        --output outputs/overhead_calibration.json

The result is the lens (OpenCV model, full 640x480 frame) and the camera's pose in
the left arm's base frame (sim/panthera/panthera.xml), plus the right base's pose
in that frame when it touched the sheet too. Dataset overhead images are the crop
[176, 224, 432, 480]: subtract (176, 224) from full-frame pixels.

    python tools/calibrate_overhead_camera.py --selftest    # synthetic check of the maths
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.calibrate_wrist_camera import Arm, board_points, rigid, still_frames

CROP = (176, 224, 432, 480)  # the dataset's overhead crop of the 640x480 frame (x0, y0, x1, y1)
GRID = (3, 2)  # tags across, down
DICTIONARY = cv2.aruco.DICT_APRILTAG_36h11


# ---------------------------------------------------------------- lens ------

def calibrate_lens(images, cols, rows, square_m):
    objp = board_points(cols, rows, square_m).astype(np.float32)
    obj, img, size, used = [], [], None, []
    for path in images:
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        size = gray.shape[::-1]
        found, corners = cv2.findChessboardCornersSB(gray, (cols, rows), flags=cv2.CALIB_CB_EXHAUSTIVE |
                                                     cv2.CALIB_CB_ACCURACY)
        if found:
            obj.append(objp)
            img.append(corners.reshape(-1, 1, 2).astype(np.float32))
            used.append(str(path))
    if len(obj) < 6:
        raise SystemExit(f'the board was found in only {len(obj)} images; need at least 6 (aim for 15)')
    rms, K, dist, _, _ = cv2.calibrateCamera(obj, img, size, None, None)
    return dict(K=K.tolist(), dist=dist.ravel().tolist(), image_size=list(size), lens_rms_px=float(rms),
                lens_images=used)


# ---------------------------------------------------------------- sheet -----

class Sheet:
    """The printed grid of tags: its board geometry and its four numbered outer corners."""

    def __init__(self, tag_m, gap_m):
        self.dictionary = cv2.aruco.getPredefinedDictionary(DICTIONARY)
        self.board = cv2.aruco.GridBoard(GRID, tag_m, gap_m, self.dictionary)
        points = np.concatenate([np.asarray(p, np.float64).reshape(4, 3) for p in self.board.getObjPoints()])
        lo, hi = points.min(0), points.max(0)
        box = np.array([[lo[0], lo[1], 0], [hi[0], lo[1], 0], [hi[0], hi[1], 0], [lo[0], hi[1], 0]])
        # Number the outer corners as they appear on the print: 1 top-left, then clockwise.
        image, H = self.render(800)
        pix = cv2.perspectiveTransform(box[None, :, :2].astype(np.float32), H)[0]
        centre = pix.mean(0)
        angle = np.arctan2(pix[:, 1] - centre[1], pix[:, 0] - centre[0])  # image y points down
        order = np.argsort(angle)  # top-left (-135 deg) first, then clockwise on screen
        start = int(np.argmin([np.hypot(*(pix[i])) for i in order]))
        self.corners = box[np.roll(order, -start)]  # touch order, in board coordinates
        self.corner_pixels = pix[np.roll(order, -start)]
        # Which way the board's z points when the sheet lies face up: x across x y up the print.
        across = self.corners[1] - self.corners[0]
        up = self.corners[0] - self.corners[3]
        self.face_up = np.diag([1., 1, 1]) if np.cross(across, up)[2] > 0 else np.diag([1., -1, -1])

    def render(self, width):
        margin = width // 20  # the detector needs white around the outer tags
        w, h = self.extent()
        size = (width, int(np.ceil((width - 2 * margin) * h / w)) + 2 * margin + 2)  # a little slack for rounding
        image = self.board.generateImage(size, marginSize=margin, borderBits=1)
        corners, ids, _ = cv2.aruco.ArucoDetector(self.dictionary).detectMarkers(image)
        obj, img = self.board.matchImagePoints(corners, ids)
        H, _ = cv2.findHomography(obj.reshape(-1, 3)[:, :2], img.reshape(-1, 2))
        return image, H

    def extent(self):
        points = np.concatenate([np.asarray(p).reshape(4, 3) for p in self.board.getObjPoints()])
        return points.max(0)[:2] - points.min(0)[:2]

    def detect(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        corners, ids, _ = cv2.aruco.ArucoDetector(self.dictionary).detectMarkers(gray)
        if ids is None or len(ids) < 2:
            raise SystemExit(f"found {0 if ids is None else len(ids)} of the sheet's 6 tags; need at least 2")
        obj, img = self.board.matchImagePoints(corners, ids)
        return obj.reshape(-1, 3).astype(np.float64), img.reshape(-1, 2).astype(np.float64), sorted(ids.ravel().tolist())


def camera_from_sheet(obj, img, K, dist):
    """The sheet's pose in the camera frame, and the corners' reprojection error (px)."""
    K, dist = np.array(K), np.array(dist)
    ok, rvec, tvec = cv2.solvePnP(obj, img, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    rvec, tvec = cv2.solvePnPRefineLM(obj, img, K, dist, rvec, tvec)
    projected, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
    error = float(np.sqrt(np.mean(np.sum((projected.reshape(-1, 2) - img) ** 2, 1))))
    return rigid(cv2.Rodrigues(rvec)[0], tvec), error


# ---------------------------------------------------------------- touches ---

def touch_points(arm: Arm, joints, fps=20, min_hold_s=1.5):
    """Fingertip positions (grip_site, between the closed fingertips) at the still holds of an episode."""
    still = set(still_frames(joints))
    holds, start = [], None
    for i in range(len(joints) + 1):
        if i in still and start is None:
            start = i
        elif i not in still and start is not None:
            if i - start >= min_hold_s * fps:
                holds.append((start, i))
            start = None
    if len(holds) < 4:
        raise SystemExit(f'found {len(holds)} still holds of {min_hold_s} s; need the 4 corner touches')
    if len(holds) > 4:  # keep the 4 longest, in time order (e.g. a rest pose at the start or end)
        holds = sorted(sorted(holds, key=lambda h: h[1] - h[0])[-4:])
    site = arm.model.site('grip_site').id
    points = []
    for a, b in holds:
        arm.link6_pose(np.median(joints[a:b], 0))
        points.append(arm.data.site_xpos[site].copy())
    return np.array(points), holds


def fit_sheet(sheet: Sheet, points, limit_m=.006):
    """The sheet's pose in the arm base from its 4 touched corners (in the numbered order).

    The sheet lies flat, face up: only its position and turn about the vertical are fitted.
    A fit worse than `limit_m` means the order (or a touch) was wrong, and stops.
    """
    model = sheet.corners @ sheet.face_up.T  # face-up board corners, z still 0
    mc, pc = model[:, :2].mean(0), points[:, :2].mean(0)
    h = (model[:, :2] - mc).T @ (points[:, :2] - pc)
    yaw = np.arctan2(h[0, 1] - h[1, 0], h[0, 0] + h[1, 1])
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    pose = rigid(Rz @ sheet.face_up, np.r_[pc - Rz[:2, :2] @ mc, points[:, 2].mean()])
    fitted = sheet.corners @ pose[:3, :3].T + pose[:3, 3]
    residual = float(np.sqrt(np.mean(np.sum((fitted - points) ** 2, 1))))
    if residual > limit_m:
        sides = np.linalg.norm(points - np.roll(points, -1, 0), axis=1) * 1000
        expected = np.linalg.norm(sheet.corners - np.roll(sheet.corners, -1, 0), axis=1) * 1000
        raise SystemExit(f'the touches do not fit the flat sheet in the numbered order (off by {residual * 1000:.1f} '
                         f'mm; touched sides {np.round(sides).tolist()} mm, expected {np.round(expected).tolist()}). '
                         'Touch corners 1, 2, 3, 4 as printed, fingertips closed, and hold each 2 s.')
    return pose, residual


def make_sheet(path: Path, sheet: Sheet, tag_mm, gap_mm, dpi=300):
    """A printable sheet at true size: the tags, the numbered outer corners and a 50 mm scale bar."""
    px = lambda mm: int(round(mm / 25.4 * dpi))
    width_mm, height_mm = sheet.extent() * 1000
    # render() keeps 1/20 of the width as white margin on each side: size it so the tags print at true size.
    image, _ = sheet.render(int(round(px(width_mm) * 20 / 18)))
    scale = image.shape[1] / 800  # sheet.corner_pixels were measured on an 800 px render
    label = px(14)
    page = np.full((image.shape[0] + label, image.shape[1]), 255, np.uint8)
    page[:image.shape[0]] = image
    font, thick = cv2.FONT_HERSHEY_SIMPLEX, px(1.2)
    for k, (x, y) in enumerate(sheet.corner_pixels * scale):
        dx = -px(12) if x < image.shape[1] / 2 else px(3)
        dy = -px(3) if y < image.shape[0] / 2 else px(12)
        cv2.putText(page, str(k + 1), (int(x + dx), int(y + dy)), font, px(9) / 22, 0, thick)
    y = image.shape[0] + px(4)
    x0 = int(sheet.corner_pixels[0][0] * scale)
    cv2.line(page, (x0, y), (x0 + px(50), y), 0, px(.8))
    cv2.putText(page, f'50 mm   |   tag36h11 ids 0-5   |   tags {tag_mm:g} mm, gaps {gap_mm:g} mm   |   touch 1, 2, 3, 4',
                (x0 + px(55), y + px(1.5)), font, px(3.5) / 22, 0, px(.4))
    cv2.imwrite(str(path), page)
    return path


def joints_from_dataset(root: Path, episode: int, offset: int):
    import pandas as pd
    import pyarrow.parquet as pq
    meta = pd.concat([pq.read_table(f).to_pandas() for f in sorted(root.glob('meta/episodes/*/*.parquet'))])
    row = meta[meta['episode_index'] == episode].iloc[0]
    path = root / f"data/chunk-{int(row['data/chunk_index']):03d}/file-{int(row['data/file_index']):03d}.parquet"
    table = pq.read_table(path, columns=['episode_index', 'frame_index', 'observation.state']).to_pandas()
    table = table[table['episode_index'] == episode].sort_values('frame_index')
    return np.stack(table['observation.state'].to_numpy())[:, offset:offset + 6].astype(np.float64)


# ---------------------------------------------------------------- self-test -

def selftest(seed=0):
    rng = np.random.default_rng(seed)
    ok = True
    # Lens: synthetic board views through a known webcam lens.
    K = np.array([[420., 0, 330], [0, 418, 244], [0, 0, 1]])
    dist = np.array([-.28, .09, .001, -.0005, -.012])
    objp = board_points(9, 6, .025).astype(np.float32)
    obj, img = [], []
    while len(obj) < 15:
        rvec = rng.normal(0, .35, 3)
        t = np.array([rng.uniform(-.15, .05), rng.uniform(-.1, .05), rng.uniform(.35, .7)])
        pts, _ = cv2.projectPoints(objp, rvec, t, K, dist)
        pts = pts.reshape(-1, 2)
        if (pts > 5).all() and (pts[:, 0] < 635).all() and (pts[:, 1] < 475).all():
            obj.append(objp)
            img.append((pts + rng.normal(0, .25, pts.shape)).reshape(-1, 1, 2).astype(np.float32))
    rms, K2, _, _, _ = cv2.calibrateCamera(obj, img, (640, 480), None, None)
    focal = np.abs(K2[[0, 1], [0, 1]] - K[[0, 1], [0, 1]]).max()
    print(f'lens: rms {rms:.2f} px, focal length within {focal:.2f} px')
    ok &= focal < 4
    # The printed sheet's own image must decode, with all six tags.
    sheet = Sheet(.10, .03)
    image, _ = sheet.render(900)
    ok &= len(sheet.detect(np.pad(image, 40, constant_values=255))[2]) == 6
    # Sheet face up on the table; a camera above and behind the arms like the solved overhead one.
    yaw = np.radians(25.)
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    sheet_in_base = rigid(Rz @ sheet.face_up, [.25, -.15, -.0584])
    eye, target = np.array([.07, 0., 1.0]), np.array([1.05, -.1, 0.])
    z = (target - eye) / np.linalg.norm(target - eye)
    x = np.cross(z, [0, 0, 1.])
    x /= np.linalg.norm(x)
    cam_in_base = rigid(np.column_stack([x, np.cross(z, x), z]), eye)
    obj = np.concatenate([np.asarray(p, np.float64).reshape(4, 3) for p in sheet.board.getObjPoints()])
    in_cam = np.linalg.inv(cam_in_base) @ sheet_in_base
    pixels, _ = cv2.projectPoints(obj, cv2.Rodrigues(in_cam[:3, :3])[0], in_cam[:3, 3], K, dist)
    pixels = pixels.reshape(-1, 2) + rng.normal(0, .2, (len(obj), 2))
    found, error = camera_from_sheet(obj, pixels, K, dist)
    touched = sheet.corners @ sheet_in_base[:3, :3].T + sheet_in_base[:3, 3]
    fitted, residual = fit_sheet(sheet, touched + rng.normal(0, .0015, touched.shape))
    try:  # a mirrored touch order must be refused, not silently fitted
        fit_sheet(sheet, touched[[1, 0, 3, 2]])
        print('a wrong touch order was accepted')
        ok = False
    except SystemExit:
        pass
    cam = fitted @ np.linalg.inv(found)
    err_mm = np.linalg.norm(cam[:3, 3] - cam_in_base[:3, 3]) * 1000
    err_deg = np.degrees(np.arccos(np.clip((np.trace(cam_in_base[:3, :3].T @ cam[:3, :3]) - 1) / 2, -1, 1)))
    print(f'sheet: {len(obj)} corners, reprojection {error:.2f} px; touch fit residual {residual * 1000:.2f} mm; '
          f'camera pose within {err_mm:.1f} mm and {err_deg:.2f} deg')
    ok &= err_mm < 15 and err_deg < 1.
    print('SELFTEST', 'PASSED' if ok else 'FAILED')
    return ok


# ---------------------------------------------------------------- main ------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--lens-images', type=Path, help='folder of full-frame checkerboard images')
    parser.add_argument('--lens-json', type=Path, help='reuse the lens from an earlier output instead')
    parser.add_argument('--cols', type=int, default=9, help='checkerboard inner corners along')
    parser.add_argument('--rows', type=int, default=6, help='checkerboard inner corners across')
    parser.add_argument('--square-mm', type=float, default=25.)
    parser.add_argument('--sheet-image', type=Path, help='full-frame overhead image showing the tag sheet')
    parser.add_argument('--tag-mm', type=float, default=100., help="side of one tag's black square, as printed")
    parser.add_argument('--gap-mm', type=float, default=30., help='gap between neighbouring tags, as printed')
    parser.add_argument('--touch-dataset', type=Path, help='local LeRobot dataset with the corner-touch episodes')
    parser.add_argument('--left-episode', type=int, help='episode where the left arm touches corners 1-4')
    parser.add_argument('--right-episode', type=int, help='the same with the right arm (gives the right base pose)')
    parser.add_argument('--output', type=Path, default=Path('outputs/overhead_calibration.json'))
    parser.add_argument('--make-sheet', type=Path, help='write the printable tag sheet (PNG, 300 dpi) and stop')
    parser.add_argument('--selftest', action='store_true')
    args = parser.parse_args()
    if args.selftest:
        sys.exit(0 if selftest() else 1)
    sheet = Sheet(args.tag_mm / 1000, args.gap_mm / 1000)
    if args.make_sheet:
        print(f'wrote {make_sheet(args.make_sheet, sheet, args.tag_mm, args.gap_mm)}: print at 100% on A3, check '
              'the 50 mm bar, measure a tag and a gap, pass them as --tag-mm and --gap-mm')
        return
    out = {}
    if args.lens_json:
        out.update({k: v for k, v in json.loads(args.lens_json.read_text()).items()
                    if k in ('K', 'dist', 'image_size', 'lens_rms_px', 'lens_images')})
    elif args.lens_images:
        images = sorted(glob.glob(str(args.lens_images / '*.jpg')) + glob.glob(str(args.lens_images / '*.png')))
        out.update(calibrate_lens(images, args.cols, args.rows, args.square_mm / 1000))
    else:
        parser.error('give --lens-images or --lens-json')
    print(f"lens: {out['lens_rms_px']:.2f} px rms over {len(out['lens_images'])} images")
    out.update(model='opencv_pinhole_distortion', crop=list(CROP),
               frame_note='full 640x480 frame; dataset overhead images are the crop [176, 224, 432, 480]; '
                          'poses are 4x4 in the left arm base frame of sim/panthera/panthera.xml')
    if args.sheet_image:
        obj, img, ids = sheet.detect(cv2.imread(str(args.sheet_image)))
        sheet_in_cam, error = camera_from_sheet(obj, img, out['K'], out['dist'])
        out.update(sheet_in_camera=sheet_in_cam.tolist(), sheet_reprojection_px=error, sheet_tags_seen=ids,
                   tag_mm=args.tag_mm, gap_mm=args.gap_mm)
        print(f'sheet: tags {ids}, {len(obj)} corners, reprojection {error:.2f} px')
        if args.touch_dataset and args.left_episode is not None:
            arm = Arm()
            left, holds = touch_points(arm, joints_from_dataset(args.touch_dataset, args.left_episode, 0))
            sheet_in_left, residual = fit_sheet(sheet, left)
            camera = sheet_in_left @ np.linalg.inv(sheet_in_cam)
            out.update(camera_in_left_base=camera.tolist(), left_touch_residual_mm=residual * 1000,
                       left_touch_holds=holds)
            print(f'left touches fit the sheet to {residual * 1000:.1f} mm; camera at '
                  f'{np.round(camera[:3, 3], 3).tolist()} m from the left base')
            if args.right_episode is not None:
                right, holds = touch_points(arm, joints_from_dataset(args.touch_dataset, args.right_episode, 7))
                sheet_in_right, residual = fit_sheet(sheet, right)
                right_base = sheet_in_left @ np.linalg.inv(sheet_in_right)
                out.update(right_base_in_left_base=right_base.tolist(), right_touch_residual_mm=residual * 1000,
                           right_touch_holds=holds)
                print(f'right touches fit to {residual * 1000:.1f} mm; right base at '
                      f'{np.round(right_base[:3, 3], 4).tolist()} m from the left base')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2) + '\n')
    print(f'wrote {args.output}')


if __name__ == '__main__':
    main()
