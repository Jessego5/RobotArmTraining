"""Checks of tools/augment_real_dataset.py that need no models, GPU or network."""
import os
import socket
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

from tools import augment_real_dataset as aug


def wood_table_scene(size=128, square=(40, 40, 88, 88)):
    """A wood-coloured table with a grey object standing on it, and a matching disparity map."""
    image = np.zeros((size, size, 3), np.uint8)
    image[:] = (60, 140, 200)  # BGR wood
    x0, y0, x1, y1 = square
    image[y0:y1, x0:x1] = (128, 128, 128)
    yy = np.mgrid[0:size, 0:size][0].astype(np.float32)
    disparity = .3 + .4 * yy / size  # the table tilts away from the camera
    disparity[y0:y1, x0:x1] += .3  # the object rises above it
    return image, disparity, square


class MaskTest(unittest.TestCase):
    def test_keeps_the_object_and_repaints_the_table(self):
        image, disparity, (x0, y0, x1, y1) = wood_table_scene()
        for clean in (True, False):
            with self.subTest(clean=clean):
                keep = aug.keep_mask(image, disparity, clean=clean)
                table = np.ones_like(keep)
                table[y0 - 12:y1 + 12, x0 - 12:x1 + 12] = False
                self.assertGreater(keep[y0 + 4:y1 - 4, x0 + 4:x1 - 4].mean(), .95)
                self.assertLess(keep[table].mean(), .05)

    def test_scramble_never_touches_kept_pixels(self):
        rng = np.random.default_rng(0)
        frames = rng.integers(0, 255, (3, 64, 64, 3), dtype=np.uint8)
        keep = np.zeros((3, 64, 64), bool)
        keep[:, 20:40, 10:30] = True
        for realistic in (True, False):
            with self.subTest(realistic=realistic):
                out, kind = aug.scramble_background(frames, keep, seed=5, realistic=realistic)
                self.assertTrue(np.array_equal(out[keep], frames[keep]))
                self.assertFalse(np.array_equal(out[~keep], frames[~keep]))
                self.assertTrue(kind.startswith('realistic' if realistic else 'wild'))

    def test_maps_round_trip(self):
        rng = np.random.default_rng(1)
        keep = rng.random((5, 64, 64)) > .5
        disparity = cv2.GaussianBlur(rng.random((5, 64, 64)).astype(np.float32), (0, 0), 4)
        with tempfile.TemporaryDirectory() as tmp:
            aug.save_maps(Path(tmp) / 'maps.npz', {camera: (keep, disparity) for camera in aug.CAMERAS})
            loaded = aug.load_maps(Path(tmp) / 'maps.npz')
        for camera in aug.CAMERAS:
            self.assertTrue(np.array_equal(loaded[camera][0], keep))  # masks are stored exactly
            self.assertEqual(loaded[camera][1].shape, disparity.shape)


class RelightTest(unittest.TestCase):
    def test_still_scene_stays_still_when_depth_scale_jitters(self):
        """Monocular depth has its own scale and offset in every frame; a still scene must not flicker."""
        rng = np.random.default_rng(2)
        frame = cv2.GaussianBlur(rng.integers(60, 200, (96, 96, 3), dtype=np.uint8), (0, 0), 3)
        frames = np.repeat(frame[None], 30, 0)
        yy, xx = np.mgrid[0:96, 0:96].astype(np.float32)
        base = yy / 96 + .2 * np.sin(xx / 9)
        disparity = np.stack([base * rng.uniform(.6, 1.4) + rng.uniform(-.3, .3) for _ in range(30)]).astype(np.float32)
        look = aug.sample_lighting(3)
        look.update(shadows=0)  # drifting shadow blobs move on purpose
        out = aug.relight(frames, disparity, look, camera='overhead').astype(np.float32).mean(-1)
        jumps = np.abs(np.diff(out.mean((1, 2))))
        # Out of 255; normalising depth once per episode instead of per frame gave 0.38 here.
        self.assertLess(np.median(jumps[5:]), .1)


class ClaimTest(unittest.TestCase):
    def test_two_pods_share_a_work_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            episode = work / 'episode_0001_restyled0'
            as_pod = lambda name: mock.patch.object(socket, 'gethostname', return_value=name)
            with as_pod('pod-a'):
                self.assertTrue(aug.claim(episode))
            with as_pod('pod-b'):
                self.assertFalse(aug.claim(episode))  # pod-a is rendering it
            with as_pod('pod-a'):
                self.assertTrue(aug.claim(episode))  # pod-a restarted: its own claim
            episode.mkdir()
            (episode / 'done.json').write_text('{}')
            with as_pod('pod-b'):
                self.assertFalse(aug.claim(episode))  # finished
            stale = work / 'episode_0002_restyled0'
            with as_pod('pod-a'):
                self.assertTrue(aug.claim(stale))
            old = time.time() - aug.CLAIM_STALE_S - 10
            os.utime(work / (stale.name + '.claim'), (old, old))
            with as_pod('pod-b'):
                self.assertTrue(aug.claim(stale))  # abandoned claims are taken over


if __name__ == '__main__':
    unittest.main()
