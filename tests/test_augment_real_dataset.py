"""Checks of tools/augment_real_dataset.py that need no models, GPU or network."""
import os
import socket
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

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


@pytest.mark.parametrize('clean', [True, False])
def test_keep_mask_keeps_the_object_and_repaints_the_table(clean):
    image, disparity, (x0, y0, x1, y1) = wood_table_scene()
    keep = aug.keep_mask(image, disparity, clean=clean)
    inner = keep[y0 + 4:y1 - 4, x0 + 4:x1 - 4]
    table = np.ones_like(keep)
    table[y0 - 12:y1 + 12, x0 - 12:x1 + 12] = False
    assert inner.mean() > .95
    assert keep[table].mean() < .05


def test_scramble_never_touches_kept_pixels():
    rng = np.random.default_rng(0)
    frames = rng.integers(0, 255, (3, 64, 64, 3), dtype=np.uint8)
    keep = np.zeros((3, 64, 64), bool)
    keep[:, 20:40, 10:30] = True
    for realistic in (True, False):
        out, kind = aug.scramble_background(frames, keep, seed=5, realistic=realistic)
        assert np.array_equal(out[keep], frames[keep])
        assert not np.array_equal(out[~keep], frames[~keep])
        assert kind.startswith('realistic' if realistic else 'wild')


def test_maps_round_trip(tmp_path):
    rng = np.random.default_rng(1)
    keep = rng.random((5, 64, 64)) > .5
    disparity = cv2.GaussianBlur(rng.random((5, 64, 64)).astype(np.float32), (0, 0), 4)
    maps = {camera: (keep, disparity) for camera in aug.CAMERAS}
    aug.save_maps(tmp_path / 'maps.npz', maps)
    loaded = aug.load_maps(tmp_path / 'maps.npz')
    for camera in aug.CAMERAS:
        assert np.array_equal(loaded[camera][0], keep)  # masks are stored exactly
        assert loaded[camera][1].shape == disparity.shape


def test_relight_does_not_flicker_when_depth_scale_jitters():
    """Monocular depth comes with its own scale and offset per frame; a still scene must stay still."""
    rng = np.random.default_rng(2)
    frame = cv2.GaussianBlur(rng.integers(60, 200, (96, 96, 3), dtype=np.uint8), (0, 0), 3)
    frames = np.repeat(frame[None], 30, 0)
    yy = np.mgrid[0:96, 0:96][0].astype(np.float32) / 96
    base = yy + .2 * np.sin(np.mgrid[0:96, 0:96][1] / 9.)
    disparity = np.stack([base * rng.uniform(.6, 1.4) + rng.uniform(-.3, .3) for _ in range(30)]).astype(np.float32)
    look = aug.sample_lighting(3)
    look.update(shadows=0)  # drifting shadow blobs move on purpose
    out = aug.relight(frames, disparity, look, camera='overhead').astype(np.float32).mean(-1)
    jumps = np.abs(np.diff(out.mean((1, 2))))
    assert np.median(jumps[5:]) < .1  # out of 255; with one normalisation per episode it was 0.38 here


def test_claims_let_two_pods_share_a_work_folder(tmp_path, monkeypatch):
    episode = tmp_path / 'episode_0001_restyled0'
    monkeypatch.setattr(socket, 'gethostname', lambda: 'pod-a')
    assert aug.claim(episode)
    monkeypatch.setattr(socket, 'gethostname', lambda: 'pod-b')
    assert not aug.claim(episode)  # pod-a is rendering it
    monkeypatch.setattr(socket, 'gethostname', lambda: 'pod-a')
    assert aug.claim(episode)  # pod-a restarted: its own claim
    episode.mkdir()
    (episode / 'done.json').write_text('{}')
    monkeypatch.setattr(socket, 'gethostname', lambda: 'pod-b')
    assert not aug.claim(episode)  # finished
    stale = tmp_path / 'episode_0002_restyled0'
    monkeypatch.setattr(socket, 'gethostname', lambda: 'pod-a')
    assert aug.claim(stale)
    old = time.time() - aug.CLAIM_STALE_S - 10
    os.utime(tmp_path / (stale.name + '.claim'), (old, old))
    monkeypatch.setattr(socket, 'gethostname', lambda: 'pod-b')
    assert aug.claim(stale)  # abandoned claims are taken over
