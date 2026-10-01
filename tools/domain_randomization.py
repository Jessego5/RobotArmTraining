"""Per-episode visual domain randomization: a placeholder for sim-to-real.

Until real camera images exist, varying what a real setup will change (table and
floor appearance, lighting, arm finish, small camera-pose errors) is the standard
cheap way to keep a policy from depending on this simulator's exact look.

Only the rendering model is changed, between episodes, so physics and recorded
trajectories are untouched and the renderer's code (and the rendered-dataset
contract that hashes it) stays as it is. Cube hues are preserved because the
task order depends on them; only their brightness/saturation vary slightly.
Every range is centred on the default scene, so ordinary simulator evaluation
stays in-distribution.
"""
from __future__ import annotations

import colorsys
from dataclasses import asdict, dataclass

import mujoco
import numpy as np

CUBES = ('cube_red', 'cube_green', 'cube_blue')


@dataclass(frozen=True)
class Ranges:
    surface_value: float = .25      # +/- relative brightness of the table
    surface_hue: float = .08        # +/- hue shift of the table (0-1 wheel)
    floor_tint: float = .35         # per-channel tint of the textured floor (1 = none)
    arm_value: float = .25          # +/- relative brightness of robot links
    cube_value: float = .12         # +/- relative brightness of cubes (hue fixed)
    cube_saturation: float = .15    # +/- relative saturation of cubes
    light_intensity: float = .35    # +/- relative diffuse/ambient light
    light_tilt_deg: float = 25.     # light direction tilt from vertical
    shoulder_angle_deg: float = 3.  # azimuth/elevation jitter of the fixed view
    shoulder_distance: float = .04  # +/- relative distance of the fixed view
    shoulder_lookat_m: float = .01
    wrist_offset_m: float = .002    # wrist camera mount translation
    wrist_angle_deg: float = 1.5    # wrist camera mount rotation
    fovy_deg: float = 2.


def _rescale(rgb, value=1., saturation=1., hue_shift=0.):
    h, s, v = colorsys.rgb_to_hsv(*np.clip(rgb, 0, 1))
    return np.array(colorsys.hsv_to_rgb((h + hue_shift) % 1, np.clip(s * saturation, 0, 1),
                                        np.clip(v * value, 0, 1)))


class VisualRandomizer:
    """Apply and undo one random appearance on a model and its render cameras."""

    def __init__(self, model: mujoco.MjModel, ranges: Ranges = Ranges()):
        self.model, self.ranges = model, ranges
        name = lambda kind, i: mujoco.mj_id2name(model, kind, i) or ''
        self.surfaces = [i for i in range(model.nmat) if name(mujoco.mjtObj.mjOBJ_MATERIAL, i) == 'table']
        # The floor's colour comes from its texture; its white material tints it.
        self.floors = [i for i in range(model.nmat) if name(mujoco.mjtObj.mjOBJ_MATERIAL, i) == 'groundplane']
        self.cubes = [i for i in range(model.ngeom) if name(mujoco.mjtObj.mjOBJ_GEOM, i) in CUBES]
        table = {mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n) for n in ('table', 'world')}
        cube_bodies = {model.geom_bodyid[i] for i in self.cubes}
        # Visible robot geoms: everything else that renders and is not scenery.
        self.arm = [i for i in range(model.ngeom)
                    if model.geom_bodyid[i] not in table | cube_bodies
                    and model.geom_group[i] < 3 and model.geom_rgba[i, 3] > 0]
        self.wrist = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, 'wrist')
        self.saved = None

    def apply(self, seed: int, shoulder: mujoco.MjvCamera | None = None) -> dict:
        """Randomize for one episode; returns the drawn parameters for provenance."""
        if self.saved is not None:
            self.restore()
        m, r = self.model, self.ranges
        rng = np.random.default_rng(seed)
        u = lambda width: float(rng.uniform(-width, width))
        self.saved = dict(mat_rgba=m.mat_rgba.copy(), geom_rgba=m.geom_rgba.copy(),
                          light_diffuse=m.light_diffuse.copy(), light_ambient=m.light_ambient.copy(),
                          light_dir=m.light_dir.copy(), cam_pos=m.cam_pos.copy(),
                          cam_quat=m.cam_quat.copy(), cam_fovy=m.cam_fovy.copy(),
                          headlight=(m.vis.headlight.ambient.copy(), m.vis.headlight.diffuse.copy()),
                          shoulder=None if shoulder is None else (
                              shoulder, shoulder.azimuth, shoulder.elevation,
                              shoulder.distance, shoulder.lookat.copy()))
        drawn = {}
        for i in self.surfaces:
            value, hue = 1 + u(r.surface_value), u(r.surface_hue)
            rgb = _rescale(m.mat_rgba[i, :3], value, 1., hue)
            m.mat_rgba[i, :3] = rgb * min(1., .8 / rgb.max())  # stay below white under bright light
            drawn[f'surface_{i}'] = dict(value=value, hue_shift=hue)
        for i in self.floors:
            tint = 1 - rng.uniform(0, r.floor_tint, size=3)
            m.mat_rgba[i, :3] = m.mat_rgba[i, :3] * tint
            drawn[f'floor_{i}'] = dict(tint=tint.tolist())
        arm_value = 1 + u(r.arm_value)
        for i in self.arm:
            m.geom_rgba[i, :3] = _rescale(m.geom_rgba[i, :3], arm_value)
        drawn['arm_value'] = arm_value
        for i in self.cubes:
            value, saturation = 1 + u(r.cube_value), 1 + u(r.cube_saturation)
            m.geom_rgba[i, :3] = _rescale(m.geom_rgba[i, :3], value, saturation)
            drawn[f'cube_{i}'] = dict(value=value, saturation=saturation)
        intensity = 1 + u(r.light_intensity)
        m.light_diffuse[:] *= intensity
        m.light_ambient[:] *= intensity
        m.vis.headlight.ambient[:] = self.saved['headlight'][0] * intensity
        m.vis.headlight.diffuse[:] = self.saved['headlight'][1] * intensity
        tilt, heading = np.deg2rad(rng.uniform(0, r.light_tilt_deg)), rng.uniform(0, 2 * np.pi)
        m.light_dir[:] = [np.sin(tilt) * np.cos(heading), np.sin(tilt) * np.sin(heading), -np.cos(tilt)]
        drawn.update(light_intensity=intensity, light_tilt_deg=float(np.rad2deg(tilt)))
        if self.wrist >= 0:
            m.cam_pos[self.wrist] += [u(r.wrist_offset_m) for _ in range(3)]
            axis = rng.normal(size=3)
            turn = np.zeros(4)
            mujoco.mju_axisAngle2Quat(turn, axis / np.linalg.norm(axis), np.deg2rad(u(r.wrist_angle_deg)))
            quat = np.zeros(4)
            mujoco.mju_mulQuat(quat, turn, m.cam_quat[self.wrist].copy())
            m.cam_quat[self.wrist] = quat
            m.cam_fovy[self.wrist] += u(r.fovy_deg)
        if shoulder is not None:
            shoulder.azimuth += u(r.shoulder_angle_deg)
            shoulder.elevation += u(r.shoulder_angle_deg)
            shoulder.distance *= 1 + u(r.shoulder_distance)
            shoulder.lookat[:] += [u(r.shoulder_lookat_m) for _ in range(3)]
        drawn['ranges'] = asdict(r)
        return drawn

    def restore(self):
        if self.saved is None:
            return
        m, saved = self.model, self.saved
        for key in ('mat_rgba', 'geom_rgba', 'light_diffuse', 'light_ambient', 'light_dir',
                    'cam_pos', 'cam_quat', 'cam_fovy'):
            getattr(m, key)[:] = saved[key]
        m.vis.headlight.ambient[:], m.vis.headlight.diffuse[:] = saved['headlight']
        if saved['shoulder'] is not None:
            camera, azimuth, elevation, distance, lookat = saved['shoulder']
            camera.azimuth, camera.elevation, camera.distance = azimuth, elevation, distance
            camera.lookat[:] = lookat
        self.saved = None
