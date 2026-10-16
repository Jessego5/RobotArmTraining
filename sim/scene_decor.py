"""Visual-only scene variety for rendering recorded two-arm episodes.

Everything added here has no collisions and no mass, so recorded physics is
untouched; it changes only what the cameras see. Geometry and lighting are what
a structure-locked restyle keeps (it repaints surfaces, guided by depth and
outlines), so this adds geometry the restyle cannot invent:

- a room: walls at random distances, furniture boxes, monitors on the far end
  of the table (the real frames show the lab behind the table);
- clutter: spare parts and small objects on the table, away from the work area;
- the operator: a hand, forearm and sleeve reaching to each robot wrist from
  behind, as in hand-guided demonstrations; they move with the robot (mocap,
  set per frame) and cast real shadows;
- table surfaces: procedural wood grain, checker, speckle or plain colour.

`decorate` edits an MjSpec before compiling; `Operators` updates the operator
bodies for each replayed frame; `perturb_model` adds wrist-camera mount jitter
and arm finish variation after compiling.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import mujoco
import numpy as np

VISUAL = dict(contype=0, conaffinity=0, group=0)
FLOOR_Z = -.70
_TEXTURE_DIR = Path(tempfile.gettempdir()) / f'scene_decor_{os.getpid()}'


def _wood_texture(rng, size=512) -> str:
    """Procedural wood grain: warped bands of two browns plus fine noise."""
    import cv2
    _TEXTURE_DIR.mkdir(parents=True, exist_ok=True)
    y, x = np.mgrid[0:size, 0:size].astype(np.float32) / size
    warp = cv2.GaussianBlur(rng.normal(0, 1, (size, size)).astype(np.float32), (0, 0), size / 12) * 6
    bands = np.sin((y * rng.uniform(6, 14) + warp * .5 + x * rng.uniform(-1, 1)) * np.pi) * .5 + .5
    fine = cv2.GaussianBlur(rng.normal(0, 1, (size, size)).astype(np.float32), (0, 0), 1.2)
    light = np.array(rng.uniform([.70, .52, .32], [.92, .76, .55]))
    dark = light * rng.uniform(.70, .88)
    image = dark + (light - dark) * bands[..., None] + .03 * fine[..., None]
    path = _TEXTURE_DIR / f'wood_{rng.integers(1 << 30)}.png'
    cv2.imwrite(str(path), (np.clip(image, 0, 1)[..., ::-1] * 255).astype(np.uint8))
    return str(path)


def _texture(spec, name, rng, kind):
    rgb = lambda lo=.25, hi=.85: rng.uniform(lo, hi, 3).tolist()
    if kind == 'wood':
        return spec.add_texture(name=name, type=mujoco.mjtTexture.mjTEXTURE_2D, file=_wood_texture(rng))
    if kind == 'checker':
        return spec.add_texture(name=name, type=mujoco.mjtTexture.mjTEXTURE_2D,
                                builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER, rgb1=rgb(), rgb2=rgb(),
                                width=256, height=256)
    if kind == 'gradient':
        return spec.add_texture(name=name, type=mujoco.mjtTexture.mjTEXTURE_2D,
                                builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT, rgb1=rgb(), rgb2=rgb(),
                                width=256, height=256)
    # speckle: flat colour with random marks
    return spec.add_texture(name=name, type=mujoco.mjtTexture.mjTEXTURE_2D,
                            builtin=mujoco.mjtBuiltin.mjBUILTIN_FLAT, rgb1=rgb(), rgb2=rgb(),
                            mark=mujoco.mjtMark.mjMARK_RANDOM, markrgb=rgb(), random=float(rng.uniform(.01, .2)),
                            width=256, height=256)


def _material(spec, name, rng, kind, repeat=(1, 1), **kw):
    material = spec.add_material(name=name, texrepeat=list(repeat), **kw)
    if kind != 'plain':
        texture = _texture(spec, name + '_tex', rng, kind)
        textures = list(material.textures)
        textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = texture.name
        material.textures = textures
    else:
        material.rgba = list(rng.uniform(.2, .85, 3)) + [1.]
    return material


def decorate(spec: mujoco.MjSpec, rng, layout: dict) -> dict:
    """Add room, clutter, operator and table surface to `spec`; returns what was added."""
    tz, world = layout['table_z'], spec.worldbody
    info = dict(table=None, furniture=0, clutter=0, operators=[])

    # Table surface.
    table = spec.material('table')
    kind = str(rng.choice(['wood', 'wood', 'wood', 'checker', 'speckle', 'plain']))
    info['table'] = kind
    if kind != 'plain':
        texture = _texture(spec, 'decor_table_tex', rng, kind)
        textures = list(table.textures)
        textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = texture.name
        table.textures = textures
        table.texrepeat = [float(rng.uniform(1, 2)), float(rng.uniform(1, 2))]
        table.rgba = [1, 1, 1, 1]
    else:
        table.rgba = list(rng.uniform([.45, .35, .25], [.9, .8, .7])) + [1.]

    # Room: walls beyond the ceiling lights so they never block them, and a back wall.
    far, side = float(rng.uniform(4.2, 5.5)), float(rng.uniform(1.5, 2.6))
    height = 1.6
    walls = [('far', [far, 0, FLOOR_Z + height], [.05, side, height]),
             ('left', [(far - 2.2) / 2, side, FLOOR_Z + height], [(far + 2.2) / 2, .05, height]),
             ('right', [(far - 2.2) / 2, -side, FLOOR_Z + height], [(far + 2.2) / 2, .05, height]),
             ('back', [-2.2, 0, FLOOR_Z + height], [.05, side, height])]
    for name, pos, size in walls:
        material = _material(spec, f'decor_wall_{name}', rng, str(rng.choice(['plain', 'plain', 'gradient', 'speckle'])),
                             repeat=(2, 2))
        world.add_geom(name=f'decor_wall_{name}', type=mujoco.mjtGeom.mjGEOM_BOX, pos=pos, size=size,
                       material=material.name, **VISUAL)
    floor = _material(spec, 'decor_floor', rng, str(rng.choice(['plain', 'checker', 'speckle'])), repeat=(8, 8))
    world.add_geom(name='decor_floor', type=mujoco.mjtGeom.mjGEOM_BOX, pos=[0, 0, FLOOR_Z - .01],
                   size=[6, 6, .01], material=floor.name, **VISUAL)

    # Furniture in the room and behind the operator, below the lights' height.
    for k in range(int(rng.integers(6, 16))):
        if rng.random() < .75:
            x, y = rng.uniform(1.3, far - .4), rng.choice([-1, 1]) * rng.uniform(.9, side - .3)
        else:
            x, y = rng.uniform(-2.0, -1.0), rng.uniform(-(side - .3), side - .3)
        size = rng.uniform([.15, .15, .2], [.5, .5, .6])
        world.add_geom(name=f'decor_furniture_{k}', type=mujoco.mjtGeom.mjGEOM_BOX,
                       pos=[x, y, FLOOR_Z + size[2]], size=size.tolist(),
                       rgba=list(rng.uniform(.1, .8, 3)) + [1.], **VISUAL)
        info['furniture'] += 1
    for k in range(int(rng.integers(0, 3))):  # monitors at the far end of the table
        w, h = rng.uniform(.18, .3), rng.uniform(.12, .2)
        world.add_geom(name=f'decor_monitor_{k}', type=mujoco.mjtGeom.mjGEOM_BOX,
                       pos=[rng.uniform(.95, 1.02), rng.uniform(-.5, .5), tz + .08 + h],
                       size=[.015, w, h], rgba=list(rng.uniform(.02, .12, 3)) + [1.], **VISUAL)

    # Clutter on the table, outside the work area (far band and front corners).
    meshes = []
    for name in layout.get('clutter_meshes', []):
        try:
            if spec.mesh(name) is not None:
                meshes.append(name)
        except Exception:
            pass
    for k in range(int(rng.integers(3, 11))):
        if rng.random() < .7:
            x, y = rng.uniform(.58, .98), rng.uniform(-.6, .6)
        else:
            x, y = rng.uniform(.05, .18), rng.choice([-1, 1]) * rng.uniform(.45, .65)
        yaw = rng.uniform(-np.pi, np.pi)
        quat = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        if meshes and rng.random() < .5:
            world.add_geom(name=f'decor_clutter_{k}', type=mujoco.mjtGeom.mjGEOM_MESH,
                           meshname=str(rng.choice(meshes)), pos=[x, y, tz + .0005], quat=quat,
                           rgba=list(rng.choice([[.75, .76, .78], [.1, .1, .11], [.6, .45, .2]])) + [1.], **VISUAL)
        else:
            shape = rng.choice(['box', 'cylinder', 'capsule'])
            size = rng.uniform([.006, .006, .004], [.04, .03, .03])
            types = dict(box=mujoco.mjtGeom.mjGEOM_BOX, cylinder=mujoco.mjtGeom.mjGEOM_CYLINDER,
                         capsule=mujoco.mjtGeom.mjGEOM_CAPSULE)
            if shape == 'capsule':  # pens, screwdrivers lying down
                world.add_geom(name=f'decor_clutter_{k}', type=types[shape], fromto=[
                    x - .06 * np.cos(yaw), y - .06 * np.sin(yaw), tz + .006,
                    x + .06 * np.cos(yaw), y + .06 * np.sin(yaw), tz + .006], size=[.006, 0, 0],
                    rgba=list(rng.uniform(.05, .9, 3)) + [1.], **VISUAL)
            else:
                world.add_geom(name=f'decor_clutter_{k}', type=types[shape], pos=[x, y, tz + size[2]],
                               quat=quat, size=size.tolist(), rgba=list(rng.uniform(.05, .9, 3)) + [1.], **VISUAL)
        info['clutter'] += 1

    # Extra lights with their own shadows: a low window light, a desk lamp, another fixture.
    info['lights'] = []
    for k in range(int(rng.choice([0, 1, 1, 2]))):
        kind = str(rng.choice(['window', 'lamp', 'ceiling']))
        side = rng.choice([-1, 1])
        if kind == 'window':  # low and far to one side: long, hard shadows across the table
            pos = [rng.uniform(-.5, 1.5), side * rng.uniform(2.0, 3.0), tz + rng.uniform(.4, 1.2)]
            colour = rng.uniform([.9, .9, .95], [1.1, 1.05, 1.2]) * rng.uniform(.4, .9)
            cutoff, softness = 60., rng.uniform(0, .1)
        elif kind == 'lamp':  # close to the table: one bright pool of light
            pos = [rng.uniform(.0, .9), side * rng.uniform(.3, .8), tz + rng.uniform(.35, .7)]
            colour = rng.uniform([1., .85, .6], [1.1, 1., .85]) * rng.uniform(.4, .9)
            cutoff, softness = rng.uniform(25, 45), rng.uniform(.02, .2)
        else:
            pos = [rng.uniform(-.5, 2.5), rng.uniform(-1.5, 1.5), tz + rng.uniform(1.6, 2.2)]
            colour = np.full(3, rng.uniform(.2, .6))
            cutoff, softness = 80., rng.uniform(.05, .6)
        target = np.array([rng.uniform(.15, .45), rng.uniform(-.15, .15), tz])
        direction = target - np.array(pos)
        world.add_light(name=f'decor_light_{k}', pos=pos, dir=(direction / np.linalg.norm(direction)).tolist(),
                        type=mujoco.mjtLightType.mjLIGHT_SPOT, cutoff=float(cutoff), castshadow=True,
                        diffuse=list(colour), specular=list(colour * .4), bulbradius=float(softness))
        info['lights'].append(dict(kind=kind, pos=[float(v) for v in pos], colour=[float(v) for v in colour]))

    # Operator hands guiding the wrists: one mocap body per arm, posed per frame.
    skin = rng.choice([[.95, .80, .68], [.85, .64, .48], [.62, .44, .32], [.40, .27, .19]])
    sleeve = list(rng.uniform(.05, .8, 3)) + [1.]
    for side, link in layout['operator_links'].items():
        if rng.random() > layout.get('operator_share', .7):
            continue
        body = world.add_body(name=f'decor_operator_{side}', mocap=True)
        body.add_geom(name=f'decor_operator_{side}_hand', type=mujoco.mjtGeom.mjGEOM_ELLIPSOID,
                      size=[.04, .025, .045], pos=[0, 0, .015], rgba=list(skin) + [1.], **VISUAL)
        body.add_geom(name=f'decor_operator_{side}_forearm', type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                      fromto=[0, 0, .045, 0, 0, .22], size=[.028, 0, 0], rgba=list(skin) + [1.], **VISUAL)
        body.add_geom(name=f'decor_operator_{side}_sleeve', type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                      fromto=[0, 0, .20, 0, 0, .34], size=[.04, 0, 0], rgba=sleeve, **VISUAL)
        sign = 1 if side == 'left' else -1
        info['operators'].append(dict(
            side=side, link=link, body=f'decor_operator_{side}',
            grip_offset=rng.uniform([-.04, -.03, .03], [.02, .03, .07]).tolist(),
            # The forearm points low and outward-back, toward an operator beside and
            # behind the base, so it stays near the bottom edge of the overhead view
            # (as in the real frames) instead of in front of the camera.
            shoulder=[float(rng.uniform(-.30, -.10)), float(sign * rng.uniform(.55, .80)),
                      float(tz + rng.uniform(.10, .25))]))
    return info


class Operators:
    """Pose each operator arm from the robot link it holds toward the operator's shoulder."""

    def __init__(self, model, operators: list):
        self.items = []
        for op in operators:
            body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, op['body'])
            link = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, op['link'])
            self.items.append((model.body_mocapid[body], link, np.array(op['grip_offset']), np.array(op['shoulder'])))

    def update(self, model, data) -> None:
        if not self.items:
            return
        for mocap, link, offset, shoulder in self.items:
            hand = data.xpos[link] + offset
            z = shoulder - hand
            z /= np.linalg.norm(z)
            x = np.cross([0, 0, 1.], z)
            x = x / np.linalg.norm(x) if np.linalg.norm(x) > 1e-6 else np.array([1., 0, 0])
            quat = np.zeros(4)
            mujoco.mju_mat2Quat(quat, np.stack([x, np.cross(z, x), z], 1).ravel())
            data.mocap_pos[mocap] = hand
            data.mocap_quat[mocap] = quat
        mujoco.mj_kinematics(model, data)


def perturb_model(model, rng, layout: dict) -> dict:
    """Wrist-camera mount jitter and arm finish, after compiling."""
    out = {}
    for name in layout['wrist_cameras']:
        cam = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name)
        model.cam_pos[cam] += rng.normal(0, .004, 3)
        axis = rng.normal(0, 1, 3)
        angle = np.deg2rad(rng.uniform(0, 3))
        turn, quat = np.zeros(4), model.cam_quat[cam].copy()
        mujoco.mju_axisAngle2Quat(turn, axis / np.linalg.norm(axis), angle)
        mujoco.mju_mulQuat(model.cam_quat[cam], quat, turn)
        out[name] = float(np.rad2deg(angle))
    for side, bodies in layout['arm_bodies'].items():
        tone = rng.uniform(.82, 1.1)
        ids = {mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, b) for b in bodies}
        for g in range(model.ngeom):
            if model.geom_group[g] == 2 and model.geom_bodyid[g] in ids:
                model.geom_rgba[g, :3] = np.clip(model.geom_rgba[g, :3] * tone, 0, 1)
        out[side] = float(tone)
    return out
