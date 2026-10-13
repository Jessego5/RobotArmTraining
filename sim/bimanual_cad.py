"""Zeb's CAD actuator parts in the real-matched two-arm layout.

`sim/bimanual_actuator_env.py` holds the Onshape carrier, gears and pins with
physical, contact-only seating, but places the arms facing each other in a
synthetic frame. The real recordings put the bases side by side, both facing
+x, 0.575 m apart (see `sim/bimanual.py`). This scene keeps Zeb's parts,
fixture and controller geometry and moves them into that layout:

- bases side by side on the table top (the real tools work 1-3 cm above base
  height, which fits the parts' table at the bases' height);
- the fixture and every station rotated as one block by FIXTURE_YAW about the
  fixture, so the gear slot faces the left arm and the carrier's lean points
  back toward the right arm's base, as in Zeb's layout;
- only the parts the real task uses: carrier, one gear, one pin;
- the real-lab lights, varnished table and estimated cameras of `sim/bimanual.py`.

The sim keeps Zeb's state contract (gripper opening in metres); the exporter
converts to the real dataset's radians.
"""
from __future__ import annotations

import copy
import math
import os
import tempfile
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from sim.actuator_env import ASSETS, TABLE_Z, ActuatorSim
from sim.bimanual import BASE_SEPARATION, LIGHTS, OVERHEAD_SETUPS, TABLE_CENTRE_X, TABLE_SIZE, _look_at
from sim.bimanual_actuator_env import Arm
from sim.panthera_env import ARM_JOINTS

USED = ('small_carrier', 'large_carrier', 'gear_3', 'pin_2')
FIXTURE_XY = np.array([.30, 0.])
FIXTURE_YAW = -math.pi / 2
# Zeb's bimanual stations around his fixture at (0.36, 0); rotated into this layout.
_ZEB_FIXTURE = np.array([.36, 0.])
_ZEB = {'large_carrier': (.43, .12), 'gear_3': (.28, -.12), 'pin_2': (.24, -.06)}
_ZEB_LEFT_CLEAR = (.24, -.14)
CAMERAS = {'overhead': 'overhead', 'wrist_port2': 'left_wrist', 'wrist_port3': 'right_wrist'}
# Colours of the real parts (Zeb's scene colours them for legibility instead).
PART_RGBA = {'large_carrier': '.78 .79 .81 1', 'small_carrier': '.62 .63 .66 1',
             'gear_3': '.10 .10 .11 1', 'pin_2': '.70 .71 .73 1'}


def _rotate(xy, angle=FIXTURE_YAW):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([c * xy[0] - s * xy[1], s * xy[0] + c * xy[1]])


def station(name):
    if name == 'small_carrier':
        return FIXTURE_XY.copy()
    return FIXTURE_XY + _rotate(np.subtract(_ZEB[name], _ZEB_FIXTURE))


LEFT_CLEAR = np.r_[FIXTURE_XY + _rotate(np.subtract(_ZEB_LEFT_CLEAR, _ZEB_FIXTURE)), .14 + TABLE_Z]


def _vec(values):
    return ' '.join(f'{v:.9g}' for v in values)


def _shift(element, offset):
    pos = np.array([float(v) for v in element.get('pos', '0 0 0').split()]) + offset
    element.set('pos', _vec(pos))


def build_scene(overhead_setup: str = 'after_oct2') -> str:
    overhead = OVERHEAD_SETUPS[overhead_setup]
    root = ET.parse(ASSETS / 'scene_pins.xml').getroot()
    root.set('model', 'panthera_bimanual_cad_parallel')
    world = root.find('worldbody')
    # Parts the real task does not use, their racks, and Zeb's lights and cameras go.
    for body in list(world.findall('body')):
        name = body.get('name', '')
        if name.startswith(('gear_', 'pin_', 'small_carrier', 'large_carrier')) and name not in USED:
            world.remove(body)
            continue
        if name.startswith('gear_'):
            _shift(body, [0, 0, TABLE_Z + .0002 - float(body.get('pos').split()[2])])
        for geom in body.findall('geom'):
            if geom.get('rgba') and name in PART_RGBA:
                geom.set('rgba', PART_RGBA[name])
    for element in list(world):
        if element.tag in ('light', 'camera') or element.get('name', '').endswith('_rack'):
            world.remove(element)
    # Real-matched table: varnished, large, under both bases.
    asset = root.find('asset')
    ET.SubElement(asset, 'material', name='table', rgba='.80 .62 .40 1', specular='.6', shininess='.9')
    table = world.find("geom[@name='table_top']")
    table.attrib.pop('rgba', None)
    table.set('material', 'table')
    table.set('pos', _vec([TABLE_CENTRE_X, 0, TABLE_Z / 2]))
    table.set('size', _vec([TABLE_SIZE[0] / 2, TABLE_SIZE[1] / 2, TABLE_Z / 2]))
    floor = world.find("geom[@name='floor']")
    floor.set('pos', '0 0 -.70')
    floor.set('rgba', '.35 .35 .38 1')
    # Left arm: its base geoms live in the worldbody; lift them onto the table top.
    left_offset = np.array([0., BASE_SEPARATION / 2, TABLE_Z])
    robot = world.find("body[@name='link1']")
    base_geoms = [g for g in world.findall('geom') if g.get('class') in ('visual', 'collision')]
    for geom in base_geoms:
        _shift(geom, left_offset)
    _shift(robot, left_offset)
    for geom in world.findall(".//geom[@class='visual']"):
        if geom.get('rgba'):
            rgba = geom.get('rgba').split(); rgba[-1] = '1'
            geom.set('rgba', ' '.join(rgba))
    # Right arm: a copy with right_ names on its own base, side by side, facing +x.
    second = copy.deepcopy(robot)
    names = {e.get('name'): 'right_' + e.get('name') for e in robot.iter() if e.get('name')}
    for e in second.iter():
        if e.get('name'):
            e.set('name', names[e.get('name')])
    _shift(second, -left_offset)
    base = ET.SubElement(world, 'body', name='right_base', pos=_vec([0, -BASE_SEPARATION / 2, TABLE_Z]))
    for geom in base_geoms:
        clone = copy.deepcopy(geom)
        _shift(clone, -left_offset)
        base.append(clone)
    base.append(second)
    second.find(".//camera[@name='right_wrist']").set('name', 'right_wrist')
    robot.find(".//camera[@name='wrist']").set('name', 'left_wrist')
    for section in ('actuator', 'equality', 'contact'):
        parent = root.find(section)
        for e in list(parent):
            clone = copy.deepcopy(e)
            for key, value in list(clone.attrib.items()):
                if key in ('name', 'joint', 'joint1', 'joint2', 'body1', 'body2'):
                    clone.set(key, 'right_base' if value == 'world' else 'right_' + value)
            parent.append(clone)
    # The right base now overlaps the table it stands on.
    ET.SubElement(root.find('contact'), 'exclude', body1='world', body2='right_base')
    # Lights and cameras of sim/bimanual.py, lifted by the table height.
    lift = np.array([0, 0, TABLE_Z])
    for name, light in LIGHTS.items():
        target = np.array([.35, 0., TABLE_Z])
        direction = target - (light['pos'] + lift)
        ET.SubElement(world, 'light', name=name, pos=_vec(light['pos'] + lift),
                      dir=_vec(direction / np.linalg.norm(direction)), type='spot', cutoff='80',
                      castshadow='true', diffuse=_vec(light['diffuse']), specular=_vec(light['specular']),
                      bulbradius='.15')
    quat = _look_at(np.add(overhead['pos'], lift), np.add(overhead['target'], lift))
    ET.SubElement(world, 'camera', name='overhead', pos=_vec(np.add(overhead['pos'], lift)),
                  quat=_vec(quat), fovy=str(overhead['fovy']))
    visual = root.find('visual')
    head = visual.find('headlight')
    head.set('ambient', '.22 .22 .22'); head.set('diffuse', '.12 .12 .12'); head.set('specular', '0 0 0')
    ET.SubElement(visual, 'quality', shadowsize='4096')
    glob = visual.find('global')
    glob.set('offwidth', '1920'); glob.set('offheight', '1080')
    # Keyframe: both arms at home, free parts where the scene put them.
    root.remove(root.find('keyframe'))
    home = [0., 1.2, 1.4, 0., 0., 0., .04, -.04]
    free = []
    for body in world.findall('body'):
        if body.find('freejoint') is not None:
            free.extend(map(float, body.get('pos').split()))
            free.extend(map(float, body.get('quat', '1 0 0 0').split()))
    ET.SubElement(ET.SubElement(root, 'keyframe'), 'key', name='home',
                  qpos=_vec(home + free + home), ctrl=_vec(home[:7] + home[:7]))
    ET.indent(root)
    path = ASSETS / f'scene_bimanual_parallel_{overhead_setup}.xml'
    with tempfile.NamedTemporaryFile(mode='w', dir=ASSETS, suffix='.xml', delete=False) as f:
        temporary = f.name
        ET.ElementTree(root).write(f, encoding='unicode')
    os.replace(temporary, path)
    return path


class ParallelCADSim(ActuatorSim):
    """Zeb's BimanualActuatorSim interface on the real-matched layout."""

    cameras = CAMERAS

    def __init__(self, seed=0, xy_jitter=.002, yaw_jitter_deg=2., overhead_setup: str = 'after_oct2'):
        self.part_names = USED
        super().__init__(seed, xy_jitter, yaw_jitter_deg, with_pins=True, scene=build_scene(overhead_setup))
        self.stations = {name: tuple(station(name)) for name in USED}
        self.arms = {side: Arm(self, side) for side in ('left', 'right')}
        self.reset()

    def reset(self, *, randomize=True, rng=None):
        if not hasattr(self, 'part_names') or not isinstance(getattr(self, 'stations', None), dict) \
                or 'gear_1' in self.stations:
            # Parent constructor: stations are not set up for this layout yet.
            mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
            mujoco.mj_forward(self.model, self.data)
            return
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        rng = np.random.default_rng(self.seed) if rng is None else rng
        for name, xy in self.stations.items():
            dxy = rng.uniform(-self.xy_jitter, self.xy_jitter, 2) if randomize else np.zeros(2)
            yaw = FIXTURE_YAW + (np.deg2rad(rng.uniform(-self.yaw_jitter_deg, self.yaw_jitter_deg))
                                 if randomize else 0.)
            pos = np.r_[np.array(xy) + dxy, TABLE_Z + .0002]
            quat = np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])
            bid = self.model.body(name).id
            if name == 'small_carrier':
                self.model.body_pos[bid] = pos
                self.model.body_quat[bid] = quat
                self.model.geom_pos[self.model.geom('fixture').id, :2] = pos[:2]
            else:
                self.set_part_pose(name, pos, quat)
        for side in ('left', 'right'):
            prefix = '' if side == 'left' else 'right_'
            for joint, value in zip(ARM_JOINTS, [0., 1.2, 1.4, 0., 0., 0.]):
                self.data.qpos[self.model.jnt_qposadr[self.model.joint(prefix + joint).id]] = value
                self.data.ctrl[self.model.actuator(prefix + joint).id] = value
            for finger, value in (('L', .04), ('R', -.04)):
                self.data.qpos[self.model.jnt_qposadr[self.model.joint(prefix + finger + '_finger_joint').id]] = value
            self.data.ctrl[self.model.actuator(prefix + 'gripper').id] = .04
        self.sync_control_state()
        mujoco.mj_forward(self.model, self.data)

    def sync_control_state(self):
        super().sync_control_state()
        for arm in getattr(self, 'arms', {}).values():
            arm.applied = self.data.ctrl[arm.act_ids].copy()

    def fixture_pose(self) -> dict:
        """The fixture is not a free body, so a qpos replay needs its pose separately."""
        bid = self.model.body('small_carrier').id
        return dict(pos=self.model.body_pos[bid].tolist(), quat=self.model.body_quat[bid].tolist())

    # Same contract as BimanualActuatorSim.
    from sim.bimanual_actuator_env import BimanualActuatorSim as _B
    state = _B.state
    action = _B.action
    step = _B.step
    del _B
