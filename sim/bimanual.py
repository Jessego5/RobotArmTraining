"""Two-arm Panthera scene matching the real gear/carrier/pin setup (first estimate).

Two copies of the single-arm model (`panthera/panthera.xml`, whose arm
kinematics are identical to the real arms' URDF) are attached with `left/` and
`right/` prefixes, so actuators come out in the real dataset's 14-D order:
left joints 1-6, left gripper, right joints 1-6, right gripper.

Placement is ESTIMATED until measured. In real demonstrations both grippers end
holding the assembly together; across 14 episodes the left-minus-right tool
position (each in its own arm's base frame) was (0.006, -0.575, 0.040) m with
1-2 cm spread. That fits parallel bases facing +x, 0.575 m apart along y, the
left arm on +y, at table height. The overhead camera pose is a guess. The
gripper is still the simulated parallel jaw, not the real fin-ray gripper, and
the wrist cameras are pinhole, not the real fisheye lenses.

    python sim/bimanual.py            # writes panthera/scene_bimanual.xml and checks it
"""
from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
ARM_XML = HERE / 'panthera' / 'panthera.xml'
SCENE_XML = HERE / 'panthera' / 'scene_bimanual.xml'

# ---- estimates; replace with measurements --------------------------------
BASE_SEPARATION = .575          # m between the two bases, along y
BASE_HEIGHT = 0.                # bases sit on the table surface (z = 0)
TABLE_SIZE = (1.0, 1.4)         # m, x by y, centred in front of the bases
TABLE_CENTRE_X = .30
# The real 'overhead' camera sits behind and above the bases, looking forward over
# the arms along +x (arms at the bottom of the image); the lens is wide.
OVERHEAD = dict(pos=(-.45, 0., .60), target=(.45, 0., 0.), fovy=85.)
# --------------------------------------------------------------------------

ARMS = ('left', 'right')
REAL_GRIPPER_RANGE_RAD = (0., 2.)  # real gripper limits from the recording config


def build_spec() -> mujoco.MjSpec:
    arm_template = mujoco.MjSpec.from_file(str(ARM_XML))
    spec = mujoco.MjSpec()
    spec.modelname = 'panthera_bimanual'
    # Physics settings come from the arm model, as in the single-arm scene.
    for key in ('timestep', 'integrator', 'cone', 'impratio'):
        setattr(spec.option, key, getattr(arm_template.option, key))
    spec.visual.global_.offwidth, spec.visual.global_.offheight = 1920, 1080
    spec.visual.headlight.diffuse = [.25, .25, .25]
    spec.visual.headlight.ambient = [.15, .15, .15]

    world = spec.worldbody
    world.add_light(pos=[TABLE_CENTRE_X, 0, 2.], dir=[0, 0, -1], diffuse=[.45, .45, .45],
                    type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL)
    world.add_light(pos=[TABLE_CENTRE_X, .5, 1.5], dir=[0, -.3, -1], diffuse=[.15, .15, .15])
    world.add_geom(name='floor', type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, .05],
                   pos=[0, 0, -.75], rgba=[.35, .35, .38, 1])
    table = world.add_body(name='table', pos=[TABLE_CENTRE_X, 0, -.02])
    table.add_geom(name='table_top', type=mujoco.mjtGeom.mjGEOM_BOX,
                   size=[TABLE_SIZE[0] / 2, TABLE_SIZE[1] / 2, .02],
                   rgba=[.82, .66, .45, 1], friction=[.4, .02, .0005])
    # Camera looks along -z: z-axis points away from the target; x-axis is image right (-y).
    forward = np.subtract(OVERHEAD['target'], OVERHEAD['pos'])
    forward /= np.linalg.norm(forward)
    right = np.array([0., -1., 0.])
    world.add_camera(name='overhead', pos=list(OVERHEAD['pos']), fovy=OVERHEAD['fovy'],
                     quat=_quat_from_axes(right, np.cross(-forward, right)))

    for side, y in zip(ARMS, (BASE_SEPARATION / 2, -BASE_SEPARATION / 2)):
        arm = mujoco.MjSpec.from_file(str(ARM_XML))
        frame = world.add_frame(pos=[0, y, BASE_HEIGHT])
        frame.attach_body(arm.worldbody.first_body(), f'{side}/', '')
    return spec


def _quat_from_axes(x_axis: np.ndarray, y_axis: np.ndarray) -> list[float]:
    z_axis = np.cross(x_axis, y_axis)
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, np.stack([x_axis, y_axis, z_axis], axis=1).ravel())
    return quat.tolist()


def build_model() -> mujoco.MjModel:
    return build_spec().compile()


class BimanualState:
    """Write recorded real joint states into the simulator kinematically."""

    def __init__(self, model: mujoco.MjModel):
        joint = lambda name: model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)]
        self.arm_qadr = {side: np.array([joint(f'{side}/joint{i}') for i in range(1, 7)]) for side in ARMS}
        self.finger_qadr = {side: (joint(f'{side}/L_finger_joint'), joint(f'{side}/R_finger_joint'))
                            for side in ARMS}
        self.finger_max = float(model.jnt_range[mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, 'left/L_finger_joint'), 1])

    def set(self, data: mujoco.MjData, joints: dict, grippers_rad: dict) -> None:
        for side in ARMS:
            data.qpos[self.arm_qadr[side]] = joints[side]
            # Placeholder: real fin-ray angle mapped linearly onto the parallel jaws.
            lo, hi = REAL_GRIPPER_RANGE_RAD
            opening = float(np.clip((grippers_rad[side] - lo) / (hi - lo), 0, 1)) * self.finger_max
            left, right = self.finger_qadr[side]
            data.qpos[left], data.qpos[right] = opening, -opening


if __name__ == '__main__':
    spec = build_spec()
    model = spec.compile()
    SCENE_XML.write_text(spec.to_xml())
    names = lambda kind, n: [mujoco.mj_id2name(model, kind, i) for i in range(n)]
    print(f'wrote {SCENE_XML.name}: nq={model.nq} nu={model.nu}')
    print('actuators:', names(mujoco.mjtObj.mjOBJ_ACTUATOR, model.nu))
    print('cameras:', names(mujoco.mjtObj.mjOBJ_CAMERA, model.ncam))
