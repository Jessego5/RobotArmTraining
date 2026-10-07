"""Two-arm Panthera scene matching the real gear/carrier/pin setup (first estimate).

Two copies of the single-arm model (`panthera/panthera.xml`, whose arm
kinematics are identical to the real arms' URDF) are attached with `left/` and
`right/` prefixes, so actuators come out in the real dataset's 14-D order:
left joints 1-6, left gripper, right joints 1-6, right gripper.

Placement is ESTIMATED until measured. In real demonstrations both grippers end
holding the assembly together; across 14 episodes the left-minus-right tool
position (each in its own arm's base frame) was (0.006, -0.575, 0.040) m with
1-2 cm spread. That fits parallel bases facing +x, 0.575 m apart along y, the
left arm on +y, at table height. Camera poses are guesses. The gripper is the
simulated parallel jaw, whose fin-shaped fingers resemble the real fin-ray ones.

The task parts are stand-ins until CAD arrives: a square carrier with a pin
hole through its centre, a small gear (ring with a bore and teeth) and a pin
standing in a stand. In the real recordings the right gripper holds the carrier
the whole time (about 1.15 rad open) while the left places the gear and the pin.

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
TABLE_SIZE = (1.2, 1.4)         # m, x by y, centred in front of the bases
TABLE_CENTRE_X = .45
# The real overhead camera is a 640x480 webcam above and behind the bases;
# the dataset keeps the 256x256 crop [176, 224, 432, 480] (x0, y0, x1, y1).
# Two setups exist (moved on 2026-10-02): first nearly straight down, then
# tilted forward. Both are rough guesses.
OVERHEAD_SETUPS = {
    # Far behind and above the bases with a narrow webcam view: in the real crop
    # the far side of the room shows along the top, the table fills the middle and
    # the grippers meet about 85% of the way down, the arms entering from below.
    # Aimed so the real meeting point (0.25, 0, 0.03) projects to that row.
    'before_oct2': dict(pos=(-.60, 0., 1.10), target=(.83, 0., 0.), fovy=38.),
    'after_oct2': dict(pos=(-.80, 0., 1.00), target=(.85, 0., .10), fovy=35.),
}
OVERHEAD = OVERHEAD_SETUPS['after_oct2']
OVERHEAD_CROP = (176, 224, 432, 480)
# Parts (m). Clearances are generous until real tolerances are known.
CARRIER_HALF = .022, .016       # square half-width, half-height
HOLE_HALF = .0045               # square pin hole through carrier and gear
GEAR_HALF = .011, .004          # outer half-width, half-thickness
PIN_RADIUS, PIN_HALF = .0025, .013
STAND_HALF = .012, .006         # pin stand: half-width, half-height (hole depth = height)
# --------------------------------------------------------------------------

ARMS = ('left', 'right')
REAL_GRIPPER_RANGE_RAD = (0., 2.)  # real gripper: 0 closed, 2 fully open
FINGER_OPEN = .04                  # sim per-finger opening at full open (m)
PARTS = ('carrier', 'gear', 'pin')
METAL = [.72, .73, .75, 1]
DARK = [.12, .12, .13, 1]


def rad_to_opening(rad):
    """Real gripper angle -> sim per-finger opening; linear placeholder."""
    lo, hi = REAL_GRIPPER_RANGE_RAD
    return np.clip((np.asarray(rad) - lo) / (hi - lo), 0, 1) * FINGER_OPEN


def opening_to_rad(opening):
    lo, hi = REAL_GRIPPER_RANGE_RAD
    return lo + np.clip(np.asarray(opening) / FINGER_OPEN, 0, 1) * (hi - lo)


def _box(body, name, half, pos, rgba, **kw):
    body.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_BOX, size=list(half), pos=list(pos), rgba=rgba, **kw)


def _square_ring(body, prefix, outer, inner, half_z, z, rgba, **kw):
    """Four boxes around a square hole: a convex-geom version of a bored part."""
    side = (outer - inner) / 2
    for sign in (1, -1):
        _box(body, f'{prefix}_x{sign:+d}', (side, outer, half_z), (sign * (inner + side), 0, z), rgba, **kw)
        _box(body, f'{prefix}_y{sign:+d}', (inner, side, half_z), (0, sign * (inner + side), z), rgba, **kw)


def _add_parts(world):
    contact = dict(condim=6, friction=[.8, .08, .004], solref=[.005, 1], solimp=[.98, .99, .001, .5, 2])
    # Carrier: solid floor under a bored top layer; the pin seats in the bore.
    half, half_z = CARRIER_HALF
    carrier = world.add_body(name='carrier', pos=[.32, -.12, half_z])
    carrier.add_freejoint(name='carrier')
    floor = .003
    _box(carrier, 'carrier', (half, half, floor), (0, 0, -half_z + floor), METAL, density=2700, **contact)
    _square_ring(carrier, 'carrier_top', half, HOLE_HALF, half_z - floor, floor, METAL, density=2700, **contact)
    carrier.add_site(name='carrier_hole', pos=[0, 0, half_z], size=[.002], group=4)
    # Gear: bored ring with visual teeth.
    outer, gz = GEAR_HALF
    gear = world.add_body(name='gear', pos=[.30, .12, gz])
    gear.add_freejoint(name='gear')
    _square_ring(gear, 'gear', outer, HOLE_HALF + .0005, gz, 0, DARK, density=1200, **contact)
    for k in range(12):
        angle = 2 * np.pi * k / 12
        quat = np.zeros(4)
        mujoco.mju_axisAngle2Quat(quat, np.array([0., 0, 1]), angle)
        gear.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[.0022, .0015, gz * .9], rgba=DARK, quat=quat,
                      pos=[np.cos(angle) * outer * 1.25, np.sin(angle) * outer * 1.25, 0],
                      contype=0, conaffinity=0, group=2)
    gear.add_site(name='gear_bore', pos=[0, 0, gz], size=[.002], group=4)
    # Pin standing in a fixed stand, so it can be taken from above and inserted upright.
    stand_half, stand_z = STAND_HALF
    stand = world.add_body(name='pin_stand', pos=[.24, .20, stand_z])
    _box(stand, 'pin_stand_floor', (stand_half, stand_half, .001), (0, 0, -stand_z + .001), DARK)
    _square_ring(stand, 'pin_stand', stand_half, HOLE_HALF, stand_z - .001, .001, DARK)
    pin = world.add_body(name='pin', pos=[.24, .20, .002 + PIN_HALF + PIN_RADIUS])
    pin.add_freejoint(name='pin')
    pin.add_geom(name='pin', type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[PIN_RADIUS, PIN_HALF, 0],
                 rgba=METAL, density=7800, **contact)
    pin.add_site(name='pin_tip', pos=[0, 0, -PIN_HALF - PIN_RADIUS], size=[.002], group=4)


def _look_at(pos, target) -> list[float]:
    """Camera quaternion looking from pos to target with image-right along -y."""
    forward = np.subtract(target, pos, dtype=float)
    forward /= np.linalg.norm(forward)
    right = np.array([0., -1., 0.])
    up = np.cross(-forward, right)
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, np.stack([right, up, -forward], axis=1).ravel())
    return quat.tolist()


def build_spec(overhead: dict = OVERHEAD, parts: bool = True) -> mujoco.MjSpec:
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
    # Light maple butcher block, like the real table.
    table.add_geom(name='table_top', type=mujoco.mjtGeom.mjGEOM_BOX,
                   size=[TABLE_SIZE[0] / 2, TABLE_SIZE[1] / 2, .02],
                   rgba=[.80, .62, .40, 1], friction=[.4, .02, .0005])
    world.add_camera(name='overhead', pos=list(overhead['pos']), fovy=overhead['fovy'],
                     quat=_look_at(overhead['pos'], overhead['target']))
    if parts:
        _add_parts(world)

    for side, y in zip(ARMS, (BASE_SEPARATION / 2, -BASE_SEPARATION / 2)):
        arm = mujoco.MjSpec.from_file(str(ARM_XML))
        frame = world.add_frame(pos=[0, y, BASE_HEIGHT])
        frame.attach_body(arm.worldbody.first_body(), f'{side}/', '')
    # The single-arm model ghosts its links for teleoperation; the real arms are solid.
    for geom in spec.geoms:
        if geom.rgba[3] < 1:
            geom.rgba[3] = 1.
    return spec


def build_model(**kw) -> mujoco.MjModel:
    return build_spec(**kw).compile()


class BimanualState:
    """Write recorded real joint states into the simulator kinematically."""

    def __init__(self, model: mujoco.MjModel):
        joint = lambda name: model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)]
        self.arm_qadr = {side: np.array([joint(f'{side}/joint{i}') for i in range(1, 7)]) for side in ARMS}
        self.finger_qadr = {side: (joint(f'{side}/L_finger_joint'), joint(f'{side}/R_finger_joint'))
                            for side in ARMS}

    def set(self, data: mujoco.MjData, joints: dict, grippers_rad: dict) -> None:
        for side in ARMS:
            data.qpos[self.arm_qadr[side]] = joints[side]
            opening = float(rad_to_opening(grippers_rad[side]))
            left, right = self.finger_qadr[side]
            data.qpos[left], data.qpos[right] = opening, -opening


class _ArmIK:
    """One arm's view of the shared model, reusing the single-arm IK solver."""

    def __init__(self, model, data, side):
        from sim.panthera_env import PantheraSim
        self._single = PantheraSim
        name = lambda kind, n: mujoco.mj_name2id(model, kind, f'{side}/{n}')
        joints = [name(mujoco.mjtObj.mjOBJ_JOINT, f'joint{i}') for i in range(1, 7)]
        self.model, self.data = model, data
        self.arm_qadr = model.jnt_qposadr[joints]
        self.arm_dofadr = model.jnt_dofadr[joints]
        self.arm_range = model.jnt_range[joints].copy()
        self.ee_site = name(mujoco.mjtObj.mjOBJ_SITE, 'grip_site')
        self._ik_data = mujoco.MjData(model)
        self._jacp = np.zeros((3, model.nv))
        self._jacr = np.zeros((3, model.nv))
        self._q_nominal = np.array([0, 1.2, 1.4, 0, 0, 0.])

    def ik(self, *args, **kw):
        return self._single.ik(self, *args, **kw)

    def _ee_error(self, *args, **kw):
        return self._single._ee_error(self, *args, **kw)

    def _dls(self, *args, **kw):
        return self._single._dls(self, *args, **kw)


class BimanualSim:
    """Both arms with interpolated joint commands and real pad contact (contact-v2 style).

    Commands are absolute joint targets per arm plus a per-finger opening; one
    control tick linearly interpolates the arm targets at physics rate, as the
    single-arm `PantheraSim.step` does.
    """

    def __init__(self, model: mujoco.MjModel | None = None):
        self.model = model or build_model()
        self.model.opt.impratio = 100
        self.data = mujoco.MjData(self.model)
        m = self.model
        ident = lambda kind, n: mujoco.mj_name2id(m, kind, n)
        self.arms = {side: _ArmIK(m, self.data, side) for side in ARMS}
        self.arm_act = {side: np.array([ident(mujoco.mjtObj.mjOBJ_ACTUATOR, f'{side}/joint{i}') for i in range(1, 7)])
                        for side in ARMS}
        self.grip_act = {side: ident(mujoco.mjtObj.mjOBJ_ACTUATOR, f'{side}/gripper') for side in ARMS}
        self.finger_qadr = {side: m.jnt_qposadr[[ident(mujoco.mjtObj.mjOBJ_JOINT, f'{side}/{j}_finger_joint')
                                                 for j in 'LR']] for side in ARMS}
        self.finger_dofadr = {side: m.jnt_dofadr[ident(mujoco.mjtObj.mjOBJ_JOINT, f'{side}/L_finger_joint')]
                              for side in ARMS}
        self.pads = {side: [ident(mujoco.mjtObj.mjOBJ_GEOM, f'{side}/{j}_finger_pad') for j in 'LR'] for side in ARMS}
        self.part_body = {p: ident(mujoco.mjtObj.mjOBJ_BODY, p) for p in PARTS}
        self.part_qadr = {p: m.jnt_qposadr[ident(mujoco.mjtObj.mjOBJ_JOINT, p)] for p in PARTS}
        self.part_geoms = {p: {g for g in range(m.ngeom) if m.geom_bodyid[g] == b and m.geom_contype[g]}
                           for p, b in self.part_body.items()}
        self.sites = {n: ident(mujoco.mjtObj.mjOBJ_SITE, n) for n in ('carrier_hole', 'gear_bore', 'pin_tip')}
        self.reset()

    def reset(self) -> None:
        mujoco.mj_resetData(self.model, self.data)
        for side in ARMS:
            self.data.qpos[self.arms[side].arm_qadr] = self.arms[side]._q_nominal
            left, right = self.finger_qadr[side]
            self.data.qpos[left], self.data.qpos[right] = FINGER_OPEN, -FINGER_OPEN
            self.data.ctrl[self.arm_act[side]] = self.arms[side]._q_nominal
            self.data.ctrl[self.grip_act[side]] = FINGER_OPEN
        mujoco.mj_forward(self.model, self.data)
        self._applied = self.data.ctrl.copy()

    def place_part(self, part: str, xy, yaw: float = 0.) -> None:
        adr = self.part_qadr[part]
        quat = np.zeros(4)
        mujoco.mju_axisAngle2Quat(quat, np.array([0., 0, 1]), yaw)
        self.data.qpos[adr:adr + 2] = xy
        self.data.qpos[adr + 3:adr + 7] = quat

    def q(self, side) -> np.ndarray:
        return self.data.qpos[self.arms[side].arm_qadr].copy()

    def opening(self, side) -> float:
        return float(self.data.qpos[self.finger_qadr[side][0]])

    def ee_pose(self, side):
        site = self.arms[side].ee_site
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, self.data.site_xmat[site])
        return self.data.site_xpos[site].copy(), quat

    def part_pose(self, part):
        body = self.part_body[part]
        return self.data.xpos[body].copy(), self.data.xquat[body].copy()

    def site(self, name) -> np.ndarray:
        return self.data.site_xpos[self.sites[name]].copy()

    def set_command(self, side, q=None, opening=None, *, immediate=False) -> None:
        if q is not None:
            r = self.arms[side].arm_range
            self.data.ctrl[self.arm_act[side]] = np.clip(q, r[:, 0], r[:, 1])
        if opening is not None:
            self.data.ctrl[self.grip_act[side]] = float(np.clip(opening, 0, FINGER_OPEN))
        if immediate:
            self._applied = self.data.ctrl.copy()

    def pinched(self, side) -> set[str]:
        """Parts touching both of this arm's pads."""
        touching = [set(), set()]
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            for geom, other in ((c.geom1, c.geom2), (c.geom2, c.geom1)):
                if other in self.pads[side]:
                    touching[self.pads[side].index(other)] |= {
                        p for p, geoms in self.part_geoms.items() if geom in geoms}
        return touching[0] & touching[1]

    def step(self, n: int) -> None:
        arm = np.concatenate(list(self.arm_act.values()))
        target, start = self.data.ctrl.copy(), self._applied
        for j in range(1, n + 1):
            self.data.ctrl[arm] = start[arm] + (target[arm] - start[arm]) * (j / n)
            mujoco.mj_step(self.model, self.data)
        self.data.ctrl[:] = target
        self._applied = target.copy()

    def state_rad(self) -> np.ndarray:
        """14-D vector in the real dataset's order and units."""
        return np.concatenate([np.r_[self.q(s), opening_to_rad(self.opening(s))] for s in ARMS]).astype(np.float32)

    def velocity_rad(self) -> np.ndarray:
        scale = (REAL_GRIPPER_RANGE_RAD[1] - REAL_GRIPPER_RANGE_RAD[0]) / FINGER_OPEN
        return np.concatenate([np.r_[self.data.qvel[self.arms[s].arm_dofadr],
                                     self.data.qvel[self.finger_dofadr[s]] * scale] for s in ARMS]).astype(np.float32)

    def effort(self) -> np.ndarray:
        """Actuator torques; the gripper entry is the sim finger force (N), not the real motor's Nm."""
        force = self.data.actuator_force
        return np.concatenate([np.r_[force[self.arm_act[s]], force[self.grip_act[s]]] for s in ARMS]).astype(np.float32)


if __name__ == '__main__':
    spec = build_spec()
    model = spec.compile()
    SCENE_XML.write_text(spec.to_xml())
    names = lambda kind, n: [mujoco.mj_id2name(model, kind, i) for i in range(n)]
    print(f'wrote {SCENE_XML.name}: nq={model.nq} nu={model.nu}')
    print('actuators:', names(mujoco.mjtObj.mjOBJ_ACTUATOR, model.nu))
    print('cameras:', names(mujoco.mjtObj.mjOBJ_CAMERA, model.ncam))
