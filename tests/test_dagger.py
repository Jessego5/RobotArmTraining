"""Expert takeover and failure triggers for expert-in-the-loop ACT collection."""
import unittest

import mujoco
import numpy as np

from tools.collect_scripted import DemoFailure, Planner, grasp_rotation, mat_to_quat
from tools.dagger_act import FailureMonitor

SEED = 230927  # a layout the scripted collector completes


def started(seed=SEED):
    """A planner with its arm initialised as in run(), nothing recorded yet."""
    planner = Planner(seed)
    sim = planner.sim
    start = planner.rng.uniform([.30, -.16, .18], [.46, .16, .28])
    quat = mat_to_quat(grasp_rotation(sim, 0))
    q, _, _ = sim.ik(start, quat, max_joint_step=None, iters=150)
    sim.data.qpos[sim.arm_qadr] = q
    sim.data.qvel[:] = 0
    sim.data.qpos[sim.finger_qadr] = [.04, -.04]
    sim.set_arm_ctrl(q, immediate=True)
    sim.set_gripper(1)
    mujoco.mj_forward(sim.model, sim.data)
    planner.target, planner.quat, planner.qctrl = sim.ee_pos(), sim.ee_quat(), q
    return planner


def hover_over_red(planner, sideways=0.):
    sim = planner.sim
    rotation = grasp_rotation(sim, 0)
    grasp = sim.object_poses()[0][0] + .018 * rotation[:, 0] + [0, sideways, 0]
    planner.move('approach', grasp + [0, 0, .07], mat_to_quat(rotation), 1.)
    try:
        planner.move('descend', grasp)
    except DemoFailure:
        pass  # pressing onto the cube is itself a realistic policy state
    return grasp


class ExpertTakeoverTest(unittest.TestCase):
    def test_finish_keeps_carrying_the_due_cube(self):
        planner = started()
        planner.pick(1, 0)
        planner.move('halfway', planner.target + [0, 0, .03])
        expert = Planner(SEED, sim=planner.sim)
        self.assertTrue(expert.finish()['three_stack'])
        self.assertEqual(expert.stages[0]['name'], '1_raise')
        self.assertFalse(planner.sim.grasped)

    def test_finish_recovers_from_a_missed_grasp(self):
        planner = started()
        hover_over_red(planner, sideways=.03)
        for _ in range(15):
            planner.tick(planner.target, planner.quat, 0.)
        self.assertFalse(planner.sim.grasp_flags()[0])
        expert = Planner(SEED, sim=planner.sim)
        self.assertTrue(expert.finish()['three_stack'])
        self.assertEqual([s['name'] for s in expert.stages[:2]], ['recover_open', 'recover_rise'])


class FailureMonitorTest(unittest.TestCase):
    def monitor(self, planner, stall_seconds=60.):
        return FailureMonitor(Planner(SEED, sim=planner.sim), 30, stall_seconds, .015)

    def test_closing_beside_the_cube_triggers(self):
        planner = started()
        hover_over_red(planner, sideways=.03)
        monitor = self.monitor(planner)
        self.assertIsNone(monitor.update(1.))
        self.assertEqual(monitor.update(.2), 'misaligned_close')

    def test_closing_at_the_grasp_point_does_not_trigger(self):
        planner = started()
        hover_over_red(planner)
        monitor = self.monitor(planner)
        self.assertIsNone(monitor.update(1.))
        self.assertIsNone(monitor.update(.2))

    def test_grasping_out_of_order_triggers(self):
        planner = started()
        planner.pick(1, 2)  # blue first
        self.assertEqual(self.monitor(planner).update(0.), 'wrong_block')

    def test_no_progress_triggers_a_stall(self):
        monitor = self.monitor(started(), stall_seconds=.1)
        results = [monitor.update(1.) for _ in range(5)]
        self.assertIn('stall', results)


if __name__ == '__main__':
    unittest.main()
