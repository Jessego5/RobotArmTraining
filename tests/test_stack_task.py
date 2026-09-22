import unittest

import numpy as np

from sim.stack_task import CUBE_EDGE, TABLE_CUBE_Z, StackReward, stack_metrics


class StackMetricsTest(unittest.TestCase):
    def setUp(self):
        z = TABLE_CUBE_Z
        self.flat = np.array([[0.32, -0.10, z], [0.40, 0.0, z], [0.49, 0.11, z]])
        self.two = np.array([[0.40, 0.0, z], [0.40, 0.0, z + CUBE_EDGE], [0.50, 0.10, z]])
        self.three = np.array([
            [0.40, 0.0, z],
            [0.40, 0.0, z + CUBE_EDGE],
            [0.40, 0.0, z + 2 * CUBE_EDGE],
        ])

    def test_stack_levels(self):
        self.assertFalse(stack_metrics(self.flat)["two_stack"])
        self.assertTrue(stack_metrics(self.two)["two_stack"])
        self.assertFalse(stack_metrics(self.two)["three_stack"])
        self.assertTrue(stack_metrics(self.three)["three_stack"])

    def test_airborne_chain_is_not_success(self):
        airborne = self.three.copy()
        airborne[:, 2] += 0.15
        self.assertFalse(stack_metrics(airborne)["three_stack"])

    def test_success_requires_hold_and_release(self):
        reward = StackReward(success_hold_steps=3)
        ee = np.array([0.40, 0.0, 0.20])
        reward.reset(self.flat, ee)
        for _ in range(4):
            _value, success, _metrics = reward.step(self.three, ee, grasped=True)
            self.assertFalse(success)
        for expected in (False, False, True):
            _value, success, _metrics = reward.step(self.three, ee, grasped=False)
            self.assertEqual(success, expected)


if __name__ == "__main__":
    unittest.main()
