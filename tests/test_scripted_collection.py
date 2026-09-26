"""Integration checks for physical scripted demos and native dataset compatibility."""
import tempfile
import unittest
from pathlib import Path

import mujoco
import numpy as np

from sim.panthera_env import PantheraSim
from sim.stack_task import stack_metrics
from teleop.episode import FIELDS
from teleop.render_vla_dataset import recording_times, interpolate
from teleop.dataset_contract import PhysicsClock
from tools.collect_scripted import Augment, DemoFailure, Planner
from tools.validate_scripted_dataset import replay
from sim.stack_task import ordered_two_stack_metrics


class ScriptedCollectionTest(unittest.TestCase):
    def test_two_blocks_fixed_start_and_replay(self):
        initial = []
        layouts = []
        for seed in (230924, 230925, 230926):
            planner = Planner(seed, blocks=2, arm_start='fixed')
            self.assertTrue(planner.run()['two_stack'])
            self.assertEqual(planner.sim.object_names, ['cube_red', 'cube_green'])
            self.assertFalse(any(s['name'].startswith('2_') for s in planner.stages))
            initial.append(planner.episode.rows[0]['q'])
            layouts.append(planner.episode.rows[0]['obj_pos'])
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory)
                planner.episode.save(path, {'scene': 'sim/panthera/scene_two_blocks.xml'}, 30, False)
                result = replay(path)
                self.assertTrue(result['success'], result)
                self.assertEqual(result['grasped_blocks'], [0])
        for q in initial[1:]:
            np.testing.assert_array_equal(q, initial[0])
        self.assertFalse(np.allclose(layouts[0], layouts[1]))

    def test_two_stack_requires_correct_order_alignment_and_table_support(self):
        positions = np.array([[.4, 0., .1175], [.4, 0., .0725]])
        self.assertTrue(ordered_two_stack_metrics(positions)['two_stack'])
        self.assertFalse(ordered_two_stack_metrics(positions[::-1])['two_stack'])
        self.assertFalse(ordered_two_stack_metrics(positions + [0, 0, .1])['two_stack'])
        positions[0, 0] += .02
        self.assertFalse(ordered_two_stack_metrics(positions)['two_stack'])

    def test_successful_demo_survives_native_save_and_act_clock_replay(self):
        planner = Planner(230927)
        self.assertTrue(planner.run()['three_stack'])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            planner.episode.save(path, {'seed': planner.seed}, 30, False)
            with np.load(path/'data.npz') as raw:
                self.assertTrue(set(FIELDS).issubset(raw.files))
                self.assertTrue({'sim_time','finger_q','finger_dq','physics_steps'}.issubset(raw.files))
                for value in raw.values():
                    self.assertTrue(np.isfinite(value).all())
                self.assertLessEqual(np.abs(np.diff(raw['ctrl'][:,:6],axis=0)).max(), .0800001)
                sim = PantheraSim()
                t, method = recording_times(raw, sim.dt)
                self.assertEqual(method, 'recorded_simulation_time')
                grid = np.arange(0, t[-1]+1e-9, 1/30)
                actions = interpolate(t, raw['ctrl'], grid)
                sim.reset(randomize=False)
                sim.data.qpos[sim.arm_qadr] = raw['q'][0]
                sim.data.qvel[sim.arm_dofadr] = raw['dq'][0]
                sim.data.qpos[sim.finger_qadr] = raw['finger_q'][0]
                sim.data.qvel[sim.finger_dofadr] = raw['finger_dq'][0]
                sim.set_object_poses(raw['obj_pos'][0], raw['obj_quat'][0])
                sim.data.ctrl[:] = actions[0]
                mujoco.mj_forward(sim.model, sim.data)
                clock = PhysicsClock(30, sim.dt)
                stable = 0
                for action in actions[1:]:
                    sim.set_arm_ctrl(action[:6])
                    sim.set_gripper(action[6]/.04)
                    sim.step(clock.next_steps())
                    success = stack_metrics(sim.object_poses()[0])['three_stack']
                    released = not any(sim.data.eq_active[e] for e in sim._grasp_eq)
                    stable = stable+1 if success and released else 0
                self.assertGreaterEqual(stable, 30)


    def test_state_waits_remove_pauses_without_changing_clean_labels(self):
        timed, state = Planner(230927), Planner(230927, waits='state')
        self.assertTrue(timed.run()['three_stack'])
        self.assertTrue(state.run()['three_stack'])
        self.assertNotIn('settle', [s['name'] for s in state.stages])
        still = lambda p: np.sum(np.abs(np.diff([r['q'] for r in p.episode.rows], axis=0)).max(1) < 1e-5)
        self.assertLess(still(state), still(timed) / 2)
        for row in state.episode.rows:
            np.testing.assert_array_equal(row['ctrl_label'], row['ctrl'])

    def test_augmented_labels_correct_the_offset_and_gate_the_jaws(self):
        augment = Augment(noise_pos=.008, noise_yaw=np.deg2rad(4), miss_prob=1.,
                          miss_offset=.025, max_attempts=3)
        for seed in range(230930, 230960):
            planner = Planner(seed, waits='state', augment=augment)
            try:
                planner.run()
            except DemoFailure:
                continue
            if planner.retries:
                break
        else:
            self.fail('no augmented episode with a re-grasp succeeded')
        rows = planner.episode.rows
        label, executed = np.array([r['ctrl_label'] for r in rows]), np.array([r['ctrl'] for r in rows])
        self.assertGreater(np.abs(label[:, :6] - executed[:, :6]).max(), .01)
        stages = sorted((s['start'], s['name']) for s in planner.stages) + [(len(rows), 'end')]
        spans = {}
        for (start, name), (end, _) in zip(stages, stages[1:]):
            spans.setdefault(name, []).append((start, end))
        # The sideways first close: the executed jaws shut, the label keeps them open.
        start, end = spans['1_close'][0]
        self.assertLess(executed[start:end, 6].min(), .01)
        self.assertGreater(label[start:end, 6].max(), .03)
        # Re-opening after a miss is labelled as opening.
        reopen = next(name for name in spans if name.endswith('_reopen'))
        for start, end in spans[reopen]:
            np.testing.assert_allclose(label[start:end, 6], executed[start:end, 6])


if __name__ == '__main__':
    unittest.main()
