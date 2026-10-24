"""Real CityLearn 2023 Phase 3.3 integration (needs citylearn, and torch for training)."""
import importlib.util
import time
import unittest

import numpy as np

HAS_CL = importlib.util.find_spec("citylearn") is not None
HAS_TORCH = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(HAS_CL, "citylearn not installed")
class TestCityLearnAdapter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from marl.envs.citylearn_adapter import CityLearnAdapter
        cls.env = CityLearnAdapter(episode_time_steps=48)

    def test_contract(self):
        from marl.envs import check_env_contract
        check_env_contract(self.env, n_steps=10)

    def test_six_buildings_battery_only(self):
        self.assertEqual(self.env.n_agents, 6)
        self.assertEqual(self.env.action_dims, [1] * 6)
        for names in self.env.action_names:
            self.assertEqual(names, ["electrical_storage"])

    def test_soc_tracks_charging(self):
        self.env.reset(seed=0)
        s0 = self.env.soc()
        for _ in range(3):
            self.env.step([np.ones(1, np.float32)] * 6)
        self.assertTrue(np.all(self.env.soc() >= s0 - 1e-6))

    def test_global_state_has_no_duplicate_shared_columns(self):
        self.assertLess(self.env.state_dim, sum(self.env.obs_dims))


@unittest.skipUnless(HAS_CL and HAS_TORCH, "citylearn and torch needed")
class TestCityLearnTraining(unittest.TestCase):
    def test_one_update_per_arm_and_speed(self):
        from marl.envs.citylearn_adapter import CityLearnAdapter
        from marl.mappo import MAPPOConfig, MAPPOTrainer
        for arm in ("M0", "M1", "M1-RND"):
            env = CityLearnAdapter(episode_time_steps=168)
            cfg = MAPPOConfig.for_arm(arm, total_steps=512, rollout_steps=512,
                                      ppo_epochs=2, num_minibatches=2)
            t0 = time.perf_counter()
            hist = MAPPOTrainer(env, cfg).train()
            dt = time.perf_counter() - t0
            sps = hist[-1]["steps_per_sec"]
            print(f"\n  {arm}: rollout {sps:.0f} steps/s; 512 steps + update in {dt:.1f}s; "
                  f"=> ~{2208 * 100 / sps / 3600:.2f} h per 100 passes over 2208 steps")
            self.assertTrue(np.isfinite(hist[-1]["v_loss"]))


if __name__ == "__main__":
    unittest.main()
