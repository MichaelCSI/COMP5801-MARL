"""CityLearnAdapter logic against a fake CityLearnEnv (numpy only).

The fake copies the CityLearn 3.0.2 behaviour the adapter depends on, read
from its source (citylearn/base.py EpisodeTracker, citylearn/citylearn.py,
citylearn/internal/runtime.py):

* window choice: rolling splits make every step a candidate start; random
  splits pick one with np.random.RandomState(random_seed * (episode + 1)),
  which raises once that product reaches 2**32;
* ``terminated`` is ``time_step >= time_steps - 1`` and ``truncated`` is always
  False;
* on the terminal transition the env clock advances but building clocks don't.

tests/test_citylearn_smoke.py runs the same checks on the real package.
"""
import sys
import types
import unittest
from unittest import mock

import numpy as np

from marl.envs import check_env_contract
from marl.envs.citylearn_adapter import (SPLITS, CityLearnAdapter, citylearn_seed,
                                         split_kwargs)

NAMES = ["hour", "electricity_pricing", "electrical_storage_soc", "non_shiftable_load"]
SHARED = ["hour", "electricity_pricing"]
POWER = [3.32, 3.32, 1.61, 1.61, 3.32, 1.61]


class _Tracker:
    def __init__(self, start, end):
        self.sim_start, self.sim_end = start, end
        self.episode = -1
        self.episode_start_time_step = self.episode_end_time_step = None

    def next_episode(self, episode_time_steps, rolling, random_split, random_seed):
        self.episode += 1
        latest = (self.sim_end + 1) - episode_time_steps
        step = 1 if rolling else episode_time_steps
        starts = list(range(self.sim_start, latest + 1, step))
        if random_split:
            nprs = np.random.RandomState(int(random_seed * (self.episode + 1)))
            ix = 0 if len(starts) == 1 else int(nprs.choice(len(starts)))
        else:
            ix = self.episode % len(starts)
        self.episode_start_time_step = starts[ix]
        self.episode_end_time_step = starts[ix] + episode_time_steps - 1


class _Space:
    shape = (1,)


class _Building:
    def __init__(self, i, n_steps):
        self.time_step = 0
        self.electrical_storage = types.SimpleNamespace(
            electricity_consumption=np.zeros(n_steps, np.float32))


class FakeCityLearnEnv:
    instances = []

    def __init__(self, schema, **kw):
        FakeCityLearnEnv.instances.append(self)
        self.kwargs = kw
        self.random_seed = 2023
        self.episode_tracker = _Tracker(kw.get("simulation_start_time_step", 0),
                                        kw.get("simulation_end_time_step", 2207))
        self.time_steps = int(kw.get("episode_time_steps") or 2208)
        self.action_space = [_Space() for _ in range(6)]
        self.action_names = [list(kw.get("active_actions")) for _ in range(6)]
        self.observation_names = [list(NAMES) for _ in range(6)]
        self.shared_observations = list(SHARED)
        self.time_step = 0

    def _obs(self):
        t = self.episode_tracker.episode_start_time_step + self.time_step
        return [[t % 24, 0.2, 0.5, 1.0 + i] for i in range(6)]

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.random_seed = seed
        self.episode_tracker.next_episode(
            self.time_steps, self.kwargs.get("rolling_episode_split", False),
            self.kwargs.get("random_episode_split", False), self.random_seed)
        self.time_step = 0
        self.buildings = [_Building(i, self.time_steps) for i in range(6)]
        return self._obs(), {}

    @property
    def terminated(self):
        return self.time_step >= self.time_steps - 1

    def step(self, actions):
        if self.terminated:
            raise RuntimeError("call reset()")
        t = self.time_step
        for b, a, p in zip(self.buildings, actions, POWER):
            b.electrical_storage.electricity_consumption[t] = a[0] * p
        last = t >= max(self.time_steps - 2, 0)
        if not last:                       # buildings don't advance on the terminal step
            for b in self.buildings:
                b.time_step += 1
        self.time_step += 1
        return self._obs(), [-1.0] * 6, self.terminated, False, {}


def fake_modules():
    pkg = types.ModuleType("citylearn")
    sub = types.ModuleType("citylearn.citylearn")
    sub.CityLearnEnv = FakeCityLearnEnv
    pkg.citylearn = sub
    return mock.patch.dict(sys.modules, {"citylearn": pkg, "citylearn.citylearn": sub})


class TestSplitKwargs(unittest.TestCase):
    def test_blocks_match_proposal(self):
        self.assertEqual(SPLITS["train"], (0, 1487))
        self.assertEqual(SPLITS["validation"], (1488, 1847))
        self.assertEqual(SPLITS["test"], (1848, 2207))

    def test_train_uses_random_windows_inside_block(self):
        k = split_kwargs("train", 168)
        self.assertEqual((k["simulation_start_time_step"], k["simulation_end_time_step"]),
                         (0, 1487))
        self.assertTrue(k["rolling_episode_split"] and k["random_episode_split"])
        self.assertEqual(k["episode_time_steps"], 168)

    def test_eval_blocks_are_one_whole_episode(self):
        for split, (a, b) in (("validation", (1488, 1847)), ("test", (1848, 2207))):
            k = split_kwargs(split, 168)
            self.assertEqual(k["episode_time_steps"], b - a + 1)
            self.assertFalse(k["random_episode_split"])

    def test_bad_arguments(self):
        with self.assertRaises(ValueError):
            split_kwargs("holdout", 168)
        with self.assertRaises(ValueError):
            split_kwargs("train", 2000)


class TestSeedMapping(unittest.TestCase):
    def test_range_and_determinism(self):
        for s in (0, 1, 99_999, 2 ** 31 - 1, 12345678):
            v = citylearn_seed(s)
            self.assertTrue(1 <= v <= 100_000)
            self.assertEqual(v, citylearn_seed(s))
        self.assertIsNone(citylearn_seed(None))
        self.assertLess(100_000 * 40_000, 2 ** 32)   # valid for 40k episodes

    def test_raw_large_seed_would_crash_citylearn(self):
        """Why the mapping exists: CityLearn's window RNG rejects big products."""
        tr = _Tracker(0, 1487)
        tr.next_episode(168, True, True, 2 ** 31 - 1)   # seed * 1
        tr.next_episode(168, True, True, 2 ** 31 - 1)   # seed * 2 < 2**32, still fine
        with self.assertRaises(ValueError):           # seed * 3 >= 2**32
            tr.next_episode(168, True, True, 2 ** 31 - 1)


class TestAdapterWithFakeCityLearn(unittest.TestCase):
    def make(self, **kw):
        with fake_modules():
            return CityLearnAdapter(**kw)

    def test_contract_and_battery_only(self):
        env = self.make(episode_time_steps=24)
        check_env_contract(env, n_steps=30)
        self.assertEqual(env.env.kwargs["active_actions"], ["electrical_storage"])
        self.assertEqual(env.state_dim, 2 + 6 * 2)

    def test_training_windows_random_inside_train_block(self):
        env = self.make(episode_time_steps=168)
        rng = np.random.default_rng(0)
        starts = []
        for _ in range(60):     # big seeds, many episodes: must not crash
            env.reset(seed=int(rng.integers(2 ** 31 - 1)))
            a, b = env.episode_window()
            self.assertGreaterEqual(a, 0)
            self.assertLessEqual(b, 1487)
            self.assertEqual(b - a + 1, 168)
            starts.append(a)
        self.assertGreater(len(set(starts)), 40)
        self.assertTrue(any(s % 168 for s in starts))   # not just weekly blocks

    def test_windows_paired_for_same_seed_sequence(self):
        e1, e2 = self.make(), self.make()
        for s in (5, 99, 123456789, 7):
            e1.reset(seed=s)
            e2.reset(seed=s)
            self.assertEqual(e1.episode_window(), e2.episode_window())

    def test_test_split_is_whole_block(self):
        env = self.make(split="test")
        env.reset(seed=0)
        self.assertEqual(env.episode_window(), (1848, 2207))

    def test_episode_end_is_truncation(self):
        env = self.make(episode_time_steps=10)
        env.reset(seed=1)
        ends = []
        for _ in range(9):     # CityLearn runs time_steps - 1 transitions
            _, _, term, trunc, _ = env.step([np.ones(1)] * 6)
            ends.append((term, trunc))
        self.assertEqual(ends[-1], (False, True))
        self.assertTrue(all(e == (False, False) for e in ends[:-1]))

    def test_battery_energy_is_the_step_just_taken_including_terminal(self):
        env = self.make(episode_time_steps=6)
        env.reset(seed=3)
        rng = np.random.default_rng(1)
        for _ in range(5):
            a = rng.uniform(-1, 1, 6).astype(np.float32)
            _, _, _, trunc, info = env.step([np.array([x]) for x in a])
            np.testing.assert_allclose(info["battery_energy"], a * np.array(POWER), rtol=1e-5)
        self.assertTrue(trunc)   # the last check above was the terminal step


if __name__ == "__main__":
    unittest.main()
