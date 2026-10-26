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

OBS_NAMES = ["hour", "electricity_pricing", "electrical_storage_soc", "non_shiftable_load"]
SHARED_OBS_NAMES = ["hour", "electricity_pricing"]
POWER_KW = [3.32, 3.32, 1.61, 1.61, 3.32, 1.61]
N_BUILDINGS = 6


class _Tracker:
    """CityLearn's EpisodeTracker window choice."""

    def __init__(self, simulation_start, simulation_end):
        self.simulation_start, self.simulation_end = simulation_start, simulation_end
        self.episode = -1
        self.episode_start_time_step = self.episode_end_time_step = None

    def next_episode(self, episode_time_steps, rolling, random_split, random_seed):
        self.episode += 1
        latest_start = (self.simulation_end + 1) - episode_time_steps
        start_spacing = 1 if rolling else episode_time_steps
        candidate_starts = list(range(self.simulation_start, latest_start + 1, start_spacing))

        if random_split:
            # raises ValueError once the seed product reaches 2**32, like CityLearn
            window_rng = np.random.RandomState(int(random_seed * (self.episode + 1)))
            window_index = (0 if len(candidate_starts) == 1
                            else int(window_rng.choice(len(candidate_starts))))
        else:
            window_index = self.episode % len(candidate_starts)

        self.episode_start_time_step = candidate_starts[window_index]
        self.episode_end_time_step = self.episode_start_time_step + episode_time_steps - 1


class _Space:
    shape = (1,)


class _Building:
    def __init__(self, n_steps):
        self.time_step = 0
        self.electrical_storage = types.SimpleNamespace(
            electricity_consumption=np.zeros(n_steps, np.float32))


class FakeCityLearnEnv:
    def __init__(self, schema, **kwargs):
        self.kwargs = kwargs
        self.random_seed = 2023
        self.episode_tracker = _Tracker(kwargs.get("simulation_start_time_step", 0),
                                        kwargs.get("simulation_end_time_step", 2207))
        self.time_steps = int(kwargs.get("episode_time_steps") or 2208)

        self.action_space = [_Space() for _ in range(N_BUILDINGS)]
        self.action_names = [list(kwargs.get("active_actions")) for _ in range(N_BUILDINGS)]
        self.observation_names = [list(OBS_NAMES) for _ in range(N_BUILDINGS)]
        self.shared_observations = list(SHARED_OBS_NAMES)
        self.time_step = 0

    def _obs(self):
        dataset_step = self.episode_tracker.episode_start_time_step + self.time_step
        return [[dataset_step % 24, 0.2, 0.5, 1.0 + building] for building in range(N_BUILDINGS)]

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.random_seed = seed
        self.episode_tracker.next_episode(
            self.time_steps, self.kwargs.get("rolling_episode_split", False),
            self.kwargs.get("random_episode_split", False), self.random_seed)

        self.time_step = 0
        self.buildings = [_Building(self.time_steps) for _ in range(N_BUILDINGS)]
        return self._obs(), {}

    @property
    def terminated(self):
        return self.time_step >= self.time_steps - 1

    def step(self, actions):
        if self.terminated:
            raise RuntimeError("call reset()")

        current = self.time_step
        for building, action, power in zip(self.buildings, actions, POWER_KW):
            building.electrical_storage.electricity_consumption[current] = action[0] * power

        # buildings don't advance on the terminal step; the env clock always does
        is_last_transition = current >= max(self.time_steps - 2, 0)
        if not is_last_transition:
            for building in self.buildings:
                building.time_step += 1
        self.time_step += 1
        return self._obs(), [-1.0] * N_BUILDINGS, self.terminated, False, {}


def fake_modules():
    """Patch ``import citylearn.citylearn`` to return the fake."""
    package = types.ModuleType("citylearn")
    module = types.ModuleType("citylearn.citylearn")
    module.CityLearnEnv = FakeCityLearnEnv
    package.citylearn = module
    return mock.patch.dict(sys.modules, {"citylearn": package, "citylearn.citylearn": module})


class TestSplitKwargs(unittest.TestCase):
    def test_blocks_match_proposal(self):
        self.assertEqual(SPLITS["train"], (0, 1487))
        self.assertEqual(SPLITS["validation"], (1488, 1847))
        self.assertEqual(SPLITS["test"], (1848, 2207))

    def test_train_uses_random_windows_inside_block(self):
        env_kwargs = split_kwargs("train", 168)
        self.assertEqual((env_kwargs["simulation_start_time_step"],
                          env_kwargs["simulation_end_time_step"]), (0, 1487))
        self.assertTrue(env_kwargs["rolling_episode_split"] and env_kwargs["random_episode_split"])
        self.assertEqual(env_kwargs["episode_time_steps"], 168)

    def test_eval_blocks_are_one_whole_episode(self):
        for split, (start, end) in (("validation", (1488, 1847)), ("test", (1848, 2207))):
            env_kwargs = split_kwargs(split, 168)
            self.assertEqual(env_kwargs["episode_time_steps"], end - start + 1)
            self.assertFalse(env_kwargs["random_episode_split"])

    def test_bad_arguments(self):
        with self.assertRaises(ValueError):
            split_kwargs("holdout", 168)
        with self.assertRaises(ValueError):
            split_kwargs("train", 2000)


class TestSeedMapping(unittest.TestCase):
    def test_range_and_determinism(self):
        for seed in (0, 1, 99_999, 2 ** 31 - 1, 12345678):
            mapped = citylearn_seed(seed)
            self.assertTrue(1 <= mapped <= 100_000)
            self.assertEqual(mapped, citylearn_seed(seed))

        self.assertIsNone(citylearn_seed(None))
        self.assertLess(100_000 * 40_000, 2 ** 32)   # valid for 40k episodes

    def test_raw_large_seed_would_crash_citylearn(self):
        """Why the mapping exists: CityLearn's window RNG rejects big products."""
        tracker = _Tracker(0, 1487)
        tracker.next_episode(168, True, True, 2 ** 31 - 1)   # seed * 1
        tracker.next_episode(168, True, True, 2 ** 31 - 1)   # seed * 2 < 2**32, still fine
        with self.assertRaises(ValueError):                # seed * 3 >= 2**32
            tracker.next_episode(168, True, True, 2 ** 31 - 1)


class TestAdapterWithFakeCityLearn(unittest.TestCase):
    def make(self, **adapter_kwargs):
        with fake_modules():
            return CityLearnAdapter(**adapter_kwargs)

    def test_contract_and_battery_only(self):
        env = self.make(episode_time_steps=24)
        check_env_contract(env, n_steps=30)
        self.assertEqual(env.env.kwargs["active_actions"], ["electrical_storage"])
        self.assertEqual(env.state_dim, 2 + N_BUILDINGS * 2)   # 2 shared + 2 local each

    def test_training_windows_random_inside_train_block(self):
        env = self.make(episode_time_steps=168)
        rng = np.random.default_rng(0)
        starts = []
        for _ in range(60):     # big seeds, many episodes: must not crash
            env.reset(seed=int(rng.integers(2 ** 31 - 1)))
            start, end = env.episode_window()
            self.assertGreaterEqual(start, 0)
            self.assertLessEqual(end, 1487)
            self.assertEqual(end - start + 1, 168)
            starts.append(start)

        self.assertGreater(len(set(starts)), 40)
        self.assertTrue(any(start % 168 for start in starts))   # not just weekly blocks

    def test_windows_paired_for_same_seed_sequence(self):
        first, second = self.make(), self.make()
        for seed in (5, 99, 123456789, 7):
            first.reset(seed=seed)
            second.reset(seed=seed)
            self.assertEqual(first.episode_window(), second.episode_window())

    def test_test_split_is_whole_block(self):
        env = self.make(split="test")
        env.reset(seed=0)
        self.assertEqual(env.episode_window(), (1848, 2207))

    def test_episode_end_is_truncation(self):
        env = self.make(episode_time_steps=10)
        env.reset(seed=1)
        end_flags = []
        for _ in range(9):     # CityLearn runs time_steps - 1 transitions
            _, _, terminated, truncated, _ = env.step([np.ones(1)] * N_BUILDINGS)
            end_flags.append((terminated, truncated))

        self.assertEqual(end_flags[-1], (False, True))
        self.assertTrue(all(flags == (False, False) for flags in end_flags[:-1]))

    def test_battery_energy_is_the_step_just_taken_including_terminal(self):
        env = self.make(episode_time_steps=6)
        env.reset(seed=3)
        rng = np.random.default_rng(1)
        for _ in range(5):
            actions = rng.uniform(-1, 1, N_BUILDINGS).astype(np.float32)
            _, _, _, truncated, info = env.step([np.array([action]) for action in actions])
            np.testing.assert_allclose(info["battery_energy"], actions * np.array(POWER_KW),
                                       rtol=1e-5)
        self.assertTrue(truncated)   # the last check above was the terminal step


if __name__ == "__main__":
    unittest.main()
