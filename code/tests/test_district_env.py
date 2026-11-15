"""Part A: DistrictBatteryEnv (reward, outage seeding, KPIs, baselines), numpy only.

Runs against a fake CityLearn that extends the one in test_citylearn_adapter.py
with what the wrapper reads: per-building net consumption and pricing (indexed
by episode step), outage models, and CityLearn's per-episode outage hook
(``Building.reset_power_outage_signal``, called from ``Building.reset``).
tests/test_citylearn_smoke.py repeats the key checks on the real package.
"""
import sys
import types
import unittest
from unittest import mock

import numpy as np

from marl.baselines import BasicBatteryRBCController, ZeroController, rollout
from marl.envs import check_env_contract
from marl.envs.district_env import DistrictBatteryEnv
from marl.envs.kpis import EpisodeLog, daily_peaks, raw_kpis, relative_kpis
from marl.envs.outages import CALENDAR_STEPS, calendar_outage_signals, outage_seed
from marl.envs.reward import H_FORMULA, DistrictReward
from tests.test_citylearn_adapter import N_BUILDINGS, OBS_NAMES, POWER_KW, FakeCityLearnEnv

FROZEN = {"observation_names": [list(OBS_NAMES) for _ in range(N_BUILDINGS)],
          "reward_norm": {"cost": 0.5, "load_sq": 4.0, "h_formula": H_FORMULA}}


def base_load(dataset_step, building):
    """Deterministic building load: daily shape, evening peak, a little PV at noon."""
    hour = dataset_step % 24
    return 0.5 + 0.1 * building + (1.0 if 17 <= hour <= 20 else 0.0) - (0.8 if 11 <= hour <= 14 else 0.0)


def price(dataset_step):
    return 0.2 if 16 <= dataset_step % 24 <= 20 else 0.05


class FakeOutageModel:
    """Like ReliabilityMetricsPowerOutage: a fixed schema seed, signal drawn from step 0."""

    def __init__(self, random_seed=57841):
        self.random_seed = random_seed

    def get_signals(self, time_steps, seconds_per_time_step=3600.0, **kwargs):
        rng = np.random.RandomState(self.random_seed)
        return (rng.random_sample(time_steps) < 0.01).astype(int)


class FakeBuilding:
    def __init__(self, index, env):
        self.index, self.env = index, env
        self.name = f"Building_{index + 1}"
        self.simulate_power_outage = self.stochastic_power_outage = True
        self.stochastic_power_outage_model = FakeOutageModel()
        self.time_step = 0

    def reset_power_outage_signal(self):
        # CityLearn's default: a fresh draw from the episode start, schema seed
        return self.stochastic_power_outage_model.get_signals(self.env.time_steps)

    def reset(self):
        n_steps = self.env.time_steps
        start = self.env.episode_tracker.episode_start_time_step
        self.time_step = 0
        self.electrical_storage = types.SimpleNamespace(
            electricity_consumption=np.zeros(n_steps, np.float32))
        self.net_electricity_consumption = np.zeros(n_steps, np.float32)
        self.pricing = types.SimpleNamespace(
            electricity_pricing=np.array([price(start + k) for k in range(n_steps)], np.float32))
        self.power_outage_signal = np.asarray(self.reset_power_outage_signal(), np.float32)


class FakeDistrictCityLearnEnv(FakeCityLearnEnv):
    def __init__(self, schema, **kwargs):
        super().__init__(schema, **kwargs)
        self.seconds_per_time_step = 3600.0
        self.buildings = [FakeBuilding(b, self) for b in range(N_BUILDINGS)]

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.random_seed = seed
        self.episode_tracker.next_episode(
            self.time_steps, self.kwargs.get("rolling_episode_split", False),
            self.kwargs.get("random_episode_split", False), self.random_seed)
        self.time_step = 0
        for building in self.buildings:
            building.reset()
        return self._obs(), {}

    def step(self, actions):
        if self.terminated:
            raise RuntimeError("call reset()")
        current = self.time_step
        dataset_step = self.episode_tracker.episode_start_time_step + current
        for building, action, power in zip(self.buildings, actions, POWER_KW):
            energy = action[0] * power
            building.electrical_storage.electricity_consumption[current] = energy
            building.net_electricity_consumption[current] = base_load(dataset_step, building.index) + energy

        is_last_transition = current >= max(self.time_steps - 2, 0)
        if not is_last_transition:
            for building in self.buildings:
                building.time_step += 1
        self.time_step += 1
        return self._obs(), [-1.0] * N_BUILDINGS, self.terminated, False, {}


def fake_district_modules():
    """Patch ``import citylearn.citylearn`` to return the district fake."""
    package = types.ModuleType("citylearn")
    module = types.ModuleType("citylearn.citylearn")
    module.CityLearnEnv = FakeDistrictCityLearnEnv
    package.citylearn = module
    return mock.patch.dict(sys.modules, {"citylearn": package, "citylearn.citylearn": module})


def make(**kwargs):
    kwargs.setdefault("frozen", FROZEN)
    kwargs.setdefault("schema", "fake-schema.json")
    with fake_district_modules():
        return DistrictBatteryEnv(**kwargs)


def run_episode(env, controller, seed):
    with fake_district_modules():     # kpis() may build the R0 reference env
        return rollout(env, controller, seed)


class TestReward(unittest.TestCase):
    def test_formula(self):
        reward = DistrictReward(cost_norm=0.5, import_sq_norm=4.0, alpha=2.0, beta=0.5)
        self.assertAlmostEqual(reward(3.0, 0.1), -(2.0 * 0.3 / 0.5 + 0.5 * 9.0 / 4.0))

    def test_exports_are_neither_costed_nor_penalized(self):
        reward = DistrictReward(1.0, 1.0)
        self.assertEqual(reward(-2.0, 0.3), 0.0)            # cost 0, max(0, L)^2 = 0
        self.assertAlmostEqual(reward(2.0, 0.3), -(0.6 + 4.0))

    def test_norms_must_be_positive(self):
        with self.assertRaises(ValueError):
            DistrictReward(0.0, 1.0)

    def test_env_reward_matches_hand_computation(self):
        env = make(episode_time_steps=24, alpha=1.5, beta=0.7)
        env.reset(seed=4)
        rng = np.random.default_rng(0)
        for _ in range(10):
            actions = rng.uniform(-1, 1, N_BUILDINGS)
            step = env.episode_window()[0] + env.env.time_step
            _, reward, _, _, info = env.step([np.array([a]) for a in actions])
            load = sum(base_load(step, b) + actions[b] * POWER_KW[b] for b in range(N_BUILDINGS))
            expected = -(1.5 * price(step) * max(0.0, load) / 0.5 + 0.7 * max(0.0, load) ** 2 / 4.0)
            self.assertAlmostEqual(info["district_load"], load, places=4)
            self.assertAlmostEqual(reward, expected, places=4)
            self.assertIsInstance(reward, float)

    def test_r0_normalization_gives_minus_alpha_plus_beta(self):
        """freeze_env's norms make the mean R0 reward on the block exactly -(alpha + beta)."""
        from scripts.freeze_env import r0_norms
        window = (0, 1487)
        raw = make(window=window, reward_norm=(1.0, 1.0), r0_reference=False)
        with fake_district_modules():
            cost_norm, import_sq_norm, _ = r0_norms(raw, seed=0)
        env = make(window=window, reward_norm=(cost_norm, import_sq_norm), r0_reference=False)
        record = run_episode(env, ZeroController(env), seed=0)
        self.assertAlmostEqual(float(np.mean(record["rewards"])), -2.0, places=4)

    def test_norms_frozen_for_another_formula_are_refused(self):
        stale = dict(FROZEN, reward_norm={"cost": 0.5, "load_sq": 4.0})     # pre-change file
        with self.assertRaises(RuntimeError):
            make(frozen=stale)
        with self.assertRaises(RuntimeError):
            make(frozen=dict(FROZEN, reward_norm=dict(FROZEN["reward_norm"], h_formula="L^2")))

    def test_missing_norm_is_an_error(self):
        with self.assertRaises(RuntimeError):
            make(frozen={"observation_names": FROZEN["observation_names"]})


class TestContractAndSplit(unittest.TestCase):
    def test_contract_every_split(self):
        for kwargs in (dict(split="train", episode_time_steps=48), dict(split="validation"),
                       dict(split="test"), dict(window=(100, 160))):
            env = make(**kwargs)
            check_env_contract(env, n_steps=30)

    def test_windows_follow_the_split(self):
        env = make(split="test")
        env.reset(seed=0)
        self.assertEqual(env.episode_window(), (1848, 2207))
        env = make(window=(24, 71))
        env.reset(seed=0)
        self.assertEqual(env.episode_window(), (24, 71))

    def test_bad_window(self):
        with self.assertRaises(ValueError):
            make(window=(0, 2000))

    def test_frozen_observation_set_is_enforced(self):
        changed = dict(FROZEN, observation_names=[["hour"]] * N_BUILDINGS)
        with self.assertRaises(RuntimeError):
            make(frozen=changed)


class TestOutageSeeding(unittest.TestCase):
    def test_seed_derivation(self):
        self.assertEqual(outage_seed(3, 1), outage_seed(3, 1))
        self.assertEqual(len({outage_seed(3, b) for b in range(6)}), 6)
        self.assertNotEqual(outage_seed(3, 0), outage_seed(4, 0))

    def test_models_are_not_modified(self):
        models = [FakeOutageModel() for _ in range(3)]
        calendar_outage_signals(models, reset_seed=9)
        self.assertTrue(all(model.random_seed == 57841 for model in models))

    def test_calendar_depends_only_on_seed_and_differs_by_building(self):
        env = make()
        first, again, other = env.outage_calendar(5), make().outage_calendar(5), env.outage_calendar(6)
        self.assertEqual(first.shape, (N_BUILDINGS, CALENDAR_STEPS))
        np.testing.assert_array_equal(first, again)
        self.assertFalse(np.array_equal(first, other))
        self.assertFalse(all(np.array_equal(first[0], first[b]) for b in range(1, N_BUILDINGS)))

    def test_citylearn_gets_the_calendar_slice(self):
        env = make(split="train", episode_time_steps=168)
        for seed in (1, 2, 3):
            env.reset(seed=seed)
            start, end = env.episode_window()
            expected = env.outage_calendar(seed)[:, start:end + 1]
            signals = np.stack([b.power_outage_signal for b in env.env.buildings])
            np.testing.assert_array_equal(signals, expected)

    def test_outages_sit_on_the_calendar_whatever_the_window(self):
        """Overlapping windows with the same seed see the same outage hours."""
        seed = next(s for s in range(200) if make().outage_calendar(s)[:, 100:300].sum() > 0)
        short, long = make(window=(100, 300)), make(window=(0, 500))
        short.reset(seed=seed)
        long.reset(seed=seed)
        short_signal = np.stack([b.power_outage_signal for b in short.env.buildings])
        long_signal = np.stack([b.power_outage_signal for b in long.env.buildings])
        np.testing.assert_array_equal(short_signal, long_signal[:, 100:301])
        self.assertGreater(short_signal.sum(), 0)

    def test_outages_off(self):
        env = make(outages=False)
        self.assertEqual(env.outage_calendar(1).sum(), 0)


class TestKPIs(unittest.TestCase):
    def test_daily_peaks_use_calendar_days(self):
        steps = np.arange(20, 52)                # day 0 (20-23), day 1, day 2 (48-51)
        loads = np.arange(len(steps), dtype=float)
        np.testing.assert_array_equal(daily_peaks(steps, loads), [3.0, 27.0, 31.0])

    def test_raw_and_relative(self):
        log = EpisodeLog()
        for step, load, step_price in ((0, 2.0, 0.1), (1, 4.0, 0.2), (24, -1.0, 0.1), (25, 3.0, 0.1)):
            log.append(step, load, step_price, -1.0)
        kpis = raw_kpis(log)
        self.assertAlmostEqual(kpis["avg_daily_peak"], (4.0 + 3.0) / 2)
        self.assertAlmostEqual(kpis["cost"], 0.2 + 0.8 + 0.0 + 0.3)
        self.assertEqual((kpis["n_days"], kpis["n_steps"]), (2, 4))
        relative = relative_kpis(kpis, {"avg_daily_peak": 7.0, "cost": 2.6})
        self.assertAlmostEqual(relative["primary"], 0.5)
        self.assertAlmostEqual(relative["cost_rel_r0"], 0.5)

    def test_r0_scores_one_and_r1_differs(self):
        env = make(split="validation")
        r0 = run_episode(env, ZeroController(env), seed=3)["kpis"]
        self.assertAlmostEqual(r0["primary"], 1.0)
        self.assertAlmostEqual(r0["cost_rel_r0"], 1.0)
        r1 = run_episode(env, BasicBatteryRBCController(env), seed=3)["kpis"]
        self.assertNotAlmostEqual(r1["primary"], 1.0)
        self.assertEqual(r1["window"], [1488, 1847])
        self.assertEqual(r1["n_steps"], 359)

    def test_r0_reference_is_cut_to_the_same_length(self):
        env = make(split="validation")
        with fake_district_modules():
            record = rollout(env, ZeroController(env), seed=3, max_steps=50)
        self.assertEqual(record["kpis"]["r0"]["n_steps"], 50)
        self.assertAlmostEqual(record["kpis"]["primary"], 1.0)

    def test_kpis_through_michaels_evaluate(self):
        from marl.mappo.evaluate import evaluate
        from marl.mappo.normalize import Preprocessor

        class ZeroPolicy:
            def act(self, actor_inputs, deterministic=True):
                return np.zeros((N_BUILDINGS, 1), np.float32), np.zeros(N_BUILDINGS, np.float32)

        env = make(split="test")
        preprocessor = Preprocessor(env.obs_dims, env.state_dim)
        with fake_district_modules():
            records = evaluate(env, ZeroPolicy(), preprocessor, use_messages=False,
                               n_replicates=2, env_seed=0, on_episode_end=lambda e: e.kpis())
        self.assertEqual(len(records), 2)
        self.assertAlmostEqual(records[0]["kpis"]["primary"], 1.0)
        self.assertAlmostEqual(records[0]["return"], records[0]["kpis"]["return"], places=3)


class TestBaselines(unittest.TestCase):
    def test_rbc_schedule(self):
        env = make()
        controller = BasicBatteryRBCController(env)
        self.assertAlmostEqual(controller.action_for_hour(10), 0.11)
        self.assertAlmostEqual(controller.action_for_hour(20), -0.067)
        self.assertAlmostEqual(controller.action_for_hour(0), controller.action_for_hour(24))

    def test_rollout_record(self):
        env = make(window=(0, 47))
        record = run_episode(env, BasicBatteryRBCController(env), seed=1)
        self.assertEqual(record["actions"].shape, (47, N_BUILDINGS))
        self.assertEqual(record["battery_energy"].shape, (47, N_BUILDINGS))
        self.assertAlmostEqual(record["return"], record["kpis"]["return"], places=3)


class TestLoadInfoAndExports(unittest.TestCase):
    def test_building_net_load_sums_to_district_load(self):
        env = make(window=(0, 47))
        with fake_district_modules():
            env.reset(seed=0)
            seen_export = False
            for _ in range(24):
                _, reward, _, _, info = env.step([np.zeros(1, np.float32)] * N_BUILDINGS)
                self.assertEqual(info["building_net_load"].shape, (N_BUILDINGS,))
                self.assertAlmostEqual(float(info["building_net_load"].sum()),
                                       info["district_load"], places=4)
                if info["district_load"] < 0:   # noon PV surplus in the fake: no penalty at all
                    seen_export = True
                    self.assertEqual(reward, 0.0)
        self.assertTrue(seen_export, "the fake district should export around noon")

    def test_export_stats(self):
        from scripts.export_check import export_stats
        stats = export_stats([3.0, -1.0, 2.0, -2.0])
        self.assertEqual(stats["export_steps"], 2)
        self.assertAlmostEqual(stats["export_fraction"], 0.5)
        self.assertAlmostEqual(stats["min_load"], -2.0)
        self.assertAlmostEqual(stats["export_share_of_old_load_sq"], 5.0 / 18.0)


class TestPeerSocFeatures(unittest.TestCase):
    def test_dataset_shapes_and_peer_order(self):
        from marl.comms.channel import peer_indices
        from marl.envs.features import collect, peer_soc_dataset
        env = make(window=(0, 47))
        with fake_district_modules():
            obs_seq, soc_seq = collect(env, BasicBatteryRBCController(env), seed=1)
        self.assertEqual(obs_seq.shape, (47, N_BUILDINGS, len(OBS_NAMES)))
        self.assertEqual(soc_seq.shape, (47, N_BUILDINGS))
        data = peer_soc_dataset(obs_seq, soc_seq, env.observation_names)
        for agent, row in enumerate(data):
            self.assertEqual(row["peers"], peer_indices(N_BUILDINGS, agent))
            self.assertEqual(row["Y"].shape, (47, N_BUILDINGS - 1))
            np.testing.assert_allclose(row["Y"], soc_seq[:, row["peers"]])
            np.testing.assert_allclose(row["X"], obs_seq[:, agent, :])

    def test_drop_own_soc(self):
        from marl.envs.features import SOC_NAME, peer_soc_dataset
        names = [list(OBS_NAMES)] * N_BUILDINGS
        obs = np.zeros((5, N_BUILDINGS, len(OBS_NAMES)))
        data = peer_soc_dataset(obs, np.zeros((5, N_BUILDINGS)), names, drop_own_soc=True)
        self.assertNotIn(SOC_NAME, data[0]["feature_names"])
        self.assertEqual(data[0]["X"].shape, (5, len(OBS_NAMES) - 1))
        with self.assertRaises(ValueError):           # no SoC column: refuse, don't keep X as is
            peer_soc_dataset(obs, np.zeros((5, N_BUILDINGS)),
                             [["a"] * len(OBS_NAMES)] * N_BUILDINGS, drop_own_soc=True)


if __name__ == "__main__":
    unittest.main()
