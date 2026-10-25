"""MessageEnv (actor/critic inputs for on-policy), env contract, evaluation (numpy only)."""
import unittest

import numpy as np

from marl.comms import DeliverySchedule, PeerSoCChannel, peer_indices
from marl.envs import EnvContractError, PeerSoCMatchEnv, ToyDistrictEnv, check_env_contract
from marl.mappo import (MAPPOConfig, MessageEnv, Preprocessor, RandomPolicy,
                        RunningMeanStd, build_actor_inputs, evaluate, make_message_env)


def run(menv, steps, seed=0):
    """Drive a MessageEnv with random actions; return per-step inputs."""
    rng = np.random.default_rng(seed)
    obs, share = menv.reset()
    out = []
    for _ in range(steps):
        soc = menv.last_soc.copy()
        out.append((obs, share, soc))
        a = rng.uniform(-1, 1, (menv.n_agents, 1))
        obs, share, rew, done, trunc, info = menv.step(a)
        out[-1] += (rew, done, trunc, info)
    return out


class TestNormalizers(unittest.TestCase):
    def test_running_moments_match_numpy(self):
        x = np.random.default_rng(0).normal(3, 2, size=(1000, 3))
        r = RunningMeanStd((3,), epsilon=1e-12)
        for chunk in np.array_split(x, 17):
            r.update(chunk)
        np.testing.assert_allclose(r.mean, x.mean(0), rtol=1e-6)
        np.testing.assert_allclose(r.var, x.var(0), rtol=1e-6)

    def test_frozen_when_update_false(self):
        pre = Preprocessor([2], 2)
        pre.local([np.array([5.0, 5.0])], update=True)
        m = pre.obs_rms[0].mean.copy()
        pre.local([np.array([100.0, 100.0])], update=False)
        np.testing.assert_array_equal(pre.obs_rms[0].mean, m)

    def test_state_dict_roundtrip(self):
        pre = Preprocessor([2, 2], 3)
        pre.local([np.ones(2), 2 * np.ones(2)], update=True)
        pre2 = Preprocessor([2, 2], 3)
        pre2.load_state_dict(pre.state_dict())
        np.testing.assert_array_equal(pre.local([np.ones(2)] * 2, False)[1],
                                      pre2.local([np.ones(2)] * 2, False)[1])


class BrokenEnv(ToyDistrictEnv):
    """Returns one reward per building, which the contract forbids."""

    def step(self, actions):
        obs, r, te, tr, info = super().step(actions)
        return obs, [r] * self.n_agents, te, tr, info


class TestEnvContract(unittest.TestCase):
    def test_toy_envs_pass(self):
        check_env_contract(ToyDistrictEnv())
        check_env_contract(PeerSoCMatchEnv())

    def test_per_building_reward_is_rejected(self):
        with self.assertRaises(EnvContractError):
            check_env_contract(BrokenEnv())


class TestMessageEnvSpaces(unittest.TestCase):
    def test_spaces_per_arm(self):
        env = ToyDistrictEnv()
        for arm, extra in (("M0", 0), ("M1", 10), ("M1-RND", 10), ("I0", 0)):
            menv = make_message_env(env, MAPPOConfig.for_arm(arm))
            self.assertEqual(menv.actor_dims, [7 + extra] * 6, arm)
            # on-policy picks the network head by class name and reads .shape
            self.assertEqual(menv.observation_space[0].__class__.__name__, "Box")
            self.assertEqual(menv.action_space[0].shape, (1,))
            if arm == "I0":
                self.assertEqual(menv.critic_dims, menv.actor_dims)
            else:
                self.assertEqual(menv.critic_dims, [env.state_dim] * 6)

    def test_needs_channel_for_messages(self):
        with self.assertRaises(ValueError):
            MessageEnv(ToyDistrictEnv(), True)


class TestMessageEnvInputs(unittest.TestCase):
    def test_critic_never_sees_dropped_messages(self):
        """share_obs identical at p=1 and p=0; actor message columns differ."""
        def make(p):
            env = ToyDistrictEnv()
            ch = PeerSoCChannel(6, 1.0, seed=0)
            return MessageEnv(env, True, ch, DeliverySchedule("fixed", p=p),
                              Preprocessor(env.obs_dims, env.state_dim), env_seed=1)
        r1, r0 = run(make(1.0), 40), run(make(0.0), 40)
        for (o1, s1, *_), (o0, s0, *_) in zip(r1, r0):
            for i in range(6):
                np.testing.assert_array_equal(s1[i], s0[i])
                np.testing.assert_array_equal(o0[i][-10:], 0.0)
                np.testing.assert_array_equal(o1[i][:-10], o0[i][:-10])
        self.assertFalse(np.array_equal(r1[3][0][0], r0[3][0][0]))

    def test_messages_are_raw_soc_of_the_current_step(self):
        env = ToyDistrictEnv()
        menv = make_message_env(env, MAPPOConfig.for_arm("M1"))
        for obs, share, soc, *_ in run(menv, 30):
            for i in range(6):
                np.testing.assert_allclose(obs[i][-10::2], soc[peer_indices(6, i)], rtol=1e-6)
                np.testing.assert_array_equal(obs[i][-9::2], 1.0)

    def test_local_features_are_normalized(self):
        env = ToyDistrictEnv()
        menv = make_message_env(env, MAPPOConfig.for_arm("M0"))
        steps = run(menv, 2000)
        x = np.stack([o[0] for o, *_ in steps[500:]])
        self.assertLess(np.abs(x.mean(0)).max(), 0.5)

    def test_shared_reward_replicated_per_agent(self):
        menv = make_message_env(ToyDistrictEnv(), MAPPOConfig.for_arm("M0"))
        for *_, rew, done, trunc, info in run(menv, 10):
            self.assertEqual(rew.shape, (6, 1))
            self.assertTrue(np.all(rew == rew[0, 0]))

    def test_auto_reset_and_truncation_flags(self):
        menv = make_message_env(ToyDistrictEnv(episode_steps=10), MAPPOConfig.for_arm("M0"))
        steps = run(menv, 35)
        done_at = [t for t, s in enumerate(steps) if s[4]]
        self.assertEqual(done_at, [9, 19, 29])
        self.assertTrue(all(steps[t][5] for t in done_at))      # time-limit truncations
        self.assertEqual(steps[9][6]["episode"]["length"], 10)
        self.assertEqual(len(menv.reset_seeds), 4)


class TestPairingAndSchedule(unittest.TestCase):
    def test_env_reset_seeds_paired_across_arms(self):
        seeds = {}
        for arm in ("M0", "M1", "M1-RND", "I0"):
            menv = make_message_env(ToyDistrictEnv(episode_steps=8),
                                    MAPPOConfig.for_arm(arm, seed=3))
            run(menv, 80)
            seeds[arm] = menv.reset_seeds
        self.assertGreater(len(seeds["M0"]), 5)
        for arm in seeds:
            self.assertEqual(seeds[arm], seeds["M0"], arm)

    def test_different_run_seeds_differ(self):
        a = make_message_env(ToyDistrictEnv(episode_steps=8), MAPPOConfig.for_arm("M0", seed=0))
        b = make_message_env(ToyDistrictEnv(episode_steps=8), MAPPOConfig.for_arm("M0", seed=1))
        run(a, 40)
        run(b, 40)
        self.assertNotEqual(a.reset_seeds, b.reset_seeds)

    def test_m1_rnd_draws_p_per_episode(self):
        menv = make_message_env(ToyDistrictEnv(episode_steps=16),
                                MAPPOConfig.for_arm("M1-RND", seed=1))
        run(menv, 16 * 200)
        ps = [e["p"] for e in menv.episodes]
        self.assertEqual(set(ps), {0.0, 0.5, 0.75, 1.0})
        for e in menv.episodes:
            self.assertAlmostEqual(e["delivery_rate"], e["p"], delta=0.12)

    def test_m1_always_full_delivery(self):
        menv = make_message_env(ToyDistrictEnv(episode_steps=16), MAPPOConfig.for_arm("M1"))
        run(menv, 64)
        self.assertTrue(all(e["delivery_rate"] == 1.0 for e in menv.episodes))

    def test_no_mask_arm_holds_mask_at_one(self):
        menv = make_message_env(ToyDistrictEnv(episode_steps=16),
                                MAPPOConfig.for_arm("M1-RND-NM", seed=2))
        for obs, *_ in run(menv, 200):
            np.testing.assert_array_equal(obs[0][-9::2], 1.0)


class OraclePolicy:
    """For PeerSoCMatchEnv: copy the right-hand peer's SoC from the message."""

    def __init__(self, n):
        self.n = n

    def act(self, x, deterministic=True):
        out = np.zeros((self.n, 1), np.float32)
        k = 2 * (self.n - 1)
        for i in range(self.n):
            pos = peer_indices(self.n, i).index((i + 1) % self.n)
            v, m = x[i][-k + 2 * pos], x[i][-k + 2 * pos + 1]
            out[i, 0] = 2 * v - 1 if m == 1 else 0.0
        return out, np.zeros(self.n, np.float32)


class TestEvaluate(unittest.TestCase):
    def test_messages_carry_true_soc_end_to_end(self):
        """Oracle reward ~ 0 at p=1 and ~ -(1-p)/3 otherwise."""
        env = PeerSoCMatchEnv(n_agents=3, episode_steps=400)
        pre = Preprocessor(env.obs_dims, env.state_dim)
        for p in (1.0, 0.5, 0.0):
            recs = evaluate(env, OraclePolicy(3), pre, True, p=p, n_replicates=3)
            mean_r = np.mean([r["rewards"].mean() for r in recs])
            self.assertAlmostEqual(mean_r, -(1 - p) / 3, delta=0.03, msg=f"p={p}")

    def test_force_mask_breaks_the_oracle(self):
        env = PeerSoCMatchEnv(n_agents=3, episode_steps=400)
        pre = Preprocessor(env.obs_dims, env.state_dim)
        good = evaluate(env, OraclePolicy(3), pre, True, p=0.5, n_replicates=2)
        forced = evaluate(env, OraclePolicy(3), pre, True, p=0.5, n_replicates=2,
                          force_mask=True)
        self.assertLess(np.mean([r["return"] for r in forced]),
                        np.mean([r["return"] for r in good]))

    def test_replicates_share_outages_but_not_drops(self):
        env = ToyDistrictEnv(episode_steps=24)
        pre = Preprocessor(env.obs_dims, env.state_dim)
        recs = evaluate(env, RandomPolicy(6, [1] * 6, seed=0), pre, True, p=0.5,
                        n_replicates=3, env_seed=11)
        for r in recs[1:]:
            np.testing.assert_array_equal(r["soc"][0], recs[0]["soc"][0])
            self.assertFalse(np.array_equal(r["delivered"], recs[0]["delivered"]))

    def test_eval_does_not_update_normalizers(self):
        env = ToyDistrictEnv(episode_steps=24)
        pre = Preprocessor(env.obs_dims, env.state_dim)
        before = pre.state_dict()
        evaluate(env, RandomPolicy(6, [1] * 6), pre, False, n_replicates=1)
        np.testing.assert_array_equal(pre.state_dict()["state"]["mean"],
                                      before["state"]["mean"])

    def test_record_format(self):
        env = ToyDistrictEnv(episode_steps=24)
        pre = Preprocessor(env.obs_dims, env.state_dim)
        r = evaluate(env, RandomPolicy(6, [1] * 6), pre, True, p=0.9, n_replicates=1,
                     on_episode_end=lambda e: {"steps": e.t})[0]
        self.assertEqual(r["actions"].shape, (24, 6))
        self.assertEqual(r["soc"].shape, (24, 6))
        self.assertEqual(r["delivered"].shape, (24, 6, 6))
        self.assertEqual(r["rewards"].shape, (24,))
        self.assertTrue(np.all(np.abs(r["actions"]) <= 1))
        self.assertIsNotNone(r["kpis"])
        r0 = evaluate(env, RandomPolicy(6, [1] * 6), pre, False, n_replicates=1)[0]
        self.assertIsNone(r0["delivered"])

    def test_battery_energy_recorded_and_differs_from_actions(self):
        """Synchrony needs what the batteries did, not what was commanded."""
        class AlwaysCharge:
            def act(self, x, deterministic=True):
                return np.ones((6, 1), np.float32), np.zeros(6, np.float32)

        env = ToyDistrictEnv(episode_steps=24)
        pre = Preprocessor(env.obs_dims, env.state_dim)
        r = evaluate(env, AlwaysCharge(), pre, False, n_replicates=1)[0]
        self.assertEqual(r["battery_energy"].shape, (24, 6))
        np.testing.assert_array_equal(r["actions"], 1.0)
        self.assertTrue(np.all(r["battery_energy"][-1] == 0))   # batteries full by the end
        self.assertGreater(r["battery_energy"][0].sum(), 0)
        env2 = PeerSoCMatchEnv(n_agents=3, episode_steps=10)
        r2 = evaluate(env2, OraclePolicy(3), Preprocessor(env2.obs_dims, env2.state_dim),
                      True, n_replicates=1)[0]
        self.assertIsNone(r2["battery_energy"])

    def test_build_actor_inputs(self):
        x = build_actor_inputs([np.ones(2), np.zeros(2)], np.array([[5.0, 1.0], [6.0, 0.0]]))
        np.testing.assert_array_equal(x[0], [1, 1, 5, 1])
        np.testing.assert_array_equal(build_actor_inputs([np.ones(2)], None)[0], [1, 1])


if __name__ == "__main__":
    unittest.main()
