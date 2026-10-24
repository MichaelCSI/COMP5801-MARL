"""Buffer/GAE, normalizers, env contract, rollout worker and evaluation (numpy only)."""
import unittest

import numpy as np

from marl.comms import DeliverySchedule, PeerSoCChannel, message_dim, peer_indices
from marl.envs import EnvContractError, PeerSoCMatchEnv, ToyDistrictEnv, check_env_contract
from marl.mappo import (ARMS, MAPPOConfig, Preprocessor, RandomPolicy, RolloutBuffer,
                        RolloutWorker, RunningMeanStd, build_actor_inputs, compute_gae,
                        evaluate)
from marl.mappo.config import STREAM_ENV, stream


# ---------------------------------------------------------------------------
class TestGAE(unittest.TestCase):
    def test_single_step_terminal(self):
        adv, ret = compute_gae([1.0], [[0.5]], [True], [[0.0]], [9.0], 0.99, 0.95)
        self.assertAlmostEqual(float(adv[0, 0]), 0.5, places=6)
        self.assertAlmostEqual(float(ret[0, 0]), 1.0, places=6)

    def test_matches_hand_computation(self):
        r = np.array([1.0, 0.0, 2.0])
        v = np.array([[0.5], [0.2], [0.1]])
        g, l, last = 0.9, 0.8, np.array([0.3])
        d = r + g * np.array([0.2, 0.1, 0.3]) - v[:, 0]
        a2 = d[2]
        a1 = d[1] + g * l * a2
        a0 = d[0] + g * l * a1
        adv, ret = compute_gae(r, v, [False] * 3, np.zeros((3, 1)), last, g, l)
        np.testing.assert_allclose(adv[:, 0], [a0, a1, a2], rtol=1e-5)
        np.testing.assert_allclose(ret[:, 0], adv[:, 0] + v[:, 0], rtol=1e-5)

    def test_lambda_one_is_discounted_return(self):
        r = np.array([1.0, 2.0, 3.0, 4.0])
        v = np.random.default_rng(0).normal(size=(4, 1))
        _, ret = compute_gae(r, v, [False, False, False, True], np.zeros((4, 1)),
                             [100.0], 0.9, 1.0)
        exp = [1 + .9 * 2 + .81 * 3 + .729 * 4, 2 + .9 * 3 + .81 * 4, 3 + .9 * 4, 4.0]
        np.testing.assert_allclose(ret[:, 0], exp, rtol=1e-5)

    def test_no_leak_across_episodes_and_bootstrap(self):
        r = np.array([0.0, 0.0, 5.0, 0.0])
        v = np.zeros((4, 1))
        dones = [False, True, False, False]
        boot = np.array([[0.0], [2.0], [0.0], [0.0]])  # truncated after step 1, V=2
        adv, _ = compute_gae(r, v, dones, boot, [0.0], 0.5, 1.0)
        # step 1: r + g*boot = 1.0 ; step 0: 0 + .5*1 = .5 ; the 5.0 in ep2 must not leak
        np.testing.assert_allclose(adv[:2, 0], [0.5, 1.0], rtol=1e-6)
        np.testing.assert_allclose(adv[2:, 0], [5.0, 0.0], rtol=1e-6)

    def test_multiple_critics(self):
        r = np.array([1.0, 1.0])
        v = np.array([[0.0, 1.0], [0.0, 1.0]])
        adv, _ = compute_gae(r, v, [False, False], np.zeros((2, 2)), [0.0, 3.0], 1.0, 1.0)
        np.testing.assert_allclose(adv[:, 0], [2.0, 1.0])   # each critic uses its own V
        np.testing.assert_allclose(adv[:, 1], [4.0, 3.0])


class TestBuffer(unittest.TestCase):
    def test_add_and_overflow(self):
        b = RolloutBuffer(3, [4, 4], [1, 1], [8])
        for _ in range(3):
            b.add([np.ones(4)] * 2, [np.ones(8)], np.zeros((2, 1)), np.zeros(2), 1.0, False,
                  np.zeros(1), np.zeros(1))
        self.assertTrue(b.full)
        with self.assertRaises(IndexError):
            b.add([np.ones(4)] * 2, [np.ones(8)], np.zeros((2, 1)), np.zeros(2), 1.0, False,
                  np.zeros(1), np.zeros(1))

    def test_minibatches_cover_all_indices_once(self):
        b = RolloutBuffer(10, [1], [1], [1])
        b.ptr = 10
        idx = np.concatenate(list(b.minibatches(3, np.random.default_rng(0))))
        self.assertEqual(sorted(idx.tolist()), list(range(10)))


class TestNormalizers(unittest.TestCase):
    def test_running_moments_match_numpy(self):
        x = np.random.default_rng(0).normal(3, 2, size=(1000, 3))
        r = RunningMeanStd((3,), epsilon=1e-12)
        for chunk in np.array_split(x, 17):
            r.update(chunk)
        np.testing.assert_allclose(r.mean, x.mean(0), rtol=1e-6)
        np.testing.assert_allclose(r.var, x.var(0), rtol=1e-6)

    def test_state_dict_roundtrip(self):
        r = RunningMeanStd((2,))
        r.update(np.array([[1.0, 2.0], [3.0, 5.0]]))
        r2 = RunningMeanStd((2,))
        r2.load_state_dict(r.state_dict())
        np.testing.assert_array_equal(r.normalize([1, 1]), r2.normalize([1, 1]))

    def test_frozen_when_update_false(self):
        pre = Preprocessor([2], 2)
        pre.local([np.array([5.0, 5.0])], update=True)
        m = pre.obs_rms[0].mean.copy()
        pre.local([np.array([100.0, 100.0])], update=False)
        np.testing.assert_array_equal(pre.obs_rms[0].mean, m)


# ---------------------------------------------------------------------------
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

    def test_toy_district_matches_proposal_shape(self):
        env = ToyDistrictEnv()
        self.assertEqual(env.n_agents, 6)
        self.assertEqual(env.action_dims, [1] * 6)


# ---------------------------------------------------------------------------
def make_worker(env, arm, seed=0, p_fixed=None, policy=None):
    cfg = MAPPOConfig.for_arm(arm, seed=seed)
    ch = sch = None
    if cfg.use_messages:
        ch = PeerSoCChannel(env.n_agents, 1.0, seed=stream(seed, 2),
                            force_mask=cfg.force_mask_train)
        if p_fixed is not None:
            sch = DeliverySchedule("fixed", p=p_fixed)
        elif cfg.train_schedule == "fixed":
            sch = DeliverySchedule("fixed", p=cfg.train_p)
        else:
            sch = DeliverySchedule("random", choices=cfg.rnd_p_choices, seed=stream(seed, 3))
    msg = message_dim(env.n_agents) if cfg.use_messages else 0
    actor_dims = [d + msg for d in env.obs_dims]
    critic_dims = [env.state_dim] if cfg.critic == "central" else actor_dims
    pol = policy or RandomPolicy(env.n_agents, env.action_dims, seed=seed,
                                 n_critics=len(critic_dims))
    pre = Preprocessor(env.obs_dims, env.state_dim)
    w = RolloutWorker(env, pol, pre, cfg.use_messages, ch, sch, cfg.critic,
                      stream(seed, STREAM_ENV))
    buf = RolloutBuffer(64, actor_dims, env.action_dims, critic_dims)
    return w, buf


class TestRolloutWorker(unittest.TestCase):
    def test_input_dims_per_arm(self):
        env = ToyDistrictEnv()
        for arm, extra in (("M0", 0), ("M1", 10), ("M1-RND", 10), ("I0", 0)):
            w, buf = make_worker(env, arm)
            w.collect(buf)
            self.assertEqual(buf.actor_inputs[0].shape[1], env.obs_dims[0] + extra, arm)
        w, buf = make_worker(env, "I0")
        self.assertEqual(buf.K, env.n_agents)       # IPPO: one critic per agent

    def test_critic_never_sees_dropped_messages(self):
        """Central-critic inputs are identical at p=1 and p=0."""
        env1, env0 = ToyDistrictEnv(), ToyDistrictEnv()
        w1, b1 = make_worker(env1, "M1", p_fixed=1.0)
        w0, b0 = make_worker(env0, "M1", p_fixed=0.0)
        w1.collect(b1)
        w0.collect(b0)
        np.testing.assert_array_equal(b1.critic_inputs[0], b0.critic_inputs[0])
        # ...while the actors' message columns differ
        self.assertFalse(np.array_equal(b1.actor_inputs[0], b0.actor_inputs[0]))
        np.testing.assert_array_equal(b0.actor_inputs[0][:, -10:], 0.0)

    def test_messages_are_raw_soc_not_normalized(self):
        env = ToyDistrictEnv()
        w, buf = make_worker(env, "M1")
        socs = []

        class Spy(RandomPolicy):
            def act(self, x, deterministic=False):
                socs.append(env.soc().copy())
                return super().act(x, deterministic)

        w.policy = Spy(6, [1] * 6, seed=0)
        w.collect(buf)
        for t in range(10):
            for i in range(6):
                np.testing.assert_allclose(buf.actor_inputs[i][t, -10::2],
                                           socs[t][peer_indices(6, i)], rtol=1e-6)
                np.testing.assert_array_equal(buf.actor_inputs[i][t, -9::2], 1.0)

    def test_env_reset_seeds_paired_across_arms(self):
        seeds = {}
        for arm in ("M0", "M1", "M1-RND"):
            w, buf = make_worker(ToyDistrictEnv(episode_steps=8), arm, seed=3)
            w.collect(buf)
            seeds[arm] = w.reset_seeds
        self.assertEqual(seeds["M0"], seeds["M1"])
        self.assertEqual(seeds["M0"], seeds["M1-RND"])
        self.assertGreater(len(seeds["M0"]), 5)

    def test_m1_rnd_draws_p_per_episode(self):
        w, buf = make_worker(ToyDistrictEnv(episode_steps=16), "M1-RND", seed=1)
        buf = RolloutBuffer(16 * 200, [17] * 6, [1] * 6, [27])
        w.collect(buf)
        ps = [e["p"] for e in w.episodes]
        self.assertEqual(set(ps), {0.0, 0.5, 0.75, 1.0})
        for e in w.episodes:   # realised delivery rate tracks that episode's p
            self.assertAlmostEqual(e["delivered"] / e["links"], e["p"], delta=0.12)

    def test_episode_boundaries_and_truncation_bootstrap(self):
        class ConstV(RandomPolicy):
            def value(self, x):
                return np.full(self.n_critics, 7.0, np.float32)

        env = ToyDistrictEnv(episode_steps=10)
        w, buf = make_worker(env, "M0", policy=ConstV(6, [1] * 6, n_critics=1))
        w.collect(buf)
        done_idx = np.where(buf.dones)[0]
        np.testing.assert_array_equal(done_idx, [9, 19, 29, 39, 49, 59])
        np.testing.assert_array_equal(buf.bootstrap[done_idx, 0], 7.0)
        self.assertEqual(buf.last_value[0], 7.0)
        self.assertEqual(len(w.episodes), 6)

    def test_rollouts_continue_across_collect_calls(self):
        w, buf = make_worker(ToyDistrictEnv(episode_steps=48), "M0")
        w.collect(buf)
        w.collect(buf)
        self.assertEqual(w.total_steps, 128)
        self.assertEqual([e["length"] for e in w.episodes], [48, 48])


# ---------------------------------------------------------------------------
class OraclePolicy:
    """For PeerSoCMatchEnv: copy the right-hand peer's SoC from the message."""

    def __init__(self, n):
        self.n = n

    def act(self, x, deterministic=True):
        out = np.zeros((self.n, 1), np.float32)
        for i in range(self.n):
            target = (i + 1) % self.n
            pos = peer_indices(self.n, i).index(target)
            v, m = x[i][-2 * (self.n - 1) + 2 * pos], x[i][-2 * (self.n - 1) + 2 * pos + 1]
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
        r_mask = evaluate(env, OraclePolicy(3), pre, True, p=0.5, n_replicates=2)
        r_force = evaluate(env, OraclePolicy(3), pre, True, p=0.5, n_replicates=2,
                           force_mask=True)
        self.assertLess(np.mean([r["return"] for r in r_force]),
                        np.mean([r["return"] for r in r_mask]))

    def test_replicates_share_outages_but_not_drops(self):
        env = ToyDistrictEnv(episode_steps=24)
        pre = Preprocessor(env.obs_dims, env.state_dim)
        recs = evaluate(env, RandomPolicy(6, [1] * 6, seed=0), pre, True, p=0.5,
                        n_replicates=3, env_seed=11)
        # env reset with the same seed -> same initial SoC in every replicate
        for r in recs[1:]:
            np.testing.assert_array_equal(r["soc"][0], recs[0]["soc"][0])
            self.assertFalse(np.array_equal(r["delivered"], recs[0]["delivered"]))

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


class TestConfig(unittest.TestCase):
    def test_arms_differ_only_in_message_fields(self):
        base = MAPPOConfig.for_arm("M0").to_dict()
        allowed = {"arm", "use_messages", "train_schedule", "train_p", "critic",
                   "force_mask_train"}
        for arm in ARMS:
            d = MAPPOConfig.for_arm(arm).to_dict()
            diff = {k for k in d if d[k] != base[k]}
            self.assertTrue(diff <= allowed, (arm, diff))
            MAPPOConfig.for_arm(arm).validate()

    def test_bad_configs(self):
        with self.assertRaises(ValueError):
            MAPPOConfig.for_arm("M2")
        with self.assertRaises(ValueError):
            MAPPOConfig(use_messages=True, train_schedule="none").validate()


if __name__ == "__main__":
    unittest.main()
