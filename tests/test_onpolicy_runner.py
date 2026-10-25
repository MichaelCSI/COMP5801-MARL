"""OnPolicyRunner: on-policy MAPPO wired to the message channel (needs torch)."""
import importlib.util
import os
import tempfile
import unittest

import numpy as np

HAS_TORCH = importlib.util.find_spec("torch") is not None

if HAS_TORCH:
    import torch

    from marl.envs import PeerSoCMatchEnv, ToyDistrictEnv
    from marl.mappo import ARMS, MAPPOConfig, OnPolicyRunner

FAST = dict(total_steps=400, rollout_steps=200, onpolicy_overrides={"ppo_epoch": 2})


@unittest.skipUnless(HAS_TORCH, "torch not installed")
class TestStructure(unittest.TestCase):
    def test_uses_onpolicy_classes(self):
        r = OnPolicyRunner(ToyDistrictEnv(), MAPPOConfig.for_arm("M1", **FAST))
        self.assertEqual(type(r.policy[0]).__module__,
                         "onpolicy.algorithms.r_mappo.algorithm.rMAPPOPolicy")
        self.assertEqual(type(r.trainer[0]).__name__, "R_MAPPO")
        self.assertEqual(type(r.buffer[0]).__name__, "SeparatedReplayBuffer")

    def test_one_actor_per_building_no_sharing(self):
        r = OnPolicyRunner(ToyDistrictEnv(), MAPPOConfig.for_arm("M1", **FAST))
        self.assertEqual(len(r.policy), 6)
        ids = [{id(p) for p in pol.actor.parameters()} for pol in r.policy]
        for i in range(6):
            for j in range(i + 1, 6):
                self.assertFalse(ids[i] & ids[j])

    def test_input_dims(self):
        env = ToyDistrictEnv()
        r = OnPolicyRunner(env, MAPPOConfig.for_arm("M1", **FAST))
        self.assertEqual(r.buffer[0].obs.shape[-1], 17)
        self.assertEqual(r.buffer[0].share_obs.shape[-1], env.state_dim)
        r0 = OnPolicyRunner(env, MAPPOConfig.for_arm("M0", **FAST))
        self.assertEqual(r0.buffer[0].obs.shape[-1], 7)
        ri = OnPolicyRunner(env, MAPPOConfig.for_arm("I0", **FAST))
        self.assertEqual(ri.buffer[0].share_obs.shape[-1], 7)   # IPPO: local critic


@unittest.skipUnless(HAS_TORCH, "torch not installed")
class TestTraining(unittest.TestCase):
    def test_every_arm_trains_and_logs_finite_stats(self):
        for arm in ARMS:
            logs = []
            r = OnPolicyRunner(ToyDistrictEnv(episode_steps=48),
                               MAPPOConfig.for_arm(arm, seed=0, **FAST), logger=logs.append)
            r.run()
            self.assertEqual(len(logs), 2, arm)
            for k in ("value_loss", "policy_loss", "dist_entropy", "ratio", "steps_per_sec"):
                self.assertTrue(np.isfinite(logs[-1][k]), (arm, k))
            if arm.startswith("M1"):
                self.assertIsNotNone(logs[-1]["train_p_mean"])

    def test_same_seed_is_reproducible(self):
        outs = []
        for _ in range(2):
            r = OnPolicyRunner(ToyDistrictEnv(), MAPPOConfig.for_arm("M1-RND", seed=5, **FAST))
            r.run()
            outs.append([p.detach().clone() for pol in r.policy for p in pol.actor.parameters()])
        for a, b in zip(*outs):
            self.assertTrue(torch.allclose(a, b))

    def test_update_changes_actor_and_critic(self):
        r = OnPolicyRunner(ToyDistrictEnv(), MAPPOConfig.for_arm("M0", **FAST))
        a0 = [p.detach().clone() for p in r.policy[0].actor.parameters()]
        c0 = [p.detach().clone() for p in r.policy[0].critic.parameters()]
        r.run()
        self.assertTrue(any(not torch.equal(x, p) for x, p in
                            zip(a0, r.policy[0].actor.parameters())))
        self.assertTrue(any(not torch.equal(x, p) for x, p in
                            zip(c0, r.policy[0].critic.parameters())))

    def test_truncation_sets_bad_masks(self):
        r = OnPolicyRunner(ToyDistrictEnv(episode_steps=50),
                           MAPPOConfig.for_arm("M0", total_steps=200, rollout_steps=200,
                                               onpolicy_overrides={"ppo_epoch": 1}))
        r.warmup()
        for step in range(200):
            vals, acts, logps, rnns, rnns_c = r.collect(step)
            obs, share, rew, done, trunc, _ = r.menv.step([a[0] for a in acts])
            r.insert(obs, share, rew, done, trunc, vals, acts, logps, rnns, rnns_c)
        b = r.buffer[0]
        ends = np.where(b.masks[1:, 0, 0] == 0)[0]
        np.testing.assert_array_equal(ends, [49, 99, 149, 199])
        np.testing.assert_array_equal(np.where(b.bad_masks[1:, 0, 0] == 0)[0], ends)

    def test_checkpoint_roundtrip(self):
        r = OnPolicyRunner(ToyDistrictEnv(), MAPPOConfig.for_arm("M1-RND", seed=2, **FAST))
        r.run()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "ck.pt")
            r.save(path)
            r2 = OnPolicyRunner.load(path, ToyDistrictEnv())
        self.assertEqual(r2.cfg.arm, "M1-RND")
        vn1, vn2 = r.trainer[0].value_normalizer, r2.trainer[0].value_normalizer
        self.assertTrue(torch.allclose(vn1.running_mean, vn2.running_mean))
        e1 = r.evaluate(ToyDistrictEnv(), p=0.5, n_replicates=2, env_seed=3)
        e2 = r2.evaluate(ToyDistrictEnv(), p=0.5, n_replicates=2, env_seed=3)
        for a, b in zip(e1, e2):
            np.testing.assert_allclose(a["actions"], b["actions"], rtol=1e-6)

    def test_evaluate_p_levels_and_mask_check(self):
        r = OnPolicyRunner(ToyDistrictEnv(), MAPPOConfig.for_arm("M1-RND", **FAST))
        r.run()
        for p in (1.0, 0.9, 0.5, 0.0):
            recs = r.evaluate(ToyDistrictEnv(episode_steps=24), p=p, n_replicates=2)
            rate = recs[0]["delivered"].sum() / (24 * 30)
            self.assertAlmostEqual(rate, p, delta=0.15)
        recs = r.evaluate(ToyDistrictEnv(episode_steps=24), p=0.0, n_replicates=1,
                          force_mask=True)
        self.assertTrue(recs[0]["force_mask"])

    def test_deterministic_eval_is_repeatable(self):
        r = OnPolicyRunner(ToyDistrictEnv(), MAPPOConfig.for_arm("M1", **FAST))
        a = r.evaluate(ToyDistrictEnv(episode_steps=24), p=1.0, n_replicates=1)[0]
        b = r.evaluate(ToyDistrictEnv(episode_steps=24), p=1.0, n_replicates=1)[0]
        np.testing.assert_array_equal(a["actions"], b["actions"])


@unittest.skipUnless(HAS_TORCH, "torch not installed")
class TestLearningSignal(unittest.TestCase):
    """On a task solvable only through messages, M1 must beat M0's ceiling (-1/3)."""

    def _train(self, arm):
        cfg = MAPPOConfig.for_arm(arm, seed=0, total_steps=60_000, rollout_steps=1000,
                                  onpolicy_overrides={"gamma": 0.5})
        r = OnPolicyRunner(PeerSoCMatchEnv(n_agents=3, episode_steps=32), cfg)
        r.run()
        recs = r.evaluate(PeerSoCMatchEnv(n_agents=3, episode_steps=200), p=1.0,
                          n_replicates=2)
        return r, float(np.mean([x["rewards"].mean() for x in recs]))

    def test_messages_are_learned(self):
        r1, m1 = self._train("M1")
        _, m0 = self._train("M0")
        print(f"\n  learning test: M1 reward {m1:.3f}, M0 reward {m0:.3f} (M0 ceiling -0.333)")
        self.assertLess(m0, -0.30)
        self.assertGreater(m1, -0.25)
        drop = r1.evaluate(PeerSoCMatchEnv(n_agents=3, episode_steps=200), p=0.0,
                           n_replicates=1)
        self.assertLess(float(drop[0]["rewards"].mean()), m1)   # fragile when messages drop


if __name__ == "__main__":
    unittest.main()
