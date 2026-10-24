"""MAPPO networks, PPO update, checkpoints and a learning test (needs torch)."""
import importlib.util
import os
import tempfile
import unittest

import numpy as np

HAS_TORCH = importlib.util.find_spec("torch") is not None

if HAS_TORCH:
    import torch

    from marl.envs import PeerSoCMatchEnv, ToyDistrictEnv
    from marl.mappo import ARMS, MAPPOConfig, MAPPOTrainer

FAST = dict(total_steps=512, rollout_steps=256, ppo_epochs=2, num_minibatches=2)


@unittest.skipUnless(HAS_TORCH, "torch not installed")
class TestPolicy(unittest.TestCase):
    def setUp(self):
        self.env = ToyDistrictEnv()
        self.tr = MAPPOTrainer(self.env, MAPPOConfig.for_arm("M1", **FAST))

    def test_no_parameter_sharing_between_actors(self):
        ids = [{id(p) for p in a.parameters()} for a in self.tr.policy.actors]
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                self.assertFalse(ids[i] & ids[j])
        # an update to actor 0 leaves actor 1 untouched
        before = [p.detach().clone() for p in self.tr.policy.actors[1].parameters()]
        with torch.no_grad():
            for p in self.tr.policy.actors[0].parameters():
                p.add_(1.0)
        for b, p in zip(before, self.tr.policy.actors[1].parameters()):
            self.assertTrue(torch.equal(b, p))

    def test_dims(self):
        self.assertEqual(self.tr.actor_dims, [7 + 10] * 6)
        self.assertEqual(self.tr.critic_dims, [self.env.state_dim])
        tr0 = MAPPOTrainer(self.env, MAPPOConfig.for_arm("M0", **FAST))
        self.assertEqual(tr0.actor_dims, [7] * 6)
        ti = MAPPOTrainer(self.env, MAPPOConfig.for_arm("I0", **FAST))
        self.assertEqual(len(ti.policy.critics), 6)

    def test_act_logp_and_determinism(self):
        x = [np.random.randn(17).astype(np.float32) for _ in range(6)]
        a, logp = self.tr.policy.act(x, deterministic=False)
        self.assertEqual(a.shape, (6, 1))
        d = self.tr.policy.actors[2].dist(torch.as_tensor(x[2])[None])
        self.assertAlmostEqual(float(d.log_prob(torch.as_tensor(a[2])[None]).sum()),
                               float(logp[2]), places=4)
        a1, _ = self.tr.policy.act(x, deterministic=True)
        a2, _ = self.tr.policy.act(x, deterministic=True)
        np.testing.assert_array_equal(a1, a2)

    def test_value_shape(self):
        v = self.tr.policy.value([np.zeros(self.env.state_dim, np.float32)])
        self.assertEqual(v.shape, (1,))


@unittest.skipUnless(HAS_TORCH, "torch not installed")
class TestTraining(unittest.TestCase):
    def test_every_arm_trains_and_logs_finite_stats(self):
        for arm in ARMS:
            logs = []
            tr = MAPPOTrainer(ToyDistrictEnv(episode_steps=48),
                              MAPPOConfig.for_arm(arm, seed=0, **FAST), logger=logs.append)
            tr.train()
            self.assertEqual(len(logs), 2, arm)
            for k in ("pi_loss", "v_loss", "entropy", "approx_kl", "steps_per_sec"):
                self.assertTrue(np.isfinite(logs[-1][k]), (arm, k))
            if arm == "M1-RND":
                self.assertIsNotNone(logs[-1]["train_p_mean"])

    def test_same_seed_is_reproducible(self):
        outs = []
        for _ in range(2):
            tr = MAPPOTrainer(ToyDistrictEnv(), MAPPOConfig.for_arm("M1-RND", seed=5, **FAST))
            tr.train()
            outs.append([p.detach().clone() for p in tr.policy.parameters()])
        for a, b in zip(*outs):
            self.assertTrue(torch.allclose(a, b))

    def test_update_changes_actors_and_critic(self):
        tr = MAPPOTrainer(ToyDistrictEnv(), MAPPOConfig.for_arm("M0", **FAST))
        before = [p.detach().clone() for p in tr.policy.parameters()]
        tr.train()
        changed = [not torch.equal(b, p) for b, p in zip(before, tr.policy.parameters())]
        self.assertTrue(all(changed[:2]))     # first actor's layers moved
        self.assertTrue(any(changed[-4:]))    # critic moved

    def test_checkpoint_roundtrip(self):
        env = ToyDistrictEnv()
        tr = MAPPOTrainer(env, MAPPOConfig.for_arm("M1-RND", seed=2, **FAST))
        tr.train()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "ck.pt")
            tr.save(path)
            tr2 = MAPPOTrainer.load(path, ToyDistrictEnv())
        self.assertEqual(tr2.cfg.arm, "M1-RND")
        r1 = tr.evaluate(ToyDistrictEnv(), p=0.5, n_replicates=2, env_seed=3)
        r2 = tr2.evaluate(ToyDistrictEnv(), p=0.5, n_replicates=2, env_seed=3)
        for a, b in zip(r1, r2):
            np.testing.assert_allclose(a["actions"], b["actions"], rtol=1e-6)

    def test_evaluate_force_mask_and_p_levels(self):
        tr = MAPPOTrainer(ToyDistrictEnv(), MAPPOConfig.for_arm("M1-RND", **FAST))
        tr.train()
        for p in (1.0, 0.9, 0.5, 0.0):
            recs = tr.evaluate(ToyDistrictEnv(episode_steps=24), p=p, n_replicates=2)
            self.assertEqual(len(recs), 2)
            rate = recs[0]["delivered"].sum() / (24 * 30)
            self.assertAlmostEqual(rate, p, delta=0.15)
        recs = tr.evaluate(ToyDistrictEnv(episode_steps=24), p=0.0, n_replicates=1,
                           force_mask=True)
        self.assertTrue(recs[0]["force_mask"])


@unittest.skipUnless(HAS_TORCH, "torch not installed")
class TestLearningSignal(unittest.TestCase):
    """The task is solvable only through messages: M1 must beat M0's ceiling.

    Best possible without messages is -1/3 (predict 0 for U(-1, 1)).
    """

    def _train(self, arm):
        env = PeerSoCMatchEnv(n_agents=3, episode_steps=32)
        cfg = MAPPOConfig.for_arm(arm, seed=0, total_steps=60_000, rollout_steps=1024,
                                  ppo_epochs=10, num_minibatches=4, gamma=0.5,
                                  actor_lr=1e-3, critic_lr=1e-3)
        tr = MAPPOTrainer(env, cfg)
        tr.train()
        recs = tr.evaluate(PeerSoCMatchEnv(n_agents=3, episode_steps=200), p=1.0,
                           n_replicates=2)
        return tr, float(np.mean([r["rewards"].mean() for r in recs]))

    def test_messages_are_learned(self):
        tr1, r1 = self._train("M1")
        _, r0 = self._train("M0")
        print(f"\n  learning test: M1 reward {r1:.3f}, M0 reward {r0:.3f} (ceiling -0.333)")
        self.assertLess(r0, -0.30)
        self.assertGreater(r1, -0.25)
        # and the trained M1 policy is fragile when messages drop
        recs = tr1.evaluate(PeerSoCMatchEnv(n_agents=3, episode_steps=200), p=0.0,
                            n_replicates=1)
        self.assertLess(float(recs[0]["rewards"].mean()), r1)


if __name__ == "__main__":
    unittest.main()
