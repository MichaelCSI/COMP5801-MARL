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

N_AGENTS = 6
# two quick updates of 200 steps each
FAST = dict(total_steps=400, rollout_steps=200, onpolicy_overrides={"ppo_epoch": 2})


@unittest.skipUnless(HAS_TORCH, "torch not installed")
class TestStructure(unittest.TestCase):
    def test_uses_onpolicy_classes(self):
        runner = OnPolicyRunner(ToyDistrictEnv(), MAPPOConfig.for_arm("M1", **FAST))
        self.assertEqual(type(runner.policies[0]).__module__,
                         "onpolicy.algorithms.r_mappo.algorithm.rMAPPOPolicy")
        self.assertEqual(type(runner.trainers[0]).__name__, "R_MAPPO")
        self.assertEqual(type(runner.buffers[0]).__name__, "SeparatedReplayBuffer")

    def test_one_actor_per_building_no_sharing(self):
        runner = OnPolicyRunner(ToyDistrictEnv(), MAPPOConfig.for_arm("M1", **FAST))
        self.assertEqual(len(runner.policies), N_AGENTS)

        param_ids = [{id(param) for param in policy.actor.parameters()}
                     for policy in runner.policies]
        for i in range(N_AGENTS):
            for j in range(i + 1, N_AGENTS):
                self.assertFalse(param_ids[i] & param_ids[j])

    def test_input_dims(self):
        env = ToyDistrictEnv()
        m1 = OnPolicyRunner(env, MAPPOConfig.for_arm("M1", **FAST))
        self.assertEqual(m1.buffers[0].obs.shape[-1], 17)          # 7 obs + 10 message
        self.assertEqual(m1.buffers[0].share_obs.shape[-1], env.state_dim)

        m0 = OnPolicyRunner(env, MAPPOConfig.for_arm("M0", **FAST))
        self.assertEqual(m0.buffers[0].obs.shape[-1], 7)

        ippo = OnPolicyRunner(env, MAPPOConfig.for_arm("I0", **FAST))
        self.assertEqual(ippo.buffers[0].share_obs.shape[-1], 7)   # IPPO: local critic


@unittest.skipUnless(HAS_TORCH, "torch not installed")
class TestTraining(unittest.TestCase):
    def test_every_arm_trains_and_logs_finite_stats(self):
        for arm in ARMS:
            logs = []
            runner = OnPolicyRunner(ToyDistrictEnv(episode_steps=48),
                                    MAPPOConfig.for_arm(arm, seed=0, **FAST),
                                    logger=logs.append)
            runner.run()

            self.assertEqual(len(logs), 2, arm)
            for name in ("value_loss", "policy_loss", "dist_entropy", "ratio", "steps_per_sec"):
                self.assertTrue(np.isfinite(logs[-1][name]), (arm, name))
            if arm.startswith("M1"):
                self.assertIsNotNone(logs[-1]["train_p_mean"])

    def test_same_seed_is_reproducible(self):
        actor_weights = []
        for _ in range(2):
            runner = OnPolicyRunner(ToyDistrictEnv(),
                                    MAPPOConfig.for_arm("M1-RND", seed=5, **FAST))
            runner.run()
            actor_weights.append([param.detach().clone() for policy in runner.policies
                                  for param in policy.actor.parameters()])

        for first, second in zip(*actor_weights):
            self.assertTrue(torch.allclose(first, second))

    def test_update_changes_actor_and_critic(self):
        runner = OnPolicyRunner(ToyDistrictEnv(), MAPPOConfig.for_arm("M0", **FAST))
        policy = runner.policies[0]
        actor_before = [param.detach().clone() for param in policy.actor.parameters()]
        critic_before = [param.detach().clone() for param in policy.critic.parameters()]

        runner.run()
        self.assertTrue(any(not torch.equal(before, after)
                            for before, after in zip(actor_before, policy.actor.parameters())))
        self.assertTrue(any(not torch.equal(before, after)
                            for before, after in zip(critic_before, policy.critic.parameters())))

    def test_truncation_sets_bad_masks(self):
        runner = OnPolicyRunner(ToyDistrictEnv(episode_steps=50),
                                MAPPOConfig.for_arm("M0", total_steps=200, rollout_steps=200,
                                                    onpolicy_overrides={"ppo_epoch": 1}))
        # one rollout by hand, without the update
        runner.warmup()
        for step in range(200):
            outputs = runner.collect(step)
            actor_inputs, critic_inputs, rewards, done, time_limit, _ = runner.message_env.step(
                [action[0] for action in outputs.actions])
            runner.insert(actor_inputs, critic_inputs, rewards, done, time_limit, outputs)

        # masks[t + 1] belongs to step t, hence the [1:]
        buffer = runner.buffers[0]
        episode_ends = np.where(buffer.masks[1:, 0, 0] == 0)[0]
        np.testing.assert_array_equal(episode_ends, [49, 99, 149, 199])
        np.testing.assert_array_equal(np.where(buffer.bad_masks[1:, 0, 0] == 0)[0], episode_ends)

    def test_checkpoint_roundtrip(self):
        runner = OnPolicyRunner(ToyDistrictEnv(), MAPPOConfig.for_arm("M1-RND", seed=2, **FAST))
        runner.run()
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = os.path.join(tmp_dir, "checkpoint.pt")
            runner.save(path)
            restored = OnPolicyRunner.load(path, ToyDistrictEnv())

        self.assertEqual(restored.config.arm, "M1-RND")
        self.assertTrue(torch.allclose(runner.trainers[0].value_normalizer.running_mean,
                                       restored.trainers[0].value_normalizer.running_mean))

        original_eval = runner.evaluate(ToyDistrictEnv(), p=0.5, n_replicates=2, env_seed=3)
        restored_eval = restored.evaluate(ToyDistrictEnv(), p=0.5, n_replicates=2, env_seed=3)
        for original, reloaded in zip(original_eval, restored_eval):
            np.testing.assert_allclose(original["actions"], reloaded["actions"], rtol=1e-6)

    def test_evaluate_p_levels_and_mask_check(self):
        runner = OnPolicyRunner(ToyDistrictEnv(), MAPPOConfig.for_arm("M1-RND", **FAST))
        runner.run()
        n_links = N_AGENTS * (N_AGENTS - 1)

        for p in (1.0, 0.9, 0.5, 0.0):
            records = runner.evaluate(ToyDistrictEnv(episode_steps=24), p=p, n_replicates=2)
            delivery_rate = records[0]["delivered"].sum() / (24 * n_links)
            self.assertAlmostEqual(delivery_rate, p, delta=0.15)

        records = runner.evaluate(ToyDistrictEnv(episode_steps=24), p=0.0, n_replicates=1,
                                  force_mask=True)
        self.assertTrue(records[0]["force_mask"])

    def test_deterministic_eval_is_repeatable(self):
        runner = OnPolicyRunner(ToyDistrictEnv(), MAPPOConfig.for_arm("M1", **FAST))
        first = runner.evaluate(ToyDistrictEnv(episode_steps=24), p=1.0, n_replicates=1)[0]
        second = runner.evaluate(ToyDistrictEnv(episode_steps=24), p=1.0, n_replicates=1)[0]
        np.testing.assert_array_equal(first["actions"], second["actions"])


@unittest.skipUnless(HAS_TORCH, "torch not installed")
class TestLearningSignal(unittest.TestCase):
    """On a task solvable only through messages, M1 must beat M0's ceiling (-1/3)."""

    def _train(self, arm):
        config = MAPPOConfig.for_arm(arm, seed=0, total_steps=60_000, rollout_steps=1000,
                                     onpolicy_overrides={"gamma": 0.5})
        runner = OnPolicyRunner(PeerSoCMatchEnv(n_agents=3, episode_steps=32), config)
        runner.run()

        records = runner.evaluate(PeerSoCMatchEnv(n_agents=3, episode_steps=200), p=1.0,
                                  n_replicates=2)
        mean_reward = float(np.mean([record["rewards"].mean() for record in records]))
        return runner, mean_reward

    def test_messages_are_learned(self):
        m1_runner, m1_reward = self._train("M1")
        _, m0_reward = self._train("M0")
        print(f"\n  learning test: M1 reward {m1_reward:.3f}, M0 reward {m0_reward:.3f} "
              f"(M0 ceiling -0.333)")
        self.assertLess(m0_reward, -0.30)
        self.assertGreater(m1_reward, -0.25)

        # and the trained M1 policy is fragile when messages drop
        no_delivery = m1_runner.evaluate(PeerSoCMatchEnv(n_agents=3, episode_steps=200), p=0.0,
                                         n_replicates=1)
        self.assertLess(float(no_delivery[0]["rewards"].mean()), m1_reward)


if __name__ == "__main__":
    unittest.main()
