"""MessageEnv (actor/critic inputs for on-policy), env contract, evaluation (numpy only)."""
import unittest

import numpy as np

from marl.comms import DeliverySchedule, PeerSoCChannel, peer_indices
from marl.envs import EnvContractError, PeerSoCMatchEnv, ToyDistrictEnv, check_env_contract
from marl.mappo import (MAPPOConfig, MessageEnv, Preprocessor, RandomPolicy,
                        RunningMeanStd, build_actor_inputs, evaluate, make_message_env)

N_AGENTS = 6
TOY_OBS_DIM = 7        # ToyDistrictEnv: 4 local + 3 shared features
MESSAGE_LEN = 10       # [SoC, mask] for each of 5 peers


def drive(message_env, n_steps, seed=0):
    """Step a MessageEnv with random actions; one record per step.

    Each record holds the inputs the agents acted on, the SoC behind them,
    and what the step returned.
    """
    rng = np.random.default_rng(seed)
    actor_inputs, critic_inputs = message_env.reset()
    records = []
    for _ in range(n_steps):
        record = {"actor_inputs": actor_inputs, "critic_inputs": critic_inputs,
                  "soc": message_env.last_soc.copy()}

        actions = rng.uniform(-1, 1, (message_env.n_agents, 1))
        actor_inputs, critic_inputs, rewards, done, time_limit, info = message_env.step(actions)

        record.update(rewards=rewards, done=done, time_limit=time_limit, info=info)
        records.append(record)
    return records


class TestNormalizers(unittest.TestCase):
    def test_running_moments_match_numpy(self):
        samples = np.random.default_rng(0).normal(3, 2, size=(1000, 3))
        stats = RunningMeanStd((3,), epsilon=1e-12)
        for chunk in np.array_split(samples, 17):
            stats.update(chunk)

        np.testing.assert_allclose(stats.mean, samples.mean(0), rtol=1e-6)
        np.testing.assert_allclose(stats.var, samples.var(0), rtol=1e-6)

    def test_frozen_when_update_false(self):
        preprocessor = Preprocessor([2], 2)
        preprocessor.normalize_local_obs([np.array([5.0, 5.0])], update=True)
        mean_before = preprocessor.local_obs_stats[0].mean.copy()

        preprocessor.normalize_local_obs([np.array([100.0, 100.0])], update=False)
        np.testing.assert_array_equal(preprocessor.local_obs_stats[0].mean, mean_before)

    def test_state_dict_roundtrip(self):
        original = Preprocessor([2, 2], 3)
        original.normalize_local_obs([np.ones(2), 2 * np.ones(2)], update=True)
        restored = Preprocessor([2, 2], 3)
        restored.load_state_dict(original.state_dict())

        obs = [np.ones(2)] * 2
        np.testing.assert_array_equal(original.normalize_local_obs(obs, False)[1],
                                      restored.normalize_local_obs(obs, False)[1])


class BrokenEnv(ToyDistrictEnv):
    """Returns one reward per building, which the contract forbids."""

    def step(self, actions):
        obs, reward, terminated, truncated, info = super().step(actions)
        return obs, [reward] * self.n_agents, terminated, truncated, info


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
        for arm, message_len in (("M0", 0), ("M1", MESSAGE_LEN), ("M1-RND", MESSAGE_LEN),
                                 ("I0", 0)):
            message_env = make_message_env(env, MAPPOConfig.for_arm(arm))
            self.assertEqual(message_env.actor_dims, [TOY_OBS_DIM + message_len] * N_AGENTS, arm)

            # on-policy picks the network head by class name and reads .shape
            self.assertEqual(message_env.observation_space[0].__class__.__name__, "Box")
            self.assertEqual(message_env.action_space[0].shape, (1,))

            if arm == "I0":
                self.assertEqual(message_env.critic_dims, message_env.actor_dims)
            else:
                self.assertEqual(message_env.critic_dims, [env.state_dim] * N_AGENTS)

    def test_needs_channel_for_messages(self):
        with self.assertRaises(ValueError):
            MessageEnv(ToyDistrictEnv(), True)


class TestMessageEnvInputs(unittest.TestCase):
    def test_critic_never_sees_dropped_messages(self):
        """Critic inputs identical at p=1 and p=0; actor message columns differ."""
        def make(p):
            env = ToyDistrictEnv()
            channel = PeerSoCChannel(N_AGENTS, 1.0, seed=0)
            return MessageEnv(env, True, channel, DeliverySchedule("fixed", p=p),
                              Preprocessor(env.obs_dims, env.state_dim), env_seed=1)

        full_delivery, no_delivery = drive(make(1.0), 40), drive(make(0.0), 40)
        for full, none in zip(full_delivery, no_delivery):
            for agent in range(N_AGENTS):
                np.testing.assert_array_equal(full["critic_inputs"][agent],
                                              none["critic_inputs"][agent])
                np.testing.assert_array_equal(none["actor_inputs"][agent][-MESSAGE_LEN:], 0.0)
                np.testing.assert_array_equal(full["actor_inputs"][agent][:-MESSAGE_LEN],
                                              none["actor_inputs"][agent][:-MESSAGE_LEN])

        self.assertFalse(np.array_equal(full_delivery[3]["actor_inputs"][0],
                                        no_delivery[3]["actor_inputs"][0]))

    def test_messages_are_raw_soc_of_the_current_step(self):
        message_env = make_message_env(ToyDistrictEnv(), MAPPOConfig.for_arm("M1"))
        for record in drive(message_env, 30):
            for agent in range(N_AGENTS):
                messages = record["actor_inputs"][agent][-MESSAGE_LEN:]
                np.testing.assert_allclose(messages[0::2],
                                           record["soc"][peer_indices(N_AGENTS, agent)],
                                           rtol=1e-6)
                np.testing.assert_array_equal(messages[1::2], 1.0)

    def test_local_features_are_normalized(self):
        message_env = make_message_env(ToyDistrictEnv(), MAPPOConfig.for_arm("M0"))
        records = drive(message_env, 2000)

        # skip the first steps, while the running statistics are still settling
        agent0_inputs = np.stack([record["actor_inputs"][0] for record in records[500:]])
        self.assertLess(np.abs(agent0_inputs.mean(0)).max(), 0.5)

    def test_shared_reward_replicated_per_agent(self):
        message_env = make_message_env(ToyDistrictEnv(), MAPPOConfig.for_arm("M0"))
        for record in drive(message_env, 10):
            rewards = record["rewards"]
            self.assertEqual(rewards.shape, (N_AGENTS, 1))
            self.assertTrue(np.all(rewards == rewards[0, 0]))

    def test_auto_reset_and_truncation_flags(self):
        message_env = make_message_env(ToyDistrictEnv(episode_steps=10),
                                       MAPPOConfig.for_arm("M0"))
        records = drive(message_env, 35)

        done_at = [step for step, record in enumerate(records) if record["done"]]
        self.assertEqual(done_at, [9, 19, 29])
        self.assertTrue(all(records[step]["time_limit"] for step in done_at))
        self.assertEqual(records[9]["info"]["episode"]["length"], 10)
        self.assertEqual(len(message_env.reset_seeds), 4)


class TestPairingAndSchedule(unittest.TestCase):
    def test_env_reset_seeds_paired_across_arms(self):
        seeds_by_arm = {}
        for arm in ("M0", "M1", "M1-RND", "I0"):
            message_env = make_message_env(ToyDistrictEnv(episode_steps=8),
                                           MAPPOConfig.for_arm(arm, seed=3))
            drive(message_env, 80)
            seeds_by_arm[arm] = message_env.reset_seeds

        self.assertGreater(len(seeds_by_arm["M0"]), 5)
        for arm, seeds in seeds_by_arm.items():
            self.assertEqual(seeds, seeds_by_arm["M0"], arm)

    def test_different_run_seeds_differ(self):
        seed0 = make_message_env(ToyDistrictEnv(episode_steps=8),
                                 MAPPOConfig.for_arm("M0", seed=0))
        seed1 = make_message_env(ToyDistrictEnv(episode_steps=8),
                                 MAPPOConfig.for_arm("M0", seed=1))
        drive(seed0, 40)
        drive(seed1, 40)
        self.assertNotEqual(seed0.reset_seeds, seed1.reset_seeds)

    def test_m1_rnd_draws_p_per_episode(self):
        message_env = make_message_env(ToyDistrictEnv(episode_steps=16),
                                       MAPPOConfig.for_arm("M1-RND", seed=1))
        drive(message_env, 16 * 200)

        self.assertEqual({episode["p"] for episode in message_env.episodes},
                         {0.0, 0.5, 0.75, 1.0})
        for episode in message_env.episodes:
            self.assertAlmostEqual(episode["delivery_rate"], episode["p"], delta=0.12)

    def test_m1_always_full_delivery(self):
        message_env = make_message_env(ToyDistrictEnv(episode_steps=16),
                                       MAPPOConfig.for_arm("M1"))
        drive(message_env, 64)
        self.assertTrue(all(episode["delivery_rate"] == 1.0
                            for episode in message_env.episodes))

    def test_no_mask_arm_holds_mask_at_one(self):
        message_env = make_message_env(ToyDistrictEnv(episode_steps=16),
                                       MAPPOConfig.for_arm("M1-RND-NM", seed=2))
        for record in drive(message_env, 200):
            mask_columns = record["actor_inputs"][0][-MESSAGE_LEN:][1::2]
            np.testing.assert_array_equal(mask_columns, 1.0)


class OraclePolicy:
    """For PeerSoCMatchEnv: copy the right-hand peer's SoC from the message."""

    def __init__(self, n_agents):
        self.n_agents = n_agents

    def act(self, actor_inputs, deterministic=True):
        actions = np.zeros((self.n_agents, 1), np.float32)
        message_len = 2 * (self.n_agents - 1)
        for agent in range(self.n_agents):
            messages = actor_inputs[agent][-message_len:]
            # where the right-hand neighbour sits in this agent's message block
            slot = peer_indices(self.n_agents, agent).index((agent + 1) % self.n_agents)
            peer_soc, mask = messages[2 * slot], messages[2 * slot + 1]
            # target is 2*SoC - 1; without a message the best guess is 0
            actions[agent, 0] = 2 * peer_soc - 1 if mask == 1 else 0.0
        return actions, np.zeros(self.n_agents, np.float32)


class TestEvaluate(unittest.TestCase):
    def test_messages_carry_true_soc_end_to_end(self):
        """Oracle reward ~ 0 at p=1 and ~ -(1-p)/3 otherwise."""
        env = PeerSoCMatchEnv(n_agents=3, episode_steps=400)
        preprocessor = Preprocessor(env.obs_dims, env.state_dim)
        for p in (1.0, 0.5, 0.0):
            records = evaluate(env, OraclePolicy(3), preprocessor, True, p=p, n_replicates=3)
            mean_reward = np.mean([record["rewards"].mean() for record in records])
            # a missed target costs Var(2U - 1) = 1/3 on average
            self.assertAlmostEqual(mean_reward, -(1 - p) / 3, delta=0.03, msg=f"p={p}")

    def test_force_mask_breaks_the_oracle(self):
        env = PeerSoCMatchEnv(n_agents=3, episode_steps=400)
        preprocessor = Preprocessor(env.obs_dims, env.state_dim)
        with_mask = evaluate(env, OraclePolicy(3), preprocessor, True, p=0.5, n_replicates=2)
        forced = evaluate(env, OraclePolicy(3), preprocessor, True, p=0.5, n_replicates=2,
                          force_mask=True)
        self.assertLess(np.mean([record["return"] for record in forced]),
                        np.mean([record["return"] for record in with_mask]))

    def test_replicates_share_outages_but_not_drops(self):
        env = ToyDistrictEnv(episode_steps=24)
        preprocessor = Preprocessor(env.obs_dims, env.state_dim)
        records = evaluate(env, RandomPolicy(N_AGENTS, [1] * N_AGENTS, seed=0), preprocessor,
                           True, p=0.5, n_replicates=3, env_seed=11)

        first = records[0]
        for record in records[1:]:
            np.testing.assert_array_equal(record["soc"][0], first["soc"][0])
            self.assertFalse(np.array_equal(record["delivered"], first["delivered"]))

    def test_eval_does_not_update_normalizers(self):
        env = ToyDistrictEnv(episode_steps=24)
        preprocessor = Preprocessor(env.obs_dims, env.state_dim)
        before = preprocessor.state_dict()

        evaluate(env, RandomPolicy(N_AGENTS, [1] * N_AGENTS), preprocessor, False,
                 n_replicates=1)
        np.testing.assert_array_equal(preprocessor.state_dict()["state"]["mean"],
                                      before["state"]["mean"])

    def test_record_format(self):
        env = ToyDistrictEnv(episode_steps=24)
        preprocessor = Preprocessor(env.obs_dims, env.state_dim)
        record = evaluate(env, RandomPolicy(N_AGENTS, [1] * N_AGENTS), preprocessor, True,
                          p=0.9, n_replicates=1,
                          on_episode_end=lambda e: {"steps": e.time_step})[0]

        self.assertEqual(record["actions"].shape, (24, N_AGENTS))
        self.assertEqual(record["soc"].shape, (24, N_AGENTS))
        self.assertEqual(record["delivered"].shape, (24, N_AGENTS, N_AGENTS))
        self.assertEqual(record["rewards"].shape, (24,))
        self.assertTrue(np.all(np.abs(record["actions"]) <= 1))
        self.assertIsNotNone(record["kpis"])

        no_messages = evaluate(env, RandomPolicy(N_AGENTS, [1] * N_AGENTS), preprocessor,
                               False, n_replicates=1)[0]
        self.assertIsNone(no_messages["delivered"])

    def test_battery_energy_recorded_and_differs_from_actions(self):
        """Synchrony needs what the batteries did, not what was commanded."""
        class AlwaysCharge:
            def act(self, actor_inputs, deterministic=True):
                return np.ones((N_AGENTS, 1), np.float32), np.zeros(N_AGENTS, np.float32)

        env = ToyDistrictEnv(episode_steps=24)
        record = evaluate(env, AlwaysCharge(), Preprocessor(env.obs_dims, env.state_dim),
                          False, n_replicates=1)[0]
        self.assertEqual(record["battery_energy"].shape, (24, N_AGENTS))
        np.testing.assert_array_equal(record["actions"], 1.0)
        self.assertTrue(np.all(record["battery_energy"][-1] == 0))   # batteries full by the end
        self.assertGreater(record["battery_energy"][0].sum(), 0)

        # an env without battery_energy in its info gives None
        match_env = PeerSoCMatchEnv(n_agents=3, episode_steps=10)
        match_record = evaluate(match_env, OraclePolicy(3),
                                Preprocessor(match_env.obs_dims, match_env.state_dim),
                                True, n_replicates=1)[0]
        self.assertIsNone(match_record["battery_energy"])

    def test_build_actor_inputs(self):
        messages = np.array([[5.0, 1.0], [6.0, 0.0]])
        inputs = build_actor_inputs([np.ones(2), np.zeros(2)], messages)
        np.testing.assert_array_equal(inputs[0], [1, 1, 5, 1])
        np.testing.assert_array_equal(build_actor_inputs([np.ones(2)], None)[0], [1, 1])


if __name__ == "__main__":
    unittest.main()
