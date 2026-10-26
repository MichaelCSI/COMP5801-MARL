"""Arms -> on-policy arguments, and integrity of the vendored on-policy code.

Runs without torch: on-policy's config.py only needs argparse.
"""
import hashlib
import os
import unittest

from marl.mappo import ARMS, FIXED_ONPOLICY_ARGS, MAPPOConfig
from marl.mappo.onpolicy_compat import ONPOLICY_COMMIT, VENDOR_ROOT, get_onpolicy_parser


class TestArms(unittest.TestCase):
    def test_arms_differ_only_in_message_fields(self):
        baseline = MAPPOConfig.for_arm("M0").to_dict()
        arm_defining = {"arm", "use_messages", "train_schedule", "train_p", "critic",
                        "force_mask_train"}
        for arm in ARMS:
            fields = MAPPOConfig.for_arm(arm).to_dict()
            differing = {name for name in fields if fields[name] != baseline[name]}
            self.assertTrue(differing <= arm_defining, (arm, differing))

    def test_onpolicy_args_identical_across_mappo_arms(self):
        baseline = vars(MAPPOConfig.for_arm("M0").to_onpolicy_args())
        for arm in ("M1", "M1-RND", "M1-RND-NM"):
            args = vars(MAPPOConfig.for_arm(arm).to_onpolicy_args())
            differing = {name for name in baseline if baseline[name] != args[name]}
            self.assertEqual(differing, {"experiment_name"}, arm)

    def test_bad_configs(self):
        with self.assertRaises(ValueError):
            MAPPOConfig.for_arm("M2")
        with self.assertRaises(ValueError):
            MAPPOConfig(use_messages=True, train_schedule="none").validate()
        with self.assertRaises(ValueError):   # design-fixed args cannot be overridden
            MAPPOConfig(onpolicy_overrides={"share_policy": True}).to_onpolicy_args()
        with self.assertRaises(ValueError):
            MAPPOConfig(onpolicy_overrides={"not_an_arg": 1}).to_onpolicy_args()


class TestOnPolicyArgs(unittest.TestCase):
    def test_design_choices(self):
        args = MAPPOConfig.for_arm("M1-RND", seed=4, rollout_steps=500,
                                   total_steps=10_000).to_onpolicy_args()
        self.assertFalse(args.share_policy)                 # one actor per building
        self.assertTrue(args.use_centralized_V)             # MAPPO critic on global state
        self.assertFalse(args.use_recurrent_policy)
        self.assertFalse(args.use_naive_recurrent_policy)
        self.assertTrue(args.use_proper_time_limits)
        self.assertFalse(args.use_feature_normalization)
        self.assertFalse(args.use_wandb)
        self.assertEqual(args.n_rollout_threads, 1)
        self.assertEqual(args.episode_length, 500)
        self.assertEqual(args.num_env_steps, 10_000)
        self.assertEqual(args.seed, 4)
        self.assertEqual(args.algorithm_name, "mappo")

        for name, value in FIXED_ONPOLICY_ARGS.items():
            self.assertEqual(getattr(args, name), value, name)

    def test_ippo_uses_local_critic(self):
        self.assertFalse(MAPPOConfig.for_arm("I0").to_onpolicy_args().use_centralized_V)

    def test_other_values_are_onpolicy_defaults(self):
        defaults = vars(get_onpolicy_parser().parse_args([]))
        args = vars(MAPPOConfig.for_arm("M0").to_onpolicy_args())
        set_by_us = set(FIXED_ONPOLICY_ARGS) | {"use_centralized_V", "seed", "episode_length",
                                                "num_env_steps", "cuda", "experiment_name"}
        for name in defaults:
            if name not in set_by_us:
                self.assertEqual(args[name], defaults[name], name)

        # the reference PPO settings we inherit
        self.assertEqual(args["clip_param"], 0.2)
        self.assertEqual(args["gamma"], 0.99)
        self.assertEqual(args["gae_lambda"], 0.95)
        self.assertTrue(args["use_valuenorm"])

    def test_overrides_apply(self):
        args = MAPPOConfig(onpolicy_overrides={"ppo_epoch": 5, "lr": 1e-4}).to_onpolicy_args()
        self.assertEqual((args.ppo_epoch, args.lr), (5, 1e-4))


class TestVendoredOnPolicy(unittest.TestCase):
    """The vendored files must match upstream byte for byte (see VENDORED.md)."""

    def test_manifest_hashes_match(self):
        with open(os.path.join(VENDOR_ROOT, "MANIFEST.sha256")) as manifest:
            entries = [line.split() for line in manifest if line.strip()]
        self.assertGreaterEqual(len(entries), 17)

        for expected_hash, relative_path in entries:
            with open(os.path.join(VENDOR_ROOT, relative_path), "rb") as vendored_file:
                actual_hash = hashlib.sha256(vendored_file.read()).hexdigest()
            self.assertEqual(actual_hash, expected_hash,
                             f"{relative_path} differs from on-policy@{ONPOLICY_COMMIT[:7]}")

    def test_license_and_provenance_present(self):
        for name in ("LICENSE", "VENDORED.md"):
            self.assertTrue(os.path.exists(os.path.join(VENDOR_ROOT, name)), name)
        with open(os.path.join(VENDOR_ROOT, "VENDORED.md")) as provenance:
            self.assertIn(ONPOLICY_COMMIT, provenance.read())


if __name__ == "__main__":
    unittest.main()
