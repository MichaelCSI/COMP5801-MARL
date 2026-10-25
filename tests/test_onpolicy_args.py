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
        base = MAPPOConfig.for_arm("M0").to_dict()
        allowed = {"arm", "use_messages", "train_schedule", "train_p", "critic",
                   "force_mask_train"}
        for arm in ARMS:
            d = MAPPOConfig.for_arm(arm).to_dict()
            diff = {k for k in d if d[k] != base[k]}
            self.assertTrue(diff <= allowed, (arm, diff))

    def test_onpolicy_args_identical_across_mappo_arms(self):
        ref = vars(MAPPOConfig.for_arm("M0").to_onpolicy_args())
        for arm in ("M1", "M1-RND", "M1-RND-NM"):
            a = vars(MAPPOConfig.for_arm(arm).to_onpolicy_args())
            diff = {k for k in ref if ref[k] != a[k]}
            self.assertEqual(diff, {"experiment_name"}, arm)

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
        a = MAPPOConfig.for_arm("M1-RND", seed=4, rollout_steps=500,
                                total_steps=10_000).to_onpolicy_args()
        self.assertFalse(a.share_policy)                 # one actor per building
        self.assertTrue(a.use_centralized_V)             # MAPPO critic on global state
        self.assertFalse(a.use_recurrent_policy)
        self.assertFalse(a.use_naive_recurrent_policy)
        self.assertTrue(a.use_proper_time_limits)
        self.assertFalse(a.use_feature_normalization)
        self.assertFalse(a.use_wandb)
        self.assertEqual(a.n_rollout_threads, 1)
        self.assertEqual(a.episode_length, 500)
        self.assertEqual(a.num_env_steps, 10_000)
        self.assertEqual(a.seed, 4)
        self.assertEqual(a.algorithm_name, "mappo")
        for k, v in FIXED_ONPOLICY_ARGS.items():
            self.assertEqual(getattr(a, k), v, k)

    def test_ippo_uses_local_critic(self):
        self.assertFalse(MAPPOConfig.for_arm("I0").to_onpolicy_args().use_centralized_V)

    def test_other_values_are_onpolicy_defaults(self):
        defaults = vars(get_onpolicy_parser().parse_args([]))
        a = vars(MAPPOConfig.for_arm("M0").to_onpolicy_args())
        ours = set(FIXED_ONPOLICY_ARGS) | {"use_centralized_V", "seed", "episode_length",
                                            "num_env_steps", "cuda", "experiment_name"}
        for k in defaults:
            if k not in ours:
                self.assertEqual(a[k], defaults[k], k)
        # the reference PPO settings we inherit
        self.assertEqual(a["clip_param"], 0.2)
        self.assertEqual(a["gamma"], 0.99)
        self.assertEqual(a["gae_lambda"], 0.95)
        self.assertTrue(a["use_valuenorm"])

    def test_overrides_apply(self):
        a = MAPPOConfig(onpolicy_overrides={"ppo_epoch": 5, "lr": 1e-4}).to_onpolicy_args()
        self.assertEqual((a.ppo_epoch, a.lr), (5, 1e-4))


class TestVendoredOnPolicy(unittest.TestCase):
    """The vendored files must match upstream byte for byte (see VENDORED.md)."""

    def test_manifest_hashes_match(self):
        manifest = os.path.join(VENDOR_ROOT, "MANIFEST.sha256")
        with open(manifest) as f:
            lines = [ln.split() for ln in f if ln.strip()]
        self.assertGreaterEqual(len(lines), 17)
        for digest, rel in lines:
            with open(os.path.join(VENDOR_ROOT, rel), "rb") as fh:
                got = hashlib.sha256(fh.read()).hexdigest()
            self.assertEqual(got, digest, f"{rel} differs from on-policy@{ONPOLICY_COMMIT[:7]}")

    def test_license_and_provenance_present(self):
        for name in ("LICENSE", "VENDORED.md"):
            self.assertTrue(os.path.exists(os.path.join(VENDOR_ROOT, name)), name)
        with open(os.path.join(VENDOR_ROOT, "VENDORED.md")) as f:
            self.assertIn(ONPOLICY_COMMIT, f.read())


if __name__ == "__main__":
    unittest.main()
