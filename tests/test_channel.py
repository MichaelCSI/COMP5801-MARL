"""Message channel, mask and delivery schedule (numpy only)."""
import unittest

import numpy as np

from marl.comms import (EVAL_P_LEVELS, RND_P_CHOICES, DeliverySchedule, PeerSoCChannel,
                        message_dim, peer_indices)

N_AGENTS = 6
N_LINKS = N_AGENTS * (N_AGENTS - 1)   # 30 directed links


class TestChannelShapes(unittest.TestCase):
    def test_dim_and_peer_order(self):
        self.assertEqual(message_dim(N_AGENTS), 10)
        self.assertEqual(peer_indices(N_AGENTS, 0), [1, 2, 3, 4, 5])
        self.assertEqual(peer_indices(N_AGENTS, 3), [0, 1, 2, 4, 5])

        channel = PeerSoCChannel(N_AGENTS, 1.0, seed=0)
        messages = channel.transmit(np.linspace(0.1, 0.6, N_AGENTS))
        self.assertEqual(messages.shape, (N_AGENTS, 10))
        self.assertEqual(messages.dtype, np.float32)

    def test_full_delivery_carries_every_peer_soc(self):
        soc = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], np.float32)
        messages = PeerSoCChannel(N_AGENTS, 1.0, seed=0).transmit(soc)
        for receiver in range(N_AGENTS):
            np.testing.assert_allclose(messages[receiver, 0::2],
                                       soc[peer_indices(N_AGENTS, receiver)])
            np.testing.assert_array_equal(messages[receiver, 1::2], 1.0)

    def test_zero_delivery_is_all_zero(self):
        messages = PeerSoCChannel(N_AGENTS, 0.0, seed=0).transmit(np.full(N_AGENTS, 0.7))
        np.testing.assert_array_equal(messages, 0.0)

    def test_no_self_message(self):
        channel = PeerSoCChannel(N_AGENTS, 1.0, seed=0)
        channel.transmit(np.full(N_AGENTS, 0.5))
        self.assertFalse(channel.last_delivered.diagonal().any())
        self.assertEqual(int(channel.last_delivered.sum()), N_LINKS)

    def test_soc_zero_is_distinguishable_from_drop(self):
        """SoC = 0 delivered -> [0, 1]; dropped -> [0, 0]."""
        soc = np.zeros(N_AGENTS, np.float32)
        delivered = np.zeros((N_AGENTS, N_AGENTS), bool)
        delivered[0, 1] = True  # 1 -> 0 delivered, everything else dropped

        messages = PeerSoCChannel(N_AGENTS, 0.5, seed=0).encode(soc, delivered)
        np.testing.assert_array_equal(messages[0, 0:2], [0.0, 1.0])
        np.testing.assert_array_equal(messages[0, 2:4], [0.0, 0.0])

    def test_wrong_soc_length_raises(self):
        with self.assertRaises(ValueError):
            PeerSoCChannel(N_AGENTS, 1.0).transmit(np.zeros(N_AGENTS - 1))

    def test_bad_p_raises(self):
        for p in (-0.1, 1.1):
            with self.assertRaises(ValueError):
                PeerSoCChannel(N_AGENTS, p)


class TestChannelStatistics(unittest.TestCase):
    def test_empirical_rate_matches_p(self):
        n_steps = 4000
        for p in (0.5, 0.75, 0.9):
            channel = PeerSoCChannel(N_AGENTS, p, seed=1)
            n_delivered = sum(channel.sample_delivery().sum() for _ in range(n_steps))
            self.assertAlmostEqual(n_delivered / (n_steps * N_LINKS), p, delta=0.01)

    def test_links_are_independent(self):
        channel = PeerSoCChannel(N_AGENTS, 0.5, seed=2)
        deliveries = np.stack([channel.sample_delivery() for _ in range(6000)]).astype(float)
        off_diagonal = ~np.eye(N_AGENTS, dtype=bool)
        links = deliveries[:, off_diagonal]   # (steps, 30): one column per link

        # no correlation between different links...
        link_corr = np.corrcoef(links.T)
        np.fill_diagonal(link_corr, 0)
        self.assertLess(np.abs(link_corr).max(), 0.06)

        # ...or between consecutive steps
        lag_corr = np.corrcoef(links[:-1].ravel(), links[1:].ravel())[0, 1]
        self.assertLess(abs(lag_corr), 0.03)

    def test_values_match_mask(self):
        channel = PeerSoCChannel(N_AGENTS, 0.5, seed=3)
        soc = np.array([0.3, 0.5, 0.7, 0.9, 0.4, 0.6], np.float32)
        for _ in range(200):
            messages = channel.transmit(soc)
            soc_values, mask = messages[:, 0::2], messages[:, 1::2]
            for receiver in range(N_AGENTS):
                peers = peer_indices(N_AGENTS, receiver)
                np.testing.assert_allclose(soc_values[receiver], soc[peers] * mask[receiver])
                # mask matches the recorded delivery matrix
                np.testing.assert_array_equal(mask[receiver] == 1,
                                              channel.last_delivered[receiver, peers])


class TestChannelSeeding(unittest.TestCase):
    def test_same_seed_same_drops(self):
        first = PeerSoCChannel(N_AGENTS, 0.5, seed=7)
        second = PeerSoCChannel(N_AGENTS, 0.5, seed=7)
        for _ in range(50):
            np.testing.assert_array_equal(first.sample_delivery(), second.sample_delivery())

    def test_different_seed_different_drops(self):
        first = PeerSoCChannel(N_AGENTS, 0.5, seed=7)
        second = PeerSoCChannel(N_AGENTS, 0.5, seed=8)
        self.assertFalse(all((first.sample_delivery() == second.sample_delivery()).all()
                             for _ in range(20)))

    def test_lower_p_drops_superset(self):
        """Same seed: links delivered at p=0.5 are also delivered at p=0.9."""
        low_p = PeerSoCChannel(N_AGENTS, 0.5, seed=4)
        high_p = PeerSoCChannel(N_AGENTS, 0.9, seed=4)
        for _ in range(100):
            delivered_low, delivered_high = low_p.sample_delivery(), high_p.sample_delivery()
            self.assertTrue(np.all(delivered_high[delivered_low]))

    def test_channel_does_not_touch_global_numpy_rng(self):
        np.random.seed(123)
        expected = np.random.rand()

        np.random.seed(123)
        PeerSoCChannel(N_AGENTS, 0.5, seed=0).transmit(np.full(N_AGENTS, 0.5))
        self.assertEqual(np.random.rand(), expected)


class TestForceMask(unittest.TestCase):
    def test_force_mask_makes_drop_look_like_soc_zero(self):
        channel = PeerSoCChannel(N_AGENTS, 0.0, seed=0, force_mask=True)
        messages = channel.transmit(np.full(N_AGENTS, 0.8))
        np.testing.assert_array_equal(messages[:, 0::2], 0.0)
        np.testing.assert_array_equal(messages[:, 1::2], 1.0)

    def test_force_mask_keeps_delivered_values(self):
        soc = np.linspace(0.2, 0.7, N_AGENTS).astype(np.float32)
        normal = PeerSoCChannel(N_AGENTS, 0.5, seed=5).transmit(soc)
        forced = PeerSoCChannel(N_AGENTS, 0.5, seed=5, force_mask=True).transmit(soc)
        np.testing.assert_array_equal(normal[:, 0::2], forced[:, 0::2])


class TestSchedule(unittest.TestCase):
    def test_constants_match_proposal(self):
        self.assertEqual(RND_P_CHOICES, (0.0, 0.5, 0.75, 1.0))
        self.assertEqual(EVAL_P_LEVELS, (1.0, 0.9, 0.5, 0.0))
        self.assertNotIn(0.9, RND_P_CHOICES)  # p=0.9 never seen in training

    def test_fixed(self):
        schedule = DeliverySchedule("fixed", p=1.0)
        self.assertEqual({schedule.sample() for _ in range(20)}, {1.0})

    def test_random_covers_choices_uniformly(self):
        schedule = DeliverySchedule("random", seed=0)
        draws = np.array([schedule.sample() for _ in range(8000)])
        self.assertEqual(set(draws), set(RND_P_CHOICES))
        for choice in RND_P_CHOICES:
            self.assertAlmostEqual(np.mean(draws == choice), 0.25, delta=0.02)

    def test_random_is_seeded(self):
        first = DeliverySchedule("random", seed=3)
        second = DeliverySchedule("random", seed=3)
        self.assertEqual([first.sample() for _ in range(30)],
                         [second.sample() for _ in range(30)])

    def test_bad_mode(self):
        with self.assertRaises(ValueError):
            DeliverySchedule("sometimes")


if __name__ == "__main__":
    unittest.main()
