"""Message channel, mask and delivery schedule (numpy only)."""
import unittest

import numpy as np

from marl.comms import (EVAL_P_LEVELS, RND_P_CHOICES, DeliverySchedule, PeerSoCChannel,
                        message_dim, peer_indices)

N = 6


class TestChannelShapes(unittest.TestCase):
    def test_dim_and_peer_order(self):
        self.assertEqual(message_dim(N), 10)
        self.assertEqual(peer_indices(N, 0), [1, 2, 3, 4, 5])
        self.assertEqual(peer_indices(N, 3), [0, 1, 2, 4, 5])
        ch = PeerSoCChannel(N, 1.0, seed=0)
        m = ch.transmit(np.linspace(0.1, 0.6, N))
        self.assertEqual(m.shape, (N, 10))
        self.assertEqual(m.dtype, np.float32)

    def test_full_delivery_carries_every_peer_soc(self):
        soc = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], np.float32)
        m = PeerSoCChannel(N, 1.0, seed=0).transmit(soc)
        for i in range(N):
            np.testing.assert_allclose(m[i, 0::2], soc[peer_indices(N, i)])
            np.testing.assert_array_equal(m[i, 1::2], 1.0)

    def test_zero_delivery_is_all_zero(self):
        m = PeerSoCChannel(N, 0.0, seed=0).transmit(np.full(N, 0.7))
        np.testing.assert_array_equal(m, 0.0)

    def test_no_self_message(self):
        ch = PeerSoCChannel(N, 1.0, seed=0)
        ch.transmit(np.full(N, 0.5))
        self.assertFalse(ch.last_delivered.diagonal().any())
        self.assertEqual(int(ch.last_delivered.sum()), N * (N - 1))  # 30 links

    def test_soc_zero_is_distinguishable_from_drop(self):
        """SoC = 0 delivered -> [0, 1]; dropped -> [0, 0]."""
        soc = np.zeros(N, np.float32)
        delivered = np.zeros((N, N), bool)
        delivered[0, 1] = True  # 1 -> 0 delivered, everything else dropped
        m = PeerSoCChannel(N, 0.5, seed=0).encode(soc, delivered)
        np.testing.assert_array_equal(m[0, 0:2], [0.0, 1.0])
        np.testing.assert_array_equal(m[0, 2:4], [0.0, 0.0])

    def test_wrong_soc_length_raises(self):
        with self.assertRaises(ValueError):
            PeerSoCChannel(N, 1.0).transmit(np.zeros(N - 1))

    def test_bad_p_raises(self):
        for p in (-0.1, 1.1):
            with self.assertRaises(ValueError):
                PeerSoCChannel(N, p)


class TestChannelStatistics(unittest.TestCase):
    def test_empirical_rate_matches_p(self):
        for p in (0.5, 0.75, 0.9):
            ch = PeerSoCChannel(N, p, seed=1)
            hits = sum(ch.sample_delivery().sum() for _ in range(4000))
            rate = hits / (4000 * N * (N - 1))
            self.assertAlmostEqual(rate, p, delta=0.01)

    def test_links_are_independent(self):
        ch = PeerSoCChannel(N, 0.5, seed=2)
        D = np.stack([ch.sample_delivery() for _ in range(6000)]).astype(float)
        off = ~np.eye(N, dtype=bool)
        links = D[:, off]                       # (T, 30)
        c = np.corrcoef(links.T)
        np.fill_diagonal(c, 0)
        self.assertLess(np.abs(c).max(), 0.06)
        # also independent across time
        lag = np.corrcoef(links[:-1].ravel(), links[1:].ravel())[0, 1]
        self.assertLess(abs(lag), 0.03)

    def test_values_match_mask(self):
        ch = PeerSoCChannel(N, 0.5, seed=3)
        soc = np.array([0.3, 0.5, 0.7, 0.9, 0.4, 0.6], np.float32)
        for _ in range(200):
            m = ch.transmit(soc)
            vals, mask = m[:, 0::2], m[:, 1::2]
            for i in range(N):
                exp = soc[peer_indices(N, i)] * mask[i]
                np.testing.assert_allclose(vals[i], exp)
            # mask matches the recorded delivery matrix
            for i in range(N):
                np.testing.assert_array_equal(mask[i] == 1,
                                              ch.last_delivered[i, peer_indices(N, i)])


class TestChannelSeeding(unittest.TestCase):
    def test_same_seed_same_drops(self):
        a, b = PeerSoCChannel(N, 0.5, seed=7), PeerSoCChannel(N, 0.5, seed=7)
        for _ in range(50):
            np.testing.assert_array_equal(a.sample_delivery(), b.sample_delivery())

    def test_different_seed_different_drops(self):
        a, b = PeerSoCChannel(N, 0.5, seed=7), PeerSoCChannel(N, 0.5, seed=8)
        self.assertFalse(all((a.sample_delivery() == b.sample_delivery()).all()
                             for _ in range(20)))

    def test_lower_p_drops_superset(self):
        """Same seed: links delivered at p=0.5 are also delivered at p=0.9."""
        lo, hi = PeerSoCChannel(N, 0.5, seed=4), PeerSoCChannel(N, 0.9, seed=4)
        for _ in range(100):
            dl, dh = lo.sample_delivery(), hi.sample_delivery()
            self.assertTrue(np.all(dh[dl]))

    def test_channel_does_not_touch_global_numpy_rng(self):
        np.random.seed(123)
        before = np.random.rand()
        np.random.seed(123)
        PeerSoCChannel(N, 0.5, seed=0).transmit(np.full(N, 0.5))
        self.assertEqual(np.random.rand(), before)


class TestForceMask(unittest.TestCase):
    def test_force_mask_makes_drop_look_like_soc_zero(self):
        ch = PeerSoCChannel(N, 0.0, seed=0, force_mask=True)
        m = ch.transmit(np.full(N, 0.8))
        np.testing.assert_array_equal(m[:, 0::2], 0.0)
        np.testing.assert_array_equal(m[:, 1::2], 1.0)

    def test_force_mask_keeps_delivered_values(self):
        soc = np.linspace(0.2, 0.7, N).astype(np.float32)
        a = PeerSoCChannel(N, 0.5, seed=5).transmit(soc)
        b = PeerSoCChannel(N, 0.5, seed=5, force_mask=True).transmit(soc)
        np.testing.assert_array_equal(a[:, 0::2], b[:, 0::2])


class TestSchedule(unittest.TestCase):
    def test_constants_match_proposal(self):
        self.assertEqual(RND_P_CHOICES, (0.0, 0.5, 0.75, 1.0))
        self.assertEqual(EVAL_P_LEVELS, (1.0, 0.9, 0.5, 0.0))
        self.assertNotIn(0.9, RND_P_CHOICES)  # p=0.9 never seen in training

    def test_fixed(self):
        s = DeliverySchedule("fixed", p=1.0)
        self.assertEqual({s.sample() for _ in range(20)}, {1.0})

    def test_random_covers_choices_uniformly(self):
        s = DeliverySchedule("random", seed=0)
        draws = np.array([s.sample() for _ in range(8000)])
        self.assertEqual(set(draws), set(RND_P_CHOICES))
        for c in RND_P_CHOICES:
            self.assertAlmostEqual(np.mean(draws == c), 0.25, delta=0.02)

    def test_random_is_seeded(self):
        a, b = DeliverySchedule("random", seed=3), DeliverySchedule("random", seed=3)
        self.assertEqual([a.sample() for _ in range(30)], [b.sample() for _ in range(30)])

    def test_bad_mode(self):
        with self.assertRaises(ValueError):
            DeliverySchedule("sometimes")


if __name__ == "__main__":
    unittest.main()
