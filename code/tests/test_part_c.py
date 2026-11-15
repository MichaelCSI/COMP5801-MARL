
"""Fast, isolated Part C unit tests; no CityLearn or checkpoint required.

From code/ : python -m unittest tests.test_part_c -v
"""
import unittest

import numpy as np
import pandas as pd

from marl.part_c.metrics import battery_synchrony, summarise_record
from marl.part_c.statistics import (paired_interval, paired_comparisons,
                                    robustness_table, robustness_arm_comparison, seed_summary)


class PartCMetricsTests(unittest.TestCase):
    def test_perfect_same_and_opposing_power(self):
        x = np.linspace(-1., 1., 24)
        power = np.column_stack((x, x, -x))
        result = battery_synchrony(power)
        self.assertEqual(result['valid_pairs'], 3)
        self.assertAlmostEqual(result['mean_abs_correlation'], 1., places=8)
        self.assertAlmostEqual(result['synchrony'], -1/3, places=8)

    def test_constant_and_missing_power(self):
        x = np.linspace(-1, 1, 20)
        result = battery_synchrony(np.column_stack((x, x, np.ones(20))))
        self.assertEqual(result['valid_pairs'], 1)
        self.assertEqual(result['total_pairs'], 3)
        self.assertTrue(np.isnan(battery_synchrony(np.ones((20, 3)))['synchrony']))
        self.assertTrue(np.isnan(battery_synchrony(None)['synchrony']))

    def test_summary_uses_realized_power_and_delivery(self):
        T, n = 24, 3
        delivery = np.ones((T, n, n), bool)
        for i in range(n):
            delivery[:, i, i] = False
        record = dict(replicate=0, p=1., hold_last=False, force_mask=False,
                      actions=np.zeros((T, n)), battery_energy=np.stack([
                          np.linspace(0, 1, T)]*n, axis=1),
                      delivered=delivery, kpis=dict(primary=.91, avg_daily_peak=12.,
                      cost=15., cost_rel_r0=.95, **{'return': -5.}))
        row = summarise_record(record, arm='M1', condition='M1 p=1.0',
                               train_seed=0, eval_env_seed=7, channel_seed=8,
                               checkpoint_sha256='fake', split='validation')
        self.assertAlmostEqual(row['realized_delivery'], 1.)
        self.assertAlmostEqual(row['synchrony'], 1.)
        self.assertEqual(row['valid_pairs'], 3)
        self.assertAlmostEqual(row['primary'], .91)
        self.assertEqual(row['steps'], T)


class PartCStatisticsTests(unittest.TestCase):
    def _frame(self):
        rows = []
        for seed in range(5):
            for condition, arm, p, primary in [
                ('M0', 'M0', np.nan, 1.0 + seed * .01),
                ('M1 p=1.0', 'M1', 1., .9 + seed * .01),
                ('M1 p=0.5', 'M1', .5, .97 + seed * .01),
                ('M1-RND p=1.0', 'M1-RND', 1., .91 + seed * .01),
                ('M1-RND p=0.5', 'M1-RND', .5, .94 + seed * .01),
            ]:
                for rep in range(10):
                    rows.append(dict(arm=arm, condition=condition, seed=seed,
                                     replicate=rep, p=p, hold_last=False,
                                     force_mask=False, primary=primary,
                                     avg_daily_peak=20*primary, cost=100*primary,
                                     cost_rel_r0=primary, **{'return': -100*primary},
                                     synchrony=.6 if arm == 'M0' else .3,
                                     mean_abs_correlation=.6 if arm == 'M0' else .3,
                                     realized_delivery=p, valid_pairs=15,
                                     steps=359))
        return pd.DataFrame(rows)

    def test_paired_ci_uses_five_seeds_not_50_replicates(self):
        by_seed = seed_summary(self._frame())
        diff = paired_comparisons(by_seed)
        value = diff[(diff.condition == 'M1 p=1.0') & (diff.metric == 'primary')].iloc[0]
        self.assertEqual(int(value.n), 5)
        self.assertGreater(value['mean'], 5.)
        self.assertGreater(value['ci_low'], 0.)

    def test_robustness_sign(self):
        by_seed = seed_summary(self._frame())
        robustness = robustness_table(by_seed)
        m1_half = robustness[(robustness.arm == 'M1') & (robustness.p == .5)].iloc[0]
        m1_full = robustness[(robustness.arm == 'M1') & (robustness.p == 1.)].iloc[0]
        self.assertGreater(m1_half['mean'], 0.)
        self.assertAlmostEqual(m1_full['mean'], 0.)

    def test_robust_training_less_fragile_than_m1(self):
        by_seed = seed_summary(self._frame())
        comparison = robustness_arm_comparison(by_seed)
        row = comparison[comparison.p == .5].iloc[0]
        self.assertEqual(row['n'], 5)
        self.assertGreater(row['mean'], 0)

    def test_constant_differences_exact_ci(self):
        ci = paired_interval([10, 10, 10, 10, 10])
        self.assertAlmostEqual(ci['mean'], 10.)
        self.assertAlmostEqual(ci['ci_low'], 10.)
        self.assertAlmostEqual(ci['ci_high'], 10.)

    def test_insufficient_seeds_not_reported_as_significant(self):
        ci = paired_interval([9.])
        self.assertEqual(ci['n'], 1)
        self.assertTrue(np.isnan(ci['ci_low']))

    def test_duplicate_replicates_rejected(self):
        df = self._frame()
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            seed_summary(pd.concat([df, df.iloc[[0]]], ignore_index=True))


if __name__ == '__main__':
    unittest.main()
