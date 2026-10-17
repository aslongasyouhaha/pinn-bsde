"""Pricing identities and monitoring consistency for the common baselines."""
import unittest

from scripts.analytic_bs import barrier_price, bsm_price
from scripts.references import reference


class PricingReferences(unittest.TestCase):
    def config(self, option):
        return dict(s0=1., r=.03, sigma=.4, T=1., steps=32, mc_paths=8192,
                    up_barrier=1.2, down_barrier=.8, option_type=option,
                    strike=1., reference_seed=1729)

    def test_barrier_parity(self):
        for option in ['call', 'put']:
            vanilla = bsm_price(1., 1., 1., .03, .4, option)
            for direction, barrier in [('up', 1.2), ('down', .8)]:
                prices = [barrier_price(1., 1., barrier, 1., .03, .4, option, f'{direction}-and-{knock}')
                          for knock in ['in', 'out']]
                self.assertAlmostEqual(sum(prices), vanilla, places=12)

    def test_bridge_and_discrete_monitoring(self):
        for option in ['call', 'put']:
            for product in ['barrier_upout', 'barrier_downout']:
                with self.subTest(option=option, product=product):
                    result = reference(self.config(option), product)
                    # Conditional survival weighting cannot exceed a grid-only payoff.
                    self.assertLessEqual(result['bridge_mc'], result['discrete_mc'])
                    self.assertGreater(result['bridge_mc_se'], 0)
                    self.assertLess(abs(result['bridge_mc'] - result['continuous_analytic']),
                                    6 * result['bridge_mc_se'])


if __name__ == '__main__':
    unittest.main()
