# -*- coding: utf-8 -*-

# Wencke Liermann - wliermann@uni-potsdam.de
# Universität Potsdam
# Bachelor Computerlinguistik
# 4. Semester

# 28/08/2020
# Python 3.7.3
# Windows 8
"""distribution.py testcases."""

import logging
import unittest

from lib.distribution import IntegerDistribution, Distribution, LOG


LOG.setLevel(logging.CRITICAL)


class DistributionTestCase(unittest.TestCase):
    def test_initialize_from_iterable(self):
        distr = Distribution("ameisenhaufen")
        self.assertEqual(distr['e'], 3)

    def test_initialize_from_iterable_total(self):
        distr = Distribution("ameisenhaufen")
        self.assertEqual(distr.total, 13)

    def test_initialize_from_mapping(self):
        distr = Distribution({'a': 25, 'b': 5, 'c': 70})
        self.assertEqual(distr['c'], 70)

    def test_initialize_from_mapping_total(self):
        distr = Distribution({'a': 25, 'b': 5, 'c': 70})
        self.assertEqual(distr.total, 100)

    def test_prob_dist_from_distribution(self):
        distr = Distribution({'a': 25, 'b': 5, 'c': 70})
        pdistr = distr.prob_dist(iterable={'c', 'd'})
        self.assertEqual(pdistr, {'c': 0.7, 'd': 0})

    def test_provided_function_clear(self):
        distr = Distribution("ameisenhaufen")
        distr.clear()
        self.assertEqual(distr.total, 0)

    def test_provided_function_pop(self):
        distr = Distribution("ameisenhaufen")
        distr.pop('a')
        self.assertEqual(distr.total, 11)

    def test_reading_in_mapping_with_non_integer_values(self):
        inventar = {"monkey": 2, "banana": 4, "water": None}
        with self.assertRaises(TypeError) as exc:
            Distribution(inventar)
        self.assertEqual(str(exc.exception),
                         "Values of Distribution need to be positive integers.")

    def test_updating_distribution_with_dict(self):
        distr = Distribution({'a': 25, 'b': 5, 'c': 70})
        distr.update({'b': 5, 'd': 5})
        self.assertEqual(distr['b'], 10)

    def test_updating_distribution_with_dict_total(self):
        distr = Distribution({'a': 25, 'b': 5, 'c': 70})
        distr.update({'b': 5, 'd': 5})
        self.assertEqual(distr.total, 110)

    def test_updating_distribution_with_iterable(self):
        distr = Distribution("ameisenhaufen")
        distr.update("hasenbaumschule")
        self.assertEqual(distr['a'], 4)

    def test_updating_distribution_with_iterable_total(self):
        distr = Distribution("ameisenhaufen")
        distr.update("hasenbaumschule")
        self.assertEqual(distr.total, 28)


class IntegerDistributionTestCase(unittest.TestCase):
    def test_mean_calculation(self):
        sample = [9, 10, 12, 13, 13, 13, 15, 15, 16, 16, 18, 22, 23, 24, 24, 25]
        distr = IntegerDistribution(sample)
        self.assertAlmostEqual(distr.mean(), 16.75, places=2)

    def test_reading_in_mapping_with_non_integer_keys(self):
        inventar = {17: 2, 5: 4, "drei": 13}
        with self.assertRaises(TypeError) as exc:
            IntegerDistribution(inventar)
        self.assertEqual(str(exc.exception), "Keys of IntegerDistribution need to be integers.")

    def test_reading_in_mapping_with_non_integer_values(self):
        inventar = {17: 2, 5: 4, 3: None}
        with self.assertRaises(TypeError) as exc:
            IntegerDistribution(inventar)
        self.assertEqual(str(exc.exception),
                         "Values of Distribution need to be positive integers.")

    def test_stdev_calculation(self):
        sample = [9, 10, 12, 13, 13, 13, 15, 15, 16, 16, 18, 22, 23, 24, 24, 25]
        distr = IntegerDistribution(sample)
        self.assertAlmostEqual(distr.stdev(), 5.29, places=2)

    def test_updating_with_integer_values_iterable(self):
        sample = {9: 1, 10: 1, 12: 1, 13: 3, 15: 2, 16: 2, 18: 1, 22: 1, 23: 1, 24: 2, 25: 1}
        distr = IntegerDistribution(sample)
        distr.update([13, 13, 7, 7, 7, 16])
        self.assertEqual(distr[16], 3)

    def test_updating_with_non_integer_values_dict(self):
        sample = [9, 10, 12, 13, 13, 13, 15, 15, 16, 16, 18, 22, 23, 24, 24, 25]
        distr = IntegerDistribution(sample)
        with self.assertRaises(TypeError) as exc:
            distr.update({13: 2, 7: 3, 16: 1, 'acht': 8})
        self.assertEqual(str(exc.exception), "Keys of IntegerDistribution need to be integers.")
