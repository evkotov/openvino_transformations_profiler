import unittest
import numpy as np
from ov_ts_profiler.common_structs import ComparisonValues, SummaryStats

class TestComparisonValues(unittest.TestCase):

    def test_add_values(self):
        comp_values = ComparisonValues('ms')
        comp_values.add(1.0, 2.0)
        comp_values.add(3.0, 4.0)
        self.assertEqual(comp_values.values1, [1.0, 3.0])
        self.assertEqual(comp_values.values2, [2.0, 4.0])

    def test_get_differences(self):
        comp_values = ComparisonValues('ms')
        comp_values.add(1.0, 3.0)
        comp_values.add(4.0, 2.0)
        differences = comp_values.get_differences()
        np.testing.assert_array_equal(differences, np.array([2.0, -2.0]))

    def test_get_ratios(self):
        comp_values = ComparisonValues('ms')
        comp_values.add(1.0, 3.0)
        comp_values.add(2.0, 8.0)
        ratios = comp_values.get_ratios()
        np.testing.assert_array_equal(ratios, np.array([200.0, 300.0]))

    def test_get_max_values(self):
        comp_values = ComparisonValues('ms')
        comp_values.add(1.0, 3.0)
        comp_values.add(2.0, 1.0)
        max_values = comp_values.get_max_values()
        self.assertEqual(max_values, [3.0, 2.0])

    def test_get_stats(self):
        comp_values = ComparisonValues('ms')
        comp_values.add(1.0, 2.0)
        comp_values.add(-1.0, -2.0)
        stats = comp_values.get_stats()
        expected_stats = SummaryStats(0.0, 0.0, 1.0, 1.0, 100.0, 100.0, 0.0, 100.0, 'ms')
        self.assertEqual(stats, expected_stats)

if __name__ == '__main__':
    unittest.main()
