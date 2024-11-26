import unittest
from table import get_comparison_values, compare_compile_time
from common_structs import ModelInfo, ModelData


class TestTableFunctions(unittest.TestCase):

    def test_get_comparison_values_with_valid_data(self):
        table = [{'key1': 1.0, 'key2': 2.0}, {'key1': 3.0, 'key2': 4.0}]
        comparison_values = get_comparison_values(table, 'key1', 'key2', 'ms')
        self.assertEqual(comparison_values.values1, [1.0, 3.0])
        self.assertEqual(comparison_values.values2, [2.0, 4.0])

    def test_get_comparison_values_with_non_float_values(self):
        table = [{'key1': 'a', 'key2': 'b'}, {'key1': 1.0, 'key2': 2.0}]
        comparison_values = get_comparison_values(table, 'key1', 'key2', 'ms')
        self.assertEqual(comparison_values.values1, [1.0])
        self.assertEqual(comparison_values.values2, [2.0])

    def test_get_comparison_values_with_empty_table(self):
        table = []
        comparison_values = get_comparison_values(table, 'key1', 'key2', 'ms')
        self.assertEqual(comparison_values.values1, [])
        self.assertEqual(comparison_values.values2, [])

    def test_get_comparison_values_with_mixed_data(self):
        table = [{'key1': 1.0, 'key2': 2.0}, {'key1': 'a', 'key2': 4.0}, {'key1': 3.0, 'key2': 'b'}]
        comparison_values = get_comparison_values(table, 'key1', 'key2', 'ms')
        self.assertEqual(comparison_values.values1, [1.0])
        self.assertEqual(comparison_values.values2, [2.0])


if __name__ == '__main__':
    unittest.main()
