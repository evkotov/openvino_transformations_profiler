import unittest
from table import get_comparison_values, compare_compile_time, sort_table
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


class TestSortTable(unittest.TestCase):

    def test_with_valid_float_keys(self):
        table = [{'value': 1.0}, {'value': 3.0}, {'value': 2.0}]
        sorted_table = sort_table(table, lambda row: row['value'])
        self.assertEqual(sorted_table, [{'value': 3.0}, {'value': 2.0}, {'value': 1.0}])

    def test_with_mixed_type_keys(self):
        table = [{'value': 1.0}, {'value': 'invalid'}, {'value': 2.0}]
        sorted_table = sort_table(table, lambda row: row['value'])
        self.assertEqual(sorted_table, [{'value': 2.0}, {'value': 1.0}, {'value': 'invalid'}])

    def test_with_all_invalid_keys(self):
        table = [{'value': 'invalid1'}, {'value': 'invalid2'}, {'value': 'invalid3'}]
        sorted_table = sort_table(table, lambda row: row['value'])
        self.assertEqual(sorted_table, [{'value': 'invalid1'}, {'value': 'invalid2'}, {'value': 'invalid3'}])

    def test_empty_table(self):
        table = []
        sorted_table = sort_table(table, lambda row: row.get('value', 0.0))
        self.assertEqual(sorted_table, [])

    def test_with_none_keys(self):
        table = [{'value': None}, {'value': 2.0}, {'value': 1.0}]
        sorted_table = sort_table(table, lambda row: row['value'] if row['value'] is not None else 0.0)
        self.assertEqual(sorted_table, [{'value': 2.0}, {'value': 1.0}, {'value': None}])


if __name__ == '__main__':
    unittest.main()
