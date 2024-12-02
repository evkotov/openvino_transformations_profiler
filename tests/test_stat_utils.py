import unittest
from ov_ts_profiler.stat_utils import join_sum_units_by_name, Total, get_comparison_values_compile_time, \
    get_comparison_values_sum_transformation_time, get_comparison_values_sum_units, get_common_models

from ov_ts_profiler.common_structs import ModelInfo, ModelData


class TestJoinSumUnitsByName(unittest.TestCase):
    def test_join_sum_units_by_name_returns_empty_dict_for_empty_input(self):
        data = {}
        result = join_sum_units_by_name(data)
        self.assertEqual(result, {})

    def test_join_sum_units_by_name_aggregates_totals_correctly(self):
        data = {
            "unit1": [Total(), Total()],
            "unit2": [Total()]
        }
        data["unit1"][0].duration = 100
        data["unit1"][0].count = 1
        data["unit1"][1].duration = 200
        data["unit1"][1].count = 2
        data["unit2"][0].duration = 300
        data["unit2"][0].count = 3

        result = join_sum_units_by_name(data)

        self.assertEqual(result["unit1"].duration, 300)
        self.assertEqual(result["unit1"].count, 3)
        self.assertEqual(result["unit2"].duration, 300)
        self.assertEqual(result["unit2"].count, 3)

    def test_join_sum_units_by_name_handles_none_values(self):
        data = {
            "unit1": [Total(), None],
            "unit2": [None, Total()]
        }
        data["unit1"][0].duration = 100
        data["unit1"][0].count = 1
        data["unit2"][1].duration = 200
        data["unit2"][1].count = 2

        result = join_sum_units_by_name(data)

        self.assertEqual(result["unit1"].duration, 100)
        self.assertEqual(result["unit1"].count, 1)
        self.assertEqual(result["unit2"].duration, 200)
        self.assertEqual(result["unit2"].count, 2)


class TestGetComparisonValuesCompileTime(unittest.TestCase):
    def test_comparison_values_compile_time_returns_empty_for_empty_data(self):
        data = []
        result = get_comparison_values_compile_time(data)
        self.assertEqual(result.values1, [])
        self.assertEqual(result.values2, [])

    def test_comparison_values_compile_time_ignores_incomplete_data(self):
        data = [
            (ModelInfo('framework', 'name1', 'precision', 'config'), [None, 1.0]),
            (ModelInfo('framework', 'name2', 'precision', 'config'), [2.0, None])
        ]
        result = get_comparison_values_compile_time(data)
        self.assertEqual(result.values1, [])
        self.assertEqual(result.values2, [])

    def test_comparison_values_compile_time_adds_valid_data(self):
        data = [
            (ModelInfo('framework', 'name1', 'precision', 'config'), [1.0, 2.0]),
            (ModelInfo('framework', 'name2', 'precision', 'config'), [3.0, 4.0])
        ]
        result = get_comparison_values_compile_time(data)
        self.assertEqual(result.values1, [1.0, 3.0])
        self.assertEqual(result.values2, [2.0, 4.0])

    def test_comparison_values_compile_time_handles_mixed_data(self):
        data = [
            (ModelInfo('framework', 'name1', 'precision', 'config'), [1.0, 2.0]),
            (ModelInfo('framework', 'name2', 'precision', 'config'), [None, 3.0]),
            (ModelInfo('framework', 'name3', 'precision', 'config'), [4.0, None]),
            (ModelInfo('framework', 'name4', 'precision', 'config'), [5.0, 6.0])
        ]
        result = get_comparison_values_compile_time(data)
        self.assertEqual(result.values1, [1.0, 5.0])
        self.assertEqual(result.values2, [2.0, 6.0])


class TestGetComparisonValuesSumTransformationTime(unittest.TestCase):
    def test_comparison_values_sum_transformation_time_returns_empty_for_empty_data(self):
        data = []
        result = get_comparison_values_sum_transformation_time(data)
        self.assertEqual(result.values1, [])
        self.assertEqual(result.values2, [])

    def test_comparison_values_sum_transformation_time_ignores_incomplete_data(self):
        data = [
            (ModelInfo('framework', 'name1', 'precision', 'config'), [None, 1.0]),
            (ModelInfo('framework', 'name2', 'precision', 'config'), [2.0, None])
        ]
        result = get_comparison_values_sum_transformation_time(data)
        self.assertEqual(result.values1, [])
        self.assertEqual(result.values2, [])

    def test_comparison_values_sum_transformation_time_adds_valid_data(self):
        data = [
            (ModelInfo('framework', 'name1', 'precision', 'config'), [1.0, 2.0]),
            (ModelInfo('framework', 'name2', 'precision', 'config'), [3.0, 4.0])
        ]
        result = get_comparison_values_sum_transformation_time(data)
        self.assertEqual(result.values1, [1.0, 3.0])
        self.assertEqual(result.values2, [2.0, 4.0])

    def test_comparison_values_sum_transformation_time_handles_mixed_data(self):
        data = [
            (ModelInfo('framework', 'name1', 'precision', 'config'), [1.0, 2.0]),
            (ModelInfo('framework', 'name2', 'precision', 'config'), [None, 3.0]),
            (ModelInfo('framework', 'name3', 'precision', 'config'), [4.0, None]),
            (ModelInfo('framework', 'name4', 'precision', 'config'), [5.0, 6.0])
        ]
        result = get_comparison_values_sum_transformation_time(data)
        self.assertEqual(result.values1, [1.0, 5.0])
        self.assertEqual(result.values2, [2.0, 6.0])


class TestGetComparisonValuesSumUnits(unittest.TestCase):
    def test_comparison_values_sum_units_returns_empty_for_empty_data(self):
        data = {}
        result = get_comparison_values_sum_units(data)
        self.assertEqual(result.values1, [])
        self.assertEqual(result.values2, [])

    def test_comparison_values_sum_units_ignores_incomplete_data(self):
        data = {
            "unit1": [Total(), None],
            "unit2": [None, Total()]
        }
        data["unit1"][0].duration = 100
        data["unit1"][0].count = 1
        data["unit2"][1].duration = 200
        data["unit2"][1].count = 2

        result = get_comparison_values_sum_units(data)
        self.assertEqual(result.values1, [])
        self.assertEqual(result.values2, [])

    def test_comparison_values_sum_units_adds_valid_data(self):
        data = {
            "unit1": [Total(), Total()],
            "unit2": [Total(), Total()]
        }
        data["unit1"][0].duration = 100
        data["unit1"][0].count = 1
        data["unit1"][1].duration = 200
        data["unit1"][1].count = 2
        data["unit2"][0].duration = 300
        data["unit2"][0].count = 3
        data["unit2"][1].duration = 400
        data["unit2"][1].count = 4

        result = get_comparison_values_sum_units(data)
        self.assertEqual(result.values1, [0.0001, 0.0003])
        self.assertEqual(result.values2, [0.0002, 0.0004])

    def test_comparison_values_sum_units_handles_mixed_data(self):
        data = {
            "unit1": [Total(), Total()],
            "unit2": [Total(), None],
            "unit3": [None, Total()],
            "unit4": [Total(), Total()]
        }
        data["unit1"][0].duration = 100
        data["unit1"][0].count = 1
        data["unit1"][1].duration = 200
        data["unit1"][1].count = 2
        data["unit2"][0].duration = 300
        data["unit2"][0].count = 3
        data["unit4"][0].duration = 500
        data["unit4"][0].count = 5
        data["unit4"][1].duration = 600
        data["unit4"][1].count = 6

        result = get_comparison_values_sum_units(data)
        self.assertEqual(result.values1, [0.0001, 0.0005])
        self.assertEqual(result.values2, [0.0002, 0.0006])


if __name__ == '__main__':
    unittest.main()


class TestGetCommonModels(unittest.TestCase):
    def test_common_models_with_no_data(self):
        data = []
        result = get_common_models(data)
        self.assertEqual(result, [])

    def test_common_models_with_single_entry(self):
        data = [{ModelInfo('framework1', 'model1', 'precision1', 'config1'): ModelData()}]
        result = get_common_models(data)
        self.assertEqual(result, [ModelInfo('framework1', 'model1', 'precision1', 'config1')])

    def test_common_models_with_multiple_entries(self):
        data = [
            {ModelInfo('framework1', 'model1', 'precision1', 'config1'): ModelData()},
            {ModelInfo('framework1', 'model1', 'precision1', 'config1'): ModelData(),
             ModelInfo('framework2', 'model2', 'precision2', 'config2'): ModelData()}
        ]
        result = get_common_models(data)
        self.assertEqual(result, [ModelInfo('framework1', 'model1', 'precision1', 'config1')])

    def test_common_models_with_no_common_entries(self):
        data = [
            {ModelInfo('framework1', 'model1', 'precision1', 'config1'): ModelData()},
            {ModelInfo('framework2', 'model2', 'precision2', 'config2'): ModelData()}
        ]
        result = get_common_models(data)
        self.assertEqual(result, [])
