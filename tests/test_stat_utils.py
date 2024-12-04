import unittest
from unittest.mock import MagicMock, patch
from ov_ts_profiler.stat_utils import join_sum_units_by_name, Total, get_comparison_values_compile_time, \
    get_comparison_values_sum_transformation_time, get_comparison_values_sum_units, get_common_models, \
    get_total_by_unit_names_by_csv, get_sum_units_comparison_data, get_total_by_unit_names, \
    get_sum_plain_manager_time_data, get_sum_plain_manager_gap_time_data

from ov_ts_profiler.common_structs import ModelInfo, ModelData, Unit


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


class TestGetTotalByUnitNamesByCsv(unittest.TestCase):

    def test_returns_empty_list_when_no_model_data(self):
        result = get_total_by_unit_names_by_csv([], 'unit_type')
        self.assertEqual(result, [])

    def test_returns_empty_dict_for_none_model_data(self):
        model_data_items = [None]
        result = get_total_by_unit_names_by_csv(model_data_items, 'unit_type')
        self.assertEqual(result, [{}])

    def test_returns_correct_totals_for_single_model_data(self):
        model_data = MagicMock()
        model_data.collect_items_by_type.return_value = {'unit1': [MagicMock(), MagicMock()]}
        model_data_items = [model_data]
        result = get_total_by_unit_names_by_csv(model_data_items, 'unit_type')
        self.assertEqual(len(result), 1)
        self.assertIn('unit1', result[0])

    def test_handles_multiple_model_data_items(self):
        model_data1 = MagicMock()
        model_data1.collect_items_by_type.return_value = {'unit1': [MagicMock()]}
        model_data2 = MagicMock()
        model_data2.collect_items_by_type.return_value = {'unit2': [MagicMock()]}
        model_data_items = [model_data1, model_data2]
        result = get_total_by_unit_names_by_csv(model_data_items, 'unit_type')
        self.assertEqual(len(result), 2)
        self.assertIn('unit1', result[0])
        self.assertIn('unit2', result[1])


class TestStatUtilsGetSumUnitComparisonData(unittest.TestCase):

    @patch('ov_ts_profiler.stat_utils.full_join_by_model_info')
    def test_returns_empty_iterator_when_no_data(self, mock_full_join):
        mock_full_join.return_value = []
        result = list(get_sum_units_comparison_data([], 'unit_type'))
        self.assertEqual(result, [])

    @patch('ov_ts_profiler.stat_utils.full_join_by_model_info')
    def test_returns_empty_dict_for_none_model_data(self, mock_full_join):
        mock_full_join.return_value = [(ModelInfo('framework', 'name', 'precision', 'config'), [None])]
        result = list(get_sum_units_comparison_data([{}], 'unit_type'))
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][1], {})

    @patch('ov_ts_profiler.stat_utils.full_join_by_model_info')
    def test_returns_correct_totals_for_single_model_data(self, mock_full_join):
        model_data = MagicMock()
        model_data.collect_items_by_type.return_value = {'unit1': [MagicMock(), MagicMock()]}
        mock_full_join.return_value = [(ModelInfo('framework', 'name', 'precision', 'config'), [model_data])]
        result = list(get_sum_units_comparison_data([{}], 'unit_type'))
        self.assertEqual(len(result), 1)
        self.assertIn('unit1', result[0][1])
        self.assertEqual(len(result[0][1]['unit1']), 1)

    @patch('ov_ts_profiler.stat_utils.full_join_by_model_info')
    def test_handles_multiple_model_data_items(self, mock_full_join):
        model_data1 = MagicMock()
        model_data1.collect_items_by_type.return_value = {'unit1': [MagicMock()]}
        model_data2 = MagicMock()
        model_data2.collect_items_by_type.return_value = {'unit2': [MagicMock()]}
        mock_full_join.return_value = [(ModelInfo('framework', 'name', 'precision', 'config'), [model_data1, model_data2])]
        result = list(get_sum_units_comparison_data([{}], 'unit_type'))
        self.assertEqual(len(result), 1)
        self.assertIn('unit1', result[0][1])
        self.assertIn('unit2', result[0][1])
        self.assertEqual(len(result[0][1]['unit1']), 1)
        self.assertEqual(len(result[0][1]['unit2']), 1)

    @patch('ov_ts_profiler.stat_utils.full_join_by_model_info')
    def test_valid_data(self, mock_full_join):
        model_info1 = MagicMock(name='model_info1')
        model_info2 = MagicMock(name='model_info2')
        model_data1 = MagicMock()
        model_data2 = MagicMock()
        unit1 = MagicMock()
        unit2 = MagicMock()
        unit1.get_duration_median.return_value = 100
        unit2.get_duration_median.return_value = 200
        model_data1.collect_items_by_type.return_value = {"unit1": [unit1]}
        model_data2.collect_items_by_type.return_value = {"unit2": [unit2]}
        data = [{model_info1: model_data1}, {model_info2: model_data2}]
        unit_type = "type1"

        mock_full_join.return_value = [(model_info1, [model_data1]), (model_info2, [model_data2])]

        result = list(get_sum_units_comparison_data(data, unit_type))

        self.assertEqual(len(result), 2)
        self.assertIn("unit1", result[0][1])
        self.assertIn("unit2", result[1][1])
        self.assertEqual(result[0][1]["unit1"][0].duration, 100)
        self.assertEqual(result[1][1]["unit2"][0].duration, 200)

    def test_empty_data(self):
        data = []
        unit_type = "type1"

        result = list(get_sum_units_comparison_data(data, unit_type))

        self.assertEqual(result, [])

    def test_none_model_data(self):
        model_info = MagicMock()
        data = [{model_info: None}]
        unit_type = "type1"

        result = list(get_sum_units_comparison_data(data, unit_type))

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][1], {})

    def test_duplicate_units(self):
        model_info = MagicMock()
        model_data = MagicMock()
        unit1 = MagicMock()
        unit2 = MagicMock()
        unit1.get_duration_median.return_value = 100
        unit2.get_duration_median.return_value = 200
        model_data.collect_items_by_type.return_value = {"unit1": [unit1, unit2]}
        data = [{model_info: model_data}]
        unit_type = "type1"

        result = list(get_sum_units_comparison_data(data, unit_type))

        self.assertEqual(len(result), 1)
        self.assertIn("unit1", result[0][1])
        self.assertEqual(result[0][1]["unit1"][0].duration, 300)


class TestGetTotalByUnitNames(unittest.TestCase):

    def test_get_total_by_unit_names_with_valid_data(self):
        unit1 = MagicMock()
        unit1.get_duration_median.return_value = 100
        unit1.status = '1'
        unit2 = MagicMock()
        unit2.get_duration_median.return_value = 200
        unit2.status = '0'
        units_by_type = {"unit1": [unit1], "unit2": [unit2]}

        result = get_total_by_unit_names(units_by_type)

        self.assertEqual(result["unit1"].duration, 100)
        self.assertEqual(result["unit1"].count, 1)
        self.assertEqual(result["unit1"].count_status_true, 1)
        self.assertEqual(result["unit2"].duration, 200)
        self.assertEqual(result["unit2"].count, 1)
        self.assertEqual(result["unit2"].count_status_true, 0)

    def test_get_total_by_unit_names_with_empty_data(self):
        units_by_type = {}

        result = get_total_by_unit_names(units_by_type)

        self.assertEqual(result, {})

    def test_get_total_by_unit_names_with_duplicate_units(self):
        unit1 = MagicMock()
        unit1.get_duration_median.return_value = 100
        unit1.status = '1'
        unit2 = MagicMock()
        unit2.get_duration_median.return_value = 200
        unit2.status = '0'
        units_by_type = {"unit1": [unit1, unit2]}

        result = get_total_by_unit_names(units_by_type)

        self.assertEqual(result["unit1"].duration, 300)
        self.assertEqual(result["unit1"].count, 2)
        self.assertEqual(result["unit1"].count_status_true, 1)

    def test_get_total_by_unit_names_with_none_units(self):
        units_by_type = {"unit1": None}

        with self.assertRaises(TypeError):
            get_total_by_unit_names(units_by_type)


class TestGetSumPlainManagerTime(unittest.TestCase):

    def test_get_sum_plain_manager_time_data_returns_correct_data(self):
        model_info_1 = ModelInfo(framework="framework1", name="model1", precision="precision1", config="config1")
        model_info_2 = ModelInfo(framework="framework2", name="model2", precision="precision2", config="config2")

        model_data_1 = MagicMock(spec=ModelData)
        model_data_1.get_manager_plain_sequence_median_sum.return_value = 2000000

        model_data_2 = MagicMock(spec=ModelData)
        model_data_2.get_manager_plain_sequence_median_sum.return_value = 3000000

        data = [
            {model_info_1: model_data_1},
            {model_info_2: model_data_2}
        ]

        result = list(get_sum_plain_manager_time_data(data))

        self.assertEqual(result, [
            (model_info_1, [2.0, None]),
            (model_info_2, [None, 3.0])
        ])

    def test_get_sum_plain_manager_time_data_handles_empty_data(self):
        data = []
        result = list(get_sum_plain_manager_time_data(data))
        self.assertEqual(result, [])

    def test_get_sum_plain_manager_time_data_handles_none_model_data(self):
        model_info_1 = ModelInfo(framework="framework1", name="model1", precision="precision1", config="config1")
        data = [
            {model_info_1: None}
        ]
        result = list(get_sum_plain_manager_time_data(data))
        self.assertEqual(result, [
            (model_info_1, [None])
        ])


class TestGetManagerPlainSequenceGap(unittest.TestCase):

    def test_get_sum_plain_manager_gap_time_data_returns_correct_data(self):
        model_info_1 = ModelInfo(framework="framework1", name="model1", precision="precision1", config="config1")
        model_info_2 = ModelInfo(framework="framework2", name="model2", precision="precision2", config="config2")

        model_data_1 = MagicMock(spec=ModelData)
        model_data_1.get_manager_plain_sequence_median_gap_sum.return_value = 2000000

        model_data_2 = MagicMock(spec=ModelData)
        model_data_2.get_manager_plain_sequence_median_gap_sum.return_value = 3000000

        data = [
            {model_info_1: model_data_1},
            {model_info_2: model_data_2}
        ]

        result = list(get_sum_plain_manager_gap_time_data(data))

        self.assertEqual(result, [
            (model_info_1, [2.0, None]),
            (model_info_2, [None, 3.0])
        ])

    def test_get_sum_plain_manager_gap_time_data_handles_empty_data(self):
        data = []
        result = list(get_sum_plain_manager_gap_time_data(data))
        self.assertEqual(result, [])

    def test_get_sum_plain_manager_gap_time_data_handles_none_model_data(self):
        model_info_1 = ModelInfo(framework="framework1", name="model1", precision="precision1", config="config1")
        data = [
            {model_info_1: None}
        ]
        result = list(get_sum_plain_manager_gap_time_data(data))
        self.assertEqual(result, [
            (model_info_1, [None])
        ])


if __name__ == '__main__':
    unittest.main()
