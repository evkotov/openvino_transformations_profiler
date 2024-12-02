import unittest
from typing import List
from unittest.mock import MagicMock
from ov_ts_profiler.table import (sort_table, compare_compile_time, compare_sum_transformation_time,
                                  compare_sum_units, get_longest_unit, create_comparison_summary_table)
from ov_ts_profiler.stat_utils import get_compile_time_data, get_sum_transformation_time_data, group_units_by_name, \
    get_sum_units_comparison_data, join_sum_units, Total
from ov_ts_profiler.common_structs import ModelInfo, ModelData, Unit, ComparisonValues


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


class TestGetCompileTimeData(unittest.TestCase):

    def test_returns_correct_compile_times(self):
        model_info_1 = ModelInfo('framework1', 'model1', 'fp32', 'config1')
        model_info_2 = ModelInfo('framework2', 'model2', 'fp16', 'config2')
        model_data_1 = MagicMock()
        model_data_1.get_compile_time.return_value = 2_000_000_000
        model_data_2 = MagicMock()
        model_data_2.get_compile_time.return_value = 3_000_000_000
        data = [{model_info_1: model_data_1}, {model_info_2: model_data_2}]

        result = list(get_compile_time_data(data))

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0][0], model_info_1)
        self.assertEqual(result[0][1], [2.0, None])
        self.assertEqual(result[1][0], model_info_2)
        self.assertEqual(result[1][1], [None, 3.0])

    def test_handles_none_model_data(self):
        model_info_1 = ModelInfo('framework1', 'model1', 'fp32', 'config1')
        model_info_2 = ModelInfo('framework2', 'model2', 'fp16', 'config2')
        model_data_1 = MagicMock()
        model_data_1.get_compile_time.return_value = 2_000_000_000
        data = [{model_info_1: model_data_1}, {model_info_2: None}]

        result = list(get_compile_time_data(data))

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0][0], model_info_1)
        self.assertEqual(result[0][1], [2.0, None])
        self.assertEqual(result[1][0], model_info_2)
        self.assertEqual(result[1][1], [None, None])

    def test_handles_empty_data(self):
        data = []
        result = list(get_compile_time_data(data))
        self.assertEqual(result, [])


class TestGetSumTransformationTimeData(unittest.TestCase):

    def test_returns_correct_transformation_times(self):
        model_info_1 = ModelInfo('framework1', 'model1', 'fp32', 'config1')
        model_info_2 = ModelInfo('framework2', 'model2', 'fp16', 'config2')
        model_data_1 = ModelData()
        model_data_1.sum_transformation_time = MagicMock(return_value=2_000_000)
        model_data_2 = ModelData()
        model_data_2.sum_transformation_time = MagicMock(return_value=3_000_000)
        data = [{model_info_1: model_data_1}, {model_info_2: model_data_2}]

        result = list(get_sum_transformation_time_data(data))

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0][0], model_info_1)
        self.assertEqual(result[0][1], [2.0, None])
        self.assertEqual(result[1][0], model_info_2)
        self.assertEqual(result[1][1], [None, 3.0])

    def test_handles_none_model_data(self):
        model_info_1 = ModelInfo('framework1', 'model1', 'fp32', 'config1')
        model_info_2 = ModelInfo('framework2', 'model2', 'fp16', 'config2')
        model_data_1 = ModelData()
        model_data_1.sum_transformation_time = MagicMock(return_value=2_000_000)
        data = [{model_info_1: model_data_1}, {model_info_2: None}]

        result = list(get_sum_transformation_time_data(data))

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0][0], model_info_1)
        self.assertEqual(result[0][1], [2.0, None])
        self.assertEqual(result[1][0], model_info_2)
        self.assertEqual(result[1][1], [None, None])

    def test_handles_empty_data(self):
        data = []
        result = list(get_sum_transformation_time_data(data))
        self.assertEqual(result, [])


class TestGroupUnitsByName(unittest.TestCase):

    @staticmethod
    def create_fake_unit():
        fake_unit = MagicMock(spec=Unit)
        fake_unit.csv_item = 'fake_csv_item'
        fake_unit.device = 'fake_device'
        fake_unit.type = 'fake_type'
        fake_unit.name = 'fake_name'
        fake_unit.transformation_name = 'fake_transformation'
        fake_unit.manager_name = 'fake_manager'
        fake_unit.status = 'fake_status'
        return fake_unit

    def test_group_units_by_name_with_multiple_units(self):
        model_data = MagicMock()
        unit_type = 'test_type'
        model_data.collect_items_by_type.return_value = {
            'unit1': [TestGroupUnitsByName.create_fake_unit(), TestGroupUnitsByName.create_fake_unit()],
            'unit2': [TestGroupUnitsByName.create_fake_unit()]
        }

        result = group_units_by_name(model_data, unit_type)

        self.assertEqual(len(result), 2)
        self.assertEqual(len(result['unit1']), 2)
        self.assertEqual(len(result['unit2']), 1)

    def test_group_units_by_name_with_no_units(self):
        model_data = MagicMock()
        unit_type = 'test_type'
        model_data.collect_items_by_type.return_value = {}

        result = group_units_by_name(model_data, unit_type)

        self.assertEqual(len(result), 0)

    def test_group_units_by_name_with_empty_unit_list(self):
        model_data = MagicMock()
        unit_type = 'test_type'
        model_data.collect_items_by_type.return_value = {
            'unit1': []
        }

        result = group_units_by_name(model_data, unit_type)

        self.assertEqual(len(result), 1)
        self.assertEqual(len(result['unit1']), 0)


class TestTotal(unittest.TestCase):

    def test_initializes_with_default_values(self):
        total = Total()
        self.assertEqual(total.duration, 0.0)
        self.assertEqual(total.count, 0)
        self.assertEqual(total.count_status_true, 0)

    def test_calculates_duration_in_ms(self):
        total = Total()
        total.duration = 1_000_000
        self.assertEqual(total.get_duration_ms(), 1.0)

    def test_appends_another_total(self):
        total1 = Total()
        total1.duration = 1_000_000
        total1.count = 1
        total1.count_status_true = 1

        total2 = Total()
        total2.duration = 2_000_000
        total2.count = 2
        total2.count_status_true = 1

        total1.append(total2)
        self.assertEqual(total1.duration, 3_000_000)
        self.assertEqual(total1.count, 3)
        self.assertEqual(total1.count_status_true, 2)

    def test_appends_unit(self):
        unit = MagicMock(spec=Unit)
        unit.get_duration_median.return_value = 1_000_000
        unit.status = '1'

        total = Total()
        total.append_unit(unit)
        self.assertEqual(total.duration, 1_000_000)
        self.assertEqual(total.count, 1)
        self.assertEqual(total.count_status_true, 1)

    def test_appends_unit_with_different_status(self):
        unit = MagicMock(spec=Unit)
        unit.get_duration_median.return_value = 1_000_000
        unit.status = '0'

        total = Total()
        total.append_unit(unit)
        self.assertEqual(total.duration, 1_000_000)
        self.assertEqual(total.count, 1)
        self.assertEqual(total.count_status_true, 0)


class TestGetSumUnitsComparisonData(unittest.TestCase):

    @staticmethod
    def create_fake_unit(name: str, duration: float, status: str):
        fake_unit = MagicMock(spec=Unit)
        fake_unit.csv_item = 'fake_csv_item'
        fake_unit.device = 'fake_device'
        fake_unit.type = 'fake_type'
        fake_unit.name = name
        fake_unit.transformation_name = 'fake_transformation'
        fake_unit.manager_name = 'fake_manager'
        fake_unit.status = status
        fake_unit.get_duration_median.return_value = duration
        return fake_unit

    def test_returns_correct_unit_comparison_data(self):
        model_info_1 = ModelInfo('framework1', 'model1', 'fp32', 'config1')
        model_info_2 = ModelInfo('framework2', 'model2', 'fp16', 'config2')
        unit_1 = TestGetSumUnitsComparisonData.create_fake_unit('unit1', 1_000_000, '1')
        unit_2 = TestGetSumUnitsComparisonData.create_fake_unit('unit2', 2_000_000, '0')
        model_data_1 = ModelData()
        model_data_1.collect_items_by_type = MagicMock(return_value={'unit1': [unit_1]})
        model_data_2 = ModelData()
        model_data_2.collect_items_by_type = MagicMock(return_value={'unit2': [unit_2]})
        data = [{model_info_1: model_data_1}, {model_info_2: model_data_2}]

        result = list(get_sum_units_comparison_data(data, 'unit_type'))

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0][0], model_info_1)
        self.assertEqual(result[1][0], model_info_2)
        self.assertEqual(result[0][1]['unit1'][0].get_duration_ms(), 1.0)
        self.assertEqual(result[1][1]['unit2'][1].get_duration_ms(), 2.0)

    def test_handles_none_model_data(self):
        model_info_1 = ModelInfo('framework1', 'model1', 'fp32', 'config1')
        model_info_2 = ModelInfo('framework2', 'model2', 'fp16', 'config2')
        unit_1 = TestGetSumUnitsComparisonData.create_fake_unit('unit1', 1_000_000, '1')
        model_data_1 = ModelData()
        model_data_1.collect_items_by_type = MagicMock(return_value={'unit1': [unit_1]})
        data = [{model_info_1: model_data_1}, {model_info_2: None}]

        result = list(get_sum_units_comparison_data(data, 'unit_type'))

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0][0], model_info_1)
        self.assertEqual(result[1][0], model_info_2)
        self.assertEqual(result[0][1]['unit1'][0].get_duration_ms(), 1.0)
        self.assertIsNone(result[1][1].get('unit1'))

    def test_handles_empty_data(self):
        data = []
        result = list(get_sum_units_comparison_data(data, 'unit_type'))
        self.assertEqual(result, [])


class TestJoinSumUnits(unittest.TestCase):

    def test_returns_empty_dict_for_empty_input(self):
        data = iter([])
        result = join_sum_units(data)
        self.assertEqual(result, {})

    def test_aggregates_totals_by_unit_name(self):
        model_info = MagicMock(spec=ModelInfo)
        total1 = MagicMock(spec=Total)
        total2 = MagicMock(spec=Total)
        data = iter([(model_info, {'unit1': [total1], 'unit2': [total2]})])
        result = join_sum_units(data)
        self.assertEqual(result, {'unit1': [total1], 'unit2': [total2]})

    def test_handles_multiple_entries_for_same_unit_name(self):
        model_info = MagicMock(spec=ModelInfo)
        total1 = MagicMock(spec=Total)
        total2 = MagicMock(spec=Total)
        total3 = MagicMock(spec=Total)
        data = iter([(model_info, {'unit1': [total1]}), (model_info, {'unit1': [total2, total3]})])
        result = join_sum_units(data)
        self.assertEqual(result, {'unit1': [total1, total2, total3]})

    def test_handles_none_totals(self):
        model_info = MagicMock(spec=ModelInfo)
        total1 = MagicMock(spec=Total)
        data = iter([(model_info, {'unit1': [total1, None]})])
        result = join_sum_units(data)
        self.assertEqual(result, {'unit1': [total1, None]})


class TestCompareCompileTime(unittest.TestCase):

    def test_returns_correct_header_for_single_csv(self):
        result = compare_compile_time([], 1)
        expected_header = ['framework', 'name', 'precision', 'config', 'compile time #1 (secs)']
        self.assertEqual(result[0], expected_header)

    def test_returns_correct_header_for_multiple_csvs(self):
        result = compare_compile_time([], 3)
        expected_header = ['framework', 'name', 'precision', 'config', 'compile time #1 (secs)', 'compile time #2 (secs)', 'compile time #3 (secs)', 'compile time #2 - #1 (secs)', 'compile time #3 - #1 (secs)', 'compile time #2/#1', 'compile time #3/#1']
        self.assertEqual(result[0], expected_header)

    def test_handles_empty_data(self):
        result = compare_compile_time([], 1)
        self.assertEqual(result[1], [])

    def test_handles_none_compile_times(self):
        model_info = MagicMock(spec=ModelInfo)
        data = [(model_info, [None, None])]
        result = compare_compile_time(data, 2)
        self.assertEqual(result[1][0]['compile time #1 (secs)'], 'N/A')
        self.assertEqual(result[1][0]['compile time #2 (secs)'], 'N/A')

    def test_calculates_deltas_and_ratios_correctly(self):
        model_info = MagicMock(spec=ModelInfo)
        data = [(model_info, [10.0, 20.0, 30.0])]
        result = compare_compile_time(data, 3)
        self.assertEqual(result[1][0]['compile time #2 - #1 (secs)'], 10.0)
        self.assertEqual(result[1][0]['compile time #3 - #1 (secs)'], 20.0)
        self.assertEqual(result[1][0]['compile time #2/#1'], 2.0)
        self.assertEqual(result[1][0]['compile time #3/#1'], 3.0)


class TestCompareSumTransformationTime(unittest.TestCase):

    def test_returns_correct_header_for_single_csv(self):
        result = compare_sum_transformation_time([], 1)
        expected_header = ['framework', 'name', 'precision', 'config', 'time #1 (ms)']
        self.assertEqual(result[0], expected_header)

    def test_returns_correct_header_for_multiple_csvs(self):
        result = compare_sum_transformation_time([], 3)
        expected_header = ['framework', 'name', 'precision', 'config', 'time #1 (ms)', 'time #2 (ms)', 'time #3 (ms)', 'time #2 - #1 (ms)', 'time #3 - #1 (ms)', 'time #2/#1', 'time #3/#1']
        self.assertEqual(result[0], expected_header)

    def test_handles_empty_data(self):
        result = compare_sum_transformation_time([], 1)
        self.assertEqual(result[1], [])

    def test_handles_none_transformation_times(self):
        model_info = MagicMock(spec=ModelInfo)
        data = [(model_info, [None, None])]
        result = compare_sum_transformation_time(data, 2)
        self.assertEqual(result[1][0]['time #1 (ms)'], 'N/A')
        self.assertEqual(result[1][0]['time #2 (ms)'], 'N/A')

    def test_calculates_deltas_and_ratios_correctly(self):
        model_info = MagicMock(spec=ModelInfo)
        data = [(model_info, [10.0, 20.0, 30.0])]
        result = compare_sum_transformation_time(data, 3)
        self.assertEqual(result[1][0]['time #2 - #1 (ms)'], 10.0)
        self.assertEqual(result[1][0]['time #3 - #1 (ms)'], 20.0)
        self.assertEqual(result[1][0]['time #2/#1'], 2.0)
        self.assertEqual(result[1][0]['time #3/#1'], 3.0)


class TestCompareSumUnits(unittest.TestCase):

    def test_returns_correct_header_for_single_csv(self):
        result = compare_sum_units({}, 1)
        expected_header = ['name', 'duration #1 (ms)', 'count #1', 'count status true #1']
        self.assertEqual(result[0], expected_header)

    def test_returns_correct_header_for_multiple_csvs(self):
        result = compare_sum_units({}, 3)
        expected_header = ['name', 'duration #1 (ms)', 'duration #2 (ms)', 'duration #3 (ms)', 'duration #2 - #1 (ms)', 'duration #3 - #1 (ms)', 'duration #2/#1', 'duration #3/#1', 'count #1', 'count #2', 'count #3', 'count #2 - #1', 'count #3 - #1', 'count status true #1', 'count status true #2', 'count status true #3', 'count status true #2 - #1', 'count status true #3 - #1']
        self.assertEqual(result[0], expected_header)

    def test_handles_empty_data(self):
        result = compare_sum_units({}, 1)
        self.assertEqual(result[1], [])

    def test_handles_none_totals(self):
        data = {'unit1': [None, None]}
        result = compare_sum_units(data, 2)
        self.assertEqual(result[1][0]['duration #1 (ms)'], 'N/A')
        self.assertEqual(result[1][0]['duration #2 (ms)'], 'N/A')

    def test_calculates_deltas_and_ratios_correctly(self):
        total1 = MagicMock(spec=Total)
        total1.get_duration_ms.return_value = 10.0
        total1.count = 1
        total1.count_status_true = 1

        total2 = MagicMock(spec=Total)
        total2.get_duration_ms.return_value = 20.0
        total2.count = 2
        total2.count_status_true = 1

        total3 = MagicMock(spec=Total)
        total3.get_duration_ms.return_value = 30.0
        total3.count = 3
        total3.count_status_true = 2

        data = {'unit1': [total1, total2, total3]}
        result = compare_sum_units(data, 3)
        self.assertEqual(result[1][0]['duration #2 - #1 (ms)'], 10.0)
        self.assertEqual(result[1][0]['duration #3 - #1 (ms)'], 20.0)
        self.assertEqual(result[1][0]['duration #2/#1'], 2.0)
        self.assertEqual(result[1][0]['duration #3/#1'], 3.0)


class TestGetLongestUnit(unittest.TestCase):
    def test_get_longest_unit_returns_correct_header(self):
        data = {}
        header, _ = get_longest_unit(data)
        self.assertEqual(header, ['name', 'total duration (ms)', 'count of executions', 'count of status true'])

    def test_get_longest_unit_returns_empty_table_for_empty_input(self):
        data = {}
        _, table = get_longest_unit(data)
        self.assertEqual(table, [])

    def test_get_longest_unit_calculates_durations_correctly(self):
        data = {
            "unit1": Total(),
            "unit2": Total()
        }
        data["unit1"].duration = 1_000_000
        data["unit1"].count = 1
        data["unit1"].count_status_true = 1
        data["unit2"].duration = 2_000_000
        data["unit2"].count = 2
        data["unit2"].count_status_true = 2
        _, table = get_longest_unit(data)
        self.assertEqual(table, [
            {'name': 'unit2', 'total duration (ms)': 2.0, 'count of executions': 2, 'count of status true': 2},
            {'name': 'unit1', 'total duration (ms)': 1.0, 'count of executions': 1, 'count of status true': 1}
        ])

    def test_get_longest_unit_handles_zero_duration(self):
        data = {
            "unit1": Total()
        }
        data["unit1"].duration = 0
        data["unit1"].count = 1
        data["unit1"].count_status_true = 1
        _, table = get_longest_unit(data)
        self.assertEqual(table, [
            {'name': 'unit1', 'total duration (ms)': 0.0, 'count of executions': 1, 'count of status true': 1}
        ])

    def test_get_longest_unit_sorts_by_duration(self):
        data = {
            "unit1": Total(),
            "unit2": Total(),
            "unit3": Total()
        }
        data["unit1"].duration = 1_000_000
        data["unit1"].count = 1
        data["unit1"].count_status_true = 1
        data["unit2"].duration = 3_000_000
        data["unit2"].count = 2
        data["unit2"].count_status_true = 2
        data["unit3"].duration = 2_000_000
        data["unit3"].count = 3
        data["unit3"].count_status_true = 3

        _, table = get_longest_unit(data)
        self.assertEqual(table, [
            {'name': 'unit2', 'total duration (ms)': 3.0, 'count of executions': 2, 'count of status true': 2},
            {'name': 'unit3', 'total duration (ms)': 2.0, 'count of executions': 3, 'count of status true': 3},
            {'name': 'unit1', 'total duration (ms)': 1.0, 'count of executions': 1, 'count of status true': 1}
        ])


class TestCreateComparisonSummaryTable(unittest.TestCase):
    @staticmethod
    def create_values(unit: str, l1: List[float], l2: List[float]) -> ComparisonValues:
        values = ComparisonValues(unit)
        values.values1 = l1
        values.values2 = l2
        return values

    def test_create_comparison_summary_table_returns_correct_header(self):
        data = {
            ModelInfo('framework', 'name1', 'precision', 'config'): TestCreateComparisonSummaryTable.create_values('ms', [1.0], [1.0]),
            ModelInfo('framework', 'name2', 'precision', 'config'): TestCreateComparisonSummaryTable.create_values('ms', [2.0], [2.0])
        }
        header, _ = create_comparison_summary_table(data)
        expected_header = ['framework', 'name', 'precision', 'config', 'delta median, ms', 'delta mean, ms', 'delta std, ms', 'delta max, ms', 'ratio median', 'ratio mean', 'ratio std', 'ratio max']
        self.assertEqual(header, expected_header)

    def test_create_comparison_summary_table_handles_empty_data(self):
        data = {}
        header, table = create_comparison_summary_table(data)
        self.assertEqual(table, [])

    def test_create_comparison_summary_table_calculates_correct_values(self):
        data = {
            ModelInfo('framework', 'name1', 'precision', 'config'): TestCreateComparisonSummaryTable.create_values('ms', [1.0, 1.2, 1.3], [1.6, 1.9, 1.7]),
            ModelInfo('framework', 'name2', 'precision', 'config'): TestCreateComparisonSummaryTable.create_values('ms', [2.1, 2.2, 2.5], [1.2, 1.0, 1.1])
        }
        _, table = create_comparison_summary_table(data)
        self.assertEqual(len(table), 2)
        self.assertEqual(table[0]['delta median, ms'], 0.6000000000000001)
        self.assertEqual(table[0]['delta mean, ms'], 0.5666666666666667)
        self.assertEqual(table[0]['delta std, ms'], 0.12472191289246475)
        self.assertEqual(table[0]['delta max, ms'], 0.7)
        self.assertEqual(table[0]['ratio median'], 58.33333333333333)
        self.assertEqual(table[0]['ratio mean'], 49.700854700854705)
        self.assertEqual(table[0]['ratio std'], 13.40396043366217)
        self.assertEqual(table[0]['ratio max'], 60.00000000000001)


if __name__ == '__main__':
    unittest.main()
