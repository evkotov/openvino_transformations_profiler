import unittest
import numpy as np
from ov_ts_profiler.common_structs import ComparisonValues, SummaryStats, CSVItem, Unit, \
    make_model_console_description
from ov_ts_profiler.common_structs import ModelInfo, ModelData, full_join_by_model_info


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


class TestFullJoinByModelInfo(unittest.TestCase):

    def test_returns_correct_joined_data(self):
        model_info_1 = ModelInfo('framework1', 'model1', 'fp32', 'config1')
        model_info_2 = ModelInfo('framework2', 'model2', 'fp16', 'config2')
        model_data_1 = ModelData()
        model_data_2 = ModelData()
        data = [{model_info_1: model_data_1}, {model_info_2: model_data_2}]

        result = list(full_join_by_model_info(data))

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0][0], model_info_1)
        self.assertEqual(result[0][1], [model_data_1, None])
        self.assertEqual(result[1][0], model_info_2)
        self.assertEqual(result[1][1], [None, model_data_2])

    def test_handles_empty_data(self):
        data = []
        result = list(full_join_by_model_info(data))
        self.assertEqual(result, [])

    def test_handles_missing_model_data(self):
        model_info_1 = ModelInfo('framework1', 'model1', 'fp32', 'config1')
        model_info_2 = ModelInfo('framework2', 'model2', 'fp16', 'config2')
        model_data_1 = ModelData()
        data = [{model_info_1: model_data_1}, {}]

        result = list(full_join_by_model_info(data))

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][0], model_info_1)
        self.assertEqual(result[0][1], [model_data_1, None])


if __name__ == '__main__':
    unittest.main()


class TestUnit(unittest.TestCase):
    def test_initializes_correctly(self):
        csv_item = CSVItem('GPU', 'path/to/model', 'model_name', 'framework', 'precision', '', '1', 'transformation', 'transformation_name', '', '10.0', '1')
        unit = Unit(csv_item)
        self.assertEqual(unit.device, 'GPU')
        self.assertEqual(unit.name, 'transformation_name')
        self.assertEqual(unit.model_path, 'path/to/model')
        self.assertEqual(unit.model_framework, 'framework')
        self.assertEqual(unit.model_precision, 'precision')
        self.assertEqual(unit.type, 'transformation')
        self.assertEqual(unit.transformation_name, 'transformation_name')
        self.assertEqual(unit.manager_name, '')
        self.assertEqual(unit.get_durations(), [10.0])

    def test_calculates_duration_median_correctly(self):
        csv_item = CSVItem('GPU', 'path/to/model', 'model_name', 'framework', 'precision', '', '1', 'transformation', 'transformation_name', '', '10.0', '1')
        unit = Unit(csv_item)
        csv_item = CSVItem('GPU', 'path/to/model', 'model_name', 'framework', 'precision', '', '1', 'transformation', 'transformation_name', '', '15.0', '1')
        unit.add(csv_item)
        csv_item = CSVItem('GPU', 'path/to/model', 'model_name', 'framework', 'precision', '', '1', 'transformation', 'transformation_name', '', '23.0', '1')
        unit.add(csv_item)
        self.assertEqual(unit.get_duration_median(), 15.0)

    def test_calculates_deviations_correctly(self):
        csv_item = CSVItem('GPU', 'path/to/model', 'model_name', 'framework', 'precision', '', '1', 'transformation', 'transformation_name', '', '10.0', '1')
        unit = Unit(csv_item)
        self.assertEqual(unit.get_deviations(), [0.0])

    def test_calculates_duration_stddev_correctly(self):
        csv_item = CSVItem('GPU', 'path/to/model', 'model_name', 'framework', 'precision', '', '1', 'transformation', 'transformation_name', '', '10.0', '1')
        unit = Unit(csv_item)
        self.assertEqual(unit.get_duration_stddev(), 0.0)

    def test_calculates_variations_as_ratio_correctly(self):
        csv_item = CSVItem('GPU', 'path/to/model', 'model_name', 'framework', 'precision', '', '1', 'transformation', 'transformation_name', '', '10.0', '1')
        unit = Unit(csv_item)
        self.assertEqual(unit.get_variations_as_ratio(), [0.0])

    def test_adds_csv_item_correctly(self):
        csv_item1 = CSVItem('GPU', 'path/to/model', 'model_name', 'framework', 'precision', '', '1', 'transformation', 'transformation_name', '', '10.0', '1')
        csv_item2 = CSVItem('GPU', 'path/to/model', 'model_name', 'framework', 'precision', '', '2', 'transformation', 'transformation_name', '', '20.0', '1')
        unit = Unit(csv_item1)
        unit.add(csv_item2)
        self.assertEqual(unit.get_durations(), [10.0, 20.0])

    def test_uses_only_first_iter_correctly(self):
        Unit.USE_ONLY_0_ITER_GPU = True
        csv_item = CSVItem('GPU', 'path/to/model', 'model_name', 'framework', 'precision', '', '1', 'transformation', 'transformation_name', '', '10.0', '1')
        unit = Unit(csv_item)
        self.assertTrue(unit.use_only_first_iter())
        Unit.USE_ONLY_0_ITER_GPU = False

    def test_gpu_first_iter_correctly(self):
        Unit.USE_ONLY_0_ITER_GPU = True
        csv_item = CSVItem('GPU', 'path/to/model', 'model_name', 'framework', 'precision', '', '1', 'transformation', 'transformation_name', '', '10.0', '1')
        unit = Unit(csv_item)
        csv_item = CSVItem('GPU', 'path/to/model', 'model_name', 'framework', 'precision', '', '2', 'transformation', 'transformation_name', '', '15.0', '1')
        unit.add(csv_item)
        self.assertEqual(unit.get_durations(), [10.0])


class TestModelData(unittest.TestCase):
    def test_appends_first_iteration_correctly(self):
        csv_item = CSVItem('GPU', 'path/to/model', 'model_name', 'framework', 'precision', '', '1', 'transformation', 'transformation_name', '', '10.0', '1')
        model_data = ModelData()
        model_data.append(csv_item)
        self.assertEqual(len(model_data.items), 1)
        self.assertEqual(model_data.items[0].get_durations(), [10.0])

    def test_appends_subsequent_iterations_correctly(self):
        csv_item1 = CSVItem('GPU', 'path/to/model', 'model_name', 'framework', 'precision', '', '1', 'transformation', 'transformation_name', '', '10.0', '1')
        csv_item2 = CSVItem('GPU', 'path/to/model', 'model_name', 'framework', 'precision', '', '2', 'transformation', 'transformation_name', '', '20.0', '1')
        model_data = ModelData()
        model_data.append(csv_item1)
        model_data.append(csv_item2)
        self.assertEqual(len(model_data.items), 1)
        self.assertEqual(model_data.items[0].get_durations(), [10.0, 20.0])

    def test_raises_error_on_non_ascending_iterations(self):
        csv_item1 = CSVItem('GPU', 'path/to/model', 'model_name', 'framework', 'precision', '', '2', 'transformation', 'transformation_name', '', '20.0', '1')
        csv_item2 = CSVItem('GPU', 'path/to/model', 'model_name', 'framework', 'precision', '', '1', 'transformation', 'transformation_name', '', '10.0', '1')
        model_data = ModelData()
        with self.assertRaises(AssertionError):
            model_data.append(csv_item1)
            model_data.append(csv_item2)

    def test_gets_device_correctly(self):
        csv_item = CSVItem('GPU', 'path/to/model', 'model_name', 'framework', 'precision', '', '1', 'transformation', 'transformation_name', '', '10.0', '1')
        model_data = ModelData()
        model_data.append(csv_item)
        self.assertEqual(model_data.get_device(), 'GPU')

    def test_collects_items_by_type_correctly(self):
        csv_item1 = CSVItem('GPU', 'path/to/model', 'model_name', 'framework', 'precision', '', '1', 'transformation', 'transformation_name', '', '10.0', '1')
        csv_item2 = CSVItem('GPU', 'path/to/model', 'model_name', 'framework', 'precision', '', '2', 'transformation', 'transformation_name', '', '20.0', '1')
        csv_item3 = CSVItem('GPU', 'path/to/model', 'model_name', 'framework', 'precision', '', '1', 'manager', 'transformation_name', '', '10.0', '1')
        model_data = ModelData()
        model_data.append(csv_item1)
        model_data.append(csv_item2)
        items_by_type = model_data.collect_items_by_type('transformation')
        self.assertEqual(len(items_by_type['transformation_name']), 1)
        self.assertEqual(items_by_type['transformation_name'][0].get_durations(), [10.0, 20.0])

    def test_calculates_compile_time_correctly(self):
        csv_item1 = CSVItem('GPU', 'path/to/model', 'model_name', 'framework', 'precision', '', '1', 'compile_time', '', '', '10.0', '1')
        csv_item2 = CSVItem('GPU', 'path/to/model', 'model_name', 'framework', 'precision', '', '2', 'compile_time', '', '', '11.0', '1')
        model_data = ModelData()
        model_data.append(csv_item1)
        model_data.append(csv_item2)
        self.assertEqual(model_data.get_compile_time(), 10.5)

    def test_sums_transformation_time_correctly(self):
        csv_item1 = CSVItem('GPU', 'path/to/model', 'model_name', 'framework', 'precision', '', '1', 'transformation', 'transformation_name1', '', '10.0', '1')
        csv_item2 = CSVItem('GPU', 'path/to/model', 'model_name', 'framework', 'precision', '', '1', 'transformation', 'transformation_name2', '', '20.0', '1')
        csv_item3 = CSVItem('GPU', 'path/to/model', 'model_name', 'framework', 'precision', '', '2', 'transformation', 'transformation_name1', '', '11.0', '1')
        csv_item4 = CSVItem('GPU', 'path/to/model', 'model_name', 'framework', 'precision', '', '2', 'transformation', 'transformation_name2', '', '21.0', '1')
        model_data = ModelData()
        model_data.append(csv_item1)
        model_data.append(csv_item2)
        model_data.append(csv_item3)
        model_data.append(csv_item4)
        self.assertEqual(model_data.sum_transformation_time(), 31.0)

    def test_gets_compile_durations_correctly(self):
        csv_item = CSVItem('GPU', 'path/to/model', 'model_name', 'framework', 'precision', '', '1', 'compile_time', '', '', '10.0', '1')
        model_data = ModelData()
        model_data.append(csv_item)
        self.assertEqual(model_data.get_compile_durations(), [10.0])

    def test_gets_duration_correctly(self):
        csv_item = CSVItem('GPU', 'path/to/model', 'model_name', 'framework', 'precision', '', '1', 'transformation', 'transformation_name', '', '10.0', '1')
        model_data = ModelData()
        model_data.append(csv_item)
        self.assertEqual(model_data.get_duration(0), 10.0)

    def test_gets_all_item_info_correctly(self):
        csv_item = CSVItem('GPU', 'path/to/model', 'model_name', 'framework', 'precision', '', '1', 'transformation', 'transformation_name', '', '10.0', '1')
        model_data = ModelData()
        model_data.append(csv_item)
        item_info = model_data.get_all_item_info()
        self.assertEqual(len(item_info), 1)
        self.assertEqual(item_info[0].type, 'transformation')

    def test_gets_item_info_correctly(self):
        csv_item = CSVItem('GPU', 'path/to/model', 'model_name', 'framework', 'precision', '', '1', 'transformation', 'transformation_name', '', '10.0', '1')
        model_data = ModelData()
        model_data.append(csv_item)
        item_info = model_data.get_item_info(0)
        self.assertEqual(item_info.type, 'transformation')

    def test_gets_n_iterations_correctly(self):
        csv_item1 = CSVItem('GPU', 'path/to/model', 'model_name', 'framework', 'precision', '', '1', 'transformation', 'transformation_name', '', '10.0', '1')
        csv_item2 = CSVItem('GPU', 'path/to/model', 'model_name', 'framework', 'precision', '', '2', 'transformation', 'transformation_name', '', '20.0', '1')
        model_data = ModelData()
        model_data.append(csv_item1)
        model_data.append(csv_item2)
        self.assertEqual(model_data.get_n_iterations(), 2)

    def test_manager_plain_sequence_correct_order(self):
        csv_items = [
            CSVItem('GPU', 'path1', 'name1', 'framework1', 'precision1', 'config1', '1', 'manager_start', 'transformation1', 'manager1', '10.0', True),
            CSVItem('GPU', 'path1', 'name1', 'framework1', 'precision1', 'config1', '1', 'manager_end', 'transformation1', 'manager1', '20.0', True)
        ]
        model_data = ModelData()
        for item in csv_items:
            model_data.append(item)
        result = model_data.get_manager_plain_sequence()
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][0].manager_name, 'manager1')
        self.assertEqual(result[0][1].manager_name, 'manager1')

    def test_manager_plain_sequence_missing_start(self):
        csv_items = [
            CSVItem('GPU', 'path1', 'name1', 'framework1', 'precision1', 'config1', '1', 'manager_end', 'transformation1', 'manager1', '20.0', True)
        ]
        model_data = ModelData()
        for item in csv_items:
            model_data.append(item)
        with self.assertRaises(AssertionError):
            model_data.get_manager_plain_sequence()

    def test_manager_plain_sequence_missing_end(self):
        csv_items = [
            CSVItem('GPU', 'path1', 'name1', 'framework1', 'precision1', 'config1', '1', 'manager_start', 'transformation1', 'manager1', '10.0', True)
        ]
        model_data = ModelData()
        for item in csv_items:
            model_data.append(item)
        result = model_data.get_manager_plain_sequence()
        self.assertEqual(len(result), 0)

    def test_manager_plain_sequence_different_names(self):
        csv_items = [
            CSVItem('GPU', 'path1', 'name1', 'framework1', 'precision1', 'config1', '1', 'manager_start', 'transformation1', 'manager1', '10.0', True),
            CSVItem('GPU', 'path1', 'name1', 'framework1', 'precision1', 'config1', '1', 'manager_end', 'transformation1', 'manager2', '20.0', True)
        ]
        model_data = ModelData()
        for item in csv_items:
            model_data.append(item)
        with self.assertRaises(AssertionError):
            model_data.get_manager_plain_sequence()

    def test_manager_plain_sequence_multiple_managers(self):
        csv_items = [
            CSVItem('GPU', 'path1', 'name1', 'framework1', 'precision1', 'config1', '1', 'manager_start', 'transformation1', 'manager1', '10.0', True),
            CSVItem('GPU', 'path1', 'name1', 'framework1', 'precision1', 'config1', '1', 'manager_end', 'transformation1', 'manager1', '20.0', True),
            CSVItem('GPU', 'path1', 'name1', 'framework1', 'precision1', 'config1', '1', 'manager_start', 'transformation1', 'manager2', '30.0', True),
            CSVItem('GPU', 'path1', 'name1', 'framework1', 'precision1', 'config1', '1', 'manager_end', 'transformation1', 'manager2', '40.0', True)
        ]
        model_data = ModelData()
        for item in csv_items:
            model_data.append(item)
        result = model_data.get_manager_plain_sequence()
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0][0].manager_name, 'manager1')
        self.assertEqual(result[0][1].manager_name, 'manager1')
        self.assertEqual(result[1][0].manager_name, 'manager2')
        self.assertEqual(result[1][1].manager_name, 'manager2')

    def test_manager_plain_sequence_interleaved_managers(self):
        csv_items = [
            CSVItem('GPU', 'path1', 'name1', 'framework1', 'precision1', 'config1', '1', 'manager_start', 'transformation1', 'manager1', '10.0', True),
            CSVItem('GPU', 'path1', 'name1', 'framework1', 'precision1', 'config1', '1', 'manager_start', 'transformation1', 'manager2', '20.0', True),
            CSVItem('GPU', 'path1', 'name1', 'framework1', 'precision1', 'config1', '1', 'manager_end', 'transformation1', 'manager2', '30.0', True),
            CSVItem('GPU', 'path1', 'name1', 'framework1', 'precision1', 'config1', '1', 'manager_end', 'transformation1', 'manager1', '40.0', True)
        ]
        model_data = ModelData()
        for item in csv_items:
            model_data.append(item)
        result = model_data.get_manager_plain_sequence()
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][0].manager_name, 'manager1')
        self.assertEqual(result[0][0].get_durations(), [10.0])
        self.assertEqual(result[0][1].get_durations(), [40.0])

    def test_manager_plain_sequence_no_managers(self):
        csv_items = [
            CSVItem('GPU', 'path1', 'name1', 'framework1', 'precision1', 'config1', '1', 'transformation', 'transformation1', 'manager1', '10.0', True),
            CSVItem('GPU', 'path1', 'name1', 'framework1', 'precision1', 'config1', '2', 'transformation', 'transformation1', 'manager1', '20.0', True)
        ]
        model_data = ModelData()
        for item in csv_items:
            model_data.append(item)
        result = model_data.get_manager_plain_sequence()
        self.assertEqual(len(result), 0)

    def test_manager_plain_sequence_median_sum_single_pair(self):
        csv_items = [
            CSVItem('GPU', 'path1', 'name1', 'framework1', 'precision1', 'config1', '1', 'manager_start', 'transformation1', 'manager1', '10.0', True),
            CSVItem('GPU', 'path1', 'name1', 'framework1', 'precision1', 'config1', '1', 'manager_end', 'transformation1', 'manager1', '20.0', True)
        ]
        model_data = ModelData()
        for item in csv_items:
            model_data.append(item)
        result = model_data.get_manager_plain_sequence_median_sum()
        self.assertEqual(result, 10.0)

    def test_manager_plain_sequence_median_sum_multiple_pairs(self):
        csv_items = [
            CSVItem('GPU', 'path1', 'name1', 'framework1', 'precision1', 'config1', '1', 'manager_start', 'transformation1', 'manager1', '10.0', True),
            CSVItem('GPU', 'path1', 'name1', 'framework1', 'precision1', 'config1', '1', 'manager_end', 'transformation1', 'manager1', '20.0', True),
            CSVItem('GPU', 'path1', 'name1', 'framework1', 'precision1', 'config1', '1', 'manager_start', 'transformation1', 'manager2', '30.0', True),
            CSVItem('GPU', 'path1', 'name1', 'framework1', 'precision1', 'config1', '1', 'manager_end', 'transformation1', 'manager2', '50.0', True)
        ]
        model_data = ModelData()
        for item in csv_items:
            model_data.append(item)
        result = model_data.get_manager_plain_sequence_median_sum()
        self.assertEqual(result, 30.0)

    def test_manager_plain_sequence_median_sum_no_pairs(self):
        csv_items = [
            CSVItem('GPU', 'path1', 'name1', 'framework1', 'precision1', 'config1', '1', 'manager_start', 'transformation1', 'manager1', '10.0', True)
        ]
        model_data = ModelData()
        for item in csv_items:
            model_data.append(item)
        result = model_data.get_manager_plain_sequence_median_sum()
        self.assertEqual(result, 0.0)

    def test_manager_plain_sequence_median_sum_interleaved_pairs(self):
        csv_items = [
            CSVItem('GPU', 'path1', 'name1', 'framework1', 'precision1', 'config1', '1', 'manager_start', 'transformation1', 'manager1', '10.0', True),
            CSVItem('GPU', 'path1', 'name1', 'framework1', 'precision1', 'config1', '1', 'manager_start', 'transformation1', 'manager2', '20.0', True),
            CSVItem('GPU', 'path1', 'name1', 'framework1', 'precision1', 'config1', '1', 'manager_end', 'transformation1', 'manager2', '30.0', True),
            CSVItem('GPU', 'path1', 'name1', 'framework1', 'precision1', 'config1', '1', 'manager_end', 'transformation1', 'manager1', '40.0', True)
        ]
        model_data = ModelData()
        for item in csv_items:
            model_data.append(item)
        result = model_data.get_manager_plain_sequence_median_sum()
        self.assertEqual(result, 30.0)

    def test_manager_plain_sequence_names_single_pair(self):
        csv_items = [
            CSVItem('GPU', 'path1', 'name1', 'framework1', 'precision1', 'config1', '1', 'manager_start', 'transformation1', 'manager1', '10.0', True),
            CSVItem('GPU', 'path1', 'name1', 'framework1', 'precision1', 'config1', '1', 'manager_end', 'transformation1', 'manager1', '20.0', True)
        ]
        model_data = ModelData()
        for item in csv_items:
            model_data.append(item)
        result = model_data.get_manager_plain_sequence_names()
        self.assertEqual(result, ['manager1'])

    def test_manager_plain_sequence_names_multiple_pairs(self):
        csv_items = [
            CSVItem('GPU', 'path1', 'name1', 'framework1', 'precision1', 'config1', '1', 'manager_start', 'transformation1', 'manager1', '10.0', True),
            CSVItem('GPU', 'path1', 'name1', 'framework1', 'precision1', 'config1', '1', 'manager_end', 'transformation1', 'manager1', '20.0', True),
            CSVItem('GPU', 'path1', 'name1', 'framework1', 'precision1', 'config1', '1', 'manager_start', 'transformation1', 'manager2', '30.0', True),
            CSVItem('GPU', 'path1', 'name1', 'framework1', 'precision1', 'config1', '1', 'manager_end', 'transformation1', 'manager2', '40.0', True)
        ]
        model_data = ModelData()
        for item in csv_items:
            model_data.append(item)
        result = model_data.get_manager_plain_sequence_names()
        self.assertEqual(result, ['manager1', 'manager2'])

    def test_manager_plain_sequence_names_no_pairs(self):
        csv_items = [
            CSVItem('GPU', 'path1', 'name1', 'framework1', 'precision1', 'config1', '1', 'manager_start', 'transformation1', 'manager1', '10.0', True)
        ]
        model_data = ModelData()
        for item in csv_items:
            model_data.append(item)
        result = model_data.get_manager_plain_sequence_names()
        self.assertEqual(result, [])

    def test_manager_plain_sequence_names_interleaved_pairs(self):
        csv_items = [
            CSVItem('GPU', 'path1', 'name1', 'framework1', 'precision1', 'config1', '1', 'manager_start', 'transformation1', 'manager1', '10.0', True),
            CSVItem('GPU', 'path1', 'name1', 'framework1', 'precision1', 'config1', '1', 'manager_start', 'transformation1', 'manager2', '20.0', True),
            CSVItem('GPU', 'path1', 'name1', 'framework1', 'precision1', 'config1', '1', 'manager_end', 'transformation1', 'manager2', '30.0', True),
            CSVItem('GPU', 'path1', 'name1', 'framework1', 'precision1', 'config1', '1', 'manager_end', 'transformation1', 'manager1', '40.0', True)
        ]
        model_data = ModelData()
        for item in csv_items:
            model_data.append(item)
        result = model_data.get_manager_plain_sequence_names()
        self.assertEqual(result, ['manager1'])

    def test_manager_plain_sequence_median_gap_sum_single_pair(self):
        csv_items = [
            CSVItem('GPU', 'path1', 'name1', 'framework1', 'precision1', 'config1', '1', 'manager_start', 'transformation1', 'manager1', '10.0', True),
            CSVItem('GPU', 'path1', 'name1', 'framework1', 'precision1', 'config1', '1', 'manager_end', 'transformation1', 'manager1', '20.0', True)
        ]
        model_data = ModelData()
        for item in csv_items:
            model_data.append(item)
        result = model_data.get_manager_plain_sequence_median_gap_sum()
        self.assertEqual(result, 0.0)

    def test_manager_plain_sequence_median_gap_sum_multiple_pairs(self):
        csv_items = [
            CSVItem('GPU', 'path1', 'name1', 'framework1', 'precision1', 'config1', '1', 'manager_start', 'transformation1', 'manager1', '10.0', True),
            CSVItem('GPU', 'path1', 'name1', 'framework1', 'precision1', 'config1', '1', 'manager_end', 'transformation1', 'manager1', '20.0', True),
            CSVItem('GPU', 'path1', 'name1', 'framework1', 'precision1', 'config1', '1', 'manager_start', 'transformation1', 'manager2', '30.0', True),
            CSVItem('GPU', 'path1', 'name1', 'framework1', 'precision1', 'config1', '1', 'manager_end', 'transformation1', 'manager2', '50.0', True)
        ]
        model_data = ModelData()
        for item in csv_items:
            model_data.append(item)
        result = model_data.get_manager_plain_sequence_median_gap_sum()
        self.assertEqual(result, 10.0)

    def test_manager_plain_sequence_median_gap_sum_no_pairs(self):
        csv_items = [
            CSVItem('GPU', 'path1', 'name1', 'framework1', 'precision1', 'config1', '1', 'manager_start', 'transformation1', 'manager1', '10.0', True)
        ]
        model_data = ModelData()
        for item in csv_items:
            model_data.append(item)
        result = model_data.get_manager_plain_sequence_median_gap_sum()
        self.assertEqual(result, 0.0)

    def test_manager_plain_sequence_median_gap_sum_interleaved_pairs(self):
        csv_items = [
            CSVItem('GPU', 'path1', 'name1', 'framework1', 'precision1', 'config1', '1', 'manager_start', 'transformation1', 'manager1', '10.0', True),
            CSVItem('GPU', 'path1', 'name1', 'framework1', 'precision1', 'config1', '1', 'manager_start', 'transformation1', 'manager2', '20.0', True),
            CSVItem('GPU', 'path1', 'name1', 'framework1', 'precision1', 'config1', '1', 'manager_end', 'transformation1', 'manager2', '30.0', True),
            CSVItem('GPU', 'path1', 'name1', 'framework1', 'precision1', 'config1', '1', 'manager_end', 'transformation1', 'manager1', '40.0', True)
        ]
        model_data = ModelData()
        for item in csv_items:
            model_data.append(item)
        result = model_data.get_manager_plain_sequence_median_gap_sum()
        self.assertEqual(result, 0.0)


class TestMakeModelConsoleDescription(unittest.TestCase):

    def test_make_model_console_description_with_config(self):
        model_info = ModelInfo('TensorFlow', 'ModelA', 'FP32', 'config1')
        result = make_model_console_description(model_info)
        self.assertEqual(result, 'TensorFlow ModelA FP32 config1')

    def test_make_model_console_description_without_config(self):
        model_info = ModelInfo('TensorFlow', 'ModelA', 'FP32', '')
        result = make_model_console_description(model_info)
        self.assertEqual(result, 'TensorFlow ModelA FP32')

    def test_make_model_console_description_with_special_characters(self):
        model_info = ModelInfo('TensorFlow', 'ModelA', 'FP32', 'config@1')
        result = make_model_console_description(model_info)
        self.assertEqual(result, 'TensorFlow ModelA FP32 config@1')

    def test_make_model_console_description_with_empty_fields(self):
        model_info = ModelInfo('', '', '', '')
        result = make_model_console_description(model_info)
        self.assertEqual(result, '  ')


if __name__ == "__main__":
    unittest.main()
