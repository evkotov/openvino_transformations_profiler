import unittest
import os
from collections import namedtuple

from compare_csv import make_model_file_name, make_model_console_description, get_common_models, CSVOutput
from parse_input import CSVItem, get_csv_header, read_csv, remove_invalid_items, get_config_value_from_path, is_header_valid
from common_structs import Unit, ModelData, ModelInfo


class TestUnit(unittest.TestCase):
    def test_initializes_correctly(self):
        csv_item = CSVItem('GPU', 'path/to/model', 'model_name', 'framework', 'precision', '', '1', 'transformation', 'transformation_name', '', '10.0')
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
        csv_item = CSVItem('GPU', 'path/to/model', 'model_name', 'framework', 'precision', '', '1', 'transformation', 'transformation_name', '', '10.0')
        unit = Unit(csv_item)
        csv_item = CSVItem('GPU', 'path/to/model', 'model_name', 'framework', 'precision', '', '1', 'transformation', 'transformation_name', '', '15.0')
        unit.add(csv_item)
        csv_item = CSVItem('GPU', 'path/to/model', 'model_name', 'framework', 'precision', '', '1', 'transformation', 'transformation_name', '', '23.0')
        unit.add(csv_item)
        self.assertEqual(unit.get_duration_median(), 15.0)

    def test_calculates_deviations_correctly(self):
        csv_item = CSVItem('GPU', 'path/to/model', 'model_name', 'framework', 'precision', '', '1', 'transformation', 'transformation_name', '', '10.0')
        unit = Unit(csv_item)
        self.assertEqual(unit.get_deviations(), [0.0])

    def test_calculates_duration_stddev_correctly(self):
        csv_item = CSVItem('GPU', 'path/to/model', 'model_name', 'framework', 'precision', '', '1', 'transformation', 'transformation_name', '', '10.0')
        unit = Unit(csv_item)
        self.assertEqual(unit.get_duration_stddev(), 0.0)

    def test_calculates_variations_as_ratio_correctly(self):
        csv_item = CSVItem('GPU', 'path/to/model', 'model_name', 'framework', 'precision', '', '1', 'transformation', 'transformation_name', '', '10.0')
        unit = Unit(csv_item)
        self.assertEqual(unit.get_variations_as_ratio(), [0.0])

    def test_adds_csv_item_correctly(self):
        csv_item1 = CSVItem('GPU', 'path/to/model', 'model_name', 'framework', 'precision', '', '1', 'transformation', 'transformation_name', '', '10.0')
        csv_item2 = CSVItem('GPU', 'path/to/model', 'model_name', 'framework', 'precision', '', '2', 'transformation', 'transformation_name', '', '20.0')
        unit = Unit(csv_item1)
        unit.add(csv_item2)
        self.assertEqual(unit.get_durations(), [10.0, 20.0])

    def test_uses_only_first_iter_correctly(self):
        Unit.USE_ONLY_0_ITER_GPU = True
        csv_item = CSVItem('GPU', 'path/to/model', 'model_name', 'framework', 'precision', '', '1', 'transformation', 'transformation_name', '', '10.0')
        unit = Unit(csv_item)
        self.assertTrue(unit.use_only_first_iter())
        Unit.USE_ONLY_0_ITER_GPU = False

    def test_gpu_first_iter_correctly(self):
        Unit.USE_ONLY_0_ITER_GPU = True
        csv_item = CSVItem('GPU', 'path/to/model', 'model_name', 'framework', 'precision', '', '1', 'transformation', 'transformation_name', '', '10.0')
        unit = Unit(csv_item)
        csv_item = CSVItem('GPU', 'path/to/model', 'model_name', 'framework', 'precision', '', '2', 'transformation', 'transformation_name', '', '15.0')
        unit.add(csv_item)
        self.assertEqual(unit.get_durations(), [10.0])


class TestModelData(unittest.TestCase):
    def test_appends_first_iteration_correctly(self):
        csv_item = CSVItem('GPU', 'path/to/model', 'model_name', 'framework', 'precision', '', '1', 'transformation', 'transformation_name', '', '10.0')
        model_data = ModelData()
        model_data.append(csv_item)
        self.assertEqual(len(model_data.items), 1)
        self.assertEqual(model_data.items[0].get_durations(), [10.0])

    def test_appends_subsequent_iterations_correctly(self):
        csv_item1 = CSVItem('GPU', 'path/to/model', 'model_name', 'framework', 'precision', '', '1', 'transformation', 'transformation_name', '', '10.0')
        csv_item2 = CSVItem('GPU', 'path/to/model', 'model_name', 'framework', 'precision', '', '2', 'transformation', 'transformation_name', '', '20.0')
        model_data = ModelData()
        model_data.append(csv_item1)
        model_data.append(csv_item2)
        self.assertEqual(len(model_data.items), 1)
        self.assertEqual(model_data.items[0].get_durations(), [10.0, 20.0])

    def test_raises_error_on_non_ascending_iterations(self):
        csv_item1 = CSVItem('GPU', 'path/to/model', 'model_name', 'framework', 'precision', '', '2', 'transformation', 'transformation_name', '', '20.0')
        csv_item2 = CSVItem('GPU', 'path/to/model', 'model_name', 'framework', 'precision', '', '1', 'transformation', 'transformation_name', '', '10.0')
        model_data = ModelData()
        with self.assertRaises(AssertionError):
            model_data.append(csv_item1)
            model_data.append(csv_item2)

    def test_gets_device_correctly(self):
        csv_item = CSVItem('GPU', 'path/to/model', 'model_name', 'framework', 'precision', '', '1', 'transformation', 'transformation_name', '', '10.0')
        model_data = ModelData()
        model_data.append(csv_item)
        self.assertEqual(model_data.get_device(), 'GPU')

    def test_collects_items_by_type_correctly(self):
        csv_item1 = CSVItem('GPU', 'path/to/model', 'model_name', 'framework', 'precision', '', '1', 'transformation', 'transformation_name', '', '10.0')
        csv_item2 = CSVItem('GPU', 'path/to/model', 'model_name', 'framework', 'precision', '', '2', 'transformation', 'transformation_name', '', '20.0')
        csv_item3 = CSVItem('GPU', 'path/to/model', 'model_name', 'framework', 'precision', '', '1', 'manager', 'transformation_name', '', '10.0')
        model_data = ModelData()
        model_data.append(csv_item1)
        model_data.append(csv_item2)
        items_by_type = model_data.collect_items_by_type('transformation')
        self.assertEqual(len(items_by_type['transformation_name']), 1)
        self.assertEqual(items_by_type['transformation_name'][0].get_durations(), [10.0, 20.0])

    def test_calculates_compile_time_correctly(self):
        csv_item1 = CSVItem('GPU', 'path/to/model', 'model_name', 'framework', 'precision', '', '1', 'compile_time', '', '', '10.0')
        csv_item2 = CSVItem('GPU', 'path/to/model', 'model_name', 'framework', 'precision', '', '2', 'compile_time', '', '', '11.0')
        model_data = ModelData()
        model_data.append(csv_item1)
        model_data.append(csv_item2)
        self.assertEqual(model_data.get_compile_time(), 10.5)

    def test_sums_transformation_time_correctly(self):
        csv_item1 = CSVItem('GPU', 'path/to/model', 'model_name', 'framework', 'precision', '', '1', 'transformation', 'transformation_name1', '', '10.0')
        csv_item2 = CSVItem('GPU', 'path/to/model', 'model_name', 'framework', 'precision', '', '1', 'transformation', 'transformation_name2', '', '20.0')
        csv_item3 = CSVItem('GPU', 'path/to/model', 'model_name', 'framework', 'precision', '', '2', 'transformation', 'transformation_name1', '', '11.0')
        csv_item4 = CSVItem('GPU', 'path/to/model', 'model_name', 'framework', 'precision', '', '2', 'transformation', 'transformation_name2', '', '21.0')
        model_data = ModelData()
        model_data.append(csv_item1)
        model_data.append(csv_item2)
        model_data.append(csv_item3)
        model_data.append(csv_item4)
        self.assertEqual(model_data.sum_transformation_time(), 31.0)

    def test_gets_compile_durations_correctly(self):
        csv_item = CSVItem('GPU', 'path/to/model', 'model_name', 'framework', 'precision', '', '1', 'compile_time', '', '', '10.0')
        model_data = ModelData()
        model_data.append(csv_item)
        self.assertEqual(model_data.get_compile_durations(), [10.0])

    def test_gets_duration_correctly(self):
        csv_item = CSVItem('GPU', 'path/to/model', 'model_name', 'framework', 'precision', '', '1', 'transformation', 'transformation_name', '', '10.0')
        model_data = ModelData()
        model_data.append(csv_item)
        self.assertEqual(model_data.get_duration(0), 10.0)

    def test_gets_all_item_info_correctly(self):
        csv_item = CSVItem('GPU', 'path/to/model', 'model_name', 'framework', 'precision', '', '1', 'transformation', 'transformation_name', '', '10.0')
        model_data = ModelData()
        model_data.append(csv_item)
        item_info = model_data.get_all_item_info()
        self.assertEqual(len(item_info), 1)
        self.assertEqual(item_info[0].type, 'transformation')

    def test_gets_item_info_correctly(self):
        csv_item = CSVItem('GPU', 'path/to/model', 'model_name', 'framework', 'precision', '', '1', 'transformation', 'transformation_name', '', '10.0')
        model_data = ModelData()
        model_data.append(csv_item)
        item_info = model_data.get_item_info(0)
        self.assertEqual(item_info.type, 'transformation')

    def test_gets_n_iterations_correctly(self):
        csv_item1 = CSVItem('GPU', 'path/to/model', 'model_name', 'framework', 'precision', '', '1', 'transformation', 'transformation_name', '', '10.0')
        csv_item2 = CSVItem('GPU', 'path/to/model', 'model_name', 'framework', 'precision', '', '2', 'transformation', 'transformation_name', '', '20.0')
        model_data = ModelData()
        model_data.append(csv_item1)
        model_data.append(csv_item2)
        self.assertEqual(model_data.get_n_iterations(), 2)


class TestCSVHeader(unittest.TestCase):
    def tearDown(self):
        files_to_remove = ['test.csv', 'empty.csv', 'header_only.csv', 'no_newline.csv']
        for file in files_to_remove:
            if os.path.exists(file):
                os.remove(file)

    def test_reads_header_correctly(self):
        with open('test.csv', 'w') as f:
            f.write('col1;col2;col3\n')
        self.assertEqual(get_csv_header('test.csv'), ['col1', 'col2', 'col3'])

    def test_raises_error_on_empty_file(self):
        with open('empty.csv', 'w') as f:
            pass
        with self.assertRaises(StopIteration):
            get_csv_header('empty.csv')

    def test_handles_file_with_only_header(self):
        with open('header_only.csv', 'w') as f:
            f.write('col1;col2;col3\n')
        self.assertEqual(get_csv_header('header_only.csv'), ['col1', 'col2', 'col3'])

    def test_handles_file_with_no_newline_at_end(self):
        with open('no_newline.csv', 'w') as f:
            f.write('col1;col2;col3')
        self.assertEqual(get_csv_header('no_newline.csv'), ['col1', 'col2', 'col3'])


class TestReadCSV(unittest.TestCase):
    def tearDown(self):
        files_to_remove = ['test_optional.csv', 'test_no_optional.csv', 'test_device.csv', 'test_no_device.csv', 'test_invalid.csv', 'missing_config.csv']
        for file in files_to_remove:
            if os.path.exists(file):
                os.remove(file)

    def test_reads_csv_with_optional_model_attr_correctly(self):
        with open('test_optional.csv', 'w') as f:
            f.write('device;model_path;model_name;model_framework;model_precision;optional_model_attribute;iteration;type;transformation_name;manager_name;duration\n')
            f.write('GPU;path/to/model;model_name;framework;precision;attr;1;transformation;transformation_name;;10.0\n')
        csv_items = list(read_csv('test_optional.csv'))
        self.assertEqual(len(csv_items), 1)
        self.assertEqual(csv_items[0].config, 'attr')

    def test_read_csv_with_missing_config_from_path(self):
        path = "missing_config.csv"
        with open(path, "w") as f:
            f.write("device;model_path;model_name;model_framework;model_precision;iteration;type;transformation_name;manager_name;duration\n")
            f.write("CPU;/path/to/model;model1;framework1;FP32;1;transformation;transformation1;manager1;0.1\n")
        result = list(read_csv(path))
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].config, "to")

    def test_reads_csv_without_optional_model_attr_correctly(self):
        with open('test_no_optional.csv', 'w') as f:
            f.write('device;model_path;model_name;model_framework;model_precision;iteration;type;transformation_name;manager_name;duration\n')
            f.write('GPU;path_to_model;model_name;framework;precision;1;transformation;transformation_name;;10.0\n')
        csv_items = list(read_csv('test_no_optional.csv'))
        self.assertEqual(len(csv_items), 1)
        self.assertEqual(csv_items[0].config, '')

    def test_reads_csv_with_device_correctly(self):
        with open('test_device.csv', 'w') as f:
            f.write('device;model_path;model_name;model_framework;model_precision;iteration;type;transformation_name;manager_name;duration\n')
            f.write('GPU;path/to/model;model_name;framework;precision;1;transformation;transformation_name;;10.0\n')
        csv_items = list(read_csv('test_device.csv'))
        self.assertEqual(len(csv_items), 1)
        self.assertEqual(csv_items[0].device, 'GPU')

    def test_reads_csv_without_device_correctly(self):
        with open('test_no_device.csv', 'w') as f:
            f.write('model_path;model_name;model_framework;model_precision;iteration;type;transformation_name;manager_name;duration\n')
            f.write('path/to/model;model_name;framework;precision;1;transformation;transformation_name;;10.0\n')
        csv_items = list(read_csv('test_no_device.csv'))
        self.assertEqual(len(csv_items), 1)
        self.assertEqual(csv_items[0].device, 'N/A')

    def test_header_with_all_required_columns(self):
        column_names = ['device', 'model_path', 'model_name', 'model_framework', 'model_precision', 'config', 'iteration', 'type', 'transformation_name', 'manager_name', 'duration']
        self.assertTrue(is_header_valid(column_names))

    def test_header_with_optional_columns(self):
        column_names = ['device', 'model_path', 'model_name', 'model_framework', 'model_precision', 'optional_model_attribute', 'iteration', 'type', 'transformation_name', 'manager_name', 'duration']
        self.assertTrue(is_header_valid(column_names))

    def test_header_with_weight_compression(self):
        column_names = ['device', 'model_path', 'model_name', 'model_framework', 'model_precision', 'weight_compression', 'iteration', 'type', 'transformation_name', 'manager_name', 'duration']
        self.assertTrue(is_header_valid(column_names))

    def test_header_with_missing_device(self):
        column_names = ['model_path', 'model_name', 'model_framework', 'model_precision', 'config', 'iteration', 'type', 'transformation_name', 'manager_name', 'duration']
        self.assertTrue(is_header_valid(column_names))

    def test_header_with_missing_config(self):
        column_names = ['device', 'model_path', 'model_name', 'model_framework', 'model_precision', 'iteration', 'type', 'transformation_name', 'manager_name', 'duration']
        self.assertTrue(is_header_valid(column_names))

    def test_header_with_extra_columns(self):
        column_names = ['device', 'model_path', 'model_name', 'model_framework', 'model_precision', 'config', 'iteration', 'type', 'transformation_name', 'manager_name', 'duration', 'extra_column']
        self.assertFalse(is_header_valid(column_names))

    def test_header_with_missing_required_columns(self):
        column_names = ['model_path', 'model_name', 'model_framework', 'model_precision', 'iteration', 'type', 'transformation_name', 'manager_name']
        self.assertFalse(is_header_valid(column_names))

    def test_header_with_incorrect_order(self):
        column_names = ['model_path', 'device', 'model_name', 'model_framework', 'model_precision', 'config', 'iteration', 'type', 'transformation_name', 'manager_name', 'duration']
        self.assertFalse(is_header_valid(column_names))


class TestGetConfigValueFromPath(unittest.TestCase):
    def test_config_value_from_valid_path(self):
        path = "/some/path/to/config"
        config_values_cache = {}
        expected_value = "to"
        self.assertEqual(get_config_value_from_path(path, config_values_cache), expected_value)

    def test_config_value_from_cached_path(self):
        path = "/cached/path/to/config"
        config_values_cache = {path: "cached_value"}
        self.assertEqual(get_config_value_from_path(path, config_values_cache), "cached_value")

    def test_config_value_from_root_path(self):
        path = "/"
        config_values_cache = {}
        expected_value = ""
        self.assertEqual(get_config_value_from_path(path, config_values_cache), expected_value)

    def test_config_value_from_empty_path(self):
        path = ""
        config_values_cache = {}
        expected_value = ""
        self.assertEqual(get_config_value_from_path(path, config_values_cache), expected_value)

    def test_config_value_from_single_component_path(self):
        path = "/single"
        config_values_cache = {}
        expected_value = ''
        self.assertEqual(get_config_value_from_path(path, config_values_cache), expected_value)


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


class TestCreateRatioStatsTable(unittest.TestCase):
    def create_ratio_stats_table(self, data):
        column_names = ['framework', 'name', 'precision', 'config', 'median', 'mean', 'std', 'max']
        table = []
        for model_info, ratio_stats in data.items():
            row = {'framework': model_info.framework, 'name': model_info.name, 'precision': model_info.precision,
                   'config': model_info.config, 'median': ratio_stats.median, 'mean': ratio_stats.mean,
                   'std': ratio_stats.std, 'max': ratio_stats.max}
            table.append(row)
        def get_ratio_max(row):
            return row['max']
        return column_names, sorted(table, key=get_ratio_max, reverse=True)

    def test_create_ratio_stats_table_with_valid_data(self):
        ModelInfo = namedtuple('ModelInfo', ['framework', 'name', 'precision', 'config'])
        RatioStats = namedtuple('RatioStats', ['median', 'mean', 'std', 'max'])
        data = {
            ModelInfo('TensorFlow', 'ModelA', 'FP32', 'config1'): RatioStats(10, 20, 5, 30),
            ModelInfo('PyTorch', 'ModelB', 'FP16', 'config2'): RatioStats(15, 25, 10, 35)
        }
        column_names, table = self.create_ratio_stats_table(data)
        self.assertEqual(column_names, ['framework', 'name', 'precision', 'config', 'median', 'mean', 'std', 'max'])
        self.assertEqual(len(table), 2)
        self.assertEqual(table[0]['framework'], 'PyTorch')
        self.assertEqual(table[1]['framework'], 'TensorFlow')

    def test_create_ratio_stats_table_with_empty_data(self):
        data = {}
        column_names, table = self.create_ratio_stats_table(data)
        self.assertEqual(column_names, ['framework', 'name', 'precision', 'config', 'median', 'mean', 'std', 'max'])
        self.assertEqual(len(table), 0)

    def test_create_ratio_stats_table_with_single_entry(self):
        ModelInfo = namedtuple('ModelInfo', ['framework', 'name', 'precision', 'config'])
        RatioStats = namedtuple('RatioStats', ['median', 'mean', 'std', 'max'])
        data = {
            ModelInfo('TensorFlow', 'ModelA', 'FP32', 'config1'): RatioStats(10, 20, 5, 30)
        }
        column_names, table = self.create_ratio_stats_table(data)
        self.assertEqual(column_names, ['framework', 'name', 'precision', 'config', 'median', 'mean', 'std', 'max'])
        self.assertEqual(len(table), 1)
        self.assertEqual(table[0]['framework'], 'TensorFlow')
        self.assertEqual(table[0]['name'], 'ModelA')
        self.assertEqual(table[0]['precision'], 'FP32')
        self.assertEqual(table[0]['config'], 'config1')
        self.assertEqual(table[0]['median'], 10)
        self.assertEqual(table[0]['mean'], 20)
        self.assertEqual(table[0]['std'], 5)
        self.assertEqual(table[0]['max'], 30)


class TestRemoveInvalidItems(unittest.TestCase):

    def test_remove_invalid_items_with_valid_data(self):
        ModelInfo = namedtuple('ModelInfo', ['framework', 'name', 'precision', 'config'])
        data = {
            ModelInfo('TensorFlow', 'ModelA', 'FP32', 'config1'): ModelData(),
            ModelInfo('PyTorch', 'ModelB', 'FP16', 'config2'): ModelData()
        }
        for model_data in data.values():
            model_data.check = lambda: True
        valid_data = remove_invalid_items(data)
        self.assertEqual(len(valid_data), 2)

    def test_remove_invalid_items_with_invalid_data(self):
        ModelInfo = namedtuple('ModelInfo', ['framework', 'name', 'precision', 'config'])
        data = {
            ModelInfo('TensorFlow', 'ModelA', 'FP32', 'config1'): ModelData(),
            ModelInfo('PyTorch', 'ModelB', 'FP16', 'config2'): ModelData()
        }
        data[ModelInfo('TensorFlow', 'ModelA', 'FP32', 'config1')].check = lambda: True
        data[ModelInfo('PyTorch', 'ModelB', 'FP16', 'config2')].check = lambda: (_ for _ in ()).throw(AssertionError)
        valid_data = remove_invalid_items(data)
        self.assertEqual(len(valid_data), 1)
        self.assertIn(ModelInfo('TensorFlow', 'ModelA', 'FP32', 'config1'), valid_data)

    def test_remove_invalid_items_with_empty_data(self):
        data = {}
        valid_data = remove_invalid_items(data)
        self.assertEqual(len(valid_data), 0)


class TestMakeModelFileName(unittest.TestCase):
    def test_make_model_file_name_with_all_fields(self):
        model_info = ModelInfo(framework='tensorflow', name='modelA', precision='fp32', config='config1')
        result = make_model_file_name('prefix', model_info, 'csv')
        self.assertEqual(result, 'prefix_tensorflow_modelA_fp32_config1.csv')

    def test_make_model_file_name_without_config(self):
        model_info = ModelInfo(framework='tensorflow', name='modelA', precision='fp32', config=None)
        result = make_model_file_name('prefix', model_info, 'csv')
        self.assertEqual(result, 'prefix_tensorflow_modelA_fp32.csv')

    def test_make_model_file_name_without_extension(self):
        model_info = ModelInfo(framework='tensorflow', name='modelA', precision='fp32', config='config1')
        result = make_model_file_name('prefix', model_info, '')
        self.assertEqual(result, 'prefix_tensorflow_modelA_fp32_config1')

    def test_make_model_file_name_with_empty_prefix(self):
        model_info = ModelInfo(framework='tensorflow', name='modelA', precision='fp32', config='config1')
        result = make_model_file_name('', model_info, 'csv')
        self.assertEqual(result, 'tensorflow_modelA_fp32_config1.csv')

    def test_make_model_file_name_with_empty_extension_and_prefix(self):
        model_info = ModelInfo(framework='tensorflow', name='modelA', precision='fp32', config='config1')
        result = make_model_file_name('', model_info, '')
        self.assertEqual(result, 'tensorflow_modelA_fp32_config1')


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


class TestCSVOutput(unittest.TestCase):

    def tearDown(self):
        if os.path.exists('test.csv'):
            os.remove('test.csv')

    def test_write_writes_rows_correctly(self):
        output = CSVOutput('test.csv', ['col1', 'col2'], None)
        rows = [{'col1': 'val1', 'col2': 'val2'}, {'col1': 'val3', 'col2': 'val4'}]
        with output:
            output.write(rows)
        with open('test.csv', 'r') as f:
            lines = f.readlines()
        self.assertEqual(lines[1].strip(), 'val1;val2')
        self.assertEqual(lines[2].strip(), 'val3;val4')

    def test_write_respects_limit_output(self):
        output = CSVOutput('test.csv', ['col1', 'col2'], 1)
        rows = [{'col1': 'val1', 'col2': 'val2'}, {'col1': 'val3', 'col2': 'val4'}]
        with output:
            output.write(rows)
        with open('test.csv', 'r') as f:
            lines = f.readlines()
        self.assertEqual(len(lines), 2)  # header + 1 row

    def test_enter_opens_file(self):
        output = CSVOutput('test.csv', ['col1', 'col2'], None)
        with output:
            self.assertIsNotNone(output.file)

    def test_exit_closes_file(self):
        output = CSVOutput('test.csv', ['col1', 'col2'], None)
        with output:
            pass
        self.assertTrue(output.file.closed)


if __name__ == "__main__":
    unittest.main()
