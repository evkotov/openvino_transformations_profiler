import os
import unittest
from collections import namedtuple

from ov_ts_profiler.common_structs import ModelData
from ov_ts_profiler.parse_input import get_csv_header, read_csv, is_header_valid, get_config_value_from_path, \
    remove_invalid_items


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

    def header_is_valid_with_correct_columns(self):
        column_names = ['model_framework', 'model_name', 'model_precision', 'device', 'status', 'config']
        self.assertTrue(is_header_valid(column_names))

    def header_is_valid_with_optional_columns(self):
        column_names = ['model_framework', 'model_name', 'model_precision', 'optional_model_attribute', 'device', 'status']
        self.assertTrue(is_header_valid(column_names))

    def header_is_valid_with_weight_compression(self):
        column_names = ['model_framework', 'model_name', 'model_precision', 'weight_compression', 'device', 'status']
        self.assertTrue(is_header_valid(column_names))

    def header_is_invalid_with_missing_columns(self):
        column_names = ['model_framework', 'model_name', 'model_precision']
        self.assertFalse(is_header_valid(column_names))

    def header_is_invalid_with_extra_columns(self):
        column_names = ['model_framework', 'model_name', 'model_precision', 'config', 'device', 'status', 'extra_column']
        self.assertFalse(is_header_valid(column_names))

    def header_is_invalid_with_wrong_order(self):
        column_names = ['model_name', 'model_framework', 'model_precision', 'config', 'device', 'status']
        self.assertFalse(is_header_valid(column_names))

    def header_is_invalid_without_config_and_optional_columns(self):
        column_names = ['model_framework', 'model_name', 'model_precision', 'device', 'status']
        self.assertFalse(is_header_valid(column_names))

    def header_is_valid_with_only_required_columns(self):
        column_names = ['model_framework', 'model_name', 'model_precision']
        self.assertTrue(is_header_valid(column_names))


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
