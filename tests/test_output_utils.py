import os
import unittest

from ov_ts_profiler.common_structs import ModelInfo
from ov_ts_profiler.output_utils import make_model_file_name, CSVOutput


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
