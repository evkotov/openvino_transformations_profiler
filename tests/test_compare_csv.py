import unittest
from unittest.mock import patch, MagicMock
from collections import namedtuple

from ov_ts_profiler.common_structs import ModelInfo
from ov_ts_profiler.output_utils import NoOutput
from compare_csv import (CSVSingleFileOutputFactory, ConsoleTableSingleFileOutputFactory,
                         MultiFileNoOutputFactory)


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


class TestCSVSingleFileOutputFactory(unittest.TestCase):

    @patch('compare_csv.CSVOutput')  # Patch CSVOutput in the module where it's used
    def test_create_table_creates_csv_output(self, MockCSVOutput):
        # Create a mock instance of CSVOutput
        mock_instance = MagicMock()
        MockCSVOutput.return_value = mock_instance  # Set the return value of the class constructor

        # Instantiate CSVSingleFileOutputFactory and call create_table()
        path_prefix = 'test_path'
        limit_output = 10
        header = ['col1', 'col2']
        factory = CSVSingleFileOutputFactory(path_prefix, limit_output)
        factory.create_table(header)

        # Assertions to verify behavior
        MockCSVOutput.assert_called_once_with('test_path.csv', header, limit_output)
        self.assertEqual(factory.create_table(header), mock_instance)  # Check if the returned object is the mock instance

    @patch('compare_csv.CSVOutput')
    def test_create_table_with_empty_header(self, MockCSVOutput):
        mock_instance = MagicMock()
        MockCSVOutput.return_value = mock_instance

        path_prefix = 'test_path'
        limit_output = 10
        header = []
        factory = CSVSingleFileOutputFactory(path_prefix, limit_output)
        factory.create_table(header)

        MockCSVOutput.assert_called_once_with('test_path.csv', header, limit_output)
        self.assertEqual(factory.create_table(header), mock_instance)

    @patch('compare_csv.CSVOutput')
    def test_create_table_with_none_limit_output(self, MockCSVOutput):
        mock_instance = MagicMock()
        MockCSVOutput.return_value = mock_instance

        path_prefix = 'test_path'
        limit_output = None
        header = ['col1', 'col2']
        factory = CSVSingleFileOutputFactory(path_prefix, limit_output)
        factory.create_table(header)

        MockCSVOutput.assert_called_once_with('test_path.csv', header, limit_output)
        self.assertEqual(factory.create_table(header), mock_instance)


class TestConsoleTableSingleFileOutputFactory(unittest.TestCase):

    @patch('compare_csv.ConsoleTableOutput')
    def test_create_table_creates_console_table_output(self, MockConsoleTableOutput):
        mock_instance = MagicMock()
        MockConsoleTableOutput.return_value = mock_instance

        description = 'test_description'
        limit_output = 10
        header = ['col1', 'col2']
        factory = ConsoleTableSingleFileOutputFactory(description, limit_output)
        factory.create_table(header)

        MockConsoleTableOutput.assert_called_once_with(header, description, limit_output)
        self.assertEqual(factory.create_table(header), mock_instance)

    @patch('compare_csv.ConsoleTableOutput')
    def test_create_table_with_empty_header(self, MockConsoleTableOutput):
        mock_instance = MagicMock()
        MockConsoleTableOutput.return_value = mock_instance

        description = 'test_description'
        limit_output = 10
        header = []
        factory = ConsoleTableSingleFileOutputFactory(description, limit_output)
        factory.create_table(header)

        MockConsoleTableOutput.assert_called_once_with(header, description, limit_output)
        self.assertEqual(factory.create_table(header), mock_instance)

    @patch('compare_csv.ConsoleTableOutput')
    def test_create_table_with_none_limit_output(self, MockConsoleTableOutput):
        mock_instance = MagicMock()
        MockConsoleTableOutput.return_value = mock_instance

        description = 'test_description'
        limit_output = None
        header = ['col1', 'col2']
        factory = ConsoleTableSingleFileOutputFactory(description, limit_output)
        factory.create_table(header)

        MockConsoleTableOutput.assert_called_once_with(header, description, limit_output)
        self.assertEqual(factory.create_table(header), mock_instance)


class TestMultiFileNoOutputFactory(unittest.TestCase):

    def test_create_table_returns_no_output(self):
        factory = MultiFileNoOutputFactory()
        header = ['col1', 'col2']
        model_info = ModelInfo('framework', 'model', 'precision', 'config')
        result = factory.create_table(header, model_info)
        self.assertIsInstance(result, NoOutput)

    def test_create_table_with_empty_header(self):
        factory = MultiFileNoOutputFactory()
        header = []
        model_info = ModelInfo('framework', 'model', 'precision', 'config')
        result = factory.create_table(header, model_info)
        self.assertIsInstance(result, NoOutput)

    def test_create_table_with_none_model_info(self):
        factory = MultiFileNoOutputFactory()
        header = ['col1', 'col2']
        model_info = None
        result = factory.create_table(header, model_info)
        self.assertIsInstance(result, NoOutput)

if __name__ == "__main__":
    unittest.main()
