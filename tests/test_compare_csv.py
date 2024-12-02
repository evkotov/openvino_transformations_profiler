import unittest
from unittest.mock import patch, MagicMock
from collections import namedtuple

from ov_ts_profiler.common_structs import ModelInfo, ModelData, make_model_console_description
from ov_ts_profiler.output_utils import NoOutput, make_model_file_name
from compare_csv import (CSVSingleFileOutputFactory, ConsoleTableSingleFileOutputFactory,
                         MultiFileNoOutputFactory, CSVMultiFileOutputFactory,
                         ConsoleTableMultiOutputFactory, CompareCompileTime,
                         CompareSumTransformationTime, GenerateLongestUnitsOverall,
                         GenerateLongestUnitsPerModel, CompareSumUnitsOverall,
                         CompareSumUnitsPerModel, PlotCompileTimeByIteration, PlotSumTSTimeByIteration)


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


class TestCSVMultiFileOutputFactory(unittest.TestCase):

    @patch('compare_csv.CSVOutput')
    def test_create_table_creates_csv_output(self, MockCSVOutput):
        mock_instance = MagicMock()
        MockCSVOutput.return_value = mock_instance

        prefix = 'test_prefix'
        limit_output = 10
        header = ['col1', 'col2']
        model_info = ModelInfo('framework', 'model', 'precision', 'config')
        factory = CSVMultiFileOutputFactory(prefix, limit_output)
        factory.create_table(header, model_info)

        MockCSVOutput.assert_called_once_with(make_model_file_name(prefix, model_info, 'csv'), header, limit_output)
        self.assertEqual(factory.create_table(header, model_info), mock_instance)

    @patch('compare_csv.CSVOutput')
    def test_create_table_with_empty_header(self, MockCSVOutput):
        mock_instance = MagicMock()
        MockCSVOutput.return_value = mock_instance

        prefix = 'test_prefix'
        limit_output = 10
        header = []
        model_info = ModelInfo('framework', 'model', 'precision', 'config')
        factory = CSVMultiFileOutputFactory(prefix, limit_output)
        factory.create_table(header, model_info)

        MockCSVOutput.assert_called_once_with(make_model_file_name(prefix, model_info, 'csv'), header, limit_output)
        self.assertEqual(factory.create_table(header, model_info), mock_instance)

    @patch('compare_csv.CSVOutput')
    def test_create_table_with_none_limit_output(self, MockCSVOutput):
        mock_instance = MagicMock()
        MockCSVOutput.return_value = mock_instance

        prefix = 'test_prefix'
        limit_output = None
        header = ['col1', 'col2']
        model_info = ModelInfo('framework', 'model', 'precision', 'config')
        factory = CSVMultiFileOutputFactory(prefix, limit_output)
        factory.create_table(header, model_info)

        MockCSVOutput.assert_called_once_with(make_model_file_name(prefix, model_info, 'csv'), header, limit_output)
        self.assertEqual(factory.create_table(header, model_info), mock_instance)


class TestConsoleTableMultiOutputFactory(unittest.TestCase):

    @patch('compare_csv.ConsoleTableOutput')
    def test_create_table_creates_console_output(self, MockConsoleTableOutput):
        mock_instance = MagicMock()
        MockConsoleTableOutput.return_value = mock_instance

        description = 'test_description'
        limit_output = 10
        header = ['col1', 'col2']
        model_info = ModelInfo('framework', 'model', 'precision', 'config')
        factory = ConsoleTableMultiOutputFactory(description, limit_output)
        factory.create_table(header, model_info)

        MockConsoleTableOutput.assert_called_once_with(header, make_model_console_description(model_info), limit_output)
        self.assertEqual(factory.create_table(header, model_info), mock_instance)

    @patch('compare_csv.ConsoleTableOutput')
    def test_create_table_with_empty_header(self, MockConsoleTableOutput):
        mock_instance = MagicMock()
        MockConsoleTableOutput.return_value = mock_instance

        description = 'test_description'
        limit_output = 10
        header = []
        model_info = ModelInfo('framework', 'model', 'precision', 'config')
        factory = ConsoleTableMultiOutputFactory(description, limit_output)
        factory.create_table(header, model_info)

        MockConsoleTableOutput.assert_called_once_with(header, make_model_console_description(model_info), limit_output)
        self.assertEqual(factory.create_table(header, model_info), mock_instance)

    @patch('compare_csv.ConsoleTableOutput')
    def test_create_table_with_none_limit_output(self, MockConsoleTableOutput):
        mock_instance = MagicMock()
        MockConsoleTableOutput.return_value = mock_instance

        description = 'test_description'
        limit_output = None
        header = ['col1', 'col2']
        model_info = ModelInfo('framework', 'model', 'precision', 'config')
        factory = ConsoleTableMultiOutputFactory(description, limit_output)
        factory.create_table(header, model_info)

        MockConsoleTableOutput.assert_called_once_with(header, make_model_console_description(model_info), limit_output)
        self.assertEqual(factory.create_table(header, model_info), mock_instance)


class TestCompareCompileTime(unittest.TestCase):

    @patch('compare_csv.get_compile_time_data')
    @patch('compare_csv.get_comparison_values_compile_time')
    @patch('compare_csv.print_summary_stats')
    @patch('compare_csv.compare_compile_time')
    def test_comparing_compile_time_with_data(self, mock_compare_compile_time, mock_print_summary_stats, mock_get_comparison_values_compile_time, mock_get_compile_time_data):
        mock_get_compile_time_data.return_value = [(MagicMock(), MagicMock())]
        mock_compare_compile_time.return_value = (MagicMock(), MagicMock())
        mock_get_comparison_values_compile_time.return_value = MagicMock()
        mock_output_factory = MagicMock()
        mock_plot_output = MagicMock()
        processor = CompareCompileTime(mock_output_factory, True, mock_plot_output)
        csv_data = [MagicMock()]

        processor.run(csv_data)

        mock_get_compile_time_data.assert_called_once_with(csv_data)
        mock_compare_compile_time.assert_called_once()
        mock_get_comparison_values_compile_time.assert_called_once()
        mock_print_summary_stats.assert_called_once()
        mock_plot_output.plot.assert_called_once()

    @patch('compare_csv.get_compile_time_data')
    @patch('compare_csv.compare_compile_time')
    def test_comparing_compile_time_without_data(self, mock_compare_compile_time, mock_get_compile_time_data):
        mock_get_compile_time_data.return_value = []
        mock_output_factory = MagicMock()
        processor = CompareCompileTime(mock_output_factory, False, None)
        csv_data = []

        processor.run(csv_data)

        mock_get_compile_time_data.assert_not_called()
        mock_compare_compile_time.assert_not_called()
        mock_output_factory.create_table.assert_not_called()

    @patch('compare_csv.get_compile_time_data')
    @patch('compare_csv.get_comparison_values_compile_time')
    @patch('compare_csv.print_summary_stats')
    @patch('compare_csv.compare_compile_time')
    def test_comparing_compile_time_without_summary_stats(self, mock_compare_compile_time, mock_print_summary_stats, mock_get_comparison_values_compile_time, mock_get_compile_time_data):
        mock_get_compile_time_data.return_value = [(MagicMock(), MagicMock())]
        mock_compare_compile_time.return_value = (MagicMock(), MagicMock())
        mock_output_factory = MagicMock()
        processor = CompareCompileTime(mock_output_factory, False, None)
        csv_data = [MagicMock()]

        processor.run(csv_data)

        mock_get_compile_time_data.assert_called_once_with(csv_data)
        mock_compare_compile_time.assert_called_once()
        mock_get_comparison_values_compile_time.assert_called_once()
        mock_print_summary_stats.assert_not_called()
        mock_output_factory.create_table.assert_called_once()


class TestCompareSumTransformationTime(unittest.TestCase):

    @patch('compare_csv.get_sum_transformation_time_data')
    @patch('compare_csv.get_comparison_values_sum_transformation_time')
    @patch('compare_csv.print_summary_stats')
    @patch('compare_csv.compare_sum_transformation_time')
    def test_comparing_sum_transformation_time_with_data(self, mock_compare_sum_transformation_time, mock_print_summary_stats, mock_get_comparison_values_sum_transformation_time, mock_get_sum_transformation_time_data):
        mock_get_sum_transformation_time_data.return_value = [(MagicMock(), MagicMock())]
        mock_compare_sum_transformation_time.return_value = (MagicMock(), MagicMock())
        mock_get_comparison_values_sum_transformation_time.return_value = MagicMock()
        mock_output_factory = MagicMock()
        mock_plot_output = MagicMock()
        processor = CompareSumTransformationTime(mock_output_factory, True, mock_plot_output)
        csv_data = [MagicMock()]

        processor.run(csv_data)

        mock_get_sum_transformation_time_data.assert_called_once_with(csv_data)
        mock_compare_sum_transformation_time.assert_called_once()
        mock_get_comparison_values_sum_transformation_time.assert_called_once()
        mock_print_summary_stats.assert_called_once()
        mock_plot_output.plot.assert_called_once()

    @patch('compare_csv.get_sum_transformation_time_data')
    @patch('compare_csv.compare_sum_transformation_time')
    def test_comparing_sum_transformation_time_without_data(self, mock_compare_sum_transformation_time, mock_get_sum_transformation_time_data):
        mock_get_sum_transformation_time_data.return_value = []
        mock_output_factory = MagicMock()
        processor = CompareSumTransformationTime(mock_output_factory, False, None)
        csv_data = []

        processor.run(csv_data)

        mock_get_sum_transformation_time_data.assert_not_called()
        mock_compare_sum_transformation_time.assert_not_called()
        mock_output_factory.create_table.assert_not_called()

    @patch('compare_csv.get_sum_transformation_time_data')
    @patch('compare_csv.get_comparison_values_sum_transformation_time')
    @patch('compare_csv.print_summary_stats')
    @patch('compare_csv.compare_sum_transformation_time')
    def test_comparing_sum_transformation_time_without_summary_stats(self, mock_compare_sum_transformation_time, mock_print_summary_stats, mock_get_comparison_values_sum_transformation_time, mock_get_sum_transformation_time_data):
        mock_get_sum_transformation_time_data.return_value = [(MagicMock(), MagicMock())]
        mock_compare_sum_transformation_time.return_value = (MagicMock(), MagicMock())
        mock_output_factory = MagicMock()
        processor = CompareSumTransformationTime(mock_output_factory, False, None)
        csv_data = [MagicMock()]

        processor.run(csv_data)

        mock_get_sum_transformation_time_data.assert_called_once_with(csv_data)
        mock_compare_sum_transformation_time.assert_called_once()
        mock_get_comparison_values_sum_transformation_time.assert_called_once()
        mock_print_summary_stats.assert_not_called()
        mock_output_factory.create_table.assert_called_once()


class TestGenerateLongestUnitsOverall(unittest.TestCase):

    @patch('compare_csv.get_sum_units_comparison_data')
    @patch('compare_csv.join_sum_units')
    @patch('compare_csv.join_sum_units_by_name')
    @patch('compare_csv.get_longest_unit')
    def test_aggregating_longest_units_overall_with_data(self, mock_get_longest_unit, mock_join_sum_units_by_name, mock_join_sum_units, mock_get_sum_units_comparison_data):
        mock_get_sum_units_comparison_data.return_value = [MagicMock()]
        mock_join_sum_units.return_value = MagicMock()
        mock_join_sum_units_by_name.return_value = MagicMock()
        mock_get_longest_unit.return_value = (MagicMock(), MagicMock())
        mock_output_factory = MagicMock()
        processor = GenerateLongestUnitsOverall(mock_output_factory, 'transformation')
        csv_data = [MagicMock()]

        processor.run(csv_data)

        mock_get_sum_units_comparison_data.assert_called_once_with(csv_data, 'transformation')
        mock_join_sum_units.assert_called_once()
        mock_join_sum_units_by_name.assert_called_once()
        mock_get_longest_unit.assert_called_once()
        mock_output_factory.create_table.assert_called_once()

    @patch('compare_csv.get_sum_units_comparison_data')
    def test_aggregating_longest_units_overall_without_data(self, mock_get_sum_units_comparison_data):
        mock_get_sum_units_comparison_data.return_value = []
        mock_output_factory = MagicMock()
        processor = GenerateLongestUnitsOverall(mock_output_factory, 'transformation')
        csv_data = []

        processor.run(csv_data)

        mock_get_sum_units_comparison_data.assert_not_called()
        mock_output_factory.create_table.assert_not_called()


class TestGenerateLongestUnitsPerModel(unittest.TestCase):

    @patch('compare_csv.get_all_models')
    @patch('compare_csv.filter_by_models')
    @patch('compare_csv.get_sum_units_comparison_data')
    @patch('compare_csv.join_sum_units')
    @patch('compare_csv.join_sum_units_by_name')
    @patch('compare_csv.get_longest_unit')
    def test_aggregating_longest_units_per_model_with_data(self, mock_get_longest_unit, mock_join_sum_units_by_name, mock_join_sum_units, mock_get_sum_units_comparison_data, mock_filter_by_models, mock_get_all_models):
        mock_get_all_models.return_value = [MagicMock()]
        mock_filter_by_models.return_value = MagicMock()
        mock_get_sum_units_comparison_data.return_value = [MagicMock()]
        mock_join_sum_units.return_value = MagicMock()
        mock_join_sum_units_by_name.return_value = MagicMock()
        mock_get_longest_unit.return_value = (MagicMock(), MagicMock())
        mock_output_factory = MagicMock()
        processor = GenerateLongestUnitsPerModel(mock_output_factory, 'transformation')
        csv_data = [MagicMock()]

        processor.run(csv_data)

        mock_get_all_models.assert_called_once_with(csv_data)
        mock_filter_by_models.assert_called()
        mock_get_sum_units_comparison_data.assert_called()
        mock_join_sum_units.assert_called_once()
        mock_join_sum_units_by_name.assert_called_once()
        mock_get_longest_unit.assert_called_once()
        mock_output_factory.create_table.assert_called()

    @patch('compare_csv.get_all_models')
    def test_aggregating_longest_units_per_model_without_data(self, mock_get_all_models):
        mock_get_all_models.return_value = []
        mock_output_factory = MagicMock()
        processor = GenerateLongestUnitsPerModel(mock_output_factory, 'transformation')
        csv_data = []

        processor.run(csv_data)

        mock_get_all_models.assert_not_called()
        mock_output_factory.create_table.assert_not_called()


class TestCompareSumUnitsOverall(unittest.TestCase):

    @patch('compare_csv.filter_common_models')
    @patch('compare_csv.get_sum_units_comparison_data')
    @patch('compare_csv.join_sum_units')
    @patch('compare_csv.compare_sum_units')
    @patch('compare_csv.get_comparison_values_sum_units')
    @patch('compare_csv.print_summary_stats')
    def test_comparing_sum_units_overall_with_data(self, mock_print_summary_stats, mock_get_comparison_values_sum_units, mock_compare_sum_units, mock_join_sum_units, mock_get_sum_units_comparison_data, mock_filter_common_models):
        mock_filter_common_models.return_value = [MagicMock()]
        mock_get_sum_units_comparison_data.return_value = [MagicMock()]
        mock_join_sum_units.return_value = MagicMock()
        mock_compare_sum_units.return_value = (MagicMock(), MagicMock(), MagicMock())
        mock_get_comparison_values_sum_units.return_value = MagicMock()
        mock_output_factory = MagicMock()
        mock_plot_output = MagicMock()
        processor = CompareSumUnitsOverall(mock_output_factory, 'transformation', True, mock_plot_output)
        csv_data = [MagicMock()]

        processor.run(csv_data)

        mock_filter_common_models.assert_called_once_with(csv_data)
        mock_get_sum_units_comparison_data.assert_called_once()
        mock_join_sum_units.assert_called_once()
        mock_compare_sum_units.assert_called_once()
        mock_get_comparison_values_sum_units.assert_called()
        mock_print_summary_stats.assert_called_once()
        mock_plot_output.plot.assert_called_once()
        mock_output_factory.create_table.assert_called_once()

    @patch('compare_csv.filter_common_models')
    @patch('compare_csv.get_sum_units_comparison_data')
    @patch('compare_csv.join_sum_units')
    @patch('compare_csv.compare_sum_units')
    def test_comparing_sum_units_overall_without_data(self, mock_compare_sum_units, mock_join_sum_units, mock_get_sum_units_comparison_data, mock_filter_common_models):
        mock_filter_common_models.return_value = []
        mock_output_factory = MagicMock()
        processor = CompareSumUnitsOverall(mock_output_factory, 'transformation', False, None)
        csv_data = []

        processor.run(csv_data)

        mock_filter_common_models.assert_not_called()
        mock_get_sum_units_comparison_data.assert_not_called()
        mock_join_sum_units.assert_not_called()
        mock_compare_sum_units.assert_not_called()
        mock_output_factory.create_table.assert_not_called()

    @patch('compare_csv.filter_common_models')
    @patch('compare_csv.get_sum_units_comparison_data')
    @patch('compare_csv.join_sum_units')
    @patch('compare_csv.compare_sum_units')
    def test_comparing_sum_units_overall_with_data_no_summary_no_plot(self, mock_compare_sum_units, mock_join_sum_units, mock_get_sum_units_comparison_data, mock_filter_common_models):
        mock_filter_common_models.return_value = [MagicMock()]
        mock_get_sum_units_comparison_data.return_value = [MagicMock()]
        mock_join_sum_units.return_value = MagicMock()
        mock_compare_sum_units.return_value = (MagicMock(), MagicMock(), MagicMock())
        mock_output_factory = MagicMock()
        processor = CompareSumUnitsOverall(mock_output_factory, 'transformation', False, None)
        csv_data = [MagicMock()]

        processor.run(csv_data)

        mock_filter_common_models.assert_called_once_with(csv_data)
        mock_get_sum_units_comparison_data.assert_called_once()
        mock_join_sum_units.assert_called_once()
        mock_compare_sum_units.assert_called_once()
        mock_output_factory.create_table.assert_called_once()

    @patch('compare_csv.filter_common_models')
    @patch('compare_csv.get_sum_units_comparison_data')
    @patch('compare_csv.join_sum_units')
    @patch('compare_csv.compare_sum_units')
    def test_comparing_sum_units_overall_without_data_no_summary_no_plot(self, mock_compare_sum_units, mock_join_sum_units, mock_get_sum_units_comparison_data, mock_filter_common_models):
        mock_filter_common_models.return_value = []
        mock_output_factory = MagicMock()
        processor = CompareSumUnitsOverall(mock_output_factory, 'transformation', False, None)
        csv_data = []

        processor.run(csv_data)

        mock_filter_common_models.assert_not_called()
        mock_get_sum_units_comparison_data.assert_not_called()
        mock_join_sum_units.assert_not_called()
        mock_compare_sum_units.assert_not_called()
        mock_output_factory.create_table.assert_not_called()


class TestCompareSumUnitsPerModel(unittest.TestCase):

    @patch('compare_csv.filter_common_models')
    @patch('compare_csv.get_all_models')
    @patch('compare_csv.filter_by_models')
    @patch('compare_csv.get_sum_units_comparison_data')
    @patch('compare_csv.join_sum_units')
    @patch('compare_csv.compare_sum_units')
    @patch('compare_csv.get_comparison_values_sum_units')
    @patch('compare_csv.create_comparison_summary_table')
    def test_aggregating_sum_units_per_model_with_data(self, mock_create_comparison_summary_table, mock_get_comparison_values_sum_units, mock_compare_sum_units, mock_join_sum_units, mock_get_sum_units_comparison_data, mock_filter_by_models, mock_get_all_models, mock_filter_common_models):
        mock_filter_common_models.return_value = [MagicMock()]
        mock_get_all_models.return_value = [MagicMock()]
        mock_filter_by_models.return_value = MagicMock()
        mock_get_sum_units_comparison_data.return_value = [MagicMock()]
        mock_join_sum_units.return_value = MagicMock()
        mock_compare_sum_units.return_value = (MagicMock(), MagicMock())
        mock_get_comparison_values_sum_units.return_value = MagicMock()
        mock_create_comparison_summary_table.return_value = (MagicMock(), MagicMock())
        mock_output_factory = MagicMock()
        mock_summary_output_factory = MagicMock()
        mock_plot_output = MagicMock()
        processor = CompareSumUnitsPerModel(mock_output_factory, mock_summary_output_factory, 'transformation', mock_plot_output)
        csv_data = [MagicMock()]

        processor.run(csv_data)

        mock_filter_common_models.assert_called_once_with(csv_data)
        mock_get_all_models.assert_called_once()
        mock_filter_by_models.assert_called()
        mock_get_sum_units_comparison_data.assert_called()
        mock_join_sum_units.assert_called()
        mock_compare_sum_units.assert_called()
        mock_get_comparison_values_sum_units.assert_called()
        mock_create_comparison_summary_table.assert_called()
        mock_output_factory.create_table.assert_called()
        mock_summary_output_factory.create_table.assert_called()
        mock_plot_output.plot.assert_called()

    @patch('compare_csv.filter_common_models')
    @patch('compare_csv.get_all_models')
    @patch('compare_csv.filter_by_models')
    @patch('compare_csv.get_sum_units_comparison_data')
    @patch('compare_csv.join_sum_units')
    @patch('compare_csv.compare_sum_units')
    def test_aggregating_sum_units_per_model_without_data(self, mock_compare_sum_units, mock_join_sum_units, mock_get_sum_units_comparison_data, mock_filter_by_models, mock_get_all_models, mock_filter_common_models):
        mock_filter_common_models.return_value = []
        mock_output_factory = MagicMock()
        mock_summary_output_factory = MagicMock()
        mock_plot_output = MagicMock()
        processor = CompareSumUnitsPerModel(mock_output_factory, mock_summary_output_factory, 'transformation', mock_plot_output)
        csv_data = []

        processor.run(csv_data)

        mock_filter_common_models.assert_not_called()
        mock_get_all_models.assert_not_called()
        mock_filter_by_models.assert_not_called()
        mock_get_sum_units_comparison_data.assert_not_called()
        mock_join_sum_units.assert_not_called()
        mock_compare_sum_units.assert_not_called()
        mock_output_factory.create_table.assert_not_called()
        mock_summary_output_factory.create_table.assert_not_called()
        mock_plot_output.plot.assert_not_called()


class TestPlotCompileTimeByIteration(unittest.TestCase):

    @patch('compare_csv.get_device')
    @patch('compare_csv.compile_time_by_iterations')
    @patch('compare_csv.gen_plot_time_by_iterations')
    def test_plotting_compile_time_with_data(self, mock_gen_plot_time_by_iterations, mock_compile_time_by_iterations, mock_get_device):
        mock_get_device.return_value = 'CPU'
        mock_compile_time_by_iterations.return_value = [(MagicMock(), MagicMock())]
        processor = PlotCompileTimeByIteration()
        csv_data = [MagicMock()]

        processor.run(csv_data)

        mock_get_device.assert_called_once_with(csv_data)
        mock_compile_time_by_iterations.assert_called_once_with(csv_data)
        mock_gen_plot_time_by_iterations.assert_called_once()

    @patch('compare_csv.get_device')
    @patch('compare_csv.compile_time_by_iterations')
    @patch('compare_csv.gen_plot_time_by_iterations')
    def test_plotting_compile_time_without_data(self, mock_gen_plot_time_by_iterations, mock_compile_time_by_iterations, mock_get_device):
        mock_get_device.return_value = 'CPU'
        mock_compile_time_by_iterations.return_value = []
        processor = PlotCompileTimeByIteration()
        csv_data = []

        processor.run(csv_data)

        mock_get_device.assert_called_once_with(csv_data)
        mock_compile_time_by_iterations.assert_called_once_with(csv_data)
        mock_gen_plot_time_by_iterations.assert_not_called()


class TestPlotSumTSTimeByIteration(unittest.TestCase):

    @patch('compare_csv.get_device')
    @patch('compare_csv.get_sum_units_durations_by_iteration')
    @patch('compare_csv.gen_plot_time_by_iterations')
    def test_plotting_sum_ts_time_with_data(self, mock_gen_plot_time_by_iterations, mock_get_sum_units_durations_by_iteration, mock_get_device):
        mock_get_device.return_value = 'CPU'
        mock_get_sum_units_durations_by_iteration.return_value = [(MagicMock(), MagicMock())]
        processor = PlotSumTSTimeByIteration()
        csv_data = [MagicMock()]

        processor.run(csv_data)

        mock_get_device.assert_called_once_with(csv_data)
        mock_get_sum_units_durations_by_iteration.assert_called_once_with(csv_data, 'transformation')
        mock_gen_plot_time_by_iterations.assert_called_once()

    @patch('compare_csv.get_device')
    @patch('compare_csv.get_sum_units_durations_by_iteration')
    @patch('compare_csv.gen_plot_time_by_iterations')
    def test_plotting_sum_ts_time_without_data(self, mock_gen_plot_time_by_iterations, mock_get_sum_units_durations_by_iteration, mock_get_device):
        mock_get_device.return_value = 'CPU'
        mock_get_sum_units_durations_by_iteration.return_value = []
        processor = PlotSumTSTimeByIteration()
        csv_data = []

        processor.run(csv_data)

        mock_get_device.assert_called_once_with(csv_data)
        mock_get_sum_units_durations_by_iteration.assert_called_once_with(csv_data, 'transformation')
        mock_gen_plot_time_by_iterations.assert_not_called()


if __name__ == "__main__":
    unittest.main()
