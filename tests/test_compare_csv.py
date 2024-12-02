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
                         CompareSumUnitsPerModel, PlotCompileTimeByIteration, PlotSumTSTimeByIteration,
                         parse_args, create_single_output_factory, SingleNoOutputFactory, Config,
                         create_multi_output_factory, create_summary_output_factory, build_data_processors,
                         PlotCompareCompileTime, PlotCompareSumTransformationTime, PlotCompareSumUnitsOverall)


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
        processor = CompareCompileTime(mock_output_factory, True)
        csv_data = [MagicMock()]

        processor.run(csv_data)

        mock_get_compile_time_data.assert_called_once_with(csv_data)
        mock_compare_compile_time.assert_called_once()
        mock_get_comparison_values_compile_time.assert_called_once()
        mock_print_summary_stats.assert_called_once()

    @patch('compare_csv.get_compile_time_data')
    @patch('compare_csv.compare_compile_time')
    def test_comparing_compile_time_without_data(self, mock_compare_compile_time, mock_get_compile_time_data):
        mock_get_compile_time_data.return_value = []
        mock_output_factory = MagicMock()
        processor = CompareCompileTime(mock_output_factory, False)
        csv_data = []

        processor.run(csv_data)

        mock_get_compile_time_data.assert_not_called()
        mock_compare_compile_time.assert_not_called()
        mock_output_factory.create_table.assert_not_called()

    @patch('compare_csv.get_compile_time_data')
    @patch('compare_csv.print_summary_stats')
    @patch('compare_csv.compare_compile_time')
    def test_comparing_compile_time_without_summary_stats(self, mock_compare_compile_time, mock_print_summary_stats, mock_get_compile_time_data):
        mock_get_compile_time_data.return_value = [(MagicMock(), MagicMock())]
        mock_compare_compile_time.return_value = (MagicMock(), MagicMock())
        mock_output_factory = MagicMock()
        processor = CompareCompileTime(mock_output_factory, False)
        csv_data = [MagicMock()]

        processor.run(csv_data)

        mock_get_compile_time_data.assert_called_once_with(csv_data)
        mock_compare_compile_time.assert_called_once()
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
        processor = CompareSumTransformationTime(mock_output_factory, True)
        csv_data = [MagicMock()]

        processor.run(csv_data)

        mock_get_sum_transformation_time_data.assert_called_once_with(csv_data)
        mock_compare_sum_transformation_time.assert_called_once()
        mock_get_comparison_values_sum_transformation_time.assert_called_once()
        mock_print_summary_stats.assert_called_once()

    @patch('compare_csv.get_sum_transformation_time_data')
    @patch('compare_csv.compare_sum_transformation_time')
    def test_comparing_sum_transformation_time_without_data(self, mock_compare_sum_transformation_time, mock_get_sum_transformation_time_data):
        mock_get_sum_transformation_time_data.return_value = []
        mock_output_factory = MagicMock()
        processor = CompareSumTransformationTime(mock_output_factory, False)
        csv_data = []

        processor.run(csv_data)

        mock_get_sum_transformation_time_data.assert_not_called()
        mock_compare_sum_transformation_time.assert_not_called()
        mock_output_factory.create_table.assert_not_called()

    @patch('compare_csv.get_sum_transformation_time_data')
    @patch('compare_csv.print_summary_stats')
    @patch('compare_csv.compare_sum_transformation_time')
    def test_comparing_sum_transformation_time_without_summary_stats(self, mock_compare_sum_transformation_time, mock_print_summary_stats, mock_get_sum_transformation_time_data):
        mock_get_sum_transformation_time_data.return_value = [(MagicMock(), MagicMock())]
        mock_compare_sum_transformation_time.return_value = (MagicMock(), MagicMock())
        mock_output_factory = MagicMock()
        processor = CompareSumTransformationTime(mock_output_factory, False)
        csv_data = [MagicMock()]

        processor.run(csv_data)

        mock_get_sum_transformation_time_data.assert_called_once_with(csv_data)
        mock_compare_sum_transformation_time.assert_called_once()
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
        mock_compare_sum_units.return_value = (MagicMock(), MagicMock())
        mock_get_comparison_values_sum_units.return_value = MagicMock()
        mock_output_factory = MagicMock()
        processor = CompareSumUnitsOverall(mock_output_factory, 'transformation', True)
        csv_data = [MagicMock()]

        processor.run(csv_data)

        mock_filter_common_models.assert_called_once_with(csv_data)
        mock_get_sum_units_comparison_data.assert_called_once()
        mock_join_sum_units.assert_called_once()
        mock_compare_sum_units.assert_called_once()
        mock_get_comparison_values_sum_units.assert_called()
        mock_print_summary_stats.assert_called_once()
        mock_output_factory.create_table.assert_called_once()

    @patch('compare_csv.filter_common_models')
    @patch('compare_csv.get_sum_units_comparison_data')
    @patch('compare_csv.join_sum_units')
    @patch('compare_csv.compare_sum_units')
    def test_comparing_sum_units_overall_without_data(self, mock_compare_sum_units, mock_join_sum_units, mock_get_sum_units_comparison_data, mock_filter_common_models):
        mock_filter_common_models.return_value = []
        mock_output_factory = MagicMock()
        processor = CompareSumUnitsOverall(mock_output_factory, 'transformation', False)
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
        mock_compare_sum_units.return_value = (MagicMock(), MagicMock())
        mock_output_factory = MagicMock()
        processor = CompareSumUnitsOverall(mock_output_factory, 'transformation', False)
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
        processor = CompareSumUnitsOverall(mock_output_factory, 'transformation', False)
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


class TestParseArgs(unittest.TestCase):

    @patch('compare_csv.sys.argv', ['script_name', '--input', '/dir1/file1.csv,/dir2/file2.csv'])
    def test_parses_input_files_correctly(self):
        config = parse_args()
        self.assertEqual(config.inputs, ['/dir1/file1.csv', '/dir2/file2.csv'])

    @patch('compare_csv.sys.argv', ['script_name', '--input', '/dir1/file1.csv,/dir2/file2.csv', '--compare_compile_time'])
    def test_sets_compare_compile_time_flag(self):
        config = parse_args()
        self.assertEqual(config.compare_compile_time, 'compile_time_comparison')

    @patch('compare_csv.sys.argv', ['script_name', '--input', '/dir1/file1.csv,/dir2/file2.csv', '--output_type', 'console'])
    def test_sets_output_type_correctly(self):
        config = parse_args()
        self.assertEqual(config.output_type, 'console')

    @patch('compare_csv.sys.argv', ['script_name', '--input', '/dir1/file1.csv,/dir2/file2.csv', '--summary_statistics'])
    def test_enables_summary_statistics(self):
        config = parse_args()
        self.assertTrue(config.summary_statistics)

    @patch('compare_csv.sys.argv', ['script_name', '--input', '/dir1/file1.csv,/dir2/file2.csv', '--plots'])
    def test_enables_plots(self):
        config = parse_args()
        self.assertTrue(config.plots)

    @patch('compare_csv.sys.argv', ['script_name', '--input', '/dir1/file1.csv,/dir2/file2.csv', '--limit_output', '10'])
    def test_sets_limit_output_correctly(self):
        config = parse_args()
        self.assertEqual(config.limit_output, 10)

    @patch('compare_csv.sys.argv', ['script_name', '--input', '/dir1/file1.csv,/dir2/file2.csv', '--model_name', 'llama-3-8b'])
    def test_filters_by_model_name(self):
        config = parse_args()
        self.assertEqual(config.model_name, 'llama-3-8b')

    @patch('compare_csv.sys.argv', ['script_name', '--input', '/dir1/file1.csv,/dir2/file2.csv', '--no_csv'])
    def test_disables_csv_output(self):
        config = parse_args()
        self.assertTrue(config.no_csv)

    @patch('compare_csv.sys.argv', ['script_name', '--input', '/dir1/file1.csv,/dir2/file2.csv', '--n_plot_segments', '3'])
    def test_sets_number_of_plot_segments(self):
        config = parse_args()
        self.assertEqual(config.n_plot_segments, 3)

    @patch('compare_csv.sys.argv', ['script_name', '--input', '/dir1/file1.csv,/dir2/file2.csv', '--plot_compile_time_by_iteration'])
    def test_enables_plot_compile_time_by_iteration(self):
        config = parse_args()
        self.assertTrue(config.plot_compile_time_by_iteration)

    @patch('compare_csv.sys.argv', ['script_name', '--input', '/dir1/file1.csv,/dir2/file2.csv', '--plot_sum_ts_time_by_iteration'])
    def test_enables_plot_sum_ts_time_by_iteration(self):
        config = parse_args()
        self.assertTrue(config.plot_sum_ts_time_by_iteration)

    @patch('sys.argv', ['script_name', '--input', '/dir1/file1.csv,/dir2/file2.csv', '--plot_compare_compile_time'])
    def test_parses_plot_compare_compile_time(self):
        config = parse_args()
        self.assertTrue(config.plot_compare_compile_time)
        self.assertEqual(config.inputs, ['/dir1/file1.csv', '/dir2/file2.csv'])


class TestCreateSingleOutputFactory(unittest.TestCase):

    def test_creates_no_output_factory_when_no_csv(self):
        config = Config(no_csv=True)
        factory = create_single_output_factory(config, 'path_prefix', 'description')
        self.assertIsInstance(factory, SingleNoOutputFactory)

    def test_creates_csv_output_factory_when_output_type_is_csv(self):
        config = Config(no_csv=False)
        factory = create_single_output_factory(config, 'path_prefix', 'description')
        self.assertIsInstance(factory, CSVSingleFileOutputFactory)
        self.assertEqual(factory.path_prefix, 'path_prefix')
        self.assertEqual(factory.limit_output, config.limit_output)

    def test_creates_console_output_factory_when_output_type_is_console(self):
        config = Config(no_csv=False)
        config.output_type = 'console'
        factory = create_single_output_factory(config, 'path_prefix', 'description')
        self.assertIsInstance(factory, ConsoleTableSingleFileOutputFactory)
        self.assertEqual(factory.description, 'description')
        self.assertEqual(factory.limit_output, config.limit_output)


class TestCreateMultiOutputFactory(unittest.TestCase):

    def test_creates_no_output_factory_when_no_csv(self):
        config = Config(no_csv=True)
        config.output_type = 'csv'
        config.limit_output = 10
        factory = create_multi_output_factory(config, 'prefix', 'description')
        self.assertIsInstance(factory, MultiFileNoOutputFactory)

    def test_creates_csv_output_factory_when_output_type_is_csv(self):
        config = Config(no_csv=False)
        config.output_type = 'csv'
        config.limit_output = 10
        factory = create_multi_output_factory(config, 'prefix', 'description')
        self.assertIsInstance(factory, CSVMultiFileOutputFactory)
        self.assertEqual(factory.prefix, 'prefix')
        self.assertEqual(factory.limit_output, 10)

    def test_creates_console_output_factory_when_output_type_is_console(self):
        config = Config(no_csv=False)
        config.output_type = 'console'
        config.limit_output = 5
        factory = create_multi_output_factory(config, 'prefix', 'description')
        self.assertIsInstance(factory, ConsoleTableMultiOutputFactory)
        self.assertEqual(factory.description, 'description')
        self.assertEqual(factory.limit_output, 5)


class TestCreateSummaryOutputFactory(unittest.TestCase):

    def test_creates_csv_summary_output_factory(self):
        factory = create_summary_output_factory('csv', 'prefix', 'description')
        self.assertIsInstance(factory, CSVSingleFileOutputFactory)
        self.assertEqual(factory.path_prefix, 'prefix_summary')
        self.assertIsNone(factory.limit_output)

    def test_creates_console_summary_output_factory(self):
        factory = create_summary_output_factory('console', 'prefix', 'description')
        self.assertIsInstance(factory, ConsoleTableSingleFileOutputFactory)
        self.assertEqual(factory.description, 'description')
        self.assertIsNone(factory.limit_output)


class TestBuildDataProcessorsFunction(unittest.TestCase):

    def test_creates_data_processors_for_compile_time_comparison(self):
        config = Config()
        config.compare_compile_time = 'compile_time_comparison'
        data_processors = build_data_processors(config)
        self.assertEqual(len(data_processors), 1)
        self.assertIsInstance(data_processors[0], CompareCompileTime)

    def test_creates_data_processors_for_sum_transformation_time_comparison(self):
        config = Config()
        config.compare_sum_transformation_time = 'transformation_sum_time_comparison'
        data_processors = build_data_processors(config)
        self.assertEqual(len(data_processors), 1)
        self.assertIsInstance(data_processors[0], CompareSumTransformationTime)

    def test_creates_data_processors_for_transformations_overall(self):
        config = Config()
        config.transformations_overall = 'transformations_overall'
        data_processors = build_data_processors(config)
        self.assertEqual(len(data_processors), 1)
        self.assertIsInstance(data_processors[0], GenerateLongestUnitsOverall)

    def test_creates_data_processors_for_manager_overall(self):
        config = Config()
        config.manager_overall = 'manager_overall'
        data_processors = build_data_processors(config)
        self.assertEqual(len(data_processors), 1)
        self.assertIsInstance(data_processors[0], GenerateLongestUnitsOverall)

    def test_creates_data_processors_for_transformations_per_model(self):
        config = Config()
        config.transformations_per_model = 'transformations_per_model'
        data_processors = build_data_processors(config)
        self.assertEqual(len(data_processors), 1)
        self.assertIsInstance(data_processors[0], GenerateLongestUnitsPerModel)

    def test_creates_data_processors_for_managers_per_model(self):
        config = Config()
        config.managers_per_model = 'managers_per_model'
        data_processors = build_data_processors(config)
        self.assertEqual(len(data_processors), 1)
        self.assertIsInstance(data_processors[0], GenerateLongestUnitsPerModel)

    def test_creates_data_processors_for_compare_transformations_overall(self):
        config = Config()
        config.compare_transformations_overall = 'compare_transformations_overall'
        data_processors = build_data_processors(config)
        self.assertEqual(len(data_processors), 1)
        self.assertIsInstance(data_processors[0], CompareSumUnitsOverall)

    def test_creates_data_processors_for_compare_managers_overall(self):
        config = Config()
        config.compare_managers_overall = 'compare_managers_overall'
        data_processors = build_data_processors(config)
        self.assertEqual(len(data_processors), 1)
        self.assertIsInstance(data_processors[0], CompareSumUnitsOverall)

    def test_creates_data_processors_for_compare_transformations_per_model(self):
        config = Config()
        config.compare_transformations_per_model = 'compare_transformations_per_model'
        data_processors = build_data_processors(config)
        self.assertEqual(len(data_processors), 1)
        self.assertIsInstance(data_processors[0], CompareSumUnitsPerModel)

    def test_creates_data_processors_for_compare_managers_per_model(self):
        config = Config()
        config.compare_managers_per_model = 'compare_managers_per_model'
        data_processors = build_data_processors(config)
        self.assertEqual(len(data_processors), 1)
        self.assertIsInstance(data_processors[0], CompareSumUnitsPerModel)

    def test_creates_data_processors_for_plot_compile_time_by_iteration(self):
        config = Config()
        config.plot_compile_time_by_iteration = True
        data_processors = build_data_processors(config)
        self.assertEqual(len(data_processors), 1)
        self.assertIsInstance(data_processors[0], PlotCompileTimeByIteration)

    def test_creates_data_processors_for_plot_sum_ts_time_by_iteration(self):
        config = Config()
        config.plot_sum_ts_time_by_iteration = True
        data_processors = build_data_processors(config)
        self.assertEqual(len(data_processors), 1)
        self.assertIsInstance(data_processors[0], PlotSumTSTimeByIteration)

    def test_does_nothing_when_no_data_processors(self):
        config = Config()
        data_processors = build_data_processors(config)
        self.assertEqual(len(data_processors), 0)

    @patch('compare_csv.PlotOutput')
    @patch('compare_csv.PlotCompareCompileTime', autospec=True)
    def test_adds_plot_compare_compile_time_processor(self, mock_plot_compare_compile_time, mock_plot_output):
        config = Config()
        config.plot_compare_compile_time = True
        mock_plot_output_instance = mock_plot_output.return_value
        data_processors = build_data_processors(config)
        mock_plot_output.assert_called_once_with('compilation', 'compilation time', config.n_plot_segments)
        mock_plot_compare_compile_time.assert_called_once_with(mock_plot_output_instance)
        self.assertEqual(len(data_processors), 1)
        self.assertIsInstance(data_processors[0], PlotCompareCompileTime)

    @patch('compare_csv.PlotOutput')
    @patch('compare_csv.PlotCompareCompileTime', autospec=True)
    def test_adds_plot_compare_compile_time_processor_with_default_prefix(self, mock_plot_compare_compile_time, mock_plot_output):
        config = Config()
        config.plot_compare_compile_time = True
        config.compare_compile_time = None
        mock_plot_output_instance = mock_plot_output.return_value
        data_processors = build_data_processors(config)
        mock_plot_output.assert_called_once_with('compilation', 'compilation time', config.n_plot_segments)
        mock_plot_compare_compile_time.assert_called_once_with(mock_plot_output_instance)
        self.assertEqual(len(data_processors), 1)
        self.assertIsInstance(data_processors[0], PlotCompareCompileTime)

    @patch('compare_csv.PlotOutput')
    @patch('compare_csv.PlotCompareCompileTime')
    def test_does_not_add_plot_compare_compile_time_processor_when_not_enabled(self, mock_plot_compare_compile_time, mock_plot_output):
        config = Config()
        config.plot_compare_compile_time = False
        data_processors = build_data_processors(config)
        mock_plot_output.assert_not_called()
        mock_plot_compare_compile_time.assert_not_called()
        self.assertEqual(len(data_processors), 0)

    @patch('compare_csv.PlotOutput')
    def test_plot_compare_transformations_overall_creates_plot_output(self, mock_plot_output):
        config = Config()
        config.plot_compare_transformations_overall = True
        config.compare_transformations_overall = None
        config.n_plot_segments = 3

        data_processors = build_data_processors(config)
        self.assertEqual(len(data_processors), 1)
        self.assertIsInstance(data_processors[0], PlotCompareSumUnitsOverall)
        mock_plot_output.assert_called_once_with('ts_overall', 'transformations overall time', 3)

    @patch('compare_csv.PlotOutput')
    def test_plot_compare_transformations_overall_uses_provided_prefix(self, mock_plot_output):
        config = Config()
        config.plot_compare_transformations_overall = True
        config.n_plot_segments = 3

        data_processors = build_data_processors(config)
        self.assertEqual(len(data_processors), 1)
        self.assertIsInstance(data_processors[0], PlotCompareSumUnitsOverall)
        mock_plot_output.assert_called_once_with('ts_overall', 'transformations overall time', 3)

    @patch('compare_csv.PlotOutput')
    def test_plot_compare_transformations_overall_no_data_processors_when_disabled(self, mock_plot_output):
        config = Config()
        config.plot_compare_transformations_overall = False

        data_processors = build_data_processors(config)
        self.assertEqual(len(data_processors), 0)
        mock_plot_output.assert_not_called()


class TestPlotCompareCompileTime(unittest.TestCase):

    @patch('compare_csv.PlotOutput')
    def test_processes_compile_time_data(self, mock_plot_output):
        config = Config()
        config.plot_compare_compile_time = True
        mock_plot_output_instance = mock_plot_output.return_value
        data_processor = PlotCompareCompileTime(mock_plot_output_instance)
        csv_data = [{ModelInfo('framework', 'name', 'precision', 'config'): ModelData()}]
        data_processor.run(csv_data)
        mock_plot_output_instance.plot.assert_called_once()

    @patch('compare_csv.PlotOutput')
    def test_does_nothing_when_no_data(self, mock_plot_output):
        config = Config()
        config.plot_compare_compile_time = True
        mock_plot_output_instance = mock_plot_output.return_value
        data_processor = PlotCompareCompileTime(mock_plot_output_instance)
        csv_data = []
        data_processor.run(csv_data)
        mock_plot_output_instance.plot.assert_not_called()


class TestBuildDataProcessors(unittest.TestCase):

    @patch('compare_csv.PlotOutput')
    def test_plot_compare_sum_transformation_time_creates_plot_output(self, mock_plot_output):
        config = Config()
        config.plot_compare_sum_transformation_time = True
        config.compare_sum_transformation_time = None
        config.n_plot_segments = 3

        data_processors = build_data_processors(config)
        self.assertEqual(len(data_processors), 1)
        self.assertIsInstance(data_processors[0], PlotCompareSumTransformationTime)
        mock_plot_output.assert_called_once_with('sum_ts', 'sum transformation time', 3)

    @patch('compare_csv.PlotOutput')
    def test_plot_compare_sum_transformation_time_uses_provided_prefix(self, mock_plot_output):
        config = Config()
        config.plot_compare_sum_transformation_time = True
        config.n_plot_segments = 3

        data_processors = build_data_processors(config)
        self.assertEqual(len(data_processors), 1)
        self.assertIsInstance(data_processors[0], PlotCompareSumTransformationTime)
        mock_plot_output.assert_called_once_with('sum_ts', 'sum transformation time', 3)

    @patch('compare_csv.PlotOutput')
    def test_plot_compare_sum_transformation_time_no_data_processors_when_disabled(self, mock_plot_output):
        config = Config()
        config.plot_compare_sum_transformation_time = False

        data_processors = build_data_processors(config)
        self.assertEqual(len(data_processors), 0)
        mock_plot_output.assert_not_called()


class TestPlotCompareSumUnitsOverall(unittest.TestCase):

    @patch('compare_csv.get_comparison_values_sum_units')
    @patch('compare_csv.join_sum_units')
    @patch('compare_csv.get_sum_units_comparison_data')
    @patch('compare_csv.filter_common_models')
    def test_plots_comparison_values_when_data_exists(self, mock_filter_common_models, mock_get_sum_units_comparison_data, mock_join_sum_units, mock_get_comparison_values_sum_units):
        mock_plot_output = MagicMock()
        processor = PlotCompareSumUnitsOverall('transformation', mock_plot_output)
        csv_data = [MagicMock()]
        mock_filter_common_models.return_value = csv_data
        mock_get_sum_units_comparison_data.return_value = [MagicMock()]
        mock_join_sum_units.return_value = MagicMock()
        mock_get_comparison_values_sum_units.return_value = MagicMock()

        processor.run(csv_data)

        mock_plot_output.plot.assert_called_once()

    @patch('compare_csv.get_comparison_values_sum_units')
    @patch('compare_csv.join_sum_units')
    @patch('compare_csv.get_sum_units_comparison_data')
    @patch('compare_csv.filter_common_models')
    def test_does_not_plot_when_no_data(self, mock_filter_common_models, mock_get_sum_units_comparison_data, mock_join_sum_units, mock_get_comparison_values_sum_units):
        mock_plot_output = MagicMock()
        processor = PlotCompareSumUnitsOverall('transformation', mock_plot_output)
        csv_data = []

        mock_filter_common_models.return_value = csv_data
        mock_get_sum_units_comparison_data.return_value = []
        mock_join_sum_units.return_value = None
        mock_get_comparison_values_sum_units.return_value = None

        processor.run(csv_data)

        mock_plot_output.plot.assert_not_called()

    @patch('compare_csv.get_comparison_values_sum_units')
    @patch('compare_csv.join_sum_units')
    @patch('compare_csv.get_sum_units_comparison_data')
    @patch('compare_csv.filter_common_models')
    def test_does_not_plot_when_no_common_models(self, mock_filter_common_models, mock_get_sum_units_comparison_data, mock_join_sum_units, mock_get_comparison_values_sum_units):
        mock_plot_output = MagicMock()
        processor = PlotCompareSumUnitsOverall('transformation', mock_plot_output)
        csv_data = [MagicMock()]
        mock_filter_common_models.return_value = []

        processor.run(csv_data)

        mock_plot_output.plot.assert_not_called()


if __name__ == "__main__":
    unittest.main()
