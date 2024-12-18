from __future__ import annotations

from abc import ABC, abstractmethod
import argparse
from dataclasses import dataclass, field
import sys
from typing import List, Dict, Optional

from ov_ts_profiler.output_utils import print_summary_stats, make_model_file_name, NoOutput, CSVOutput, ConsoleTableOutput
from ov_ts_profiler.parse_input import get_csv_data, get_input_csv_files
from ov_ts_profiler.common_structs import ModelData, ModelInfo, ComparisonValues, make_model_console_description
from ov_ts_profiler.plot_utils import PlotOutput, gen_plot_time_by_iterations, PlotOutputRatioSimple
from ov_ts_profiler.stat_utils import filter_by_models, filter_by_model_name, filter_common_models, get_device, \
    get_all_models, \
    compile_time_by_iterations, get_sum_units_durations_by_iteration, get_compile_time_data, \
    get_sum_transformation_time_data, get_sum_units_comparison_data, join_sum_units, join_sum_units_by_name, \
    get_comparison_values_compile_time, get_comparison_values_sum_transformation_time, \
    get_comparison_values_sum_units, get_sum_plain_manager_time_data, get_sum_plain_manager_gap_time_data, \
    get_plain_manager_time_by_iteration, get_plain_manager_gap_time_by_iteration, \
    join_mem_rss_by_model, join_mem_virtual_by_model
from ov_ts_profiler.table import compare_compile_time, compare_sum_transformation_time, get_longest_unit, compare_sum_units, \
    create_comparison_summary_table, compare_compilation_and_plain_manager_sum_time, compare_mem_rss


class SingleOutputFactory(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def create_table(self, header: List[str]):
        pass


class SingleNoOutputFactory(SingleOutputFactory):
    def __init__(self):
        super().__init__()

    def create_table(self, header: List[str]):
        return NoOutput()


class CSVSingleFileOutputFactory(SingleOutputFactory):
    def __init__(self, path_prefix: str, limit_output):
        super().__init__()
        self.path_prefix = path_prefix
        self.limit_output = limit_output

    def create_table(self, header: List[str]):
        path = self.path_prefix + '.csv'
        return CSVOutput(path, header, self.limit_output)


class ConsoleTableSingleFileOutputFactory(SingleOutputFactory):
    def __init__(self, description: str, limit_output):
        super().__init__()
        self.description = description
        self.limit_output = limit_output

    def create_table(self, header: List[str]):
        return ConsoleTableOutput(header, self.description, self.limit_output)


class MultiOutputFactory(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def create_table(self, header: List[str], model_info: ModelInfo):
        pass


class MultiFileNoOutputFactory(MultiOutputFactory):
    def __init__(self):
        super().__init__()

    def create_table(self, header: List[str], model_info: ModelInfo):
        return NoOutput()


class CSVMultiFileOutputFactory(MultiOutputFactory):
    def __init__(self, prefix: str, limit_output):
        super().__init__()
        self.prefix = prefix
        self.limit_output = limit_output

    def create_table(self, header: List[str], model_info: ModelInfo):
        return CSVOutput(make_model_file_name(self.prefix, model_info, 'csv'), header, self.limit_output)


class ConsoleTableMultiOutputFactory(MultiOutputFactory):
    def __init__(self, description: str, limit_output):
        super().__init__()
        self.description = description
        self.limit_output = limit_output

    def create_table(self, header: List[str], model_info: ModelInfo):
        return ConsoleTableOutput(header, make_model_console_description(model_info), self.limit_output)


class DataProcessor(ABC):
    def __init__(self, output_factory):
        self.output_factory = output_factory

    @abstractmethod
    def run(self, csv_data: List[Dict[ModelInfo, ModelData]]) -> None:
        pass


class PlotCompareCompileTime(DataProcessor):
    def __init__(self, plot_output: PlotOutput):
        super().__init__(None)
        self.__plot_output = plot_output

    def run(self, csv_data: List[Dict[ModelInfo, ModelData]]) -> None:
        print('comparing compile time ...')
        # CSV files can store different models info
        compile_time_data = list(get_compile_time_data(csv_data))
        comparison_values = get_comparison_values_compile_time(compile_time_data)
        self.__plot_output.plot(comparison_values)


class CompareCompileTime(DataProcessor):
    def __init__(self, output_factory: SingleOutputFactory, summary_stats: bool):
        super().__init__(output_factory)
        self.__summary_stats = summary_stats

    def run(self, csv_data: List[Dict[ModelInfo, ModelData]]) -> None:
        print('comparing compile time in plots ...')
        # CSV files can store different models info
        n_csv_files = len(csv_data)
        compile_time_data = list(get_compile_time_data(csv_data))
        header, table = compare_compile_time(compile_time_data, n_csv_files)
        with self.output_factory.create_table(header) as output:
            output.write(table)

        if self.__summary_stats:
            comparison_values = get_comparison_values_compile_time(compile_time_data)
            print_summary_stats(comparison_values)


class PlotCompareSumTransformationTime(DataProcessor):
    def __init__(self, plot_output: PlotOutput):
        super().__init__(None)
        self.__plot_output = plot_output

    def run(self, csv_data: List[Dict[ModelInfo, ModelData]]) -> None:
        print('comparing sum transformation time plot ...')
        # CSV files can store different models info
        sum_ts_data = list(get_sum_transformation_time_data(csv_data))
        comparison_values = get_comparison_values_sum_transformation_time(sum_ts_data)
        self.__plot_output.plot(comparison_values)


class CompareSumTransformationTime(DataProcessor):
    def __init__(self, output_factory: SingleOutputFactory, summary_stats: bool):
        super().__init__(output_factory)
        self.__summary_stats = summary_stats

    def run(self, csv_data: List[Dict[ModelInfo, ModelData]]) -> None:
        print('comparing sum transformation time ...')
        # CSV files can store different models info
        n_csv_files = len(csv_data)
        sum_ts_data = list(get_sum_transformation_time_data(csv_data))
        header, table = compare_sum_transformation_time(sum_ts_data, n_csv_files)
        with self.output_factory.create_table(header) as output:
            output.write(table)

        if self.__summary_stats:
            comparison_values = get_comparison_values_sum_transformation_time(sum_ts_data)
            print_summary_stats(comparison_values)


class PlotCompareSumPlainManagerTime(DataProcessor):
    def __init__(self, plot_output: PlotOutput):
        super().__init__(None)
        self.__plot_output = plot_output

    def run(self, csv_data: List[Dict[ModelInfo, ModelData]]) -> None:
        print('comparing sum manager plain time plot ...')
        # CSV files can store different models info
        if not csv_data:
            return
        sum_ts_data = list(get_sum_plain_manager_time_data(csv_data))
        comparison_values = get_comparison_values_sum_transformation_time(sum_ts_data)
        self.__plot_output.plot(comparison_values)


class CompareSumPlainManagerTime(DataProcessor):
    def __init__(self, output_factory: SingleOutputFactory, summary_stats: bool):
        super().__init__(output_factory)
        self.__summary_stats = summary_stats

    def run(self, csv_data: List[Dict[ModelInfo, ModelData]]) -> None:
        print('comparing sum plain manager time ...')
        # CSV files can store different models info
        n_csv_files = len(csv_data)
        sum_data = list(get_sum_plain_manager_time_data(csv_data))
        header, table = compare_sum_transformation_time(sum_data, n_csv_files)
        with self.output_factory.create_table(header) as output:
            output.write(table)

        if self.__summary_stats:
            comparison_values = get_comparison_values_sum_transformation_time(sum_data)
            print_summary_stats(comparison_values)


class PlotCompareCompilationAndSumPlainManagerTime(DataProcessor):
    def __init__(self, plot_output: PlotOutput):
        super().__init__(None)
        self.__plot_output = plot_output

    def run(self, csv_data: List[Dict[ModelInfo, ModelData]]) -> None:
        print('comparing compilation and sum plain manager time ...')
        # CSV files can store different models info
        sum_data = list(get_sum_plain_manager_time_data(csv_data))
        sum_data_d = {item[0]: item[1][0] for item in sum_data if item[1]}
        compile_time_data = list(get_compile_time_data(csv_data))
        compile_time_data_d = {item[0]: item[1][0] for item in compile_time_data if item[1]}
        comparison_values = ComparisonValues('sec')
        for model_info in sum_data_d:
            if model_info not in compile_time_data_d:
                continue
            compile_value = compile_time_data_d[model_info]
            if compile_value is None:
                continue
            sum_value = sum_data_d[model_info] / 1_000
            assert sum_value < compile_value
            comparison_values.add(compile_value, sum_value)
        self.__plot_output.plot(comparison_values)


class PlotCompareCompilationAndSumPlainManagerGapTime(DataProcessor):
    def __init__(self, plot_output: PlotOutput):
        super().__init__(None)
        self.__plot_output = plot_output

    def run(self, csv_data: List[Dict[ModelInfo, ModelData]]) -> None:
        print('comparing compilation and sum plain manager time ...')
        # CSV files can store different models info
        sum_data = list(get_sum_plain_manager_gap_time_data(csv_data))
        sum_data_d = {item[0]: item[1][0] for item in sum_data if item[1]}
        compile_time_data = list(get_compile_time_data(csv_data))
        compile_time_data_d = {item[0]: item[1][0] for item in compile_time_data if item[1]}
        comparison_values = ComparisonValues('sec')
        for model_info in sum_data_d:
            if model_info not in compile_time_data_d:
                continue
            compile_value = compile_time_data_d[model_info]
            if compile_value is None:
                continue
            sum_value = sum_data_d[model_info] / 1_000
            assert sum_value < compile_value
            comparison_values.add(compile_value, sum_value)
        self.__plot_output.plot(comparison_values)


class CompareCompilationAndSumPlainManagerTime(DataProcessor):
    def __init__(self, output_factory: SingleOutputFactory):
        super().__init__(output_factory)

    def run(self, csv_data: List[Dict[ModelInfo, ModelData]]) -> None:
        print('comparing compilation and sum plain manager time ...')
        # CSV files can store different models info
        n_csv_files = len(csv_data)
        sum_data = list(get_sum_plain_manager_time_data(csv_data))
        sum_data_d = {item[0]: item[1][0] for item in sum_data if item[1]}
        compile_time_data = list(get_compile_time_data(csv_data))
        compile_time_data_d = {item[0]: item[1][0] for item in compile_time_data if item[1]}
        table_data = []
        comparison_values = ComparisonValues('sec')
        for model_info in sum_data_d:
            if model_info not in compile_time_data_d:
                continue
            compile_value = compile_time_data_d[model_info]
            if compile_value is None:
                continue
            sum_value = sum_data_d[model_info] / 1_000
            assert sum_value <= compile_value
            comparison_values.add(compile_value, sum_value)
            table_data.append((model_info, compile_value, sum_value))
        header, table = compare_compilation_and_plain_manager_sum_time(table_data)
        with self.output_factory.create_table(header) as output:
            output.write(table)


class CompareMemRSS(DataProcessor):
    def __init__(self, output_factory: SingleOutputFactory):
        super().__init__(output_factory)

    def run(self, csv_data: List[Dict[ModelInfo, ModelData]]) -> None:
        print('comparing memory RSS ...')
        # CSV files can store different models info
        n_csv_files = len(csv_data)
        if n_csv_files != 2:
            return
        comparison_values = ComparisonValues('bytes')
        table_data = []
        for model_info, mem_values in join_mem_rss_by_model(csv_data):
            comparison_values.add(mem_values[0], mem_values[1])
            table_data.append((model_info, mem_values[0], mem_values[1]))
        header, table = compare_mem_rss(table_data)
        with self.output_factory.create_table(header) as output:
            output.write(table)


class CompareMemVirtual(DataProcessor):
    def __init__(self, output_factory: SingleOutputFactory):
        super().__init__(output_factory)

    def run(self, csv_data: List[Dict[ModelInfo, ModelData]]) -> None:
        print('comparing virtual memory ...')
        # CSV files can store different models info
        n_csv_files = len(csv_data)
        if n_csv_files != 2:
            return
        comparison_values = ComparisonValues('bytes')
        table_data = []
        for model_info, mem_values in join_mem_virtual_by_model(csv_data):
            comparison_values.add(mem_values[0], mem_values[1])
            table_data.append((model_info, mem_values[0], mem_values[1]))
        header, table = compare_mem_rss(table_data)
        with self.output_factory.create_table(header) as output:
            output.write(table)


class PlotCompareSumPlainManagerGapTime(DataProcessor):
    def __init__(self, plot_output: PlotOutput):
        super().__init__(None)
        self.__plot_output = plot_output

    def run(self, csv_data: List[Dict[ModelInfo, ModelData]]) -> None:
        print('comparing sum manager plain time plot ...')
        # CSV files can store different models info
        sum_ts_data = list(get_sum_plain_manager_gap_time_data(csv_data))
        comparison_values = get_comparison_values_sum_transformation_time(sum_ts_data)
        self.__plot_output.plot(comparison_values)


class CompareSumPlainManagerGapTime(DataProcessor):
    def __init__(self, output_factory: SingleOutputFactory, summary_stats: bool):
        super().__init__(output_factory)
        self.__summary_stats = summary_stats

    def run(self, csv_data: List[Dict[ModelInfo, ModelData]]) -> None:
        print('comparing sum plain manager time ...')
        # CSV files can store different models info
        n_csv_files = len(csv_data)
        sum_data = list(get_sum_plain_manager_gap_time_data(csv_data))
        header, table = compare_sum_transformation_time(sum_data, n_csv_files)
        with self.output_factory.create_table(header) as output:
            output.write(table)

        if self.__summary_stats:
            comparison_values = get_comparison_values_sum_transformation_time(sum_data)
            print_summary_stats(comparison_values)


class GenerateLongestUnitsOverall(DataProcessor):
    def __init__(self, output_factory: SingleOutputFactory, unit_type: str):
        super().__init__(output_factory)
        self.unit_type = unit_type

    def run(self, csv_data: List[Dict[ModelInfo, ModelData]]) -> None:
        print(f'aggregating longest {self.unit_type} overall data ...')
        sum_units_data = get_sum_units_comparison_data(csv_data, self.unit_type)
        sum_units_data_all = join_sum_units(sum_units_data)
        sum_units_by_name = join_sum_units_by_name(sum_units_data_all)
        header, table = get_longest_unit(sum_units_by_name)
        with self.output_factory.create_table(header) as output:
            output.write(table)


class GenerateLongestUnitsPerModel(DataProcessor):
    def __init__(self, output_factory: MultiOutputFactory, unit_type: str):
        super().__init__(output_factory)
        self.unit_type = unit_type

    def run(self, csv_data: List[Dict[ModelInfo, ModelData]]) -> None:
        print(f'aggregating longest {self.unit_type} per model data ...')
        for model_info in get_all_models(csv_data):
            model_data = filter_by_models(csv_data, [model_info])
            sum_units_data = get_sum_units_comparison_data(model_data, self.unit_type)
            sum_units_data_all = join_sum_units(sum_units_data)
            sum_units_by_name = join_sum_units_by_name(sum_units_data_all)
            header, table = get_longest_unit(sum_units_by_name)
            with self.output_factory.create_table(header, model_info) as output:
                output.write(table)


class PlotCompareSumUnitsOverall(DataProcessor):
    def __init__(self, unit_type: str, plot_output: PlotOutput):
        super().__init__(None)
        self.unit_type = unit_type
        self.__plot_output = plot_output

    def run(self, csv_data: List[Dict[ModelInfo, ModelData]]) -> None:
        print(f'compare sum {self.unit_type} overall data plot ...')
        # CSV files can store different models info
        csv_data_common_models = filter_common_models(csv_data)
        if not csv_data_common_models:
            print('no models to get sum units overall ...')
            return
        sum_units_data = get_sum_units_comparison_data(csv_data_common_models, self.unit_type)
        sum_units_data_all = join_sum_units(sum_units_data)
        comparison_values = get_comparison_values_sum_units(sum_units_data_all)
        self.__plot_output.plot(comparison_values)


class CompareSumUnitsOverall(DataProcessor):
    def __init__(self, output_factory: SingleOutputFactory, unit_type: str, summary_stats: bool):
        super().__init__(output_factory)
        self.unit_type = unit_type
        self.__summary_stats = summary_stats

    def run(self, csv_data: List[Dict[ModelInfo, ModelData]]) -> None:
        print(f'compare sum {self.unit_type} overall data ...')
        csv_data_common_models = filter_common_models(csv_data)
        n_csv_files = len(csv_data)
        sum_units_data = get_sum_units_comparison_data(csv_data_common_models, self.unit_type)
        sum_units_data_all = join_sum_units(sum_units_data)
        for model_info, total_list in sum_units_data_all.items():
            assert len(total_list) <= n_csv_files
        header, table = compare_sum_units(sum_units_data_all, n_csv_files)
        with self.output_factory.create_table(header) as output:
            output.write(table)
        if self.__summary_stats:
            comparison_values = get_comparison_values_sum_units(sum_units_data_all)
            print_summary_stats(comparison_values)


class CompareSumUnitsPerModel(DataProcessor):
    def __init__(self, output_factory: MultiOutputFactory,
                 summary_output_factory: Optional[SingleOutputFactory],
                 unit_type: str, plot_output: Optional[PlotOutput]):
        super().__init__(output_factory)
        self.unit_type = unit_type
        self.__summary_output_factory = summary_output_factory
        self.__plot_output = plot_output

    def run(self, csv_data: List[Dict[ModelInfo, ModelData]]) -> None:
        print(f'compare sum {self.unit_type} per model data ...')
        n_csv_files = len(csv_data)
        csv_data_common_models = filter_common_models(csv_data)
        comparison_values_overall = {}
        models = get_all_models(csv_data_common_models)
        for model_info in models:
            model_data = filter_by_models(csv_data_common_models, [model_info])
            sum_units_data = get_sum_units_comparison_data(model_data, self.unit_type)
            sum_units_data_all = join_sum_units(sum_units_data)
            header, table = compare_sum_units(sum_units_data_all, n_csv_files)
            with self.output_factory.create_table(header, model_info) as output:
                output.write(table)
            comparison_values_overall[model_info] = get_comparison_values_sum_units(sum_units_data_all)
            if self.__plot_output:
                prefix = make_model_file_name(f'compare_{self.unit_type}', model_info, '')
                self.__plot_output.plot_into_file(comparison_values_overall[model_info], prefix)
        if self.__summary_output_factory:
            header, table = create_comparison_summary_table(comparison_values_overall)
            with self.__summary_output_factory.create_table(header) as output:
                output.write(table)
        if self.__plot_output and comparison_values_overall:
            unit = next(iter(comparison_values_overall.values())).unit
            combined_comparison_values = ComparisonValues(unit)
            for values in comparison_values_overall.values():
                combined_comparison_values.values1.extend(values.values1)
                combined_comparison_values.values2.extend(values.values2)
            self.__plot_output.plot(combined_comparison_values)


class PlotCompileTimeByIteration(DataProcessor):
    def __init__(self):
        super().__init__(None)

    def run(self, csv_data: List[Dict[ModelInfo, ModelData]]) -> None:
        device = get_device(csv_data)
        for model_info, durations in compile_time_by_iterations(csv_data):
            gen_plot_time_by_iterations('.', device, model_info, durations, 'Compile time', 'compile_time')


class PlotSumTSTimeByIteration(DataProcessor):
    def __init__(self):
        super().__init__(None)

    def run(self, csv_data: List[Dict[ModelInfo, ModelData]]) -> None:
        device = get_device(csv_data)
        for model_info, durations in get_sum_units_durations_by_iteration(csv_data, 'transformation'):
            gen_plot_time_by_iterations('.', device, model_info, durations, 'Sum of transformations', 'sum_ts')


class PlotPlainManagerTimeByIteration(DataProcessor):
    def __init__(self):
        super().__init__(None)

    def run(self, csv_data: List[Dict[ModelInfo, ModelData]]) -> None:
        device = get_device(csv_data)
        for model_info, durations in get_plain_manager_time_by_iteration(csv_data):
            gen_plot_time_by_iterations('.', device, model_info, durations, 'Plain manager time', 'plain_manager_time')


class PlotPlainManagerGapTimeByIteration(DataProcessor):
    def __init__(self):
        super().__init__(None)

    def run(self, csv_data: List[Dict[ModelInfo, ModelData]]) -> None:
        device = get_device(csv_data)
        for model_info, durations in get_plain_manager_gap_time_by_iteration(csv_data):
            gen_plot_time_by_iterations('.', device, model_info, durations, 'Plain manager gap time', 'plain_manager_gap_time')


@dataclass
class Config:
    compare_compile_time = None
    compare_sum_transformation_time = None
    compare_plain_manager_sum_time: Optional[str] = None
    compare_plain_manager_gap_sum_time: Optional[str] = None
    transformations_overall = None
    manager_overall = None
    transformations_per_model = None
    managers_per_model = None
    compare_transformations_overall = None
    compare_managers_overall = None
    compare_transformations_per_model = None
    compare_managers_per_model = None
    output_type = 'csv'
    model_name = None
    limit_output = None
    inputs: List[str] = field(default_factory=list)
    summary_statistics: bool = False
    summary_ratio_histogram: bool = False
    plots: bool = False
    no_csv: bool = False
    n_plot_segments: int = 1
    plot_compile_time_by_iteration: bool = False
    plot_sum_ts_time_by_iteration: bool = False
    plot_compare_compile_time: bool = False
    plot_compare_sum_transformation_time: bool = False
    plot_compare_transformations_overall: bool = False
    plot_compare_plain_manager_sum_time: bool = False
    plot_compare_plain_manager_gap_sum_time: bool = False
    plot_plain_time_by_iteration: bool = False
    plot_plain_gap_time_by_iteration: bool = False
    plot_compare_compile_vs_plain_time: bool = False
    compare_compile_vs_plain_time: Optional[str] = None
    plot_compare_compile_vs_plain_gap_time: bool = False
    compare_mem_rss: Optional[str] = None
    compare_mem_virtual: Optional[str] = None


def parse_args() -> Config:
    script_bin = sys.argv[0]
    args_parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    args_parser.add_argument('--input', type=str,
                             help=f'''input CSV files separated by comma
For example, if you have 2 input CSV files /dir1/file1.csv and /dir2/file2.csv, generated by dev_trigger job, you can specify them
{script_bin} --input /dir1/file1.csv,/dir2/file2.csv
''')
    args_parser.add_argument('--compare_compile_time', nargs='?', type=str, default=None,
                             const='compile_time_comparison',
                             help='compare compile time between input files; for common models between inputs')
    args_parser.add_argument('--compare_sum_transformation_time', nargs='?', type=str, default=None,
                             const='transformation_sum_time_comparison',
                             help='compare sum transformation time between input files; for common models between inputs')
    args_parser.add_argument('--compare_plain_manager_sum_time', nargs='?', type=str, default=None,
                             const='plain_manager_sum_time_comparison',
                             help='compare sum managers plain time between input files; for common models between inputs')
    args_parser.add_argument('--compare_plain_manager_gap_sum_time', nargs='?', type=str, default=None,
                             const='plain_manager_gap_sum_time_comparison',
                             help='compare sum managers plain time between input files; for common models between inputs')
    args_parser.add_argument('--transformations_overall', nargs='?', type=str, default=None,
                             const='transformations_overall',
                             metavar='path to output file',
                             help=f'''aggregate transformations overall models and input CSVs data
For example, if you have 2 input CSV files /dir1/file1.csv and /dir2/file2.csv, generated by dev_trigger job, you can specify them
{script_bin} --inputs /dir1/file1.csv,/dir2/file2.csv --transformations_overall
This option will generate CSV output file transformations_overall.csv with columns
name - name of transformation	
total duration (ms) - total duration (in milliseconds) of all execution of this transformations in both dev_trigger jobs for all models
count of executions - total count of all execution of this transformations in both dev_trigger jobs for all models
You can specify output path, for example
{script_bin} --inputs /dir1/file1.csv,/dir2/file2.csv --transformations_overall /output_dir/output_file.csv
''')
    args_parser.add_argument('--manager_overall', type=str,
                             metavar='path to output file',
                             help=f'''aggregate managers overall data
For example, if you have 2 input CSV files /dir1/file1.csv and /dir2/file2.csv, generated by dev_trigger job, you can specify them
{script_bin} --inputs /dir1/file1.csv,/dir2/file2.csv --manager_overall
This option will generate CSV output file manager_overall.csv with columns
name - name of manager	
total duration (ms) - total duration (in milliseconds) of all execution of this manager in both dev_trigger jobs for all models
count of executions - total count of all execution of this manager in both dev_trigger jobs for all models
You can specify output path, for example
{script_bin} --inputs /dir1/file1.csv,/dir2/file2.csv --manager_overall /output_dir/output_file.csv
''',
                             const='manager_overall', nargs='?', default=None)
    args_parser.add_argument('--transformations_per_model', nargs='?', type=str, default=None,
                             const='transformations_per_model',
                             metavar='output files name prefix',
                             help=f'''aggregate transformations per model data; output in different CSV files
For example, if you have 2 input CSV files /dir1/file1.csv and /dir2/file2.csv, generated by dev_trigger job, you can specify them
{script_bin} --inputs /dir1/file1.csv,/dir2/file2.csv --transformations_per_model
This option will generate CSV multiple output files with names transformations_per_model_<framework>_<model>_<precision>[_<additional_model_attribute>].csv with columns
name - name of transformation
total duration (ms) - total duration (in milliseconds) of all execution of this transformation in both dev_trigger jobs for one particular model
count of executions - total count of all execution of this transformation in both dev_trigger jobs for one particular model
You can specify output names prefix, for example
{script_bin} --inputs /dir1/file1.csv,/dir2/file2.csv --transformations_per_model prefix_to_name
and there will be files prefix_to_name_<framework>_<model>_<precision>[_<additional_model_attribute>].csv
''')
    args_parser.add_argument('--managers_per_model', nargs='?', type=str, default=None,
                             const='managers_per_model',
                             metavar='output files name prefix',
                             help=f'''aggregate managers per model data; output in different CSV files
For example, if you have 2 input CSV files /dir1/file1.csv and /dir2/file2.csv, generated by dev_trigger job, you can specify them
{script_bin} --inputs /dir1/file1.csv,/dir2/file2.csv --managers_per_model
This option will generate CSV multiple output files with names managers_per_model_<framework>_<model>_<precision>[_<additional_model_attribute>].csv with columns
name - name of manager
total duration (ms) - total duration (in milliseconds) of all execution of this manager in both dev_trigger jobs for one particular model
count of executions - total count of all execution of this manager in both dev_trigger jobs for one particular model
You can specify output names prefix, for example
{script_bin} --inputs /dir1/file1.csv,/dir2/file2.csv --managers_per_model prefix_to_name
and there will be files prefix_to_name_<framework>_<model>_<precision>[_<additional_model_attribute>].csv
''')
    args_parser.add_argument('--compare_transformations_overall', nargs='?', type=str, default=None,
                             const='comparison_transformations_overall',
                             metavar='path to output file',
                             help=f'''aggregate transformations overall models data and compare between input CSV files
For example, if you have 2 input CSV files /dir1/file1.csv and /dir2/file2.csv, generated by dev_trigger job, you can specify them
{script_bin} --inputs /dir1/file1.csv,/dir2/file2.csv --compare_transformations_overall
This option will generate CSV output file comparison_transformations_overall.csv with columns
name - name of transformation
duration #1 (ms) - duration (in milliseconds) of all execution of this transformation in first dev_trigger job (/dir1/file1.csv) for all models
duration #2 (ms) - duration (in milliseconds) of all execution of this transformation in second dev_trigger job (/dir2/file2.csv) for all models
duration #2 - #1 (ms) - delta between duration#2 and duration#1  (ms)
duration #2/#1 - ratio duration#2 / duration #1 
count #1 - count of all execution of this transformation in first dev_trigger job (/dir1/file1.csv) for all models
count #2 - count of all execution of this transformation in second dev_trigger job (/dir2/file2.csv) for all models
count #2 - #1 (secs) - delta between count#2 and count#1
It takes into account only those models, that exists in all input CSV files.
You can specify output names prefix, for example
{script_bin} --inputs /dir1/file1.csv,/dir2/file2.csv --compare_transformations_overall /dir3/output3
''')
    args_parser.add_argument('--compare_managers_overall', nargs='?', type=str, default=None,
                             const='compare_managers_overall',
                             metavar='path to output file',
                             help=f'''aggregate managers overall models data and compare between input CSV files
For example, if you have 2 input CSV files /dir1/file1.csv and /dir2/file2.csv, generated by dev_trigger job, you can specify them
{script_bin} --inputs /dir1/file1.csv,/dir2/file2.csv --compare_managers_overall
This option will generate CSV output file comparison_transformations_overall.csv with columns
name - name of manager
duration #1 (ms) - duration (in milliseconds) of all execution of this manager in first dev_trigger job (/dir1/file1.csv) for all models
duration #2 (ms) - duration (in milliseconds) of all execution of this manager in second dev_trigger job (/dir2/file2.csv) for all models
duration #2 - #1 (ms) - delta between duration#2 and duration#1  (ms)
duration #2/#1 - ratio duration#2 / duration #1 
count #1 - count of all execution of this manager in first dev_trigger job (/dir1/file1.csv) for all models
count #2 - count of all execution of this manager in second dev_trigger job (/dir2/file2.csv) for all models
count #2 - #1 (secs) - delta between count#2 and count#1
It takes into account only those models, that exists in all input CSV files.
You can specify output names prefix, for example
{script_bin} --inputs /dir1/file1.csv,/dir2/file2.csv --compare_managers_overall /dir3/output3
''')
    args_parser.add_argument('--compare_transformations_per_model', nargs='?', type=str, default=None,
                             const='compare_transformations',
                             metavar='output files name prefix',
                             help=f'''aggregate transformations per model data and compare between input CSV files
For example, if you have 2 input CSV files /dir1/file1.csv and /dir2/file2.csv, generated by dev_trigger job, you can specify them
{script_bin} --inputs /dir1/file1.csv,/dir2/file2.csv --compare_transformations_per_model
This option will generate CSV multiple output files with names compare_transformations_<framework>_<model>_<precision>[_<additional_model_attribute>].csv with columns
name - name of transformation
duration #1 (ms) - duration (in milliseconds) of all execution of this transformation in first dev_trigger job (/dir1/file1.csv) for one particular model
duration #2 (ms) - duration (in milliseconds) of all execution of this transformation in second dev_trigger job (/dir2/file2.csv) for one particular model
duration #2 - #1 (ms) - delta between duration#2 and duration#1  (ms)
duration #2/#1 - ratio duration#2 / duration #1 
count #1 - count of all execution of this transformation in first dev_trigger job (/dir1/file1.csv) for one particular model
count #2 - count of all execution of this transformation in second dev_trigger job (/dir2/file2.csv) for one particular model
count #2 - #1 (secs) - delta between count#2 and count#1
You can specify output names prefix, for example
{script_bin} --inputs /dir1/file1.csv,/dir2/file2.csv --compare_transformations_per_model prefix_to_name
and there will be files prefix_to_name_<framework>_<model>_<precision>[_<additional_model_attribute>].csv
''')
    args_parser.add_argument('--compare_managers_per_model', nargs='?', type=str, default=None,
                             const='compare_managers',
                             metavar='output files name prefix',
                             help=f'''aggregate managers per model data and compare between input CSV files
For example, if you have 2 input CSV files /dir1/file1.csv and /dir2/file2.csv, generated by dev_trigger job, you can specify them
{script_bin} --inputs /dir1/file1.csv,/dir2/file2.csv --compare_managers_per_model
This option will generate CSV multiple output files with names compare_managers_<framework>_<model>_<precision>[_<additional_model_attribute>].csv with columns
name - name of manager
duration #1 (ms) - duration (in milliseconds) of all execution of this manager in first dev_trigger job (/dir1/file1.csv) for one particular model
duration #2 (ms) - duration (in milliseconds) of all execution of this manager in second dev_trigger job (/dir2/file2.csv) for one particular model
duration #2 - #1 (ms) - delta between duration#2 and duration#1  (ms)
duration #2/#1 - ratio duration#2 / duration #1 
count #1 - count of all execution of this manager in first dev_trigger job (/dir1/file1.csv) for one particular model
count #2 - count of all execution of this manager in second dev_trigger job (/dir2/file2.csv) for one particular model
count #2 - #1 (secs) - delta between count#2 and count#1
You can specify output names prefix, for examples
{script_bin} --inputs /dir1/file1.csv,/dir2/file2.csv --compare_managers_per_model prefix_to_name
and there will be files prefix_to_name_<framework>_<model>_<precision>[_<additional_model_attribute>].csv
''')
    args_parser.add_argument('--model_name', type=str, default=None,
                             help=f"""filter input data by specified model name
If you want to get information only about models with specified model name,
For example to dump only for models with name 'llama-3-8b',
{script_bin} --inputs /dir1/file1.csv,/dir2/file2.csv --compare_managers_per_model --model_name llama-3-8b
""")
    args_parser.add_argument('--output_type', type=str, default='csv',
                             help='csv or console')
    args_parser.add_argument('--limit_output', type=int, default=None,
                             help=f'''
Output maximum number of rows
For example, to output only first 15 rows in table
{script_bin} --inputs /dir1/file1.csv,/dir2/file2.csv --compare_managers_overall --limit_output 15
''')
    args_parser.add_argument('--summary_statistics', action='store_true',
                             help=f'''
Output summary statistics if compare 2 input CSV files
{script_bin} --inputs /dir1/file1.csv,/dir2/file2.csv --compare_compile_time --summary_statistics
''')
    args_parser.add_argument('--plots', action='store_true',
                             help=f'''
Output histograms and scatter plots if compare 2 input CSV files
Is useful with --compare_managers_per_model, --compare_transformations_per_model
{script_bin} --inputs /dir1/file1.csv,/dir2/file2.csv --compare_compile_time --plots
''')
    args_parser.add_argument('--plot_compare_compile_time', action='store_true',
                             help=f'''
Output histograms and scatter plots compilation time comparison if compare 2 input CSV files
{script_bin} --inputs /dir1/file1.csv,/dir2/file2.csv --plot_compare_compile_time
''')
    args_parser.add_argument('--no_csv', action='store_true',
                             help=f'''
Don't generate CSV output files. Is useful with --plots option
{script_bin} --inputs /dir1/file1.csv,/dir2/file2.csv --compare_compile_time --plots --no_csv
''')
    args_parser.add_argument('--n_plot_segments', type=int, default=1,
                             help=f'''
Number of plot segments. Is useful with --plots option
{script_bin} --inputs /dir1/file1.csv,/dir2/file2.csv --compare_compile_time --plots --no_csv --n_plot_segments 3
''')
    args_parser.add_argument('--plot_compile_time_by_iteration', action='store_true',
                             help=f'''
Plot graph with Y - compilation time and X - iteration number
{script_bin} --inputs /dir1/file1.csv,/dir2/file2.csv --plot_compile_time_by_iteration
''')
    args_parser.add_argument('--plot_compare_sum_transformation_time', nargs='?', type=str, default=None,
                             const='transformation_sum_time_comparison',
                             help='compare sum transformation time between input files; for common models between inputs')
    args_parser.add_argument('--plot_compare_plain_manager_sum_time', nargs='?', type=str, default=None,
                             const='manager_plain_sum_time_comparison',
                             help='compare sum manager plain time between input files; for common models between inputs')
    args_parser.add_argument('--plot_compare_plain_manager_gap_sum_time', nargs='?', type=str, default=None,
                             const='manager_plain_gap_sum_time_comparison',
                             help='compare sum manager plain gap time between input files; for common models between inputs')
    args_parser.add_argument('--plot_compare_transformations_overall', nargs='?', type=str, default=None,
                             const='compare_transformations_overall',
                             help='compare sum transformation time between input files; for common models between inputs')
    args_parser.add_argument('--plot_sum_ts_time_by_iteration', action='store_true',
                             help=f'''
Plot graph with Y - sum transformation time and X - iteration number
{script_bin} --inputs /dir1/file1.csv,/dir2/file2.csv --plot_sum_ts_time_by_iteration
''')
    args_parser.add_argument('--plot_plain_time_by_iteration', action='store_true',
                             help=f'''
Plot graph with Y - sum plain manager time and X - iteration number
{script_bin} --inputs /dir1/file1.csv,/dir2/file2.csv --plot_plain_time_by_iteration
''')
    args_parser.add_argument('--plot_plain_gap_time_by_iteration', action='store_true',
                             help=f'''
Plot graph with Y - sum plain manager gap time and X - iteration number
{script_bin} --inputs /dir1/file1.csv,/dir2/file2.csv --plot_plain_gap_time_by_iteration
''')
    args_parser.add_argument('--plot_compare_compile_vs_plain_time', action='store_true',
                             help=f'''
{script_bin} --inputs /dir1/file1.csv,/dir2/file2.csv --plot_compare_compile_vs_plain_time
''')
    args_parser.add_argument('--compare_compile_vs_plain_time', nargs='?', type=str, default=None,
                             const='compare_compile_vs_plain_time',
                             help='compare compile time and plain manager time')
    args_parser.add_argument('--plot_compare_compile_vs_plain_gap_time', action='store_true',
                             help=f'''
{script_bin} --inputs /dir1/file1.csv,/dir2/file2.csv --plot_compare_compile_vs_plain_gap_time
''')
    args_parser.add_argument('--compare_mem_rss', nargs='?', type=str, default=None,
                             const='compare_mem_rss',
                             help='compare RSS memory between 2 input files')
    args_parser.add_argument('--compare_mem_virtual', nargs='?', type=str, default=None,
                             const='compare_mem_virtual',
                             help='compare virtual memory between 2 input files')

    args = args_parser.parse_args()
    if not args.input:
        print('specify input CSV files separated by comma')
        sys.exit(1)


    config = Config()
    config.inputs = args.input.split(',')

    if any(not s for s in config.inputs):
        print('input file cannot be empty')
        sys.exit(1)

    if args.output_type not in ('csv', 'console'):
        raise Exception(f'unknown output type {args.output_type}')
    config.output_type = args.output_type

    config.compare_compile_time = args.compare_compile_time
    config.compare_sum_transformation_time = args.compare_sum_transformation_time
    config.compare_plain_manager_sum_time = args.compare_plain_manager_sum_time
    config.compare_plain_manager_gap_sum_time = args.compare_plain_manager_gap_sum_time
    config.transformations_overall = args.transformations_overall
    config.manager_overall = args.manager_overall
    config.transformations_per_model = args.transformations_per_model
    config.managers_per_model = args.managers_per_model
    config.compare_transformations_overall = args.compare_transformations_overall
    config.compare_managers_overall = args.compare_managers_overall
    config.compare_transformations_per_model = args.compare_transformations_per_model
    config.compare_managers_per_model = args.compare_managers_per_model
    config.model_name = args.model_name
    config.limit_output = args.limit_output
    config.n_plot_segments = args.n_plot_segments
    if args.no_csv:
        config.no_csv = True
    if args.summary_statistics:
        config.summary_statistics = True
    if args.plots:
        config.plots = True
    if args.plot_compile_time_by_iteration:
        config.plot_compile_time_by_iteration = True
    if args.plot_sum_ts_time_by_iteration:
        config.plot_sum_ts_time_by_iteration = True
    if args.plot_compare_compile_time:
        config.plot_compare_compile_time = True
    if args.plot_compare_sum_transformation_time:
        config.plot_compare_sum_transformation_time = True
    if args.plot_compare_transformations_overall:
        config.plot_compare_transformations_overall = True
    if args.plot_compare_plain_manager_sum_time:
        config.plot_compare_plain_manager_sum_time = True
    if args.plot_compare_plain_manager_gap_sum_time:
        config.plot_compare_plain_manager_gap_sum_time = True
    if args.plot_plain_time_by_iteration:
        config.plot_plain_time_by_iteration = True
    if args.plot_plain_gap_time_by_iteration:
        config.plot_plain_gap_time_by_iteration = True
    if args.plot_compare_compile_vs_plain_time:
        config.plot_compare_compile_vs_plain_time = True
    config.compare_compile_vs_plain_time = args.compare_compile_vs_plain_time
    if args.plot_compare_compile_vs_plain_gap_time:
        config.plot_compare_compile_vs_plain_gap_time = True
    config.compare_mem_rss = args.compare_mem_rss
    config.compare_mem_virtual = args.compare_mem_virtual

    return config


def create_single_output_factory(config: Config, path_prefix: str, description: str):
    if config.no_csv:
        return SingleNoOutputFactory()
    if config.output_type == 'csv':
        return CSVSingleFileOutputFactory(path_prefix, config.limit_output)
    return ConsoleTableSingleFileOutputFactory(description, config.limit_output)


def create_multi_output_factory(config: Config, prefix: str, description: str):
    if config.no_csv:
        return MultiFileNoOutputFactory()
    if config.output_type == 'csv':
        return CSVMultiFileOutputFactory(prefix, config.limit_output)
    return ConsoleTableMultiOutputFactory(description, config.limit_output)


def create_summary_output_factory(output_type: str, prefix: str, description: str):
    if output_type == 'csv':
        path = f'{prefix}_summary'
        return CSVSingleFileOutputFactory(path, None)
    return ConsoleTableSingleFileOutputFactory(description, None)


def build_data_processors(config):
    data_processors = []
    if config.compare_compile_time:
        output_factory = create_single_output_factory(config,
                                                      config.compare_compile_time,
                                                      'compilation time')
        data_processors.append(CompareCompileTime(output_factory, config.summary_statistics))

    if config.plot_compare_compile_time:
        path_prefix = config.compare_compile_time
        if not path_prefix:
            path_prefix = 'compilation'
        title_prefix = 'compilation time'
        plot_output_factory = PlotOutput(path_prefix, title_prefix, config.n_plot_segments)
        data_processors.append(PlotCompareCompileTime(plot_output_factory))

    if config.compare_sum_transformation_time:
        output_factory = create_single_output_factory(config,
                                                      config.compare_sum_transformation_time,
                                                      'sum transformation time')
        data_processors.append(CompareSumTransformationTime(output_factory, config.summary_statistics))

    if config.compare_plain_manager_sum_time:
        output_factory = create_single_output_factory(config,
                                                      config.compare_plain_manager_sum_time,
                                                      'sum plain manager time')
        data_processors.append(CompareSumPlainManagerTime(output_factory, config.summary_statistics))

    if config.plot_compare_plain_manager_sum_time:
        path_prefix = config.compare_plain_manager_sum_time
        if not path_prefix:
            path_prefix = 'sum_plain_manager'
        title_prefix = 'sum plain manager time'
        plot_output_factory = PlotOutput(path_prefix, title_prefix, config.n_plot_segments)
        data_processors.append(PlotCompareSumPlainManagerTime(plot_output_factory))

    if config.compare_plain_manager_gap_sum_time:
        output_factory = create_single_output_factory(config,
                                                      config.compare_plain_manager_gap_sum_time,
                                                      'sum plain manager gap time')
        data_processors.append(CompareSumPlainManagerGapTime(output_factory, config.summary_statistics))

    if config.plot_compare_sum_transformation_time:
        path_prefix = config.compare_sum_transformation_time
        if not path_prefix:
            path_prefix = 'sum_ts'
        title_prefix = 'sum transformation time'
        plot_output_factory = PlotOutput(path_prefix, title_prefix, config.n_plot_segments)
        data_processors.append(PlotCompareSumTransformationTime(plot_output_factory))

    if config.plot_compare_plain_manager_gap_sum_time:
        path_prefix = config.compare_plain_manager_gap_sum_time
        if not path_prefix:
            path_prefix = 'sum_plain_manager_gap'
        title_prefix = 'sum plain manager gap time'
        plot_output_factory = PlotOutput(path_prefix, title_prefix, config.n_plot_segments)
        data_processors.append(PlotCompareSumPlainManagerGapTime(plot_output_factory))

    if config.transformations_overall:
        output_factory = create_single_output_factory(config,
                                                      config.transformations_overall,
                                                      'transformations overall')
        data_processors.append(GenerateLongestUnitsOverall(output_factory, unit_type='transformation'))
    if config.manager_overall:
        output_factory = create_single_output_factory(config,
                                                      config.manager_overall,
                                                      'managers overall')
        data_processors.append(GenerateLongestUnitsOverall(output_factory, unit_type='manager'))
    if config.transformations_per_model:
        output_factory = create_multi_output_factory(config,
                                                     config.transformations_per_model,
                                                     'transformations per model')
        data_processors.append(GenerateLongestUnitsPerModel(output_factory, unit_type='transformation'))
    if config.managers_per_model:
        output_factory = create_multi_output_factory(config,
                                                     config.managers_per_model,
                                                     'managers per model')
        data_processors.append(GenerateLongestUnitsPerModel(output_factory, unit_type='manager'))
    if config.compare_transformations_overall:
        output_factory = create_single_output_factory(config,
                                                      config.compare_transformations_overall,
                                                      'compare transformations overall')
        data_processors.append(CompareSumUnitsOverall(output_factory, unit_type='transformation',
                                                      summary_stats=config.summary_statistics))
    if config.plot_compare_transformations_overall:
        path_prefix = config.compare_transformations_overall
        if not path_prefix:
            path_prefix = 'ts_overall'
        title_prefix = 'transformations overall time'
        plot_output_factory = PlotOutput(path_prefix, title_prefix, config.n_plot_segments)
        data_processors.append(PlotCompareSumUnitsOverall(unit_type='transformation', plot_output=plot_output_factory))

    if config.compare_managers_overall:
        output_factory = create_single_output_factory(config,
                                                      config.compare_managers_overall,
                                                      'compare managers overall')
        data_processors.append(CompareSumUnitsOverall(output_factory, unit_type='manager',
                                                      summary_stats=config.summary_statistics))
    if config.compare_transformations_per_model:
        output_factory = create_multi_output_factory(config,
                                                     config.compare_transformations_per_model,
                                                     'compare transformations per model')
        summary_output_factory = None
        if config.summary_statistics:
            summary_output_factory = create_summary_output_factory(config.output_type,
                                                                   config.compare_transformations_per_model,
                                                                   'compare transformations per model')
        plot_output_factory = None
        if config.plots:
            path_prefix = config.compare_transformations_per_model
            if not path_prefix:
                path_prefix = 'compare_ts'
            title_prefix = 'compare transformations time'
            plot_output_factory = PlotOutput(path_prefix, title_prefix, config.n_plot_segments)
        data_processors.append(CompareSumUnitsPerModel(output_factory, summary_output_factory,
                                                       unit_type='transformation', plot_output=plot_output_factory))
    if config.compare_managers_per_model:
        output_factory = create_multi_output_factory(config,
                                                     config.compare_managers_per_model,
                                                     'compare managers per model')
        summary_output_factory = None
        if config.summary_statistics:
            summary_output_factory = create_summary_output_factory(config.output_type,
                                                                   config.compare_managers_per_model,
                                                                   'compare managers per model')
        plot_output_factory = None
        if config.plots:
            path_prefix = config.compare_managers_per_model
            if not path_prefix:
                path_prefix = 'compare_managers'
            title_prefix = 'compare managers time'
            plot_output_factory = PlotOutput(path_prefix, title_prefix, config.n_plot_segments)
        data_processors.append(CompareSumUnitsPerModel(output_factory, summary_output_factory, unit_type='manager',
                                                       plot_output=plot_output_factory))
    if config.plot_compile_time_by_iteration:
        data_processors.append(PlotCompileTimeByIteration())
    if config.plot_sum_ts_time_by_iteration:
        data_processors.append(PlotSumTSTimeByIteration())
    if config.plot_plain_time_by_iteration:
        data_processors.append(PlotPlainManagerTimeByIteration())
    if config.plot_plain_gap_time_by_iteration:
        data_processors.append(PlotPlainManagerGapTimeByIteration())

    if config.plot_compare_compile_vs_plain_time:
        path_prefix = 'compare_compile_plain'
        if not path_prefix:
            path_prefix = 'compare_compile_plain'
        title_prefix = ''
        plot_output_factory = PlotOutput(path_prefix, title_prefix, config.n_plot_segments)
        plot_ratio = PlotOutputRatioSimple()
        plot_ratio.set_label('Transformations time / Compile Time, %')
        plot_output_factory.set_get_ratio_func(plot_ratio)
        plot_output_factory.set_label_y_hist('Number of Models')
        plot_output_factory.set_label_x_scatter('Compile Time, sec')
        data_processors.append(PlotCompareCompilationAndSumPlainManagerTime(plot_output_factory))

    if config.compare_compile_vs_plain_time:
        output_factory = create_single_output_factory(config,
                                                      'compare_compile_plain_manager',
                                                      'compare compile and plain manager time')
        data_processors.append(CompareCompilationAndSumPlainManagerTime(output_factory))

    if config.plot_compare_compile_vs_plain_gap_time:
        path_prefix = 'compare_compile_plain_gap'
        if not path_prefix:
            path_prefix = 'compare_compile_plain_gap'
        title_prefix = ''
        plot_output_factory = PlotOutput(path_prefix, title_prefix, config.n_plot_segments)
        plot_ratio = PlotOutputRatioSimple()
        plot_ratio.set_label('Transformations time / Compile Time, %')
        plot_output_factory.set_get_ratio_func(plot_ratio)
        plot_output_factory.set_label_y_hist('Number of Models')
        plot_output_factory.set_label_x_scatter('Compile Time, sec')
        data_processors.append(PlotCompareCompilationAndSumPlainManagerGapTime(plot_output_factory))

    if config.compare_mem_rss:
        output_factory = create_single_output_factory(config,
                                                      'compare_mem_rss',
                                                      'compare memory rss')
        data_processors.append(CompareMemRSS(output_factory))

    if config.compare_mem_virtual:
        output_factory = create_single_output_factory(config,
                                                      'compare_mem_virtual',
                                                      'compare memory virtual')
        data_processors.append(CompareMemVirtual(output_factory))

    return data_processors


def main(config: Config) -> None:
    data_processors = build_data_processors(config)

    if not data_processors:
        print('nothing to do ...')
        return

    csv_data = get_csv_data(get_input_csv_files(config.inputs))
    if config.model_name:
        csv_data = filter_by_model_name(csv_data, config.model_name)
    if not csv_data:
        print('no data to process ...')
        return
    for proc in data_processors:
        proc.run(csv_data)


if __name__ == '__main__':
    main(parse_args())
