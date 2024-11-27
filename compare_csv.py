from __future__ import annotations

from abc import ABC, abstractmethod
import argparse
import csv
from dataclasses import dataclass, field
import sys
import os
from tabulate import tabulate
from typing import List, Dict, Set, Optional, Iterator

import numpy as np

import plot_utils
from parse_input import get_csv_data
from common_structs import Unit, ModelData, ModelInfo, ComparisonValues, full_join_by_model_info
from table import compare_compile_time, compare_sum_transformation_time, get_longest_unit, compare_sum_units, \
    create_comparison_summary_table


def get_common_models(data: List[Dict[ModelInfo, ModelData]]) -> List[ModelInfo]:
    if len(data) == 0:
        return []
    common_keys = data[0].keys()
    for csv_data in data:
        common_keys &= csv_data.keys()
    return list(common_keys)


def filter_by_models(data: List[Dict[ModelInfo, ModelData]],
                     models: List[ModelInfo]) -> List[Dict[ModelInfo, ModelData]]:
    new_data = []
    for csv_data in data:
        new_dict = {}
        for model_info in models:
            if model_info in csv_data:
                new_dict[model_info] = csv_data[model_info]
        if new_dict:
            new_data.append(new_dict)
    return new_data


def filter_by_model_name(data: List[Dict[ModelInfo, ModelData]],
                         model_name: str) -> List[Dict[ModelInfo, ModelData]]:
    new_data = []
    for csv_data in data:
        new_dict = {}
        for model_info in csv_data:
            if model_info.name != model_name:
                continue
            new_dict[model_info] = csv_data[model_info]
        if new_dict:
            new_data.append(new_dict)
    return new_data


def filter_common_models(data: List[Dict[ModelInfo, ModelData]]) -> List[Dict[ModelInfo, ModelData]]:
    common_models: List[ModelInfo] = get_common_models(data)
    return filter_by_models(data, common_models)


def print_summary_stats(values: ComparisonValues):
    stats = values.get_stats()
    header = ['value', 'median', 'mean', 'std', 'max abs']
    rows = [{'value': f'delta ({stats.unit})',
             'median': f'{stats.delta_median:.2f}',
             'mean': f'{stats.delta_mean:.2f}',
             'std': f'{stats.delta_std:.2f}',
             'max abs': f'{stats.delta_max_abs:.2f}',
             }, {'value': 'ratio (%)',
                 'median': f'{stats.ratio_median:.2f}',
                 'mean': f'{stats.ratio_mean:.2f}',
                 'std': f'{stats.ratio_std:.2f}',
                 'max abs': f'{stats.ratio_max_abs:.2f}'}]
    ordered_rows_str = [{key: str(row[key]) for key in header} for row in rows]
    print(tabulate(ordered_rows_str, headers="keys"))


def get_device(data: List[Dict[ModelInfo, ModelData]]) -> str:
    if not data:
        return ''
    key = next(iter(data[0]))
    return data[0][key].get_device()


def get_all_models(data: List[Dict[ModelInfo, ModelData]]) -> Set[ModelInfo]:
    return set(model_info for csv_data in data for model_info in csv_data)


def make_model_file_name(prefix: str, model_info: ModelInfo, extension: str) -> str:
    name = []
    if prefix:
        name = [prefix]
    name.extend([model_info.framework,
                 model_info.name,
                 model_info.precision])
    if model_info.config:
        name.append(model_info.config)
    name = '_'.join(name)
    if extension:
        name = name + '.' + extension
    return name


def make_model_console_description(model_info: ModelInfo) -> str:
    name = [model_info.framework,
            model_info.name,
            model_info.precision]
    if model_info.config:
        name.append(model_info.config)
    return ' '.join(name)


def find_iqr_outlier_indexes(values) -> Set[int]:
    q1 = np.percentile(values, 25)
    q3 = np.percentile(values, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    indexes = (i for i, x in enumerate(values) if x < lower_bound or x > upper_bound)
    return set(indexes)


class PlotOutput:
    def __init__(self, path_prefix, title_prefix: str, n_segments: int):
        self.path_prefix = path_prefix
        self.title_prefix = title_prefix
        self.n_segments = n_segments

    def plot_for_model(self, model_info: ModelInfo, values: ComparisonValues):
        prefix = make_model_file_name(self.path_prefix, model_info, '')
        self.plot_into_file(values, prefix)

    def plot_into_file(self, values: ComparisonValues, prefix: str):
        ratios = values.get_ratios()
        max_values = values.get_max_values()
        assert len(ratios) == len(max_values)

        ratio_outlier_indexes = find_iqr_outlier_indexes(ratios)
        log_numbers = np.log(max_values)
        min_log = np.min(log_numbers)
        max_log = np.max(log_numbers)
        bins = np.linspace(min_log, max_log, self.n_segments  + 1)
        indices = np.digitize(log_numbers, bins)

        for i in range(1, self.n_segments  + 1):
            x_part = []
            y_part = []
            n_outliers = 0
            for j in range(len(max_values)):
                if indices[j] != i:
                    continue
                if j in ratio_outlier_indexes:
                    n_outliers += 1
                    continue
                x_part.append(max_values[j])
                y_part.append(ratios[j])

            if not x_part:
                continue

            y_values = np.array(y_part)
            y_median = float(np.median(y_values))

            outliers_percent = n_outliers / len(max_values) * 100.0
            title = f'{self.title_prefix} ratio part {i} without outliers ({outliers_percent:.2f} %)'
            scatter_path = f'{prefix}_scatter_part{i}.png'

            y_label = f'ratio (value #2/value #1 - 1), %'
            x_label = f'max (value #1, value #2), {values.unit}'
            scatter = plot_utils.ScatterPlot(title, x_label, y_label)
            scatter.set_values(x_part, y_part)
            scatter.add_horizontal_line(y_median, f'median {y_median:.2f} %')
            scatter.plot(scatter_path)

    def plot(self, values: ComparisonValues):
        self.plot_into_file(values, self.path_prefix)


class Output(ABC):
    def __init__(self, header: List[str]):
        self.header = header

    @abstractmethod
    def __enter__(self):
        return self

    @abstractmethod
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    @abstractmethod
    def write(self, row: List[Dict[str, str]]):
        pass


class NoOutput(Output):
    def __init__(self):
        super().__init__([])

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def write(self, row: List[Dict[str, str]]):
        pass


class CSVOutput(Output):
    def __init__(self, path: str, header: List[str], limit_output):
        super().__init__(header)
        self.path = path
        self.limit_output = limit_output
        self.file = None

    def write(self, rows: List[Dict[str, str]]):
        assert self.file is not None
        assert self.header is not None
        csv_writer = csv.DictWriter(self.file, fieldnames=self.header, delimiter=';')
        csv_writer.writeheader()
        if self.limit_output:
            rows = rows[:self.limit_output]
        for row in rows:
            csv_writer.writerow(row)

    def __enter__(self):
        self.file = open(self.path, 'w', newline='')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file:
            self.file.close()


class ConsoleTableOutput(Output):
    def __init__(self, header: List[str], description: str, limit_output):
        super().__init__(header)
        self.description = description
        self.limit_output = limit_output
        self.file = None

    def write(self, rows: List[Dict[str, str]]):
        assert self.header is not None
        if self.limit_output:
            rows = rows[:self.limit_output]
        print(self.description)
        ordered_rows_str = [{key: str(row[key]) for key in self.header} for row in rows]
        table = tabulate(ordered_rows_str, headers="keys")
        print(table)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


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


class CompareCompileTime(DataProcessor):
    def __init__(self, output_factory: SingleOutputFactory, summary_stats: bool, plot_output: Optional[PlotOutput]):
        super().__init__(output_factory)
        self.__summary_stats = summary_stats
        self.__plot_output = plot_output

    def run(self, csv_data: List[Dict[ModelInfo, ModelData]]) -> None:
        print('comparing compile time ...')
        # CSV files can store different models info
        if not csv_data:
            print('no common models to compare compilation time ...')
            return
        header, table, comparison_values = compare_compile_time(csv_data)
        if self.__summary_stats:
            print_summary_stats(comparison_values)
        if self.__plot_output:
            self.__plot_output.plot(comparison_values)
        with self.output_factory.create_table(header) as output:
            output.write(table)


class CompareSumTransformationTime(DataProcessor):
    def __init__(self, output_factory: SingleOutputFactory, summary_stats: bool, plot_output: Optional[PlotOutput]):
        super().__init__(output_factory)
        self.__summary_stats = summary_stats
        self.__plot_output = plot_output

    def run(self, csv_data: List[Dict[ModelInfo, ModelData]]) -> None:
        print('comparing sum transformation time ...')
        # CSV files can store different models info
        if not csv_data:
            print('no common models to compare compilation time ...')
            return
        header, table, comparison_values = compare_sum_transformation_time(csv_data)
        if self.__summary_stats:
            print_summary_stats(comparison_values)
        if self.__plot_output:
            self.__plot_output.plot(comparison_values)
        with self.output_factory.create_table(header) as output:
            output.write(table)


class GenerateLongestUnitsOverall(DataProcessor):
    def __init__(self, output_factory: SingleOutputFactory, unit_type: str):
        super().__init__(output_factory)
        self.unit_type = unit_type

    def run(self, csv_data: List[Dict[ModelInfo, ModelData]]) -> None:
        print(f'aggregating longest {self.unit_type} overall data ...')
        header, table = get_longest_unit(csv_data, self.unit_type)
        with self.output_factory.create_table(header) as output:
            output.write(table)


class GenerateLongestUnitsPerModel(DataProcessor):
    def __init__(self, output_factory: MultiOutputFactory, unit_type: str):
        super().__init__(output_factory)
        self.unit_type = unit_type

    def run(self, csv_data: List[Dict[ModelInfo, ModelData]]) -> None:
        print(f'aggregating longest {self.unit_type} per model data ...')
        for model_info in get_all_models(csv_data):
            header, table = get_longest_unit(filter_by_models(csv_data, [model_info]), self.unit_type)
            with self.output_factory.create_table(header, model_info) as output:
                output.write(table)


class CompareSumUnitsOverall(DataProcessor):
    def __init__(self, output_factory: SingleOutputFactory, unit_type: str, summary_stats: bool, plot_output: Optional[PlotOutput]):
        super().__init__(output_factory)
        self.unit_type = unit_type
        self.__summary_stats = summary_stats
        self.__plot_output = plot_output

    def run(self, csv_data: List[Dict[ModelInfo, ModelData]]) -> None:
        print(f'compare sum {self.unit_type} overall data ...')
        csv_data_common_models = filter_common_models(csv_data)
        header, table, comparison_values = compare_sum_units(csv_data_common_models, self.unit_type)
        if self.__summary_stats:
            print_summary_stats(comparison_values)
        if self.__plot_output:
            self.__plot_output.plot(comparison_values)
        with self.output_factory.create_table(header) as output:
            output.write(table)


class ProgressStatus:
    def __init__(self, total_iterations: int):
        self.total_iterations = total_iterations
        self.current_iteration = 0
        self.last_length = 0

    def update(self, status: str):
        self.current_iteration += 1
        percent_complete = (self.current_iteration / self.total_iterations) * 100
        message = f'Progress: {percent_complete:.2f}% - {status}'
        sys.stdout.write('\r' + ' ' * self.last_length)
        sys.stdout.write('\r' + message)
        sys.stdout.flush()
        self.last_length = len(message)

    def complete(self):
        sys.stdout.write('\n')
        sys.stdout.flush()


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
        csv_data_common_models = filter_common_models(csv_data)
        comparison_values_overall = {}
        models = get_all_models(csv_data_common_models)
        progress_status = ProgressStatus(len(models))
        for model_info in models:
            progress_status.update(f'{model_info.framework} {model_info.name} {model_info.precision} {model_info.config}')
            header, table, comparison_values = compare_sum_units(filter_by_models(csv_data,[model_info]),
                                                                 self.unit_type)
            comparison_values_overall[model_info] = comparison_values
            with self.output_factory.create_table(header, model_info) as output:
                output.write(table)
            if self.__plot_output:
                self.__plot_output.plot_for_model(model_info, comparison_values)
        progress_status.complete()
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


def get_compile_durations(model_data_items: Iterator[ModelData]) -> Iterator[List[float]]:
    return (model_data.get_compile_durations() for model_data in model_data_items if model_data is not None)


def compile_time_by_iterations(csv_data: List[Dict[ModelInfo, ModelData]]) -> Iterator[Tuple[ModelInfo, Iterator[List[float]]]]:
    for model_info, model_data_items in full_join_by_model_info(csv_data):
        yield model_info, get_compile_durations(model_data_items)
    return


def gen_compile_time_by_iterations_one_common_median(output_dir: str,
                                                     device: str, model_info: ModelInfo, model_data_items: Iterator[List[float]],
                                                     what: str, file_prefix: str):
    title = f'{what} {device} {model_info.framework} {model_info.name} {model_info.precision}'
    if model_info.config:
        title += f' {model_info.config}'
    x_label = 'iteration number'
    y_label = f'{what} (seconds)'

    plot = plot_utils.Plot(title, x_label, y_label)
    plot.set_x_ticks_func(plot_utils.generate_x_ticks_cast_to_int)

    all_compile_time_values = []
    for durations in model_data_items:
        compile_time_values = [float(duration) / 1_000_000_000 for duration in durations]
        iterations = [i for i in range(1, len(compile_time_values) + 1)]
        all_compile_time_values.extend(compile_time_values)

        assert len(compile_time_values) == len(iterations)
        plot.add(iterations, compile_time_values)

    if not all_compile_time_values:
        print(f'no values for {model_info}')
        return

    # Calculate the median value of y_values
    median_value = float(np.median(all_compile_time_values))
    plot.append_x_line(median_value, f'Median: {"%.2f" % median_value} seconds', 'red', '--')

    # maximum deviation from median in %
    max_deviation_abs = max((item for item in all_compile_time_values), key=lambda e: abs(e - median_value))
    max_deviation = abs(median_value - max_deviation_abs) * 100.0 / median_value

    if max_deviation > 1.0:
        # Calculate 10% deviation from the median
        deviation = 0.01 * median_value
        lower_bound = median_value - deviation
        upper_bound = median_value + deviation
        plot.set_stripe(lower_bound, upper_bound, label='1% deviation from the median')

    path = os.path.join(output_dir, f'{file_prefix}_{device}_{model_info.framework}_{model_info.name}_{model_info.precision}')
    if model_info.config:
        path += f'_{model_info.config}'
    path += '.png'
    plot.plot(path)


class PlotCompileTimeByIteration(DataProcessor):
    def __init__(self):
        super().__init__(None)

    def run(self, csv_data: List[Dict[ModelInfo, ModelData]]) -> None:
        device = get_device(csv_data)
        for model_info, durations in compile_time_by_iterations(csv_data):
            gen_compile_time_by_iterations_one_common_median('.', device, model_info, durations, 'Compile time', 'compile_time')


@dataclass
class Config:
    compare_compile_time = None
    compare_sum_transformation_time = None
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
                             const='transformation_sum_time_comparison.csv',
                             help='compare sum transformation time between input files; for common models between inputs')
    args_parser.add_argument('--transformations_overall', nargs='?', type=str, default=None,
                             const='transformations_overall.csv',
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
                             const='manager_overall.csv', nargs='?', default=None)
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
{script_bin} --inputs /dir1/file1.csv,/dir2/file2.csv --compare_compile_time --plots
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
    args_parser.add_argument('--plot_compile_time_by_iteration', type=int, default=1,
                             help=f'''
Plot graph with Y - compilation time and X - iteration number
{script_bin} --inputs /dir1/file1.csv,/dir2/file2.csv --plot_compile_time_by_iteration
''')
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


def main(config: Config) -> None:
    data_processors = []
    if config.compare_compile_time:
        output_factory = create_single_output_factory(config,
                                                      config.compare_compile_time,
                                                      'compilation time')
        plot_output_factory = None
        if config.plots:
            path_prefix = config.compare_compile_time
            if not path_prefix:
                path_prefix = 'compilation'
            title_prefix = 'compilation time'
            plot_output_factory = PlotOutput(path_prefix, title_prefix, config.n_plot_segments)
        data_processors.append(CompareCompileTime(output_factory, config.summary_statistics,
                                                  plot_output_factory))
    if config.compare_sum_transformation_time:
        output_factory = create_single_output_factory(config,
                                                      config.compare_sum_transformation_time,
                                                      'sum transformation time')
        plot_output_factory = None
        if config.plots:
            path_prefix = config.compare_sum_transformation_time
            if not path_prefix:
                path_prefix = 'sum_ts'
            title_prefix = 'sum transformation time'
            plot_output_factory = PlotOutput(path_prefix, title_prefix, config.n_plot_segments)
        data_processors.append(CompareSumTransformationTime(output_factory, config.summary_statistics,
                                                            plot_output_factory))
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
        plot_output_factory = None
        if config.plots:
            path_prefix = config.compare_transformations_overall
            if not path_prefix:
                path_prefix = 'ts_overall'
            title_prefix = 'transformations overall time'
            plot_output_factory = PlotOutput(path_prefix, title_prefix, config.n_plot_segments)
        data_processors.append(CompareSumUnitsOverall(output_factory, unit_type='transformation',
                                                      summary_stats=config.summary_statistics,
                                                      plot_output=plot_output_factory))
    if config.compare_managers_overall:
        output_factory = create_single_output_factory(config,
                                                      config.compare_managers_overall,
                                                      'compare managers overall')
        plot_output_factory = None
        if config.plots:
            path_prefix = config.compare_managers_overall
            if not path_prefix:
                path_prefix = 'managers_overall'
            title_prefix = 'compare managers overall time'
            plot_output_factory = PlotOutput(path_prefix, title_prefix, config.n_plot_segments)
        data_processors.append(CompareSumUnitsOverall(output_factory, unit_type='manager',
                                                      summary_stats=config.summary_statistics,
                                                      plot_output=plot_output_factory))
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

    if not data_processors:
        print('nothing to do ...')
        return

    csv_data = get_csv_data(config.inputs)
    if config.model_name:
        csv_data = filter_by_model_name(csv_data, config.model_name)
    for proc in data_processors:
        proc.run(csv_data)


if __name__ == '__main__':
    Unit.USE_ONLY_0_ITER_GPU = True
    Unit.USE_ONLY_0_ITER = False
    main(parse_args())
