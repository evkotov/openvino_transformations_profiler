from __future__ import annotations

from abc import ABC, abstractmethod
import argparse
from collections import namedtuple
from dataclasses import dataclass, field
import sys
from typing import List, Dict, Optional, Tuple, Iterator

import numpy as np

from ov_ts_profiler.output_utils import print_summary_stats, make_model_file_name, NoOutput, CSVOutput, ConsoleTableOutput
from ov_ts_profiler.parse_input import get_csv_data, get_input_csv_files
from ov_ts_profiler.common_structs import ModelData, ModelInfo, ComparisonValues, make_model_console_description, full_join_by_model_info, \
    Unit, get_measurement_date
from ov_ts_profiler.plot_utils import PlotOutput, gen_plot_time_by_iterations, PlotOutputRatioSimple, gen_plot_debug_items, \
    gen_CompareCompileTimeWithBenchmarking, gen_plot_by_date, Hist, ScatterPlot, gen_plot_key_value_float, \
    gen_plot_scatter_colors
from ov_ts_profiler.stat_utils import filter_by_models, filter_by_model_name, filter_common_models, get_device, \
    get_all_models, \
    compile_time_by_iterations, get_sum_units_durations_by_iteration, get_compile_time_data, \
    get_sum_transformation_time_data, get_sum_units_comparison_data, join_sum_units, join_sum_units_by_name, \
    get_comparison_values_compile_time, get_comparison_values_sum_transformation_time, \
    get_comparison_values_sum_units, get_sum_plain_manager_time_data, get_sum_plain_manager_gap_time_data, \
    get_plain_manager_time_by_iteration, get_plain_manager_gap_time_by_iteration, \
    join_mem_rss_by_model, join_mem_virtual_by_model, get_debug_mem_rss, get_debug_vmpeak, get_debug_mem_rss_and_shared
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
            gen_plot_time_by_iterations('.', device, model_info, durations, 'Compile time', 'compile_time_by_iteration')


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
            gen_plot_time_by_iterations('.', device, model_info, durations, 'Plain manager time', 'plain_manager_time_by_iteration')


class PlotPlainManagerGapTimeByIteration(DataProcessor):
    def __init__(self):
        super().__init__(None)

    def run(self, csv_data: List[Dict[ModelInfo, ModelData]]) -> None:
        device = get_device(csv_data)
        for model_info, durations in get_plain_manager_gap_time_by_iteration(csv_data):
            gen_plot_time_by_iterations('.', device, model_info, durations, 'Plain manager gap time', 'plain_manager_gap_time')


class PlotMemRSSDebug(DataProcessor):
    def __init__(self):
        super().__init__(None)

    def run(self, csv_data: List[Dict[ModelInfo, ModelData]]) -> None:
        device = get_device(csv_data)
        for model_info, mem_rss in get_debug_mem_rss(csv_data):
            values = [[(unit.get_status()['timestamp_ns'], unit.get_status()['rss_bytes_used']) for unit in mem_rss_one_csv] for mem_rss_one_csv in mem_rss]
            # normalize timestamps
            min_timestamp = min((pair[0] for single_csv_values in values for pair in single_csv_values))
            values = [[(pair[0] - min_timestamp, pair[1]) for pair in single_csv_values] for single_csv_values in values]
            # to seconds
            values = [[(pair[0] / 1_000_000_000, pair[1]) for pair in single_csv_values] for single_csv_values in values]
            # to megabytes
            values = [[(pair[0], pair[1] / (1024 * 1024)) for pair in single_csv_values] for single_csv_values in values]
            gen_plot_debug_items('.', device, model_info, values, 'memory consumption RSS', 'mem_rss_debug')


class PlotVMpeakDebug(DataProcessor):
    def __init__(self):
        super().__init__(None)

    def run(self, csv_data: List[Dict[ModelInfo, ModelData]]) -> None:
        device = get_device(csv_data)
        for model_info, mem_rss in get_debug_vmpeak(csv_data):
            values = [[(unit.get_status()['timestamp_ns'], unit.get_status()['bytes_used']) for unit in mem_rss_one_csv] for mem_rss_one_csv in mem_rss]
            print(values)
            # normalize timestamps
            min_timestamp = min((pair[0] for single_csv_values in values for pair in single_csv_values))
            values = [[(pair[0] - min_timestamp, pair[1]) for pair in single_csv_values] for single_csv_values in values]
            # to seconds
            values = [[(pair[0] / 1_000_000_000, pair[1]) for pair in single_csv_values] for single_csv_values in values]
            # to megabytes
            values = [[(pair[0], pair[1] / (1024 * 1024)) for pair in single_csv_values] for single_csv_values in values]
            gen_plot_debug_items('.', device, model_info, values, 'memory consumption RSS', 'mem_vmpeak')


class PlotMemRSSAndSharedDebug(DataProcessor):
    def __init__(self):
        super().__init__(None)

    def run(self, csv_data: List[Dict[ModelInfo, ModelData]]) -> None:
        device = get_device(csv_data)
        for model_info, mem_rss in get_debug_mem_rss_and_shared(csv_data):
            values = [[(unit.get_status()['timestamp_ns'], unit.get_status()['bytes_used']) for unit in mem_rss_one_csv] for mem_rss_one_csv in mem_rss]
            # normalize timestamps
            min_timestamp = min((pair[0] for single_csv_values in values for pair in single_csv_values))
            values = [[(pair[0] - min_timestamp, pair[1]) for pair in single_csv_values] for single_csv_values in values]
            # to seconds
            values = [[(pair[0] / 1_000_000_000, pair[1]) for pair in single_csv_values] for single_csv_values in values]
            # to megabytes
            values = [[(pair[0], pair[1] / (1024 * 1024)) for pair in single_csv_values] for single_csv_values in values]
            gen_plot_debug_items('.', device, model_info, values, 'memory consumption RSS', 'mem_rss_and_shared_debug')


class PlotCompareCompileTimeWithBenchmarking(DataProcessor):
    def __init__(self):
        super().__init__(None)

    def get_compile_time_data(self, data: List[Dict[ModelInfo, ModelData]]) -> Iterator[Tuple[ModelInfo, List[Optional[float]]]]:
        for model_info, model_data_items in full_join_by_model_info(data):
            compile_times = [
                (model_data.get_compile_durations()[0] / 1_000_000_000 if model_data is not None and len(model_data.get_compile_durations()) != 0 else None)
                for model_data in model_data_items
            ]
            yield model_info, compile_times

    def run(self, csv_data: List[Dict[ModelInfo, ModelData]]) -> None:
        print('comparing compile time ...')
        device = get_device(csv_data)
        # CSV files can store different models info
        for model_info, model_data_iter in self.get_compile_time_data(csv_data):
            model_data_list = list(model_data_iter)
            gen_CompareCompileTimeWithBenchmarking('.', device, model_info, [model_data_list], 'compilation', 'compilation')



class PlotMemRSS(DataProcessor):
    def __init__(self):
        super().__init__(None)

    def run(self, csv_data: List[Dict[ModelInfo, ModelData]]) -> None:
        device = get_device(csv_data)
        for model_info, model_data_items in full_join_by_model_info(csv_data):
            values = {}
            for model_data in model_data_items:
                if model_data is None:
                    continue
                values[model_data.get_measurement_date()] = model_data.get_mem_rss() / (1024 * 1024)
            gen_plot_by_date('.', device, model_info, values, 'memory consumption RSS', 'mem_rss', 'Mb')


class PlotPlainManagerTimeByDate(DataProcessor):
    def __init__(self):
        super().__init__(None)

    def run(self, csv_data: List[Dict[ModelInfo, ModelData]]) -> None:
        device = get_device(csv_data)
        for model_info, model_data_items in full_join_by_model_info(csv_data):
            values = {}
            for model_data in model_data_items:
                if model_data is None:
                    continue
                values[model_data.get_measurement_date()] = model_data.get_manager_plain_sequence_median_sum() / 1_000_000_000
            gen_plot_by_date('.', device, model_info, values, 'plain manager time', 'plain_time', 'sec')


class PlotCompileTimeByDate(DataProcessor):
    def __init__(self):
        super().__init__(None)

    def run(self, csv_data: List[Dict[ModelInfo, ModelData]]) -> None:
        device = get_device(csv_data)
        for model_info, model_data_items in full_join_by_model_info(csv_data):
            values = {}
            for model_data in model_data_items:
                if model_data is None:
                    continue
                values[model_data.get_measurement_date()] = model_data.get_compile_time() / 1_000_000_000
            gen_plot_by_date('.', device, model_info, values, 'compile time', 'compile_time', 'sec')


class PlotVariationByIteration(DataProcessor):
    def __init__(self):
        super().__init__(None)

    def run(self, csv_data: List[Dict[ModelInfo, ModelData]]) -> None:
        device = get_device(csv_data)
        model_data_items = [model_data for csv_data_d in csv_data for model_data in csv_data_d.values()]
        for i in range(1, 10):
            print(f'iteration {i}')
            compile_times = [model_data.get_compile_time() / 1_000_000_000 for model_data in model_data_items
                             if model_data is not None and len(model_data.get_compile_durations()) >= i]
            assert all(compile_time is not None and not np.isnan(compile_time) and compile_time >= 0 for compile_time in compile_times)

            stddev_values = [100.0 * model_data.get_compile_time_stddev_n_iterations(i) / model_data.get_compile_time() for model_data in model_data_items
                             if model_data is not None and len(model_data.get_compile_durations()) >= i]

            assert all(stddev_value is not None and not np.isnan(stddev_value) and stddev_value >= 0 for stddev_value in stddev_values)
            hist = Hist(f"compile time std deviation for {i} iterations", "std deviation / compile time, %", "number of values")
            hist.set_values(stddev_values)
            hist.plot(f'stddev_{i}_iter.png')

            scatter = ScatterPlot(f"scatter std deviation for {i} iterations", "std deviation / compile time, %", "compile time, sec")
            scatter.set_values(stddev_values, compile_times)
            scatter.plot(f'scatter_stddev_{i}_iter.png')

            mad_values = [100.0 * model_data.get_compile_time_mad_n_iterations(i) / model_data.get_compile_time() for model_data in model_data_items
                          if model_data is not None and len(model_data.get_compile_durations()) >= i]
            assert all(mad_value is not None and not np.isnan(mad_value) and mad_value >= 0 for mad_value in mad_values)
            hist = Hist(f"compile time MAD for {i} iterations", "MAD / compile time, %", "number of values")
            hist.set_values(mad_values)
            hist.plot(f'mad_{i}_iter.png')

            scatter = ScatterPlot(f"scatter MAD for {i} iterations", "MAD / compile time, %", "compile time, sec")
            scatter.set_values(mad_values, compile_times)
            scatter.plot(f'scatter_mad_{i}_iter.png')


class PlotPlainSeqErrorByIteration(DataProcessor):
    def __init__(self):
        super().__init__(None)

    def run(self, csv_data: List[Dict[ModelInfo, ModelData]]) -> None:
        class Occurrence:
            def __init__(self):
                self.n_total = 0
                self.n_out_of_90_percentile = 0
                self.n_out_of_90_percentile_percent = None
                self.n_out_of_99_percentile = 0
                self.n_out_of_99_percentile_percent = None
                self.median10 = []
                self.median10_max = None
                self.median10_median = None
                self.median10_median_90_percentile = None

            def calculate(self):
                self.median10_max = np.max(self.median10)
                self.median10_median = np.median(self.median10)
                self.median10_median_90_percentile = np.percentile(self.median10, 90)
                self.n_out_of_90_percentile_percent = 100.0 * self.n_out_of_90_percentile / self.n_total
                self.n_out_of_99_percentile_percent = 100.0 * self.n_out_of_99_percentile / self.n_total

        device = get_device(csv_data)
        plain_time_items = []
        for csv_data_d in csv_data:
            for model_info, model_data in csv_data_d.items():
                data_list = model_data.get_manager_plain_sequence_sum_by_iteration()
                plain_time_items.append((model_info, data_list))
        median_values = {}
        max_values = {}
        mean_values = {}
        p90_values = {}
        p99_values = {}
        occurrence = {}
        for i in range(3, 11):
            print(f'iteration {i}')
            deltas = []
            for model_info, plain_time_list in plain_time_items:
                if plain_time_list is None:
                    continue
                if len(plain_time_list) < i:
                    continue
                median_i = np.median(plain_time_list[:i])
                median_10 = np.median(plain_time_list)
                assert not np.isnan(median_10)
                assert median_10 != 0.0
                deltas.append(100.0 * abs(median_10 - median_i) / median_10)
            median_values[i] = np.median(deltas)
            max_values[i] = np.max(deltas)
            mean_values[i] = np.mean(deltas)
            p90_values[i] = np.percentile(deltas, 90)
            p99_values[i] = np.percentile(deltas, 99)

            for model_info, plain_time_list in plain_time_items:
                if model_info not in occurrence:
                    occurrence[model_info] = Occurrence()
                occurrence[model_info].n_total += 1
                median_i = np.median(plain_time_list[:i])
                median_10 = np.median(plain_time_list)
                assert not np.isnan(median_10)
                assert median_10 != 0.0
                delta = 100.0 * abs(median_10 - median_i) / median_10
                if delta > p90_values[i]:
                    occurrence[model_info].n_out_of_90_percentile += 1
                if delta > p99_values[i]:
                    occurrence[model_info].n_out_of_99_percentile += 1
                occurrence[model_info].median10.append(median_10)

        gen_plot_key_value_float('.',
                                 {'median': median_values, 'maximum': max_values, 'mean': mean_values,
                                  'percentile 90': p90_values,
                                  'percentile 99': p99_values},
                                 'plain manager time error by iteration number',
                                 f'{device}_plain_seq_error_by_iteration', 'number of iterations', '(median[10] - median[:i])/median[10], %')

        for model_info, occ in occurrence.items():
            occ.calculate()
        occurrence_items = [(model_info, occ) for model_info, occ in occurrence.items()]
        sorted_by_90_percentile = sorted(occurrence_items, key=lambda x: x[1].n_out_of_90_percentile_percent, reverse=True)
        table = []
        for model_info, occ in sorted_by_90_percentile:
            if occ.n_out_of_90_percentile == 0:
                continue
            row = {'framework': model_info.framework,
                   'name': model_info.name,
                   'precision': model_info.precision,
                   'config': model_info.config}
            row['total'] = occ.n_total
            row['out of 90 percentile'] = occ.n_out_of_90_percentile
            row['out of 90 percentile, %'] = f'{occ.n_out_of_90_percentile_percent:.2f}%'
            row['median10_max'] = occ.median10_max
            row['median10_median'] = occ.median10_median
            row['median10_median_90_percentile'] = occ.median10_median_90_percentile
            table.append(row)
        header_90_p = ['framework', 'name', 'precision', 'config', 'total', 'out of 90 percentile', 'out of 90 percentile, %',
                       'median10_max', 'median10_median', 'median10_median_90_percentile']
        with CSVOutput(f'{device}_plain_seq_error_by_iteration_out_of_90_percentile.csv', header_90_p, None) as csv_file:
            csv_file.write(table)
        sorted_by_99_percentile = sorted(occurrence_items, key=lambda x: x[1].n_out_of_99_percentile_percent, reverse=True)
        table = []
        for model_info, occ in sorted_by_99_percentile:
            if occ.n_out_of_99_percentile == 0:
                continue
            row = {'framework': model_info.framework,
                   'name': model_info.name,
                   'precision': model_info.precision,
                   'config': model_info.config}
            row['total'] = occ.n_total
            row['out of 99 percentile'] = occ.n_out_of_99_percentile
            row['out of 99 percentile, %'] = f'{occ.n_out_of_99_percentile_percent:.2f}%'
            row['median10_max'] = occ.median10_max
            row['median10_median'] = occ.median10_median
            row['median10_median_90_percentile'] = occ.median10_median_90_percentile
            table.append(row)
        header_99_p = ['framework', 'name', 'precision', 'config', 'total', 'out of 99 percentile', 'out of 99 percentile, %',
                       'median10_max', 'median10_median', 'median10_median_90_percentile']
        with CSVOutput(f'{device}_plain_seq_error_by_iteration_out_of_99_percentile.csv', header_99_p, None) as csv_file:
            csv_file.write(table)


class PlotCompileTimeErrorByIteration(DataProcessor):
    def __init__(self):
        super().__init__(None)

    def run(self, csv_data: List[Dict[ModelInfo, ModelData]]) -> None:
        device = get_device(csv_data)
        model_data_items = [model_data for csv_data_d in csv_data for model_data in csv_data_d.values()]
        plain_times = [model_data.get_compile_time_by_iteration() if model_data is not None else None for model_data in model_data_items]
        median_values = {}
        max_values = {}
        mean_values = {}
        p90_values = {}
        p99_values = {}
        for i in range(3, 11):
            print(f'iteration {i}')
            deltas = []
            for plain_time_list in plain_times:
                if plain_time_list is None:
                    continue
                if len(plain_time_list) < i:
                    continue
                median_i = np.median(plain_time_list[:i])
                median_10 = np.median(plain_time_list)
                assert not np.isnan(median_10)
                assert median_10 != 0.0
                deltas.append(100.0 * abs(median_10 - median_i) / median_10)
            median_values[i] = np.median(deltas)
            max_values[i] = np.max(deltas)
            mean_values[i] = np.mean(deltas)
            p90_values[i] = np.percentile(deltas, 90)
            p99_values[i] = np.percentile(deltas, 99)
        gen_plot_key_value_float('.',
                                 {'median': median_values, 'maximum': max_values, 'mean': mean_values,
                                  'percentile 90': p90_values,
                                  'percentile 99': p99_values},
                                 f'{device} compile time error by iteration number',
                                 f'{device}_compile_time_error_by_iteration', 'number of iterations', '(median[10] - median[:i])/median[10], %')


class PlotCompare2InputsPlainSeqErrorByIteration(DataProcessor):
    def __init__(self):
        super().__init__(None)

    def run(self, csv_data: List[Dict[ModelInfo, ModelData]]) -> None:
        class Occurrence:
            def __init__(self):
                self.medians = []
                self.deltas = {}
                self.n_total = 0
                self.median_median = None
                self.n_out_of_90_percentile = 0
                self.n_out_of_90_percentile_percent = None
                self.n_out_of_99_percentile = 0
                self.n_out_of_99_percentile_percent = None

            def calculate(self):
                self.n_out_of_90_percentile_percent = 100.0 * self.n_out_of_90_percentile / self.n_total
                self.n_out_of_99_percentile_percent = 100.0 * self.n_out_of_99_percentile / self.n_total
                self.median_median = float(np.median(self.medians))

        def get_model_data_median(data: Optional[ModelData], i: int) -> Optional[float]:
            if data is None:
                return None
            seqs = data.get_manager_plain_sequence_sum_by_iteration()
            if len(seqs) < i:
                return None
            median = np.median(seqs[:i])
            assert not np.isnan(median)
            assert median != 0.0
            return float(median)

        device = get_device(csv_data)

        median_values = {}
        max_values = {}
        mean_values = {}
        p90_values = {}
        p99_values = {}
        occurrence = {}
        for i in range(1, 11):
            print(f'iteration {i}')
            deltas = []
            for model_info, model_data_items in full_join_by_model_info(csv_data):
                if model_info not in occurrence:
                    occurrence[model_info] = Occurrence()
                assert len(model_data_items) == 2
                median_i_0 = get_model_data_median(model_data_items[0], i)
                median_i_1 = get_model_data_median(model_data_items[1], i)
                if median_i_0 is None or median_i_1 is None:
                    occurrence[model_info].deltas[i] = None
                    continue
                delta = 100.0 * abs(median_i_0 - median_i_1) / max(median_i_0, median_i_1)
                deltas.append(delta)
                occurrence[model_info].deltas[i] = delta
                occurrence[model_info].medians.append(max(median_i_0, median_i_1))
            median_values[i] = np.median(deltas)
            max_values[i] = np.max(deltas)
            mean_values[i] = np.mean(deltas)
            p90_values[i] = np.percentile(deltas, 90)
            p99_values[i] = np.percentile(deltas, 99)

            for model_info, occ in occurrence.items():
                occ.n_total += 1
                if occ.deltas[i] is not None and occ.deltas[i] > p90_values[i]:
                    occ.n_out_of_90_percentile += 1
                if occ.deltas[i] is not None and occ.deltas[i] > p99_values[i]:
                    occ.n_out_of_99_percentile += 1

        gen_plot_key_value_float('.',
                                 {'median': median_values, 'maximum': max_values, 'mean': mean_values,
                                  'percentile 90': p90_values,
                                  'percentile 99': p99_values},
                                 f'{device} plain manager time error by iteration number',
                                 f'{device}_plain_seq_error_by_iteration_2inputs', 'number of iterations', 'delta median / median, %')

        for model_info, occ in occurrence.items():
            occ.calculate()
        occurrence_items = [(model_info, occ) for model_info, occ in occurrence.items()]
        sorted_by_90_percentile = sorted(occurrence_items, key=lambda x: x[1].n_out_of_90_percentile_percent, reverse=True)
        table = []
        for model_info, occ in sorted_by_90_percentile:
            if occ.n_out_of_90_percentile == 0:
                continue
            row = {'framework': model_info.framework,
                   'name': model_info.name,
                   'precision': model_info.precision,
                   'config': model_info.config}
            row['total'] = occ.n_total
            row['out of 90 percentile'] = occ.n_out_of_90_percentile
            row['out of 90 percentile, %'] = f'{occ.n_out_of_90_percentile_percent:.2f}%'
            row['median plain time'] = occ.median_median
            table.append(row)
        header_90_p = ['framework', 'name', 'precision', 'config', 'total', 'out of 90 percentile', 'out of 90 percentile, %',
                       'median plain time']
        with CSVOutput(f'{device}_plain_seq_error_by_iteration_out_of_90_percentile.csv', header_90_p, None) as csv_file:
            csv_file.write(table)

        sorted_by_99_percentile = sorted(occurrence_items, key=lambda x: x[1].n_out_of_99_percentile_percent, reverse=True)
        table = []
        for model_info, occ in sorted_by_99_percentile:
            if occ.n_out_of_99_percentile == 0:
                continue
            row = {'framework': model_info.framework,
                   'name': model_info.name,
                   'precision': model_info.precision,
                   'config': model_info.config}
            row['total'] = occ.n_total
            row['out of 99 percentile'] = occ.n_out_of_99_percentile
            row['out of 99 percentile, %'] = f'{occ.n_out_of_99_percentile_percent:.2f}%'
            row['median plain time'] = occ.median_median
            table.append(row)
        header_99_p = ['framework', 'name', 'precision', 'config', 'total', 'out of 99 percentile', 'out of 99 percentile, %',
                       'median plain time']
        with CSVOutput(f'{device}_plain_seq_error_by_iteration_out_of_99_percentile.csv', header_99_p, None) as csv_file:
            csv_file.write(table)


class PlotCompare2InputsCompileTimeByIteration(DataProcessor):
    def __init__(self):
        super().__init__(None)

    def run(self, csv_data: List[Dict[ModelInfo, ModelData]]) -> None:
        class Occurrence:
            def __init__(self):
                self.medians = []
                self.deltas = {}
                self.n_total = 0
                self.median_median = None
                self.n_out_of_90_percentile = 0
                self.n_out_of_90_percentile_percent = None
                self.n_out_of_99_percentile = 0
                self.n_out_of_99_percentile_percent = None

            def calculate(self):
                self.n_out_of_90_percentile_percent = 100.0 * self.n_out_of_90_percentile / self.n_total
                self.n_out_of_99_percentile_percent = 100.0 * self.n_out_of_99_percentile / self.n_total
                self.median_median = float(np.median(self.medians))

        def get_model_data_median(data: Optional[ModelData], i: int) -> Optional[float]:
            if data is None:
                return None
            seqs = data.get_compile_time_by_iteration()
            if len(seqs) < i:
                return None
            median = np.median(seqs[:i])
            assert not np.isnan(median)
            assert median != 0.0
            return float(median)

        device = get_device(csv_data)

        median_values = {}
        max_values = {}
        mean_values = {}
        p90_values = {}
        p99_values = {}
        occurrence = {}
        for i in range(1, 11):
            print(f'iteration {i}')
            deltas = []
            for model_info, model_data_items in full_join_by_model_info(csv_data):
                if model_info not in occurrence:
                    occurrence[model_info] = Occurrence()
                assert len(model_data_items) == 2
                median_i_0 = get_model_data_median(model_data_items[0], i)
                median_i_1 = get_model_data_median(model_data_items[1], i)
                if median_i_0 is None or median_i_1 is None:
                    occurrence[model_info].deltas[i] = None
                    continue
                delta = 100.0 * abs(median_i_0 - median_i_1) / max(median_i_0, median_i_1)
                deltas.append(delta)
                occurrence[model_info].deltas[i] = delta
                occurrence[model_info].medians.append(max(median_i_0, median_i_1))
            median_values[i] = np.median(deltas)
            max_values[i] = np.max(deltas)
            mean_values[i] = np.mean(deltas)
            p90_values[i] = np.percentile(deltas, 90)
            p99_values[i] = np.percentile(deltas, 99)

            for model_info, occ in occurrence.items():
                occ.n_total += 1
                if occ.deltas[i] is not None and occ.deltas[i] > p90_values[i]:
                    occ.n_out_of_90_percentile += 1
                if occ.deltas[i] is not None and occ.deltas[i] > p99_values[i]:
                    occ.n_out_of_99_percentile += 1

        gen_plot_key_value_float('.',
                                 {'median': median_values, 'maximum': max_values, 'mean': mean_values,
                                  'percentile 90': p90_values,
                                  'percentile 99': p99_values},
                                 f'{device} compile time error by iteration number',
                                 f'{device}_compile_time_error_by_iteration_2inputs', 'number of iterations', 'delta median / median, %')

        for model_info, occ in occurrence.items():
            occ.calculate()
        occurrence_items = [(model_info, occ) for model_info, occ in occurrence.items()]
        sorted_by_90_percentile = sorted(occurrence_items, key=lambda x: x[1].n_out_of_90_percentile_percent, reverse=True)
        table = []
        for model_info, occ in sorted_by_90_percentile:
            if occ.n_out_of_90_percentile == 0:
                continue
            row = {'framework': model_info.framework,
                   'name': model_info.name,
                   'precision': model_info.precision,
                   'config': model_info.config}
            row['total'] = occ.n_total
            row['out of 90 percentile'] = occ.n_out_of_90_percentile
            row['out of 90 percentile, %'] = f'{occ.n_out_of_90_percentile_percent:.2f}%'
            row['median compile time'] = occ.median_median
            table.append(row)
        header_90_p = ['framework', 'name', 'precision', 'config', 'total', 'out of 90 percentile', 'out of 90 percentile, %',
                       'median compile time']
        with CSVOutput(f'{device}_compile_time_error_by_iteration_out_of_90_percentile.csv', header_90_p, None) as csv_file:
            csv_file.write(table)

        sorted_by_99_percentile = sorted(occurrence_items, key=lambda x: x[1].n_out_of_99_percentile_percent, reverse=True)
        table = []
        for model_info, occ in sorted_by_99_percentile:
            if occ.n_out_of_99_percentile == 0:
                continue
            row = {'framework': model_info.framework,
                   'name': model_info.name,
                   'precision': model_info.precision,
                   'config': model_info.config}
            row['total'] = occ.n_total
            row['out of 99 percentile'] = occ.n_out_of_99_percentile
            row['out of 99 percentile, %'] = f'{occ.n_out_of_99_percentile_percent:.2f}%'
            row['median compile time'] = occ.median_median
            table.append(row)
        header_99_p = ['framework', 'name', 'precision', 'config', 'total', 'out of 99 percentile', 'out of 99 percentile, %',
                       'median compile time']
        with CSVOutput(f'{device}_compile_time_error_by_iteration_out_of_99_percentile.csv', header_99_p, None) as csv_file:
            csv_file.write(table)


class PlotCompare2InputsPlainSeqErrorByTime(DataProcessor):
    def __init__(self, step_sec: float):
        super().__init__(None)
        self.step_sec = step_sec

    def run(self, csv_data: List[Dict[ModelInfo, ModelData]]) -> None:

        def get_model_data_median(data: Optional[ModelData], i: int) -> Optional[float]:
            if data is None:
                return None
            seqs = data.get_manager_plain_sequence_sum_by_iteration()
            if len(seqs) < i:
                return None
            median = np.median(seqs[:i])
            assert not np.isnan(median)
            assert median != 0.0
            return float(median)

        deltas = {}
        for i in range(1, 11):
            print(f'iteration {i}')
            for model_info, model_data_items in full_join_by_model_info(csv_data):
                assert len(model_data_items) == 2
                median_i_0 = get_model_data_median(model_data_items[0], i)
                median_i_1 = get_model_data_median(model_data_items[1], i)
                if median_i_0 is None or median_i_1 is None:
                    continue
                delta = 100.0 * abs(median_i_0 - median_i_1) / max(median_i_0, median_i_1)
                plain_seq_duration0 = model_data_items[0].get_manager_plain_sequence_first_n_iter_duration(i)
                plain_seq_duration1 = model_data_items[1].get_manager_plain_sequence_first_n_iter_duration(i)
                if plain_seq_duration0 is None or plain_seq_duration1 is None:
                    continue
                mean_duration = float(np.mean([plain_seq_duration0, plain_seq_duration1])) / 1_000_000_000
                deltas[mean_duration] = delta

        median_values = {}
        max_values = {}
        mean_values = {}
        p90_values = {}
        p99_values = {}

        max_duration = max(deltas.keys())
        for duration in np.arange(self.step_sec, max_duration, self.step_sec):
            duration = float(duration)
            deltas_for_duration = [delta for mean_duration, delta in deltas.items() if
                                   duration > mean_duration >= duration - self.step_sec]
            assert len(deltas_for_duration) > 0
            median_values[duration] = float(np.median(deltas_for_duration))
            max_values[duration] = float(np.max(deltas_for_duration))
            mean_values[duration] = float(np.mean(deltas_for_duration))
            p90_values[duration] = float(np.percentile(deltas_for_duration, 90))
            p99_values[duration] = float(np.percentile(deltas_for_duration, 99))

        device = get_device(csv_data)
        gen_plot_key_value_float('.',
                                 {'median': median_values, 'maximum': max_values, 'mean': mean_values,
                                  'percentile 90': p90_values,
                                  'percentile 99': p99_values},
                                 f'{device} plain manager time error by time',
                                 f'{device}_plain_seq_error_by_time_2inputs', 'time, s', 'delta median / median, %')


class PlotCompareMultipleInputsPlainSeqErrorByIteration(DataProcessor):
    def __init__(self):
        super().__init__(None)
        self.__only_medians = False

    def run(self, csv_data: List[Dict[ModelInfo, ModelData]]) -> None:
        class Occurrence:
            def __init__(self):
                self.n_total = 0
                self.n_out_of_95_percentile = 0
                self.n_out_of_95_percentile_percent = None
                self.n_in_5_percentile = 0
                self.n_in_5_percentile_percent = None

            def calculate(self):
                self.n_out_of_95_percentile_percent = 100.0 * self.n_out_of_95_percentile / self.n_total
                self.n_in_5_percentile_percent = 100.0 * self.n_in_5_percentile / self.n_total

        def get_model_data_median(data: Optional[ModelData], i: int) -> Optional[float]:
            if data is None:
                return None
            seqs = data.get_manager_plain_sequence_sum_by_iteration()
            if len(seqs) < i:
                return None
            median = np.median(seqs[:i])
            assert not np.isnan(median)
            assert median != 0.0
            return float(median)

        device = get_device(csv_data)
        n_inputs = len(csv_data)
        joined_models = list(full_join_by_model_info(csv_data))

        median_values = {}
        max_values = {}
        p95_values = {}
        p5_values = {}

        model_occurence = {}

        for k in range(1, 11):
            print(f'iteration {k}')
            for i in range(0, n_inputs):
                for j in range(i + 1, n_inputs):
                    model_deltas = {}
                    deltas = []
                    for model_info, model_data_items in joined_models:
                        assert n_inputs == len(model_data_items)

                        median_i = get_model_data_median(model_data_items[i], k)
                        median_j = get_model_data_median(model_data_items[j], k)
                        if median_i is None or median_j is None:
                            continue
                        delta = 100.0 * abs(median_i - median_j) / max(median_i, median_j)
                        deltas.append(delta)
                        model_deltas[model_info] = delta

                    delta_median = np.median(deltas)
                    delta_max = np.max(deltas)
                    delta_p95 = np.percentile(deltas, 95)
                    delta_p5 = np.percentile(deltas, 5)

                    if i not in median_values:
                        median_values[i] = {}
                        max_values[i] = {}
                        p95_values[i] = {}
                        p5_values[i] = {}
                    if j not in median_values[i]:
                        median_values[i][j] = {}
                        max_values[i][j] = {}
                        p95_values[i][j] = {}
                        p5_values[i][j] = {}

                    median_values[i][j][k] = delta_median
                    max_values[i][j][k] = delta_max
                    p95_values[i][j][k] = delta_p95
                    p5_values[i][j][k] = delta_p5

                    for model_info, delta in model_deltas.items():
                        if model_info not in model_occurence:
                            model_occurence[model_info] = Occurrence()
                        model_occurence[model_info].n_total += 1
                        if delta > delta_p95:
                            model_occurence[model_info].n_out_of_95_percentile += 1
                        if delta < delta_p5:
                            model_occurence[model_info].n_in_5_percentile += 1

        plot_data = {}
        for i in range(0, n_inputs):
            for j in range(i + 1, n_inputs):
                plot_data[f'median #{i} - #{j}'] = median_values[i][j]
                if not self.__only_medians:
                    plot_data[f'max #{i} - #{j}'] = max_values[i][j]
                    plot_data[f'p95 #{i} - #{j}'] = p95_values[i][j]
        gen_plot_key_value_float('.',
                                 plot_data,
                                 f'{device} transformations time error',
                                 f'{device}_plain_seq_error_by_iteration_multi_inputs', 'number of iterations', '%')

        for model_info, occ in model_occurence.items():
            occ.calculate()

        in_p5_models = [model_info for model_info, occ in model_occurence.items() if occ.n_in_5_percentile > 0]
        in_p5_models = sorted(in_p5_models, key=lambda x: model_occurence[x].n_in_5_percentile_percent, reverse=True)

        table = []
        for model_info in in_p5_models:
            occ = model_occurence[model_info]
            row = {'framework': model_info.framework,
                   'name': model_info.name,
                   'precision': model_info.precision,
                   'config': model_info.config}
            row['total'] = occ.n_total
            row['in 5 percentile'] = occ.n_in_5_percentile
            row['in 5 percentile, %'] = f'{occ.n_in_5_percentile_percent:.2f}%'
            table.append(row)
        header = ['framework', 'name', 'precision', 'config', 'total', 'in 5 percentile', 'in 5 percentile, %']
        with CSVOutput(f'{device}_compile_time_error_by_iteration_in_5_percentile.csv', header, None) as csv_file:
            csv_file.write(table)

        out_of_p95_models = [model_info for model_info, occ in model_occurence.items() if occ.n_out_of_95_percentile > 0]
        out_of_p95_models = sorted(out_of_p95_models, key=lambda x: model_occurence[x].n_out_of_95_percentile_percent, reverse=True)

        table = []
        for model_info in out_of_p95_models:
            occ = model_occurence[model_info]
            row = {'framework': model_info.framework,
                   'name': model_info.name,
                   'precision': model_info.precision,
                   'config': model_info.config}
            row['total'] = occ.n_total
            row['out of 95 percentile'] = occ.n_out_of_95_percentile
            row['out of 95 percentile, %'] = f'{occ.n_out_of_95_percentile_percent:.2f}%'
            table.append(row)
        header = ['framework', 'name', 'precision', 'config', 'total', 'out of 95 percentile', 'out of 95 percentile, %']
        with CSVOutput(f'{device}_compile_time_error_by_iteration_out_if_95_percentile.csv', header, None) as csv_file:
            csv_file.write(table)


class PlotCompareMultipleInputsCompileTimeErrorByIteration(DataProcessor):
    def __init__(self):
        super().__init__(None)

    def run(self, csv_data: List[Dict[ModelInfo, ModelData]]) -> None:
        def get_model_data_median(data: Optional[ModelData], i: int) -> Optional[float]:
            if data is None:
                return None
            seqs = data.get_compile_time_by_iteration()
            if len(seqs) < i:
                return None
            median = np.median(seqs[:i])
            assert not np.isnan(median)
            assert median != 0.0
            return float(median)

        device = get_device(csv_data)
        n_inputs = len(csv_data)

        median_values = {}
        max_values = {}
        p95_values = {}
        for i in range(1, 11):
            print(f'iteration {i}')
            deltas = {}
            for model_info, model_data_items in full_join_by_model_info(csv_data):
                assert n_inputs == len(model_data_items)
                median_i_0 = get_model_data_median(model_data_items[0], i)
                if median_i_0 is None:
                    continue
                for j in range(1, len(model_data_items)):
                    median_i_j = get_model_data_median(model_data_items[j], i)
                    if median_i_j is None:
                        continue
                    delta = 100.0 * abs(median_i_0 - median_i_j) / max(median_i_0, median_i_j)
                    if j not in deltas:
                        deltas[j] = []
                    deltas[j].append(delta)
            for j in range(1, n_inputs):
                if j not in median_values:
                    median_values[j] = {}
                    max_values[j] = {}
                    p95_values[j] = {}
                median_values[j][i] = np.median(deltas[j])
                max_values[j][i] = np.max(deltas[j])
                p95_values[j][i] = np.percentile(deltas[j], 95)

        plot_data = {}
        for j in range(1, n_inputs):
            plot_data[f'median #{j}'] = median_values[j]
            plot_data[f'max #{j}'] = max_values[j]
            plot_data[f'p95 #{j}'] = p95_values[j]
        gen_plot_key_value_float('.',
                                 plot_data,
                                 f'{device} compile time error',
                                 f'{device}_compile_time_error_by_iteration_3inputs', 'number of iterations', '%')


class PlotCompareMultipleInputsTransformationErrorByIteration(DataProcessor):
    def __init__(self):
        super().__init__(None)
        self.__only_medians = False

    def run(self, csv_data: List[Dict[ModelInfo, ModelData]]) -> None:

        TransformationInfo = namedtuple('TransformationInfo', ['model_info', 'name', 'manager_name'])

        class Occurrence:
            def __init__(self):
                self.n_total = 0
                self.n_out_of_95_percentile = 0
                self.n_out_of_95_percentile_percent = None
                self.n_in_5_percentile = 0
                self.n_in_5_percentile_percent = None

            def calculate(self):
                self.n_out_of_95_percentile_percent = 100.0 * self.n_out_of_95_percentile / self.n_total
                self.n_in_5_percentile_percent = 100.0 * self.n_in_5_percentile / self.n_total

        def get_transformations(data: Optional[ModelData]) -> Dict[Tuple[str, str], Unit]:
            if data is None:
                return {}
            d = {}
            for unit in data.get_units_with_type('transformation'):
                d[(unit.name, unit.manager_name)] = unit
            return d

        def full_join_by_transformation_name(data1: Optional[ModelData], data2: Optional[ModelData])  -> Iterator[Tuple[str, str, List[Optional[Unit]]]]:
            transformations1 = get_transformations(data1)
            transformations2 = get_transformations(data2)
            for (name, manager_name) in set(transformations1.keys()) | set(transformations2.keys()):
                yield name, manager_name, [transformations1.get((name, manager_name)), transformations2.get((name, manager_name))]

        def get_longest_transformations(data: Optional[ModelData], i: int) -> List[Unit]:
            if data is None:
                return []
            units = list(data.get_units_with_type('transformation'))
            units = sorted(units, key=lambda x: x.get_duration_median(), reverse=True)
            return units[:i]

        def get_unit_median(unit: Unit, i: int) -> Optional[float]:
            if unit is None:
                return None
            durations = unit.get_durations()
            if durations is None:
                return None
            if len(durations) < i:
                return None
            median = np.median(durations[:i])
            assert not np.isnan(median)
            assert median != 0.0
            return float(median)

        device = get_device(csv_data)
        n_inputs = len(csv_data)
        joined_models = list(full_join_by_model_info(csv_data))

        median_values = {}
        max_values = {}
        p95_values = {}
        p5_values = {}

        ts_occurence: Dict[TransformationInfo, Occurrence] = {}

        for k in range(1, 11):
            print(f'iteration {k}')
            for i in range(0, n_inputs):
                for j in range(i + 1, n_inputs):
                    ts_deltas: Dict[TransformationInfo, float] = {}
                    deltas = []
                    for model_info, model_data_items in joined_models:
                        assert n_inputs == len(model_data_items)
                        longest_transformations = {(unit.name, unit.manager_name) for unit in get_longest_transformations(model_data_items[i], 10)}
                        for name, manager_name, units in full_join_by_transformation_name(model_data_items[i], model_data_items[j]):
                            if units[0] is None or units[1] is None:
                                continue
                            if (units[0].name, units[0].manager_name) not in longest_transformations:
                                continue
                            median_i = get_unit_median(units[0], k)
                            median_j = get_unit_median(units[1], k)
                            if median_i is None or median_j is None:
                                continue
                            delta = 100.0 * abs(median_i - median_j) / max(median_i, median_j)
                            deltas.append(delta)
                            ts_deltas[TransformationInfo(model_info, name, manager_name)] = delta

                    delta_median = np.median(deltas)
                    delta_max = np.max(deltas)
                    delta_p95 = np.percentile(deltas, 95)
                    delta_p5 = np.percentile(deltas, 5)

                    if i not in median_values:
                        median_values[i] = {}
                        max_values[i] = {}
                        p95_values[i] = {}
                        p5_values[i] = {}
                    if j not in median_values[i]:
                        median_values[i][j] = {}
                        max_values[i][j] = {}
                        p95_values[i][j] = {}
                        p5_values[i][j] = {}

                    median_values[i][j][k] = delta_median
                    max_values[i][j][k] = delta_max
                    p95_values[i][j][k] = delta_p95
                    p5_values[i][j][k] = delta_p5

                    for ts_info, delta in ts_deltas.items():
                        if ts_info not in ts_occurence:
                            ts_occurence[ts_info] = Occurrence()
                        ts_occurence[ts_info].n_total += 1
                        if delta > delta_p95:
                            ts_occurence[ts_info].n_out_of_95_percentile += 1
                        if delta < delta_p5:
                            ts_occurence[ts_info].n_in_5_percentile += 1

        plot_data = {}
        for i in range(0, n_inputs):
            for j in range(i + 1, n_inputs):
                plot_data[f'median #{i} - #{j}'] = median_values[i][j]
                if not self.__only_medians:
                    plot_data[f'max #{i} - #{j}'] = max_values[i][j]
                    plot_data[f'p95 #{i} - #{j}'] = p95_values[i][j]
        gen_plot_key_value_float('.',
                                 plot_data,
                                 f'{device} separate longest transformations time error',
                                 f'{device}_ts_error_by_iteration_multi_inputs', 'number of iterations', '%')

        for model_info, occ in ts_occurence.items():
            occ.calculate()

        in_p5_ts = [ts_info for ts_info, occ in ts_occurence.items() if occ.n_in_5_percentile > 0]
        in_p5_ts = sorted(in_p5_ts, key=lambda x: ts_occurence[x].n_in_5_percentile_percent, reverse=True)

        table = []
        for ts_info in in_p5_ts:
            occ = ts_occurence[ts_info]
            row = {'name': ts_info.name,
                   'manager name': ts_info.manager_name,
                   'framework': ts_info.model_info.framework,
                   'model name': ts_info.model_info.name,
                   'precision': ts_info.model_info.precision,
                   'config': ts_info.model_info.config}
            row['total'] = occ.n_total
            row['in 5 percentile'] = occ.n_in_5_percentile
            row['in 5 percentile, %'] = f'{occ.n_in_5_percentile_percent:.2f}%'
            table.append(row)
        header = ['name', 'manager name', 'framework', 'model name', 'precision', 'config', 'total', 'in 5 percentile', 'in 5 percentile, %']
        with CSVOutput(f'{device}_compile_time_error_by_iteration_in_5_percentile.csv', header, None) as csv_file:
            csv_file.write(table)

        out_of_p95_ts = [ts_info for ts_info, occ in ts_occurence.items() if occ.n_out_of_95_percentile > 0]
        out_of_p95_ts = sorted(out_of_p95_ts, key=lambda x: ts_occurence[x].n_out_of_95_percentile_percent, reverse=True)

        table = []
        for ts_info in out_of_p95_ts:
            occ = ts_occurence[ts_info]
            row = {'name': ts_info.name,
                   'manager name': ts_info.manager_name,
                   'framework': ts_info.model_info.framework,
                   'model name': ts_info.model_info.name,
                   'precision': ts_info.model_info.precision,
                   'config': ts_info.model_info.config}
            row['total'] = occ.n_total
            row['out of 95 percentile'] = occ.n_out_of_95_percentile
            row['out of 95 percentile, %'] = f'{occ.n_out_of_95_percentile_percent:.2f}%'
            table.append(row)
        header = ['name', 'manager name', 'framework', 'model name', 'precision', 'config', 'total', 'out of 95 percentile', 'out of 95 percentile, %']
        with CSVOutput(f'{device}_ts_error_by_iteration_out_if_95_percentile.csv', header, None) as csv_file:
            csv_file.write(table)

'''
class PlotTransformationByIteration(DataProcessor):
    def __init__(self, model_info: ModelInfo, ts_name: str, manager_name: str):
        super().__init__(None)
        self.model_info = model_info
        self.ts_name = ts_name
        self.manager_name = manager_name

    def run(self, csv_data: List[Dict[ModelInfo, ModelData]]) -> None:

        def get_transformations(data: Optional[ModelData]) -> Dict[Tuple[str, str], Unit]:
            if data is None:
                return {}
            d = {}
            for unit in data.get_units_with_type('transformation'):
                d[(unit.name, unit.manager_name)] = unit
            return d

        def get_transformation_time(data: Optional[ModelData], ts_name: str, manager_name: str) -> Optional[List[float]]:
            units = get_transformations(data)
            if (ts_name, manager_name) not in units:
                return None
            return units[(ts_name, manager_name)].get_durations()

        device = get_device(csv_data)
        n_inputs = len(csv_data)
        joined_models = list(full_join_by_model_info(csv_data))

        plot_values = {}

        for k in range(1, 11):
            for i in range(0, n_inputs):
                for model_info, model_data_items in joined_models:
                    if model_info != self.model_info:
                        continue
                    assert n_inputs == len(model_data_items)
                    plot_values[i][k] = get_transformation_time(model_data_items[i], self.ts_name, self.manager_name)[k]

        plot_data = {}
        for i in range(0, n_inputs):
            plot_data[f'#{i}'] = plot_values[i]
        gen_plot_key_value_float('.',
                                 plot_data,
                                 f'{device} transformations time {self.model_info} {self.ts_name} {self.manager_name}',
                                 f'{device}_{self.ts_name}_{self.manager_name}_ts_by_iteration_multi_inputs', 'number of iterations', '%')
'''





class PlotPlainSeqScatterColors(DataProcessor):
    def __init__(self):
        super().__init__(None)

    def run(self, csv_data: List[Dict[ModelInfo, ModelData]]) -> None:
        def get_ratio(data: Dict[ModelInfo, Dict[int, float]]) -> Dict[ModelInfo, Dict[int, float]]:
            for model_info in data:
                times = [x for x in data[model_info].values()]
                median = np.median(times)
                for i in data[model_info].keys():
                    data[model_info][i] = (data[model_info][i]/median - 1.0) * 100.0
            return data

        def get_common_models(csv_data: List[Dict[ModelInfo, ModelData]]) -> List[ModelInfo]:
            model_infos = [set(d.keys()) for d in csv_data]
            common_models = model_infos[0]
            for model_info in model_infos:
                common_models &= model_info
            return list(common_models)

        def get_plain_time(models, csv_data):
            plain_times = {}
            for i in range(len(csv_data)):
                csv_data_d = csv_data[i]
                for model_info in models:
                    model_data = csv_data_d[model_info]
                    if model_info not in plain_times:
                        plain_times[model_info] = {}
                    plain_times[model_info][i] = model_data.get_manager_plain_sequence_median_sum()
            return plain_times

        common_models = get_common_models(csv_data)
        plain_times = get_plain_time(common_models, csv_data)
        ratios = get_ratio(plain_times)

        N = len(csv_data)
        M = len(common_models)
        x = np.arange(N + 1)
        y = np.arange(M + 1)

        X, Y = np.meshgrid(x, y)
        values = np.full((len(common_models), len(csv_data)), np.nan)
        for i, model_info in enumerate(common_models):
            for index, value in ratios[model_info].items():
                values[i, index] = value

        device = get_device(csv_data)
        gen_plot_scatter_colors('.', X, Y, values, f'{device} sum of transformations',
                                f'{device}_sum_ts',
                                'nightly job', 'model', '(one nightly/all nightly median - 1), %')


class PlotCompileTimeScatterColors(DataProcessor):
    def __init__(self):
        super().__init__(None)

    def run(self, csv_data: List[Dict[ModelInfo, ModelData]]) -> None:
        def get_ratio(data: Dict[ModelInfo, Dict[int, float]]) -> Dict[ModelInfo, Dict[int, float]]:
            for model_info in data:
                times = [x for x in data[model_info].values()]
                median = np.median(times)
                for i in data[model_info].keys():
                    data[model_info][i] = (data[model_info][i]/median - 1.0) * 100.0
            return data

        def get_common_models(csv_data: List[Dict[ModelInfo, ModelData]]) -> List[ModelInfo]:
            model_infos = [set(d.keys()) for d in csv_data]
            common_models = model_infos[0]
            for model_info in model_infos:
                common_models &= model_info
            return list(common_models)

        def get_compile_time(models, csv_data):
            plain_times = {}
            for i in range(len(csv_data)):
                csv_data_d = csv_data[i]
                for model_info in models:
                    model_data = csv_data_d[model_info]
                    if model_info not in plain_times:
                        plain_times[model_info] = {}
                    plain_times[model_info][i] = model_data.get_compile_time()
            return plain_times

        common_models = get_common_models(csv_data)
        plain_times = get_compile_time(common_models, csv_data)
        ratios = get_ratio(plain_times)

        N = len(csv_data)
        M = len(common_models)
        x = np.arange(N + 1)
        y = np.arange(M + 1)

        X, Y = np.meshgrid(x, y)
        values = np.full((len(common_models), len(csv_data)), np.nan)
        for i, model_info in enumerate(common_models):
            for index, value in ratios[model_info].items():
                values[i, index] = value

        device = get_device(csv_data)
        gen_plot_scatter_colors('.', X, Y, values, f'{device} compile time',
                                f'{device}_compile_time',
                                'nightly job', 'model', '(one nightly/all nightly median - 1), %')


class FindDegradation(DataProcessor, ABC):
    def __init__(self):
        super().__init__(None)

    @abstractmethod
    def get_measurements(self, data: ModelData) -> List[float]:
        pass

    @abstractmethod
    def get_output_file_name(self, device: str) -> str:
        pass

    def get_model_data_median(self, data: Optional[ModelData], i: int) -> Optional[float]:
        if data is None:
            return None
        seqs = self.get_measurements(data)
        if len(seqs) < i:
            return None
        median = np.median(seqs[:i])
        assert not np.isnan(median)
        assert median != 0.0
        return float(median)

    def generate_thresholds(self) -> List[float]:
        thresholds = []
        threshold = 5.0
        while threshold < 100.0:
            thresholds.append(threshold)
            threshold += 0.5
        return thresholds

    def generate_iterations(self) -> List[float]:
        return list(range(1, 11, 2))

    def run(self, csv_data: List[Dict[ModelInfo, ModelData]]) -> None:
        device = get_device(csv_data)

        deltas_by_iteration = {} # key - num of iterations, value - list of deltas

        iterations = self.generate_iterations()

        for model_info, model_data_items in full_join_by_model_info(csv_data):
            assert len(model_data_items) == len(csv_data)
            for csv_idx in range(1, len(model_data_items)):
                for n_iter in iterations:
                    median_i_0 = self.get_model_data_median(model_data_items[csv_idx - 1], n_iter)
                    median_i_1 = self.get_model_data_median(model_data_items[csv_idx], n_iter)
                    if median_i_0 is None or median_i_1 is None:
                        continue
                    avg = (median_i_1 + median_i_0) / 2.0
                    delta = 100.0 * (median_i_1 - median_i_0) / avg
                    if n_iter not in deltas_by_iteration:
                        deltas_by_iteration[n_iter] = []
                    deltas_by_iteration[n_iter].append(delta)

        thresholds = self.generate_thresholds()

        count_by_iteration = {} # key - num of iterations, value : Dict[threshold, count]
        for n_iter, deltas in deltas_by_iteration.items():
            if n_iter not in count_by_iteration:
                count_by_iteration[n_iter] = {}
            for threshold in thresholds:
                count = sum(1 for x in deltas if x > threshold)
                count_by_iteration[n_iter][threshold] = count

        header = ['threshold, %']
        for n_iter in iterations:
            header.append(f'first {n_iter} iterations')
        table = []
        for threshold in thresholds:
            row = {'threshold, %': threshold}
            for n_iter, counts in count_by_iteration.items():
                row[f'first {n_iter} iterations'] = counts[threshold]
            table.append(row)
        with CSVOutput(self.get_output_file_name(device), header, None) as csv_file:
            csv_file.write(table)


class FindCompileTimeDegradation(FindDegradation):
    def __init__(self):
        super().__init__()

    def get_measurements(self, data: ModelData) -> List[float]:
        return data.get_compile_time_by_iteration()

    def get_output_file_name(self, device: str) -> str:
        return f'{device}_compile_time_error_table_by_iteration.csv'


class FindTransformationSumDegradation(FindDegradation):
    def __init__(self):
        super().__init__()

    def get_measurements(self, data: ModelData) -> List[float]:
        return data.get_manager_plain_sequence_sum_by_iteration()

    def get_output_file_name(self, device: str) -> str:
        return f'{device}_ts_sum_error_table_by_iteration.csv'


class FindMemoryDegradation(FindDegradation):
    def __init__(self):
        super().__init__()

    def get_measurements(self, data: ModelData) -> List[float]:
        return []

    def get_output_file_name(self, device: str) -> str:
        return f'{device}_memory_error_table_by_iteration.csv'

    def get_model_data_median(self, data: Optional[ModelData], i: int) -> Optional[float]:
        if data is None:
            return None
        return data.get_mem_rss()

    def generate_thresholds(self) -> List[float]:
        thresholds = []
        threshold = 0.2
        while threshold < 20.0:
            thresholds.append(threshold)
            threshold += 0.2
        return thresholds

    def generate_iterations(self) -> List[float]:
        return [1]


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

    #data_processors.append(FindCompileTimeDegradation())
    #data_processors.append(FindTransformationSumDegradation())
    #data_processors.append(FindMemoryDegradation())

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
