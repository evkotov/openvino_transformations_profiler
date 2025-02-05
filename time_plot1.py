import copy
from typing import List, Dict, Optional, Iterator, Tuple
import os

from ov_ts_profiler.common_structs import ModelInfo, ModelData, full_join_by_model_info
from ov_ts_profiler.stat_utils import get_device
from ov_ts_profiler.parse_input import get_csv_data, get_input_csv_files
from ov_ts_profiler.plot_utils import Plot, generate_x_ticks_cast_to_int, save_subplots
import csv
from collections import namedtuple
import numpy as np

BenchCSVColumnNames = ('Time', "compilation_time", "topology", "framework", "precision", "device", "system_hardware")
BenchCSVItem = namedtuple('BenchCSVItem', BenchCSVColumnNames)


def parse_benchmark_input(path: str) -> Iterator[BenchCSVItem]:
    with open(path, 'r', encoding='utf-8-sig') as f:
        csv_reader = csv.DictReader(f, delimiter=',')
        for row in csv_reader:
            yield BenchCSVItem(
                row['Time'],
                float(row['compilation_time']),
                row['topology'],
                row['framework'],
                row['precision'],
                row['device'],
                row['system_hardware']
            )


def join_benchmark_input(items: List[BenchCSVItem]) -> List[Tuple[ModelInfo, List[Optional[float]]]]:
    model_data = {}
    for item in items:
        model_info = ModelInfo(item.framework, item.topology, item.precision, '')
        if model_info not in model_data:
            model_data[model_info] = []
        model_data[model_info].append(item.compilation_time)
    return [(model_info, model_data[model_info]) for model_info in model_data]


def join_benchmark_input_csv_items(items: List[BenchCSVItem]) -> List[Tuple[ModelInfo, List[Optional[BenchCSVItem]]]]:
    model_data = {}
    for item in items:
        model_info = ModelInfo(item.framework, item.topology, item.precision, '')
        if model_info not in model_data:
            model_data[model_info] = []
        model_data[model_info].append(item)
    return [(model_info, model_data[model_info]) for model_info in model_data]


'''
class FilterChain(ABC):
    def __init__(self):
        self.prev_filter = None

    def set_prev_filter(self, prev_filter):
        self.prev_filter = prev_filter

    def __call__(self, item: BenchCSVItem) -> bool:
        if self.prev_filter:
            return self.prev_filter.filter(item)
        return True

    @abstractmethod
    def filter(self, item: BenchCSVItem) -> bool:
        pass


class FilterByDevice(FilterChain):
    def __init__(self, device: str):
        super().__init__()
        self.device = device

    def filter(self, item: BenchCSVItem) -> bool:
        return item.device == self.device


class FilterByModelInfo(FilterChain):
    def __init__(self, model_info: ModelInfo):
        super().__init__()
        self.model_info = model_info

    def filter(self, item: BenchCSVItem) -> bool:
        return item.framework == self.model_info.framework and \
               item.topology == self.model_info.name and \
               item.precision == self.model_info.precision
'''

class BenchmarkData:
    def __init__(self):
        self.csv_items = []

    def add_csv_item(self, item: BenchCSVItem):
        self.csv_items.append(item)

    def get_items_by_filter(self, filter_func) -> Iterator[BenchCSVItem]:
        for item in self.csv_items:
            if filter_func(item):
                yield item


def gen_plot(output_dir: str,
             device: str, model_info: ModelInfo, model_data_list: List[List[Optional[float]]],
             what: str, file_prefix: str):
    title = f'{what} {device}\n{model_info.framework} {model_info.name} {model_info.precision}'
    if model_info.config:
        title += f' {model_info.config}'
    x_label = 'iteration number'
    y_label = f'{what} (seconds)'

    plot = Plot(title, x_label, y_label)
    plot.set_x_ticks_func(generate_x_ticks_cast_to_int)

    for durations in model_data_list:
        y_values = []
        x_values = []
        for i, duration in enumerate(durations):
            if duration is None:
                continue
            y_values.append(duration)
            x_values.append(i + 1)
        assert len(y_values) == len(x_values)
        plot.add(x_values, y_values)

    if len(model_data_list) == 1:
        all_compile_time_values = [item for item in model_data_list[0] if item is not None]
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

    path = os.path.join(output_dir, f'{device}_{model_info.framework}_{model_info.name}_{model_info.precision}')
    if model_info.config:
        path += f'_{model_info.config}'
    path += f'_{file_prefix}'
    path += '.png'
    plot.plot(path)


def gen_plot_with_subplots(device: str,
                           model_info: ModelInfo,
                           model_data_list: List[List[Optional[float]]],
                           what: str):
    title = ''
    x_label = 'iteration number'
    y_label = f'seconds'

    plot = Plot(title, x_label, y_label)
    plot.set_x_ticks_func(generate_x_ticks_cast_to_int)

    all_compile_time_values = []
    for durations in model_data_list:
        y_values = []
        x_values = []
        for i, duration in enumerate(durations):
            if duration is None:
                continue
            y_values.append(duration)
            x_values.append(i + 1)
        assert len(y_values) == len(x_values)
        assert all(value is not None for value in y_values)
        assert all(value is not None for value in x_values)
        plot.add(x_values, y_values)
        all_compile_time_values.extend(y_values)

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

    title = f'{what} {device}\n{model_info.framework} {model_info.name} {model_info.precision}'
    if model_info.config:
        title += f' {model_info.config}'
    title += f' max deviation {max_deviation :.2f}%'
    plot.set_title(title)

    return plot


class PlotCompareCompileTimeWithBenchmarking:
    def get_compile_time_data(self, data: List[Dict[ModelInfo, ModelData]]) -> Iterator[Tuple[ModelInfo, List[Optional[float]]]]:
        for model_info, model_data_items in full_join_by_model_info(data):
            compile_times = [
                (model_data.get_compile_durations()[0] / 1_000_000_000 if model_data is not None and len(model_data.get_compile_durations()) != 0 else None)
                for model_data in model_data_items
            ]
            yield model_info, compile_times

    def run_ts_stats(self, csv_data: List[Dict[ModelInfo, ModelData]]) -> None:
        print('run_ts_stats ...')
        device = get_device(csv_data)
        # CSV files can store different models info
        for model_info, model_data_iter in self.get_compile_time_data(csv_data):
            model_data_list = list(model_data_iter)
            gen_plot('.', device, model_info, [model_data_list], 'ts stats compile time', 'compilation')

    def run_benchmark_items(self, items: List[BenchCSVItem], device: str, hardware: str) -> None:
        print('run_benchmark_items ...')
        filtered_items = (item for item in items if item.device == device and item.system_hardware == hardware)
        for model_info, model_data_list in join_benchmark_input(filtered_items):
            gen_plot('.', device, model_info, [model_data_list], 'compilation', 'compilation')


ts_stats_inputs = [
    '/home/ekotov/WORK/DEBUG/TRANSFORMATIONS_STATS/INPUT/nightly/28.01.2025/04.1_cpu.zip.dir/archive/CONCATED.csv',
    '/home/ekotov/WORK/DEBUG/TRANSFORMATIONS_STATS/INPUT/nightly/28.01.2025/04.2_cpu.zip.dir/archive/CONCATED.csv',
    '/home/ekotov/WORK/DEBUG/TRANSFORMATIONS_STATS/INPUT/nightly/28.01.2025/04.4_cpu.zip.dir/archive/CONCATED.csv',
    '/home/ekotov/WORK/DEBUG/TRANSFORMATIONS_STATS/INPUT/nightly/28.01.2025/04.5_cpu.zip.dir/archive/CONCATED.csv',
    '/home/ekotov/WORK/DEBUG/TRANSFORMATIONS_STATS/INPUT/nightly/28.01.2025/05.2_cpu.zip.dir/archive/CONCATED.csv',
    '/home/ekotov/WORK/DEBUG/TRANSFORMATIONS_STATS/INPUT/nightly/28.01.2025/05.3_cpu.zip.dir/archive/CONCATED.csv',
    '/home/ekotov/WORK/DEBUG/TRANSFORMATIONS_STATS/INPUT/nightly/28.01.2025/05.4_cpu.zip.dir/archive/CONCATED.csv',
    '/home/ekotov/WORK/DEBUG/TRANSFORMATIONS_STATS/INPUT/nightly/28.01.2025/05.5_cpu.zip.dir/archive/CONCATED.csv',
    '/home/ekotov/WORK/DEBUG/TRANSFORMATIONS_STATS/INPUT/nightly/28.01.2025/06.01_cpu.zip.dir/archive/CONCATED.csv',
]

csv_data = get_csv_data(get_input_csv_files(ts_stats_inputs))
data_processor = PlotCompareCompileTimeWithBenchmarking()
#data_processor.run_ts_stats(csv_data)

#benchmark_input = '/home/ekotov/WORK/DEBUG/TRANSFORMATIONS_STATS/INPUT/nightly/28.01.2025/Compilation time from benchmarks [s]-data-2025-01-27 18_58_21.csv'
benchmark_input = '/home/ekotov/WORK/DEBUG/TRANSFORMATIONS_STATS/INPUT/nightly/28.01.2025/Compilation time from benchmarks [s]-data-2025-02-04 17_08_43.csv'


def print_benchmark_hardware(benchmark_input: List[BenchCSVItem]):
    hardware = set(item.system_hardware for item in benchmark_input)
    hardware = sorted(list(hardware))
    print('\n'.join(hardware))


def get_benchmark_hardware(benchmark_input: List[BenchCSVItem]) -> List[str]:
    return list(set(item.system_hardware for item in benchmark_input))


def print_benchmark_device_hardware(benchmark_input: List[BenchCSVItem]):
    hardware = set((item.device, item.system_hardware) for item in benchmark_input)
    hardware = sorted(list(hardware), key=lambda x: (x[0], x[1]))
    for item in hardware:
        print('{} {}'.format(item[0], item[1]))

benchmark_items = list(parse_benchmark_input(benchmark_input))
#print_benchmark_device_hardware(benchmark_items)

#data_processor.run_benchmark_items(benchmark_items, 'CPU', 'ubuntu22_i9-12900K_17')

def check_config_duplicates(csv_data: List[Dict[ModelInfo, ModelData]]):
    model_info_items = {}
    for d in csv_data:
        for model_info in d.keys():
            new_model_info = copy.deepcopy(model_info)
            new_model_info = new_model_info._replace(config = '')
            if new_model_info not in model_info_items:
                model_info_items[new_model_info] = set()
            model_info_items[new_model_info].add(model_info)
    for model_info, items in model_info_items.items():
        if len(items) > 1:
            print(f'Config {model_info} duplicates:')
            for item in items:
                print(f'    {item}')

check_config_duplicates(csv_data)

def plot_join_ts_and_benchmark(csv_data: List[Dict[ModelInfo, ModelData]], benchmark_items: List[BenchCSVItem], device: str):
    ts_stats_data_first_compile_time = {}
    for model_info, model_data_items in full_join_by_model_info(csv_data):
        compile_times = [
            (model_data.get_compile_durations()[0] / 1_000_000_000 if model_data is not None and len(model_data.get_compile_durations()) != 0 else None)
            for model_data in model_data_items
        ]
        new_model_info = copy.deepcopy(model_info)
        new_model_info = new_model_info._replace(config = '')
        if new_model_info not in ts_stats_data_first_compile_time:
            ts_stats_data_first_compile_time[new_model_info] = []
        ts_stats_data_first_compile_time[new_model_info].append(compile_times)

    ts_stats_data_median_compile_time = {}
    for model_info, model_data_items in full_join_by_model_info(csv_data):
        compile_times = [
            (model_data.get_compile_time() / 1_000_000_000 if model_data is not None and model_data.get_compile_time() is not None else None)
            for model_data in model_data_items
        ]
        new_model_info = copy.deepcopy(model_info)
        new_model_info = new_model_info._replace(config = '')
        if new_model_info not in ts_stats_data_median_compile_time:
            ts_stats_data_median_compile_time[new_model_info] = []
        ts_stats_data_median_compile_time[new_model_info].append(compile_times)

    benchmark_items = [item for item in benchmark_items if item.device == device]
    benchmark_data = {}
    for model_info, model_data_list in join_benchmark_input_csv_items(benchmark_items):
        benchmark_data[model_info] = model_data_list
    get_benchmark_hardware_items = get_benchmark_hardware(benchmark_items)

    for model_info in ts_stats_data_first_compile_time.keys():
        plots = []
        if model_info not in benchmark_data:
            continue
        plot = gen_plot_with_subplots(device, model_info, ts_stats_data_first_compile_time[model_info], 'transformation stats first compile time')
        plots.append(plot)
        plot = gen_plot_with_subplots(device, model_info, ts_stats_data_median_compile_time[model_info], 'transformation stats median compile time')
        plots.append(plot)
        for hardware in get_benchmark_hardware_items:
            values = [item.compilation_time for item in benchmark_data[model_info] if item.system_hardware == hardware]
            if not values:
                continue
            plot = gen_plot_with_subplots(device, model_info, [values], 'benchmark compile time {}'.format(hardware))
            plots.append(plot)
        path = os.path.join('.', f'{device}_{model_info.framework}_{model_info.name}_{model_info.precision}')
        path += '_ts_and_benchmarks.png'
        save_subplots(plots, path)

plot_join_ts_and_benchmark(csv_data, benchmark_items, 'CPU')
