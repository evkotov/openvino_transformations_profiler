import csv
import os
import sys
from typing import List, Iterator, Dict, Optional, Tuple
import numpy as np

from compare_csv import get_device
from table import sort_table
from parse_input import get_csv_data
from stat_utils import (
    get_iter_time_values, compile_time_by_iterations,
    get_stddev_unit_durations_all_csv, get_sum_units_durations_by_iteration,
    get_model_unit_sum_by_iterations_all_csv, get_model_units_deviations_by_iter_all_csv
)
from common_structs import Unit, ModelInfo
from plot_utils import Plot, Hist, ScatterPlot, generate_x_ticks_cast_to_int


def gen_compile_time_by_iterations_multiple_median(device: str, model_info: ModelInfo, model_data_items: Iterator[List[float]],
                                   what: str, file_prefix: str):
    title = f'{what} {device} {model_info.framework} {model_info.name} {model_info.precision}'
    if model_info.config:
        title += f' {model_info.config}'
    x_label = 'Job run time (seconds)'
    y_label = f'{what} (seconds)'
    plot = Plot(title, x_label, y_label)
    for durations in model_data_items:
        compile_time_values = [float(duration) / 1_000_000_000 for duration in durations]
        iterations = get_iter_time_values(compile_time_values)
        assert len(compile_time_values) == len(iterations)

        median_value = float(np.median(compile_time_values))
        median_label = f'Median: {"%.3f" % median_value} seconds'
        plot.add_with_xline(iterations, compile_time_values, median_value, ':', median_label)

    path = f'{file_prefix}_{device}_{model_info.framework}_{model_info.name}_{model_info.precision}'
    if model_info.config:
        path += f'_{model_info.config}'
    path += '.png'
    plot.plot(path)


def gen_compile_time_by_iterations_one_common_median(output_dir: str,
                                                     device: str, model_info: ModelInfo, model_data_items: Iterator[List[float]],
                                                     what: str, file_prefix: str):
    title = f'{what} {device} {model_info.framework} {model_info.name} {model_info.precision}'
    if model_info.config:
        title += f' {model_info.config}'
    x_label = 'iteration number'
    y_label = f'{what} (seconds)'

    plot = Plot(title, x_label, y_label)
    plot.set_x_ticks_func(generate_x_ticks_cast_to_int)

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


def gen_hist_values(device: str, model_info: ModelInfo, values: List[float]):
    title = f'Ratio stddev/median {device} {model_info.framework} {model_info.name} {model_info.precision}'
    if model_info.config:
        title += f' {model_info.config}'
    x_label = 'Ratio stddev/median'
    y_label = 'Number of values'
    hist = Hist(title, x_label, y_label)
    hist.set_values(values)
    hist.set_bins(160)
    path = f'hist_stddev_{device}_{model_info.framework}_{model_info.name}_{model_info.precision}'
    if model_info.config:
        path += f'_{model_info.config}'
    path += '.png'
    hist.plot(path)


def gen_compile_time_by_iterations_from_input(output_dir: str, inputs: List[str], model_name: Optional[str]):
    csv_data = get_csv_data(inputs)
    if not csv_data:
        return
    device = get_device(csv_data)
    for model_info, durations in compile_time_by_iterations(csv_data):
        if model_name and model_info.name != model_name:
            continue
        gen_compile_time_by_iterations_one_common_median(output_dir, device, model_info, durations, 'Compile time', 'compile_time')


def gen_transformations_sum_time_by_iterations_from_input(output_dir: str, inputs: List[str], unit_type: str, model_name: Optional[str]):
    csv_data = get_csv_data(inputs)
    if not csv_data:
        return
    device = get_device(csv_data)
    for model_info, durations in get_sum_units_durations_by_iteration(csv_data, unit_type):
        if model_name and model_info.name != model_name:
            continue
        gen_compile_time_by_iterations_one_common_median(output_dir, device, model_info, durations, 'Sum of transformations', 'sum_ts')


def gen_stddev_units_from_input(inputs: List[str], unit_type: str, min_median: float):
    csv_data = get_csv_data(inputs)
    if not csv_data:
        return
    device = get_device(csv_data)
    for model_info, values in get_stddev_unit_durations_all_csv(csv_data, unit_type, min_median):
        gen_hist_values(device, model_info, values)


def gen_csv_sum_iterations(device: str, model_info, unit_sums: Dict[str, List[float]], prefix: str):
    def create_header(n_iterations: int):
        column_names = ['name', 'median']
        for i in range(n_iterations):
            column_names.append(f'iteration #{i + 1} (ms)')
        for i in range(n_iterations):
            column_names.append(f'iteration #{i + 1} - median (ms)')
        return column_names

    def get_delta_header_names(n_iterations: int) -> List[str]:
        column_names = []
        for n_iteration in range(n_iterations):
            column_names.append(f'iteration #{n_iteration + 1} - median (ms)')
        return column_names

    def get_n_iterations(unit_sums: Dict[str, List[float]]) -> int:
        first_key, first_value = next(iter(unit_sums.items()))
        return len(first_value)

    n_iterations = get_n_iterations(unit_sums)

    table = []
    for name in unit_sums.keys():
        durations = [e / 1_000_000 for e in unit_sums[name]]
        median = np.median(durations)
        row = {'name' : name, 'median' : float(median)}
        for n_iteration in range(n_iterations):
            row[f'iteration #{n_iteration + 1} (ms)'] = float(durations[n_iteration])
            delta = durations[n_iteration] - median
            row[f'iteration #{n_iteration + 1} - median (ms)'] = float(delta)
        table.append(row)

    header = create_header(n_iterations)
    delta_header_names = get_delta_header_names(n_iterations)

    def get_max_delta(row: Dict) -> float:
        return max((abs(row[key]) for key in delta_header_names))
    sorted_table = sort_table(table, get_max_delta)

    path = f'{prefix}_{device}_{model_info.framework}_{model_info.name}_{model_info.precision}'
    if model_info.optional_attribute:
        path += f'_{model_info.optional_attribute}'
    path += '.csv'

    with open(path, mode="w", newline="") as f_out:
        csv_writer = csv.DictWriter(f_out, fieldnames=header, delimiter=';')
        csv_writer.writeheader()
        for row in sorted_table:
            csv_writer.writerow(row)

def gen_csv_transformations_sum_time_by_iterations_from_input(inputs: List[str], unit_type: str):
    csv_data = get_csv_data(inputs)
    if not csv_data:
        return
    device = get_device(csv_data)
    for model_info, unit_sums in get_model_unit_sum_by_iterations_all_csv(csv_data, unit_type):
        i = 0
        for unit_sums_dict in unit_sums:
            gen_csv_sum_iterations(device, model_info, unit_sums_dict, f'sum_ts_iterations_{i}')
            i += 1


def gen_deviation_scatter_units_values(device: str,
                                       model_info: ModelInfo,
                                       n_iter: int,
                                       x_values: List[float],
                                       y_values: List[float]):
    title = f'Deviation scatter iteration #{n_iter} {device} {model_info.framework} {model_info.name} {model_info.precision}'
    if model_info.optional_attribute:
        title += f' {model_info.optional_attribute}'
    x_label = 'Median (ms)'
    y_label = f'Duration on {n_iter} iteration  - Median (ms)'
    plot = ScatterPlot(title, x_label, y_label)
    plot.set_values(x_values, y_values)
    path = f'scatter_deviation_{device}_{model_info.framework}_{model_info.name}_{model_info.precision}_{n_iter}'
    if model_info.optional_attribute:
        path += f'_{model_info.optional_attribute}'
    path += '.png'
    plot.plot(path)


def gen_deviation_units_from_input(inputs: List[str], unit_type: str, min_median: float):
    csv_data = get_csv_data(inputs)
    if not csv_data:
        return
    device = get_device(csv_data)
    min_median_ms = min_median * 1_000
    for model_info, deviations in get_model_units_deviations_by_iter_all_csv(csv_data, unit_type):
        for n_iter in range(len(deviations)):
            iter_deviations = [e for e in deviations[n_iter] if e.median >= min_median_ms]
            x_values = [e.median / 1_000_000 for e in iter_deviations]
            y_values = [e.deviation / 1_000_000 for e in iter_deviations]
            gen_deviation_scatter_units_values(device, model_info, n_iter, x_values, y_values)


def get_all_csv(dir_path: str) -> List[str]:
    csv_files: List[str] = []
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            if file.endswith('.csv'):
                csv_files.append(os.path.join(root, file))
    return csv_files


def get_input_csv_files(inputs: List[str]) -> List[str]:
    csv_files: List[str] = []
    for input_path in inputs:
        if os.path.isdir(input_path):
            csv_files.extend(get_all_csv(input_path))
        else:
            csv_files.append(input_path)
    return sorted(csv_files)


def find_csv_files_in_subdirs(input_dir: str) -> Iterator[Tuple[str, List[str]]]:
    for root, dirs, files in os.walk(input_dir):
        csv_files = [os.path.join(root, file) for file in files if file.endswith('.csv')]
        if csv_files:
            yield root, csv_files
    return


def process_and_replicate_structure(input_dir: str, output_dir: str, function):
    for input_dir_path, csv_file_paths in find_csv_files_in_subdirs(input_dir):
        output_dir_path = os.path.join(output_dir, os.path.relpath(input_dir_path, input_dir))
        if not os.path.exists(output_dir_path):
            os.makedirs(output_dir_path)
        function(csv_file_paths, output_dir_path)


if __name__ == '__main__':
    Unit.USE_ONLY_0_ITER_GPU = False
    '''
    def gen_compile_time(input_files: List[str], output_dir: str):
        gen_compile_time_by_iterations_from_input(output_dir, input_files, None)
    process_and_replicate_structure(sys.argv[1], sys.argv[2], gen_compile_time)
    '''
    inputs = get_input_csv_files(sys.argv[1:])
    gen_compile_time_by_iterations_from_input('.', inputs, None)
    #gen_stddev_units_from_input(inputs, 'transformation', 1.0)
    gen_transformations_sum_time_by_iterations_from_input('.', inputs,'transformation', None)
    #gen_csv_transformations_sum_time_by_iterations_from_input(inputs, 'transformation')
    #gen_deviation_units_from_input(inputs, 'transformation', 1.0)
