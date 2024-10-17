import csv
import sys
from typing import List, Iterator, Dict

from compare_csv import ModelInfo, get_csv_data, get_device, sort_table
from stat_utils import get_iter_time_values, compile_time_by_iterations, get_stddev_unit_durations_all_csv
from stat_utils import get_sum_units_durations_by_iteration, get_model_unit_sum_by_iterations_all_csv
from stat_utils import get_model_units_deviations_by_iter_all_csv
from plot_utils import Plot, Hist, ScatterPlot

import numpy as np


def gen_compile_time_by_iterations(device: str, model_info: ModelInfo, model_data_items: Iterator[List[float]],
                                   what: str, file_prefix: str):
    title = f'{what} {device} {model_info.framework} {model_info.name} {model_info.precision}'
    if model_info.optional_attribute:
        title += f' {model_info.optional_attribute}'
    x_label = 'Job run time (seconds)'
    y_label = f'{what} (seconds)'
    plot = Plot(title, x_label, y_label)
    all_compile_time_values = []
    for durations in model_data_items:
        compile_time_values = [float(duration) / 1_000_000_000 for duration in durations]
        iterations = get_iter_time_values(compile_time_values)
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
    plot.append_x_line(max_deviation_abs, f'{"%.2f" % abs(max_deviation)}% max deviation from the median',
                           'blue', ':')

    if max_deviation > 1.0:
        # Calculate 10% deviation from the median
        deviation = 0.01 * median_value
        lower_bound = median_value - deviation
        upper_bound = median_value + deviation
        plot.set_stripe(lower_bound, upper_bound, label='1% deviation from the median')

    path = f'{file_prefix}_{device}_{model_info.framework}_{model_info.name}_{model_info.precision}'
    if model_info.optional_attribute:
        path += f'_{model_info.optional_attribute}'
    path += '.png'
    plot.plot(path)


def gen_hist_values(device: str, model_info: ModelInfo, values: List[float]):
    title = f'Ratio stddev/median {device} {model_info.framework} {model_info.name} {model_info.precision}'
    if model_info.optional_attribute:
        title += f' {model_info.optional_attribute}'
    x_label = 'Ratio stddev/median'
    y_label = 'Number of values'
    hist = Hist(title, x_label, y_label)
    hist.set_values(values)
    hist.set_bins(160)
    path = f'hist_stddev_{device}_{model_info.framework}_{model_info.name}_{model_info.precision}'
    if model_info.optional_attribute:
        path += f'_{model_info.optional_attribute}'
    path += '.png'
    hist.plot(path)


def gen_compile_time_by_iterations_from_input(inputs: List[str]):
    csv_data = get_csv_data(inputs)
    if not csv_data:
        return
    device = get_device(csv_data)
    for model_info, durations in compile_time_by_iterations(csv_data):
        gen_compile_time_by_iterations(device, model_info, durations, 'Compile time', 'compile_time')


def gen_transformations_sum_time_by_iterations_from_input(inputs: List[str], unit_type: str):
    csv_data = get_csv_data(inputs)
    if not csv_data:
        return
    device = get_device(csv_data)
    for model_info, durations in get_sum_units_durations_by_iteration(csv_data, unit_type):
        gen_compile_time_by_iterations(device, model_info, durations, 'Sum of transformations', 'sum_ts')


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


if __name__ == '__main__':
    #gen_compile_time_by_iterations_from_input(sys.argv[1:])
    #gen_stddev_units_from_input(sys.argv[1:], 'transformation', 1.0)
    #gen_transformations_sum_time_by_iterations_from_input(sys.argv[1:], 'transformation')
    #gen_csv_transformations_sum_time_by_iterations_from_input(sys.argv[1:], 'transformation')
    gen_deviation_units_from_input(sys.argv[1:], 'transformation', 1.0)
