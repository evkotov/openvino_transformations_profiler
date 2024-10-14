import sys
from typing import List, Iterator

from compare_csv import ModelInfo, get_csv_data, get_device
from stat_utils import get_iter_time_values, compile_time_by_iterations, get_stddev_unit_durations_all_csv
from stat_utils import get_sum_units_durations_by_iteration
from plot_utils import Plot, Hist

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


if __name__ == '__main__':
    gen_compile_time_by_iterations_from_input(sys.argv[1:])
    #gen_stddev_units_from_input(sys.argv[1:], 'transformation', 1.0)
    #gen_transformations_sum_time_by_iterations_from_input(sys.argv[1:], 'transformation')
