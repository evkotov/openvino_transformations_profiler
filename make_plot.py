import sys
from typing import List, Iterator

from compare_csv import ModelInfo, get_csv_data
from stat_utils import get_iter_time_values, compile_time_by_iterations
from plot_utils import Plot

import numpy as np


def gen_compile_time_by_iterations(model_info: ModelInfo, model_data_items: Iterator[List[float]]):
    title = f'Compile time {model_info.framework} {model_info.name} {model_info.precision}'
    if model_info.optional_attribute:
        title += f' {model_info.optional_attribute}'
    x_label = 'Job run time (seconds)'
    y_label = 'Compile time (seconds)'
    plot = Plot(title, x_label, y_label)
    all_compile_time_values = []
    for durations in model_data_items:
        compile_time_values = [float(duration) / 1_000_000_000 for duration in durations]
        iterations = get_iter_time_values(compile_time_values)
        all_compile_time_values.extend(compile_time_values)

        assert len(compile_time_values) == len(iterations)
        plot.add(iterations, compile_time_values)

    # Calculate the median value of y_values
    median_value = float(np.median(all_compile_time_values))
    plot.append_x_line(median_value, f'Median: {"%.2f" % median_value} secs')

    # Calculate 10% deviation from the median
    deviation = 0.01 * median_value
    lower_bound = median_value - deviation
    upper_bound = median_value + deviation
    plot.set_stripe(lower_bound, upper_bound, label='1% Deviation from the median')

    # maximum variation from median in %
    if all_compile_time_values:
        max_variation_abs = max((item for item in all_compile_time_values), key=lambda e: abs(e - median_value))
        max_variation = abs(median_value - max_variation_abs) * 100.0 / median_value
        plot.append_x_line(max_variation_abs, f'{"%.2f" % abs(max_variation)}% max variation from median')

    path = f'compile_time_{model_info.framework}_{model_info.name}_{model_info.precision}'
    if model_info.optional_attribute:
        path += f'_{model_info.optional_attribute}'
    path += '.png'
    plot.plot(path)


def gen_compile_time_by_iterations_from_input(inputs: List[str]):
    csv_data = get_csv_data(inputs)
    for model_info, model_data_items in compile_time_by_iterations(csv_data):
        gen_compile_time_by_iterations(model_info, model_data_items)


if __name__ == '__main__':
    gen_compile_time_by_iterations_from_input(sys.argv[1:])
