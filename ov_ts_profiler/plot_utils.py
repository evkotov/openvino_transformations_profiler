from __future__ import annotations

import os
from collections import namedtuple
from typing import List, Tuple, Optional, Iterator

import matplotlib.pyplot as plt
import numpy as np

from ov_ts_profiler.common_structs import ModelInfo, ComparisonValues
from ov_ts_profiler.output_utils import make_model_file_name

PlotDots = namedtuple('PlotDots', ['x_values', 'y_values', 'label'])
Stripe = namedtuple('Stripe', ['lower_bound', 'upper_bound', 'label'])
Xline = namedtuple('Xline', ['y_value', 'label', 'color', 'style'])
PlotDotsWithXline = namedtuple('PlotDotsWithXline', ['x_values', 'y_values', 'y_line_value', 'line_style', 'line_label'])
StripeNoneLine = namedtuple('StripeNoneLine', ['x_values', 'min_y_values', 'max_y_values', 'label'])

def generate_x_ticks_cast_to_int(x_values: List[float]):
    return np.arange(int(min(x_values)), int(max(x_values)) + 1, 1)


class Plot:
    def __init__(self, title: str, x_label: str, y_label: str):
        self.__title = title
        self.__x_label = x_label
        self.__y_label = y_label
        self.__graphs: List[PlotDots] = []
        self.__graphs_with_xline: List[PlotDotsWithXline] = []
        self.__x_lines = []
        self.__stripe: Optional[Stripe] = None
        self.__plot_size = (8, 5)
        self.__legends = []
        self.__gen_x_ticks_func = None
        self.__stripe_none_line = None

    def append_legend(self, label: str):
        self.__legends.append(label)

    def set_x_ticks_func(self, func):
        self.__gen_x_ticks_func = func

    def add(self, x_values: List, y_values: List[float], label: Optional[str] = None):
        self.__graphs.append(PlotDots(x_values, y_values, label))

    def add_stripe_non_line(self, x_values, min_y_values, max_y_values, label):
        self.__stripe_none_line = StripeNoneLine(x_values, min_y_values, max_y_values, label)

    def add_with_xline(self, x_values: List, y_values: List[float], y_line_value: float, line_style: str, line_label: str):
        self.__graphs_with_xline.append(PlotDotsWithXline(x_values, y_values, y_line_value, line_style, line_label))

    def set_stripe(self, lower_bound: float, upper_bound: float, label: str):
        self.__stripe = Stripe(lower_bound, upper_bound, label)

    def append_x_line(self, y_value: float, label: str, color: str, style: str):
        self.__x_lines.append(Xline(y_value, label, color, style))

    def set_plot_size(self, size: Tuple[int, int]) -> None:
        self.__plot_size = size

    def plot(self, path: str):
        plt.rcParams.update({'font.size': 14})  # Set global font size
        need_a_legend = False

        plt.figure(figsize=self.__plot_size)

        all_x_values = set()
        for graph_item in self.__graphs:
            plt.plot(graph_item.x_values, graph_item.y_values, marker='o', label=graph_item.label)
            if graph_item.label:
                need_a_legend = True
            all_x_values.update(graph_item.x_values)
        for graph_item in self.__graphs_with_xline:
            line, = plt.plot(graph_item.x_values, graph_item.y_values, marker='o')
            line_color = line.get_color()
            plt.axhline(y=graph_item.y_line_value, linestyle=graph_item.line_style,
                        label=graph_item.line_label, color=line_color)
            need_a_legend = True
            all_x_values.update(graph_item.x_values)

        # Adding labels and title
        plt.title(self.__title)
        plt.xlabel(self.__x_label)
        plt.ylabel(self.__y_label)

        for x_line in self.__x_lines:
            # Add a horizontal line at the median value
            plt.axhline(y=x_line.y_value, linestyle=x_line.style,
                        label=x_line.label, color=x_line.color)
            need_a_legend = True

        if self.__stripe is not None:
            x_min, x_max = plt.gca().get_xlim()
            # Add a stripe representing 10% deviation from the median
            plt.fill_between([x_min, x_max],
                             self.__stripe.lower_bound, self.__stripe.upper_bound,
                             color='green', alpha=0.2, label=self.__stripe.label)
            need_a_legend = True

        if self.__stripe_none_line is not None:
            plt.fill_between(self.__stripe_none_line.x_values,
                             self.__stripe_none_line.min_y_values,
                             self.__stripe_none_line.max_y_values,
                             color='green', alpha=0.2, label=self.__stripe_none_line.label)

        if self.__gen_x_ticks_func:
            plt.xticks(ticks=self.__gen_x_ticks_func(all_x_values))

        # Add a legend
        if need_a_legend:
            plt.legend()

        # Show the graph
        plt.grid(True)
        plt.savefig(path)
        # Close the plot so it doesn't show up
        plt.close()

class Hist:
    def __init__(self, title: str, x_label: str, y_label: str):
        self.__title = title
        self.__x_label = x_label
        self.__y_label = y_label
        self.__bins = 10
        self.__values = []
        self.__plot_size = (10, 8)

    def set_bins(self, bins: int):
        self.__bins = bins

    def set_values(self, values: List[float]):
        self.__values = values

    def plot(self, path: str):
        plt.rcParams.update({'font.size': 14})  # Set global font size
        plt.figure(figsize=self.__plot_size)
        plt.hist(self.__values, bins=self.__bins, edgecolor='black')

        # Add titles and labels
        plt.title(self.__title)
        plt.xlabel(self.__x_label)
        plt.ylabel(self.__y_label)

        plt.savefig(path)
        # Close the plot so it doesn't show up
        plt.close()


class ScatterPlot:
    HLine = namedtuple('HLine', ['y_value', 'label'])
    HFill = namedtuple('HFill', ['y_min', 'y_max', 'label'])
    def __init__(self, title: str, x_label: str, y_label: str):
        self.__title = title
        self.__x_label = x_label
        self.__y_label = y_label
        self.__x_values: List[float] = []
        self.__y_values: List[float] = []
        self.__plot_size = (10, 8)
        self.__h_line: Optional[ScatterPlot.HLine] = None
        self.__h_fill: Optional[ScatterPlot.HFill] = None

    def set_values(self, x_values: List[float], y_values: List[float]):
        self.__x_values = x_values
        self.__y_values = y_values

    def add_horizontal_line(self, y_value: float, label: str):
        self.__h_line = ScatterPlot.HLine(y_value, label)

    def add_horizontal_fill(self, y_min: float, y_max: float, label: str):
        self.__h_fill = ScatterPlot.HFill(y_min, y_max, label)

    def plot(self, path: str):
        plt.rcParams.update({'font.size': 14})  # Set global font size
        plt.figure(figsize=self.__plot_size)
        assert len(self.__x_values) == len(self.__y_values)
        plt.scatter(self.__x_values, self.__y_values)

        # Add titles and labels
        plt.title(self.__title)
        plt.xlabel(self.__x_label)
        plt.ylabel(self.__y_label)

        need_legend = False
        if self.__h_line:
            plt.axhline(y=self.__h_line.y_value, color='b', linestyle='-', label=self.__h_line.label)
            need_legend = True
        if self.__h_fill:
            plt.fill_between(
                x=np.linspace(min(self.__x_values), max(self.__x_values), 100),  # Ensure the stripe spans the x-range
                y1=self.__h_fill.y_min,
                y2=self.__h_fill.y_max,
                color='blue',
                alpha=0.2,
                label=self.__h_fill.label
            )
            need_legend = True

        if need_legend:
            plt.legend()

        plt.savefig(path)
        # Close the plot so it doesn't show up
        plt.close()


def divide_into_segments_by_y_values(x_values, y_values, n_segments: int):
    if n_segments == 1:
        yield x_values, y_values
        return
    if not x_values:
        return
    log_numbers = np.log(np.abs(y_values) + 1)  # Use absolute values and add 1 to handle zero and negative values
    min_log = np.min(log_numbers)
    max_log = np.max(log_numbers)
    bins = np.linspace(min_log, max_log, n_segments)
    indices = np.digitize(log_numbers, bins)

    for i in range(1, n_segments  + 1):
        x_part = []
        y_part = []
        for j in range(len(y_values)):
            if indices[j] != i:
                continue
            x_part.append(x_values[j])
            y_part.append(y_values[j])
        if not x_part:
            continue
        yield x_part, y_part


class PlotOutputRatio:
    def __init__(self):
        self.__label = ''

    def get_base_label(self):
        return self.__label

    def set_label(self, label: str):
        self.__label = label

    def get_label(self, value1_label: str, value2_label: str):
        if self.get_base_label():
            return self.get_base_label()
        return f'ratio ({value2_label}/{value1_label} - 1), %'

    def get_ratio(self, values: ComparisonValues):
        return values.get_ratios()


class PlotOutputRatioSimple(PlotOutputRatio):
    def __init__(self):
        super().__init__()

    def get_ratio(self, values: ComparisonValues):
        return values.get_simple_ratios()

    def get_label(self, value1_label: str, value2_label: str):
        if self.get_base_label():
            return self.get_base_label()
        return f'ratio ({value2_label}/{value1_label}), %'


class PlotOutput:
    def __init__(self, path_prefix, title_prefix: str, n_segments: int):
        self.path_prefix = path_prefix
        self.title_prefix = title_prefix
        self.n_segments = n_segments
        self.__get_ratio_func = PlotOutputRatio()
        self.__label1 = ''
        self.__label2 = ''
        self.__label_y_hist = ''
        self.__label_x_scatter = ''

    def set_label_y_hist(self, label: str):
        self.__label_y_hist = label

    def set_label_x_scatter(self, label: str):
        self.__label_x_scatter = label

    def set_value1_label(self, label: str):
        self.__label1 = label

    def set_value2_label(self, label: str):
        self.__label2 = label

    def set_get_ratio_func(self, get_ratio_func):
        self.__get_ratio_func = get_ratio_func

    def __get_title(self):
        title = f'{self.title_prefix}'
        if self.n_segments > 1:
            title += ' part {i}'
        return title

    def __get_output_path(self, prefix: str, plot_type: str) -> str:
        path = f'{prefix}_{plot_type}'
        if self.n_segments > 1:
            path += '_part{i}'
        path += '.png'
        return path

    def plot_scatter(self, prefix: str, i: int, max_values_part: List[float], ratios_part: List[float], y_median: float, unit: str):
        y_label = self.__get_ratio_func.get_label(self.__label1, self.__label2)
        x_label = f'max ({self.__label1}, {self.__label2}), {unit}'
        if self.__label_x_scatter:
            x_label = self.__label_x_scatter

        scatter = ScatterPlot(self.__get_title(), x_label, y_label)
        scatter.set_values(max_values_part, ratios_part)
        scatter.add_horizontal_line(y_median, f'median {y_median:.1f} %')
        scatter.plot(self.__get_output_path(prefix, 'scatter'))

    def plot_hist(self, prefix: str, i: int, ratios_part: List[float]):
        x_label = self.__get_ratio_func.get_label(self.__label1, self.__label2)

        y_label = f'number of values'
        if self.__label_y_hist:
            y_label = self.__label_y_hist

        hist = Hist(self.__get_title(), x_label, y_label)
        hist.set_values(ratios_part)
        hist.plot(self.__get_output_path(prefix, 'hist'))

    def __plot_values(self, i, max_values_part, ratios_part, unit: str, prefix: str):
        y_values = np.array(ratios_part)
        y_median = float(np.median(y_values))
        self.plot_scatter(prefix, i, max_values_part, ratios_part, y_median, unit)
        self.plot_hist(prefix, i, ratios_part)

    def plot_into_file(self, values: ComparisonValues, prefix: str):
        ratios = self.__get_ratio_func.get_ratio(values)
        max_values = values.get_max_values()
        assert len(ratios) == len(max_values)
        for i, (max_values_part, ratios_part) in enumerate(divide_into_segments_by_y_values(max_values, ratios, self.n_segments)):
            self.__plot_values(i, max_values_part, ratios_part, values.unit, prefix)

    def plot(self, values: ComparisonValues):
        self.plot_into_file(values, self.path_prefix)


def gen_plot_time_by_iterations(output_dir: str,
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

    path = os.path.join(output_dir, f'{device}_{model_info.framework}_{model_info.name}_{model_info.precision}')
    if model_info.config:
        path += f'_{model_info.config}'
    path += f'_{file_prefix}'
    path += '.png'
    plot.plot(path)
