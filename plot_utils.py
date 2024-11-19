from collections import namedtuple
from typing import List, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np


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
    def __init__(self, title: str, x_label: str, y_label: str):
        self.__title = title
        self.__x_label = x_label
        self.__y_label = y_label
        self.__x_values: List[float] = []
        self.__y_values: List[float] = []
        self.__plot_size = (10, 8)

    def set_values(self, x_values: List[float], y_values: List[float]):
        self.__x_values = x_values
        self.__y_values = y_values

    def plot(self, path: str):
        plt.figure(figsize=self.__plot_size)
        plt.scatter(self.__x_values, self.__y_values)

        # Add titles and labels
        plt.title(self.__title)
        plt.xlabel(self.__x_label)
        plt.ylabel(self.__y_label)

        plt.savefig(path)
        # Close the plot so it doesn't show up
        plt.close()
