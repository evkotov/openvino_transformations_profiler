from typing import List, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np


class Plot:
    def __init__(self, title: str, x_label: str, y_label: str):
        self.__title = title
        self.__x_label = x_label
        self.__y_label = y_label
        self.__graphs = [] # list of tuples (x_values, y_values)
        self.__x_lines = []
        self.__stripe_bounds = None
        self.__stripe_label = None
        self.__plot_size = (8, 5)
        self.__legends = []

    def append_legend(self, label: str):
        self.__legends.append(label)

    def add(self, x_values: List, y_values: List[float], label: Optional[str] = None):
        self.__graphs.append((x_values, y_values, label))

    def set_stripe(self, lower_bound: float, upper_bound: float, label: str):
        self.__stripe_bounds = (lower_bound, upper_bound)
        self.__stripe_label = label

    def append_x_line(self, y_value: float, label: str):
        self.__x_lines.append((y_value, label))

    def set_plot_size(self, size: Tuple[int, int]) -> None:
        self.__plot_size = size

    def plot(self, path: str):
        need_a_legend = False

        plt.figure(figsize=self.__plot_size)

        first_x_values, first_y_values, label = self.__graphs[0]
        all_x_values = set(first_x_values)
        for graph_item in self.__graphs:
            x_values, y_values, label = graph_item
            plt.plot(x_values, y_values, marker='o', label=label)
            if label:
                need_a_legend = True
            all_x_values.update(x_values)
        # Adding labels and title
        plt.title(self.__title)
        plt.xlabel(self.__x_label)
        plt.ylabel(self.__y_label)

        all_x_values_sorted = sorted(all_x_values)

        for x_line_value, x_line_label in self.__x_lines:
            # Add a horizontal line at the median value
            plt.axhline(y=x_line_value, linestyle='--', label=x_line_label)
            need_a_legend = True

        if self.__stripe_bounds is not None and self.__stripe_label is not None:
            # Add a stripe representing 10% deviation from the median
            lower_bound, upper_bound = self.__stripe_bounds
            plt.fill_between(all_x_values_sorted,
                             lower_bound, upper_bound, color='green', alpha=0.2, label=self.__stripe_label)
            need_a_legend = True

        # Show only integers on the X-axis
        plt.xticks(ticks=np.arange(int(all_x_values_sorted[0]), int(all_x_values_sorted[-1]) + 1, 1))

        # Add a legend
        if need_a_legend:
            plt.legend()

        # Show the graph
        plt.grid(True)
        plt.savefig(path)
        # Close the plot so it doesn't show up
        plt.close()
