from collections import namedtuple
import csv
import os
import sys
from typing import List, Dict, Tuple, Generator, Set
import math
import numpy as np
from dataclasses import dataclass
import matplotlib.pyplot as plt


def get_csv_header(path: str) -> List[str]:
    with open(path, 'r') as f_in:
        csv_reader = csv.reader(f_in, delimiter=';')
        return next(csv_reader)


CSVColumnNames = ('model_path',
                  'model_name',
                  'model_framework',
                  'model_precision',
                  'optional_model_attribute',
                  'iteration',
                  'type',
                  'transformation_name',
                  'manager_name',
                  'duration')


CSVItem = namedtuple('CSVItem', CSVColumnNames)


class Graph:
    def __init__(self, title: str, x_label: str, y_label: str):
        self.__title = title
        self.__x_label = x_label
        self.__y_label = y_label
        self.__graphs = [] # list of tuples (x_values, y_values)
        self.__x_line_value = None
        self.__x_line_label = None
        self.__stripe_bounds = None
        self.__stripe_label = None

    def add(self, x_values: List[int], y_values: List[float], label: str):
        self.__graphs.append((x_values, y_values, label))

    def set_stripe(self, lower_bound: float, upper_bound: float, label: str):
        self.__stripe_bounds = (lower_bound, upper_bound)
        self.__stripe_label = label

    def set_x_line(self, y_value: float, label: str):
        self.__x_line_value = y_value
        self.__x_line_label = label

    def plot(self, path: str):
        need_a_legend = False

        #plt.figure(figsize=(20, 15))
        plt.figure(figsize=(8, 5))

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

        if self.__x_line_value is not None and self.__x_line_label is not None:
            # Add a horizontal line at the median value
            plt.axhline(y=self.__x_line_value, color='r', linestyle='--', label=self.__x_line_label)
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


class Unit:
    def __init__(self, csv_item: CSVItem):
        self.name = None
        if csv_item.type == 'transformation':
            self.name = csv_item.transformation_name
        elif csv_item.type == 'manager':
            self.name = csv_item.manager_name
        self.model_path = csv_item.model_path
        self.model_framework = csv_item.model_framework
        self.model_precision = csv_item.model_precision
        self.type = csv_item.type
        self.transformation_name = csv_item.transformation_name
        self.manager_name = csv_item.manager_name
        self.__durations: List[float] = [float(csv_item.duration)]
        self.__duration_median: float = None
        self.__variations: List[float] = None
        self.__duration_medians_first_iterations: List[float] = None

    def get_n_durations(self) -> int:
        return len(self.__durations)

    def get_duration_median(self) -> float:
        if not self.__durations:
            return 0.0
        if self.__duration_median is None:
            self.__duration_median = float(np.median(self.__durations))
        return self.__duration_median

    def get_duration_median_first_iterations(self, n_iterations: int) -> float:
        assert self.__durations is not None
        if self.__duration_medians_first_iterations is None:
            self.__duration_medians_first_iterations = []
            for i in range(1, len(self.__durations) + 1):
                self.__duration_medians_first_iterations.append(float(np.median(self.__durations[:i])))
        return self.__duration_medians_first_iterations[n_iterations]

    def get_variations_as_ratio(self) -> List[float]:
        if self.__variations is None:
            median = self.get_duration_median()
            self.__variations = []
            for item in self.__durations:
                if median != 0.0:
                    self.__variations.append(abs(item - median) / median)
                else:
                    self.__variations.append(0.0)
        return self.__variations


    def add(self, csv_item: CSVItem) -> None:
        assert self.model_path == csv_item.model_path
        assert self.model_framework == csv_item.model_framework
        assert self.model_precision == csv_item.model_precision
        assert self.type == csv_item.type
        assert self.transformation_name == csv_item.transformation_name
        assert self.manager_name == csv_item.manager_name
        self.__durations.append(float(csv_item.duration))
        self.__duration_median = None


def check_header(column_names: List[str]) -> None:
    column_names = set(column_names)
    for name in CSVColumnNames:
        if name == 'optional_model_attribute':
            # no such column in old CSV files
            continue
        assert name in column_names


def csv_has_optional_model_attr(path: str) -> bool:
    with open(path, 'r') as f_in:
        column_names = get_csv_header(path)
        return 'optional_model_attribute' in column_names


def read_csv(path: str) -> Generator:
    with open(path, 'r') as f_in:
        column_names = get_csv_header(path)
        check_header(column_names)
        has_optional_model_attr = 'optional_model_attribute' in column_names
        csv_reader = csv.reader(f_in, delimiter=';')
        for row in csv_reader:
            if row[-1] == 'duration':
                continue
            if not has_optional_model_attr:
                row.insert(4, '')
            yield CSVItem(*row)


UnitInfo = namedtuple('UnitInfo', ['type',
                                   'transformation_name',
                                   'manager_name'])


class ModelData:
    def __init__(self):
        self.items: List[Unit] = []
        self.__item_last_idx = None
        self.__max_n_iterations: int = None

    def append(self, csv_item: CSVItem) -> None:
        n_iteration = int(csv_item.iteration)
        assert n_iteration > 0
        if n_iteration == 1:
            self.items.append(Unit(csv_item))
        else:
            if (self.__item_last_idx is None or
                    self.__item_last_idx == len(self.items) - 1):
                self.__item_last_idx = 0
            else:
                self.__item_last_idx += 1
            self.items[self.__item_last_idx].add(csv_item)

    def get_items(self, filter_item_func) -> Generator[Unit, None, None]:
        for item in self.items:
            if filter_item_func(item):
                yield item

    def get_items_by_type(self, type_name: str) -> Generator[Unit, None, None]:
        return self.get_items(lambda item: item.type == type_name)

    def get_compile_time(self) -> float:
        item = next(self.get_items_by_type('compile_time'))
        return item.get_duration_median()

    def get_duration(self, i: int) -> float:
        return self.items[i].get_duration_median()

    def get_all_item_info(self) -> List[UnitInfo]:
        data = []
        for item in self.items:
            data.append(UnitInfo(item.type,
                                 item.transformation_name,
                                 item.manager_name))
        return data

    def get_max_n_iterations(self):
        if self.__max_n_iterations is None:
            self.__max_n_iterations = max(item.get_n_durations() for item in self.items)
        return self.__max_n_iterations

    def get_item_info(self, i: int) -> UnitInfo:
        item = self.items[i]
        return UnitInfo(item.type, item.transformation_name, item.manager_name)

    def check(self) -> None:
        if len(self.items) == 0:
            return
        n_iteration_items = [item.get_n_durations() for item in self.items]
        assert all(e == n_iteration_items[0] for e in n_iteration_items), \
            f'different number of items in different iterations: {n_iteration_items}'
        # check if there is compile time in each iteration
        n_compile_time_items = sum(1 for _ in self.get_items_by_type('compile_time'))
        assert n_compile_time_items == 1, \
            f'iteration data must consists exact 1 compile_time item but there are: {n_compile_time_items}'


ModelInfo = namedtuple('ModelInfo', ['path',
                                     'framework',
                                     'name',
                                     'precision',
                                     'optional_attribute'])


optional_model_attr_cache = {}


def get_optional_model_attr(path: str) -> str:
    if path in optional_model_attr_cache:
        return optional_model_attr_cache[path]
    if 'compressed_weights' not in path:
        return ''

    def get_path_components(path: str) -> List[str]:
        result = []
        while path != '/' and path != '\\':
            result.append(os.path.basename(path))
            path = os.path.dirname(path)
        return result
    components = get_path_components(path)
    attr = components[1]
    optional_model_attr_cache[path] = attr
    return attr


def read_csv_data(path: str) -> Dict[ModelInfo, ModelData]:
    print(f'reading {path} ...')
    csv_rows = read_csv(path)
    has_optional_model_attr = csv_has_optional_model_attr(path)
    data: Dict[ModelInfo, ModelData] = {}
    for item in csv_rows:
        if has_optional_model_attr:
            opt_model_attr = item.optional_model_attribute
        else:
            opt_model_attr = get_optional_model_attr(item.model_path)
        model_info = ModelInfo(item.model_path,
                               item.model_framework,
                               item.model_name,
                               item.model_precision,
                               opt_model_attr)
        if model_info not in data:
            data[model_info] = ModelData()
        data[model_info].append(item)
    return data

def check_csv_data(data: Dict[ModelInfo, ModelData]) -> None:
    for info, model_data in data.items():
        try:
            model_data.check()
        except AssertionError:
            print(f'assertion error while checking model data {info}')
            raise


def get_csv_data(csv_paths: List[str]) -> List[Dict[ModelInfo, ModelData]]:
    csv_data = []
    for csv_path in csv_paths:
        current_csv_data = read_csv_data(csv_path)
        check_csv_data(current_csv_data)
        csv_data.append(current_csv_data)
    return csv_data


def get_all_models(data: List[Dict[ModelInfo, ModelData]]) -> Set[ModelInfo]:
    return set(model_info for csv_data in data for model_info in csv_data)


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


'''
def gen_variation_std_dev_graph(model_info: ModelInfo, model_data: List[float]):
    title = f'Standart deviation transformations durations {model_info.framework} {model_info.name} {model_info.precision}'
    x_label = 'Iteration number'
    y_label = 'Standart deviation durations ratio std_dev[(value - median) / median]'

    print(f'generate variation std dev graph {title}')

    graph = Graph(title, x_label, y_label)
    iterations = list(range(1, len(model_data) + 1))
    graph.add(iterations, model_data)
    path = f'compile_time_{model_info.framework}_{model_info.name}_{model_info.precision}.png'
    graph.plot(path)



@dataclass
class Total:
    variation: float
    count: int


if __name__ == '__main__':
    csv_data = get_csv_data(sys.argv[1:])
    # collect variations
    variation_dict: Dict[ModelInfo, List[Total]] = {} # model_info: variation each item for each iteration
    for model_info in get_all_models(csv_data):
        if model_info not in variation_dict:
            variation_dict[model_info] = []
        model_variations = variation_dict[model_info]
        for csv_item in csv_data:
            model_data = csv_item[model_info]
            # TODO: check managers
            for item in model_data.get_items_by_type('transformation'):
                if item.get_duration_median() < 1_000:
                    continue
                # each list item for each iteration
                variations: List[float] = item.get_variations_as_ratio()
                while len(model_variations) < len(variations):
                    model_variations.append(Total(0.0, 0))
                for i, var in enumerate(variations):
                    model_variations[i].variation += math.ldexp(var, 1)
                    model_variations[i].count += 1
    # calculate data
    result_std_dev: Dict[ModelInfo, List[float]] = {} # model_info: std_dev each item for each iteration
    for model_info in variation_dict:
        variations: List[Total] = variation_dict[model_info]
        result_std_dev[model_info] = []
        for var in variations:
            result_std_dev[model_info].append(math.sqrt(var.variation / var.count))
        gen_variation_std_dev_graph(model_info, result_std_dev[model_info])
'''


def gen_std_dev_graph_deltas(model_info: ModelInfo, duration_deltas_stddev: List[float]):
    x_label = 'Iteration number'
    y_label = 'Standard deviation #2 - #1 (ms)'

    title = f'Standard deviation #2 - #1 {model_info.framework} {model_info.name} {model_info.precision}'
    print(f'generate variation std dev graph {title}')

    graph = Graph(title, x_label, y_label)
    iterations = list(range(1, len(duration_deltas_stddev) + 1))
    graph.add(iterations, duration_deltas_stddev)
    graph.set_stripe(0.0, 1.0, '1 ms')
    path = f'delta_{model_info.framework}_{model_info.name}_{model_info.precision}.png'
    graph.plot(path)


def gen_std_dev_graph_ratio(model_info: ModelInfo, duration_deltas_stddev: List[float]):
    x_label = 'Iteration number'
    y_label = 'Standard deviation ratio #2/#1'

    title = f'Standard deviation #2/#1 {model_info.framework} {model_info.name} {model_info.precision}'
    print(f'generate variation std dev graph {title}')

    graph = Graph(title, x_label, y_label)
    iterations = list(range(1, len(duration_deltas_stddev) + 1))
    graph.add(iterations, duration_deltas_stddev)
    graph.set_stripe(0.0, 0.02, '2%')
    path = f'ratio_{model_info.framework}_{model_info.name}_{model_info.precision}.png'
    graph.plot(path)


def iteration_statistics_in_comparison(data: List[Dict[ModelInfo, ModelData]],
                                       unit_type: str):
    assert len(data) == 2

    @dataclass
    class Total:
        duration: float
        count: int


    def aggregate_unit_data(data: Dict[ModelInfo, ModelData],
                            n_iterations: int,
                            unit_type: str) -> Dict[str, Total]:
        result: Dict[str, Total] = {} # ts name: Total
        for model_info, model_data in data.items():
            for item in model_data.get_items_by_type(unit_type):
                if item.name not in result:
                    result[item.name] = Total(0.0, 0)
                item_duration = item.get_duration_median_first_iterations(n_iterations)
                assert not math.isnan(item_duration)
                result[item.name].duration += item_duration / 1_000_000
                result[item.name].count += 1
        return result


    def get_min_n_iterations(data: List[Dict[ModelInfo, ModelData]]) -> int:
        return min(model_data.get_max_n_iterations() for csv_data in data for _, model_data in csv_data.items())


    def get_duration(aggregated_data_item: Dict[str, Total], name: str) -> float:
        duration = 0.0
        if name in aggregated_data_item:
            duration = aggregated_data_item[name].duration
        return duration


    duration_deltas_stddev = [] # each index - for each iteration
    duration_ratios_stddev = [] # each index - for each iteration
    for n_iterations in range(1, get_min_n_iterations(data)):
        aggregated_data = [aggregate_unit_data(csv_data, n_iterations, unit_type) for csv_data in data]
        all_transformations = set(ts_name for aggregated_data_item in aggregated_data
                              for ts_name in aggregated_data_item)
        duration_deltas = []
        duration_ratios = []
        for name in all_transformations:
            duration1 = get_duration(aggregated_data[0], name)
            duration2 = get_duration(aggregated_data[1], name)

            if duration1 < 1.0:
                continue

            duration_deltas.append(duration2 - duration1)

            ratio = 0.0
            if duration2 != 0.0:
                ratio = duration1/duration2
            duration_ratios.append(ratio)
        duration_deltas_stddev.append(float(np.std(duration_deltas)))
        duration_ratios_stddev.append(float(np.std(duration_ratios)))
        assert 0.0 not in duration_ratios_stddev, f'stddev {duration_ratios_stddev} ratios {duration_ratios}'

    return duration_deltas_stddev, duration_ratios_stddev


def compare_compile_time(data: List[Dict[ModelInfo, ModelData]]):
    if len(data) == 0:
        return [], []

    n_cvs_files = len(data)
    table = []
    models = [model_info for model_info in data[0]]
    for model_info in models:
        row = [model_info]
        compile_times = []
        for csv_idx in range(n_cvs_files):
            model_data = data[csv_idx][model_info]
            compile_time = model_data.get_compile_time() / 1_000_000_000
            compile_times.append(compile_time)
        for csv_idx in range(n_cvs_files):
            row.append(compile_times[csv_idx])
        for csv_idx in range(1, n_cvs_files):
            delta = compile_times[csv_idx] - compile_times[0]
            row.append(delta)
        for csv_idx in range(1, n_cvs_files):
            ratio = compile_times[csv_idx] / compile_times[0]
            row.append(ratio)
        table.append(row)
    return table


def get_common_models(data: List[Dict[ModelInfo, ModelData]]) -> List[ModelInfo]:
    if len(data) == 0:
        return []
    common_keys = data[0].keys()
    for csv_data in data:
        common_keys = common_keys & csv_data.keys()
    return common_keys


def filter_common_models(data: List[Dict[ModelInfo, ModelData]]) -> List[Dict[ModelInfo, ModelData]]:
    common_models: List[ModelInfo] = get_common_models(data)
    return filter_by_models(data, common_models)


def main_iteration_statistics_in_comparison():
    UNIT_TYPE = 'transformation'
    csv_data = get_csv_data(sys.argv[1:])

    for model_info in get_all_models(csv_data):
        duration_deltas_stddev, duration_ratios_stddev = iteration_statistics_in_comparison(filter_by_models(csv_data,
                                                                                                         [model_info]),
                                                                                        UNIT_TYPE)
        assert len(duration_deltas_stddev) == len(duration_ratios_stddev)
        assert 0.0 not in duration_deltas_stddev, f'{duration_deltas_stddev}'
        gen_std_dev_graph_deltas(model_info, duration_deltas_stddev)
        assert 0.0 not in duration_ratios_stddev, f'{duration_ratios_stddev}'
        gen_std_dev_graph_ratio(model_info, duration_ratios_stddev)


def compare_compile_time_for_series(inputs: List[str]) -> Dict[ModelInfo, List[float]]:
    csv_data = get_csv_data(inputs)
    csv_data_common_models = filter_common_models(csv_data)
    if not csv_data_common_models:
        print('no common models to compare compilation time ...')
        return {}
    rows = compare_compile_time(csv_data_common_models)
    n_csv_files = len(inputs)
    result = {}
    for row in rows:
        result[row[0]] = row[1:n_csv_files + 1]
    return result


def gen_compile_time_graph(model_info: ModelInfo, compile_time_series: List[Tuple[str, List[float]]]):
    x_label = 'Dev trigger job run number'
    y_label = 'Compilation time (secs)'

    title = f'Compilation time {model_info.framework} {model_info.name} {model_info.precision}'
    print(f'generate graph {title}')

    graph = Graph(title, x_label, y_label)
    for label, serie in compile_time_series:
        iterations = list(range(1, len(serie) + 1))
        graph.add(iterations, serie, label)

    # median of first serie
    median_value = float(np.median(compile_time_series[0][1]))
    graph.set_x_line(median_value, f'Median {compile_time_series[0][0]}: {"%.2f" % median_value} secs')
    deviation = 0.1 * median_value
    lower_bound = median_value - deviation
    upper_bound = median_value + deviation
    graph.set_stripe(lower_bound, upper_bound, label='10% Deviation from the median')

    path = f'compile_time_{model_info.framework}_{model_info.name}_{model_info.precision}.png'
    graph.plot(path)


def main_compare_compile_time_rpl_05_12():
    inputs1 = ['C:\\Users\\ekotov\\WORK\\TO_DELETE\\Ivan_experiment3\\rpl_05_01.csv',
              "C:\\Users\\ekotov\\WORK\\TO_DELETE\\Ivan_experiment3\\rpl_05_03.csv",
              "C:\\Users\\ekotov\\WORK\\TO_DELETE\\Ivan_experiment3\\rpl_05_07.csv",
              "C:\\Users\\ekotov\\WORK\\TO_DELETE\\Ivan_experiment3\\rpl_05_08.csv",
              "C:\\Users\\ekotov\\WORK\\TO_DELETE\\Ivan_experiment3\\rpl_05_05.csv"]
    inputs2 = ["C:\\Users\\ekotov\\WORK\\TO_DELETE\\Ivan_experiment3\\rpl_12_04.csv",
               "C:\\Users\\ekotov\\WORK\\TO_DELETE\\Ivan_experiment3\\rpl_12_09.csv",
               "C:\\Users\\ekotov\\WORK\\TO_DELETE\\Ivan_experiment3\\rpl_12_09.csv",
               "C:\\Users\\ekotov\\WORK\\TO_DELETE\\Ivan_experiment3\\rpl_12_last.csv"]
    series1 = compare_compile_time_for_series(inputs1)
    series2 = compare_compile_time_for_series(inputs2)
    for model_info, serie1 in series1.items():
        serie2 = series2[model_info]
        gen_compile_time_graph(model_info, [('rpl-05', serie1), ('rpl-12', serie2)])


def main_compare_compile_time_my_jobs():
    inputs = ['C:\\Users\\ekotov\\WORK\\TO_DELETE\\ov_transformations_stats_196.csv',
              'C:\\Users\\ekotov\\WORK\\TO_DELETE\\ov_transformations_stats_197.csv',
              'C:\\Users\\ekotov\\WORK\\TO_DELETE\\ov_transformations_stats_203.csv',
              'C:\\Users\\ekotov\\WORK\\TO_DELETE\\ov_transformations_stats_204_dev_10_iter.csv',
              'C:\\Users\\ekotov\\WORK\\TO_DELETE\\ov_transformations_stats_186.csv',
              'C:\\Users\\ekotov\\WORK\\TO_DELETE\\ov_transformations_stats_191.csv',
              'C:\\Users\\ekotov\\WORK\\TO_DELETE\\ov_transformations_stats_193.csv',
              'C:\\Users\\ekotov\\WORK\\TO_DELETE\\ov_transformations_stats_193_new.csv']
    series = compare_compile_time_for_series(inputs)
    for model_info, serie in series.items():
        gen_compile_time_graph(model_info, [('', serie)])


if __name__ == '__main__':
    #main_compare_compile_time_my_jobs()
    main_compare_compile_time_rpl_05_12()