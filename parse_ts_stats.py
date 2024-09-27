import csv
from collections import namedtuple
import typing
import sys
import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass


# model_path;model_name;model_framework;model_precision;iteration;type;transformation_name;manager_name;duration

def read_csv(path: str) -> typing.Generator:
    def get_csv_header(path):
        with open(path, 'r') as f_in:
            csv_reader = csv.reader(f_in, delimiter=';')
            return next(csv_reader)
    
    with open(path, 'r') as f_in:
        column_names = get_csv_header(path)
        Item = namedtuple('Item', column_names)
        csv_reader = csv.reader(f_in, delimiter=';')
        for row in csv_reader:
            if row[-1] == 'duration':
                continue
            yield Item(*row)


def read_csv_data(path: str) -> typing.Dict[str, typing.List[typing.List]]:
    print('read csv ...')
    csv_rows = read_csv(path)
    model_data = {} # model_path: [[Item]] - list of iterations
    for item in csv_rows:
        model_path = item.model_path
        if not model_path in model_data:
            model_data[model_path] = []
        data_value = model_data[model_path]
        n_iter = int(item.iteration)
        if len(data_value) < n_iter:
            data_value.append([])
        data_value[n_iter - 1].append(item)
    return model_data


@dataclass
class ModelInfo:
    framework: str
    name: str
    precision: str


def get_model_info(data: typing.List[typing.List]) -> ModelInfo:
    first_item = data[0][0]
    return ModelInfo(first_item.model_framework, first_item.model_name, first_item.model_precision)


# not sure that there are the same count of iterations in each CSV file
# return list - each item from the list - for appropriate CSV file
def get_durations(data: typing.List[typing.List[typing.List]]) -> typing.List[np.ndarray]:
    durations = []
    for csv_data in data:
        duration_items = [[float(item.duration) for item in iter_data] for iter_data in csv_data]
        durations.append(np.array(duration_items))
    return durations


def get_median(data: np.ndarray) -> np.ndarray:
    return np.median(data, axis=0, keepdims=True)

def get_medians(data: typing.List[np.ndarray]) -> typing.List[np.ndarray]:
    medians = []
    for csv_data in data:
        medians.append(get_median(csv_data))
    return medians


def get_max_deviations(data: typing.List[np.ndarray], values: typing.List[np.ndarray]) -> typing.List[np.ndarray]:
    max_deviations = []
    assert len(data) == len(values)
    for i in range(len(data)):
        csv_data = data[i]
        #print(csv_data.shape)
        #print(f'durations #152 {csv_data[:,152]}')
        medians = values[i]
        #print(f'medians.shape {medians.shape}')
        #print(f'median #152 {medians[0, 152]}')
        deviations = np.abs(medians - csv_data) / medians * 100.0
        #print(f'deviations  #152 {deviations[:, 152]}')
        max_deviation = np.max(deviations, axis=0)
        #print(f' max_deviation.shape {max_deviation.shape}')
        #print(f'max_deviation  #152 {max_deviation[152]}')
        max_deviations.append(max_deviation)
    return max_deviations


def get_std_deviations(data: np.ndarray) -> np.ndarray:
    pass # TODO


def get_median_percent_deviations(data: typing.List[typing.List]) -> np.ndarray:
    '''
    # DEBUG
    items = [[item for item in sublist] for sublist in data]
    item_durations = [item[152].duration for item in items]
    item_objs = [item[152] for item in items]
    print(item_objs)
    print(f'item_durations {' '.join(item_durations)}')
    # DEBUG
    durations = get_durations(data)
    print(f'durations {durations[:,152]}')
    medians = get_median(durations)
    print(f'medians {medians[:,152]}')
    #return (durations - medians) / medians * 100
    retval = (durations - medians) / medians * 100
    print(f'retval {retval[:,152]}')
    print(f'max retval {np.max(retval[:,152])}')
    return retval
    '''
    durations = get_durations(data)
    medians = get_median(durations)
    return (durations - medians) / medians * 100


def check_no_duplicate_model_path(path: str):
    print('checking duplicate model_path ...')
    csv_rows = read_csv(path)
    data_dict = {} # model_path: (model_name, model_framework, model_precision)
    for item in csv_rows:
        if item.model_path not in data_dict:
            data_dict[item.model_path] = (item.model_name, item.model_framework, item.model_precision)
            continue
        value = data_dict[item.model_path]
        assert value == (item.model_name, item.model_framework, item.model_precision)


def check_iterations(model_data: typing.List[typing.List]):
    for n_iter in range(1, len(model_data)):
        data1 = model_data[n_iter - 1]
        data2 = model_data[n_iter]
        assert len(data1) == len(data2), f'assertion failed: {len(data1)} != {len(data2)}'
        for n_item in range(len(data1)):
            item1 = data1[n_item]
            item2 = data2[n_item]
            assert int(item1.iteration) == n_iter
            assert int(item2.iteration) == n_iter + 1
            assert item1.model_precision == item2.model_precision
            assert item1.model_path == item2.model_path
            assert item1.model_name == item2.model_name
            assert item1.model_framework == item2.model_framework
            assert item1.type == item2.type
            assert item1.transformation_name == item2.transformation_name
            assert item1.manager_name == item2.manager_name


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

    def add(self, x_values: typing.List[int], y_values: typing.List[float]):
        self.__graphs.append((x_values, y_values))

    def set_stripe(self, lower_bound: float, upper_bound: float, label: str):
        self.__stripe_bounds = (lower_bound, upper_bound)
        self.__stripe_label = label

    def set_x_line(self, y_value: float, label: str):
        self.__x_line_value = y_value
        self.__x_line_label = label

    def plot(self, path: str):
        #plt.figure(figsize=(20, 15))
        plt.figure(figsize=(8, 5))
        first_x_values, first_y_values = self.__graphs[0]
        for graph_item in self.__graphs:
            x_values, y_values = graph_item
            plt.plot(x_values, y_values, marker='o')
        # Adding labels and title
        plt.title(self.__title)
        plt.xlabel(self.__x_label)
        plt.ylabel(self.__y_label)

        if self.__x_line_value is not None and self.__x_line_label is not None:
            # Add a horizontal line at the median value
            plt.axhline(y=self.__x_line_value, color='r', linestyle='--', label=self.__x_line_label)

        if self.__stripe_bounds is not None and self.__stripe_label is not None:
            # Add a stripe representing 10% deviation from the median
            lower_bound, upper_bound = self.__stripe_bounds
            plt.fill_between(first_x_values,
                             lower_bound, upper_bound, color='green', alpha=0.2, label=self.__stripe_label)

        # Show only integers on the X-axis
        plt.xticks(ticks=np.arange(int(min(first_x_values)), int(max(first_x_values))+1, 1))

        # Add a legend
        plt.legend()

        # Show the graph
        plt.grid(True)
        plt.savefig(path)
        # Close the plot so it doesn't show up
        plt.close()


def gen_compile_time_graph(model_info: ModelInfo, model_data: typing.List[typing.List[typing.List]]):

    def get_iter_time_values(time_values: typing.List[float]):
        # get values in seconds
        prev = 0.0
        result = []
        for value in time_values:
            new_value = prev + value
            result.append(new_value)
            prev = new_value
        return result

    title = f'Compile time {model_info.framework} {model_info.name} {model_info.precision}'
    x_label = 'Job run time (seconds)'
    y_label = 'Compile time (seconds)'

    print(f'generate compile time graph {title}')

    all_compile_time_values = []
    graph = Graph(title, x_label, y_label)
    for current_data in model_data:
        compile_time_values = [float(item.duration) / 1_000_000_000
                               for l in current_data for item in l if item.type == 'compile_time']
        iterations = get_iter_time_values(compile_time_values)

        all_compile_time_values.extend(compile_time_values)

        assert len(compile_time_values) == len(iterations)
        graph.add(iterations, compile_time_values)

    # Calculate the median value of y_values
    median_value = float(np.median(all_compile_time_values))
    graph.set_x_line(median_value, f'Median: {"%.2f" % median_value} secs')
    # Calculate 10% deviation from the median
    deviation = 0.01 * median_value
    lower_bound = median_value - deviation
    upper_bound = median_value + deviation
    graph.set_stripe(lower_bound, upper_bound, label='1% Deviation from the median')

    path = f'compile_time_{model_info.framework}_{model_info.name}_{model_info.precision}.png'
    graph.plot(path)


def gen_scatter_graph_agg(x_values, x_label, y_values, y_label, title, path):
    plt.figure(figsize=(20, 15))
    plt.scatter(x_values, y_values)

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    plt.xscale('log')

    # Show the graph
    plt.grid(True)

    plt.savefig(path)
    # Close the plot so it doesn't show up
    plt.close()


def gen_median_max_deviation_scatter_graph_pro_model(model_info, x_values_list: typing.List[np.ndarray],
                                                     y_values_list: typing.List[np.ndarray]):
    print(f'generating median deviation scatter graph for {model_info.framework} {model_info.name} {model_info.precision} ...')
    assert len(x_values_list) == len(y_values_list)
    x_values = np.array([])
    y_values = np.array([])
    for i in range(len(x_values_list)):
        x_values = np.append(x_values, x_values_list[i])
        y_values = np.append(y_values, y_values_list[i])
    title = f'Dependency median and max deviation from median {model_info.framework} {model_info.name} {model_info.precision}'
    x_label = 'Max deviation from median %'
    y_label = 'Median (nano seconds)'
    path = f'median_deviation_scatter_{model_info.framework}_{model_info.name}_{model_info.precision}.png'
    gen_scatter_graph_agg(x_values, x_label, y_values, y_label, title, path)


def gen_median_max_deviation_scatter_graph(x_values_list: typing.List[typing.List[np.ndarray]],
                                           y_values_list: typing.List[typing.List[np.ndarray]]):
    print(f'generating median deviation scatter graph for all models ...')
    assert len(x_values_list) == len(y_values_list)
    x_values = np.array([])
    y_values = np.array([])
    for model_data in x_values_list:
        for csv_data in model_data:
            x_values = np.append(x_values, csv_data)
    for model_data in y_values_list:
        for csv_data in model_data:
            y_values = np.append(y_values, csv_data)
    title = f'Dependency median and max deviation from median all models'
    x_label = 'Max deviation from median %'
    y_label = 'Median (nano seconds)'
    path = f'median_deviation_scatter_all_models.png'
    gen_scatter_graph_agg(x_values, x_label, y_values, y_label, title, path)


def get_model_paths(data: typing.List) -> typing.Set[str]:
    model_paths = set()
    for path_data in data:
        for path in path_data:
            model_paths.add(path)
    return model_paths


def get_model_data(data: typing.List) -> typing.Generator:
    paths = get_model_paths(data)
    for path in paths:
        model_data = []
        for path_data in data:
            if path not in path_data:
                continue
            model_data.append(path_data[path])
        if len(model_data) == 0:
            continue
        yield get_model_info(model_data[0]), model_data


@dataclass
class Config:
    check_duplicates: bool = False
    gen_compile_time_graphs: bool = True
    median_max_deviation_scatter_per_model_graphs: bool = False
    median_max_deviation_scatter_all_models_graph: bool = False

if __name__ == '__main__':
    csv_paths = sys.argv[1:]
    config = Config()

    all_model_data = []
    for csv_path in csv_paths:
        print(f'proceeding {csv_path} ...')
        if config.check_duplicates:
            check_no_duplicate_model_path(csv_path)
        model_data = read_csv_data(csv_path)
        all_model_data.append(model_data)

    investigate_deviations = (config.median_max_deviation_scatter_per_model_graphs or
                              config.median_max_deviation_scatter_all_models_graph)

    all_model_medians = []
    all_model_max_median_deviations = []
    for model_info, model_data in get_model_data(all_model_data):
        if config.gen_compile_time_graphs:
            gen_compile_time_graph(model_info, model_data)
        if investigate_deviations:
            durations = get_durations(model_data)
            medians = get_medians(durations)
            max_deviations = get_max_deviations(durations, medians)
            if config.median_max_deviation_scatter_per_model_graphs:
                gen_median_max_deviation_scatter_graph_pro_model(model_info, medians, max_deviations)
            all_model_medians.append(medians)
            all_model_max_median_deviations.append(max_deviations)
    if config.median_max_deviation_scatter_all_models_graph:
        gen_median_max_deviation_scatter_graph(all_model_medians, all_model_max_median_deviations)
