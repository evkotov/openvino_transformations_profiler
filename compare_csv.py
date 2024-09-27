import argparse
from collections import namedtuple
import csv
from dataclasses import dataclass, field
import os
import sys
from typing import List, Dict, Tuple, Generator, Set

import numpy as np


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

    def get_n_durations(self) -> int:
        return len(self.__durations)

    def get_duration_median(self) -> float:
        if not self.__durations:
            return 0.0
        if self.__duration_median is None:
            self.__duration_median = float(np.median(self.__durations))
        return self.__duration_median

    def get_variations(self) -> List[float]:
        if self.__variations is None:
            median = self.get_duration_median()
            self.__variations = [abs(item - median) for item in self.__durations]
        return self.__variations


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


ModelInfo = namedtuple('ModelInfo', ['framework',
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
        model_info = ModelInfo(item.model_framework,
                               item.model_name,
                               item.model_precision,
                               opt_model_attr)
        if model_info not in data:
            data[model_info] = ModelData()
        data[model_info].append(item)
    return data


def get_common_models(data: List[Dict[ModelInfo, ModelData]]) -> List[ModelInfo]:
    if len(data) == 0:
        return []
    common_keys = data[0].keys()
    for csv_data in data:
        common_keys = common_keys & csv_data.keys()
    return common_keys


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


def filter_by_model_name(data: List[Dict[ModelInfo, ModelData]],
                         model_name: str) -> List[Dict[ModelInfo, ModelData]]:
    new_data = []
    for csv_data in data:
        new_dict = {}
        for model_info in csv_data:
            if model_info.name != model_name:
                continue
            new_dict[model_info] = csv_data[model_info]
        if new_dict:
            new_data.append(new_dict)
    return new_data


def filter_common_models(data: List[Dict[ModelInfo, ModelData]]) -> List[Dict[ModelInfo, ModelData]]:
    common_models: List[ModelInfo] = get_common_models(data)
    return filter_by_models(data, common_models)


def sort_table(table: List[List], get_row_key_func) -> List[List]:
    sorting_table: List[Tuple[int, float]] = []
    for row_idx, row in enumerate(table):
        sorting_table.append((row_idx, get_row_key_func(row)))
    sorted_table = sorted(sorting_table, key=lambda e: e[1], reverse=True)
    result_table = []
    for row in sorted_table:
        result_table.append(table[row[0]])
    return result_table


def compare_compile_time(data: List[Dict[ModelInfo, ModelData]]):
    if len(data) == 0:
        return [], []

    def create_header(n_csv_files: int):
        column_names = ['framework',
                        'name',
                        'precision',
                        'optional model attribute']
        for i in range(n_csv_files):
            column_names.append(f'compile time #{i + 1} (secs)')
        for i in range(1, n_csv_files):
            column_names.append(f'compile time #{i + 1} - #1 (secs)')
        for i in range(1, n_csv_files):
            column_names.append(f'compile time #{i + 1}/#1')
        return column_names

    n_cvs_files = len(data)
    header = create_header(n_cvs_files)
    table = []
    models = [model_info for model_info in data[0]]
    for model_info in models:
        row = [model_info.framework,
               model_info.name,
               model_info.precision,
               model_info.optional_attribute]
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

    def get_max_delta(row: List) -> float:
        delta_values = (abs(item) for item in row[-2 * (n_cvs_files - 1): -1 * (n_cvs_files - 1)])
        return max(delta_values)
    return header, sort_table(table, get_max_delta)

'''
# FIXME
def find_longest_common_subsequence(list1: List[UnitInfo],
                                    list2: List[UnitInfo]) -> List[List[int]]:
    TransformationInfo = namedtuple('TransformationInfo',
                                    ['type',
                                     'transformation_name',
                                     'manager_name'])

    def get_ts_info(model_info_list: List[UnitInfo]) -> List[TransformationInfo]:
        result = []
        for item in model_info_list:
            result.append(TransformationInfo(item.type,
                                             item.transformation_name,
                                             item.manager_name))
        return result

    list1 = get_ts_info(list1)
    list2 = get_ts_info(list2)

    n = len(list1)
    m = len(list2)

    # Create a (n+1) x (m+1) DP table initialized to 0
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    # Fill the DP table
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if list1[i - 1] == list2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    # Backtrack to find the LCS sequence
    list1_indexes = []
    list2_indexes = []
    i, j = n, m
    while i > 0 and j > 0:
        if list1[i - 1] == list2[j - 1]:
            list1_indexes.append(i - 1)
            list2_indexes.append(j - 1)
            i -= 1
            j -= 1
        elif dp[i - 1][j] > dp[i][j - 1]:
            i -= 1
        else:
            j -= 1

    # Since we are backtracking, the LCS is constructed backwards, reverse it
    list1_indexes.reverse()
    list2_indexes.reverse()

    assert len(list1_indexes) == len(list2_indexes)

    return [list1_indexes, list2_indexes]


def create_durations_header(n_csv_files: int):
    column_names = ['is not a common',
                    'type',
                    'transformation_name',
                    'manager_name']
    for i in range(n_csv_files):
        column_names.append(f'duration #{i + 1} (milliseconds)')
        if i > 0:
            column_names.append(f'duration #{i + 1} to #1 ratio')
    return column_names


def compare_model_durations(data: List[Dict[ModelInfo, ModelData]],
                            model_info: ModelInfo):
    n_cvs_files = len(data)
    assert n_cvs_files == 2
    model_data_objs = [obj[model_info] for obj in data]
    # get longest common subsequence of transformations
    all_item_info1 = model_data_objs[0].get_all_item_info()
    all_item_info2 = model_data_objs[1].get_all_item_info()
    all_item_info_list = [all_item_info1, all_item_info2]
    subsequence_indexes = find_longest_common_subsequence(all_item_info1,
                                                          all_item_info2)
    assert len(subsequence_indexes) == n_cvs_files
    #
    header = create_durations_header(n_cvs_files)
    table = []

    # add common subsequence to the table
    for common_subsequence_idx in range(len(subsequence_indexes[0])):
        item_info_ref = all_item_info_list[0][subsequence_indexes[0][common_subsequence_idx]]
        row = ['',
               item_info_ref.type,
               item_info_ref.transformation_name,
               item_info_ref.manager_name]
        first_duration = None
        for csv_idx in range(n_cvs_files):
            subsequence = subsequence_indexes[csv_idx]
            item_idx = subsequence[common_subsequence_idx]
            duration = model_data_objs[csv_idx].get_duration(item_idx) / 1_000_000
            row.append(duration)
            if csv_idx == 0:
                first_duration = duration
            else:
                row.append((duration - first_duration)/first_duration)
        table.append(row)

    # add non-common items to the table
    for csv_idx in range(n_cvs_files):
        subsequence = set(subsequence_indexes[csv_idx])
        item_info_list = all_item_info_list[csv_idx]
        for item_idx, item_info in enumerate(item_info_list):
            if item_idx in subsequence:
                # we already add this item as it is common
                continue
            row = ['true',
                   item_info.type,
                   item_info.transformation_name,
                   item_info.manager_name]
            for column_idx in range(n_cvs_files):
                duration = 0.0
                if column_idx == csv_idx:
                    duration = model_data_objs[csv_idx].get_duration(item_idx) / 1_000_000
                row.append(duration)
                # add ratio
                if column_idx > 0:
                    row.append(0.0)
            table.append(row)

    def get_max_duration(row: List) -> float:
        durations = [row[4]] + row[5::2]
        return max(durations)
    return header, sort_table(table, get_max_duration)
'''


def get_longest_unit(data: List[Dict[ModelInfo, ModelData]],
                     unit_type: str):
    @dataclass
    class Total:
        duration: float
        count: int

    def aggregate_unit_data(data: List[Dict[ModelInfo, ModelData]],
                            unit_type: str) -> Dict[str, Total]:
        result: Dict[str, Total] = {} # ts name: Total
        for csv_item in data:
            for model_info, model_data in csv_item.items():
                for item in model_data.get_items_by_type(unit_type):
                    if item.name not in result:
                        result[item.name] = Total(0.0, 0)
                    result[item.name].duration += item.get_duration_median() / 1_000_000
                    result[item.name].count += 1
        return result

    header = ['name', 'total duration (ms)', 'count of executions']
    table = []
    for name, total in aggregate_unit_data(data, unit_type).items():
        table.append([name, total.duration, total.count])
    def get_duration(row: List) -> float:
        return row[1]
    return header, sort_table(table, get_duration)


def compare_sum_units(data: List[Dict[ModelInfo, ModelData]],
                      unit_type: str):
    @dataclass
    class Total:
        duration: float
        count: int

    def aggregate_unit_data(data: Dict[ModelInfo, ModelData],
                            unit_type: str) -> Dict[str, Total]:
        result: Dict[str, Total] = {} # ts name: Total
        for model_info, model_data in data.items():
            for item in model_data.get_items_by_type(unit_type):
                if item.name not in result:
                    result[item.name] = Total(0.0, 0)
                result[item.name].duration += item.get_duration_median() / 1_000_000
                result[item.name].count += 1
        return result

    def get_duration(aggregated_data_item: Dict[str, Total], name: str) -> float:
        duration = 0.0
        if name in aggregated_data_item:
            duration = aggregated_data_item[name].duration
        return duration

    def get_count(aggregated_data_item: Dict[str, Total], name: str) -> int:
        count = 0
        if name in aggregated_data_item:
            count = aggregated_data_item[name].count
        return count

    def create_header(n_csv_files: int):
        column_names = ['name']
        for i in range(n_csv_files):
            column_names.append(f'duration #{i + 1} (ms)')
        for i in range(1, n_csv_files):
            column_names.append(f'duration #{i + 1} - #1 (secs)')
        for i in range(1, n_csv_files):
            column_names.append(f'duration #{i + 1}/#1')
        for i in range(n_csv_files):
            column_names.append(f'count #{i + 1}')
        for i in range(1, n_csv_files):
            column_names.append(f'count #{i + 1} - #1 (secs)')
        return column_names

    n_csv_files = len(data)

    table = []

    aggregated_data = [aggregate_unit_data(csv_data, unit_type) for csv_data in data]
    all_transformations = set(ts_name for aggregated_data_item in aggregated_data
                              for ts_name in aggregated_data_item)

    for name in all_transformations:
        row = [name]
        durations = []
        counters = []
        for csv_idx in range(n_csv_files):
            durations.append(get_duration(aggregated_data[csv_idx], name))
            counters.append(get_count(aggregated_data[csv_idx], name))

        for csv_idx in range(n_csv_files):
            row.append(durations[csv_idx])
        for csv_idx in range(1, n_csv_files):
            row.append(durations[csv_idx] - durations[0])
        for csv_idx in range(1, n_csv_files):
            ratio = 0.0
            if durations[csv_idx] != 0.0:
                ratio = durations[0]/durations[csv_idx]
            row.append(ratio)

        for csv_idx in range(n_csv_files):
            row.append(counters[csv_idx])
        for csv_idx in range(1, n_csv_files):
            row.append(counters[csv_idx] - counters[0])
        table.append(row)


    header = create_header(n_csv_files)

    def get_max_delta(row: List) -> float:
        delta_values = (abs(item) for item in row[-4 * n_csv_files + 3: -3 * n_csv_files + 2])
        return max(delta_values)
    return header, sort_table(table, get_max_delta)


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


def save_csv(header, table, path: str) -> None:
    with open(path, 'w', newline='') as f_out:
        csv_writer = csv.writer(f_out, delimiter=';')
        csv_writer.writerow(header)
        csv_writer.writerows(table)


def make_model_file_name(model_info: ModelInfo, prefix: str) -> str:
    name = [prefix,
            model_info.framework,
            model_info.name,
            model_info.precision]
    if model_info.optional_attribute:
        name.append(model_info.optional_attribute)
    return '_'.join(name) + '.csv'


# TODO: make it abstract
class DataProcessor:
    def __init__(self):
        pass
    def run(self, csv_data: List[Dict[ModelInfo, ModelData]]) -> None:
        pass


class CompareCompileTime(DataProcessor):
    def __init__(self, output: str):
        super().__init__()
        self.output = output

    def run(self, csv_data: List[Dict[ModelInfo, ModelData]]) -> None:
        print('comparing compile time ...')
        # CSV files can store different models info
        csv_data_common_models = filter_common_models(csv_data)
        if not csv_data_common_models:
            print('no common models to compare compilation time ...')
            return
        header, table = compare_compile_time(csv_data_common_models)
        save_csv(header, table, self.output)


class GenerateLongestUnitsOverall(DataProcessor):
    def __init__(self, output: str, unit_type: str):
        super().__init__()
        self.output = output
        self.unit_type = unit_type

    def run(self, csv_data: List[Dict[ModelInfo, ModelData]]) -> None:
        print(f'aggregating longest {self.unit_type} overall data ...')
        header, table = get_longest_unit(csv_data, self.unit_type)
        save_csv(header, table, self.output)


class GenerateLongestUnitsPerModel(DataProcessor):
    def __init__(self, output: str, unit_type: str):
        super().__init__()
        self.output = output
        self.unit_type = unit_type

    def run(self, csv_data: List[Dict[ModelInfo, ModelData]]) -> None:
        print(f'aggregating longest {self.unit_type} per model data ...')
        for model_info in get_all_models(csv_data):
            header, table = get_longest_unit(filter_by_models(csv_data, [model_info]), self.unit_type)
            file_name = make_model_file_name(model_info, self.output)
            save_csv(header, table, file_name)


class CompareSumUnitsOverall(DataProcessor):
    def __init__(self, output: str, unit_type: str):
        super().__init__()
        self.output = output
        self.unit_type = unit_type

    def run(self, csv_data: List[Dict[ModelInfo, ModelData]]) -> None:
        print(f'compare sum {self.unit_type} overall data ...')
        csv_data_common_models = filter_common_models(csv_data)
        header, table = compare_sum_units(csv_data_common_models, self.unit_type)
        save_csv(header, table, self.output)


class CompareSumUnitsPerModel(DataProcessor):
    def __init__(self, output: str, unit_type: str):
        super().__init__()
        self.output = output
        self.unit_type = unit_type

    def run(self, csv_data: List[Dict[ModelInfo, ModelData]]) -> None:
        print(f'compare sum {self.unit_type} per model data ...')
        csv_data_common_models = filter_common_models(csv_data)
        for model_info in get_all_models(csv_data_common_models):
            header, table = compare_sum_units(filter_by_models(csv_data, [model_info]), self.unit_type)
            file_name = make_model_file_name(model_info, self.output)
            save_csv(header, table, file_name)


'''
# FIXME
def generate_model_durations_csv(csv_data):
    for model_info in get_all_models(csv_data):
        header, table = compare_model_durations(csv_data, model_info)
        file_name = file_name = make_model_file_name(model_info, 'duration_comparison')
        save_csv(header, table, file_name)
'''


@dataclass
class Config:
    compare_compile_time = None
    transformations_overall = None
    manager_overall = None
    transformations_per_model = None
    managers_per_model = None
    compare_transformations_overall = None
    compare_managers_overall = None
    compare_transformations_per_model = None
    compare_managers_per_model = None
    model_name = None
    inputs: List[str] = field(default_factory=list)


def parse_args() -> Config:
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--input', type=str, help='input CSV files separated by comma')
    args_parser.add_argument('--compare_compile_time', nargs='?', type=str, default=None,
                             const='compile_time_comparison.csv',
                             help='compare compile time between input files; for common models between inputs')
    args_parser.add_argument('--transformations_overall', nargs='?', type=str, default=None,
                             const='longest_transformations_overall.csv',
                             help='aggregate transformations overall models and input CSVs data')
    args_parser.add_argument('--manager_overall', type=str, help='aggregate managers overall data',
                             const='longest_managers_overall.csv', nargs='?', default=None)
    args_parser.add_argument('--transformations_per_model', nargs='?', type=str, default=None,
                             const='longest_transformations',
                             help='aggregate transformations per model data; output in different CSV files; --output is file prefix')
    args_parser.add_argument('--managers_per_model', nargs='?', type=str, default=None,
                             const='longest_managers',
                             help='aggregate managers per model data; output in different CSV files; --output is file prefix')
    args_parser.add_argument('--compare_transformations_overall', nargs='?', type=str, default=None,
                             const='longest_transformations_overall.csv',
                             help='aggregate transformations overall models data and compare between input CSV files')
    args_parser.add_argument('--compare_managers_overall', nargs='?', type=str, default=None,
                             const='longest_managers_overall.csv',
                             help='aggregate managers overall models data and compare between input CSV files')
    args_parser.add_argument('--compare_transformations_per_model', nargs='?', type=str, default=None,
                             const='compare_sum_transformations',
                             help='aggregate transformations per model data and compare between input CSV files')
    args_parser.add_argument('--compare_managers_per_model', nargs='?', type=str, default=None,
                             const='compare_sum_managers',
                             help='aggregate managers per model data and compare between input CSV files')
    args_parser.add_argument('--model_name', nargs='?', type=str, default=None,
                             const='model_name',
                             help='filter input data by specified model name')
    args = args_parser.parse_args()

    config = Config()

    if not args.input:
        print('specify input CSV files separated by comma')
        sys.exit(1)
    config.inputs = args.input.split(',')

    if any(not s for s in config.inputs):
        print('input file cannot be empty')
        sys.exit(1)

    config.compare_compile_time = args.compare_compile_time
    config.transformations_overall = args.transformations_overall
    config.managers_overall = args.manager_overall
    config.transformations_per_model = args.transformations_per_model
    config.managers_per_model = args.managers_per_model
    config.compare_transformations_overall = args.compare_transformations_overall
    config.compare_managers_overall = args.compare_managers_overall
    config.compare_transformations_per_model = args.compare_transformations_per_model
    config.compare_managers_per_model = args.compare_managers_per_model
    config.model_name = args.model_name

    return config


def main(config: Config) -> None:
    data_processors = []
    if config.compare_compile_time:
        data_processors.append(CompareCompileTime(config.compare_compile_time))
    if config.transformations_overall:
        data_processors.append(GenerateLongestUnitsOverall(config.transformations_overall, unit_type='transformation'))
    if config.manager_overall:
        data_processors.append(GenerateLongestUnitsOverall(config.manager_overall, unit_type='manager'))
    if config.transformations_per_model:
        data_processors.append(GenerateLongestUnitsPerModel(config.transformations_per_model, unit_type='transformation'))
    if config.managers_per_model:
        data_processors.append(GenerateLongestUnitsPerModel(config.managers_per_model, unit_type='manager'))
    if config.compare_transformations_overall:
        data_processors.append(CompareSumUnitsOverall(config.compare_transformations_overall, unit_type='transformation'))
    if config.compare_managers_overall:
        data_processors.append(CompareSumUnitsOverall(config.compare_managers_overall, unit_type='manager'))
    if config.compare_transformations_per_model:
        data_processors.append(CompareSumUnitsPerModel(config.compare_transformations_per_model, unit_type='transformation'))
    if config.compare_managers_per_model:
        data_processors.append(CompareSumUnitsPerModel(config.compare_managers_per_model, unit_type='manager'))

    '''
    if len(csv_data) == 2:
        print('comparing transformation and manager duration ...')
        generate_model_durations_csv(csv_data)
    '''

    if not data_processors:
        print('nothing to do ...')
        return

    csv_data = get_csv_data(config.inputs)
    if config.model_name:
        csv_data = filter_by_model_name(csv_data, config.model_name)
    for proc in data_processors:
        proc.run(csv_data)


if __name__ == '__main__':
    main(parse_args())
