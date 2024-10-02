from abc import ABC, abstractmethod
import argparse
from collections import namedtuple
import csv
from dataclasses import dataclass, field
import os
import sys
from tabulate import tabulate
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


def sort_table(table: List[Dict], get_row_key_func) -> List[Dict]:
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
        for csv_idx in range(n_csv_files):
            column_names.append(f'compile time #{csv_idx + 1} (secs)')
        for csv_idx in range(1, n_csv_files):
            column_names.append(f'compile time #{csv_idx + 1} - #1 (secs)')
        for csv_idx in range(1, n_csv_files):
            column_names.append(f'compile time #{csv_idx + 1}/#1')
        return column_names

    def get_delta_header_names(n_csv_files: int) -> List[str]:
        column_names = []
        for csv_idx in range(1, n_csv_files):
            column_names.append(f'compile time #{csv_idx + 1} - #1 (secs)')
        return column_names


    n_cvs_files = len(data)
    header = create_header(n_cvs_files)
    table = []
    models = [model_info for model_info in data[0]]
    for model_info in models:
        row = {'framework': model_info.framework,
               'name': model_info.name,
               'precision': model_info.precision,
               'optional model attribute': model_info.optional_attribute}
        compile_times = []
        for csv_idx in range(n_cvs_files):
            model_data = data[csv_idx][model_info]
            compile_time = model_data.get_compile_time() / 1_000_000_000
            compile_times.append(compile_time)
        for csv_idx in range(n_cvs_files):
            row[f'compile time #{csv_idx + 1} (secs)'] = compile_times[csv_idx]
        for csv_idx in range(1, n_cvs_files):
            delta = compile_times[csv_idx] - compile_times[0]
            row[f'compile time #{csv_idx + 1} - #1 (secs)'] = delta
        for csv_idx in range(1, n_cvs_files):
            ratio = 'N/A'
            if compile_times[0] != 0.0:
                ratio = compile_times[csv_idx] / compile_times[0]
            row[f'compile time #{csv_idx + 1}/#1'] = ratio
        table.append(row)


    delta_header_names = get_delta_header_names(n_cvs_files)
    def get_max_delta(row: Dict) -> float:
        return max((abs(row[key]) for key in delta_header_names))
    return header, sort_table(table, get_max_delta)


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
        row = {'name': name,
               'total duration (ms)': total.duration,
               'count of executions': total.count}
        table.append(row)
    def get_duration(row: Dict) -> float:
        return row['total duration (ms)']
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
            column_names.append(f'duration #{i + 1} - #1 (ms)')
        for i in range(1, n_csv_files):
            column_names.append(f'duration #{i + 1}/#1')
        for i in range(n_csv_files):
            column_names.append(f'count #{i + 1}')
        for i in range(1, n_csv_files):
            column_names.append(f'count #{i + 1} - #1')
        return column_names

    def get_delta_header_names(n_csv_files: int) -> List[str]:
        column_names = []
        for csv_idx in range(1, n_csv_files):
            column_names.append(f'duration #{csv_idx + 1} - #1 (ms)')
        return column_names

    n_csv_files = len(data)

    table = []

    aggregated_data = [aggregate_unit_data(csv_data, unit_type) for csv_data in data]
    all_transformations = set(ts_name for aggregated_data_item in aggregated_data
                              for ts_name in aggregated_data_item)

    for name in all_transformations:
        row = {'name' : name}
        durations = []
        counters = []
        for csv_idx in range(n_csv_files):
            durations.append(get_duration(aggregated_data[csv_idx], name))
            counters.append(get_count(aggregated_data[csv_idx], name))

        for csv_idx in range(n_csv_files):
            row[f'duration #{csv_idx + 1} (ms)'] = durations[csv_idx]
        for csv_idx in range(1, n_csv_files):
            delta = durations[csv_idx] - durations[0]
            row[f'duration #{csv_idx + 1} - #1 (ms)'] = delta
        for csv_idx in range(1, n_csv_files):
            ratio = 'N/A'
            if durations[0] != 0.0:
                ratio = durations[csv_idx]/durations[0]
            row[f'duration #{csv_idx + 1}/#1'] = ratio
        for csv_idx in range(n_csv_files):
            row[f'count #{csv_idx + 1}'] = counters[csv_idx]
        for csv_idx in range(1, n_csv_files):
            delta = counters[csv_idx] - counters[0]
            row[f'count #{csv_idx + 1} - #1'] = delta
        table.append(row)

    header = create_header(n_csv_files)

    delta_header_names = get_delta_header_names(n_csv_files)
    def get_max_delta(row: Dict) -> float:
        return max((abs(row[key]) for key in delta_header_names))
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


def make_model_file_name(model_info: ModelInfo, prefix: str) -> str:
    name = [prefix,
            model_info.framework,
            model_info.name,
            model_info.precision]
    if model_info.optional_attribute:
        name.append(model_info.optional_attribute)
    return '_'.join(name) + '.csv'


def make_model_console_description(model_info: ModelInfo) -> str:
    name = [model_info.framework,
            model_info.name,
            model_info.precision]
    if model_info.optional_attribute:
        name.append(model_info.optional_attribute)
    return ' '.join(name)


class Output(ABC):
    def __init__(self, header: List[str]):
        self.header = header

    @abstractmethod
    def __enter__(self):
        return self

    @abstractmethod
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    @abstractmethod
    def write_header(self):
        pass

    @abstractmethod
    def write(self, row: List[Dict[str, str]]):
        pass


class CSVOutput(Output):
    def __init__(self, path: str, header: List[str], limit_output):
        super().__init__(header)
        self.path = path
        self.limit_output = limit_output
        self.file = None

    def write_header(self):
        assert self.header is not None
        csv_writer = csv.DictWriter(self.file, fieldnames=self.header, delimiter=';')
        csv_writer.writeheader()

    def write(self, rows: List[Dict[str, str]]):
        assert self.file is not None
        assert self.header is not None
        csv_writer = csv.DictWriter(self.file, fieldnames=self.header, delimiter=';')
        if self.limit_output:
            rows = rows[:self.limit_output]
        for row in rows:
            csv_writer.writerow(row)

    def __enter__(self):
        self.file = open(self.path, 'w', newline='')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file:
            self.file.close()


class ConsoleTableOutput(Output):
    def __init__(self, header: List[str], description: str, limit_output):
        super().__init__(header)
        self.description = description
        self.limit_output = limit_output
        self.file = None

    def write_header(self):
        pass

    def write(self, rows: List[Dict[str, str]]):
        assert self.header is not None
        if self.limit_output:
            rows = rows[:self.limit_output]
        print(self.description)
        ordered_rows_str = [{key: str(row[key]) for key in self.header} for row in rows]
        table = tabulate(ordered_rows_str, headers="keys")
        print(table)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class SingleOutputFactory(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def create(self, header: List[str]):
        pass


class CSVSingleFileOutputFactory(SingleOutputFactory):
    def __init__(self, path: str, limit_output):
        super().__init__()
        self.path = path
        self.limit_output = limit_output

    def create(self, header: List[str]):
        return CSVOutput(self.path, header, self.limit_output)


class ConsoleTableSingleFileOutputFactory(SingleOutputFactory):
    def __init__(self, description: str, limit_output):
        super().__init__()
        self.description = description
        self.limit_output = limit_output

    def create(self, header: List[str]):
        return ConsoleTableOutput(header, self.description, self.limit_output)


class MultiOutputFactory(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def create(self, header: List[str], model_info: ModelInfo):
        pass


class CSVMultiFileOutputFactory(MultiOutputFactory):
    def __init__(self, prefix: str, limit_output):
        super().__init__()
        self.prefix = prefix
        self.limit_output = limit_output

    def create(self, header: List[str], model_info: ModelInfo):
        return CSVOutput(make_model_file_name(model_info, self.prefix), header, self.limit_output)


class ConsoleTableMultiOutputFactory(MultiOutputFactory):
    def __init__(self, description: str, limit_output):
        super().__init__()
        self.description = description
        self.limit_output = limit_output

    def create(self, header: List[str], model_info: ModelInfo):
        return ConsoleTableOutput(header, make_model_console_description(model_info), self.limit_output)


class DataProcessor(ABC):
    def __init__(self, output_factory):
        self.output_factory = output_factory

    @abstractmethod
    def run(self, csv_data: List[Dict[ModelInfo, ModelData]]) -> None:
        pass


class CompareCompileTime(DataProcessor):
    def __init__(self, output_factory: SingleOutputFactory):
        super().__init__(output_factory)

    def run(self, csv_data: List[Dict[ModelInfo, ModelData]]) -> None:
        print('comparing compile time ...')
        # CSV files can store different models info
        csv_data_common_models = filter_common_models(csv_data)
        if not csv_data_common_models:
            print('no common models to compare compilation time ...')
            return
        header, table = compare_compile_time(csv_data_common_models)
        with self.output_factory.create(header) as output:
            output.write_header()
            output.write(table)


class GenerateLongestUnitsOverall(DataProcessor):
    def __init__(self, output_factory: SingleOutputFactory, unit_type: str):
        super().__init__(output_factory)
        self.unit_type = unit_type

    def run(self, csv_data: List[Dict[ModelInfo, ModelData]]) -> None:
        print(f'aggregating longest {self.unit_type} overall data ...')
        header, table = get_longest_unit(csv_data, self.unit_type)
        with self.output_factory.create(header) as output:
            output.write_header()
            output.write(table)


class GenerateLongestUnitsPerModel(DataProcessor):
    def __init__(self, output_factory: MultiOutputFactory, unit_type: str):
        super().__init__(output_factory)
        self.unit_type = unit_type

    def run(self, csv_data: List[Dict[ModelInfo, ModelData]]) -> None:
        print(f'aggregating longest {self.unit_type} per model data ...')
        for model_info in get_all_models(csv_data):
            header, table = get_longest_unit(filter_by_models(csv_data, [model_info]), self.unit_type)
            with self.output_factory.create(header, model_info) as output:
                output.write_header()
                output.write(table)


class CompareSumUnitsOverall(DataProcessor):
    def __init__(self, output_factory: SingleOutputFactory, unit_type: str):
        super().__init__(output_factory)
        self.unit_type = unit_type

    def run(self, csv_data: List[Dict[ModelInfo, ModelData]]) -> None:
        print(f'compare sum {self.unit_type} overall data ...')
        csv_data_common_models = filter_common_models(csv_data)
        header, table = compare_sum_units(csv_data_common_models, self.unit_type)
        with self.output_factory.create(header) as output:
            output.write_header()
            output.write(table)


class CompareSumUnitsPerModel(DataProcessor):
    def __init__(self, output_factory: MultiOutputFactory, unit_type: str):
        super().__init__(output_factory)
        self.unit_type = unit_type

    def run(self, csv_data: List[Dict[ModelInfo, ModelData]]) -> None:
        print(f'compare sum {self.unit_type} per model data ...')
        csv_data_common_models = filter_common_models(csv_data)
        for model_info in get_all_models(csv_data_common_models):
            header, table = compare_sum_units(filter_by_models(csv_data, [model_info]), self.unit_type)
            with self.output_factory.create(header, model_info) as output:
                output.write_header()
                output.write(table)


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
    output_type = 'csv'
    model_name = None
    limit_output = None
    inputs: List[str] = field(default_factory=list)


def parse_args() -> Config:
    script_bin = sys.argv[0]
    args_parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    args_parser.add_argument('--input', type=str,
                             help=f'''input CSV files separated by comma
For example, if you have 2 input CSV files /dir1/file1.csv and /dir2/file2.csv, generated by dev_trigger job, you can specify them
{script_bin} --input /dir1/file1.csv,/dir2/file2.csv
''')
    args_parser.add_argument('--compare_compile_time', nargs='?', type=str, default=None,
                             const='compile_time_comparison.csv',
                             help='compare compile time between input files; for common models between inputs')
    args_parser.add_argument('--transformations_overall', nargs='?', type=str, default=None,
                             const='transformations_overall.csv',
                             metavar='path to output file',
                             help=f'''aggregate transformations overall models and input CSVs data
For example, if you have 2 input CSV files /dir1/file1.csv and /dir2/file2.csv, generated by dev_trigger job, you can specify them
{script_bin} --inputs /dir1/file1.csv,/dir2/file2.csv --transformations_overall
This option will generate CSV output file transformations_overall.csv with columns
name - name of transformation	
total duration (ms) - total duration (in milliseconds) of all execution of this transformations in both dev_trigger jobs for all models
count of executions - total count of all execution of this transformations in both dev_trigger jobs for all models
You can specify output path, for example
{script_bin} --inputs /dir1/file1.csv,/dir2/file2.csv --transformations_overall /output_dir/output_file.csv
''')
    args_parser.add_argument('--manager_overall', type=str,
                             metavar='path to output file',
                             help=f'''aggregate managers overall data
For example, if you have 2 input CSV files /dir1/file1.csv and /dir2/file2.csv, generated by dev_trigger job, you can specify them
{script_bin} --inputs /dir1/file1.csv,/dir2/file2.csv --manager_overall
This option will generate CSV output file manager_overall.csv with columns
name - name of manager	
total duration (ms) - total duration (in milliseconds) of all execution of this manager in both dev_trigger jobs for all models
count of executions - total count of all execution of this manager in both dev_trigger jobs for all models
You can specify output path, for example
{script_bin} --inputs /dir1/file1.csv,/dir2/file2.csv --manager_overall /output_dir/output_file.csv
''',
                             const='manager_overall.csv', nargs='?', default=None)
    args_parser.add_argument('--transformations_per_model', nargs='?', type=str, default=None,
                             const='transformations_per_model',
                             metavar='output files name prefix',
                             help=f'''aggregate transformations per model data; output in different CSV files
For example, if you have 2 input CSV files /dir1/file1.csv and /dir2/file2.csv, generated by dev_trigger job, you can specify them
{script_bin} --inputs /dir1/file1.csv,/dir2/file2.csv --transformations_per_model
This option will generate CSV multiple output files with names transformations_per_model_<framework>_<model>_<precision>[_<additional_model_attribute>].csv with columns
name - name of transformation
total duration (ms) - total duration (in milliseconds) of all execution of this transformation in both dev_trigger jobs for one particular model
count of executions - total count of all execution of this transformation in both dev_trigger jobs for one particular model
You can specify output names prefix, for example
{script_bin} --inputs /dir1/file1.csv,/dir2/file2.csv --transformations_per_model prefix_to_name
and there will be files prefix_to_name_<framework>_<model>_<precision>[_<additional_model_attribute>].csv
''')
    args_parser.add_argument('--managers_per_model', nargs='?', type=str, default=None,
                             const='managers_per_model',
                             metavar='output files name prefix',
                             help=f'''aggregate managers per model data; output in different CSV files
For example, if you have 2 input CSV files /dir1/file1.csv and /dir2/file2.csv, generated by dev_trigger job, you can specify them
{script_bin} --inputs /dir1/file1.csv,/dir2/file2.csv --managers_per_model
This option will generate CSV multiple output files with names managers_per_model_<framework>_<model>_<precision>[_<additional_model_attribute>].csv with columns
name - name of manager
total duration (ms) - total duration (in milliseconds) of all execution of this manager in both dev_trigger jobs for one particular model
count of executions - total count of all execution of this manager in both dev_trigger jobs for one particular model
You can specify output names prefix, for example
{script_bin} --inputs /dir1/file1.csv,/dir2/file2.csv --managers_per_model prefix_to_name
and there will be files prefix_to_name_<framework>_<model>_<precision>[_<additional_model_attribute>].csv
''')
    args_parser.add_argument('--compare_transformations_overall', nargs='?', type=str, default=None,
                             const='comparison_transformations_overall.csv',
                             metavar='path to output file',
                             help=f'''aggregate transformations overall models data and compare between input CSV files
For example, if you have 2 input CSV files /dir1/file1.csv and /dir2/file2.csv, generated by dev_trigger job, you can specify them
{script_bin} --inputs /dir1/file1.csv,/dir2/file2.csv --compare_transformations_overall
This option will generate CSV output file comparison_transformations_overall.csv with columns
name - name of transformation
duration #1 (ms) - duration (in milliseconds) of all execution of this transformation in first dev_trigger job (/dir1/file1.csv) for all models
duration #2 (ms) - duration (in milliseconds) of all execution of this transformation in second dev_trigger job (/dir2/file2.csv) for all models
duration #2 - #1 (ms) - delta between duration#2 and duration#1  (ms)
duration #2/#1 - ratio duration#2 / duration #1 
count #1 - count of all execution of this transformation in first dev_trigger job (/dir1/file1.csv) for all models
count #2 - count of all execution of this transformation in second dev_trigger job (/dir2/file2.csv) for all models
count #2 - #1 (secs) - delta between count#2 and count#1
You can specify output names prefix, for example
{script_bin} --inputs /dir1/file1.csv,/dir2/file2.csv --compare_transformations_overall /dir3/output3
''')
    args_parser.add_argument('--compare_managers_overall', nargs='?', type=str, default=None,
                             const='compare_managers_overall.csv',
                             metavar='path to output file',
                             help=f'''aggregate managers overall models data and compare between input CSV files
For example, if you have 2 input CSV files /dir1/file1.csv and /dir2/file2.csv, generated by dev_trigger job, you can specify them
{script_bin} --inputs /dir1/file1.csv,/dir2/file2.csv --compare_managers_overall
This option will generate CSV output file comparison_transformations_overall.csv with columns
name - name of manager
duration #1 (ms) - duration (in milliseconds) of all execution of this manager in first dev_trigger job (/dir1/file1.csv) for all models
duration #2 (ms) - duration (in milliseconds) of all execution of this manager in second dev_trigger job (/dir2/file2.csv) for all models
duration #2 - #1 (ms) - delta between duration#2 and duration#1  (ms)
duration #2/#1 - ratio duration#2 / duration #1 
count #1 - count of all execution of this manager in first dev_trigger job (/dir1/file1.csv) for all models
count #2 - count of all execution of this manager in second dev_trigger job (/dir2/file2.csv) for all models
count #2 - #1 (secs) - delta between count#2 and count#1
You can specify output names prefix, for example
{script_bin} --inputs /dir1/file1.csv,/dir2/file2.csv --compare_managers_overall /dir3/output3
''')
    args_parser.add_argument('--compare_transformations_per_model', nargs='?', type=str, default=None,
                             const='compare_transformations',
                             metavar='output files name prefix',
                             help=f'''aggregate transformations per model data and compare between input CSV files
For example, if you have 2 input CSV files /dir1/file1.csv and /dir2/file2.csv, generated by dev_trigger job, you can specify them
{script_bin} --inputs /dir1/file1.csv,/dir2/file2.csv --compare_transformations_per_model
This option will generate CSV multiple output files with names compare_transformations_<framework>_<model>_<precision>[_<additional_model_attribute>].csv with columns
name - name of transformation
duration #1 (ms) - duration (in milliseconds) of all execution of this transformation in first dev_trigger job (/dir1/file1.csv) for one particular model
duration #2 (ms) - duration (in milliseconds) of all execution of this transformation in second dev_trigger job (/dir2/file2.csv) for one particular model
duration #2 - #1 (ms) - delta between duration#2 and duration#1  (ms)
duration #2/#1 - ratio duration#2 / duration #1 
count #1 - count of all execution of this transformation in first dev_trigger job (/dir1/file1.csv) for one particular model
count #2 - count of all execution of this transformation in second dev_trigger job (/dir2/file2.csv) for one particular model
count #2 - #1 (secs) - delta between count#2 and count#1
You can specify output names prefix, for example
{script_bin} --inputs /dir1/file1.csv,/dir2/file2.csv --compare_transformations_per_model prefix_to_name
and there will be files prefix_to_name_<framework>_<model>_<precision>[_<additional_model_attribute>].csv
''')
    args_parser.add_argument('--compare_managers_per_model', nargs='?', type=str, default=None,
                             const='compare_managers',
                             metavar='output files name prefix',
                             help=f'''aggregate managers per model data and compare between input CSV files
For example, if you have 2 input CSV files /dir1/file1.csv and /dir2/file2.csv, generated by dev_trigger job, you can specify them
{script_bin} --inputs /dir1/file1.csv,/dir2/file2.csv --compare_managers_per_model
This option will generate CSV multiple output files with names compare_managers_<framework>_<model>_<precision>[_<additional_model_attribute>].csv with columns
name - name of manager
duration #1 (ms) - duration (in milliseconds) of all execution of this manager in first dev_trigger job (/dir1/file1.csv) for one particular model
duration #2 (ms) - duration (in milliseconds) of all execution of this manager in second dev_trigger job (/dir2/file2.csv) for one particular model
duration #2 - #1 (ms) - delta between duration#2 and duration#1  (ms)
duration #2/#1 - ratio duration#2 / duration #1 
count #1 - count of all execution of this manager in first dev_trigger job (/dir1/file1.csv) for one particular model
count #2 - count of all execution of this manager in second dev_trigger job (/dir2/file2.csv) for one particular model
count #2 - #1 (secs) - delta between count#2 and count#1
You can specify output names prefix, for examples
{script_bin} --inputs /dir1/file1.csv,/dir2/file2.csv --compare_managers_per_model prefix_to_name
and there will be files prefix_to_name_<framework>_<model>_<precision>[_<additional_model_attribute>].csv
''')
    args_parser.add_argument('--model_name', type=str, default=None,
                             help=f"""filter input data by specified model name
If you want to get information only about models with specified model name,
For example to dump only for models with name 'llama-3-8b',
{script_bin} --inputs /dir1/file1.csv,/dir2/file2.csv --compare_managers_per_model --model_name llama-3-8b
""")
    args_parser.add_argument('--output_type', type=str, default='csv',
                             help='csv or console')
    args_parser.add_argument('--limit_output', type=int, default=None,
                             help=f'''
Output maximum number of rows
For example, to output only first 15 rows in table
{script_bin} --inputs /dir1/file1.csv,/dir2/file2.csv --compare_managers_overall --limit_output 15
''')

    args = args_parser.parse_args()

    config = Config()

    if not args.input:
        print('specify input CSV files separated by comma')
        sys.exit(1)
    config.inputs = args.input.split(',')

    if any(not s for s in config.inputs):
        print('input file cannot be empty')
        sys.exit(1)

    if args.output_type not in ('csv', 'console'):
        raise Exception(f'unknown output type {args.output_type}')
    config.output_type = args.output_type

    config.compare_compile_time = args.compare_compile_time
    config.transformations_overall = args.transformations_overall
    config.manager_overall = args.manager_overall
    config.transformations_per_model = args.transformations_per_model
    config.managers_per_model = args.managers_per_model
    config.compare_transformations_overall = args.compare_transformations_overall
    config.compare_managers_overall = args.compare_managers_overall
    config.compare_transformations_per_model = args.compare_transformations_per_model
    config.compare_managers_per_model = args.compare_managers_per_model
    config.model_name = args.model_name
    config.limit_output = args.limit_output

    return config


def create_single_output_factory(output_type: str, path: str, description: str, limit_output):
    if output_type == 'csv':
        return CSVSingleFileOutputFactory(path, limit_output)
    return ConsoleTableSingleFileOutputFactory(description, limit_output)

def create_multi_output_factory(output_type: str, prefix: str, description: str, limit_output):
    if output_type == 'csv':
        return CSVMultiFileOutputFactory(prefix, limit_output)
    return ConsoleTableMultiOutputFactory(description, limit_output)


def main(config: Config) -> None:
    data_processors = []
    if config.compare_compile_time:
        output_factory = create_single_output_factory(config.output_type,
                                                      config.compare_compile_time,
                                                      'compilation time',
                                                      config.limit_output)
        data_processors.append(CompareCompileTime(output_factory))
    if config.transformations_overall:
        output_factory = create_single_output_factory(config.output_type,
                                                      config.transformations_overall,
                                                      'transformations overall',
                                                      config.limit_output)
        data_processors.append(GenerateLongestUnitsOverall(output_factory, unit_type='transformation'))
    if config.manager_overall:
        output_factory = create_single_output_factory(config.output_type,
                                                      config.manager_overall,
                                                      'managers overall',
                                                      config.limit_output)
        data_processors.append(GenerateLongestUnitsOverall(output_factory, unit_type='manager'))
    if config.transformations_per_model:
        output_factory = create_multi_output_factory(config.output_type,
                                                     config.transformations_per_model,
                                                     'transformations per model',
                                                     config.limit_output)
        data_processors.append(GenerateLongestUnitsPerModel(output_factory, unit_type='transformation'))
    if config.managers_per_model:
        output_factory = create_multi_output_factory(config.output_type,
                                                     config.managers_per_model,
                                                     'managers per model',
                                                     config.limit_output)
        data_processors.append(GenerateLongestUnitsPerModel(output_factory, unit_type='manager'))
    if config.compare_transformations_overall:
        output_factory = create_single_output_factory(config.output_type,
                                                      config.compare_transformations_overall,
                                                      'compare transformations overall',
                                                      config.limit_output)
        data_processors.append(CompareSumUnitsOverall(output_factory, unit_type='transformation'))
    if config.compare_managers_overall:
        output_factory = create_single_output_factory(config.output_type,
                                                      config.compare_managers_overall,
                                                      'compare managers overall',
                                                      config.limit_output)
        data_processors.append(CompareSumUnitsOverall(output_factory, unit_type='manager'))
    if config.compare_transformations_per_model:
        output_factory = create_multi_output_factory(config.output_type,
                                                     config.compare_transformations_per_model,
                                                     'compare transformations per model',
                                                     config.limit_output)
        data_processors.append(CompareSumUnitsPerModel(output_factory, unit_type='transformation'))
    if config.compare_managers_per_model:
        output_factory = create_multi_output_factory(config.output_type,
                                                     config.compare_managers_per_model,
                                                     'compare managers per model',
                                                     config.limit_output)
        data_processors.append(CompareSumUnitsPerModel(output_factory, unit_type='manager'))


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
