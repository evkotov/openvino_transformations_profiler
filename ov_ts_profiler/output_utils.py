from __future__ import annotations

import csv
from abc import ABC, abstractmethod
from typing import List, Dict

from tabulate import tabulate

from ov_ts_profiler.common_structs import ComparisonValues, ModelInfo


def print_summary_stats(values: ComparisonValues):
    stats = values.get_stats()
    header = ['value', 'median', 'mean', 'std', 'max abs']
    rows = [{'value': f'delta ({stats.unit})',
             'median': f'{stats.delta_median:.2f}',
             'mean': f'{stats.delta_mean:.2f}',
             'std': f'{stats.delta_std:.2f}',
             'max abs': f'{stats.delta_max_abs:.2f}',
             }, {'value': 'ratio (%)',
                 'median': f'{stats.ratio_median:.2f}',
                 'mean': f'{stats.ratio_mean:.2f}',
                 'std': f'{stats.ratio_std:.2f}',
                 'max abs': f'{stats.ratio_max_abs:.2f}'}]
    ordered_rows_str = [{key: str(row[key]) for key in header} for row in rows]
    print(tabulate(ordered_rows_str, headers="keys"))


def make_model_file_name(prefix: str, model_info: ModelInfo, extension: str) -> str:
    name = []
    if prefix:
        name = [prefix]
    name.extend([model_info.framework,
                 model_info.name,
                 model_info.precision])
    if model_info.config:
        name.append(model_info.config)
    name = '_'.join(name)
    if extension:
        name = name + '.' + extension
    return name


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
    def write(self, row: List[Dict[str, str]]):
        pass


class NoOutput(Output):
    def __init__(self):
        super().__init__([])

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def write(self, row: List[Dict[str, str]]):
        pass


class CSVOutput(Output):
    def __init__(self, path: str, header: List[str], limit_output):
        super().__init__(header)
        self.path = path
        self.limit_output = limit_output
        self.file = None

    def write(self, rows: List[Dict[str, str]]):
        assert self.file is not None
        assert self.header is not None
        csv_writer = csv.DictWriter(self.file, fieldnames=self.header, delimiter=';')
        csv_writer.writeheader()
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
