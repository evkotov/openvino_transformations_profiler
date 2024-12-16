import csv
from typing import Optional, List, Iterator, Dict, Tuple
import sys
from ov_ts_profiler.plot_utils import PlotOutput
from ov_ts_profiler.common_structs import ComparisonValues
import os


def read_csv(path: str) -> Iterator[List[str]]:
    with open(path, 'r') as f_in:
        csv_reader = csv.reader(f_in, delimiter=';')
        for row in csv_reader:
            if row[0] == 'model name':
                continue
            yield row


def filter_gt5(items: Iterator[List[str]]) -> Iterator[List[str]]:
    for item in items:
        if abs(float(item[3])) > 5.0:
            yield item


def get_csv_data(input_path: str) -> List[List[str]]:
    return list(filter_gt5(read_csv(input_path)))


def get_csv_data_dict(input_path: str) -> Dict[str, List[str]]:
    return {item[0]: item for item in get_csv_data(input_path)}


def find_common_models(data: List[Dict[str, List[str]]]) -> List[str]:
    common_models = None
    for item in data:
        if common_models is None:
            common_models = set(item.keys())
            continue
        common_models &= set(item.keys())
    return list(common_models)


if __name__ == '__main__':
    inputs = sys.argv[1:]
    data = []
    for input_path in inputs:
        data.append(get_csv_data_dict(input_path))
    comparison_values = ComparisonValues('')
    for name in sorted(find_common_models(data)):
        print(f'model: {name}')
        for i in range(len(data)):
            input_name = inputs[i]
            data_obj = data[i]
            print(f'{input_name}: {data_obj[name][1]} {data_obj[name][2]}')
            comparison_values.add(float(data_obj[name][1]), float(data_obj[name][2]))
    plot = PlotOutput('comparison', "", 1)
    plot.set_value1_label('master')
    plot.set_value2_label('branch')
    plot.plot(comparison_values)
