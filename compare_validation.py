from typing import Dict, List, Tuple
import openpyxl
import sys
import os

from ov_ts_profiler.plot_utils import PlotOutput
from ov_ts_profiler.common_structs import ComparisonValues
from ov_ts_profiler.table import sort_table
from ov_ts_profiler.output_utils import CSVOutput


COLUMN_MODEL_NAME = 0
COLUMN_VALUE = 4


def parse_data(path: str):
    wb = openpyxl.load_workbook(path)
    sheet = wb.active
    data = {}
    for row in sheet.iter_rows(min_row=2, values_only=True):
        try:
            value = float(row[COLUMN_VALUE])
        except ValueError:
            continue
        data[row[COLUMN_MODEL_NAME]] = row[COLUMN_VALUE]
    return data



def full_join_by_model_name(data1: Dict[str, float], data2: Dict[str, float]) -> List[Tuple[str, float, float]]:
    names = set(key for key in data1 if key in data2)
    result = []
    for model_name in names:
        result.append((model_name, data1[model_name], data2[model_name]))
    return result


class Table:
    def __init__(self):
        self.__table = []
        self.__table_header = set()

    def add_row(self, row: Dict):
        row_keys = row.keys()
        if not self.__table_header:
            self.__table_header = row_keys
        elif self.__table_header != row_keys:
                raise ValueError('Row keys do not match header')
        self.__table.append(row)

    def get_table(self):
        return sort_table(self.__table, self.sort_key)

    def get_header(self):
        return list(self.__table_header)

    def sort_key(self, row: Dict) -> str:
        return row['ratio (branch/master - 1), %']


if __name__ == '__main__':
    input1 = sys.argv[1]
    input2 = sys.argv[2]
    data1 = parse_data(input1)
    data2 = parse_data(input2)
    table = Table()
    comparison_values = ComparisonValues('')
    for data in full_join_by_model_name(data1, data2):
        #print(f'{data[0]}: {data[1]} -> {data[2]}')
        comparison_values.add(data[1], data[2])
        ratio = (data[2] / data[1] - 1.0) * 100.0
        table.add_row({'model name': data[0], 'master': data[1], 'branch': data[2], 'ratio (branch/master - 1), %': ratio})
    prefix = os.path.splitext(os.path.basename(input1))[0] + os.path.splitext(os.path.basename(input2))[0]
    plot = PlotOutput(prefix, "", 1)
    plot.set_value1_label('master')
    plot.set_value2_label('branch')
    plot.plot(comparison_values)
    with CSVOutput(prefix + '.csv', table.get_header(), 0) as output:
        output.write(table.get_table())
