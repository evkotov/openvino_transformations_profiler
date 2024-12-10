from typing import Dict, List, Tuple
import openpyxl
import sys
import os

from ov_ts_profiler.plot_utils import PlotOutput
from ov_ts_profiler.common_structs import ComparisonValues

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


if __name__ == '__main__':
    input1 = sys.argv[1]
    input2 = sys.argv[2]
    data1 = parse_data(input1)
    data2 = parse_data(input2)
    comparison_values = ComparisonValues('')
    for data in full_join_by_model_name(data1, data2):
        comparison_values.add(data[1], data[2])
    prefix = os.path.basename(input1) + os.path.basename(input2)
    plot = PlotOutput(prefix, "", 1)
    plot.plot(comparison_values)
