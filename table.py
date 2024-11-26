from __future__ import annotations
from typing import List, Dict, Optional, Tuple, Iterator
from common_structs import ModelInfo, ModelData, ComparisonValues, SummaryStats


def get_comparison_values(table: List[Dict], key1: str, key2: str, unit: str):
    comparison_values = ComparisonValues(unit)
    for row in table:
        value1 = row[key1]
        value2 = row[key2]
        if isinstance(value1, float) and isinstance(value2, float):
            comparison_values.add(value1, value2)
    return comparison_values


def compare_compile_time(data: List[Dict[ModelInfo, ModelData]]):
    if len(data) == 0:
        return [], []

    def create_header(n_csv_files: int):
        column_names = ['framework',
                        'name',
                        'precision',
                        'config']
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
    for model_info, model_data_items in full_join_by_model_info(data):
        row = {'framework': model_info.framework,
               'name': model_info.name,
               'precision': model_info.precision,
               'config': model_info.config}
        compile_times = [model_data.get_compile_time() / 1_000_000_000 if model_data is not None else None
                         for model_data in model_data_items]
        for csv_idx in range(n_cvs_files):
            value = compile_times[csv_idx] if compile_times[csv_idx] is not None else 'N/A'
            row[f'compile time #{csv_idx + 1} (secs)'] = value
        for csv_idx in range(1, n_cvs_files):
            delta = 'N/A'
            if compile_times[0] is not None and compile_times[csv_idx] is not None:
                delta = compile_times[csv_idx] - compile_times[0]
            row[f'compile time #{csv_idx + 1} - #1 (secs)'] = delta
        for csv_idx in range(1, n_cvs_files):
            ratio = 'N/A'
            if compile_times[0] is not None and compile_times[csv_idx] is not None and compile_times[0] != 0.0:
                ratio = compile_times[csv_idx] / compile_times[0]
            row[f'compile time #{csv_idx + 1}/#1'] = ratio
        table.append(row)

    comparison_values = get_comparison_values(table,
                                              'compile time #1 (secs)',
                                              'compile time #2 (secs)',
                                              'sec')

    delta_header_names = get_delta_header_names(n_cvs_files)
    def get_max_delta(row: Dict) -> float:
        return max((abs(row[key]) for key in delta_header_names if isinstance(row[key], float)),
                   default=None)
    return header, sort_table(table, get_max_delta), comparison_values


def compare_sum_transformation_time(data: List[Dict[ModelInfo, ModelData]]):
    if len(data) == 0:
        return [], []

    def create_header(n_csv_files: int):
        column_names = ['framework',
                        'name',
                        'precision',
                        'config']
        for csv_idx in range(n_csv_files):
            column_names.append(f'time #{csv_idx + 1} (ms)')
        for csv_idx in range(1, n_csv_files):
            column_names.append(f'time #{csv_idx + 1} - #1 (ms)')
        for csv_idx in range(1, n_csv_files):
            column_names.append(f'time #{csv_idx + 1}/#1')
        return column_names

    def get_delta_header_names(n_csv_files: int) -> List[str]:
        column_names = []
        for csv_idx in range(1, n_csv_files):
            column_names.append(f'time #{csv_idx + 1} - #1 (ms)')
        return column_names


    n_cvs_files = len(data)
    header = create_header(n_cvs_files)
    table = []
    for model_info, model_data_items in full_join_by_model_info(data):
        row = {'framework': model_info.framework,
               'name': model_info.name,
               'precision': model_info.precision,
               'config': model_info.config}
        compile_times = [model_data.sum_transformation_time() / 1_000_000 if model_data is not None else None
                         for model_data in model_data_items]
        for csv_idx in range(n_cvs_files):
            value = compile_times[csv_idx] if compile_times[csv_idx] is not None else 'N/A'
            row[f'time #{csv_idx + 1} (ms)'] = value
        for csv_idx in range(1, n_cvs_files):
            delta = 'N/A'
            if compile_times[0] is not None and compile_times[csv_idx] is not None:
                delta = compile_times[csv_idx] - compile_times[0]
            row[f'time #{csv_idx + 1} - #1 (ms)'] = delta
        for csv_idx in range(1, n_cvs_files):
            ratio = 'N/A'
            if compile_times[0] is not None and compile_times[csv_idx] is not None and compile_times[0] != 0.0:
                ratio = compile_times[csv_idx] / compile_times[0]
            row[f'time #{csv_idx + 1}/#1'] = ratio
        table.append(row)

    comparison_values = get_comparison_values(table,
                                              'time #1 (ms)',
                                              'time #2 (ms)',
                                              'ms')

    delta_header_names = get_delta_header_names(n_cvs_files)
    def get_max_delta(row: Dict) -> float:
        return max((abs(row[key]) for key in delta_header_names if isinstance(row[key], float)),
                   default=None)
    return header, sort_table(table, get_max_delta), comparison_values


class Total:
    def __init__(self):
        self.duration: float = 0.0
        self.count: int = 0

    def append(self, total):
        self.duration += total.duration
        self.count += total.count


def get_items_by_type(data: Dict[ModelInfo, ModelData],
                      unit_type: str) -> Dict[str, List[Unit]]:
    result: Dict[str, List[Unit]] = {}
    for model_info, model_data in data.items():
        for name, units in model_data.collect_items_by_type(unit_type).items():
            if name not in result:
                result[name] = []
            result[name].extend(units)
    return result


def get_sum_duration(data: Dict[ModelInfo, ModelData],
                     unit_type: str) -> Dict[str, Total]:
    result: Dict[str, Total] = {} # ts name: Total
    for name, units in get_items_by_type(data, unit_type).items():
        total = Total()
        total.duration = sum((unit.get_duration_median() for unit in units))
        total.count = len(units)
        result[name] = total
    return result


def get_sum_duration_all_csv(data: List[Dict[ModelInfo, ModelData]],
                             unit_type: str):
    result: Dict[str, Total] = {} # ts name: Total
    for csv_item in data:
        for name, total in get_sum_duration(csv_item, unit_type).items():
            if name not in result:
                result[name] = Total()
            result[name].append(total)
    return result


def get_longest_unit(data: List[Dict[ModelInfo, ModelData]],
                     unit_type: str):
    header = ['name', 'total duration (ms)', 'count of executions']
    table = []
    for name, total in get_sum_duration_all_csv(data, unit_type).items():
        row = {'name': name,
               'total duration (ms)': total.duration / 1_000_000,
               'count of executions': total.count}
        table.append(row)
    def get_duration(row: Dict) -> float:
        return row['total duration (ms)']
    return header, sort_table(table, get_duration)


def compare_sum_units(data: List[Dict[ModelInfo, ModelData]],
                      unit_type: str):
    def get_duration(aggregated_data_item: Dict[str, Total], name: str) -> float:
        duration = 0.0
        if name in aggregated_data_item:
            duration = aggregated_data_item[name].duration / 1_000_000
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

    aggregated_data = [get_sum_duration(csv_data, unit_type) for csv_data in data]
    transformation_names = set(ts_name for aggregated_data_item in aggregated_data
                              for ts_name in aggregated_data_item)

    for name in transformation_names:
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

    comparison_values = get_comparison_values(table,
                                              'duration #1 (ms)',
                                              'duration #2 (ms)',
                                              'ms')

    header = create_header(n_csv_files)

    delta_header_names = get_delta_header_names(n_csv_files)
    def get_max_delta(row: Dict) -> float:
        return max((abs(row[key]) for key in delta_header_names))
    return header, sort_table(table, get_max_delta), comparison_values


def create_comparison_summary_table(data: Dict[ModelInfo, ComparisonValues]):
    unit = data[next(iter(data))].unit
    column_names = ['framework', 'name', 'precision',
                    'config',
                    f'delta median, {unit}', f'delta mean, {unit}', f'delta std, {unit}', f'delta max, {unit}',
                    'ratio median', 'ratio mean', 'ratio std', 'ratio max']
    table = []
    for model_info, values in data.items():
        summary_stats = values.get_stats()
        row = {'framework': model_info.framework,
               'name': model_info.name,
               'precision': model_info.precision,
               'config': model_info.config,
               f'delta median, {unit}': summary_stats.delta_median,
               f'delta mean, {unit}': summary_stats.delta_mean,
               f'delta std, {unit}': summary_stats.delta_std,
               f'delta max, {unit}': summary_stats.delta_max_abs,
               'ratio median': summary_stats.ratio_median,
               'ratio mean': summary_stats.ratio_mean,
               'ratio std': summary_stats.ratio_std,
               'ratio max': summary_stats.ratio_max_abs
                }
        table.append(row)
    def get_delta_max(row: Dict) -> float:
        return row[f'delta max, {unit}']
    return column_names, sort_table(table, get_delta_max)


def sort_table(table: List[Dict], get_row_key_func) -> List[Dict]:
    sorting_table: List[Tuple[int, float]] = []
    for row_idx, row in enumerate(table):
        row_key = get_row_key_func(row)
        if not isinstance(row_key, float):
            row_key = 0.0
        sorting_table.append((row_idx, row_key))
    sorted_table = sorted(sorting_table, key=lambda e: e[1], reverse=True)
    result_table = []
    for row in sorted_table:
        result_table.append(table[row[0]])
    return result_table


def full_join_by_model_info(data: List[Dict[ModelInfo, ModelData]]) -> Iterator[Tuple[
    ModelInfo, Iterator[Optional[ModelData]]]]:
    keys = set(info for data_item in data for info in data_item)
    for model_info in keys:
        items = (item[model_info] if model_info in item else None for item in data)
        yield model_info, items
