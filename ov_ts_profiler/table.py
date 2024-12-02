from __future__ import annotations
from typing import List, Dict, Tuple, Optional
from ov_ts_profiler.common_structs import ModelInfo, ModelData, ComparisonValues, Unit
from ov_ts_profiler.stat_utils import Total


def compare_compile_time(data: List[Tuple[ModelInfo, List[Optional[float]]]], n_csv_files: int):
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

    header = create_header(n_csv_files)
    table = []
    for model_info, compile_times in data:
        row = {'framework': model_info.framework,
               'name': model_info.name,
               'precision': model_info.precision,
               'config': model_info.config}
        for csv_idx in range(n_csv_files):
            value = compile_times[csv_idx] if compile_times[csv_idx] is not None else 'N/A'
            row[f'compile time #{csv_idx + 1} (secs)'] = value
        for csv_idx in range(1, n_csv_files):
            delta = 'N/A'
            if compile_times[0] is not None and compile_times[csv_idx] is not None:
                delta = compile_times[csv_idx] - compile_times[0]
            row[f'compile time #{csv_idx + 1} - #1 (secs)'] = delta
        for csv_idx in range(1, n_csv_files):
            ratio = 'N/A'
            if compile_times[0] is not None and compile_times[csv_idx] is not None and compile_times[0] != 0.0:
                ratio = compile_times[csv_idx] / compile_times[0]
            row[f'compile time #{csv_idx + 1}/#1'] = ratio
        table.append(row)

    delta_header_names = get_delta_header_names(n_csv_files)
    def get_max_delta(row: Dict) -> float:
        return max((abs(row[key]) for key in delta_header_names if isinstance(row[key], float)),
                   default=None)
    return header, sort_table(table, get_max_delta)


def compare_sum_transformation_time(data: List[Tuple[ModelInfo, List[Optional[float]]]], n_csv_files: int):
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


    header = create_header(n_csv_files)
    table = []
    for model_info, compile_times in data:
        row = {'framework': model_info.framework,
               'name': model_info.name,
               'precision': model_info.precision,
               'config': model_info.config}
        for csv_idx in range(n_csv_files):
            value = compile_times[csv_idx] if compile_times[csv_idx] is not None else 'N/A'
            row[f'time #{csv_idx + 1} (ms)'] = value
        for csv_idx in range(1, n_csv_files):
            delta = 'N/A'
            if compile_times[0] is not None and compile_times[csv_idx] is not None:
                delta = compile_times[csv_idx] - compile_times[0]
            row[f'time #{csv_idx + 1} - #1 (ms)'] = delta
        for csv_idx in range(1, n_csv_files):
            ratio = 'N/A'
            if compile_times[0] is not None and compile_times[csv_idx] is not None and compile_times[0] != 0.0:
                ratio = compile_times[csv_idx] / compile_times[0]
            row[f'time #{csv_idx + 1}/#1'] = ratio
        table.append(row)

    delta_header_names = get_delta_header_names(n_csv_files)
    def get_max_delta(row: Dict) -> float:
        return max((abs(row[key]) for key in delta_header_names if isinstance(row[key], float)),
                   default=None)
    return header, sort_table(table, get_max_delta)


def get_longest_unit(data: Dict[str, Total]):
    header = ['name', 'total duration (ms)', 'count of executions', 'count of status true']
    table = []
    for name, total in data.items():
        row = {'name': name,
               'total duration (ms)': total.duration / 1_000_000,
               'count of executions': total.count,
               'count of status true': total.count_status_true}
        table.append(row)
    def get_duration(row: Dict) -> float:
        return row['total duration (ms)']
    return header, sort_table(table, get_duration)


def compare_sum_units(data: Dict[str, List[Optional[Total]]], n_csv_files: int):
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
        for i in range(n_csv_files):
            column_names.append(f'count status true #{i + 1}')
        for i in range(1, n_csv_files):
            column_names.append(f'count status true #{i + 1} - #1')
        return column_names

    def get_delta_header_names(n_csv_files: int) -> List[str]:
        column_names = []
        for csv_idx in range(1, n_csv_files):
            column_names.append(f'duration #{csv_idx + 1} - #1 (ms)')
        return column_names

    table = []

    for name, totals in data.items():
        row = {'name' : name}

        for csv_idx in range(n_csv_files):
            duration = totals[csv_idx].get_duration_ms() if totals[csv_idx] is not None else 'N/A'
            row[f'duration #{csv_idx + 1} (ms)'] = duration
        for csv_idx in range(1, n_csv_files):
            if totals[0] is None or totals[csv_idx] is None:
                delta = 'N/A'
            else:
                delta = totals[csv_idx].get_duration_ms() - totals[0].get_duration_ms()
            row[f'duration #{csv_idx + 1} - #1 (ms)'] = delta
        for csv_idx in range(1, n_csv_files):
            if totals[0] is None or totals[csv_idx] is None or totals[0].get_duration_ms() == 0.0:
                ratio = 'N/A'
            else:
                ratio = totals[csv_idx].get_duration_ms() / totals[0].get_duration_ms()
            row[f'duration #{csv_idx + 1}/#1'] = ratio
        for csv_idx in range(n_csv_files):
            count = totals[csv_idx].count if totals[csv_idx] is not None else 'N/A'
            row[f'count #{csv_idx + 1}'] = count
        for csv_idx in range(1, n_csv_files):
            if totals[0] is None or totals[csv_idx] is None:
                delta = 'N/A'
            else:
                delta = totals[csv_idx].count - totals[0].count
            row[f'count #{csv_idx + 1} - #1'] = delta
        for csv_idx in range(n_csv_files):
            count_status_true = totals[csv_idx].count_status_true if totals[csv_idx] is not None else 'N/A'
            row[f'count status true #{csv_idx + 1}'] = count_status_true
        for csv_idx in range(1, n_csv_files):
            if totals[0] is None or totals[csv_idx] is None:
                delta = 'N/A'
            else:
                delta = totals[csv_idx].count_status_true - totals[0].count_status_true
            row[f'count status true #{csv_idx + 1} - #1'] = delta
        table.append(row)

    header = create_header(n_csv_files)

    delta_header_names = get_delta_header_names(n_csv_files)
    def get_max_delta(row: Dict) -> float:
        return max((abs(row[key]) if isinstance(row[key], float) else 0.0 for key in delta_header_names))
    return header, sort_table(table, get_max_delta)


def create_comparison_summary_table(data: Dict[ModelInfo, ComparisonValues]):
    if not data:
        return [], []
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
