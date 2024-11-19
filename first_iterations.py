import sys
from typing import List, Dict, Iterator, Tuple
from compare_csv import ModelInfo, ModelData, Unit, get_csv_data
import csv
import numpy as np
from plot_utils import Plot, generate_x_ticks_cast_to_int


def get_compile_time(csv_data: Dict[ModelInfo, ModelData]) -> List[Tuple[ModelInfo, List[float]]]:
    Unit.USE_NO_CACHE = True
    Unit.USE_ONLY_0_ITER_GPU = False
    Unit.USE_ONLY_0_ITER = False
    Unit.ONLY_FIRST_N_ITER_NUM = None
    result = []
    for model_info, model_data in csv_data.items():
        compile_times = []
        for n_iter in range(1, model_data.get_n_iterations() + 1):
            Unit.ONLY_FIRST_N_ITER_NUM = n_iter
            compile_times.append(model_data.get_compile_time())
        Unit.ONLY_FIRST_N_ITER_NUM = None
        result.append((model_info, compile_times))
    return result


def get_unit_sum(csv_data: Dict[ModelInfo, ModelData], unit_type: str) -> List[Tuple[ModelInfo, List[float]]]:
    Unit.USE_NO_CACHE = True
    Unit.USE_ONLY_0_ITER_GPU = False
    Unit.USE_ONLY_0_ITER = False
    Unit.ONLY_FIRST_N_ITER_NUM = None
    result = []
    for model_info, model_data in csv_data.items():
        units = list(model_data.get_units_with_type(unit_type))
        unit_sums = []
        for n_iter in range(1, model_data.get_n_iterations() + 1):
            Unit.ONLY_FIRST_N_ITER_NUM = n_iter
            durations = (e.get_duration_median() for e in units)
            unit_sums.append(sum(durations))
        Unit.ONLY_FIRST_N_ITER_NUM = None
        result.append((model_info, unit_sums))
    return result


def get_ts_sum(csv_data: Dict[ModelInfo, ModelData]) -> List[Tuple[ModelInfo, List[float]]]:
    return get_unit_sum(csv_data, 'transformation')


def get_compile_time_table(data: List[Tuple[ModelInfo, List[float]]]) -> Tuple[List[Dict[str, float]], List[str], int]:
    max_iterations = max(len(e[1]) for e in data)

    def create_header(max_iterations: int):
        column_names = ['framework',
                        'name',
                        'precision',
                        'config']
        for i in range(max_iterations, 0, -1):
            column_names.append(f'time #{i} (secs)')
        for i in range(max_iterations - 1, 0, -1):
            column_names.append(f'abs(time #{i} - #{max_iterations}) (secs)')
        for i in range(max_iterations - 1, 0, -1):
            column_names.append(f'time #{i}/#{max_iterations}')
        for i in range(max_iterations - 1, 0, -1):
            column_names.append(f'abs(1 - time #{i}/#{max_iterations})')
        return column_names

    table = []
    for model_info, time_values in data:
        row = {'framework': model_info.framework,
               'name': model_info.name,
               'precision': model_info.precision,
               'config': model_info.config}
        compile_times = [value / 1_000_000_000 for value in time_values]
        while len(compile_times) < max_iterations:
            compile_times.append(None)
        for i in range(0, max_iterations):
            value = compile_times[i] if compile_times[i] is not None else 'N/A'
            row[f'time #{i + 1} (secs)'] = value
        for i in range(0, max_iterations - 1):
            delta = 'N/A'
            if compile_times[max_iterations - 1] is not None and compile_times[i] is not None:
                delta = abs(compile_times[i] - compile_times[max_iterations - 1])
            row[f'abs(time #{i + 1} - #{max_iterations}) (secs)'] = delta
        for i in range(0, max_iterations - 1):
            ratio = 'N/A'
            if compile_times[max_iterations - 1] is not None and compile_times[i] is not None and compile_times[max_iterations - 1] != 0.0:
                ratio = compile_times[i] / compile_times[max_iterations - 1]
            row[f'time #{i + 1}/#{max_iterations}'] = ratio
        for i in range(0, max_iterations - 1):
            ratio = 'N/A'
            if compile_times[max_iterations - 1] is not None and compile_times[i] is not None and compile_times[max_iterations - 1] != 0.0:
                ratio = abs(1 - compile_times[i] / compile_times[max_iterations - 1])
            row[f'abs(1 - time #{i + 1}/#{max_iterations})'] = ratio
        table.append(row)
    return table, create_header(max_iterations), max_iterations


def get_table(path: str, func) -> Tuple[List[Dict[str, float]], List[str], int]:
    csv_data = get_csv_data([path])
    if not csv_data:
        sys.exit(1)
    compile_time_iter = func(csv_data[0])
    return get_compile_time_table(compile_time_iter)


def save_table(table: List[Dict[str, float]], column_names: List[str], output_path: str) -> None:
    with open(output_path, 'w', newline='') as f_out:
        csv_writer = csv.DictWriter(f_out, fieldnames=column_names, delimiter=';')
        csv_writer.writeheader()
        for row in table:
            csv_writer.writerow(row)


def get_ratios(table: List[Dict[str, float]], n_iter: int, max_iterations: int) -> List[float]:
        column_name = f'abs(1 - time #{n_iter}/#{max_iterations})'
        return [row[column_name] for row in table if row[column_name] is not None]


def get_ratio_stats(table: List[Dict[str, float]], max_iterations: int, get_stat_func) -> List[float]:
    result = []
    for i in range(1, max_iterations):
        ratios = get_ratios(table, i, max_iterations)
        result.append(get_stat_func(ratios))
    return result


def gen_graph(title: str, output_name: str, table: List[Dict[str, float]], max_iterations: int):
    x_label = 'iteration number'
    y_label = f'abs(1 - time #i/#{max_iterations}), %'

    plot = Plot(title, x_label, y_label)
    plot.set_x_ticks_func(generate_x_ticks_cast_to_int)

    def get_stat_mean(nums: List[float]) -> float:
        return float(np.mean(nums)) * 100.0
    def get_stat_max(nums: List[float]) -> float:
        return float(np.max(nums)) * 100.0
    def get_stat_std(nums: List[float]) -> float:
        return float(np.std(nums)) * 100.0

    mean_values = get_ratio_stats(table, max_iterations, get_stat_mean)
    max_values = get_ratio_stats(table, max_iterations, get_stat_max)

    std_values = get_ratio_stats(table, max_iterations, get_stat_std)
    min_std_values = [max(0.0, mean - std) for mean, std in zip(mean_values, std_values)]
    max_std_values = [(mean + std) for mean, std in zip(mean_values, std_values)]

    iterations = [i for i in range(1, len(max_values) + 1)]
    plot.add(iterations, max_values, 'maximum')
    plot.add(iterations, mean_values, 'mean')
    plot.add_stripe_non_line(iterations, min_std_values, max_std_values, 'std deviation')
    plot.plot(output_name)


if __name__ == '__main__':
    input_path = sys.argv[1]
    table, column_names, max_iterations = get_table(input_path, get_compile_time)
    save_table(table, column_names, 'compile_time.csv')
    gen_graph(f'Relative Change in Median Compilation Time on GPU of First i Iterations\n(Averaged Across All Models)', 'compile_time_ratio.png', table, max_iterations)

    table, column_names, max_iterations = get_table(input_path, get_ts_sum)
    save_table(table, column_names, 'ts_sum.csv')
    gen_graph('Relative Change in Median Sum of All Transformations Time on GPU of First i Iterations\n(Averaged Across All Models)', 'sum_ts_ratio.png', table, max_iterations)
