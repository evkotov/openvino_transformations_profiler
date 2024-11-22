import sys
from typing import List, Dict, Tuple, Optional
from compare_csv import ModelInfo, ModelData, Unit, get_csv_data, sort_table, full_join_by_model_info
import csv
import numpy as np
from plot_utils import Plot, generate_x_ticks_cast_to_int
from dataclasses import dataclass, field
import copy
import os


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


def get_model_compile_time(model_data: ModelData) -> List[float]:
    Unit.USE_NO_CACHE = True
    Unit.USE_ONLY_0_ITER_GPU = False
    Unit.USE_ONLY_0_ITER = False
    Unit.ONLY_FIRST_N_ITER_NUM = None
    compile_times = []
    for n_iter in range(1, model_data.get_n_iterations() + 1):
        Unit.ONLY_FIRST_N_ITER_NUM = n_iter
        compile_times.append(model_data.get_compile_time() / 1_000_000_000)
    Unit.ONLY_FIRST_N_ITER_NUM = None
    return compile_times


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


def get_unit_model_sum(model_data: ModelData, unit_type: str) -> List[float]:
    Unit.USE_NO_CACHE = True
    Unit.USE_ONLY_0_ITER_GPU = False
    Unit.USE_ONLY_0_ITER = False
    Unit.ONLY_FIRST_N_ITER_NUM = None
    units = list(model_data.get_units_with_type(unit_type))
    unit_sums = []
    for n_iter in range(1, model_data.get_n_iterations() + 1):
        Unit.ONLY_FIRST_N_ITER_NUM = n_iter
        durations = (e.get_duration_median() / 1_000_000_000 for e in units)
        unit_sums.append(sum(durations))
    Unit.ONLY_FIRST_N_ITER_NUM = None
    return unit_sums


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


@dataclass
class UnitRatioStats:
    mean_values: List[float]
    max_values: List[float]
    stddev_values: List[float]


@dataclass
class UnitValues:
    values: List[float] = field(default_factory=list)
    ratios: List[float] = field(default_factory=list)


def get_unit_ratio_stats(csv_data: List[Dict[ModelInfo, ModelData]], unit_type: str, min_value: Optional[float], n_iterations: int, exclude_unit_names: List[str]) -> Tuple[UnitRatioStats, Dict[str, UnitValues]]:
    Unit.USE_NO_CACHE = True
    Unit.USE_ONLY_0_ITER_GPU = False
    Unit.USE_ONLY_0_ITER = False
    Unit.ONLY_FIRST_N_ITER_NUM = None
    units = []

    all_unit_names = set()
    exclude_unit_names = set(exclude_unit_names)
    for data in csv_data:
        for model_info, model_data in data.items():
            Unit.ONLY_FIRST_N_ITER_NUM = None
            if model_data.get_n_iterations() < n_iterations:
                continue
            model_units = list(model_data.get_units_with_type(unit_type))
            all_unit_names.update(e.name for e in model_units)
            model_units = [e for e in model_units if e.name not in exclude_unit_names]
            if min_value is not None:
                # filter units with duration less than min_value
                Unit.ONLY_FIRST_N_ITER_NUM = n_iterations
                model_units = [e for e in model_units if e.get_duration_median() >= min_value]
                Unit.ONLY_FIRST_N_ITER_NUM = None
            units.extend(model_units)

    unit_names = set(e.name for e in units)
    print(f'used unit names : (len = {len(unit_names)}) {unit_names}')
    print(f'all unit names : (len = {len(all_unit_names)}) {all_unit_names}')

    durations = []
    for n_iter in range(1, n_iterations + 1):
        Unit.ONLY_FIRST_N_ITER_NUM = n_iter
        durations.append([e.get_duration_median() for e in units])
    Unit.ONLY_FIRST_N_ITER_NUM = None

    durations = np.array(durations)
    print(durations.shape)
    mean_values = []
    max_values = []
    stddev_values = []

    above_threshold = {}

    for i in range(durations.shape[0] - 1):
        ratios = np.abs(1 - durations[i] / durations[-1])
        mean_values.append(float(np.mean(ratios)))
        max_values.append(float(np.max(ratios)))
        stddev_values.append(float(np.std(ratios)))

        '''
        threshold = mean_values[-1] + stddev_values[-1]
        count_above_threshold = sum(1 for ratio in ratios if ratio > threshold)
        percent_above_threshold = (count_above_threshold / len(ratios)) * 100
        print(f'{i}. Count above threshold: {count_above_threshold} from {len(ratios)}')
        print(f'{i}. Percent above threshold: {percent_above_threshold:.2f}%')
        '''

        for idx in range(len(units)):
            unit_name = units[idx].name
            if unit_name not in above_threshold:
                above_threshold[unit_name] = UnitValues()
            above_threshold[unit_name].values.append(float(durations[i][idx]))
            above_threshold[unit_name].ratios.append(float(ratios[idx]))

    return UnitRatioStats(mean_values=mean_values, max_values=max_values, stddev_values=stddev_values), above_threshold


def get_unit_values_table(data: Dict[str, UnitValues]) -> Tuple[List[Dict[str, float]], List[str]]:
    column_names = ['name', 'mean value', 'max value', 'stddev value',
                    'mean ratio', 'max ratio', 'stddev ratio']
    table = []
    for unit_name in data:
        values = data[unit_name]
        row = {'name': unit_name,
               'mean value': np.mean(values.values),
               'max value': np.max(values.values),
               'stddev value': np.std(values.values),
               'mean ratio': np.mean(values.ratios),
               'max ratio': np.max(values.ratios),
               'stddev ratio': np.std(values.ratios)}
        table.append(row)
    def get_max_ratio_item(row: Dict) -> float:
        return row['max ratio']
    return sort_table(table, get_max_ratio_item), column_names


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
    x_label = 'number of iterations'
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


def gen_unit_ratio_stats_graph(stats: UnitRatioStats, output_name: str, max_iterations: int):
    title = 'Relative Change in Median Average Transformation Time on GPU of First i Iterations\n(Averaged Across All Models)'
    x_label = 'number of iterations'
    y_label = f'abs(1 - time #i/#{max_iterations}), %'

    plot = Plot(title, x_label, y_label)
    plot.set_x_ticks_func(generate_x_ticks_cast_to_int)

    iterations = [i for i in range(1, len(stats.mean_values) + 1)]
    plot.add(iterations, stats.max_values, 'max')
    plot.add(iterations, stats.mean_values, 'mean')
    min_std_values = [max(0.0, mean - std) for mean, std in zip(stats.mean_values, stats.stddev_values)]
    max_std_values = [(mean + std) for mean, std in zip(stats.mean_values, stats.stddev_values)]
    plot.add_stripe_non_line(iterations, min_std_values, max_std_values, 'std deviation')
    plot.plot(output_name)


def filter_csv_data_by_num_iterations(data: Dict[ModelInfo, ModelData], n_iterations: int) -> Dict[ModelInfo, ModelData]:
    return {model_info: model_data for model_info, model_data in data.items() if model_data.get_n_iterations() == n_iterations}


def filter_csv_data_list_by_num_iterations(data: List[Dict[ModelInfo, ModelData]], n_iterations: int) -> List[Dict[ModelInfo, ModelData]]:
    result = []
    for items in data:
        result.append(filter_csv_data_by_num_iterations(items, n_iterations))
    return result


def gen_graph_compile_time(title: str, file_name: str, durations_list: List[List[float]]):
    x_label = 'iteration number'
    y_label = f'time median (secs)'

    plot = Plot(title, x_label, y_label)
    plot.set_x_ticks_func(generate_x_ticks_cast_to_int)

    all_values = []
    for durations in durations_list:
        iterations = [i for i in range(1, len(durations) + 1)]
        all_values.extend(durations)

        assert len(durations) == len(iterations)
        plot.add(iterations, durations)

    if not all_values:
        print(f'no values for {model_info}')
        return

    # Calculate the median value of y_values
    median_value = float(np.median(all_values))
    plot.append_x_line(median_value, f'Median: {"%.2f" % median_value} seconds', 'red', '--')

    # maximum deviation from median in %
    max_deviation_abs = max((item for item in all_values), key=lambda e: abs(e - median_value))
    max_deviation = abs(median_value - max_deviation_abs) * 100.0 / median_value

    if max_deviation > 1.0:
        # Calculate 10% deviation from the median
        deviation = 0.01 * median_value
        lower_bound = median_value - deviation
        upper_bound = median_value + deviation
        plot.set_stripe(lower_bound, upper_bound, label='1% deviation from the median')

    plot.plot(file_name)


class ProgressStatus:
    def __init__(self, total_iterations: int):
        self.total_iterations = total_iterations
        self.current_iteration = 0
        self.last_length = 0

    def update(self, status: str):
        self.current_iteration += 1
        percent_complete = (self.current_iteration / self.total_iterations) * 100
        message = f'Progress: {percent_complete:.2f}% - {status}'
        sys.stdout.write('\r' + ' ' * self.last_length)
        sys.stdout.write('\r' + message)
        sys.stdout.flush()
        self.last_length = len(message)

    def complete(self):
        sys.stdout.write('\n')
        sys.stdout.flush()


def main_gen_graphs_time_series(data: Dict[ModelInfo, List[List[float]]], what: str):
    # TODO
    progress_status = ProgressStatus(len(data.keys()))
    for model_info, model_data_list in data:
        title = f'{what} on GPU of First i Iterations\n' \
                f'{model_info.framework} {model_info.name} {model_info.precision} {model_info.config}'
        file_name = f'{model_info.framework}_{model_info.name}_{model_info.precision}_{model_info.config}_compile_time.png'
        progress_status.update(f'{model_info.framework} {model_info.name} {model_info.precision} {model_info.config}')
        gen_graph_compile_time(title, file_name, model_data_list)
    progress_status.complete()


def main_gen_graphs_compilation_time(data: Dict[ModelInfo, List[ModelData]]):
    progress_status = ProgressStatus(len(data.keys()))
    for model_info, model_data_list in data.items():
        title = 'Median Compilation Time on GPU of First i Iterations\n' \
                f'{model_info.framework} {model_info.name} {model_info.precision} {model_info.config}'
        file_name = f'{model_info.framework}_{model_info.name}_{model_info.precision}_{model_info.config}_compile_time.png'
        progress_status.update(f'{model_info.framework} {model_info.name} {model_info.precision} {model_info.config}')
        compile_times_list = [get_model_compile_time(e) for e in model_data_list]
        gen_graph_compile_time(title, file_name, compile_times_list)
    progress_status.complete()


def main_gen_graphs_ts_sum(data: Dict[ModelInfo, List[ModelData]]):
    progress_status = ProgressStatus(len(data.keys()))
    for model_info, model_data_list in data.items():
        title = 'Median Sum of All Transformations Time on GPU of First i Iterations\n' \
                f'{model_info.framework} {model_info.name} {model_info.precision} {model_info.config}'
        file_name = f'{model_info.framework}_{model_info.name}_{model_info.precision}_{model_info.config}_ts_sum.png'
        progress_status.update(f'{model_info.framework} {model_info.name} {model_info.precision} {model_info.config}')
        times_list = [get_unit_model_sum(e, 'transformation') for e in model_data_list]
        gen_graph_compile_time(title, file_name, times_list)
    progress_status.complete()


def main_gen_graphs_manager_sum(data: Dict[ModelInfo, List[ModelData]]):
    progress_status = ProgressStatus(len(data.keys()))
    for model_info, model_data_list in data.items():
        title = 'Median Sum of All Managers Time on GPU of First i Iterations\n' \
                f'{model_info.framework} {model_info.name} {model_info.precision} {model_info.config}'
        file_name = f'{model_info.framework}_{model_info.name}_{model_info.precision}_{model_info.config}_manager_sum.png'
        progress_status.update(f'{model_info.framework} {model_info.name} {model_info.precision} {model_info.config}')
        times_list = [get_unit_model_sum(e, 'manager') for e in model_data_list]
        gen_graph_compile_time(title, file_name, times_list)
    progress_status.complete()


def get_all_csv(dir_path: str) -> List[str]:
    csv_files: List[str] = []
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            if file.endswith('.csv'):
                csv_files.append(os.path.join(root, file))
    return csv_files


def get_input_csv_files(inputs: List[str]) -> List[str]:
    csv_files: List[str] = []
    for input_path in inputs:
        if os.path.isdir(input_path):
            csv_files.extend(get_all_csv(input_path))
        else:
            csv_files.append(input_path)
    return sorted(csv_files)


if __name__ == '__main__':
    inputs = sys.argv[1:]
    csv_data = get_csv_data(get_input_csv_files(inputs))
    if not csv_data:
        sys.exit(1)
    join_iter = full_join_by_model_info(csv_data)
    data = {}
    for model_info, model_data_list in join_iter:
        model_data = [e for e in model_data_list if e is not None]
        if model_data:
            data[model_info] = model_data
    main_gen_graphs_compilation_time(copy.deepcopy(data))
    main_gen_graphs_ts_sum(copy.deepcopy(data))
    #main_gen_graphs_manager_sum(copy.deepcopy(data))

    '''
    input_path = sys.argv[1]
    table, column_names, max_iterations = get_table(input_path, get_compile_time)
    save_table(table, column_names, 'compile_time.csv')
    gen_graph(f'Relative Change in Median Compilation Time on GPU of First i Iterations\n(Averaged Across All Models)', 'compile_time_ratio.png', table, max_iterations)
    table, column_names, max_iterations = get_table(input_path, get_ts_sum)
    save_table(table, column_names, 'ts_sum.csv')
    gen_graph('Relative Change in Median Sum of All Transformations Time on GPU of First i Iterations\n(Averaged Across All Models)', 'sum_ts_ratio.png', table, max_iterations)
    csv_data = get_csv_data(sys.argv[1:])
    input_data = filter_csv_data_list_by_num_iterations(csv_data, 10)
    #unit_ratio_stats = get_unit_ratio_stats(input_data, 'transformation', float(1_000_000_000), 10)
    exclude_unit_names = ['ov::pass::ConstantFolding', 'ov::pass::SimplifyShapeOfSubGraph', 'ov::pass::ConvertPrecision', 'ov::pass::OptimizeSymbolsUsedAsValues',
                          'WrapInterpolateIntoTransposes', 'GroupNormComposition', 'ov::pass::SimplifyShapeOfSubGraph', 'EliminateGatherUnsqueeze']
    ts_duration_min = float(1_000_000)
    unit_ratio_stats, unit_values = get_unit_ratio_stats(input_data, 'transformation', ts_duration_min, 10, exclude_unit_names)
    print(f'mean {unit_ratio_stats.mean_values}')
    print(f'max {unit_ratio_stats.max_values}')
    print(f'std {unit_ratio_stats.stddev_values}')
    #table, column_names = get_unit_values_table(unit_values)
    #save_table(table, column_names, 'unit_values.csv')
    unit_ratio_stats.mean_values = [e * 100.0 for e in unit_ratio_stats.mean_values]
    unit_ratio_stats.max_values = [e * 100.0 for e in unit_ratio_stats.max_values]
    unit_ratio_stats.stddev_values = [e * 100.0 for e in unit_ratio_stats.stddev_values]
    gen_unit_ratio_stats_graph(unit_ratio_stats, 'avg_ts_ratio.png', 10)
    '''
