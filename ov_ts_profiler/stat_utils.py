from __future__ import annotations

from typing import List, Dict, Set, Iterator, Tuple, Optional

import numpy as np

from ov_ts_profiler.common_structs import ModelInfo, ModelData, full_join_by_model_info, Unit, ComparisonValues


def get_common_models(data: List[Dict[ModelInfo, ModelData]]) -> List[ModelInfo]:
    if len(data) == 0:
        return []
    common_keys = data[0].keys()
    for csv_data in data:
        common_keys &= csv_data.keys()
    return list(common_keys)


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


def get_device(data: List[Dict[ModelInfo, ModelData]]) -> str:
    if not data:
        return ''
    key = next(iter(data[0]))
    return data[0][key].get_device()


def get_all_models(data: List[Dict[ModelInfo, ModelData]]) -> Set[ModelInfo]:
    return set(model_info for csv_data in data for model_info in csv_data)


def get_compile_durations(model_data_items: Iterator[ModelData]) -> Iterator[List[float]]:
    return (model_data.get_compile_durations() for model_data in model_data_items if model_data is not None)


def compile_time_by_iterations(csv_data: List[Dict[ModelInfo, ModelData]]) -> Iterator[Tuple[ModelInfo, Iterator[List[float]]]]:
    for model_info, model_data_items in full_join_by_model_info(csv_data):
        yield model_info, get_compile_durations(iter(model_data_items))


def get_model_sum_units_durations_by_iteration(model_data: ModelData, unit_type: str) -> List[float]:
    units = model_data.get_units_with_type(unit_type)
    durations = np.fromiter((num for unit in units for num in unit.get_durations()), float)
    n_iterations = model_data.get_n_iterations()
    durations = durations.reshape(-1, n_iterations)
    return np.sum(durations, axis=0).tolist()


def get_sum_units_durations_by_iteration(csv_data: List[Dict[ModelInfo, ModelData]],
                                         unit_type: str) -> Iterator[Tuple[ModelInfo, Iterator[List[float]]]]:
    for model_info, model_data_items in full_join_by_model_info(csv_data):
        durations = (get_model_sum_units_durations_by_iteration(data, unit_type) for data in model_data_items
                     if data is not None)
        yield model_info, durations


def get_plain_manager_time_by_iteration(csv_data: List[Dict[ModelInfo, ModelData]]) -> Iterator[Tuple[ModelInfo, Iterator[List[float]]]]:
    for model_info, model_data_items in full_join_by_model_info(csv_data):
        durations = (data.get_manager_plain_sequence_sum_by_iteration() for data in model_data_items
                     if data is not None)
        yield model_info, durations


def get_plain_manager_gap_time_by_iteration(csv_data: List[Dict[ModelInfo, ModelData]]) -> Iterator[Tuple[ModelInfo, Iterator[List[float]]]]:
    for model_info, model_data_items in full_join_by_model_info(csv_data):
        durations = (data.get_manager_plain_sequence_median_gap_sum_by_iteration() for data in model_data_items
                     if data is not None)
        yield model_info, durations


# return compilation time in seconds
def get_compile_time_data(data: List[Dict[ModelInfo, ModelData]]) -> Iterator[Tuple[ModelInfo, List[Optional[float]]]]:
    for model_info, model_data_items in full_join_by_model_info(data):
        compile_times = [
            (model_data.get_compile_time() / 1_000_000_000 if model_data is not None and model_data.get_compile_time() is not None else None)
            for model_data in model_data_items
        ]
        yield model_info, compile_times


def get_comparison_values_compile_time(data: List[Tuple[ModelInfo, List[Optional[float]]]]) -> ComparisonValues:
    values = ComparisonValues('sec')
    for _, compile_times in data:
        if len(compile_times) != 2 or compile_times[0] is None or compile_times[1] is None:
            continue
        values.add(compile_times[0], compile_times[1])
    return values


# return transformation time in milliseconds
def get_sum_transformation_time_data(data: List[Dict[ModelInfo, ModelData]]) -> Iterator[Tuple[ModelInfo, List[Optional[float]]]]:
    for model_info, model_data_items in full_join_by_model_info(data):
        compile_times = [model_data.sum_transformation_time() / 1_000_000 if model_data is not None else None
                         for model_data in model_data_items]
        yield model_info, compile_times


def get_comparison_values_sum_transformation_time(data: List[Tuple[ModelInfo, List[Optional[float]]]]) -> ComparisonValues:
    values = ComparisonValues('ms')
    for _, compile_times in data:
        if len(compile_times) != 2 or compile_times[0] is None or compile_times[1] is None:
            continue
        values.add(compile_times[0], compile_times[1])
    return values


def get_sum_plain_manager_time_data(data: List[Dict[ModelInfo, ModelData]]) -> Iterator[Tuple[ModelInfo, List[Optional[float]]]]:
    for model_info, model_data_items in full_join_by_model_info(data):
        compile_times = [model_data.get_manager_plain_sequence_median_sum() / 1_000_000 if model_data is not None else None
                         for model_data in model_data_items]
        yield model_info, compile_times


def get_sum_plain_manager_gap_time_data(data: List[Dict[ModelInfo, ModelData]]) -> Iterator[Tuple[ModelInfo, List[Optional[float]]]]:
    for model_info, model_data_items in full_join_by_model_info(data):
        compile_times = [model_data.get_manager_plain_sequence_median_gap_sum() / 1_000_000 if model_data is not None else None
                         for model_data in model_data_items]
        yield model_info, compile_times


def get_total_by_unit_names(units_by_type):
    """
    Calculate the total duration and count of units grouped by their names.

    Args:
        units_by_type (Dict[str, List[Unit]]): A dictionary where the keys are unit names and the values are lists of units.

    Returns:
        Dict[str, Total]: A dictionary where the keys are unit names and the values are Total objects containing the total duration and count of units.
    """
    total_by_unit_names: Dict[str, Total] = {}
    for name, units in units_by_type.items():
        if name not in total_by_unit_names:
            total_by_unit_names[name] = Total()
        for unit in units:
            total_by_unit_names[name].append_unit(unit)
    return total_by_unit_names


class Total:
    def __init__(self):
        self.duration: float = 0.0
        self.count: int = 0
        self.count_status_true: int = 0

    def get_duration_ms(self) -> float:
        return self.duration / 1_000_000

    def append(self, total):
        self.duration += total.duration
        self.count += total.count
        self.count_status_true += total.count_status_true

    def append_unit(self, unit: Unit):
        self.duration += unit.get_duration_median()
        self.count += 1
        if unit.status == '1':
            self.count_status_true += 1


def get_total_by_unit_names_by_csv(model_data_items: List[Optional[ModelData]], unit_type: str) -> List[Dict[str, Total]]:
    """
    Calculate the total duration and count of units by their names for each ModelData in the list.

    Args:
        model_data_items (List[Optional[ModelData]]): A list of ModelData objects or None.
        unit_type (str): The type of units to be grouped and totaled.

    Returns:
        List[Dict[str, Total]]: A list of dictionaries where each dictionary contains the total duration and count of units
                                grouped by their names for each ModelData in the input list.
    """
    total_by_unit_names_by_csv: List[Dict[str, Total]] = []
    for model_data in model_data_items:
        if not model_data:
            total_by_unit_names_by_csv.append({})
            continue
        units_by_type = model_data.collect_items_by_type(unit_type)
        total_by_unit_names = get_total_by_unit_names(units_by_type)
        total_by_unit_names_by_csv.append(total_by_unit_names)
    return total_by_unit_names_by_csv


def get_sum_units_comparison_data(data: List[Dict[ModelInfo, ModelData]], unit_type: str) -> Iterator[Tuple[ModelInfo, Dict[str, List[Optional[Total]]]]]:
    """
    Calculate the total duration and count of units by their names for each ModelData in the list and group them by ModelInfo.

    Args:
        data (List[Dict[ModelInfo, ModelData]]): A list of dictionaries where each dictionary maps ModelInfo to ModelData.
        unit_type (str): The type of units to be grouped and totaled.

    Returns:
        Iterator[Tuple[ModelInfo, Dict[str, List[Optional[Total]]]]]: An iterator of tuples where each tuple contains a ModelInfo and a dictionary.
                                                                      The dictionary maps unit names to lists of Total objects or None.
        return time in milliseconds
    """
    for model_info, model_data_items in full_join_by_model_info(data):
        total_by_unit_names_by_csv = get_total_by_unit_names_by_csv(model_data_items, unit_type)
        unit_names = set(name for csv in total_by_unit_names_by_csv for name in csv)
        total_list_by_unit_name: Dict[str, List[Optional[Total]]] = {}
        for name in unit_names:
            if name not in total_list_by_unit_name:
                total_list_by_unit_name[name] = []
            for item in total_by_unit_names_by_csv:
                if name not in item:
                    continue
                total_list_by_unit_name[name].append(item[name])
        yield model_info, total_list_by_unit_name


def join_sum_units(data: Iterator[Tuple[ModelInfo, Dict[str, List[Optional[Total]]]]]) -> Dict[str, List[Optional[Total]]]:
    """
    Combine the total duration and count of units by their names from multiple ModelData objects.

    Args:
        data (Iterator[Tuple[ModelInfo, Dict[str, List[Optional[Total]]]]]): An iterator of tuples where each tuple contains a ModelInfo and a dictionary.
                                                                             The dictionary maps unit names to lists of Total objects or None.

    Returns:
        Dict[str, List[Optional[Total]]]: A dictionary where the keys are unit names and the values are lists of Total objects or None.
                                          The lists are combined from multiple ModelData objects.
    """
    result: Dict[str, List[Optional[Total]]] = {}
    for model_info, total_list_by_unit_name in data:
        for name, total_list in total_list_by_unit_name.items():
            if name not in result:
                result[name] = total_list
                continue
            assert len(result[name]) == len(total_list)
            for i in range(len(total_list)):
                if total_list[i] is None:
                    continue
                if result[name][i] is None:
                    result[name][i] = total_list[i]
                    continue
                result[name][i].append(total_list[i])
    return result


def get_comparison_values_sum_units(data: Dict[str, List[Optional[Total]]]) -> ComparisonValues:
    values = ComparisonValues('ms')
    for _, total_list in data.items():
        if len(total_list) != 2 or total_list[0] is None or total_list[1] is None:
            continue
        values.add(total_list[0].get_duration_ms(), total_list[1].get_duration_ms())
    return values


def join_sum_units_by_name(data: Dict[str, List[Optional[Total]]]) -> Dict[str, Total]:
    result: Dict[str, Total] = {}
    for name, total_list in data.items():
        if name not in result:
            result[name] = Total()
        for total in total_list:
            if total is not None:
                result[name].append(total)
    return result


def join_mem_rss_by_model(csv_data: List[Dict[ModelInfo, ModelData]]) -> Iterator[Tuple[ModelInfo, List[Optional[int]]]]:
    for model_info, model_data_items in full_join_by_model_info(csv_data):
        values = [model_data.get_mem_rss() if model_data is not None else None for model_data in model_data_items]
        if not all(item is not None for item in values):
            continue
        yield model_info, [model_data.get_mem_rss() if model_data is not None else None for model_data in model_data_items]


def join_mem_virtual_by_model(csv_data: List[Dict[ModelInfo, ModelData]]) -> Iterator[Tuple[ModelInfo, List[Optional[int]]]]:
    for model_info, model_data_items in full_join_by_model_info(csv_data):
        values = [model_data.get_mem_virtual() if model_data is not None else None for model_data in model_data_items]
        if not all(item is not None for item in values):
            continue
        yield model_info, values
