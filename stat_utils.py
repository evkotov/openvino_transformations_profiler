from typing import List, Dict, Iterator, Tuple
from compare_csv import ModelInfo, ModelData, full_join_by_model_info, get_items_by_type

import numpy as np


def get_compile_durations(model_data_items: Iterator[ModelData]) -> Iterator[List[float]]:
    return (model_data.get_compile_durations() for model_data in model_data_items if model_data is not None)


def compile_time_by_iterations(csv_data: List[Dict[ModelInfo, ModelData]]) -> Iterator[Tuple[ModelInfo, Iterator[List[float]]]]:
    for model_info, model_data_items in full_join_by_model_info(csv_data):
        yield model_info, get_compile_durations(model_data_items)


def get_iter_time_values(time_values: List[float]) -> List[float]:
    prev = 0.0
    result = []
    for value in time_values:
        new_value = prev + value
        result.append(new_value)
        prev = new_value
    return result


def get_stddev_unit_durations(model_data: ModelData, unit_type: str, min_median: float) -> List[float]:
    results = []
    for name, units in model_data.collect_items_by_type(unit_type).items():
        for unit in units:
            median = unit.get_duration_median()
            if median == 0.0 or median < min_median:
                continue
            stddev = unit.get_duration_stddev()
            ratio = stddev / median
            results.append(ratio)
    return results


def get_stddev_unit_durations_all_csv(csv_data: List[Dict[ModelInfo, ModelData]],
                                      unit_type: str,
                                      min_median: float) -> Iterator[Tuple[ModelInfo, List[float]]]:
    for model_info, model_data_items in full_join_by_model_info(csv_data):
        result = []
        for model_data in model_data_items:
            if model_data is None:
                continue
            result.extend(get_stddev_unit_durations(model_data, unit_type, min_median))
            yield model_info, result


def get_model_sum_units_durations_by_iteration(model_data: ModelData, unit_type: str) -> List[float]:
    results_durations = None
    for unit in model_data.get_items_with_type(unit_type):
        unit_durations = np.array(unit.get_durations())
        if results_durations is None:
            results_durations = unit_durations
            continue
        assert results_durations.size == unit_durations.size
        results_durations += unit_durations
    return results_durations.tolist()


def get_sum_units_durations_by_iteration(csv_data: List[Dict[ModelInfo, ModelData]],
                                         unit_type: str) -> Iterator[Tuple[ModelInfo, Iterator[List[float]]]]:
    for model_info, model_data_items in full_join_by_model_info(csv_data):
        durations = (get_model_sum_units_durations_by_iteration(data, unit_type) for data in model_data_items
                     if data is not None)
        yield model_info, durations
