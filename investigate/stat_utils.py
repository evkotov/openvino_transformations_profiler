from typing import List, Dict, Iterator, Tuple
from compare_csv import get_items_by_type
from table import full_join_by_model_info
from common_structs import Unit, ModelData, ModelInfo
from collections import namedtuple
import numpy as np


def get_compile_durations(model_data_items: Iterator[ModelData]) -> Iterator[List[float]]:
    return (model_data.get_compile_durations() for model_data in model_data_items if model_data is not None)


def compile_time_by_iterations(csv_data: List[Dict[ModelInfo, ModelData]]) -> Iterator[Tuple[ModelInfo, Iterator[List[float]]]]:
    for model_info, model_data_items in full_join_by_model_info(csv_data):
        yield model_info, get_compile_durations(model_data_items)
    return


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


def get_model_unit_sum_by_iterations(model_data: ModelData, unit_type: str) -> Dict[str, List[float]]:
    units: Dict[str, List[Unit]] = {}
    for unit in model_data.get_units_with_type(unit_type):
        if unit.name not in units:
            units[unit.name] = []
        units[unit.name].append(unit)
    results: Dict[str, List[float]] = {}
    n_iterations = model_data.get_n_iterations()
    for name in units:
        unit_list = units[name]
        durations = np.fromiter((num for unit in unit_list for num in unit.get_durations()), float)
        durations = durations.reshape(-1, n_iterations)
        results[name] = np.sum(durations, axis=0).tolist()
    return results


def get_model_unit_sum_by_iterations_all_csv(csv_data: List[Dict[ModelInfo, ModelData]],
                                             unit_type: str) -> Iterator[Dict[str, List[float]]]:
    for model_info, model_data_items in full_join_by_model_info(csv_data):
        unit_sums = (get_model_unit_sum_by_iterations(data, unit_type) for data in model_data_items
                     if data is not None)
        yield model_info, unit_sums
    return


Deviation = namedtuple('Deviation', ['median', 'deviation'])
Deviations = List[Deviation]

def get_model_units_deviations_by_iter(model_data: ModelData, unit_type: str) -> List[Deviations]:
    units = model_data.get_units_with_type(unit_type)
    n_iterations = model_data.get_n_iterations()
    deviations: List[List] = [[] for _ in range(n_iterations)]
    for unit in units:
        unit_deviations = unit.get_deviations()
        median = unit.get_duration_median()
        assert len(unit_deviations) == n_iterations
        for i in range(n_iterations):
            deviations[i].append(Deviation(median, unit_deviations[i]))
    return deviations


def get_model_units_deviations_by_iter_all_csv(csv_data: List[Dict[ModelInfo, ModelData]],
                                               unit_type: str) -> Iterator[Tuple[ModelInfo, List[Deviations]]]:
    for model_info, model_data_items in full_join_by_model_info(csv_data):
        deviations = []
        for data in model_data_items:
            if data is None:
                continue
            csv_deviations = get_model_units_deviations_by_iter(data, unit_type)
            while len(deviations) < len(csv_deviations):
                deviations.append([])
            for i in range(len(csv_deviations)):
                deviations[i].extend(csv_deviations[i])
        yield model_info, deviations
    return
