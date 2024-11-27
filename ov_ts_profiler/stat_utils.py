from __future__ import annotations

from typing import List, Dict, Set, Iterator, Tuple

import numpy as np

from ov_ts_profiler.common_structs import ModelInfo, ModelData, full_join_by_model_info


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
        yield model_info, get_compile_durations(model_data_items)
    return


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


def find_iqr_outlier_indexes(values) -> Set[int]:
    q1 = np.percentile(values, 25)
    q3 = np.percentile(values, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    indexes = (i for i, x in enumerate(values) if x < lower_bound or x > upper_bound)
    return set(indexes)
