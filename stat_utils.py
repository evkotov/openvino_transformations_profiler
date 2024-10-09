from typing import List, Dict, Iterator, Tuple

from compare_csv import ModelInfo, ModelData


def full_join_by_model_info(data: List[Dict[ModelInfo, ModelData]]) -> Iterator[Tuple[ModelInfo, Iterator[ModelData]]]:
    keys = set(info for data_item in data for info in data_item)
    for model_info in keys:
        items = (item[model_info] for item in data if model_info in item)
        yield model_info, items


def get_compile_durations(model_data_items: Iterator[ModelData]) -> Iterator[List[float]]:
    return (model_data.get_compile_durations() for model_data in model_data_items)


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
