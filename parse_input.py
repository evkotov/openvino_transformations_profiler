from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import List, Dict, Iterator

from common_structs import ModelInfo, ModelData, CSVItem, CSVColumnNames


def is_header_valid(column_names: List[str]) -> bool:
    optional_columns = {'device', 'status', 'config', 'optional_model_attribute', 'weight_compression'}
    expected_required_names = [e for e in CSVColumnNames if e not in optional_columns]
    real_required_names = [e for e in column_names if e not in optional_columns]
    return expected_required_names == real_required_names


def check_header(column_names: List[str]) -> None:
    assert is_header_valid(column_names),\
        f"invalid header expected {CSVColumnNames} but got {column_names}"


def get_config_value_from_path(path: str, config_values_cache: Dict[str, str]) -> str:
    if path in config_values_cache:
        return config_values_cache[path]

    def get_attr(path: str) -> str:
        try:
            p = Path(path)
            # Attempt to resolve the path to validate its syntax
            p.resolve()
            parts = [str(e) for e in p.parts]
            if len(parts) < 3:
                return ''
            return parts[-2]
        except Exception:
            return ''
    attr = get_attr(path)
    config_values_cache[path] = attr
    return attr


def get_csv_header(path: str) -> List[str]:
    with open(path) as f_in:
        csv_reader = csv.reader(f_in, delimiter=';')
        return next(csv_reader)


def read_csv(path: str) -> Iterator[CSVItem]:
    config_values_cache = {}
    with open(path) as f_in:
        csv_reader = csv.reader(f_in, delimiter=';')
        column_names = next(csv_reader)
        check_header(column_names)
        has_config = 'optional_model_attribute' in column_names or \
                     'weight_compression' in column_names or \
                     'config' in column_names
        has_device = 'device' in column_names
        has_status = 'status' in column_names
        for row in csv_reader:
            # if it's header inside CSV file
            if row[-1] == 'duration':
                continue
            if not has_device:
                row.insert(0, 'N/A')
            if not has_config:
                row.insert(5, '')
            if not has_status:
                row.append('N/A')
            if not row[5]:
                model_path = row[1]
                row[5] = get_config_value_from_path(model_path, config_values_cache)
            try:
                csv_item = CSVItem(*row)
            except:
                print(f'exception in row {row}')
                raise
            yield csv_item


def read_csv_data(csv_rows: Iterator[CSVItem]) -> Dict[ModelInfo, ModelData]:
    data: Dict[ModelInfo, ModelData] = {}
    last_model_info = None
    for item in csv_rows:
        model_info = ModelInfo(item.model_framework,
                               item.model_name,
                               item.model_precision,
                               item.config)
        if model_info not in data:
            data[model_info] = ModelData()
        else:
            '''consistency check for duplicates in CSV
            - If there is already such a model_info in data we have proceeded the same IR.
            - If previous entry in CSV file was not the same as current, we proceeded the same IR,
              than there was another IR and now we have duplicate of model IR data   
            '''
            assert last_model_info is None or last_model_info == model_info, \
                f'duplicate of {model_info} in CSV'
        data[model_info].append(item)
        last_model_info = model_info
    return data


def check_csv_data(data: List[Dict[ModelInfo, ModelData]]) -> None:
    if not data:
        return
    first_info = next(iter(data[0]))
    first_device = data[0][first_info].get_device()
    assert all(m_data.get_device() == first_device for d in data for _, m_data in d.items()), \
        f'different devices found in input data'


def remove_invalid_items(data: Dict[ModelInfo, ModelData]) -> Dict[ModelInfo, ModelData]:
    valid_data = {}
    for model_info, model_data in data.items():
        try:
            model_data.check()
            valid_data[model_info] = model_data
        except AssertionError as e:
            print(f'Removed invalid model data: {model_info} due to: {e}')
    return valid_data


def get_csv_data(csv_paths: List[str]) -> List[Dict[ModelInfo, ModelData]]:
    csv_data = []
    for csv_path in csv_paths:
        print(f'reading {csv_path} ...')
        csv_rows = read_csv(csv_path)
        current_csv_data = read_csv_data(csv_rows)
        current_csv_data = remove_invalid_items(current_csv_data)
        if current_csv_data:
            csv_data.append(current_csv_data)
    check_csv_data(csv_data)
    return csv_data


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
