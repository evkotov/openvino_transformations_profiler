import os
import sys
from compare_csv import ModelInfo, get_csv_data
from typing import List


# TODO: move and use in compare_csv
def get_all_csv(dir_path: str) -> List[str]:
    csv_files: List[str] = []
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            if file.endswith('.csv'):
                csv_files.append(os.path.join(root, file))
    return csv_files


# TODO: move and use in compare_csv
def get_input_csv_files(inputs: List[str]) -> List[str]:
    csv_files: List[str] = []
    for input_path in inputs:
        if os.path.isdir(input_path):
            csv_files.extend(get_all_csv(input_path))
        elif input_path.endswith('.csv'):
            csv_files.append(input_path)
    return sorted(csv_files)


def print_stats():
    inputs = get_input_csv_files(sys.argv[1:])
    csv_data = get_csv_data(inputs)
    for i in range(len(csv_data)):
        print(f'{inputs[i]}')
        models = [m for m in csv_data[i].keys()]
        for m in sorted(models):
            print(m)


if __name__ == '__main__':
    print_stats()
