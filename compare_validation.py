import json
import sys
from typing import Dict, List
import numpy as np

from ov_ts_profiler.plot_utils import Hist


def load_json(json_file):
    with open(json_file) as f:
        return json.load(f)


def get_values(data) -> Dict[str, float]:
    result = {}
    for model_data in data:
        value = model_data[4]
        if not value:
            continue
        model_name = model_data[0]
        result[model_name] = value
    return result


def get_ratios(data1, data2) -> List[float]:
    ratios = []
    values1 = get_values(data1)
    values2 = get_values(data2)
    for name in values1:
        if name not in values2:
            continue
        ratios.append(abs(1 - values1[name]/values2[name]))
    return ratios


if __name__ == '__main__':
    data1 = load_json(sys.argv[1])
    data2 = load_json(sys.argv[2])
    ratios = get_ratios(data1, data2)
    mean = np.mean(ratios)
    std = np.std(ratios)
    maximum = np.max(ratios)
    print(f'mean: {mean}, std: {std}, max: {maximum}')
    hist = Hist('Ratio', 'Count', 'Ratio distribution')
    hist.set_values(ratios)
    hist.plot('diff_distribution.png')
