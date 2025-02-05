import json
import os

from typing import List, Tuple
from ov_ts_profiler.plot_utils import Plot


def gen_plot_debug_items(model_data_items: List[Tuple[float, float]]):
    title = f'mem rss'
    x_label = 'seconds'
    y_label = f'mem RSS (Mb)'

    plot = Plot(title, x_label, y_label)
    plot.set_plot_size((30, 20))

    for single_csv_items in model_data_items:
        x_values = [item[0] for item in single_csv_items]
        y_values = [item[1] for item in single_csv_items]
        plot.add(x_values, y_values)    

    plot.plot('mem_rss.png')

values = []
with open('monitor.json') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        monitor = json.loads(line)
        values.append((monitor['timestamp_ns'], monitor['bytes_used']))

min_timestamp = min((pair[0] for pair in values))
for i in range(len(values)):
    values[i] = ((values[i][0] - min_timestamp) / 1_000_000_000, values[i][1] / 1024 / 1024)

gen_plot_debug_items([values])
