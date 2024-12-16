import sys
import ov_ts_profiler.plot_utils as plot_utils
import numpy as np

if __name__ == '__main__':
    title = f'RSS memory consumption'
    x_label = 'number of run'
    y_label = f'Mb'

    plot = plot_utils.Plot(title, x_label, y_label)
    plot.set_x_ticks_func(plot_utils.generate_x_ticks_cast_to_int)

    model_data_items = [[432803840,
                         432746496,
                         432918528,
                         433025024,
                         433098752,
                         432607232,
                         433238016,
                         432828416,
                         432947200,
                         432807936,
                         433004544]]

    all_compile_time_values = []
    for durations in model_data_items:
        compile_time_values = [float(duration) / 1024 / 1024 for duration in durations]
        iterations = [i for i in range(1, len(compile_time_values) + 1)]
        all_compile_time_values.extend(compile_time_values)
        plot.add(iterations, compile_time_values)
        #
        median = np.median(durations)
        ratios = [abs(1 - duration / median) * 100.0 for duration in durations]
        max_ratio = max(ratios)
        print(f'Max ratio: {max_ratio} %')

    # Calculate the median value of y_values
    median_value = float(np.median(all_compile_time_values))
    plot.append_x_line(median_value, f'Median: {"%.2f" % median_value} Mb', 'red', '--')

    # maximum deviation from median in %
    max_deviation_abs = max((item for item in all_compile_time_values), key=lambda e: abs(e - median_value))
    max_deviation = abs(median_value - max_deviation_abs) * 100.0 / median_value

    if max_deviation > 1.0:
        # Calculate 10% deviation from the median
        deviation = 0.01 * median_value
        lower_bound = median_value - deviation
        upper_bound = median_value + deviation
        plot.set_stripe(lower_bound, upper_bound, label='1% deviation from the median')

    path = 'mem_rss_iterations.png'
    plot.plot(path)
