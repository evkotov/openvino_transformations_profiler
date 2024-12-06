from __future__ import annotations

from collections import namedtuple, deque
from typing import Optional, List, Iterator, Dict, Tuple

import numpy as np

CSVColumnNames = ('device',
                  'model_path',
                  'model_name',
                  'model_framework',
                  'model_precision',
                  'config', # optional_model_attribute, weight_compression
                  'iteration',
                  'type',
                  'transformation_name',
                  'manager_name',
                  'duration',
                  'status')


CSVItem = namedtuple('CSVItem', CSVColumnNames)


class Unit:
    USE_ONLY_0_ITER_GPU: bool = False
    USE_ONLY_0_ITER: bool = False
    ONLY_FIRST_N_ITER_NUM: Optional[int] = None
    USE_NO_CACHE: bool = False

    def __init__(self, csv_item: CSVItem):
        self.name = None
        self.device = csv_item.device
        assert csv_item.type in ['compile_time', 'transformation', 'manager', 'manager_start', 'manager_end']
        if csv_item.type == 'transformation':
            self.name = csv_item.transformation_name
        elif csv_item.type == 'manager' or csv_item.type == 'manager_start' or csv_item.type == 'manager_end':
            self.name = csv_item.manager_name
        self.model_path = csv_item.model_path
        self.model_framework = csv_item.model_framework
        self.model_precision = csv_item.model_precision
        self.type = csv_item.type
        self.transformation_name = csv_item.transformation_name
        self.manager_name = csv_item.manager_name
        self.status: Optional[bool] = csv_item.status
        self.__durations: List[float] = [float(csv_item.duration)]
        self.__duration_median: Optional[float] = None
        self.__deviations: Optional[List[float]] = None
        self.__n_durations: Optional[int] = None
        self.__duration_mean: Optional[float] = None

    def get_n_durations(self) -> int:
        if self.__n_durations is None or Unit.USE_NO_CACHE:
            num_iterations = [len(self.__durations)]
            if Unit.ONLY_FIRST_N_ITER_NUM is not None:
                num_iterations.append(Unit.ONLY_FIRST_N_ITER_NUM)
            if self.use_only_first_iter():
                num_iterations.append(1)
            self.__n_durations = min(num_iterations)
        return self.__n_durations

    def get_durations(self) -> List[float]:
        return self.__durations[:self.get_n_durations()]

    def use_only_first_iter(self) -> bool:
        return Unit.USE_ONLY_0_ITER_GPU and self.device == 'GPU' or \
               Unit.USE_ONLY_0_ITER

    def get_duration_median(self) -> float:
        if self.__duration_median is None or Unit.USE_NO_CACHE:
            durations = self.get_durations()
            if not durations:
                self.__duration_median = 0.0
            else:
                self.__duration_median = float(np.median(durations))
        return self.__duration_median

    def get_duration_mean(self) -> float:
        if self.__duration_mean is None or Unit.USE_NO_CACHE:
            durations = self.get_durations()
            if not durations:
                self.__duration_mean = 0.0
            else:
                self.__duration_mean = float(np.mean(durations))
        return self.__duration_mean

    def get_deviations(self) -> List[float]:
        if self.__deviations is None or Unit.USE_NO_CACHE:
            median = self.get_duration_median()
            self.__deviations = [abs(item - median) for item in self.get_durations()]
        return self.__deviations

    def get_duration_stddev(self) -> float:
        return float(np.std(self.get_durations()))

    def get_variations_as_ratio(self) -> List[float]:
        if self.__deviations is None or Unit.USE_NO_CACHE:
            median = self.get_duration_median()
            self.__deviations = []
            for item in self.__durations:
                if median != 0.0:
                    self.__deviations.append(abs(item - median) / median)
                else:
                    self.__deviations.append(0.0)
        return self.__deviations

    def add(self, csv_item: CSVItem) -> None:
        assert self.model_path == csv_item.model_path
        assert self.model_framework == csv_item.model_framework
        assert self.model_precision == csv_item.model_precision
        assert self.type == csv_item.type
        assert self.transformation_name == csv_item.transformation_name
        assert self.manager_name == csv_item.manager_name
        assert self.status == csv_item.status
        self.__durations.append(float(csv_item.duration))
        self.__duration_median = None


UnitInfo = namedtuple('UnitInfo', ['type',
                                   'transformation_name',
                                   'manager_name'])


class ModelData:
    def __init__(self):
        self.items: List[Unit] = []
        self.__item_last_idx = None
        self.__last_iter_num: int = 0
        self.__manager_plain_sequence: Optional[List[Tuple[Unit, Unit]]] = None
        self.__manager_plain_sequence_sum: Optional[float] = None
        self.__manager_plain_sequence_median_gap_sum: Optional[float] = None
        self.__manager_plain_sequence_sum_by_iteration: Optional[List[float]] = None
        self.__manager_plain_sequence_median_gap_sum_by_iteration: Optional[List[float]] = None
        self.__manager_plain_sequence_median_gap_sum: Optional[float] = None

    def append(self, csv_item: CSVItem) -> None:
        n_iteration = int(csv_item.iteration)
        assert n_iteration > 0
        assert n_iteration >= self.__last_iter_num, \
            'consistency error: the iteration numbers must follow in ascending order'
        assert n_iteration == self.__last_iter_num or \
            n_iteration == self.__last_iter_num + 1
        self.__last_iter_num = n_iteration
        if n_iteration == 1:
            self.items.append(Unit(csv_item))
        else:
            if (self.__item_last_idx is None or
                    self.__item_last_idx == len(self.items) - 1):
                self.__item_last_idx = 0
            else:
                self.__item_last_idx += 1
            self.items[self.__item_last_idx].add(csv_item)

    def get_device(self):
        # assume that all data were collected on one device
        return self.items[0].device

    def get_units(self, filter_item_func) -> Iterator[Unit]:
        for item in self.items:
            if filter_item_func(item):
                yield item

    def get_units_with_type(self, type_name: str) -> Iterator[Unit]:
        return self.get_units(lambda item: item.type == type_name)

    def __make_manager_plain_sequence(self) -> List[Tuple[Unit, Unit]]:
        manager_timestamp_units = list(self.get_units_with_type('manager_start'))
        manager_timestamp_units.extend(list(self.get_units_with_type('manager_end')))
        # sorting by 0 item, since I don't know how median values will intersect each other
        # sequence should be the same for all iterations, it is checked in check_manager_plain_sequence()
        manager_timestamp_units = sorted(manager_timestamp_units, key=lambda e: e.get_durations()[0])
        plain_seq = []
        stack = deque()
        for item in manager_timestamp_units:
            if item.type == 'manager_start':
                stack.append(item)
            else:
                assert stack, 'manager_end without manager_start'
                start_item = stack.pop()
                assert start_item.name == item.name, 'manager_start and manager_end have different names'
                if not stack:
                    plain_seq.append((start_item, item))
        return plain_seq

    def get_manager_plain_sequence(self) -> List[Tuple[Unit, Unit]]:
        if self.__manager_plain_sequence is None:
            self.__manager_plain_sequence = self.__make_manager_plain_sequence()
        return self.__manager_plain_sequence

    def get_manager_plain_sequence_names(self) -> List[str]:
        return [start.name for start, end in self.get_manager_plain_sequence()]

    def __make_manager_plain_sequence_median_sum(self) -> float:
        sums = self.get_manager_plain_sequence_sum_by_iteration()
        if not sums:
            return float(0.0)
        return float(np.median(sums))

    def __make_manager_plain_sequence_sum_by_iteration(self) -> List[float]:
        """
        Calculate the sum of manager plain sequence durations for each iteration.

        Returns:
            List[float]: A list of sums of manager plain sequence durations for each iteration.
        """
        manager_plain_seq = self.get_manager_plain_sequence()
        if not manager_plain_seq:
            return []
        n_durations = manager_plain_seq[0][0].get_n_durations()
        result = []
        for i in range(n_durations):
            result_sum = 0.0
            for start, end in manager_plain_seq:
                unit_starts = start.get_durations()[i]
                unit_ends = end.get_durations()[i]
                delta = unit_ends - unit_starts
                assert delta >= 0.0, f'negative plain sequence unit time {start.name} delta'
                result_sum += delta
            result.append(result_sum)
        return result

    def get_manager_plain_sequence_sum_by_iteration(self) -> List[float]:
        """
        Calculate the sum of manager plain sequence durations for each iteration.

        Returns:
            List[float]: A list of sums of manager plain sequence durations for each iteration.
        """
        if self.__manager_plain_sequence_sum_by_iteration is None:
            self.__manager_plain_sequence_sum_by_iteration = self.__make_manager_plain_sequence_sum_by_iteration()
        return self.__manager_plain_sequence_sum_by_iteration

    def get_manager_plain_sequence_median_sum(self) -> float:
        if self.__manager_plain_sequence_sum is None:
             self.__manager_plain_sequence_sum = self.__make_manager_plain_sequence_median_sum()
        return self.__manager_plain_sequence_sum

    def __make_manager_plain_sequence_median_gap_sum_by_iteration(self) -> List[float]:
        manager_plain_seq = self.get_manager_plain_sequence()
        if not manager_plain_seq:
            return []
        n_durations = manager_plain_seq[0][0].get_n_durations()
        result = []
        for i in range(n_durations):
            result_sum = 0.0
            prev_ends = None
            for start, end in manager_plain_seq:
                unit_starts = start.get_durations()[i]
                if prev_ends is not None:
                    delta = unit_starts - prev_ends
                    assert delta >= 0.0, f'negative plain sequence gap unit time delta'
                    result_sum += delta
                prev_ends = end.get_durations()[i]
            result.append(result_sum)
        return result

    def get_manager_plain_sequence_median_gap_sum_by_iteration(self) -> List[float]:
        if self.__manager_plain_sequence_median_gap_sum_by_iteration is None:
            self.__manager_plain_sequence_median_gap_sum_by_iteration = (
                self.__make_manager_plain_sequence_median_gap_sum_by_iteration())
        return self.__manager_plain_sequence_median_gap_sum_by_iteration

    def __make_manager_plain_sequence_median_gap_sum(self) -> float:
        sums = self.get_manager_plain_sequence_median_gap_sum_by_iteration()
        if not sums:
            return float(0.0)
        return float(np.median(sums))

    def get_manager_plain_sequence_median_gap_sum(self) -> float:
        if self.__manager_plain_sequence_median_gap_sum is None:
            self.__manager_plain_sequence_median_gap_sum = self.__make_manager_plain_sequence_median_gap_sum()
        return self.__manager_plain_sequence_median_gap_sum

    def collect_items_by_type(self, type_name: str) -> Dict[str, List[Unit]]:
        result: Dict[str, List[Unit]] = {}
        for item in self.get_units_with_type(type_name):
            if item.name not in result:
                result[item.name] = []
            result[item.name].append(item)
        return result

    def get_compile_time(self) -> Optional[float]:
        try:
            item = next(self.get_units_with_type('compile_time'))
            return item.get_duration_median()
        except StopIteration:
            return None

    def sum_transformation_time(self) -> float:
        units = self.get_units_with_type('transformation')
        return sum(unit.get_duration_median() for unit in units)

    def get_compile_durations(self) -> List[float]:
        item = next(self.get_units_with_type('compile_time'))
        return item.get_durations()

    def get_duration(self, i: int) -> float:
        return self.items[i].get_duration_median()

    def get_all_item_info(self) -> List[UnitInfo]:
        data = []
        for item in self.items:
            data.append(UnitInfo(item.type,
                                 item.transformation_name,
                                 item.manager_name))
        return data

    def get_item_info(self, i: int) -> UnitInfo:
        item = self.items[i]
        return UnitInfo(item.type, item.transformation_name, item.manager_name)

    def get_n_iterations(self):
        if not self.items:
            return 0
        return self.items[0].get_n_durations()

    def check_manager_plain_sequence(self):
        manager_timestamp_units = list(self.get_units_with_type('manager_start'))
        manager_timestamp_units.extend(list(self.get_units_with_type('manager_end')))

        if not manager_timestamp_units:
            return

        durations = [item.get_n_durations() for item in manager_timestamp_units]
        assert all(e == durations[0] for e in durations), \
            f'different number of items in different iterations: {durations}'
        n_durations = durations[0]

        first_sequence = None
        current_sequence = []
        for i in range(n_durations):
            manager_timestamp_units = sorted(manager_timestamp_units, key=lambda e: e.get_durations()[i])
            stack = deque()
            for item in manager_timestamp_units:
                if item.type == 'manager_start':
                    stack.append(item)
                else:
                    assert stack, 'manager_end without manager_start'
                    start_item = stack.pop()
                    assert start_item.name == item.name, 'manager_start and manager_end have different names'
                    assert start_item.get_durations()[i] <= item.get_durations()[i], \
                        f'manager_start time is greater than manager_end time'
                    current_sequence.append((start_item, item))
            assert not stack, 'manager_start without manager_end'
            if first_sequence is None:
                first_sequence = current_sequence
            else:
                assert first_sequence == current_sequence, 'different manager plain sequences in different iterations'
            current_sequence = []

    def check(self) -> None:
        if len(self.items) == 0:
            return
        assert all(e.device == self.items[0].device for e in self.items), \
            f'different devices found in input data'
        n_iteration_items = [item.get_n_durations() for item in self.items]
        assert all(e == n_iteration_items[0] for e in n_iteration_items), \
            f'different number of items in different iterations: {n_iteration_items}'
        # check if there is compile time in each iteration
        n_compile_time_items = sum(1 for _ in self.get_units_with_type('compile_time'))
        assert n_compile_time_items == 1, \
            f'iteration data must consists exact 1 compile_time item but there are: {n_compile_time_items}'
        self.check_manager_plain_sequence()


ModelInfo = namedtuple('ModelInfo', ['framework',
                                     'name',
                                     'precision',
                                     'config'])


SummaryStats = namedtuple('SummaryStats', ['delta_median', 'delta_mean', 'delta_std', 'delta_max_abs',
                                           'ratio_median', 'ratio_mean', 'ratio_std', 'ratio_max_abs', 'unit'])


class ComparisonValues:
    def __init__(self, unit: str):
        self.values1: List[float] = []
        self.values2: List[float] = []
        self.unit: str = unit

    def add(self, value1: float, value2: float):
        if value1 is None or value2 is None:
            return
        self.values1.append(value1)
        self.values2.append(value2)

    def get_differences(self):
        return np.array(self.values2) - np.array(self.values1)

    def get_ratios(self):
        return (np.array(self.values2) / np.array(self.values1) - 1.0) * 100.0

    def get_max_values(self):
        return [max(item1, item2) for item1, item2 in zip(self.values1, self.values2)]

    @staticmethod
    def get_stat_values(values: np.array):
        median = float(np.median(values))
        mean = float(np.mean(values))
        std = float(np.std(values))
        maximum = max(values, key=abs)
        return median, mean, std, float(maximum)

    def get_stats(self) -> SummaryStats:
        assert len(self.values1) == len(self.values2)
        ratios = self.get_ratios()
        deltas = self.get_differences()
        delta_median, delta_mean, delta_std, delta_max = ComparisonValues.get_stat_values(deltas)
        ratio_median, ratio_mean, ratio_std, ratio_max = ComparisonValues.get_stat_values(ratios)
        return SummaryStats(delta_median, delta_mean, delta_std, delta_max,
                            ratio_median, ratio_mean, ratio_std, ratio_max, self.unit)


def full_join_by_model_info(data: List[Dict[ModelInfo, ModelData]]) -> Iterator[Tuple[
    ModelInfo, List[Optional[ModelData]]]]:
    keys = set(info for data_item in data for info in data_item)
    for model_info in sorted(keys):
        items = [item[model_info] if model_info in item else None for item in data]
        yield model_info, items


def make_model_console_description(model_info: ModelInfo) -> str:
    name = [model_info.framework,
            model_info.name,
            model_info.precision]
    if model_info.config:
        name.append(model_info.config)
    return ' '.join(name)
