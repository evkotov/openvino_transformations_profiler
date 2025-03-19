from collections import namedtuple
import logging
from typing import List, Dict, Iterator
import parse_input
from ov_ts_profiler.common_structs import CSVItem, ModelInfo, ModelData


output_fieldnames = [
    "device",
    "model_path",
    "model_name",
    "model_framework",
    "model_precision",
    "config",
    "iteration",
    "type",
    "transformation_name",
    "manager_name",
    "duration",
    "status",
]


Stats = namedtuple("Stats", ["transformation_name", "manager_name", "duration", "type", "status"])


def create_stats_transformation(ts_name: str, manager_name: str, duration: str, status: str) -> Stats:
    return Stats(ts_name, manager_name, duration, "transformation", status)


def create_stats_manager(name: str, duration: str, status: str) -> Stats:
    return Stats("", name, duration, "manager", status)


def create_stats_manager_start(name: str, m_time: str) -> Stats:
    return Stats("", name, m_time, "manager_start", "")


def create_stats_manager_end(name: str, m_time: str) -> Stats:
    return Stats("", name, m_time, "manager_end", "")


class ResultCollector:
    def __init__(self):
        self.stats: list[Stats] = []

    def __add_transformation_info(self, ts_name: str, manager_name: str, duration: str, status: str) -> None:
        stat = create_stats_transformation(ts_name, manager_name, duration, status)
        self.stats.append(stat)

    def __add_manager_info(self, name: str, duration: str, status: str) -> None:
        stats = create_stats_manager(name, duration, status)
        self.stats.append(stats)

    def __add_manager_start_info(self, name: str, m_time: str):
        stat = create_stats_manager_start(name, m_time)
        self.stats.append(stat)

    def __add_manager_end_info(self, name: str, m_time: str):
        stat = create_stats_manager_end(name, m_time)
        self.stats.append(stat)

    def parse(self, log_path: str) -> None:
        with open(log_path, "r", encoding="utf-8") as f_in:
            csv_reader = csv.reader(f_in, delimiter=";")
            for line in csv_reader:
                if line[0] == "t":
                    self.__add_transformation_info(line[1], line[2], line[3], line[4])
                elif line[0] == "m":
                    self.__add_manager_info(line[1], line[2], line[3])
                elif line[0] == "m_start":
                    self.__add_manager_start_info(line[1], line[2])
                elif line[0] == "m_end":
                    self.__add_manager_end_info(line[1], line[2])
                else:
                    logging.warning(f"unknown line type {line[0]}")


def get_output_items(iter_items: list[Stats]) -> Iterator[Dict]:
    for item in iter_items:
        output_dict = item._asdict()
        output_dict["iteration"] = 1
        output_dict["model_path"] = 'N/A'
        output_dict["model_name"] = 'N/A'
        output_dict["model_framework"] = 'N/A'
        output_dict["model_precision"] = 'N/A'
        output_dict["config"] = 'N/A'
        output_dict["device"] = 'N/A'
        yield output_dict


def parse_job_dump(path: str) -> List[Stats]:
    result_collector = ResultCollector()
    result_collector.parse(path)
    return result_collector.stats


def read_csv(path: str) -> Iterator[CSVItem]:
    parse_input.check_header(output_fieldnames)

    stats = parse_job_dump(path)
    dict_items = get_output_items(stats)
    list_items = [item for item in dict_items for name in output_fieldnames]

    has_config = 'optional_model_attribute' in output_fieldnames or \
                 'weight_compression' in output_fieldnames or \
                 'config' in output_fieldnames
    has_device = 'device' in output_fieldnames
    has_status = 'status' in output_fieldnames
    for row in list_items:
        if not row:
            continue
        if not has_device:
            row.insert(0, 'N/A')
        if not has_config:
            row.insert(5, '')
        if not has_status:
            row.append('N/A')
        if not row[5]:
            row[5] = 'N/A'
        try:
            csv_item = CSVItem(*row)
            # if it's header inside CSV file
            if csv_item.iteration == 'iteration':
                continue
        except:
            print(f'exception in row {row}')
            raise
        yield csv_item


def get_csv_data(csv_paths: List[str]) -> List[Dict[ModelInfo, ModelData]]:
    csv_data = []
    for i in range(len(csv_paths)):
        csv_path = csv_paths[i]
        print(f'reading {csv_path} ... - it will be value#{i + 1}')
        csv_rows = read_csv(csv_path)
        current_csv_data = parse_input.read_csv_data(csv_rows)
        current_csv_data = parse_input.remove_invalid_items(current_csv_data)
        if current_csv_data:
            csv_data.append(current_csv_data)
        dt = parse_input.get_measurement_date(current_csv_data)
        if dt is not None:
            print(f'  measurement date: {dt}')
    parse_input.check_csv_data(csv_data)
    return csv_data
