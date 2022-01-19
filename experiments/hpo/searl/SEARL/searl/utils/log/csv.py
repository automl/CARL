"""
save log and show log

"""
import csv
from pathlib import Path
from typing import Dict, List

from ..handler.base_handler import Handler


class LogCSV(Handler):

    def __init__(self, log_dir, file_name="train_log.csv"):
        super().__init__()

        self.log_dir = Path(log_dir)
        self.log_file = file_name

    def fieldnames(self, fieldnames_list: List):
        self.csv_columns = fieldnames_list

        with open(self.log_dir / self.log_file, 'a') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.csv_columns)
            writer.writeheader()

    def log_csv(self, dict_data: Dict):

        dict_data["time_string"] = f"{self.time_stamp()}"

        for key in self.csv_columns:
            if key not in dict_data.keys():
                dict_data[key] = None
        with open(self.log_dir / self.log_file, 'a') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.csv_columns)
            writer.writerow(dict_data)
