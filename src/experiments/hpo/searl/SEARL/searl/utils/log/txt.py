"""
save log and show log

"""
import os
import sys
from pathlib import Path

from ..handler.base_handler import Handler


class LogTXT(Handler):

    def __init__(self, log_dir, file_name="log_file.txt"):
        super().__init__()

        self.log_dir = Path(log_dir)
        self.log_file = file_name

        self.start_log()

    def start_log(self):
        if os.path.isfile(self.log_dir / self.log_file) and os.access(self.log_dir / self.log_file, os.R_OK):
            self.log("LOGGER: continue logging")
        else:
            with open(self.log_dir / self.log_file, 'w+') as file:
                file.write(
                    f"{self.time_stamp()} LOGGER: start logging with Python version: {str(sys.version).split('(')[0]} \n")

    def log(self, string: str):
        timed_string = f"{self.time_stamp()} {string}"
        print(timed_string)
        with open(self.log_dir / self.log_file, 'a') as file:
            file.write(f"{timed_string} \n")
