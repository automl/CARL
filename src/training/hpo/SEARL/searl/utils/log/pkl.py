"""
save log and show log

"""
import os
import pickle
import time
from pathlib import Path

from ..handler.base_handler import Handler


class LogPKL(Handler):

    def __init__(self, log_dir, file_name="value_dump.pkl"):
        super().__init__()

        self.log_dir = Path(log_dir)

        self.pickle_file = file_name

    def dump(self, key: str, value=None, time_step=None):
        if value:
            data = {"key": key, 'value': value, 'time_step': time_step, 'time_stamp': self.time_stamp(),
                    'time': time.time()}
        else:
            data = key
        with open(self.log_dir / self.pickle_file, 'ab') as f:
            pickle.dump(data, f)

    def check_dump(self):
        return os.path.isfile(self.log_dir / self.pickle_file)

    def load_pickle(self):
        data = []
        with open(self.log_dir / self.pickle_file, 'rb') as f:
            while True:
                try:
                    data.append(pickle.load(f))
                except EOFError:
                    break
        return data
