"""
save log and show log

"""
import json
import os
import time
from pathlib import Path
import numpy as np

from ..handler.base_handler import Handler


class NpEncoder(json.JSONEncoder):
    """Custom json.JSONEncoder.

    Able to decode ``np.integer``, ``np.floating`` and ``np.ndarray`` types.

    Credits: https://stackoverflow.com/a/57915246/13080859
    """

    def default(self, obj):
        # first half copied from NumpyEncoder
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):

            return int(obj)

        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)

        elif isinstance(obj, (np.complex_, np.complex64, np.complex128)):
            return {'real': obj.real, 'imag': obj.imag}

        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()

        elif isinstance(obj, (np.bool_)):
            return bool(obj)

        elif isinstance(obj, (np.void)):
            return None

        elif isinstance(obj, Path):
            return str(obj)

        return json.JSONEncoder.default(self, obj)


class LogJSON(Handler):

    def __init__(self, log_dir, file_name="json_log.json"):
        super().__init__()

        self.log_dir = Path(log_dir)

        self.file_name = file_name
        self.json_file = file_name

    def __enter__(self):
        self.open()
        return self

    def open(self):
        self.json_file = self.counting_name(self.log_dir, self.json_file, suffix=True)

        data = {"start": {'value': 0, 'time_step': None, 'time_stamp': self.time_stamp(), 'time': time.time()}}
        with open(self.log_dir / self.json_file, 'w+') as file:
            file.write(f"[ \n")
            file.write(json.dumps(data))

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        with open(self.log_dir / self.json_file, 'a') as file:
            file.write(f"\n]")

    def jlog(self, key: str, value, time_step=None):
        data = {key: {'value': value, 'time_step': time_step, 'time_stamp': self.time_stamp(), 'time': time.time()}}
        with open(self.log_dir / self.json_file, 'a') as file:
            file.write(", \n")
            file.write(json.dumps(data, cls=NpEncoder))

    def load_json(self):
        data_list = []
        counter = 0
        counting_file_name = self.file_name.split('.')[0] + f"-{counter}" + self.file_name.split('.')[1]
        while os.path.isfile(self.log_dir / counting_file_name):
            with open(self.log_dir / counting_file_name, 'r') as file:
                data = json.load(file)
            data_list.append(data)

            counter += 1
            counting_file_name = self.file_name.split('.')[0] + f"-{counter}" + self.file_name.split('.')[1]

        return data_list
