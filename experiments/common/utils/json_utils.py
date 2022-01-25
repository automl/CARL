# -*- coding: utf-8 -*-
import numpy as np
import json
import os
from pathlib import Path
import inspect
from numpyencoder import NumpyEncoder


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

        elif inspect.isclass(obj):
            return obj.__name__

        elif inspect.isfunction(obj):
            return obj.__name__

        return json.JSONEncoder.default(self, obj)


class NpDecoder(json.JSONDecoder):
    pass


def lazy_json_dump(data, filename):
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    with open(filename, "w") as file:
        json.dump(data, file, cls=NpEncoder, indent="\t")


def lazy_json_load(filename):
    with open(filename, "r") as file:
        data = json.load(file, cls=NpDecoder)

    return data
