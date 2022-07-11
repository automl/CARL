from typing import Any, Dict, List, TypeVar, Union

import numpy as np

Context = Dict[str, Any]
Contexts = Dict[Any, Context]
Vector = Union[List[Any], np.ndarray]
ObsType = TypeVar("ObsType")
