from typing import Any, Dict, Union, List, TypeVar
import numpy as np

Context = Dict[str, Any]
Contexts = Dict[Any, Context]
Vector = Union[List[Any], np.ndarray]
ObsType = TypeVar("ObsType")
