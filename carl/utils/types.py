from typing import Any, Dict, Union, List, TypeVar
import numpy as np

Context = Dict[str, Any]
Vector = Union[List[Any], np.typing.NDArray]
ObsType = TypeVar("ObsType")
