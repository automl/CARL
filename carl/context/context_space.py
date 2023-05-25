from typing import List, Union, Any, Dict, Tuple
from numpy.typing import NDArray
import numpy as np

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import Hyperparameter, UniformFloatHyperparameter
import gymnasium.spaces as spaces


class ContextSpace(ConfigurationSpace):
    """
    Baseclass for contexts of environments implemented in CARL. DO NOT USE DIRECTLY. USE SUBCLASSES!
    """
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

        self.__context_features: List[str] = []
        self.__context_space: List[Hyperparameter] = []
        self.__default_context: Dict[str, Union[int, float]] = {}

    def __build_context(self):
        """
        Initializes the context based on its context space and features.
        """
        for parameter in self.__context_space:
            if parameter.name in self.__context_features:
                # add context feature as hyperparameter to enable sampling
                self.add_hyperparameter(parameter)
            else:
                # set lower = upper = default to simulate non-varying context features
                self.add_hyperparameter(UniformFloatHyperparameter(
                    parameter.name,
                    lower=parameter.default_value,
                    upper=parameter.default_value
                ))

    @property
    def gym_space(self, as_dict: bool = False) -> spaces.Space:
        if as_dict:
            context_space = {}

            for parameter in self.__context_space:
                context_space[parameter.name] = spaces.Box(
                    low=parameter.lower,
                    high=parameter.upper
                )
            return spaces.Dict(context_space)
        else:
            low = np.array([param.lower for param in self.__context_space])
            high = np.array([param.upper for param in self.__context_space])

            return spaces.Box(low=low, high=high, dtype=np.float32)

    @property
    def context_features(self) -> List[str]:
        """
        Context features that are sampled.

        Returns
        -------
        List[str]
            Context features.
        """
        return self.__context_features

    @property
    def default_context(self) -> NDArray:
        """
        Default context of CARL environment.

        Returns
        -------
        NDArray
            Default context values.
        """
        return np.array([default_value for _, default_value in self.__context_space])


class CartPoleContextSpace(ContextSpace):
    """
    Context space for CARLCartPoleEnv.

    Parameters
    ----------
    context_features : List[str]
        Context features to vary, i.e. sample.

    seed : int = None
        Seed for PRNG.
    """
    def __init__(self, context_features: List[str], seed: int = None):
        super().__init__(name=self.__class__.__name__, seed=seed)

        self.__context_features = context_features

        # -----------------------------------------------------------------------------------------
        # ENVIRONMENT SPECIFIC CONTEXT PARAMETERS AND DEFAULT VALUES
        # -----------------------------------------------------------------------------------------
        self.__context = [
            UniformFloatHyperparameter("gravity", lower=0.1, upper=np.inf, default_value=9.8),
            UniformFloatHyperparameter("masscart", lower=0.1, upper=10, default_value=1.0),
            UniformFloatHyperparameter("masspole", lower=0.01, upper=1, default_value=0.1),
            UniformFloatHyperparameter("pole_length", lower=0.05, upper=5, default_value=0.5),
            UniformFloatHyperparameter("force_magnifier", lower=1, upper=100, default_value=10.0),
            UniformFloatHyperparameter("update_interval", lower=0.002, upper=0.2, default_value=0.02),
            UniformFloatHyperparameter("initial_state_lower", lower=-np.inf, upper=np.inf, default_value=-0.1),
            UniformFloatHyperparameter("initial_state_upper", lower=-np.inf, upper=np.inf, default_value=0.1)
        ]
        # -----------------------------------------------------------------------------------------

        self.__build_context()

