from typing import Union, Optional, List
import json

from ConfigSpace import ConfigurationSpace
from omegaconf import OmegaConf, DictConfig, ListConfig
from ConfigSpace.read_and_write import json as csjson


class JSONCfgEncoder(json.JSONEncoder):
    """Encode DictConfigs.

    Convert DictConfigs to normal dicts.
    """

    def default(self, obj):
        if isinstance(obj, DictConfig):
            return dict(obj)
        elif isinstance(obj, ListConfig):
            parsed_list = []
            for o in obj:
                if type(o) == DictConfig:
                    o = dict(o)
                elif type(o) == ListConfig:
                    o = list(o)
                parsed_list.append(o)

            return parsed_list  # [dict(o) for o in obj]
        return json.JSONEncoder.default(self, obj)


def search_space_to_config_space(search_space: Union[str, DictConfig], seed: Optional[int] = None) -> ConfigurationSpace:
    """
    Convert hydra search space to SMAC's configuration space.

    See the [ConfigSpace docs]() for information of how to define a configuration (search) space.

    In a yaml (hydra) config file, the smac.search space must take the form of:

    search_space:
        hyperparameters:
            hyperparameter_name_0:
                key1: value1
                ...
            hyperparameter_name_1:
                key1: value1
                key2: value2
                ...


    Parameters
    ----------
    search_space : Union[str, DictConfig]
        The search space, either a DictConfig from a hydra yaml config file, or a path to a json configuration space
        file in the format required of ConfigSpace.
    seed : Optional[int]
        Optional seed to seed configuration space.


    Example of a json-serialized ConfigurationSpace file.
    {
      "hyperparameters": [
        {
          "name": "x0",
          "type": "uniform_float",
          "log": false,
          "lower": -512.0,
          "upper": 512.0,
          "default": -3.0
        },
        {
          "name": "x1",
          "type": "uniform_float",
          "log": false,
          "lower": -512.0,
          "upper": 512.0,
          "default": -4.0
        }
      ],
      "conditions": [],
      "forbiddens": [],
      "python_module_version": "0.4.17",
      "json_format_version": 0.2
    }


    Returns
    -------
    ConfigurationSpace
    """
    if type(search_space) == str:
        with open(search_space, 'r') as f:
            jason_string = f.read()
    elif type(search_space) == DictConfig:
        search_space = OmegaConf.to_container(search_space, resolve=True)
        # reorder hyperparameters as List[Dict]
        hyperparameters = []
        for name, cfg in search_space["hyperparameters"].items():
            cfg["name"] = name
            if "default" not in cfg:
                cfg["default"] = None
            if "log" not in cfg:
                cfg["log"] = False
            hyperparameters.append(cfg)
        search_space["hyperparameters"] = hyperparameters

        if "conditions" not in search_space:
            search_space["conditions"] = []
        
        if "forbiddens" not in search_space:
            search_space["forbiddens"] = []
        

        jason_string = json.dumps(search_space, cls=JSONCfgEncoder)

    cs = csjson.read(jason_string)
    if seed is not None:
        cs.seed(seed=seed)
    return cs
