import unittest

from ConfigSpace import ConfigurationSpace
from omegaconf import DictConfig

from carl.context.search_space_encoding import search_space_to_config_space

dict_space = {
    "uniform_integer": (1, 10),
    "uniform_float": (1.0, 10.0),
    "categorical": ["a", "b", "c"],
    "constant": 1337,
}

dict_space_2 = {
    "hyperparameters": [
        {
            "name": "x0",
            "type": "uniform_float",
            "log": False,
            "lower": -512.0,
            "upper": 512.0,
            "default": -3.0,
        },
        {
            "name": "x1",
            "type": "uniform_float",
            "log": False,
            "lower": -512.0,
            "upper": 512.0,
            "default": -4.0,
        },
    ],
    "conditions": [],
    "forbiddens": [],
    "python_module_version": "0.4.17",
    "json_format_version": 0.2,
}

str_space = """{
                "uniform_integer": (1, 10),
                "uniform_float": (1.0, 10.0),
                "categorical": ["a", "b", "c"],
                "constant": 1337,
            }"""


class TestSearchSpacEncoding(unittest.TestCase):
    def setUp(self):
        self.test_space = None
        self.test_space = ConfigurationSpace(name="myspace", space=dict_space)
        return super().setUp()

    def test_init(self):
        self.test_space = ConfigurationSpace(name="myspace", space=dict_space_2)

        self.test_space = ConfigurationSpace(name="myspace", space=str_space)

    def test_config_spaces(self):
        try:
            search_space_to_config_space(self.test_space)
        except Exception as e:
            print(f"Cannot encode search space --  {self.test_space}.")
            raise e

    def test_dict_configs(self):
        try:
            dict_space = DictConfig({"hyperparameters": {}})

            search_space_to_config_space(dict_space)
        except Exception as e:
            print(f"Cannot encode search space --  {dict_space}.")
            raise e


if __name__ == "__main__":
    unittest.main()
