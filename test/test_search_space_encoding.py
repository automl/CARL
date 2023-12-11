import unittest

from ConfigSpace import ConfigurationSpace
from omegaconf import DictConfig

from carl.context.search_space_encoding import search_space_to_config_space


class TestSearchSpacEncoding(unittest.TestCase):
    def setUp(self):
        self.test_space = None
        self.test_space = ConfigurationSpace(
            name="myspace",
            space={
                "uniform_integer": (1, 10),
                "uniform_float": (1.0, 10.0),
                "categorical": ["a", "b", "c"],
                "constant": 1337,
            },
        )
        return super().setUp()

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
