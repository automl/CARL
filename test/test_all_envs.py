import unittest
import numpy as np

from carl.envs import *


class TestInitEnvs(unittest.TestCase):
    def test_init_all_envs(self):
        global_vars = globals().copy()
        mustinclude = "CARL"
        forbidden = ["defaults", "bounds"]
        for varname, var in global_vars.items():
            if mustinclude in varname and not np.any([f in varname for f in forbidden]):
                try:
                    env = var()
                except Exception as e:
                    print(f"Cannot instantiate {var} environment.")
                    raise e


