import unittest

import numpy as np

import carl.envs


class TestInitEnvs(unittest.TestCase):
    def test_init_all_envs(self):
        global_vars = vars(carl.envs)
        mustinclude = "CARL"
        forbidden = ["defaults", "bounds", "mask"]
        for varname, var in global_vars.items():
            if mustinclude in varname and not np.any([f in varname for f in forbidden]):
                try:
                    env = (  # noqa: F841 local variable is assigned to but never used
                        var()
                    )
                except Exception as e:
                    print(f"Cannot instantiate {var} environment.")
                    raise e
