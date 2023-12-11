__license__ = "Apache-2.0 License"
__version__ = "1.0.0"
__author__ = "Carolin Benjamins, Theresa Eimer, Frederik Schubert, André Biedenkapp, Aditya Mohan, Sebastian Döhler"


import datetime
import importlib.util as iutil
import warnings

from gymnasium.envs.registration import register

from carl import envs

name = "CARL"
package_name = "carl-bench"
author = __author__

author_email = "benjamins@tnt.uni-hannover.de"
description = "CARL- Contextually Adaptive Reinforcement Learning"
url = "https://www.automl.org/"
project_urls = {
    "Documentation": "https://automl.github.io/CARL",
    "Source Code": "https://github.com/https://github.com/automl/CARL",
}
copyright = f"""
    Copyright {datetime.date.today().strftime('%Y')}, AutoML.org Freiburg-Hannover
"""
version = __version__


for e in envs.gymnasium.classic_control.__all__:
    register(
        id=f"carl/{e}-v0",
        entry_point=f"carl.envs.gymnasium.classic_control:{e}",
    )


def check_spec(spec_name: str) -> bool:
    """Check if the spec is installed

    Parameters
    ----------
    spec_name : str
        Name of package that is necessary for the environment suite.

    Returns
    -------
    bool
        Whether the spec was found.
    """
    spec = iutil.find_spec(spec_name)
    found = spec is not None
    if not found:
        with warnings.catch_warnings():
            warnings.simplefilter("once")
            warnings.warn(
                f"""Module {spec_name} not found. If you want to use these environments,
                please follow the installation guide."""
            )
    return found


# Environment loading
found = check_spec("Box2D")
if found:
    for e in envs.gymnasium.box2d.__all__:
        register(
            id=f"carl/{e}-v0",
            entry_point=f"carl.envs.gymnasium.box2d:{e}",
        )

found = check_spec("py4j")
if found:
    register(
        id="carl/CARLMario-v0",
        entry_point="carl.envs.gymnasium:CARLMarioEnv",
    )
