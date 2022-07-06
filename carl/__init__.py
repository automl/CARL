__license__ = "Apache-2.0 License"
__version__ = "0.2.0"
__author__ = "Carolin Benjamins, Theresa Eimer, Frederik Schubert, André Biedenkapp, Aditya Mohan, Sebastian Döhler"


import datetime
import os
import sys
import warnings

name = "CARL"
package_name = "carl"
author = __author__

author_email = "benjamins@tnt.uni-hannover.de"
description = "CARL- Contextually Adaptive Reinforcement Learning"
url = "https://www.automl.org/"
project_urls = {
    "Documentation": "https://carl.readthedocs.io/en/latest/",
    "Source Code": "https://github.com/https://github.com/automl/CARL",
}
copyright = f"""
    Copyright {datetime.date.today().strftime('%Y')}, AutoML.org Freiburg-Hannover
"""
version = __version__


if os.name != "posix":
    warnings.warn(
        f"Detected unsupported operating system: {sys.platform}."
        "Please be aware, that SMAC might not run on this system."
    )