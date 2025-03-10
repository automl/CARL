"""Setup file."""
from __future__ import annotations

import json
import os
from setuptools import find_packages, setup


def get_other_requirements():
    """Get other requirements."""
    other_requirements = {}
    for file in os.listdir("./other_requirements"):
        with open(f"./other_requirements/{file}", encoding="utf-8") as rq:
            requirements = json.load(rq)
            other_requirements.update(requirements)
            return other_requirements
    return None


setup(
    version="1.1.0",
    packages=find_packages(
        exclude=[
            "tests",
            "examples",
        ]
    ),
)