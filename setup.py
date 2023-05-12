import os

import setuptools

from carl import (
    author,
    author_email,
    description,
    package_name,
    project_urls,
    url,
    version,
)

HERE = os.path.dirname(os.path.realpath(__file__))


def read_file(filepath: str) -> str:
    with open(filepath, "r", encoding="utf-8") as fh:
        return fh.read()


extras_require = {
    "box2d": [
        "gym[box2d]==0.24.1",
    ],
    "brax": [
        "brax>=0.0.10,<=0.0.16",
        "protobuf>=3.17.3",
    ],
    "dm_control": [
        "dm_control>=1.0.3",
    ],
    "mario": [
        "torch",
        "py4j",
        "Pillow",
        "opencv-python",
        "jdk4py",
        "wandb", 
        "pyvirtualdisplay", 
        "hydra-core", 
        "hydra-submitit-launcher", 
        "hydra_colorlog", 
        "torchinfo",
        "tqdm"
    ],
    "dev": [
        "pytest>=6.1.1",
        "pytest-cov",
        "mypy",
        "black",
        "flake8",
        "isort",
        "pydocstyle",
        "pre-commit",
    ],
    "docs": [
        "sphinx>=4.2.0",
        "sphinx-gallery>=0.10.0",
        "image>=1.5.33",
        "sphinx-autoapi>=1.8.4",
        "gym[box2d]==0.24.1",
        "brax>=0.0.10",
        "protobuf>=3.17.3",
        "dm_control>=1.0.3",
        "torch>=1.9.0",
        "Pillow>=8.3.1",
        "py4j>=0.10.9.2"
    ]
}

setuptools.setup(
    name=package_name,
    author=author,
    author_email=author_email,
    description=description,
    long_description=read_file(os.path.join(HERE, "README.md")),
    long_description_content_type="text/markdown",
    license="Apache 2.0",
    license_file="LICENSE",
    url=url,
    project_urls=project_urls,
    keywords=[
        "RL",
        "Generalization",
        "Context",
        "Reinforcement Learning"
    ],
    version=version,
    packages=setuptools.find_packages(exclude=["tests"]),
    include_package_data=True,
    python_requires=">=3.9",
    install_requires=[
        "gym==0.24.1",
        "scipy",
        "ConfigArgParse",
        "numpy==1.20.3",
        "pandas",
        "xvfbwrapper",
        "matplotlib",
        "dataclasses",
        "numpyencoder",
        "pyglet",
        "pytablewriter",
        "PyYAML",
        "tabulate",
        "bs4",
    ],
    extras_require=extras_require,
    test_suite="pytest",
    platforms=["Linux"],
    entry_points={
        "console_scripts": ["smac = smac.smac_cli:cmd_line_call"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Natural Language :: English",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: POSIX :: Linux",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development",
    ],
)
