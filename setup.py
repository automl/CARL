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
        "gymnasium[box2d]>=0.27.1",
    ],
    "brax": [
        "brax>=0.0.10,<=0.0.16",
        "protobuf>=3.17.3",
    ],
    "dm_control": [
        "dm_control>=1.0.3",
    ],
    "gymnax": [
        "gymnax>=0.0.6",
    ],
    "mario": [
        "torch>=1.9.0",
        "Pillow>=8.3.1",
        "py4j>=0.10.9.2",
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
        "gymnasium>=0.27.1",
        "scipy>=1.7.0",
        "ConfigArgParse>=1.5.1",
        "numpy>=1.19.5",
        "pandas>=1.3.0",
        "xvfbwrapper>=0.2.9",
        "matplotlib>=3.4.2",
        "dataclasses>=0.6",
        "numpyencoder>=0.3.0",
        "pyglet>=1.5.15",
        "pytablewriter>=0.62.0",
        "PyYAML>=5.4.1",
        "tabulate>=0.8.9",
        "bs4>=0.0.1",
    ],
    extras_require=extras_require,
    test_suite="pytest",
    platforms=["Linux"],
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
