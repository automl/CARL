#!/usr/bin/env python

from distutils.core import setup

req_file = "requirements.txt"

def parse_requirements(filename):
    lineiter = (line.strip() for line in open(filename))
    return [line for line in lineiter if line and not line.startswith("#")]

install_reqs = parse_requirements(req_file)

setup(name='searl',
      version='latest',
      install_requires=install_reqs,
      dependency_links=[],
      )
