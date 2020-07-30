#!/usr/bin/env python3

import sys
from skbuild import setup

setup(
    name = "pyklb",
    version = "0.0.3.dev0",
    description = "Python wrapper of the KLB file format, a high-performance file format for up to 5-dimensional arrays.",
    long_description = "See https://bitbucket.org/fernandoamat/keller-lab-block-filetype",
    url = "https://github.com/bhoeckendorf/pyklb",
    packages=["pyklb"],
    setup_requires = ["numpy"],
    install_requires = ["cython", "numpy"]
)
