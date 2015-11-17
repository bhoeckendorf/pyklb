#!/usr/bin/env python

import os
import numpy
from Cython.Build import cythonize
from distutils.core import setup, Extension

with open(os.path.abspath(os.path.join("..", "README.md")), "r") as f:
    readme = f.read()

setup(
    name = "pyklb",
    version = "0.0.1.dev",
    description = readme.split("#")[2].split(".")[0].strip(), # from readme, use first sentence after header
    long_description = readme,
    url = "https://bitbucket.org/fernandoamat/keller-lab-block-filetype/",
    ext_modules = cythonize([
        Extension("pyklb", ["pyklb.pyx"], libraries=["klb"])
        ])
    #py_modules = ["pyklb"],
    #setup_requires = ["numpy"],
    #install_requires = ["numpy"]
)
