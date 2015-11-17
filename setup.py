#!/usr/bin/env python

import os
import numpy
import urllib
import platform
from Cython.Build import cythonize
from distutils.core import setup, Extension


# version (by commit id) of main library to use
klbCommitId = "75c5eb72f91a"


# downlad required headers
downloadFiles = [
    ("src/common.h", "include/common.h"),
    ("src/klb_Cwrapper.h", "include/klb_Cwrapper.h")
    ]
for (source, target) in downloadFiles:
    targetDir = os.path.abspath(
        os.path.join("build", os.path.split(target)[0]) )
    if not os.path.exists(targetDir):
        os.makedirs(targetDir)
    urllib.urlretrieve(
        "https://bitbucket.org/fernandoamat/keller-lab-block-filetype/raw/%s/%s" % (klbCommitId, source),
        "build/%s" % target)


# download main library dependency
errorMsg = """

    *****************************************************************************++******
    * No precompiled binary of main KLB library available.                              *
    * Please download main KLB library source code from the link below and build it.    *
    * https://bitbucket.org/fernandoamat/keller-lab-block-filetype/get/%s.zip *
    *****************************************************************************++******

    """ % klbCommitId
downloadLib = None
if platform.architecture()[0].startswith("64"):
    platformName = platform.uname()[0].lower()
    if "linux" in platformName:
        downloadLib = "libklb.so"
    elif "win" in platformName:
        downloadLib = "klb.dll"
    elif "mac" in platformName:
        downloadLib = "libklb.dylib"
if downloadLib == None:
    print(errorMsg)
else:
    urllib.urlretrieve(
        "https://bitbucket.org/fernandoamat/keller-lab-block-filetype/raw/%s/bin/%s" % (klbCommitId, downloadLib),
        "pyklb/lib/%s" % downloadLib)

setup(
    name = "pyklb",
    version = "0.0.1.dev",
    description = "Python wrapper of the KLB file format, a high-performance file format for up to 5-dimensional arrays.",
    long_description = "See https://bitbucket.org/fernandoamat/keller-lab-block-filetype",
    url = "https://github.com/bhoeckendorf/pyklb",
    ext_modules = cythonize([
        Extension("pyklb", ["pyklb/pyklb.pyx"], libraries=["klb"])
        ]),
    setup_requires = ["numpy"],
    install_requires = ["numpy"]
)
