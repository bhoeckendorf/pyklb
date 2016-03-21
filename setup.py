#!/usr/bin/env python

import os
import numpy as np
import urllib
import platform
from Cython.Build import cythonize
try:
    from setuptools import setup
    from setuptools import Extension
except ImportError:
    from distutils.core import setup
    from distutils.extension import Extension


# version (by commit id) of main library to use
klbCommitId = "5edcaecc858911c7b3855579bde5cb3116cb4680"


# download required KLB headers,
includeDirs = ["build/include"]
libraryDirs = ["build/lib"]
klbUrl = "https://bitbucket.org/fernandoamat/keller-lab-block-filetype/raw/%s" % klbCommitId

downloadFiles = [
    # collect downloads, in format (sourceFileUrl, targetDir)
    ("%s/src/common.h" % klbUrl, includeDirs[0]),
    ("%s/src/klb_Cwrapper.h" % klbUrl, includeDirs[0])
    ]


# download main library dependency
errorMsg = """

    *************************************************************************************
    * No precompiled binary of main KLB library available.                              *
    * Please download main KLB library source code from the link below and build it.    *
    * https://bitbucket.org/fernandoamat/keller-lab-block-filetype/get/%s.zip *
    *************************************************************************************

    """ % klbCommitId
platformName = platform.uname()[0].lower()
if platform.architecture()[0].startswith("64"):
    if "linux" in platformName:
        downloadFiles.append(( "%s/bin/libklb.so" % klbUrl, "pyklb/lib" ))
    elif "darwin" in platformName:
        downloadFiles.append(( "%s/bin/libklb.dylib" % klbUrl, "pyklb/lib" ))
    elif "win" in platformName:
        downloadFiles.append(( "%s/bin/klb.dll" % klbUrl, "pyklb/lib" ))
        downloadFiles.append(( "%s/bin/klb.lib" % klbUrl, libraryDirs[0] ))
        # fix windows build with msvc
        downloadFiles.append(( "http://msinttypes.googlecode.com/svn/trunk/stdint.h", includeDirs[0] ))
    else:
        print(errorMsg)
else:
    print(errorMsg)


# download
for (source, targetDir) in downloadFiles:
    if not os.path.exists(targetDir):
        os.makedirs(targetDir)
    target = os.path.join(targetDir, os.path.split(source)[1])
    urllib.urlretrieve( source, target )


includeDirs.append( np.get_include() )
setup(
    name = "pyklb",
    version = "0.0.1.dev1",
    description = "Python wrapper of the KLB file format, a high-performance file format for up to 5-dimensional arrays.",
    long_description = "See https://bitbucket.org/fernandoamat/keller-lab-block-filetype",
    url = "https://github.com/bhoeckendorf/pyklb",
    ext_modules = cythonize([
        Extension("pyklb", ["pyklb/pyklb.pyx"], include_dirs=includeDirs, library_dirs=libraryDirs + ["pyklb/lib"], libraries=["klb"])
        ]),
    setup_requires = ["numpy"],
    install_requires = ["numpy"]
)
