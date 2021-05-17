# pyklb

Python wrapper of the [KLB file format](https://bitbucket.org/fernandoamat/keller-lab-block-filetype), a high-performance file format for up to 5-dimensional arrays. For more details, see https://bitbucket.org/fernandoamat/keller-lab-block-filetype

## Installation

`pip install git+https://github.com/bhoeckendorf/pyklb.git@skbuild`

## Build

Dependencies

- [NumPy](http://www.numpy.org/)
- [Cython](http://cython.org/)
- C compiler, see [this link](https://github.com/cython/cython/wiki/CythonExtensionsOnWindows) if compiling for Windows with MSVC
- [KLB main library](https://bitbucket.org/fernandoamat/keller-lab-block-filetype),  precompiled binaries (64-bit only) are available for Linux (>= Ubuntu LTS), Windows, and Mac, and will be automatically downloaded, if appropriate. Other platforms require building the main library separately from source. KLB's main library is self-contained and uses CMake. Afterwards, place the binary in <code>./build/lib/</code> (relative to the top level folder of this repository). On Windows, additionally place the <code>klb.lib</code> in the same folder.

The recommended build and installation method is via [Wheels](http://pythonwheels.com/). If it isn't already installed, you have to add the <code>wheel</code> package by running <code>pip install wheel</code>. You can then build pyklb by executing <code>python setup.py bdist_wheel</code> from the top level folder of this repository. This should create a <code>dist</code> subfolder that contains the resulting [wheel](http://pythonwheels.com/). To install, run <code>pip install /path/to/wheel.whl</code>. Conversely, you can uninstall pyklb using <code>pip uninstall pyklb</code>.
