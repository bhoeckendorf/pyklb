# pyklb

Python wrapper of the [KLB file format](https://bitbucket.org/fernandoamat/keller-lab-block-filetype), a high-performance file format for up to 5-dimensional arrays. For more details, see https://bitbucket.org/fernandoamat/keller-lab-block-filetype

## Build

Dependencies

- [NumPy](http://www.numpy.org/)
- [Cython](http://cython.org/)
- C compiler, see [this link](https://github.com/cython/cython/wiki/CythonExtensionsOnWindows) if compiling for Windows with MSVC
- [KLB main library](https://bitbucket.org/fernandoamat/keller-lab-block-filetype),  precompiled binaries (64-bit only) are available for Linux (>= Ubuntu LTS), Windows, and Mac, and will be automatically downloaded, if appropriate. Other platforms require building the main library separately from source. KLB's main library is self-contained and uses CMake. Afterwards, place the binary in <code>./pyklb/lib/</code>. On Windows, additionally place the <code>klb.lib</code> in <code>./build/lib/</code> (both paths are relative to the top level folder of this repository)

To build, execute <code>python setup.py build_ext --inplace</code> from the top level folder of this repository.