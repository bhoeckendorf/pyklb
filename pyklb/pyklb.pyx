#!python
#cython: initializedcheck=False, boundscheck=False, overflowcheck=False

import cython
cimport cython
from libc.stdint cimport uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t
from cpython cimport bool
import numpy as _np
cimport numpy as _np
import multiprocessing as _mpc



###########################################################
# Declarations from C headers                             #
###########################################################


cdef extern from "common.h":
    ctypedef float  float32_t
    ctypedef double float64_t
    cdef enum KLB_DATA_TYPE:
        UINT8_TYPE = 0
        UINT16_TYPE = 1
        UINT32_TYPE = 2
        UINT64_TYPE = 3
        INT8_TYPE = 4
        INT16_TYPE = 5
        INT32_TYPE = 6
        INT64_TYPE = 7
        FLOAT32_TYPE = 8
        FLOAT64_TYPE = 9
    cdef enum KLB_COMPRESSION_TYPE:
        NONE = 0
        BZIP2 = 1
        ZLIB = 2


cdef extern from "klb_Cwrapper.h":
    cdef int readKLBheader(const char* filename, uint32_t xyzct[5], KLB_DATA_TYPE *dataType, float32_t pixelSize[5], uint32_t blockSize[5], KLB_COMPRESSION_TYPE *compressionType, char metadata[256])
    cdef int readKLBstackInPlace(const char* filename, void* im, KLB_DATA_TYPE *dataType, int numThreads)
    cdef int readKLBroiInPlace(const char* filename, void* im, uint32_t xyzctLB[5], uint32_t xyzctUB[5], int numThreads)
    cdef int writeKLBstack(const void* im, const char* filename, uint32_t xyzct[5], KLB_DATA_TYPE dataType, int numThreads, float32_t pixelSize[5], uint32_t blockSize[5], KLB_COMPRESSION_TYPE compressionType, char metadata[256])



###########################################################
# Reading KLB files, allocating memory as needed          #
###########################################################


def readheader(
    str filepath
    ):
    """
    Read header of KLB file

    Arguments:
    ----------
    filepath : string
        File system path to KLB file

    Returns
    -------
    header : Dict
        Key-value dict of header fields,
        incl. 'imagesize_tczyx', 'datatype', 'pixelspacing_tczyx'

    Raises
    ------
    IOError
    """
    cdef _np.ndarray[_np.uint32_t, ndim=1] imagesize = _np.empty((5,), _np.uint32)
    cdef _np.ndarray[_np.uint32_t, ndim=1] blocksize = _np.empty((5,), _np.uint32)
    cdef _np.ndarray[_np.float32_t, ndim=1] pixelspacing = _np.empty((5,), _np.float32)
    cdef KLB_DATA_TYPE ktype = UINT8_TYPE
    cdef KLB_COMPRESSION_TYPE kcompression = NONE
    cdef _np.ndarray[_np.int8_t, ndim=1] metadata = _np.empty((256,), _np.int8)
    cdef int errid = readKLBheader(filepath, &imagesize[0], &ktype, &pixelspacing[0], &blocksize[0], &kcompression, <char*> &metadata[0])
    if errid != 0:
        raise IOError("Could not read KLB header of file '%s'. Error code %d" % (filepath, errid))

    return {
        "imagesize_tczyx": _np.flipud(imagesize),
        "blocksize_tczyx": _np.flipud(blocksize),
        "pixelspacing_tczyx": _np.flipud(pixelspacing),
        "metadata": metadata,
        "datatype": _pytype(ktype),
        "compression": _pycompression(kcompression)
        }



def readfull(
    str filepath,
    const int numthreads = _mpc.cpu_count()
    ):
    """
    Read entire array from KLB file

    Arguments
    ---------
    filepath : string
        File system path to KLB file
    numthreads : int, optional, default = multiprocessing.cpu_count()
        Number of threads to use for decompression

    Returns
    -------
    A : array, shape([t,c,z,y,]x)
        The entire array stored in the KLB file, note dimension order.
        Leading singleton dimensions are dropped, but the order is
        preserved to distinguish e.g. xyt (shape(t,1,1,y,x))
        from xyz (shape(z,y,x)).

    Raises
    ------
    TypeError
        mismatch of source (KLB file) and target (numpy array) data type
    IOError
    """
    header = readheader(filepath)
    cdef _np.ndarray[_np.uint32_t, ndim=1] imagesize = header["imagesize_tczyx"]
    cdef _np.dtype dtype = header["datatype"]

    # Drop leading singleton dimensions.
    # Don't squeeze the array to preserve the difference between e.g. xyz and xyt.
    cdef int d = 0
    while d < len(imagesize) and imagesize[d] == 1:
        d = d + 1

    cdef _np.ndarray A = _np.empty(imagesize[d:], dtype)
    readfull_inplace(A, filepath, numthreads, True)
    return A



def readroi(
    str filepath,
    tczyx_min,
    tczyx_max,
    const int numthreads = _mpc.cpu_count()
    ):
    """
    Read bounding box from KLB file

    Arguments
    ---------
    filepath : string
        File system path to KLB file
    tczyx_min : vector/list/tuple of length 1-5, order ([t,c,z,y,]x)
        Start of bounding box to read
    tczyx_max : vector/list/tuple of length 1-5, order ([t,c,z,y,]x)
        End of bounding box to read, inclusive
    numthreads, int, optional, default = multiprocessing.cpu_count()
        Number of threads to use for decompression

    Returns
    -------
    A : array, shape([t,c,z,y,]x)
        The content of the bounding box, note dimension order.
        Leading singleton dimensions are dropped, but the order is
        preserved to distinguish e.g. xyt (shape(t,1,1,y,x))
        from xyz (shape(z,y,x)).

    Raises
    ------
    TypeError
        mismatch of source (KLB file) and target (numpy array) data type
    IndexError
        when requested bounding box is out of bounds
    IOError
    """
    header = readheader(filepath)
    cdef _np.ndarray[_np.uint32_t, ndim=1] roisize = 1 + _np.array(tczyx_max).astype(_np.uint32) - _np.array(tczyx_min).astype(_np.uint32)
    cdef _np.ndarray A = _np.empty(roisize, header["datatype"])
    readroi_inplace(A, filepath, tczyx_min, tczyx_max, numthreads, False)
    return A



###########################################################
# Reading KLB files, into pre-allocated memory            #
###########################################################


def readfull_inplace(
    _np.ndarray A,
    str filepath,
    const int numthreads = _mpc.cpu_count(),
    bool nochecks = False
    ):
    """
    Read entire array from KLB file into pre-allocated array

    Arguments
    ---------
    A : array, shape([t,c,z,y,]x)
        Target array, note dimension order
    filepath : string
        File system path to KLB file
    numthreads : int, optional, default = multiprocessing.cpu_count()
        Number of threads to use for decompression
    nochecks : bool, optional, default = False
        Whether to skip type and bounds checks

    Raises
    ------
    TypeError
        mismatch of source (KLB file) and target (numpy array) data type
    IOError
    """
    if not nochecks:
        header = readheader(filepath)
        if A.dtype != header["datatype"]:
            raise TypeError("KLB type: %s, numpy type: %s; file at %s." % (header["datatype"], A.dtype, filepath))
        klbsize = _np.prod( header["imagesize_tczyx"] ) * A.itemsize
        if A.nbytes != klbsize:
            raise IOError("KLB size: %s, target size: %s (in bytes); file at %s." % (klbsize, A.nbytes, filepath))

    cdef _np.ndarray[_np.int8_t, ndim=1] buffer = _np.frombuffer(A, _np.int8)
    cdef KLB_DATA_TYPE ktype = INT8_TYPE # placeholder, overwritten by function call below
    cdef int errid = readKLBstackInPlace(filepath, &buffer[0], &ktype, numthreads)
    if errid != 0:
        raise IOError("Could not read KLB file '%s'. Error code %d" % (filepath, errid))



def readroi_inplace(
    _np.ndarray A,
    str filepath,
    tczyx_min,
    tczyx_max,
    const int numthreads = _mpc.cpu_count(),
    bool nochecks = False
    ):
    """
    Read bounding box from KLB file into pre-allocated array

    Arguments
    ---------
    A : array, shape([t,c,z,y,]x)
        Target array, note dimension order
    filepath : string
        File system path to KLB file
    tczyx_min : vector/list/tuple of length 1-5, order ([t,c,z,y,]x)
        Start of bounding box to read
    tczyx_max : vector/list/tuple of length 1-5, order ([t,c,z,y,]x)
        End of bounding box to read, inclusive
    numthreads, int, optional, default = multiprocessing.cpu_count()
        Number of threads to use for decompression
    nochecks : bool, optional, default = False
        Whether to skip type and bounds checks

    Raises
    ------
    TypeError
        mismatch of source (KLB file) and target (numpy array) data type
    IndexError
        when requested bounding box is out of bounds
    IOError
    """
    # convert data type and dimension order, pad to len = 5, expected by C function
    cdef _np.ndarray[_np.uint32_t, ndim=1] lb = _np.zeros((5,), _np.uint32)
    cdef _np.ndarray[_np.uint32_t, ndim=1] ub = _np.zeros((5,), _np.uint32)
    lb[:len(tczyx_min)] = _np.flipud(tczyx_min)
    ub[:len(tczyx_max)] = _np.flipud(tczyx_max)

    if not nochecks:
        header = readheader(filepath)
        if A.dtype != header["datatype"]:
            raise TypeError("KLB type: %s, numpy type: %s; file at %s." % (header["datatype"], A.dtype, filepath))
        klbsize = _np.prod( 1 + ub - lb ) * A.itemsize
        if A.nbytes != klbsize:
            raise IOError("KLB ROI size: %s, target size: %s (in bytes); file at %s." % (klbsize, A.nbytes, filepath))
        fullsize = header["imagesize_tczyx"]
        for d in range(5):
            if lb[d] < 0 or ub[d] >= fullsize[-1-d] or lb[d] > ub[d]:
                raise IndexError("ROI index out of bounds: KLB size: %s, requested ROI: %s-%s; file at %s" % (fullsize, _np.flipud(lb), _np.flipud(ub), filepath))

    cdef _np.ndarray[_np.int8_t, ndim=1] buffer = _np.frombuffer(A, _np.int8)
    cdef int errid = readKLBroiInPlace(filepath, &buffer[0], &lb[0], &ub[0], numthreads)
    if errid != 0:
        raise IOError("Could not read KLB file '%s'. Error code %d" % (filepath, errid))



###########################################################
# Writing KLB files                                       #
###########################################################


def writefull(
    _np.ndarray A,
    str filepath,
    const int numthreads = _mpc.cpu_count(),
    pixelspacing_tczyx = None,
    str metadata = None,
    _np.ndarray[_np.uint32_t, ndim=1] blocksize_tczyx = None,
    str compression = "bzip2"
    ):
    """
    Save array as KLB file, an existing file will be overwritten

    Arguments
    ---------
    A : array, shape([t,c,z,y,]x)
        Target array, note dimension order
    filepath : string
        File system path to KLB file
    numthreads : int, optional, default = multiprocessing.cpu_count()
        Number of threads to use for decompression
    pixelspacing_tczyx : vector/list/tuple of length 1-5, order ([t,c,z,y,]x)
        Spatial and temporal sampling, in a.u., µm, sec.
    metadata : string, optional, default=None
        Metadata to store in file, currently unsupported by pyklb.
    blocksize_tczyx : array, dtype=uint32, shape(5,), optional
        Shape of compression blocks
    compression : string, optional, default='bzip2'
        Compression method. Valid arguments are 'none', 'bzip2', 'zlib'

    Raises
    ------
    IOError
    """
    cdef _np.ndarray[_np.uint32_t, ndim=1] imagesize = _np.ones((5,), _np.uint32)
    cdef _np.ndarray[_np.float32_t, ndim=1] sampling = _np.ones((5,), _np.float32)
    
    imagesize[:A.ndim] = _np.flipud([A.shape[i] for i in range(A.ndim)]).astype(_np.uint32)
    if pixelspacing_tczyx != None:
        sampling[:len(pixelspacing_tczyx)] = _np.flipud(pixelspacing_tczyx).astype(_np.float32)

    cdef KLB_DATA_TYPE ktype = _klbtype(A.dtype)
    cdef KLB_COMPRESSION_TYPE kcompression = _klbcompression(compression)
    cdef _np.ndarray[_np.int8_t, ndim=1] buffer = _np.frombuffer(A, _np.int8)
    cdef int errid = writeKLBstack(&buffer[0], filepath, &imagesize[0], ktype, numthreads, &sampling[0], &blocksize_tczyx[0], kcompression, NULL)
    if errid != 0:
        raise IOError("Could not write KLB file '%s'. Error code %d" % (filepath, errid))



###########################################################
# Type conversion helper functions, not exported          #
###########################################################


cdef inline _np.dtype _pytype(const KLB_DATA_TYPE ktype):
    if ktype == UINT8_TYPE:
        return _np.dtype(_np.uint8)
    elif ktype == UINT16_TYPE:
        return _np.dtype(_np.uint16)
    elif ktype == UINT32_TYPE:
        return _np.dtype(_np.uint32)
    elif ktype == UINT64_TYPE:
        return _np.dtype(_np.uint64)
    elif ktype == INT8_TYPE:
        return _np.dtype(_np.int8)
    elif ktype == INT16_TYPE:
        return _np.dtype(_np.int16)
    elif ktype == INT32_TYPE:
        return _np.dtype(_np.int32)
    elif ktype == INT64_TYPE:
        return _np.dtype(_np.int64)
    elif ktype == FLOAT32_TYPE:
        return _np.dtype(_np.float32)
    elif ktype == FLOAT64_TYPE:
        return _np.dtype(_np.float64)
    raise Exception("Unknown or unsupported data type of KLB array: %d" % ktype)


cdef inline KLB_DATA_TYPE _klbtype(_np.dtype ptype):
    if ptype == _np.uint8:
        return UINT8_TYPE
    elif ptype == _np.uint16:
        return UINT16_TYPE
    elif ptype == _np.uint32:
        return UINT32_TYPE
    elif ptype == _np.uint64:
        return UINT64_TYPE
    elif ptype == _np.int8:
        return INT8_TYPE
    elif ptype == _np.int16:
        return INT16_TYPE
    elif ptype == _np.int32:
        return INT32_TYPE
    elif ptype == _np.int64:
        return INT64_TYPE
    elif ptype == _np.float32:
        return FLOAT32_TYPE
    elif ptype == _np.float64:
        return FLOAT64_TYPE
    raise Exception("Unknown or unsupported data type of KLB array: %d" % ptype)


cdef inline str _pycompression(const KLB_COMPRESSION_TYPE kcompression):
    if kcompression == NONE:
        return "none"
    elif kcompression == BZIP2:
        return "bzip2"
    elif kcompression == ZLIB:
        return "zlib"
    raise Exception("Unknown or unsupported compression of KLB array: %d" % kcompression)


cdef inline KLB_COMPRESSION_TYPE _klbcompression(str pcompression):
    if pcompression == "none":
        return NONE
    elif pcompression == "bzip2":
        return BZIP2
    elif pcompression == "zlib":
        return ZLIB
    raise Exception("Unknown or unsupported compression of KLB array: %d" % pcompression)
