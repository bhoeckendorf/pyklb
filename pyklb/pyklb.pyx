#!python
#cython: initializedcheck=False, boundscheck=False, overflowcheck=False

import cython
cimport cython
from libc.stdint cimport uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t
from cpython cimport bool
import numpy as np
cimport numpy as np



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
        File system path to klb file
        
    Returns
    -------
    header : Dict
        Key-value dict of header fields,
        incl. 'imagesize_yxzct', 'datatype', 'pixelspacing_yxzct'
    
    Raises
    ------
    IOError
    """
    cdef np.ndarray[np.uint32_t, ndim=1] imagesize = np.empty((5,), np.uint32)
    cdef np.ndarray[np.uint32_t, ndim=1] blocksize = np.empty((5,), np.uint32)
    cdef np.ndarray[np.float32_t, ndim=1] pixelspacing = np.empty((5,), np.float32)
    cdef KLB_DATA_TYPE ktype = UINT8_TYPE
    cdef KLB_COMPRESSION_TYPE kcompression = NONE
    cdef np.ndarray[np.int8_t, ndim=1] metadata = np.empty((256,), np.int8)
    cdef int errid = readKLBheader(filepath, &imagesize[0], &ktype, &pixelspacing[0], &blocksize[0], &kcompression, <char*> &metadata[0])
    if errid != 0:
        raise IOError("Could not read KLB header of file '%s'. Error code %d" % (filepath, errid))

    # xyz to yxz (KLB to numpy)
    cdef np.uint32_t tempui = imagesize[0]
    imagesize[0] = imagesize[1]
    imagesize[1] = tempui

    tempui = blocksize[0]
    blocksize[0] = blocksize[1]
    blocksize[1] = tempui

    cdef np.float32_t tempfl = pixelspacing[0]
    pixelspacing[0] = pixelspacing[1]
    pixelspacing[1] = tempfl

    return {
        "imagesize_yxzct": imagesize,
        "blocksize_yxzct": blocksize,
        "pixelspacing_yxzct": pixelspacing,
        "metadata": metadata,
        "datatype": _pytype(ktype),
        "compression": _pycompression(kcompression)
        }



def readfull(
    str filepath,
    const int numthreads = 1
    ):
    """
    Read entire array from KLB file
    
    Arguments
    ---------
    filepath : string
        File system path to KLB file
    numthreads : int, optional, default = 1
        Number of threads to use for decompression
    
    Returns
    -------
    A : array, shape(y[,x,z,c,t])
        The entire array stored in the KLB file.
        Trailing singleton dimensions are dropped, but the order is
        preserved to distinguish e.g. xyt (shape(y,x,1,1,t))
        from xyz (shape(y,x,z)).

        Indexing order is geometric, i.e. column-major.
        First 2 dimensions are swapped to enable numpy-style indexing.

    Raises
    ------
    IOError
    """
    header = readheader(filepath)
    cdef np.ndarray[np.uint32_t, ndim=1] imagesize = header["imagesize_yxzct"]
    cdef np.dtype dtype = header["datatype"]

    # yxz to xyz (numpy to KLB)
    cdef np.uint32_t temp = imagesize[0]
    imagesize[0] = imagesize[1]
    imagesize[1] = temp

    # Drop trailing singleton dimensions.
    # Don't squeeze the array to preserve the difference between e.g. xyz and xyt.
    cdef int ndim = len(imagesize)
    while ndim > 0 and imagesize[ndim-1] == 1:
        ndim = ndim - 1

    cdef np.ndarray A = np.empty(imagesize[0:ndim], dtype, order="F")
    readfull_inplace(A, filepath, numthreads, True)
    return A.swapaxes(0,1)


    
def readroi(
    str filepath,
    np.ndarray[np.uint32_t, ndim=1] yxzct_min,
    np.ndarray[np.uint32_t, ndim=1] yxzct_max,
    const int numthreads = 1
    ):
    """
    Read bounding box from KLB file
    
    Arguments
    ---------
    filepath : string
        File system path to KLB file
    yxzct_min : array, dtype=uint32, shape(1[,1,1,1,1])
        Start of bounding box to read, vector of length 1-5
    yxzct_max : array, dtype=uint32, shape(1[,1,1,1,1])
        End of bounding box to read (inclusive), vector of length 1-5
    numthreads, int, optional, default = 1
        Number of threads to use for decompression
    
    Returns
    -------
    A : array, shape(y[,x,z,c,t])
        The content of the bounding box.
        Trailing singleton dimensions are dropped, but the order is
        preserved to distinguish e.g. xyt (shape(y,x,1,1,t))
        from xyz (shape(y,x,z)).

        Indexing order is geometric, i.e. column-major.
        First 2 dimensions are swapped to enable numpy-style indexing.
    
    Raises
    ------
    IndexError
        when indices of requested bounding box is out of bounds
    IOError
    """
    cdef np.ndarray[np.uint32_t, ndim=1] roisize = 1 + yxzct_max - yxzct_min

    # Drop trailing singleton dimensions.
    # Don't squeeze the array to preserve the difference between e.g. xyz and xyt.
    cdef int ndim = len(roisize)
    while ndim > 0 and roisize[ndim-1] == 1:
        ndim = ndim - 1

    # if needed, pad bounds with 0 until len = 5, which is expected by C function
    if len(yxzct_min) < 5:
        yxzct_min = np.hstack(( yxzct_min, np.array([0 for i in range(5-len(yxzct_min))], np.uint32) ))
    if len(yxzct_max) < 5:
        yxzct_max = np.hstack(( yxzct_max, np.array([0 for i in range(5-len(yxzct_max))], np.uint32) ))

    header = readheader(filepath)
    cdef np.ndarray[np.uint32_t, ndim=1] imagesize = header["imagesize_yxzct"]
    cdef np.dtype dtype = header["datatype"]
    for d in range(5):
        if yxzct_min[d] > yxzct_max[d] or yxzct_max[d] > imagesize[d] - 1:
            raise IndexError("Invalid bounding box: %s -> %s, image size %s (all shapes in order yxczt); file at %s."
                             % (yxzct_min, yxzct_max, imagesize, filepath))

    # yxz to xyz (numpy to KLB)
    cdef np.uint32_t temp = roisize[0]
    roisize[0] = roisize[1]
    roisize[1] = temp

    cdef np.ndarray A = np.empty(roisize[0:ndim], dtype, order="F")
    readroi_inplace(A, filepath, yxzct_min, yxzct_max, numthreads, True)
    return A.swapaxes(0,1)



###########################################################
# Reading KLB files, into pre-allocated memory            #
###########################################################


def allocate(
    imagesize_yxzct,
    datatype
    ):
    """
    Allocate an array to be used with the _inplace fuctions of KLB.
    The returned array is in xyzct and "F" order, although it will appear to be in yxzct shape.

    Arguments
    ---------
    imagesize_yxzct : shape of target array, in yxzct order (as in the 'imagesize_yxzct' field returned by pyklb.readheader(...))
    datatype : NumPy dtype

    Returns
    -------
    NumPy array compatible with pyklb's inplace functions.
    """
    # yxz to xyz (numpy to KLB)
    temp = imagesize_yxzct[0]
    imagesize_yxzct[0] = imagesize_yxzct[1]
    imagesize_yxzct[1] = temp
    return np.empty(imagesize_yxzct, datatype, order="F").swapaxes(0,1)



def readfull_inplace(
    np.ndarray A,
    str filepath,
    const int numthreads = 1,
    bool nochecks = False
    ):
    """
    Read entire array from KLB file into pre-allocated array
    
    Arguments
    ---------
    A : array, shape(x[,y,z,c,t]), order='F'
        Target array, note dimension order!
        If needed, call A.swapaxes(0,1) to get numpy convention yxzct
    filepath : string
        File system path to KLB file
    numthreads : int, optional, default = 1
        Number of threads to use for decompression
    nochecks : bool, optional, default = False
        Whether to skip type and bounds checks
    
    Raises
    ------
    TypeError
        when dtypes or memory layout of target array and KLB file don't match
    IndexError
        when size of target array and KLB file don't match
    IOError
    """
    if A.flags["F_CONTIGUOUS"] == False:
        A = A.swapaxes(0,1)
        if A.flags["F_CONTIGUOUS"] == False:
            raise TypeError("Target array must be in xyzct shape and order='F'. Use pyklb.allocate(...) function to create target array.")

    if not nochecks:
        header = readheader(filepath)
        if A.dtype != header["datatype"]:
            raise TypeError("KLB type: %s, target type: %s; file at %s." % (header["datatype"], A.dtype, filepath))

        insize = header["imagesize_yxzct"]
        # yxz to xyz (numpy to KLB)
        temp = insize[0]
        insize[0] = insize[1]
        insize[1] = temp

        outsize = A.shape
        for d in range(A.ndim):
            if insize[d] != outsize[d]:
                raise IndexError("KLB size: %s, target size: %s (all shapes in order xyzct); file at %s." % (insize, [A.shape[i] for i in range(A.ndim)], filepath))
        # handle trailing singleton dimensions, if any
        for d in range(A.ndim, 5):
            if insize[d] != 1:
                raise IndexError("KLB size: %s, target size: %s (all shapes in order xyzct); file at %s." % (insize, [A.shape[i] for i in range(A.ndim)], filepath))

    cdef np.ndarray[np.int8_t, ndim=1] buffer = np.frombuffer(A, np.int8)
    cdef KLB_DATA_TYPE ktype = INT8_TYPE # placeholder, overwritten by function call below
    cdef int errid = readKLBstackInPlace(filepath, &buffer[0], &ktype, numthreads)
    if errid != 0:
        raise IOError("Could not read KLB file '%s'. Error code %d" % (filepath, errid))



def readroi_inplace(
    np.ndarray A,
    str filepath,
    np.ndarray[np.uint32_t, ndim=1] yxzct_min,
    np.ndarray[np.uint32_t, ndim=1] yxzct_max,
    const int numthreads = 1,
    bool nochecks = False
    ):
    """
    Read bounding box from KLB file into pre-allocated array
    
    Arguments
    ---------
    A : array, shape(x[,y,z,c,t]), order='F'
        Target array, note dimension order!
        If needed, call A.swapaxes(0,1) to get numpy convention yxzct
    filepath : string
        File system path to KLB file
    yxzct_min : array, dtype=uint32, shape(1[,1,1,1,1])
        Start of bounding box to read, vector of length 1-5
    yxzct_max : array, dtype=uint32, shape(1[,1,1,1,1])
        End of bounding box to read (inclusive), vector of length 1-5
    numthreads, int, optional, default = 1
        Number of threads to use for decompression
    nochecks : bool, optional, default = False
        Whether to skip type and bounds checks
    
    Raises
    ------
    TypeError
        when dtypes or memory layout of target array and KLB file don't match
    IndexError
        when size of target array and KLB file don't match
    IOError
    """
    # yxz to xyz (numpy to KLB)
    cdef np.uint32_t temp = yxzct_min[0]
    yxzct_min[0] = yxzct_min[1]
    yxzct_min[1] = temp

    temp = yxzct_max[0]
    yxzct_max[0] = yxzct_max[1]
    yxzct_max[1] = temp

    cdef np.ndarray[np.uint32_t, ndim=1] roisize = 1 + yxzct_max - yxzct_min

    # if needed, pad bounds with 0 until len = 5, which is expected by C function
    if len(yxzct_min) < 5:
        yxzct_min = np.hstack(( yxzct_min, np.array([0 for i in range(5-len(yxzct_min))], np.uint32) ))
    if len(yxzct_max) < 5:
        yxzct_max = np.hstack(( yxzct_max, np.array([0 for i in range(5-len(yxzct_max))], np.uint32) ))

    if A.flags["F_CONTIGUOUS"] == False:
        A = A.swapaxes(0,1)
        if A.flags["F_CONTIGUOUS"] == False:
            raise TypeError("Target array must be in xyzct shape and order='F'. Use pyklb.allocate(...) function to create target array.")

    if not nochecks:
        header = readheader(filepath)
        if A.dtype != header["datatype"]:
            raise TypeError("KLB type: %s, target type: %s; file at %s." % (header["datatype"], A.dtype, filepath))
        insize = header["imagesize_yxzct"]
        outsize = A.shape
        for d in range(A.ndim):
            if yxzct_min[d] > yxzct_max[d] or yxzct_max[d] > insize[d] - 1:
                raise IndexError("Invalid bounding box: %s -> %s, image size %s (all shapes in order xyzct); file at %s." % (yxzct_min, yxzct_max, insize, filepath))

    cdef np.ndarray[np.int8_t, ndim=1] buffer = np.frombuffer(A, np.int8)
    cdef int errid = readKLBroiInPlace(filepath, &buffer[0], &yxzct_min[0], &yxzct_max[0], numthreads)
    if errid != 0:
        raise IOError("Could not read KLB file '%s'. Error code %d" % (filepath, errid))



###########################################################
# Writing KLB files                                       #
###########################################################

    
def writefull(
    np.ndarray A,
    str filepath,
    const int numthreads = 1,
    np.ndarray[np.float32_t, ndim=1] pixelspacing = None,
    str metadata = None,
    np.ndarray[np.uint32_t, ndim=1] blocksize = None,
    str compression = "bzip2"
    ):
    """
    Save array as KLB file, an existing file will be overwritten
    
    Arguments
    ---------
    A : array, shape(x[,y,z,c,t])
        Target array
    filepath : string
        File system path to KLB file
    numthreads : int, optional, default = 1
        Number of threads to use for decompression
    pixelspacing : array, dtype=float32, shape(1,1,1,1,1), optional, default=[1,1,1,1,1]
        Spatial and temporal sampling, in µm and s.
    metadata : string, optional, default=None
        Metadata to store in file.
    blocksize : array, dtype=uint32, shape(1,1,1,1,1), optional
        Shape of compression blocks
    compression : string, optional, default='bzip2'
        Compression method. Valid arguments are 'none', 'bzip2', 'zlib'
    
    Raises
    ------
    IOError
    """
    cdef np.ndarray[np.uint32_t, ndim=1] imagesize = np.ones((5,), np.uint32)
    for d in range(A.ndim):
        imagesize[d] = A.shape[d]
    
    cdef KLB_DATA_TYPE ktype = _klbtype(A.dtype)
    cdef KLB_COMPRESSION_TYPE kcompression = _klbcompression(compression)
    cdef np.ndarray[np.int8_t, ndim=1] buffer = np.frombuffer(A, np.int8)
    cdef int errid = writeKLBstack(&buffer[0], filepath, &imagesize[0], ktype, numthreads, &pixelspacing[0], &blocksize[0], kcompression, NULL)
    if errid != 0:
        raise IOError("Could not write KLB file '%s'. Error code %d" % (filepath, errid))



###########################################################
# Type conversion helper functions, not exported          #
###########################################################


cdef inline np.dtype _pytype(const KLB_DATA_TYPE ktype):
    if ktype == UINT8_TYPE:
        return np.dtype(np.uint8)
    elif ktype == UINT16_TYPE:
        return np.dtype(np.uint16)
    elif ktype == UINT32_TYPE:
        return np.dtype(np.uint32)
    elif ktype == UINT64_TYPE:
        return np.dtype(np.uint64)
    elif ktype == INT8_TYPE:
        return np.dtype(np.int8)
    elif ktype == INT16_TYPE:
        return np.dtype(np.int16)
    elif ktype == INT32_TYPE:
        return np.dtype(np.int32)
    elif ktype == INT64_TYPE:
        return np.dtype(np.int64)
    elif ktype == FLOAT32_TYPE:
        return np.dtype(np.float32)
    elif ktype == FLOAT64_TYPE:
        return np.dtype(np.float64)
    raise Exception("Unknown or unsupported data type of KLB array: %d" % ktype)


cdef inline KLB_DATA_TYPE _klbtype(np.dtype ptype):
    if ptype == np.uint8:
        return UINT8_TYPE
    elif ptype == np.uint16:
        return UINT16_TYPE
    elif ptype == UINT32_TYPE:
        return np.uint32
    elif ptype == UINT64_TYPE:
        return np.uint64
    elif ptype == INT8_TYPE:
        return np.int8
    elif ptype == INT16_TYPE:
        return np.int16
    elif ptype == INT32_TYPE:
        return np.int32
    elif ptype == INT64_TYPE:
        return np.int64
    elif ptype == FLOAT32_TYPE:
        return np.float32
    elif ptype == FLOAT64_TYPE:
        return np.float64
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
