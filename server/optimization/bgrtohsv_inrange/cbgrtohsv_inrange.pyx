#!/usr/bin/env python3

# Combine CV2 cvtColor and inRange in one call
# This is definitely faster than calling the 2 routines separately.
# This uses Cython to compile to C code, which is why it is fast.

import cython

import numpy as np
cimport numpy as np

import cv2
from libc.math cimport round

DTYPE = np.uint8
ctypedef np.uint8_t DTYPE_t

# define what gets imported
__all__ = ('bgrtohsv_inrange', 'bgrtohsv_inrange_table', 'bgrtohsv_inrange_preparetable', 'bgrtohsv_inrange_preparetable_gpu')


@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
cpdef unsigned char[:, :] bgrtohsv_inrange(unsigned char[:, :, :] image,
                                           unsigned char[:] low_limit, unsigned char[:] high_limit,
                                           unsigned char[:, :] output):
    '''Perform BGR2HSV and thresholding in one pass'''

    cdef unsigned int xSize, ySize
    cdef unsigned int iX, iY
    cdef int b, g, r, hDelta, hOffset
    cdef int v, vMinusMin
    cdef double h, s

    # assert(low_limit[1] > 0)
    # assert(low_limit[2] > 0)

    xSize, ySize = image.shape[0:2]

    for iX in range(xSize):
        for iY in range(ySize):
            b = int(image[iX, iY, 0])
            g = int(image[iX, iY, 1])
            r = int(image[iX, iY, 2])

            v = r
            hDelta = g - b
            hOffset = 0
            if g > v:
                v = g
                hDelta = b - r
                hOffset = 60
            if b > v:
                v = b
                hDelta = r - g
                hOffset = 120

            if v < low_limit[2] or v > high_limit[2]:
                output[iX, iY] = 0
                continue

            vMinusMin = v - min(b, g, r)
            if vMinusMin == 0:
                output[iX, iY] = 0
                continue

            # round() is slightly more accurate compared to OpenCV version, but slower
            s = round((255.0 * vMinusMin) / v + 0.001)  # note "/" is float division
            # s = (255.0 * vMinusMin) / v # note "/" is float division
            if s < low_limit[1] or s > high_limit[1]:
                output[iX, iY] = 0
                continue

            # round() is slightly more accurate compared to OpenCV version, but slower
            h = 30.0 * hDelta / vMinusMin
            h = hOffset + round(h + 0.001)
            # h = hOffset + 30.0 * hDelta / vMinusMin
            if h < 0:
                h += 180

            if h < low_limit[0] or h > high_limit[0]:
                output[iX, iY] = 0
            else:
                output[iX, iY] = 255

    return output


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef bgrtohsv_inrange_preparetable(np.ndarray low_limit, np.ndarray high_limit):
    '''Prepare lookup table for very fast BGR2HSV+thresholding.'''

    cdef int b, g, r, i

    cdef np.ndarray[DTYPE_t, ndim=3] bgr = np.empty(shape=(256, 256*256, 3), dtype=DTYPE)

    for b in range(256):
        i = 0
        for g in range(256):
            for r in range(256):
                bgr[b, i, 0] = b
                bgr[b, i, 1] = g
                bgr[b, i, 2] = r
                i += 1

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    # don't know why, but this seems to be a *little* faster, at least on my laptop
    lowLimitHSV = np.array(low_limit)
    highLimitHSV = np.array(high_limit)

    thres = cv2.inRange(hsv, lowLimitHSV, highLimitHSV)
    return thres


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef bgrtohsv_inrange_preparetable_gpu(np.ndarray low_limit, np.ndarray high_limit):
    '''Prepare lookup table for very fast BGR2HSV+thresholding.'''

    cdef int b, g, r, i

    cdef np.ndarray[DTYPE_t, ndim=3] bgr = np.empty(shape=(256, 256*256, 3), dtype=DTYPE)

    for b in range(256):
        i = 0
        for g in range(256):
            for r in range(256):
                bgr[b, i, 0] = b
                bgr[b, i, 1] = g
                bgr[b, i, 2] = r
                i += 1

    bgr_umat = cv2.UMat(bgr)
    hsv_umat = cv2.cvtColor(bgr_umat, cv2.COLOR_BGR2HSV)

    # don't know why, but this seems to be a *little* faster, at least on my laptop
    lowLimitHSV = np.array(low_limit)
    highLimitHSV = np.array(high_limit)

    thres_umat = cv2.inRange(hsv_umat, lowLimitHSV, highLimitHSV)
    thres = cv2.UMat.get(thres_umat)

    return thres


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef unsigned char[:, :] bgrtohsv_inrange_table(unsigned char[:, :] lookup, unsigned char[:, :, :] image,
                                                 unsigned char[:, :] output):
    '''Use a lookup table for very fast BGR2HSV+thresholding.
    The lookup table is created by bgrtohsv_inrange_preparetable and then passed to this routine each iteration.
    Processing an image frame is very fast, but there is significant overhead in creating the lookup table.'''

    cdef unsigned int xSize, ySize
    cdef unsigned int iX, iY
    cdef unsigned char b
    cdef int g, r, i

    xSize, ySize = image.shape[0:2]

    for iX in range(xSize):
        for iY in range(ySize):
            b = image[iX, iY, 0]
            g = int(image[iX, iY, 1])
            r = int(image[iX, iY, 2])
            i = 256 * g + r

            output[iX, iY] = lookup[b, i]

    return output

# --------------------------------------------------------------------------------
# useful during development

# def checkEqualHSV(openCVRes, myRes):
#     xSize, ySize = openCVRes.shape[0:2]

#     maxDelta = 0
#     sumDeltaH = 0
#     sumDeltaS = 0
#     sumDeltaV = 0
#     for iX in range(xSize):
#         for iY in range(ySize):
#             # the hard way ;-)
#             delta = max(abs(int(openCVRes[iX,iY,0]) - int(myRes[iX,iY,0])),
#                          abs(int(openCVRes[iX,iY,1]) - int(myRes[iX,iY,1])),
#                          abs(int(openCVRes[iX,iY,2]) - int(myRes[iX,iY,2])))
#             if delta > 0:
#                 maxDelta = max(maxDelta, delta)

#                 sumDeltaH += abs(int(openCVRes[iX,iY,0]) - int(myRes[iX,iY,0]))
#                 sumDeltaS += abs(int(openCVRes[iX,iY,1]) - int(myRes[iX,iY,1]))
#                 sumDeltaV += abs(int(openCVRes[iX,iY,2]) - int(myRes[iX,iY,2]))

#     return maxDelta, sumDeltaH, sumDeltaS, sumDeltaV
