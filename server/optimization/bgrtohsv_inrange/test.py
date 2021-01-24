#!/usr/bin/env python3

# Combine CV2 cvtColor and inRange in one call
# Hopefully this is faster than the two separate
# Eventually, use Cython to speed it up

import argparse
import numpy
import cv2


from optimization.bgrtohsv_inrange import *
from util.codetimer import CodeTimer


def bgrtohsv_inrange_cv2(image, lowLimitHSV, highLimitHSV, hsv_frame, thres_frame):
    hsv_frame = cv2.cvtColor(image, cv2.COLOR_BGR2HSV, dst=hsv_frame)
    thres_frame = cv2.inRange(hsv_frame, lowLimitHSV, highLimitHSV, dst=thres_frame)
    return thres_frame


def main():
    parser = argparse.ArgumentParser(description='bgrtohsv_inrange test program')
    parser.add_argument('files', nargs='+', help='input files')

    args = parser.parse_args()

    print("haveOpenCL =", cv2.ocl.haveOpenCL())

    # Color threshold values, in HSV space
    low_limit_hsv = numpy.array((70, 60, 30), dtype=numpy.uint8)
    high_limit_hsv = numpy.array((100, 255, 255), dtype=numpy.uint8)

    # if args.no_opencl:
    #     print('Disabling OpenCL')
    #     cv2.ocl.setUseOpenCL(False)

    # prime the routines for more accurate timing
    bgr_frame = cv2.imread(args.files[0])

    for _ in range(5):
        with CodeTimer("TableInit_GPU"):
            lookup_table = bgrtohsv_inrange_preparetable_gpu(low_limit_hsv, high_limit_hsv)
            print('lookup table:', lookup_table.shape)

        with CodeTimer("TableInit"):
            lookup_table = bgrtohsv_inrange_preparetable(low_limit_hsv, high_limit_hsv)
            print('lookup table:', lookup_table.shape)

    func_out = numpy.empty(shape=bgr_frame.shape[0:2], dtype=numpy.uint8)
    table_out = numpy.empty(shape=bgr_frame.shape[0:2], dtype=numpy.uint8)
    hsv_frame = numpy.empty(shape=bgr_frame.shape, dtype=numpy.uint8)
    thres_frame = numpy.empty(shape=bgr_frame.shape[0:2], dtype=numpy.uint8)

    for _ in range(100):
        for image_file in args.files:
            bgr_frame = cv2.imread(image_file)

            with CodeTimer("OpenCV"):
                bgrtohsv_inrange_cv2(bgr_frame, low_limit_hsv, high_limit_hsv, hsv_frame, thres_frame)
            with CodeTimer("FunctionConversion"):
                bgrtohsv_inrange(bgr_frame, low_limit_hsv, high_limit_hsv, func_out)
            with CodeTimer("TableConversion"):
                bgrtohsv_inrange_table(lookup_table, bgr_frame, table_out)

            ##print(image_file, "equal =", checkEqualHSV(cv2Out, func_out))
            # if not numpy.array_equal(thres_frame, func_out):
            #    print(image_file, " function conversion failed")
            if not numpy.array_equal(thres_frame, table_out):
                print(image_file, " table conversion failed")
    CodeTimer.output_timers()
    return


if __name__ == '__main__':
    main()
