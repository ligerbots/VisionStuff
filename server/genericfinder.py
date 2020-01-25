#!/usr/bin/env python3

"""Generic finder used when not searching for any targets.
name and finder_id are instance variables so that we can create multiple
GenericFinders for different purposes.

finder_id may not need to be changed, depending on circumstances."""

import sys
import math
import cv2


class GenericFinder:
    def __init__(self, name, camera, finder_id=1.0, exposure=0, rotation=None, line_coords=None):
        self.name = name
        self.finder_id = float(finder_id)   # id needs to be float! "id" is a reserved word.
        self.camera = camera                # string with camera name
        self.stream_camera = None           # None means same camera
        self.exposure = exposure
        self.rotation = rotation            # cv2.ROTATE_90_CLOCKWISE = 0, cv2.ROTATE_180 = 1, cv2.ROTATE_90_COUNTERCLOCKWISE = 2
        self.line_coords = line_coords      # coordinates to draw a line on the image
        return

    def process_image(self, camera_frame):
        '''Main image processing routine'''
        return (1.0, self.finder_id, 0.0, 0.0, 0.0)

    def prepare_output_image(self, input_frame):
        '''Prepare output image for drive station. Rotate image if needed, otherwise nothing to do.'''
        # WARNING rotation=0 is actually 90deg clockwise (dumb!!)
        if self.rotation is not None:
            # rotate function makes a copy, so no need to do that ahead.
            output_frame = cv2.rotate(input_frame, self.rotation)
        else:
            output_frame = input_frame.copy()
        if self.line_coords is not None:
            cv2.line(output_frame, self.line_coords[0], self.line_coords[1], (255, 255, 255), 2)

        return output_frame

    # ----------------------------------------------------------------------
    # the routines below are not needed here, but are used by lots of Finders,
    #  so keep them in one place

    @staticmethod
    def contour_center_width(contour):
        '''Find boundingRect of contour, but return center and width/height'''

        x, y, w, h = cv2.boundingRect(contour)
        return (x + int(w / 2), y + int(h / 2)), (w, h)

    @staticmethod
    def quad_fit(contour, approx_dp_error):
        '''Simple polygon fit to contour with error related to perimeter'''

        peri = cv2.arcLength(contour, True)
        return cv2.approxPolyDP(contour, approx_dp_error * peri, True)


# Main routines, used for running the finder by itself for debugging and timing

def process_files(line_finder, input_files, output_dir):
    '''Process the files and output the marked up image'''
    import os.path

    for image_file in input_files:
        # print()
        # print(image_file)
        bgr_frame = cv2.imread(image_file)
        result = line_finder.process_image(bgr_frame)
        print(image_file, result[0], result[1], result[2], math.degrees(result[3]), math.degrees(result[4]))

        bgr_frame = line_finder.prepare_output_image(bgr_frame)

        outfile = os.path.join(output_dir, os.path.basename(image_file))
        # print('{} -> {}'.format(image_file, outfile))
        cv2.imwrite(outfile, bgr_frame)

        # cv2.imshow("Window", bgr_frame)
        # q = cv2.waitKey(-1) & 0xFF
        # if q == ord('q'):
        #     break
    return


def time_processing(cube_processor, input_files):
    '''Time the processing of the test files'''

    from codetimer import CodeTimer
    from time import time

    startt = time()

    cnt = 0

    # Loop 100x over the files. This is needed to make it long enough
    #  to get reasonable statistics. If we have 100s of files, we could reduce this.
    # Need the total time to be many seconds so that the timing resolution is good.
    for _ in range(100):
        for image_file in input_files:
            with CodeTimer("Read Image"):
                bgr_frame = cv2.imread(image_file)

            with CodeTimer("Main Processing"):
                cube_processor.process_image(bgr_frame)

            cnt += 1

    deltat = time() - startt

    print("{0} frames in {1:.3f} seconds = {2:.2f} msec/call, {3:.2f} FPS".format(
        cnt, deltat, 1000.0 * deltat / cnt, cnt / deltat))
    CodeTimer.output_timers()
    return


def main(finder_type):
    '''Main routine for testing this Finder'''
    import argparse

    parser = argparse.ArgumentParser(description='2019 rrtarget finder')
    parser.add_argument('--output_dir', help='Output directory for processed images')
    parser.add_argument('--time', action='store_true', help='Loop over files and time it')
    parser.add_argument('--calib_file', help='Calibration file')
    parser.add_argument('input_files', nargs='+', help='input files')

    args = parser.parse_args()

    finder = finder_type(args.calib_file)

    if sys.platform == "win32":
        # windows does not expand the "*" files on the command line
        #  so we have to do it.
        import glob

        infiles = []
        for f in args.input_files:
            infiles.extend(glob.glob(f))
        args.input_files = infiles

    if args.output_dir is not None:
        process_files(finder, args.input_files, args.output_dir)
    elif args.time:
        time_processing(finder, args.input_files)

    return
