#!/usr/bin/env python3

"""Generic finder used when not searching for any targets.
name and finder_id are instance variables so that we can create multiple
GenericFinders for different purposes.

finder_id may not need to be changed, depending on circumstances."""

import sys
import math
import cv2
import numpy
import hough_fit

two_pi = 2 * math.pi
pi_by_2 = math.pi / 2


class GenericFinder:
    def __init__(self, name, camera, finder_id=1.0, exposure=0, line_coords=None):
        self.name = name
        self.finder_id = float(finder_id)   # id needs to be float! "id" is a reserved word.
        self.camera = camera                # string with camera name
        self.stream_camera = None           # None means same camera
        self.exposure = exposure
        self.line_coords = line_coords      # coordinates to draw a line on the image
        return

    def process_image(self, camera_frame):
        '''Main image processing routine'''

        # for 2020, standard result includes position, angle of a 2nd ball
        return (1.0, self.finder_id, 0.0, 0.0, 0.0, -1.0, -1.0)

    def prepare_output_image(self, input_frame):
        '''Prepare output image for drive station. Rotate image if needed, otherwise nothing to do.'''

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
    def approxPolyDP_adaptive(contour, nsides, max_dp_error=0.1):
        '''Use approxPolyDP to fit a polygon to a contour.
        Find the smallest dp_error that gets the correct number of sides.
        The results seem to often be a little wrong, but they are a quick starting point.'''

        step = 0.0005
        peri = cv2.arcLength(contour, True)
        dp_err = step
        while dp_err <= max_dp_error:
            res = cv2.approxPolyDP(contour, dp_err * peri, True)
            if len(res) <= nsides:
                # print('approxPolyDP_adaptive found at step', step)
                return res
            dp_err += step
        # print('approxPolyDP_adaptive failed')
        return None

    @staticmethod
    def quad_from_diagonals(contour):
        '''Find the furthest points along the diagonal to be the quadrilateral.
        Taken from Robot Casserole 1736. Thanks!'''

        x_plus_y = contour[:, :, 0] + contour[:, :, 1]
        x_minus_y = contour[:, :, 0] - contour[:, :, 1]
        cornerIndices = (x_plus_y.argmin(), x_minus_y.argmin(), x_plus_y.argmax(), x_minus_y.argmax())
        return contour[cornerIndices, :]

    @staticmethod
    def quad_fit(contour, image_frame=None):
        '''Best fit of a quadrilateral to the contour.
        Pass in image_frame to get some debugging info.'''

        # approx = GenericFinder.approxPolyDP_adaptive(contour, nsides=4)
        approx = GenericFinder.quad_from_diagonals(contour)
        return hough_fit.hough_fit(contour, nsides=4, approx_fit=approx, image_frame=image_frame)

    @staticmethod
    def sort_corners(contour, center=None):
        '''Sort the contour in our standard order, starting upper-left and going counter-clockwise'''

        # Note: the inputs are all numpy arrays, so it is fast to operate on the whole array at once

        if center is None:
            center = contour.mean(axis=0)

        d = contour - center

        # remember that y-axis increases down, so flip the sign
        angle = (numpy.arctan2(-d[:, :, 1], d[:, :, 0]) - pi_by_2) % two_pi
        angle = angle[:, 0] # contours have 3 dimensions, drop one here
        return contour[numpy.argsort(angle)]

    @staticmethod
    def major_minor_axes(moments):
        '''Compute the major/minor axes and orientation of an object from the moments'''

        # See https://en.wikipedia.org/wiki/Image_moment
        # Be careful, different sites define the normalized central moments differently
        # See also http://raphael.candelier.fr/?blog=Image%20Moments

        m00 = moments['m00']
        mu20 = moments['mu20'] / m00
        mu02 = moments['mu02'] / m00
        mu11 = moments['mu11'] / m00

        descr = math.sqrt(4.0 * mu11*mu11 + (mu20 - mu02)**2)

        major = math.sqrt(2.0 * (mu20 + mu02 + descr))
        minor = math.sqrt(2.0 * (mu20 + mu02 - descr))

        # note this does not use atan2.
        angle = 0.5 * math.atan(2*mu11 / (mu20-mu02))
        if mu20 < mu02:
            angle += pi_by_2

        return major, minor, angle


# --------------------------------------------------------------------------------
# Main routines, used for running the finder by itself for debugging and timing

def process_files(line_finder, input_files, output_dir):
    '''Process the files and output the marked up image'''
    import os.path

    for image_file in input_files:
        # print()
        # print(image_file)
        bgr_frame = cv2.imread(image_file)

        result = line_finder.process_image(bgr_frame)
        print(image_file, result[0], result[1], round(result[2], 1),
              round(math.degrees(result[3]), 1), round(math.degrees(result[4]), 1),
              round(result[5], 1), round(math.degrees(result[6]), 1))

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
    for _ in range(10):
        for image_file in input_files:
            with CodeTimer("Read Image"):
                bgr_frame = cv2.imread(image_file)

            with CodeTimer("Main Processing"):
                cube_processor.process_image(bgr_frame)

            cnt += 1

    deltat = time() - startt

    print("{0} frames in {1:.3f} seconds = {2:.2f} ms/call, {3:.2f} FPS".format(
        cnt, deltat, 1000.0 * deltat / cnt, cnt / deltat))
    CodeTimer.output_timers()
    return


def main(finder_type):
    '''Main routine for testing this Finder'''
    import argparse
    import cameras

    parser = argparse.ArgumentParser(description='2019 rrtarget finder')
    parser.add_argument('--output_dir', help='Output directory for processed images')
    parser.add_argument('--time', action='store_true', help='Loop over files and time it')
    parser.add_argument('--calib_file', help='Calibration file')
    parser.add_argument('--rotate_calib', action='store_true', help='Rotate the calibration file upon load')
    parser.add_argument('input_files', nargs='+', help='input files')

    args = parser.parse_args()

    rot = 90 if args.rotate_calib else 0
    calib_matrix, dist_matrix = cameras.Camera.load_calibration_file(args.calib_file, rotation=rot)
    # print('calib', calib_matrix)
    # print('dist', dist_matrix)
    finder = finder_type(calib_matrix, dist_matrix)

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
