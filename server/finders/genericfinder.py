#!/usr/bin/env python3

"""Generic finder used when not searching for any targets.
name and finder_id are instance variables so that we can create multiple
GenericFinders for different purposes.

finder_id may not need to be changed, depending on circumstances."""

import sys
import math
import cv2
import numpy
import fitters.hough_fit

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
        '''Prepare output image for drive station.
        Add a guide line if that is called for.'''

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
    def quad_fit(contour, image_frame=None):
        '''Best fit of a quadrilateral to the contour.
        Pass in image_frame to get some debugging info.'''

        approx = hough_fit.approxPolyDP_adaptive(contour, nsides=4)
        return hough_fit.hough_fit(contour, nsides=4, approx_fit=approx, image_frame=image_frame)

    @staticmethod
    def sort_corners(contour, center=None):
        '''Sort the contour in our standard order, starting upper-left and going counter-clockwise'''

        # Note: the inputs are all numpy arrays, so it is fast to operate on the whole array at once

        if center is None:
            center = contour.mean(axis=0)

        d = contour - center
        # remember that y-axis increases down, so flip the sign
        angle = (numpy.arctan2(-d[:, 1], d[:, 0]) - pi_by_2) % two_pi
        return contour[numpy.argsort(angle)]

    @staticmethod
    def center_cross(points):
        """Finds the intersection of two lines given in Hesse normal form.

        Returns closest integer pixel locations.
        See https://stackoverflow.com/a/383527/5087436
        """

        rho1, theta1 = GenericFinder._hesse_form(points[0], points[2])
        rho2, theta2 = GenericFinder._hesse_form(points[1], points[3])
        if abs(theta1 - theta2) < 1e-6:
            # parallel
            return None

        cos1 = math.cos(theta1)
        sin1 = math.sin(theta1)
        cos2 = math.cos(theta2)
        sin2 = math.sin(theta2)

        denom = cos1*sin2 - sin1*cos2
        x = (sin2*rho1 - sin1*rho2) / denom
        y = (cos1*rho2 - cos2*rho1) / denom
        res = numpy.array((x, y))
        return res

    @staticmethod
    def _hesse_form(pt1, pt2):
        '''Compute the Hesse form for the line through the points'''

        delta = pt2 - pt1
        mag2 = delta.dot(delta)
        vec = pt2 - pt2.dot(delta) * delta / mag2

        rho = math.sqrt(vec.dot(vec))
        if abs(rho) < 1e-6:
            # through 0. Need to compute theta differently
            theta = math.atan2(delta[1], delta[0]) + pi_by_2
            if theta > two_pi:
                theta -= two_pi
        else:
            theta = math.atan2(vec[1], vec[0])

        return rho, theta

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

    @staticmethod
    def distance_angle_from_point(point, height_diff, camera_matrix, distortion_matrix, tilt_angle=0.0):
        '''Calculate the angle and distance from the camera to the center point of the robot
        This routine uses the cameraMatrix from the calibration to convert to normalized coordinates'''

        # use the distortion and camera arrays to correct the location of the center point
        # got this from
        #  https://stackoverflow.com/questions/8499984/how-to-undistort-points-in-camera-shot-coordinates-and-obtain-corresponding-undi

        ptlist = numpy.array([[point]])
        out_pt = cv2.undistortPoints(ptlist, camera_matrix, distortion_matrix, P=camera_matrix)
        undist_center = out_pt[0, 0]

        x_prime = (undist_center[0] - camera_matrix[0, 2]) / camera_matrix[0, 0]
        y_prime = -(undist_center[1] - camera_matrix[1, 2]) / camera_matrix[1, 1]

        # now have all pieces to convert to angle:
        ax = math.atan2(x_prime, 1.0)     # horizontal angle

        # naive expression
        # ay = math.atan2(y_prime, 1.0)     # vertical angle

        # corrected expression.
        # As horizontal angle gets larger, real vertical angle gets a little smaller
        ay = math.atan2(y_prime * math.cos(ax), 1.0)     # vertical angle
        # print("ax, ay", math.degrees(ax), math.degrees(ay))

        # now use the x and y angles to calculate the distance to the target:
        d = height_diff / math.tan(tilt_angle + ay)    # distance to the target

        return d, ax    # return distance and horizontal angle


# --------------------------------------------------------------------------------
# Main routines, used for running the finder by itself for debugging and timing

def process_files(line_finder, input_files, output_dir):
    '''Process the files and output the marked up image'''
    import os.path
    import re

    print('File,Success,Mode,Distance1,RobotAngle1,TargetAngle1,Distance2,RobotAngle2')
    for image_file in input_files:
        # print()
        # print(image_file)
        bgr_frame = cv2.imread(image_file)

        result = line_finder.process_image(bgr_frame)
        print(image_file, result[0], result[1], round(result[2], 1),
              round(math.degrees(result[3]), 1), round(math.degrees(result[4]), 1),
              round(result[5], 1), round(math.degrees(result[6]), 1), sep=',')

        bgr_frame = line_finder.prepare_output_image(bgr_frame)

        outfile = os.path.join(output_dir, os.path.basename(image_file))
        # output as PNG, because that does not do compression
        outfile = re.sub(r'\.jpg$', '.png', outfile, re.IGNORECASE)
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

    print("{0} frames in {1:.3f} seconds = {2:.2f} ms/call, {3:.2f} FPS".format(
        cnt, deltat, 1000.0 * deltat / cnt, cnt / deltat))
    CodeTimer.output_timers()
    return


def main(finder_type):
    '''Main routine for testing a Finder'''
    import argparse
    import camerautil

    parser = argparse.ArgumentParser(description='finder test routine')
    parser.add_argument('--output_dir', help='Output directory for processed images')
    parser.add_argument('--time', action='store_true', help='Loop over files and time it')
    parser.add_argument('--calib_file', help='Calibration file')
    parser.add_argument('--rotate_calib', action='store_true', help='Rotate the calibration file upon load')
    parser.add_argument('input_files', nargs='+', help='input files')

    args = parser.parse_args()
    calib_matrix = None
    dist_matrix = None

    rot = 90 if args.rotate_calib else 0
    if args.calib_file:
        calib_matrix, dist_matrix = camerautil.load_calibration_file(args.calib_file, rotation=rot)
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
