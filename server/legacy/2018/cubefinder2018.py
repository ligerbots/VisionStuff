#!/usr/bin/env python3

import cv2
import numpy
import json
import math


class CubeFinder2018(object):
    '''Find power cube for PowerUp 2018'''

    CUBE_FINDER_MODE = 2.0

    # CUBE_HEIGHT = 11    #inches
    # CUBE_WIDTH = 13     #inches
    # CUBE_LENGTH = 13    #inches

    HFOV = 64.0                  # horizontal angle of the field of view
    VFOV = 52.0                  # vertical angle of the field of view

    # create imaginary view plane on 3d coords to get height and width
    # place the view place on 3d coordinate plane 1.0 unit away from (0, 0) for simplicity
    VP_HALF_WIDTH = math.tan(math.radians(HFOV)/2.0)  # view plane 1/2 height
    VP_HALF_HEIGHT = math.tan(math.radians(VFOV)/2.0)  # view plane 1/2 width

    def __init__(self, calib_file):
        self.name = 'cube'
        self.finder_id = self.CUBE_FINDER_MODE
        self.camera = 'intake'
        self.exposure = 0

        # Color threshold values, in HSV space -- TODO: in 2018 server (yet to be created) make the low and high hsv limits
        # individual properties
        self.low_limit_hsv = numpy.array((25, 95, 95), dtype=numpy.uint8)
        self.high_limit_hsv = numpy.array((75, 255, 255), dtype=numpy.uint8)

        # pixel area of the bounding rectangle - just used to remove stupidly small regions
        self.contour_min_area = 250

        # maximum no. of vertices in the fitted contour
        # 12 = max # of corners if all corners are flat
        # seems to be OK with 8. Allows a few truncated corners.
        self.max_num_vertices = 8

        # Allowed "error" in the perimeter when fitting using approxPolyDP (in quad_fit)
        self.approx_polydp_error = 0.015

        self.erode_kernel = numpy.ones((3, 3), numpy.uint8)
        self.erode_iterations = 0

        # some variables to save results for drawing
        self.center = None
        self.hull_fit = None
        self.biggest_contour = None

        self.cameraMatrix = None
        self.distortionMatrix = None
        if calib_file:
            with open(calib_file) as f:
                json_data = json.load(f)
                self.cameraMatrix = numpy.array(json_data["camera_matrix"])
                self.distortionMatrix = numpy.array(json_data["distortion"])

        # self.target_coords = numpy.array([[-CubeFinder2018.CUBE_LENGTH/2.0, -CubeFinder2018.CUBE_HEIGHT/2.0, 0.0],
        #                                   [-CubeFinder2018.CUBE_LENGTH/2.0,  CubeFinder2018.CUBE_HEIGHT/2.0, 0.0],
        #                                   [ CubeFinder2018.CUBE_LENGTH/2.0,  CubeFinder2018.CUBE_HEIGHT/2.0, 0.0],
        #                                   [ CubeFinder2018.CUBE_LENGTH/2.0, -CubeFinder2018.CUBE_HEIGHT/2.0, 0.0]])

        self.tilt_angle = math.radians(-7.5)  # camera mount angle (radians)
        self.camera_height = 23.0            # height of camera off the ground (inches)
        self.target_height = 0.0             # height of target off the ground (inches)

        return

    def set_color_thresholds(self, hue_low, hue_high, sat_low, sat_high, val_low, val_high):
        self.low_limit_hsv = numpy.array((hue_low, sat_low, val_low), dtype=numpy.uint8)
        self.high_limit_hsv = numpy.array((hue_high, sat_high, val_high), dtype=numpy.uint8)
        return

    @staticmethod
    def contour_center_width(contour):
        '''Find boundingRect of contour, but return center, width, and height'''

        x, y, w, h = cv2.boundingRect(contour)
        return (x + int(w / 2), y + int(h / 2)), (w, h)

    @staticmethod
    def quad_fit(contour, approx_dp_error):
        '''Simple polygon fit to contour with error related to perimeter'''

        peri = cv2.arcLength(contour, True)
        return cv2.approxPolyDP(contour, approx_dp_error * peri, True)

    @staticmethod
    def sort_corners(cnrlist, check):
        '''Sort a list of corners -- if check == true then returns x sorted 1st, y sorted 2nd. Otherwise the opposite'''

        # recreate the list of corners to get rid of excess dimensions
        corners = []
        for c in cnrlist:
            corners.append(c[0].tolist())

        # sort the corners by x values (1st column) first and then by y values (2nd column)
        if check:
            return sorted(corners, key=lambda x: (x[0], x[1]))
        # y's first then x's
        else:
            return sorted(corners, key=lambda x: (x[1], x[0]))

    @staticmethod
    def split_xs_ys(corners):
        '''Split a list of corners into sorted lists of x and y values'''
        xs = []
        ys = []

        for i in range(len(corners)):
            xs.append(corners[i][0])
            ys.append(corners[i][1])
        #sort the lists highest to lowest
        xs.sort(reverse=True)
        ys.sort(reverse=True)
        return xs, ys

    @staticmethod
    def choose_corners_frontface(img, cnrlist):
        '''Sort a list of corners and return the bottom and side corners (one side -- 3 in total - .: or :.)
        of front face'''
        corners = CubeFinder2018.sort_corners(cnrlist, False)    # get rid of extra dimensions

        happy_corner = corners[len(corners) - 1]
        lonely_corner = corners[len(corners) - 2]

        xs, ys = CubeFinder2018.split_xs_ys(corners)

        #lonely corner is green and happy corner is red
        #cv2.circle(img, (lonely_corner[0], lonely_corner[1]), 5, (0, 255, 0), thickness=10, lineType=8, shift=0)
        #cv2.circle(img, (happy_corner[0], happy_corner[1]), 5, (0, 0, 255), thickness=10, lineType=8, shift=0)

        corners = CubeFinder2018.sort_corners(cnrlist, True)

        if happy_corner[0] > lonely_corner[0]:
            top_corner = corners[len(corners) - 1]
        else:
            top_corner = corners[0]
        #top corner is in blue
        #cv2.circle(img, (top_corner[0], top_corner[1]), 5, (255, 0, 0), thickness=10, lineType=8, shift=0)
        return ([lonely_corner, happy_corner, top_corner])

    @staticmethod
    def get_cube_facecenter(img, cnrlist):
        '''Compute the center of a cube face from a list of the three face corners'''
        #get the three corners of the front face
        front_corners = CubeFinder2018.choose_corners_frontface(img, cnrlist)
        #average of x, y values of opposite corners of front face of cube
        x = int((front_corners[0][0] + front_corners[2][0]) / 2)
        y = int((front_corners[0][1] + front_corners[2][1]) / 2)

        #middle point in white
        #cv2.circle(img, (x, y), 5, (255, 255, 255), thickness=10, lineType=8, shift=0)
        return [x, y]       #return center point of cube front face'''


    @staticmethod
    def get_cube_center(img, cnrlist):
        '''return the center of the cube'''
        #sort just to format correctly -- get rid of extra dimensions
        corners = CubeFinder2018.sort_corners(cnrlist, True)
        #xs and ys only needed for drawing the point on the image
        xs, ys = CubeFinder2018.split_xs_ys(corners)

        sum_x = 0
        sum_y = 0
        for corner in corners:
            sum_x += corner[0]
            sum_y += corner[1]
        center = numpy.array([ int(sum_x / (len(corners) / 2)), int(sum_y / (len(corners) / 2)) ])
        #cv2.circle(img, (center[0], center[1]), 5, (255, 0, 0), thickness=50, lineType=8, shift=0)
        return sum_x / (len(corners) / 2), sum_y / (len(corners) / 2)

    @staticmethod
    def get_cube_bottomcenter(cnrlist):
        '''return the center of the bottom-front side of the cube'''
        corners = CubeFinder2018.sort_corners(cnrlist, False)

        bottom_corner_a = corners[-1]
        bottom_corner_b = corners[-2]

        center = [int((bottom_corner_a[0] + bottom_corner_b[0]) / 2), int((bottom_corner_a[1] + bottom_corner_b[1]) / 2)]
        return center

    def get_cube_values_calib(self, center):
        '''Calculate the angle and distance from the camera to the center point of the robot
        This routine uses the cameraMatrix from the calibration to convert to normalized coordinates'''

        # use the distortion and camera arrays to correct the location of the center point
        # got this from
        #  https://stackoverflow.com/questions/8499984/how-to-undistort-points-in-camera-shot-coordinates-and-obtain-corresponding-undi
        # (Needs lots of brackets! Buy shares in the Bracket Company now!)

        center_np = numpy.array([[[float(self.center[0]), float(self.center[1])]]])
        out_pt = cv2.undistortPoints(center_np, self.cameraMatrix, self.distortionMatrix, P=self.cameraMatrix)
        undist_center = out_pt[0, 0]

        x_prime = (undist_center[0] - self.cameraMatrix[0, 2]) / self.cameraMatrix[0, 0]
        y_prime = -(undist_center[1] - self.cameraMatrix[1, 2]) / self.cameraMatrix[1, 1]

        # now have all pieces to convert to angle:
        ax = math.atan2(x_prime, 1.0)     # horizontal angle

        # naive expression
        # ay = math.atan2(y_prime, 1.0)     # vertical angle

        # corrected expression.
        # As horizontal angle gets larger, real vertical angle gets a little smaller
        ay = math.atan2(y_prime * math.cos(ax), 1.0)     # vertical angle
        # print("ax, ay", math.degrees(ax), math.degrees(ay))

        # now use the x and y angles to calculate the distance to the target:
        d = (self.target_height - self.camera_height) / math.tan(self.tilt_angle + ay)    # distance to the target

        return ax, d    # return horizontal angle and distance

    def get_cube_values(self, center, shape):
        '''Calculate the angle and distance from the camera to the center point of the robot
        This routine uses the FOV numbers and the default center to convert to normalized coordinates'''

        # center is in pixel coordinates, 0,0 is the upper-left, positive down and to the right
        # (nx,ny) = normalized pixel coordinates, 0,0 is the center, positive right and up
        # WARNING: shape is (h, w, nbytes) not (w,h,...)
        image_w = shape[1] / 2.0
        image_h = shape[0] / 2.0

        # NOTE: the 0.5 is to place the location in the center of the pixel
        nx = (center[0] - image_w + 0.5) / image_w
        ny = (image_h - 0.5 - center[1]) / image_h

        # convert normal pixel coords to pixel coords
        x = CubeFinder2018.VP_HALF_WIDTH * nx
        y = CubeFinder2018.VP_HALF_HEIGHT * ny
        # print("values", center[0], center[1], nx, ny, x, y)

        # now have all pieces to convert to angle:
        ax = math.atan2(x, 1.0)     # horizontal angle

        # naive expression
        # ay = math.atan2(y, 1.0)     # vertical angle

        # corrected expression.
        # As horizontal angle gets larger, real vertical angle gets a little smaller
        ay = math.atan2(y * math.cos(ax), 1.0)     # vertical angle
        # print("ax, ay", math.degrees(ax), math.degrees(ay))

        # now use the x and y angles to calculate the distance to the target:
        d = (self.target_height - self.camera_height) / math.tan(self.tilt_angle + ay)    # distance to the target

        return ax, d    # return horizontal angle and distance

    def process_image(self, camera_frame):
        '''Main image processing routine'''

        # clear out result variables
        angle = None
        distance = None
        self.center = None
        self.hull_fit = None
        self.biggest_contour = None

        hsv_frame = cv2.cvtColor(camera_frame, cv2.COLOR_BGR2HSV)
        threshold_frame = cv2.inRange(hsv_frame, self.low_limit_hsv, self.high_limit_hsv)

        if self.erode_iterations > 0:
            erode_frame = cv2.erode(threshold_frame, self.erode_kernel, iterations=self.erode_iterations)
        else:
            erode_frame = threshold_frame

        # OpenCV 3 returns 3 parameters!
        # Only need the contours variable
        res = cv2.findContours(erode_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(res) == 2:
            contours = res[0]
        else:
            contours = res[1]

        contour_list = []
        for c in contours:
            center, widths = CubeFinder2018.contour_center_width(c)
            area = widths[0] * widths[1]
            if area > self.contour_min_area:
                # TODO: use a simple class? Maybe use "attrs" package?
                contour_list.append({'contour': c, 'center': center, 'widths': widths, 'area': area})

        # Sort the list of contours from biggest area to smallest
        contour_list.sort(key=lambda c: c['area'], reverse=True)

        # test first 3 biggest contours only (optimization)
        for cnt in contour_list[0:3]:
            self.hull_fit = self.test_candidate_contour(cnt)
            if self.hull_fit is not None:
                self.biggest_contour = cnt['contour']
                break

        # NOTE: testing a list returns true if there is something in the list
        if self.hull_fit is not None:
            self.center = CubeFinder2018.get_cube_bottomcenter(self.hull_fit)

            # print('center', self.center)
            if self.cameraMatrix is not None:
                angle, distance = self.get_cube_values_calib(self.center)
            else:
                angle, distance = self.get_cube_values(self.center, camera_frame.shape)

        # return values: (success, cube or switch, distance, angle, -- still deciding here?)
        if distance is None or angle is None:
            return (0.0, CubeFinder2018.CUBE_FINDER_MODE, 0.0, 0.0, 0.0)

        return (1.0, CubeFinder2018.CUBE_FINDER_MODE, distance, angle, 0.0)

    def test_candidate_contour(self, contour_entry):
        cnt = contour_entry['contour']

        real_area = cv2.contourArea(cnt)
        # print('areas:', real_area, contour_entry['area'], real_area / contour_entry['area'])
        if real_area / contour_entry['area'] > 0.5:
            hull = cv2.convexHull(cnt)
            # hull_fit contains the corners for the contour
            hull_fit = CubeFinder2018.quad_fit(hull, self.approx_polydp_error)

            vertices = len(hull_fit)
            if vertices >= 4 and vertices <= self.max_num_vertices:
                return hull_fit

        return None

    def prepare_output_image(self, input_frame):
        '''Prepare output image for drive station. Draw the found target contour.'''

        output_frame = input_frame.copy()

        # Draw the contour on the image
        if self.biggest_contour is not None:
            cv2.drawContours(output_frame, [self.biggest_contour], -1, (0, 0, 255), 2)

        if self.hull_fit is not None:
            cv2.drawContours(output_frame, [self.hull_fit], -1, (255, 0, 0), 2)

        if self.center is not None:
            cv2.drawMarker(output_frame, tuple(self.center), (0, 255, 255), cv2.MARKER_CROSS, 10, 2)
            # cv2.circle(output_frame, tuple(self.center), 5, (255, 0, 0), thickness=10, lineType=8, shift=0)

        return output_frame


def process_files(cube_processor, input_files, output_dir):
    '''Process the files and output the marked up image'''
    import os.path

    for image_file in input_files:
        # print()
        # print(image_file)
        bgr_frame = cv2.imread(image_file)
        result = cube_processor.process_image(bgr_frame)
        print(image_file, result[0], result[1], result[2], math.degrees(result[3]), math.degrees(result[4]))

        cube_processor.prepare_output_image(bgr_frame)

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
    CodeTimer.outputTimers()
    return


def main():
    '''Main routine'''
    import argparse

    parser = argparse.ArgumentParser(description='2018 cube finder')
    parser.add_argument('--output_dir', help='Output directory for processed images')
    parser.add_argument('--time', action='store_true', help='Loop over files and time it')
    parser.add_argument('--calib_file', help='Calibration file')
    parser.add_argument('input_files', nargs='+', help='input files')

    args = parser.parse_args()

    cube_processor = CubeFinder2018(args.calib_file)

    if args.output_dir is not None:
        process_files(cube_processor, args.input_files, args.output_dir)
    elif args.time:
        time_processing(cube_processor, args.input_files)

    return


# Main routine
# This is for development/testing
if __name__ == '__main__':
    main()
