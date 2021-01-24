#!/usr/bin/env python3

import cv2
import numpy
import json
import math

class LineFinder2019(object):
    '''Find line following for Deep Space 2019'''

    HFOV = 64.0                  # horizontal angle of the field of view
    VFOV = 52.0                  # vertical angle of the field of view

    # create imaginary view plane on 3d coords to get height and width
    # place the view place on 3d coordinate plane 1.0 unit away from (0, 0) for simplicity
    VP_HALF_WIDTH = math.tan(math.radians(HFOV)/2.0)  # view plane 1/2 height
    VP_HALF_HEIGHT = math.tan(math.radians(VFOV)/2.0)  # view plane 1/2 width

    def __init__(self, calib_file):
        self.name = "linefinder"
        self.finder_id = 4.0   # id needs to be float! "id" is a reserved word.
        self.desired_camera = "floor"        # string with camera name
        self.exposure = 0

        # Color threshold values, in HSV space -- TODO: in 2019 server (yet to be created) make the low and high hsv limits
        # individual properties
        self.low_limit_hsv = numpy.array((0, 0, 185), dtype=numpy.uint8)
        self.high_limit_hsv = numpy.array((255, 60, 255), dtype=numpy.uint8)

        # Allowed "error" in the perimeter when fitting using approxPolyDP (in quad_fit)
        self.approx_polydp_error = 0.015

        self.erode_kernel = numpy.ones((3, 3), numpy.uint8)
        self.erode_iterations = 0

        # some variables to save results for drawing
        self.hull_fit = None
        self.biggest_contour = None
        self.contour_min_area = 20
        #TODO: make sure min_contour_area is correct -- what if it doesn't see the whole line...
        #line is 18 inches long and 2 inches wide, area of 36
        self.max_num_vertices = 7   #TODO test the max and min

        self.cameraMatrix = None
        self.distortionMatrix = None
        if calib_file:
            with open(calib_file) as f:
                json_data = json.load(f)
                self.cameraMatrix = numpy.array(json_data["camera_matrix"])
                self.distortionMatrix = numpy.array(json_data["distortion"])

        self.tilt_angle = math.radians(0)  # camera mount angle (radians)
        self.camera_height = 0.0            # height of camera off the ground (inches)
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
    
    def test_candidate_contour(self, test_pass, contour_entry, frame_shape):
        cnt = contour_entry['contour']

        if test_pass == 0:
            c = contour_entry['center']
            w = contour_entry['widths']
            if c[0] + (w[0] / 2.0) >= frame_shape[1] - 5 or c[0] - (w[0] / 2.0) <= 5:
                #print("Failed edge cut")
                return None

        #hull = cv2.convexHull(cnt)
        # hull_fit contains the corners for the contour
        hull_fit = LineFinder2019.quad_fit(cnt, self.approx_polydp_error)

        vertices = len(hull_fit)
        print("Vertices %d" % vertices)
        if vertices >= 4 and vertices <= self.max_num_vertices:
            return hull_fit

        return None

    def get_line_values_calib(self, center):
        '''Calculate the angle and distance from the camera to the center point of the robot
        This routine uses the cameraMatrix from the calibration to convert to normalized coordinates'''

        # use the distortion and camera arrays to correct the location of the center point
        # got this from
        # https://stackoverflow.com/questions/8499984/how-to-undistort-points-in-camera-shot-coordinates-and-obtain-corresponding-undi
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


    def process_image(self, camera_frame):
        '''Main image processing routine'''

        # clear out result variables each iteration
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
        _, contours, _ = cv2.findContours(erode_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        contour_list = []
        for c in contours:
            center, widths = LineFinder2019.contour_center_width(c)
            area = widths[0] * widths[1]
            if area > self.contour_min_area:
                # TODO: use a simple class? Maybe use "attrs" package?
                contour_list.append({'contour': c, 'center': center, 'widths': widths, 'area': area})

        # Sort the list of contours from biggest area to smallest
        contour_list.sort(key=lambda c: c['area'], reverse=True)
        self.found_contours = contour_list
        # test first 3 biggest contours only (optimization)
        for test_pass in [0, 1]:
            for cnt in contour_list[0:3]:
                self.hull_fit = self.test_candidate_contour(test_pass, cnt, camera_frame.shape)
                self.biggest_contour = cnt['contour']
                if self.hull_fit is not None:
                    break
            if self.hull_fit is not None:
                break
            print("Pass 0 failed")
        # NOTE: testing a list returns true if there is something in the list
        #if self.hull_fit is not None:
        #    if self.cameraMatrix is not None:
        #        angle, distance = self.get_cube_values_calib(self.center)
        #    else:
        #        angle, distance = self.get_cube_values(self.center, camera_frame.shape)

        # return values: (success, cube or switch, distance, angle, -- still deciding here?)
        #if distance is None or angle is None:
        #    return (0.0, CubeFinder2018.CUBE_FINDER_MODE, 0.0, 0.0, 0.0)

        #return (1.0, CubeFinder2018.CUBE_FINDER_MODE, distance, angle, 0.0)
        return (1.0, self.finder_id, 0.0, 0.0, 0.0)

    def prepare_output_image(self, input_frame):
        '''Prepare output image for drive station. Draw the found target contour.'''

        output_frame = input_frame.copy()

        for c in self.found_contours:
            cv2.drawContours(output_frame, [c['contour']], -1, (255, 0, 255), 2)

        # Draw the contour on the image
        #if self.biggest_contour is not None:
        #    print("Drawing contour on image...")
        #    cv2.drawContours(output_frame, [self.biggest_contour], -1, (255, 0, 255), 2)

        if self.hull_fit is not None:
            print("Drawing hull_fit on image...")
            cv2.drawContours(output_frame, [self.hull_fit], -1, (255, 0, 0), 2)

        return output_frame


def process_files(finder, input_files, output_dir):
    '''Process the files and output the marked up image'''
    import os.path

    for image_file in input_files:
        # print()
        # print(image_file)
        bgr_frame = cv2.imread(image_file)
        result = finder.process_image(bgr_frame)
        #print(image_file, result[0], result[1], result[2], math.degrees(result[3]), math.degrees(result[4]))

        output_frame = finder.prepare_output_image(bgr_frame)

        outfile = os.path.join(output_dir, os.path.basename(image_file))
        # print('{} -> {}'.format(image_file, outfile))
        cv2.imwrite(outfile, output_frame)

        # cv2.imshow("Window", bgr_frame)
        # q = cv2.waitKey(-1) & 0xFF
        # if q == ord('q'):
        #     break
    return


def time_processing(finder, input_files):
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
                finder.process_image(bgr_frame)

            cnt += 1

    deltat = time() - startt

    print("{0} frames in {1:.3f} seconds = {2:.2f} msec/call, {3:.2f} FPS".format(
        cnt, deltat, 1000.0 * deltat / cnt, cnt / deltat))
    CodeTimer.outputTimers()
    return


def main():
    '''Main routine'''
    import argparse

    parser = argparse.ArgumentParser(description='2019 line finder')
    parser.add_argument('--output_dir', help='Output directory for processed images')
    parser.add_argument('--time', action='store_true', help='Loop over files and time it')
    parser.add_argument('--calib_file', help='Calibration file')
    parser.add_argument('input_files', nargs='+', help='input files')

    args = parser.parse_args()

    line_finder = LineFinder2019(args.calib_file)

    if args.output_dir is not None:
        process_files(line_finder, args.input_files, args.output_dir)
    elif args.time:
        time_processing(line_finder, args.input_files)

    return


# Main routine
# This is for development/testing
if __name__ == '__main__':
    main()
