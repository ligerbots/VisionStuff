#!/usr/bin/env python3

import cv2
import numpy
import json
import math


class SwitchTarget2018(object):
    '''Find switch target for PowerUp 2018'''

    SWITCH_FINDER_MODE = 1.0

    # real world dimensions of the switch target
    # These are the full dimensions around both strips
    TARGET_WIDTH = 8.0           # inches
    TARGET_HEIGHT = 15.3         # inches
    TARGET_STRIP_WIDTH = 2.0     # inches

    def __init__(self, calib_file):
        self.name = 'switch'
        self.finder_id = self.SWITCH_FINDER_MODE
        self.camera = 'intake'
        self.exposure = 6

        # Color threshold values, in HSV space
        self.low_limit_hsv = numpy.array((70, 100, 130), dtype=numpy.uint8)
        self.high_limit_hsv = numpy.array((100, 255, 255), dtype=numpy.uint8)

        # distance between the two target bars, in units of the width of a bar
        # self.target_separation = 3.0  # theoretical value
        # tuned value. Seems to work better on pictures from our field elements.
        self.target_separation = 2.6

        # max distance in pixels that a contour can from the guessed location
        self.max_target_dist = 50

        # pixel area of the bounding rectangle - just used to remove stupidly small regions
        self.contour_min_area = 100

        # Allowed "error" in the perimeter when fitting using approxPolyDP (in quad_fit)
        self.approx_polydp_error = 0.06

        # ratio of height to width of one retroreflective strip
        self.one_strip_height_ratio = SwitchTarget2018.TARGET_HEIGHT / SwitchTarget2018.TARGET_STRIP_WIDTH

        # camera mount angle (radians)
        # NOTE: not sure if this should be positive or negative
        self.tilt_angle = math.radians(-7.5)

        self.hsv_frame = None
        self.threshold_frame = None

        # DEBUG values
        self.top_contours = None
        self.target_locations = None

        # output results
        self.target_contour = None

        with open(calib_file) as f:
            json_data = json.load(f)
            self.cameraMatrix = numpy.array(json_data["camera_matrix"])
            self.distortionMatrix = numpy.array(json_data["distortion"])

        # Corners of the switch target in real world dimensions
        # TODO: Change?
        self.target_coords = numpy.array([[-SwitchTarget2018.TARGET_WIDTH/2.0,  SwitchTarget2018.TARGET_HEIGHT/2.0, 0.0],
                                          [-SwitchTarget2018.TARGET_WIDTH/2.0, -SwitchTarget2018.TARGET_HEIGHT/2.0, 0.0],
                                          [ SwitchTarget2018.TARGET_WIDTH/2.0, -SwitchTarget2018.TARGET_HEIGHT/2.0, 0.0],
                                          [ SwitchTarget2018.TARGET_WIDTH/2.0,  SwitchTarget2018.TARGET_HEIGHT/2.0, 0.0]])

        return

    def set_color_thresholds(self, hue_low, hue_high, sat_low, sat_high, val_low, val_high):
        self.low_limit_hsv = numpy.array((hue_low, sat_low, val_low), dtype=numpy.uint8)
        self.high_limit_hsv = numpy.array((hue_high, sat_high, val_high), dtype=numpy.uint8)
        return

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

    @staticmethod
    def sort_corners(cnrlist):
        '''Sort a list of 4 corners so that it goes in a known order. Does it in place!!'''
        cnrlist.sort()

        # now, swap the pairs to make sure in proper Y order
        if cnrlist[0][1] > cnrlist[1][1]:
            cnrlist[0], cnrlist[1] = cnrlist[1], cnrlist[0]
        if cnrlist[2][1] < cnrlist[3][1]:
            cnrlist[2], cnrlist[3] = cnrlist[3], cnrlist[2]
        return

    def preallocate_arrays(self, shape):
        '''Pre-allocate work arrays to save time'''

        self.hsv_frame = numpy.empty(shape=shape, dtype=numpy.uint8)
        # threshold_fame is grey, so only 2 dimensions
        self.threshold_frame = numpy.empty(shape=shape[:2], dtype=numpy.uint8)
        return

    def process_image(self, camera_frame):
        '''Main image processing routine'''

        self.target_contour = None

        # DEBUG values
        self.top_contours = None
        self.target_locations = []

        shape = camera_frame.shape
        if self.hsv_frame is None or self.hsv_frame.shape != shape:
            self.preallocate_arrays(shape)

        self.hsv_frame = cv2.cvtColor(camera_frame, cv2.COLOR_BGR2HSV, dst=self.hsv_frame)
        self.threshold_frame = cv2.inRange(self.hsv_frame, self.low_limit_hsv, self.high_limit_hsv,
                                           dst=self.threshold_frame)

        # OpenCV 3 returns 3 parameters!
        # Only need the contours variable
        _, contours, _ = cv2.findContours(self.threshold_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        contour_list = []
        for c in contours:
            center, widths = SwitchTarget2018.contour_center_width(c)
            area = widths[0] * widths[1]
            if area > self.contour_min_area:
                # TODO: use a simple class? Maybe use "attrs" package?
                contour_list.append({'contour': c, 'center': center, 'widths': widths, 'area': area})

        # Sort the list of contours from biggest area to smallest
        contour_list.sort(key=lambda c: c['area'], reverse=True)

        # DEBUG
        self.top_contours = [x['contour'] for x in contour_list[0:2]]

        # try only the 5 biggest regions
        for candidate_index in range(min(5, len(contour_list))):
            self.target_contour = self.test_candidate_contour(contour_list, candidate_index, width=shape[1])
            if self.target_contour is not None:
                break

        if self.target_contour is not None:
            # The target was found. Convert to real world co-ordinates.

            # Need to convert the contour (integer) into a matrix of corners (float)
            # Contour starts at arbitary place around the quad, so need to sort after
            # Could also do this by finding the first point, but that is complicated, probably no faster
            cnrlist = []
            for cnr in self.target_contour:
                cnrlist.append((float(cnr[0][0]), float(cnr[0][1])))
            SwitchTarget2018.sort_corners(cnrlist)   # in place sort
            image_corners = numpy.array(cnrlist)

            retval, rvec, tvec = cv2.solvePnP(self.target_coords, image_corners,
                                              self.cameraMatrix, self.distortionMatrix)
            if retval:
                result = [1.0, SwitchTarget2018.SWITCH_FINDER_MODE, ]
                result.extend(self.compute_output_values(rvec, tvec))
                return result

        # no target found. Return "failure"
        return [0.0, SwitchTarget2018.SWITCH_FINDER_MODE, 0.0, 0.0, 0.0]

    def prepare_output_image(self, input_frame):
        '''Prepare output image for drive station. Draw the found target contour.'''

        output_frame = input_frame.copy()

        for loc in self.target_locations:
            cv2.drawMarker(output_frame, loc, (0, 255, 255), cv2.MARKER_TILTED_CROSS, 15, 3)

        if self.top_contours:
            cv2.drawContours(output_frame, self.top_contours, -1, (0, 0, 255), 2)

        if self.target_contour is not None:
            cv2.drawContours(output_frame, [self.target_contour], -1, (255, 0, 0), 2)

        return output_frame

    def test_candidate_contour(self, contour_list, cand_index, width):
        '''Given a contour as the candidate for the closest (unobscured) target region,
        try to find 1 or 2 other regions which make up the other side of the target

        In any test over the list of contours, start *after* cand_index. Those above it have
        been rejected.
        '''

        candidate = contour_list[cand_index]

        # these are going to be used a few times
        cand_x = candidate['center'][0]
        cand_width = candidate['widths'][0]
        cand_height = candidate['widths'][1]

        # Test that the candidate region has roughly the proportions of the real retroreflective tape
        ratio = cand_height / (cand_width * self.one_strip_height_ratio)
        if ratio < 0.5 or ratio > 1.3:
            return None

        # Based on the candidate location and x-width, compute guesses where the other bar should be
        test_locations = []
        cy = candidate['center'][1]
        dx = int(self.target_separation * cand_width)
        x = cand_x + dx
        if x < width:
            test_locations.append((x, cy))
        x = cand_x - dx
        if x > 0:
            test_locations.append((x, cy))

        # DEBUG
        self.target_locations = test_locations

        # if neither location is inside the image, reject
        if not test_locations:
            return None

        # find the closest contour to either of the guessed locations
        second_cont_index = None
        distance = self.max_target_dist
        for ci in range(cand_index+1, len(contour_list)):
            c = contour_list[ci]

            for test_loc in test_locations:
                # negative is outside. I want the other sign
                dist = -cv2.pointPolygonTest(c['contour'], test_loc, measureDist=True)
                if dist < distance:
                    second_cont_index = ci
                    distance = dist
                    if dist <= 0:
                        # guessed location is inside this contour, so this is the one. No need to search further
                        break

            if distance <= 0:
                break
        if second_cont_index is None:
            return None

        # We now have all the pieces for this candidate

        # Important cut: the actual distance between the two bars in proportion to the actual
        #    average width should be similar to the target_separation variable

        delta_x = abs(cand_x - contour_list[second_cont_index]['center'][0])
        ave_width = (cand_width + contour_list[second_cont_index]['widths'][0]) / 2
        ratio = delta_x / (ave_width * self.target_separation)
        # print('delta_x', delta_x, ave_width, ratio)
        if ratio > 1.3 or ratio < 0.5:
            # not close enough to 1
            return None

        # DONE! We have a winner! Maybe!
        # Now form a quadrilateral around all the contours.
        # First, combine them using convexHull and the fit a polygon to it. If we get a 4-sided shape we are set.

        all_contours = [candidate['contour'], contour_list[second_cont_index]['contour']]

        combined = numpy.vstack(all_contours)
        hull = cv2.convexHull(combined)

        target_contour = SwitchTarget2018.quad_fit(hull, self.approx_polydp_error)
        if len(target_contour) == 4:
            return target_contour

        return None

    def compute_output_values(self, rvec, tvec):
        '''Compute the necessary output distance and angles'''

        # The tilt angle only affects the distance and angle1 calcs

        x = tvec[0][0]
        z = math.sin(self.tilt_angle) * tvec[1][0] + math.cos(self.tilt_angle) * tvec[2][0]

        # distance in the horizontal plane between camera and target
        distance = math.sqrt(x**2 + z**2)

        # horizontal angle between camera center line and target
        angle1 = math.atan2(x, z)

        rot, _ = cv2.Rodrigues(rvec)
        rot_inv = rot.transpose()
        pzero_world = numpy.matmul(rot_inv, -tvec)
        angle2 = math.atan2(pzero_world[0][0], pzero_world[2][0])

        return distance, angle1, angle2


# --------------------------------------------------------------------------------

def process_files(switch_target_processor, input_files, output_dir):
    '''Process the files and output the marked up image'''
    import os.path

    success = 0
    for image_file in input_files:
        bgr_frame = cv2.imread(image_file)
        result = switch_target_processor.process_image(bgr_frame)
        print('{} {} {} {:.2f} {:.2f} {:.2f}'.format(image_file, result[0], result[1], result[2], math.degrees(result[3]),  math.degrees(result[4])))
        if result[0]:
            success += 1

        switch_target_processor.prepare_output_image(bgr_frame)
        outfile = os.path.join(output_dir, os.path.basename(image_file))
        # print('{} -> {}'.format(image_file, outfile))
        cv2.imwrite(outfile, bgr_frame)

    print('Found target in {} out of {} files'.format(success, len(input_files)))
    return


def time_processing(switch_target_processor, input_files):
    '''Time the processing of the test files'''

    from codetimer import CodeTimer
    from time import time

    startt = time()

    cnt = 0
    for _ in range(100):
        for image_file in input_files:
            with CodeTimer("Read Image"):
                bgr_frame = cv2.imread(image_file)

            with CodeTimer("Main Processing"):
                switch_target_processor.process_image(bgr_frame)

            cnt += 1

    deltat = time() - startt

    print("{0} frames in {1:.3f} seconds = {2:.2f} msec/call, {3:.2f} FPS".format(
        cnt, deltat, 1000.0 * deltat / cnt, cnt / deltat))
    CodeTimer.outputTimers()
    return


def main():
    '''Main routine, for testing'''
    import argparse

    parser = argparse.ArgumentParser(description='2018 switch target')
    parser.add_argument('--output_dir', help='Output directory for processed images')
    parser.add_argument('--time', action='store_true', help='Loop over files and time it')
    parser.add_argument('--calib', help='Calibration file')
    parser.add_argument('input_files', nargs='+', help='input files')

    args = parser.parse_args()

    switch_target_processor = SwitchTarget2018(args.calib)
    if args.output_dir is not None:
        process_files(switch_target_processor, args.input_files, args.output_dir)
    elif args.time:
        time_processing(switch_target_processor, args.input_files)

    return


# Main routine
# This is for development/testing
if __name__ == '__main__':
    main()
