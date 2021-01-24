#!/usr/bin/env python3

import cv2
import numpy
import json
import math


class RRTargetFinder2019(object):
    '''Find switch target for Deep Space 2019'''

    # real world dimensions of the switch target
    # These are the dimensions of one strip
    TARGET_STRIP_WIDTH = 2.0             # inches
    TARGET_STRIP_LENGTH = 5.5            # inches

    # offset from center of the upper corner
    TARGET_STRIP_CORNER_OFFSET = 4.0     # inches

    # tilt of strip from vertical
    TARGET_STRIP_ROT = math.radians(14.5)

    # parameters of the camera mount: tilt (up/down), angle inward, and offset from the robot center
    # NOTE: the rotation matrix for going "camera coord" -> "robot coord" is computed as R_inward * R_tilt
    #   Make sure angles are measured in the right order
    CAMERA_TILT = math.radians(-10.0)
    CAMERA_ANGLE_INWARD = math.radians(-10.0)
    CAMERA_OFFSET_X = -9.25                # inches left/right from center of rotation
    CAMERA_OFFSET_Z = -3.0                 # inches front/back from C.o.R.

    def __init__(self, calib_file, name='rrtarget', finder_id=3.0, intake_finder=None):
        self.name = name
        self.finder_id = finder_id
        self.camera = 'target'
        self.exposure = 1

        self.intake_finder = intake_finder
        self.stream_camera = 'intake' if intake_finder is not None else None

        # Color threshold values, in HSV space
        self.low_limit_hsv = numpy.array((65, 75, 135), dtype=numpy.uint8)
        self.high_limit_hsv = numpy.array((100, 255, 255), dtype=numpy.uint8)

        # distance between the two target bars, in units of the width of a bar
        # tuned value. Seems to work better on pictures from our field elements.
        self.target_separation = 3.45

        # self.width_height_ratio_max = 0.7
        # Allow this to be a bit large to accommodate partially block tape, which appears square
        self.width_height_ratio_max = 1.0
        self.width_height_ratio_min = 0.3

        # max distance in pixels that a contour can be from the guessed location
        self.max_target_dist = 50

        # pixel area of the bounding rectangle - just used to remove stupidly small regions
        self.contour_min_area = 60
        self.contour_max_area = 6000

        # Allowed "error" in the perimeter when fitting using approxPolyDP (in quad_fit)
        self.approx_polydp_error = 0.05     # TODO: maybe tighten this value to get a 5 sided quad fit rather than 4 (tighter=more sides + more accurately)

        # ratio of height to width of one retroreflective strip
        # TODO is this still correct???
        self.one_strip_height_ratio = RRTargetFinder2019.TARGET_STRIP_LENGTH / RRTargetFinder2019.TARGET_STRIP_WIDTH

        self.hsv_frame = None
        self.threshold_frame = None

        # DEBUG values
        self.top_contours = None
        self.target_locations = None
        self.outer_corners = None
        self.target_found = False

        # output results
        self.target_contours = None

        if calib_file:
            with open(calib_file) as f:
                json_data = json.load(f)
                self.cameraMatrix = numpy.array(json_data["camera_matrix"])
                self.distortionMatrix = numpy.array(json_data["distortion"])

        # Corners of the switch target in real world dimensions

        # self.all_target_coords = numpy.concatenate([self.right_strip, self.left_strip])

        # [left_bottom, left_top, right_top, right_bottom]
        self.outside_target_coords = numpy.array([self.left_strip[2], self.left_strip[1],
                                                  self.right_strip[1], self.right_strip[2]])
        # print(self.outside_target_coords)

        return

    @classmethod
    def init_class_variables(cls):
        '''Initialize class-level statics.
        This requires a bit of code, so keep it in a routine.'''

        cos_a = math.cos(cls.TARGET_STRIP_ROT)
        sin_a = math.sin(cls.TARGET_STRIP_ROT)

        pt = [cls.TARGET_STRIP_CORNER_OFFSET, 0.0, 0.0]
        strip = [tuple(pt), ]  # this makes a copy, so we are safe
        pt[0] += cls.TARGET_STRIP_WIDTH * cos_a
        pt[1] += cls.TARGET_STRIP_WIDTH * sin_a
        strip.append(tuple(pt))
        pt[0] += cls.TARGET_STRIP_LENGTH * sin_a
        pt[1] -= cls.TARGET_STRIP_LENGTH * cos_a
        strip.append(tuple(pt))
        pt[0] -= cls.TARGET_STRIP_WIDTH * cos_a
        pt[1] -= cls.TARGET_STRIP_WIDTH * sin_a
        strip.append(tuple(pt))

        cls.right_strip = strip
        # left strip is mirror of right strip
        cls.left_strip = [(-p[0], p[1], p[2]) for p in cls.right_strip]

        # matrices used to compute coordinates
        cls.t_robot = numpy.array(((cls.CAMERA_OFFSET_X,), (0.0,), (cls.CAMERA_OFFSET_Z,)))

        c_a = math.cos(cls.CAMERA_TILT)
        s_a = math.sin(cls.CAMERA_TILT)
        r_tilt = numpy.array(((1.0, 0.0, 0.0), (0.0, c_a, -s_a), (0.0, s_a, c_a)))

        c_a = math.cos(cls.CAMERA_ANGLE_INWARD)
        s_a = math.sin(cls.CAMERA_ANGLE_INWARD)
        r_inward = numpy.array(((c_a, 0.0, -s_a), (0.0, 1.0, 0.0), (s_a, 0.0, c_a)))

        cls.rot_robot = numpy.matmul(r_inward, r_tilt)
        cls.camera_offset_rotated = numpy.matmul(cls.rot_robot.transpose(), -cls.t_robot)

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

        rr = cv2.minAreaRect(contour)
        return numpy.array(cv2.boxPoints(rr))
        # peri = cv2.arcLength(contour, True)
        # return cv2.approxPolyDP(contour, approx_dp_error * peri, True)

    @staticmethod
    def get_outside_corners_single(contour, is_left):
        '''For a single contour, find the 2 corners to the farside of the middle.
        Split the set vertically and the find the farthest in each set.'''

        y_ave = 0.0
        y_min = 10000
        y_max = 0
        for cnr in contour:
            y = cnr[1]
            y_ave += y
            if y > y_max:
                y_max = y
            if y < y_min:
                y_min = y

        y_ave /= len(contour)
        y_delta = (y_max - y_min) / 4

        corners = [None, None]

        # split the loop on left/right on the outside
        #  much faster and simpler than trying to figure out how to flip the comparison on every iteration
        if is_left:
            for cnr in contour:
                # larger y (lower in picture) at index 1
                if abs(cnr[1] - y_ave) > y_delta:
                    index = 1 if cnr[1] > y_ave else 0
                    if corners[index] is None or cnr[0] < corners[index][0]:
                        corners[index] = cnr
                # else:
                #     print('middle corner:', cnr)
        else:
            for cnr in contour:
                # larger y (lower in picture) at index 1
                if abs(cnr[1] - y_ave) > y_delta:
                    index = 1 if cnr[1] > y_ave else 0
                    if corners[index] is None or cnr[0] > corners[index][0]:
                        corners[index] = cnr
                # else:
                #     print('middle corner:', cnr)

        return corners

    @staticmethod
    def get_outside_corners(cnt_left, cnt_right):
        '''Return the outer two corners of a contour'''

        lx_ave = 0
        ly_ave = 0
        rx_ave = 0
        ry_ave = 0

        for cnr in cnt_left:
            lx_ave += cnr[0]
            ly_ave += cnr[1]
        lx_ave /= len(cnt_left)
        ly_ave /= len(cnt_left)
        for cnr in cnt_right:
            rx_ave += cnr[0]
            ry_ave += cnr[1]
        rx_ave /= len(cnt_right)
        ry_ave /= len(cnt_right)

        # now we have the average (center) point of each contour
        left_cnrs = []
        for cnr in cnt_left:
            if cnr[0] < lx_ave:
                left_cnrs.append(cnr)
        right_cnrs = []
        for cnr in cnt_right:
            if cnr[0] > rx_ave:
                right_cnrs.append(cnr)

        # now, if len(cnrs) is greater than two (theoretically should be a max of three),
        # choose cnr with the lowest y (highest on image), remove it, and the cnr with the smallest x

        image_cnrs = []
        if len(left_cnrs) > 2:      # double checking left cnt cnrs

            highest_cnr = [lx_ave, ly_ave]
            for cnr in left_cnrs:
                if cnr[1] < highest_cnr[1]:
                    highest_cnr = cnr
            image_cnrs.append(highest_cnr)

            leftmost_cnr = [lx_ave, ly_ave]
            for cnr in left_cnrs:
                if cnr[0] < leftmost_cnr[0]:
                    leftmost_cnr = cnr
            image_cnrs.append(leftmost_cnr)
        else:
            image_cnrs = left_cnrs

        if len(right_cnrs) > 2:      # double checking right cnt cnrs

            print("Right cnrs are: \n", right_cnrs)
            print("Left cnrs are: \n", left_cnrs)
            highest_cnr = [rx_ave, ry_ave]
            for cnr in right_cnrs:
                if cnr[1] < highest_cnr[1]:
                    highest_cnr = cnr
            image_cnrs.append(highest_cnr)

            rightmost_cnr = [rx_ave, ry_ave]
            for cnr in right_cnrs:
                if cnr[0] > rightmost_cnr[0]:
                    rightmost_cnr = cnr
            image_cnrs.append(rightmost_cnr)
        else:
            image_cnrs.append(right_cnrs[0])    # easier than looping since already know only 2 elements
            image_cnrs.append(right_cnrs[1])

        if len(image_cnrs) == 4:
            return image_cnrs
        return None     # cnr searching did not work if not have 4 pts. Also, solvePNP will crash without exactly 4 pts

    def preallocate_arrays(self, shape):
        '''Pre-allocate work arrays to save time'''

        self.hsv_frame = numpy.empty(shape=shape, dtype=numpy.uint8)
        # threshold_fame is grey, so only 2 dimensions
        self.threshold_frame = numpy.empty(shape=shape[:2], dtype=numpy.uint8)
        return

    def process_image(self, camera_frame):
        '''Main image processing routine'''

        self.target_contours = None

        # debug/output values; clear any values from previous image
        self.top_contours = None
        self.target_locations = []
        self.outer_corners = None
        self.target_found = False

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
        rightlim = shape[1] - 20
        for c in contours:
            center, widths = RRTargetFinder2019.contour_center_width(c)
            area = widths[0] * widths[1]
            left = center[0] - widths[0]/2.0
            right = center[0] + widths[0]/2.0
            # print('area', area, 'left', left)
            if area > self.contour_min_area and area < self.contour_max_area and left > 20 and right < rightlim:
                # TODO: use a simple class? Maybe use "attrs" package?
                contour_list.append({'contour': c, 'center': center, 'widths': widths, 'area': area})

        # Sort the list of contours from biggest area to smallest
        contour_list.sort(key=lambda c: c['area'], reverse=True)

        # DEBUG
        self.top_contours = [x['contour'] for x in contour_list]

        # try only the 3 biggest regions at most
        for candidate_index in range(min(5, len(contour_list))):
            # shape[0] is height, shape[1] is the width
            self.target_contours = self.test_candidate_contour(contour_list, candidate_index, width=shape[1])
            if self.target_contours is not None:
                break

        if self.target_contours is not None:
            # The target was found. Convert to real world co-ordinates.

            cnt_left = numpy.squeeze(self.target_contours[0]).tolist()
            cnt_right = numpy.squeeze(self.target_contours[1]).tolist()

            # Need to convert the contour (integer) into a matrix of corners (float; all 4 outside cnrs)

            # image_corners = RRTargetFinder2019.get_outside_corners(cnt_left, cnt_right)

            # Important to get the corners in the right order, matching the real world ones
            # Remember that y in the image increases *down*
            left = RRTargetFinder2019.get_outside_corners_single(cnt_left, True)
            right = RRTargetFinder2019.get_outside_corners_single(cnt_right, False)

            # [left_bottom, left_top, right_top, right_bottom]
            self.outer_corners = numpy.float32(numpy.array((left[1], left[0], right[0], right[1])))

            # print()
            # print("all cnt_left corners:\n", cnt_left)
            # print("all cnt_right corners:\n", cnt_right)
            # print("Outside corners:\n", self.outer_corners)
            # print("Real World target_coords:\n", self.outside_target_coords)

            retval, rvec, tvec = cv2.solvePnP(self.outside_target_coords, self.outer_corners,
                                              self.cameraMatrix, self.distortionMatrix)
            if retval:
                result = [1.0, self.finder_id, ]
                result.extend(self.compute_output_values(rvec, tvec))
                self.target_found = True
                return result

        # no target found. Return "failure"
        return [0.0, self.finder_id, 0.0, 0.0, 0.0]

    def prepare_output_image(self, input_frame):
        '''Prepare output image for drive station. Draw the found target contour.'''

        if self.intake_finder is not None:
            # hack for now: other camera is rotated!
            output_frame = self.intake_finder.prepare_output_image(input_frame)

            dotrad = 8 if min(output_frame.shape[0], output_frame.shape[1]) < 400 else 12
            if self.target_found:
                # green dot
                cv2.circle(output_frame, (20, 20), dotrad, (0, 255, 0), thickness=dotrad)
            else:
                # red x
                cv2.drawMarker(output_frame, (20, 20), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 2*dotrad, 5)
        else:
            output_frame = input_frame.copy()

            # if self.top_contours:
            #     cv2.drawContours(output_frame, self.top_contours, -1, (0, 0, 255), 1)

            if self.outer_corners is not None:
                cv2.drawContours(output_frame, [numpy.int32(self.outer_corners), ], -1, (0, 0, 255), 1)
                # for cnr in self.outer_corners:
                #     cv2.circle(output_frame, (cnr[0], cnr[1]), 2, (0, 255, 0), -1, lineType=8, shift=0)

            # for loc in self.target_locations:
            #     cv2.drawMarker(output_frame, loc, (0, 255, 255), cv2.MARKER_TILTED_CROSS, 5, 1)

            # if self.target_contours is not None:
            #     cv2.drawContours(output_frame, self.target_contours, -1, (255, 0, 0), 1)

        return output_frame

    def test_candidate_contour(self, contour_list, cand_index, width):
        '''Given a contour as the candidate for the closest (unobscured) target region,
        try to find 1 or 2 other regions which makes up the other side of the target (due to the
        possible splitting of the target by cables lifting the elevator)

        In any test over the list of contours, start *after* cand_index. Those above it have
        been rejected.
        '''

        candidate = contour_list[cand_index]

        # these are going to be used a few times
        cand_x = candidate['center'][0]
        cand_y = candidate['center'][1]
        cand_width = candidate['widths'][0]
        cand_height = candidate['widths'][1]

        # print("Target center at: (" + str(cand_x) + ", " + str(cand_y) + ")")

        wh_ratio = cand_width / cand_height
        # print('wh_ratio:', wh_ratio)
        if wh_ratio > self.width_height_ratio_max or wh_ratio < self.width_height_ratio_min:
            # print('failed ratio test:', wh_ratio)
            return None

        # Based on the candidate location and x-width, compute guesses where the other bar should be
        test_locations = []

        dx = int(self.target_separation * cand_width)
        # print("dx: ", dx)
        x = cand_x + dx
        # print("x1: ", x)
        if x < width:
            test_locations.append((x, cand_y))
        x = cand_x - dx
        # print("x2: ", x)
        if x > 0:
            test_locations.append((x, cand_y))
        # print("Test location coord is: ", test_locations)
        # if neither location is inside the image, reject
        self.target_locations = test_locations
        if not test_locations:
            print('failed: no valid locations')
            return None

        # find the closest contour to either of the guessed locations
        second_cont_index = None
        distance = self.max_target_dist
        for ci in range(cand_index + 1, len(contour_list)):
            c = contour_list[ci]

            for test_loc in test_locations:
                # negative is outside. I want the other sign
                dist = -cv2.pointPolygonTest(c['contour'], test_loc, measureDist=True)
                # print("Current center:", c['center'][0], c['center'][1])
                # print("Dist from cadidate to test location is: %s" % dist)
                # print("Distance (target seperation) is: %s" % distance)
                # diff = dist < distance
                # print("Is dist < distance? %s" % diff)
                if dist < distance:
                    second_cont_index = ci
                    # print("Resetting the second_cont_index to", c['center'][0], c['center'][1])
                    distance = dist
                    if dist <= 0:   # WARNING: the docs say that dist is (-) when outside the contour, not inside?!
                        # guessed location is inside this contour, so this is the one. No need to search further
                        break
            if distance <= 0:
                break
        if second_cont_index is None:
            # print('failed: no second contour found')
            # print('test locations:', test_locations)
            return None

        # see if there is a second contour below. This happens if the peg obscures part of it.
        third_cont_index = None

        second_cont_x = contour_list[second_cont_index]['center'][0]
        second_cont_y = contour_list[second_cont_index]['center'][1]
        second_cont_width = contour_list[second_cont_index]['widths'][0]
        second_cont_height = contour_list[second_cont_index]['widths'][1]

        for cont3 in range(cand_index + 1, len(contour_list)):
            if cont3 == second_cont_index:
                continue

            center3 = contour_list[cont3]['center']
            width3 = contour_list[cont3]['widths']

            # distance between 2nd and 3rd contours should be less than height of main contour
            delta_x = abs(second_cont_x - center3[0])

            # 2nd and 3rd contour need to have almost the same X and almost the same width (correct??)
            if (not delta_x > cand_width) and abs(second_cont_y - center3[1]) < 10 and abs(second_cont_height - width3[0]) < 10:  # too wide to be split contour
                third_cont_index = cont3
                break

        # We now have all the pieces for this candidate

        # Important cut: the actual distance between the two bars in porportion to the actual
        #    average width should be similar to the peg_target_separation variable
        ave_second_bar_x = second_cont_x
        ave_second_bar_width = second_cont_width
        if third_cont_index is not None:
            ave_second_bar_x = (ave_second_bar_x + contour_list[third_cont_index]['center'][0]) / 2
            # just add because side by side they should add to the actual target width
            ave_second_bar_width = (ave_second_bar_width + contour_list[third_cont_index]['widths'][0])

        delta_x = abs(cand_x - ave_second_bar_x)
        ave_width = (cand_width + ave_second_bar_width) / 2
        ratio = delta_x / (ave_width * self.target_separation)
        # print('deltaX', deltaX, aveW, ratio)
        if ratio > 1.3 or ratio < 0.7:
            # not close enough to 1
            # print('failed: bad deltax ratio:', ratio)
            return None

        # DONE! We have a winner! Maybe!
        # Now form a quadrilateral around all the contours.
        # First, combine them using convexHull and the fit a polygon to it. If we get a 4-sided shape we are set.

        all_contours = [candidate['contour'], contour_list[second_cont_index]['contour']]

        # print("Is third contour present: ", (third_cont_index is not None))

        if third_cont_index is not None:    # TODO: if 2 cables on elevator, introduces possibility of perhaps 4 contours
            all_contours.append(contour_list[third_cont_index]['contour'])
            full_strip = numpy.vstack((all_contours[1], all_contours[2]))
            hull_a = cv2.convexHull(full_strip)
        else:
            hull_a = cv2.convexHull(all_contours[1])

        # combined = numpy.vstack(all_contours)
        hull_b = cv2.convexHull(all_contours[0])

        target_contour_a = RRTargetFinder2019.quad_fit(hull_a, self.approx_polydp_error)
        target_contour_b = RRTargetFinder2019.quad_fit(hull_b, self.approx_polydp_error)

        # if (len(target_contour_a) == 4 or len(target_contour_a) == 5) and (len(target_contour_b) == 4 or len(target_contour_b) == 5):
        if len(target_contour_a) <= 5 and len(target_contour_b) <= 5:
            # print("****returning the 2 contours :-)")
            if candidate['center'][0] < contour_list[second_cont_index]['center'][0]:
                # [left_contour, right_contour]
                return [target_contour_b, target_contour_a]  # candidate contour (largest) is under b
            else:
                return [target_contour_a, target_contour_b]
        print("****returning None :-(")
        return None

    def compute_output_values(self, rvec, tvec):
        '''Compute the necessary output distance and angles'''

        x_r_w0 = numpy.matmul(RRTargetFinder2019.rot_robot, tvec) + RRTargetFinder2019.t_robot
        x = x_r_w0[0][0]
        z = x_r_w0[2][0]

        # distance in the horizontal plane between robot center and target
        distance = math.sqrt(x**2 + z**2)

        # horizontal angle between robot center line and target
        angle1 = math.atan2(x, z)

        rot, _ = cv2.Rodrigues(rvec)
        rot_inv = rot.transpose()

        # location of Robot (0,0,0) in World coordinates
        x_w_r0 = numpy.matmul(rot_inv, RRTargetFinder2019.camera_offset_rotated - tvec)

        angle2 = math.atan2(x_w_r0[0][0], x_w_r0[2][0])

        return distance, angle1, angle2


# initialize the class-level "static" variables
# needs to be called after the class definition is loaded.
RRTargetFinder2019.init_class_variables()


def process_files(finder, input_files, output_dir):
    '''Process the files and output the marked up image'''
    import os.path

    for image_file in input_files:
        # print()
        # print(image_file)
        bgr_frame = cv2.imread(image_file)
        result = finder.process_image(bgr_frame)
        strafe_dist = result[2] * math.sin(result[3])
        perp_dist = result[2] * math.cos(result[3])
        print("{},{},{},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f}".format(
            image_file, result[0], result[1], result[2], math.degrees(result[3]), math.degrees(result[4]),
            perp_dist, strafe_dist)
        )

        bgr_frame = finder.prepare_output_image(bgr_frame)

        outfile = os.path.join(output_dir, os.path.basename(image_file))
        # print('{} -> {}'.format(image_file, outfile))
        cv2.imwrite(outfile, bgr_frame)

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

    parser = argparse.ArgumentParser(description='2019 rrtarget finder')
    parser.add_argument('--output_dir', help='Output directory for processed images')
    parser.add_argument('--time', action='store_true', help='Loop over files and time it')
    parser.add_argument('--calib_file', help='Calibration file')
    parser.add_argument('input_files', nargs='+', help='input files')

    args = parser.parse_args()

    # for testing, want all the found contour drawn on this image
    rrtarget_finder = RRTargetFinder2019(args.calib_file, intake_finder=None)

    if args.output_dir is not None:
        process_files(rrtarget_finder, args.input_files, args.output_dir)
    elif args.time:
        time_processing(rrtarget_finder, args.input_files)

    return


# Main routine
# This is for development/testing
if __name__ == '__main__':
    main()
