#!/usr/bin/env python3

# Vision finder to find the retro-reflective target around the goal

import cv2
import numpy
import math

from finders.genericfinder import GenericFinder, main


class GoalFinder2020(GenericFinder):
    '''Find high goal target for Infinite Recharge 2020'''

    # real world dimensions of the goal target
    # These are the full dimensions around both strips
    TARGET_STRIP_LENGTH = 19.625    # inches
    TARGET_HEIGHT = 17.0            # inches
    TARGET_TOP_WIDTH = 39.25        # inches
    TARGET_BOTTOM_WIDTH = TARGET_TOP_WIDTH - 2*TARGET_STRIP_LENGTH*math.cos(math.radians(60))

    # [0, 0] is center of the quadrilateral drawn around the high goal target
    # [top_left, bottom_left, bottom_right, top_right]
    real_world_coordinates = numpy.array([
        [-TARGET_TOP_WIDTH / 2, TARGET_HEIGHT / 2, 0.0],
        [-TARGET_BOTTOM_WIDTH / 2, -TARGET_HEIGHT / 2, 0.0],
        [TARGET_BOTTOM_WIDTH / 2, -TARGET_HEIGHT / 2, 0.0],
        [TARGET_TOP_WIDTH / 2, TARGET_HEIGHT / 2, 0.0]
    ])

    # camera offsets and tilt angles
    CAMERA_TILT = math.radians(25.0)
    CAMERA_OFFSET_X = -7.5
    CAMERA_OFFSET_Z = 0.0

    def __init__(self, calib_matrix=None, dist_matrix=None):
        super().__init__('goalfinder', camera='shooter', finder_id=1.0, exposure=1)

        # Color threshold values, in HSV space
        self.low_limit_hsv = numpy.array((65, 75, 75), dtype=numpy.uint8)
        self.high_limit_hsv = numpy.array((100, 255, 255), dtype=numpy.uint8)

        # pixel area of the bounding rectangle - just used to remove stupidly small regions
        self.contour_min_area = 80

        # candidate cut thresholds
        self.min_dim_ratio = 1
        self.max_area_ratio = 0.25

        # matrices used to correct coordinates for camera location/tilt
        self.t_robot = numpy.array(((self.CAMERA_OFFSET_X,), (0.0,), (self.CAMERA_OFFSET_Z,)))

        c_a = math.cos(self.CAMERA_TILT)
        s_a = math.sin(self.CAMERA_TILT)
        self.rot_robot = numpy.array(((1.0, 0.0, 0.0), (0.0, c_a, -s_a), (0.0, s_a, c_a)))
        self.camera_offset_rotated = numpy.matmul(self.rot_robot.transpose(), -self.t_robot)

        self.hsv_frame = None
        self.threshold_frame = None

        # DEBUG values
        self.top_contours = None

        # output results
        self.target_contour = None

        self.cameraMatrix = calib_matrix
        self.distortionMatrix = dist_matrix

        self.outer_corners = []
        return

    @staticmethod
    def contour_diagonal_corners(contour):
        '''Find the 4 points on the contour which are the farthest out along the Cartesian diagonals.
        Algorithm from Robot Casserole, Team 1736'''

        # find the index, within the contour, of each corner
        top_left = (contour[:, :, 0] + contour[:, :, 1]).argmin()
        top_right = (contour[:, :, 0] - contour[:, :, 1]).argmax()

        # for bottom, look at an angle of 30deg from vertical, instead of 45
        # tan(30) = 0.577
        bottom_left = (0.577*contour[:, :, 0] - contour[:, :, 1]).argmin()
        bottom_right = (0.577*contour[:, :, 0] + contour[:, :, 1]).argmax()

        corners = numpy.array([contour[top_left][0], contour[top_right][0], contour[bottom_right][0], contour[bottom_left][0]])

        return corners

    def set_color_thresholds(self, hue_low, hue_high, sat_low, sat_high, val_low, val_high):
        self.low_limit_hsv = numpy.array((hue_low, sat_low, val_low), dtype=numpy.uint8)
        self.high_limit_hsv = numpy.array((hue_high, sat_high, val_high), dtype=numpy.uint8)
        return

    @staticmethod
    def get_outer_corners(cnt):
        '''Return the outer four corners of a contour'''

        return GenericFinder.sort_corners(cnt)  # Sort by x value of cnr in increasing value

    def preallocate_arrays(self, shape):
        '''Pre-allocate work arrays to save time'''

        self.hsv_frame = numpy.empty(shape=shape, dtype=numpy.uint8)
        # threshold_fame is grey, so only 2 dimensions
        self.threshold_frame = numpy.empty(shape=shape[:2], dtype=numpy.uint8)
        return

    def process_image(self, camera_frame):
        '''Main image processing routine'''
        self.target_contour = None

        # DEBUG values; clear any values from previous image
        self.top_contours = None
        self.outer_corners = None

        shape = camera_frame.shape
        if self.hsv_frame is None or self.hsv_frame.shape != shape:
            self.preallocate_arrays(shape)

        self.hsv_frame = cv2.cvtColor(camera_frame, cv2.COLOR_BGR2HSV, dst=self.hsv_frame)
        self.threshold_frame = cv2.inRange(self.hsv_frame, self.low_limit_hsv, self.high_limit_hsv,
                                           dst=self.threshold_frame)

        # OpenCV 3 returns 3 parameters, OpenCV 4 returns 2!
        # Only need the contours variable
        res = cv2.findContours(self.threshold_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(res) == 2:
            contours = res[0]
        else:
            contours = res[1]

        contour_list = []
        for c in contours:
            center, widths = GoalFinder2020.contour_center_width(c)
            area = widths[0] * widths[1]
            if area > self.contour_min_area:        # area cut
                contour_list.append({'contour': c, 'center': center, 'widths': widths, 'area': area})

        # Sort the list of contours from biggest area to smallest
        contour_list.sort(key=lambda c: c['area'], reverse=True)

        # DEBUG
        self.top_contours = [x['contour'] for x in contour_list]

        # try only the 5 biggest regions at most
        target_center = None
        for candidate_index in range(min(5, len(contour_list))):
            self.target_contour = self.test_candidate_contour(contour_list[candidate_index], shape)
            if self.target_contour is not None:
                target_center = contour_list[candidate_index]['center']
                break

        if self.target_contour is not None:
            # The target was found. Convert to real world co-ordinates.

            # need the corners in proper sorted order, and as floats
            self.outer_corners = GenericFinder.sort_corners(self.target_contour, target_center).astype(numpy.float)

            # print("Outside corners: ", self.outer_corners)
            # print("Real World target_coords: ", self.real_world_coordinates)

            retval, rvec, tvec = cv2.solvePnP(self.real_world_coordinates, self.outer_corners,
                                              self.cameraMatrix, self.distortionMatrix)
            if retval:
                result = [1.0, self.finder_id, ]
                result.extend(self.compute_output_values(rvec, tvec))
                result.extend((-1.0, -1.0))
                return result

        # no target found. Return "failure"
        return [0.0, self.finder_id, 0.0, 0.0, 0.0, -1.0, -1.0]

    def prepare_output_image(self, input_frame):
        '''Prepare output image for drive station. Draw the found target contour.'''

        output_frame = input_frame.copy()

        # if self.top_contours:
        #     cv2.drawContours(output_frame, self.top_contours, -1, (0, 0, 255), 1)

        if self.target_contour is not None:
            cv2.drawContours(output_frame, [self.target_contour.astype(int)], -1, (0, 0, 255), 2)

        if self.outer_corners is not None:
            for indx, cnr in enumerate(self.outer_corners):
                cv2.drawMarker(output_frame, tuple(cnr.astype(int)), (0, 255, 255), cv2.MARKER_CROSS, 15, 2)
                # cv2.putText(output_frame, str(indx), tuple(cnr.astype(int)), 0, .5, (255, 255, 255))

        return output_frame

    def test_candidate_contour(self, candidate, shape):
        '''Determine the true target contour out of potential candidates'''

        cand_width = candidate['widths'][0]
        cand_height = candidate['widths'][1]

        cand_dim_ratio = cand_width / cand_height
        if cand_dim_ratio < self.min_dim_ratio:
            return None
        cand_area_ratio = cv2.contourArea(candidate["contour"]) / (cand_width * cand_height)
        if cand_area_ratio > self.max_area_ratio:
            return None

        hull = cv2.convexHull(candidate['contour'])
        contour = self.quad_fit(hull)

        # different way of finding the corners
        # faster, pretty good, but maybe a little less stable???
        # contour = self.contour_diagonal_corners(candidate['contour'])

        if contour is not None and len(contour) == 4:
            return contour

        return None

    def compute_output_values(self, rvec, tvec):
        '''Compute the necessary output distance and angles'''

        x_r_w0 = numpy.matmul(self.rot_robot, tvec) + self.t_robot
        x = x_r_w0[0][0]
        z = x_r_w0[2][0]

        # distance in the horizontal plane between robot center and target
        distance = math.sqrt(x**2 + z**2)

        # horizontal angle between robot center line and target
        angle1 = math.atan2(x, z)

        rot, _ = cv2.Rodrigues(rvec)
        rot_inv = rot.transpose()

        # location of Robot (0,0,0) in World coordinates
        x_w_r0 = numpy.matmul(rot_inv, self.camera_offset_rotated - tvec)

        angle2 = math.atan2(x_w_r0[0][0], x_w_r0[2][0])

        return distance, angle1, angle2


# Main routine
# This is for development/testing without running the whole server
if __name__ == '__main__':
    main(GoalFinder2020)
