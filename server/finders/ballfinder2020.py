#!/usr/bin/env python3

import cv2
import numpy
import math

from finders.genericfinder import GenericFinder, main


class BallFinder2020(GenericFinder):
    '''Ball finder for Infinite Recharge 2020'''

    BALL_DIAMETER = 7            # inches

    HFOV = 64.0                  # horizontal angle of the field of view
    VFOV = 52.0                  # vertical angle of the field of view

    # create imaginary view plane on 3d coords to get height and width
    # place the view place on 3d coordinate plane 1.0 unit away from (0, 0) for simplicity
    VP_HALF_WIDTH = math.tan(math.radians(HFOV)/2.0)  # view plane 1/2 height
    VP_HALF_HEIGHT = math.tan(math.radians(VFOV)/2.0)  # view plane 1/2 width

    def __init__(self, calib_matrix=None, dist_matrix=None):
        super().__init__('ballfinder', camera='intake', finder_id=2.0, exposure=0)

        # individual properties
        self.low_limit_hsv = numpy.array((15, 95, 95), dtype=numpy.uint8)
        self.high_limit_hsv = numpy.array((75, 255, 255), dtype=numpy.uint8)

        # pixel area of the bounding rectangle - just used to remove stupidly small regions
        self.contour_min_area = 80

        # self.erode_kernel = numpy.ones((3, 3), numpy.uint8)
        # self.erode_iterations = 0

        # some variables to save results for drawing
        self.top_contours = []
        self.found_contours = []
        self.center_points = []

        self.cameraMatrix = calib_matrix
        self.distortionMatrix = dist_matrix

        self.tilt_angle = math.radians(-20.0)  # camera mount angle (radians)
        self.camera_height = 20.5              # height of camera off the ground (inches)
        self.target_height = 3.5               # height of target off the ground (inches)

        return

    def set_color_thresholds(self, hue_low, hue_high, sat_low, sat_high, val_low, val_high):
        self.low_limit_hsv = numpy.array((hue_low, sat_low, val_low), dtype=numpy.uint8)
        self.high_limit_hsv = numpy.array((hue_high, sat_high, val_high), dtype=numpy.uint8)
        return

    def get_ball_values(self, center, shape):
        '''Calculate the angle and distance from the camera to the center point of the robot
        This routine uses the FOV numbers and the default center to convert to normalized coordinates'''

        # center is in pixel coordinates, 0,0 is the upper-left, positive down and to the right
        # (nx,ny) = normalized pixel coordinates, 0,0 is the center, positive right and up
        # WARNING: shape is (h, w, nbytes) not (w,h,...)
        image_w = shape[1] / 2.0
        image_h = shape[0] / 2.0

        # NOTE: the 0.5 is to place the location in the center of the pixel
        # print("center", center, "shape", shape)
        nx = (center[0] - image_w + 0.5) / image_w
        ny = (image_h - 0.5 - center[1]) / image_h

        # convert normal pixel coords to pixel coords
        x = BallFinder2020.VP_HALF_WIDTH * nx
        y = BallFinder2020.VP_HALF_HEIGHT * ny
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

    def get_ball_values_calib(self, center):
        '''Calculate the angle and distance from the camera to the center point of the robot
        This routine uses the cameraMatrix from the calibration to convert to normalized coordinates'''

        # use the distortion and camera arrays to correct the location of the center point
        # got this from
        #  https://stackoverflow.com/questions/8499984/how-to-undistort-points-in-camera-shot-coordinates-and-obtain-corresponding-undi

        ptlist = numpy.array([[center]])
        out_pt = cv2.undistortPoints(ptlist, self.cameraMatrix, self.distortionMatrix, P=self.cameraMatrix)
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

        # clear out member result variables
        self.center_points = []
        self.top_contours = []
        self.found_contours = []

        hsv_frame = cv2.cvtColor(camera_frame, cv2.COLOR_BGR2HSV)
        threshold_frame = cv2.inRange(hsv_frame, self.low_limit_hsv, self.high_limit_hsv)

        # if self.erode_iterations > 0:
        #     erode_frame = cv2.erode(threshold_frame, self.erode_kernel, iterations=self.erode_iterations)
        # else:
        #     erode_frame = threshold_frame

        # OpenCV 3 returns 3 parameters, OpenCV 4 returns 2!
        # Only need the contours variable
        res = cv2.findContours(threshold_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(res) == 2:
            contours = res[0]
        else:
            contours = res[1]

        contour_list = []
        for c in contours:
            center, widths = self.contour_center_width(c)
            area = widths[0] * widths[1]
            if area > self.contour_min_area:
                contour_list.append({'contour': c, 'center': center, 'widths': widths, 'area': area})

        self.top_contours = [cnt['contour'] for cnt in contour_list]
        # Sort the list of contours from lowest in the image to the highest in the image
        contour_list.sort(key=lambda c: c['center'][1], reverse=True)

        # test the lowest three contours only (optimization)
        for cnt in contour_list[0:3]:
            result_cnt = self.test_candidate_contour(cnt)
            if result_cnt is not None:
                self.found_contours.append(result_cnt)

                img_moments = cv2.moments(result_cnt)
                center = numpy.array((img_moments['m10']/img_moments['m00'], img_moments['m01']/img_moments['m00']))

                # note: these are the major/minor axes, equivalent to the radius (not the diameter)
                major, minor, angle = self.major_minor_axes(img_moments)

                # if the ratio is large, probably is 2 balls merged into 1 contour
                # print("major/minor axis ratio", major / minor)
                if major / minor > 1.4:
                    # compute the offset, otherwise just use the centroid (more reliable)
                    direction = numpy.array((math.cos(angle), math.sin(angle)))

                    # the factor of 1.3 is total arbitrary, but seems to make the point closer to the center
                    # essentially the minor axis underestimates the radius of the front ball,
                    #  which is bigger in the image
                    self.center_points.append(center + ((major - 1.3 * minor) * direction))

                    # if the contour is made up of all three balls, must return 2 centers, so return both anyways
                    self.center_points.append(center - ((major - 0.8 * minor) * direction))

                    # TODO may need to change the 1.3 and 0.8 for three vs. two balls?
                else:
                    self.center_points.append(center)
                # print("Center point:", center)

                if len(self.center_points) >= 2:
                    break

        # done with the contours. Pick two centers to return

        # return values: (success, finder_id, distance1, robot_angle1, target_angle1, distance2, robot_angle2)
        # -1.0 means that something failed
        # 0 means that the entry is not currently being used

        if not self.center_points:
            # failed, found no ball
            return (0.0, self.finder_id, -1.0, -1.0, 0.0, -1.0, -1.0)

        # remember y goes up as you move down the image
        # self.center_points.sort(key=lambda c: c[1], reverse=True) #no need b/c it should already be in order

        if self.cameraMatrix is not None:
            # use the camera calibration if we have it
            angle1, distance1 = self.get_ball_values_calib(self.center_points[0])
            if len(self.center_points) > 1:
                angle2, distance2 = self.get_ball_values_calib(self.center_points[1])
            else:
                angle2 = -1.0
                distance2 = -1.0
        else:
            angle1, distance1 = self.get_ball_values(self.center_points[0], camera_frame.shape)
            if len(self.center_points) > 1:
                angle2, distance2 = self.get_ball_values(self.center_points[1], camera_frame.shape)
            else:
                angle2 = -1.0
                distance2 = -1.0

        return (1.0, self.finder_id, distance1, angle1, 0.0, distance2, angle2)

    def test_candidate_contour(self, contour_entry):
        cnt = contour_entry['contour']

        # real_area = cv2.contourArea(cnt)
        # print('areas:', real_area, contour_entry['area'], real_area / contour_entry['area'])
        # print("ratio"+str(contour_entry['widths'][1] / contour_entry['widths'][0] ))

        # contour_entry['widths'][1] is the height
        # contour_entry['widths'][0] is the width

        ratio = contour_entry['widths'][1] / contour_entry['widths'][0]
        # TODO if balls cut out at bottom of screen it returns none,
        #   so might want to change the lower value depending on camera location
        if ratio < 0.8 or ratio > 3.1:
            return None

        ratio = cv2.contourArea(cnt) / contour_entry['area']
        if ratio < (math.pi / 4) - 0.1 or ratio > (math.pi / 4) + 0.1:  # TODO refine the 0.1 error range
            return None

        return cnt

    def prepare_output_image(self, input_frame):
        '''Prepare output image for drive station. Draw the found target contour.'''

        output_frame = input_frame.copy()

        # Draw the contour on the image
        # if self.top_contours:
        #     cv2.drawContours(output_frame, self.top_contours, -1, (255, 0, 0), 1)

        if self.found_contours:
            cv2.drawContours(output_frame, self.found_contours, -1, (0, 0, 255), 2)

        for cnt in self.center_points:
            cv2.drawMarker(output_frame, tuple(cnt.astype(int)), (255, 125, 0), cv2.MARKER_CROSS, 10, 2)

        return output_frame


# Main routine
# This is for development/testing
if __name__ == '__main__':
    main(BallFinder2020)
