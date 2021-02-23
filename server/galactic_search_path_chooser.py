#!/usr/bin/env python3

import cv2
import numpy
from genericfinder import GenericFinder, main
import math

class GalacticSearchPathChooser(GenericFinder):
    '''Path chooser for galactic search skill challenge'''

    RED_CLOSE_Y = 408  # min y value of close ball for path to be red


    def __init__(self, calib_matrix=None, dist_matrix=None, result_nt_entry=None):
        super().__init__('galactic_search_path_chooser', camera='intake', finder_id=6.0, exposure=0)

        # individual properties
        self.low_limit_hsv = numpy.array((21, 111, 111), dtype=numpy.uint8)
        self.high_limit_hsv = numpy.array((52, 255, 255), dtype=numpy.uint8)

        # pixel area of the bounding rectangle - just used to remove stupidly small regions
        self.contour_min_area = 80

        # some variables to save results for drawing
        self.top_contours = []
        self.found_contours = []
        self.center_points = []

        self.cameraMatrix = calib_matrix
        self.distortionMatrix = dist_matrix

        self.tilt_angle = math.radians(-15.0)  # camera mount angle (radians)
        self.camera_height = 20.5              # height of camera off the ground (inches)
        self.target_height = 3.5               # height of target (the ball) off the ground (inches)

        self.contour_debug_info = []
        self.chosen_path = ""

        self.result_nt_entry = result_nt_entry
        return

    def set_color_thresholds(self, hue_low, hue_high, sat_low, sat_high, val_low, val_high):
        self.low_limit_hsv = numpy.array((hue_low, sat_low, val_low), dtype=numpy.uint8)
        self.high_limit_hsv = numpy.array((hue_high, sat_high, val_high), dtype=numpy.uint8)
        return

    def process_image(self, camera_frame):
        '''Main image processing routine'''

        # clear out member result variables
        self.center_points = []
        self.top_contours = []
        self.found_contours = []
        self.contour_debug_info=[]
        hsv_frame = cv2.cvtColor(camera_frame, cv2.COLOR_BGR2HSV)
        threshold_frame = cv2.inRange(hsv_frame, self.low_limit_hsv, self.high_limit_hsv)

        image_height = hsv_frame.shape[0]

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

        # Sort the list of contours from largest area to smallest area
        contour_list.sort(key=lambda c: c['area'], reverse=True)

        for cnt in contour_list:
            result_cnt, error_reason = self.test_candidate_contour(cnt)

            if result_cnt is not None:
                img_moments = cv2.moments(result_cnt)
                center = numpy.array((img_moments['m10']/img_moments['m00'], img_moments['m01']/img_moments['m00']))

                # TODO: this could be done faster on the contour center (inside test_candidate_contour)
                #    - saves calcing the moments
                if image_height - center[1] < 237:  # filter out the wheels near the bottom of the robot
                    self.contour_debug_info.append((center, "too low"))

                    continue
                if image_height - center[1] > 544:  # filter out the ladder
                    self.contour_debug_info.append((center, "too high"))

                    continue

                self.found_contours.append(result_cnt)
                self.center_points.append(center)

                if len(self.center_points) >= 3:
                    break
            else:
                self.contour_debug_info.append((cnt["center"], error_reason))

        if len(self.center_points) < 2:
            self.chosen_path = "Fail - Not enough balls"
            return (0.0, self.finder_id, 0, 0, 0.0, 0, 0)


        # sort from bottom to top
        self.center_points.sort(key=lambda p: p[1], reverse=True)

        chosen_path = None

        if self.center_points[0][1] > self.RED_CLOSE_Y:  # is the path red?
            b1_a, b1_d = self.get_ball_values_calib(self.center_points[0])
            b2_a, b2_d = self.get_ball_values_calib(self.center_points[-1])
            b1_coords = numpy.array((math.sin(b1_a)*b1_d,math.cos(b1_a)*b1_d))
            b2_coords = numpy.array((math.sin(b2_a)*b2_d,math.cos(b2_a)*b2_d))
            real_b1_b2_distance = numpy.linalg.norm(b1_coords-b2_coords)

            self.contour_debug_info.append((self.center_points[0], str(b1_coords)))
            self.contour_debug_info.append((self.center_points[-1], str(b2_coords)))

            print(real_b1_b2_distance)

            if real_b1_b2_distance > 68:
                chosen_path = "b-red"
            else:
                chosen_path = "a-red"
        else:  # is the path blue?
            if self.center_points[0][0] - 50 > self.center_points[-1][0]:
                chosen_path = "a-blue"
            else:
                chosen_path = "b-blue"

        if chosen_path is None:
            self.chosen_path = "Fail - could not choose path"
            return (0.0, self.finder_id, 0, 0, 0.0, 0, 0)

        self.chosen_path = chosen_path
        if self.result_nt_entry is not None:
            self.result_nt_entry.setString(self.chosen_path)

        return (1.0, self.finder_id, 0, 0, 0.0, 0, 0)

    def test_candidate_contour(self, contour_entry):
        cnt = contour_entry['contour']
        rect_area = contour_entry['area']
        real_area = cv2.contourArea(cnt)

        # contour_entry['widths'][1] is the height
        # contour_entry['widths'][0] is the width
        ratio = contour_entry['widths'][1] / contour_entry['widths'][0]
        # TODO if balls cut out at bottom of screen it returns none,
        # so might want to change the lower value depending on camera location
        if ratio < 0.6 or ratio > 1.1:
            #print("Contour at",contour_entry["center"], "rejected due to bad w/h ratio")
            #print("w/h ratio:", str(contour_entry['widths'][1] / contour_entry['widths'][0]))

            return (None, "bad w/h ratio")

        ratio = real_area / contour_entry['area']
        expected_real_rect_ratio = math.pi / 4


        if rect_area*expected_real_rect_ratio*.85 - 40 > real_area or rect_area*expected_real_rect_ratio*1.1 + 30 < real_area:  # allow for greater varation in smaller contours
            #print("Contour at", contour_entry["center"], "rejected due to bad rect/real area ratio")
            #print("rect_area:", rect_area, "real_area:", real_area, "rect/real area ratio:", real_area / rect_area)
            return (None, "bad rect/real area ratio")

        return (cnt, None)

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

        return ax, d

    def prepare_output_image(self, input_frame):
        '''Prepare output image for drive station. Draw found contours.'''
        output_frame = input_frame.copy()

        # Draw the contour on the image
        if self.top_contours:
            cv2.drawContours(output_frame, self.top_contours, -1, (255, 0, 0), 1)


        if self.found_contours:
            cv2.drawContours(output_frame, self.found_contours, -1, (0, 0, 255), 2)

        for (center, reason) in self.contour_debug_info:
            cv2.putText(output_frame, reason, (int(center[0]),int(center[1])), cv2.FONT_HERSHEY_SIMPLEX, .3, (255, 255, 255), thickness=1)

        for cnt in self.center_points:
            cv2.drawMarker(output_frame, tuple(cnt.astype(int)), (255, 125, 0), cv2.MARKER_CROSS, 10, 2)

        cv2.putText(output_frame, self.chosen_path, (60, input_frame.shape[0]-30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), thickness=2)

        return output_frame


# Main routine
# This is for development/testing
if __name__ == '__main__':
    main(GalacticSearchPathChooser)
