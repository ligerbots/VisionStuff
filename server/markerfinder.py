#!/usr/bin/env python3

import cv2
import numpy as np
import math

from genericfinder import GenericFinder, main

import numba as nb
from numba.np.extensions import cross2d

@nb.njit(nb.float32(nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:]))
def line_ray_intersect(line1, line2, ray1, ray_dir):
    # https://stackoverflow.com/a/565282/5771000
    # p=line1
    # r=line_dir
    # q=ray1
    # s=ray_dir

    line_dir = line2 - line1
    line_dir_cross_ray_dir = cross2d(line_dir, ray_dir)
    if line_dir_cross_ray_dir == 0.:
        return -1.
    ray1_minus_line1 = ray1 - line1
    t = cross2d(ray1_minus_line1, ray_dir) / line_dir_cross_ray_dir
    u = cross2d(ray1_minus_line1, line_dir) / line_dir_cross_ray_dir
    if 0. <= t <= 1. and 0. <= u:
        return u
    return -1.

@nb.njit(nb.float32(nb.float64[:,:], nb.float64[:], nb.float64[:]))
def poly_ray_intersect(poly, ray1, ray_dir):
    min_d = -1
    for i in range(len(poly)):
        d = line_ray_intersect(poly[i-1], poly[i], ray1, ray_dir)
        if d >= 0:
            if(min_d < 0):
                min_d = d
            elif(d < min_d):
                min_d = d
    return min_d



class MarkerFinder2020(GenericFinder):
    '''Marker finder for Infinite Recharge at home'''
    target_world_coordinates = np.array([
        [4,  .5, 0],
        [4, 2.5, 0],
        [0, 2.5, 0],
        [0,  .5, 0]
    ])
    grid_world_coordinates = np.transpose(
        np.meshgrid([.5, 1.5, 2.5, 3.5], [.5, 1.5, 2.5, 3.5, 4.5, 5.5], [0]),
        [1, 2, 3, 0]
    )

    grid_center_world_coordinates = np.array([[2, 3, 0], ])

    totransform_world_coordinates = np.concatenate([
        np.reshape(grid_world_coordinates, [-1, 3]),
        grid_center_world_coordinates
    ])

    def __init__(self, calib_matrix=None, dist_matrix=None):
        super().__init__('markerfinder', camera='intake', finder_id=2.0, exposure=0)

        # individual properties
        self.low_limit_hsv = np.array((0, 0, 0), dtype=np.uint8)
        self.high_limit_hsv = np.array((255, 255, 110), dtype=np.uint8)

        # some variables to save results for drawing
        self.contour_list = []
        self.grid_points = []
        self.result_points = {}

        self.cameraMatrix = calib_matrix
        self.distortionMatrix = dist_matrix

        self.tilt_angle = math.radians(-15.0)  # camera mount angle (radians)
        self.camera_height = 20.5              # height of camera off the ground (inches)

        return

    def process_image(self, camera_frame):
        '''Main image processing routine'''
        image_height, image_width, _ = camera_frame.shape

        hsv_frame = cv2.cvtColor(camera_frame, cv2.COLOR_BGR2HSV)
        threshold_frame = cv2.inRange(hsv_frame, self.low_limit_hsv, self.high_limit_hsv)

        contours, hierarchy = cv2.findContours(
            threshold_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        self.contour_list = []

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)

            if(w > image_width/2 or h > image_height/2):
                continue
            if (w*h < 50):
                continue

            real_area = cv2.contourArea(contour)

            if(real_area < w*h*.5):
                continue

            img_moments = cv2.moments(contour)
            center = np.array((img_moments['m10']/img_moments['m00'],
                               img_moments['m01']/img_moments['m00']))
            major, minor, angle = self.major_minor_axes(img_moments)

            if(major < minor*2):
                continue

            direction = np.array((math.cos(angle), math.sin(angle)))

            self.contour_list.append({
                "contour": contour,
                "center": center,
                "direction": direction,
                "major": major,
                "minor": minor,
                "lower": None,
                "islower": False
            })

        self.contour_list.sort(key=lambda c: c['center'][1])

        for i in range(len(self.contour_list)):
            contour = self.contour_list[i]

            if(contour["islower"]):
                continue

            center = contour["center"]
            lower = center[1] + contour["minor"]*8
            left = center[0] - contour["major"]
            right = center[0] + contour["major"]

            for j in range(i+1, len(self.contour_list)):
                potential_pair_contour = self.contour_list[j]
                if(potential_pair_contour["center"][1] > lower):
                    break
                if(potential_pair_contour["lower"] or potential_pair_contour["islower"]):
                    continue
                if(potential_pair_contour["center"][0] < left):
                    continue
                if(potential_pair_contour["center"][0] > right):
                    continue
                contour["lower"] = potential_pair_contour
                potential_pair_contour["islower"] = True

                break

        self.grid_points = []
        self.result_points = {}

        for contour in self.contour_list:
            if not contour["lower"]:
                continue
            lower = contour["lower"]

            contour_np = contour["contour"].astype(np.float)
            lower_np = lower["contour"].astype(np.float)
            top_right = poly_ray_intersect(
                contour_np[:, 0, :], contour["center"], contour["direction"])
            top_left = poly_ray_intersect(
                contour_np[:, 0, :], contour["center"], -contour["direction"])
            bottom_right = poly_ray_intersect(
                lower_np[:, 0, :], lower["center"], lower["direction"])
            bottom_left = poly_ray_intersect(
                lower_np[:, 0, :], lower["center"], -lower["direction"])

            if top_right < 0:
                continue
            if top_left < 0:
                continue
            if bottom_right < 0:
                continue
            if bottom_left < 0:
                continue

            top_right_point = contour["center"] + contour["direction"]*top_right
            top_left_point = contour["center"] - contour["direction"]*top_left
            bottom_right_point = lower["center"] + lower["direction"]*bottom_right
            bottom_left_point = lower["center"] - lower["direction"]*bottom_left

            if top_left_point[0] > top_right_point[0]:
                top_left_point, top_right_point = top_right_point, top_left_point

            if bottom_left_point[0] > bottom_right_point[0]:
                bottom_left_point, bottom_right_point = bottom_right_point, bottom_left_point

            points = np.array([top_right_point, bottom_right_point,
                               bottom_left_point, top_left_point])

            retval, rvec, tvec = cv2.solvePnP(self.target_world_coordinates, points,
                                              self.cameraMatrix, self.distortionMatrix)

            if retval:
                imgpts, jac = cv2.projectPoints(
                    self.totransform_world_coordinates, rvec, tvec, self.cameraMatrix, self.distortionMatrix)
                grid_pts, [center] = np.split(np.squeeze(imgpts), [-1])

                self.grid_points.extend(grid_pts)

                colors_bool = self.extract_grid(threshold_frame, grid_pts)
                if colors_bool is not None:
                    if (np.all(colors_bool[0, :] == True) and np.all(colors_bool[1, :] == False) and
                            np.all(colors_bool[2, :] == True) and np.all(colors_bool[3, :] == False)):

                        payload_bool = np.concatenate([colors_bool[4, :], colors_bool[5, :]])
                        payload_value = np.packbits(payload_bool)[0]

                        self.result_points[payload_value] = center

        return (1.0, self.finder_id, 0, 0, 0.0, 0, 0)

    def extract_grid(self, threshold_frame, grid_pts):
        image_height, image_width = threshold_frame.shape

        grid_colors = np.empty((4*6))
        for i in range(len(grid_pts)):
            point = grid_pts[i].astype(np.int)

            if(point[0] < 0 or point[1] < 0 or point[0] >= image_width or point[1] >= image_height):
                return None

            grid_colors[i] = threshold_frame[point[1], point[0]]

        colors_bool = np.reshape(grid_colors, (6, 4)) > 126
        return(colors_bool)

    def prepare_output_image(self, input_frame):
        '''Prepare output image for drive station. Draw the found target contour.'''

        output_frame = input_frame.copy()
        for contour in self.contour_list:
            p1 = tuple(contour["center"].astype(np.int))
            p2 = tuple((contour["center"]+contour["direction"]*contour["major"]).astype(np.int))
            cv2.line(output_frame, p1, p2, (0, 255, 0), 1)

        upper_contours = [contour["contour"] for contour in self.contour_list if contour["lower"]]
        lower_contours = [contour["contour"] for contour in self.contour_list if contour["islower"]]

        cv2.drawContours(output_frame, upper_contours, -1, (0, 0, 255), 1)
        cv2.drawContours(output_frame, lower_contours, -1, (255, 0, 0), 1)

        for point in self.grid_points:
            cv2.circle(output_frame, tuple(point.astype(np.int)), 1, (255, 0, 255), -1)

        for id in self.result_points:
            center_tup = tuple(self.result_points[id].astype(np.int))

            cv2.circle(output_frame,center_tup, 3, (0,255,255), -1)

            cv2.putText(output_frame, str(id),
                center_tup,
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255,255,0),
                2)

        return output_frame


# Main routine
# This is for development/testing
if __name__ == '__main__':
    main(MarkerFinder2020)
