#!/usr/bin/env python3

import cv2
import numpy as np
import math

from genericfinder import GenericFinder

import numba as nb
from numba.np.extensions import cross2d
import layout

import markerfinder_position_solver

@nb.njit(nb.float32(nb.float32[:], nb.float32[:], nb.float32[:], nb.float32[:]))
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


@nb.njit(nb.float32(nb.float32[:, :], nb.float32[:], nb.float32[:]))
def poly_ray_intersect(poly, ray1, ray_dir):
    min_d = -1
    for i in range(len(poly)):
        d = line_ray_intersect(poly[i - 1], poly[i], ray1, ray_dir)
        if d >= 0:
            if(min_d < 0):
                min_d = d
            elif(d < min_d):
                min_d = d
    return min_d

def undo_tilt_on_image_plane_coords(point, theta):
    x, y = point
    return(np.array([
        x / (math.cos(theta) - y * math.sin(theta)),
        (math.sin(theta) + y * math.cos(theta)) /
        (math.cos(theta) - y * math.sin(theta))
    ]))

def image_coords_to_image_plane_coords(point, camera_matrix, distortion_matrix):
    ptlist = np.array([[point]])
    out_pt = cv2.undistortPoints(
        ptlist, camera_matrix, distortion_matrix, P=camera_matrix)
    x, y = out_pt[0, 0]

    x_prime = (x - camera_matrix[0, 2]) / camera_matrix[0, 0]
    y_prime = -(y - camera_matrix[1, 2]) / camera_matrix[1, 1]
    return(np.array([x_prime, y_prime]))


@nb.njit(nb.float32(nb.float32))
def make_nonzero(n):
    if(n == 0):
        return np.finfo(np.float32).eps
    return n


@nb.njit(nb.float32[:](nb.float32, nb.float32, nb.float32))
def get_direction_in_image_plane_coords(x, y, tilt):
    return np.array([
        x * math.sin(tilt),
        y * math.sin(tilt) + math.cos(tilt)
    ])

@nb.njit(nb.float32[:](nb.float32, nb.float32, nb.float32, nb.float32[:, :]))
def get_direction_in_image_coords(x, y, tilt, camera_matrix):
    x_prime = (x - camera_matrix[0, 2]) / camera_matrix[0, 0]
    y_prime = (y - camera_matrix[1, 2]) / camera_matrix[1, 1]
    slope_prime = get_direction_in_image_plane_coords(x_prime, y_prime, tilt)
    return np.array([
        slope_prime[0] * camera_matrix[0, 0],
        slope_prime[1] * camera_matrix[1, 1]
    ])

@nb.njit(nb.int32[:](nb.uint8[:, :, :], nb.int32[:], nb.int32[:]))
def scanline(image, start, end):
    image_height, image_width, _ = image.shape

    swap_i = 0
    swap_t = np.full((3,), -1000)
    prev_swap = -1000

    offset = end - start

    num_prev_cols = 2

    REQUIRED_DIFF = 150.
    MAX_TOLERANCE = .1

    prev_cols = np.zeros((num_prev_cols,3), dtype=np.uint8)
    for i in range(num_prev_cols):
        x = start[0] + offset[0] * i // offset[1]
        y = start[1] + i
        if(0<=x<image_width and 0<=y<image_height):
            prev_cols[i] = image[y, x]

    i = num_prev_cols
    while i <= offset[1]:
        x = start[0] + offset[0] * i // offset[1]
        y = start[1] + i
        if(0<=x<image_width and 0<=y<image_height):
            this_col = image[y, x].astype(np.float32)

            prev_col = prev_cols[i % num_prev_cols].astype(np.float32)
            prev_cols[i % num_prev_cols] = image[y, x]


            if np.linalg.norm(this_col - prev_col) > REQUIRED_DIFF:
                swap_t[swap_i % 3] = i - prev_swap
                swap_i += 1
                prev_swap = i

                total = np.sum(swap_t, dtype=np.int32)

                if(np.amax(swap_t) - np.amin(swap_t) < total / 3 * MAX_TOLERANCE + 3 and total > 6):
                    return np.array([i-num_prev_cols//2, total])

                for j in range(num_prev_cols - 1):
                    i += 1

                    x = start[0] + offset[0] * i // offset[1]
                    y = start[1] + i
                    if(0<=x<image_width and 0<=y<image_height):
                        prev_cols[i % num_prev_cols] = image[y, x]
                    else:
                        prev_cols[i % num_prev_cols] = np.zeros((3,))

        i += 1
    return np.array([-1, -1])

@nb.njit(nb.float32[:](nb.float32[:],nb.float32[:]))
def linreg(x,y):
    x_m = np.mean(x)
    y_m = np.mean(y)
    slope = np.sum((x-x_m)*(y-y_m))/np.sum((x-x_m)**2)
    y_cept = y_m - x_m * slope
    return np.array([y_cept,slope])

@nb.njit(nb.types.Tuple((nb.float32[:,:,:], nb.float32[:]))(nb.uint8[:, :, :], nb.float32, nb.float32, nb.float32[:], nb.float32, nb.float32[:,:]), parallel=True)
def scanlines(image, half_width, third_height, lower_center, tilt_angle, camera_matrix):

    xs = np.arange(-half_width, half_width, 1)

    f_xs = np.full((len(xs),), -1, dtype=nb.float32)
    f_ys = np.full((len(xs),), -1, dtype=nb.float32)
    totals = np.full((len(xs),), -1, dtype=nb.float32)

    for i in nb.prange(len(xs)):
        center = lower_center + np.array([xs[i],0])

        dir = get_direction_in_image_coords(center[0], center[1],tilt_angle, camera_matrix)
        dir /= np.linalg.norm(dir)

        top = center - dir*third_height/2*2.5
        bottom = center + dir*third_height/2*3.5

        dy, total = scanline(image,top.astype(np.int32),bottom.astype(np.int32))


        if(dy>=0):
            offset = bottom - top
            f_xs[i]=top[0] + offset[0] * dy // offset[1]
            f_ys[i]=top[1] + dy
            totals[i] = total
    success = totals>=0

    s_f_xs = f_xs[success]
    s_f_ys = f_ys[success]
    s_totals = totals[success]

    if(len(s_f_xs) < 2):
        return (np.zeros((0,0,0),dtype=np.float32),np.array([-1,-1, -1],dtype=np.float32))

    y_b, y_m = linreg(s_f_xs, s_f_ys)
    t_b, t_m = linreg(s_f_xs, s_totals)

    left_x = s_f_xs[0]
    right_x = s_f_xs[-1]
    center_x = (left_x+right_x)/2
    center_y = y_b + y_m * center_x

    ret = np.empty((6,4,2), dtype=np.float32)

    for xi in range(4):
        interp_x = (xi)/3
        x = left_x + interp_x*(right_x - left_x)
        y = y_b + x * y_m

        dir = get_direction_in_image_coords(x,y,tilt_angle, camera_matrix)
        dir /= dir[1]
        dir *= t_b + t_m * x


        for yi in range(6):
            interp_y = (yi - 2.5)/3
            ret[yi, xi] = np.array([x,y]) + dir*interp_y

    return (ret, np.array([center_x, center_y, t_b + t_m*center_x]))

def extract_payload(image, grid):
    image_height, image_width, _ = image.shape

    colors = np.zeros((6, 4, 3), dtype=np.uint8)
    for yi in range(6):
        for xi in range(4):
            x, y = grid[yi, xi]
            x_d = int(x+.5)
            y_d = int(y+.5)
            if(x_d<0 or y_d<0 or x_d >= image_width or y_d >= image_height):
                return None
            colors[yi,xi] = image[y_d,x_d]
    should_be_white = np.concatenate([colors[1, :], colors[3, :]])
    should_be_black = np.concatenate([colors[0, :], colors[2, :]])

    data_color = np.concatenate([colors[4, :], colors[5, :]])

    white_min = np.min(should_be_white, axis=0)
    black_max = np.max(should_be_black, axis=0)
    if(np.all(white_min > black_max)):
        white_avg = np.mean(should_be_white, axis=0)
        black_avg = np.mean(should_be_black, axis=0)
        data_distance_to_white = np.linalg.norm(
            data_color - white_avg, axis=1)
        data_distance_to_black = np.linalg.norm(
            data_color - black_avg, axis=1)

        data = data_distance_to_white > data_distance_to_black  # which colors are black?
        data_value = np.packbits(data)[0]
        return((data_value))
    else:
        return None

class MarkerFinder2021(GenericFinder):
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

    def __init__(self, calib_matrix, dist_matrix, camera_name):
        super().__init__("markerfinder_"+camera_name, camera=camera_name, finder_id=0.0, exposure=0)

        self.low_limit_hsv = np.array((0, 0, 0), dtype=np.uint8)
        self.high_limit_hsv = np.array((255, 255, 120), dtype=np.uint8)

        self.raw_contours = []
        self.contour_list = []
        self.scan_lines = []
        self.target_points = []
        self.grid_points = []
        self.result_points = []

        self.cameraMatrix = calib_matrix
        self.distortionMatrix = dist_matrix

        self.tilt_angle = layout.camera_pos[camera_name]["tilt"]
        self.camera_height = layout.camera_pos[camera_name]["height"]

        self.camera_transform_inv = layout.camera_pos[camera_name]["transform_inv"]

        self.camera_name = camera_name
        return

    def calculate_camera_plane_height(self, upper_center_image, lower_center_image):
        upper_center = undo_tilt_on_image_plane_coords(image_coords_to_image_plane_coords(
            upper_center_image, self.cameraMatrix, self.distortionMatrix), self.tilt_angle)

        lower_center = undo_tilt_on_image_plane_coords(image_coords_to_image_plane_coords(
            lower_center_image, self.cameraMatrix, self.distortionMatrix), self.tilt_angle)

        return(upper_center[1] - lower_center[1])

    def filter_raw_contours(self, image_width, image_height):
        self.contour_list = []

        for contour in self.raw_contours:
            x, y, w, h = cv2.boundingRect(contour)

            if(w > image_width / 2 or h > image_height / 2):
                continue

            real_area = cv2.contourArea(contour)

            if (real_area < 5):
                continue


            if(w == h or w <= 0 or h <= 0 or real_area == 0):
                continue

            img_moments = cv2.moments(contour)
            center = np.array((img_moments['m10'] / img_moments['m00'],
                               img_moments['m01'] / img_moments['m00']))

            if(img_moments['mu20'] == img_moments['mu02']):
                continue

            major, minor, angle = self.major_minor_axes(img_moments)

            if(major < minor * 2):
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

    def pair_contours(self):
        self.contour_list.sort(key=lambda c: c['center'][1])

        for i in range(len(self.contour_list)):
            contour = self.contour_list[i]

            if(contour["islower"]):
                continue

            center = contour["center"]
            lower = center[1] + contour["minor"] * 18
            left = center[0] - contour["major"]
            right = center[0] + contour["major"]

            for j in range(i + 1, len(self.contour_list)):
                potential_pair_contour = self.contour_list[j]
                if(potential_pair_contour["center"][1] > lower):
                    break
                if(potential_pair_contour["lower"] or potential_pair_contour["islower"]):
                    continue
                if(potential_pair_contour["center"][0] < left):
                    continue
                if(potential_pair_contour["center"][0] > right):
                    continue
                if(max(contour["major"], potential_pair_contour["major"]) / min(contour["major"], potential_pair_contour["major"]) > 2):
                    continue
                if(max(contour["minor"], potential_pair_contour["minor"]) / min(contour["minor"], potential_pair_contour["minor"]) > 2):
                    continue
                contour["lower"] = potential_pair_contour
                potential_pair_contour["islower"] = True

                break
    def process_small_contour_pair(self, contour, camera_frame):
        lower = contour["lower"]

        half_width = lower["major"]*1.5
        third_height = np.linalg.norm(contour["center"]-lower["center"])*1.5


        grid, (x,y, total) = scanlines(camera_frame, half_width, third_height, lower["center"], self.tilt_angle, self.cameraMatrix)

        if(total>=0):
            self.target_points.extend(grid.reshape(-1,2))
            payload = extract_payload(camera_frame,grid)
            if payload is not None and payload in layout.markers:
                center = np.array([x,y])
                relative_to_robot = self.calculate_robot_pos_from_contour_center(contour["center"],lower["center"],center)
                self.result_points.append({
                    "center": center,
                    "id": payload,
                    "relative_to_robot": relative_to_robot
                })


    def process_large_contour_pair(self, contour, threshold_frame):
        image_height, image_width = threshold_frame.shape
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
            return
        if top_left < 0:
            return
        if bottom_right < 0:
            return
        if bottom_left < 0:
            return

        top_right_point = contour["center"] + \
            contour["direction"] * top_right
        top_left_point = contour["center"] - \
            contour["direction"] * top_left
        bottom_right_point = lower["center"] + \
            lower["direction"] * bottom_right
        bottom_left_point = lower["center"] - \
            lower["direction"] * bottom_left

        if top_left_point[0] > top_right_point[0]:
            top_left_point, top_right_point = top_right_point, top_left_point

        if bottom_left_point[0] > bottom_right_point[0]:
            bottom_left_point, bottom_right_point = bottom_right_point, bottom_left_point

        points = np.array([top_right_point, bottom_right_point,
                           bottom_left_point, top_left_point])

        self.target_points.extend(points)

        retval, rvec, tvec = cv2.solvePnP(self.target_world_coordinates, points,
                                          self.cameraMatrix, self.distortionMatrix)

        if retval:
            imgpts, jac = cv2.projectPoints(
                self.totransform_world_coordinates, rvec, tvec, self.cameraMatrix, self.distortionMatrix)

            grid_pts, [center] = np.split(np.squeeze(imgpts), [-1])

            if(center[0] < 0 or center[1] < 0 or center[0] >= image_width or center[1] >= image_height):
                return

            self.grid_points.extend(grid_pts)

            colors_bool = self.extract_grid(threshold_frame, grid_pts)

            if colors_bool is not None:
                if (np.all(colors_bool[0, :] == True) and np.all(colors_bool[1, :] == False) and
                        np.all(colors_bool[2, :] == True) and np.all(colors_bool[3, :] == False)):

                    payload_bool = np.concatenate(
                        [colors_bool[4, :], colors_bool[5, :]])
                    payload_value = np.packbits(payload_bool)[0]

                    if(payload_value in layout.markers):

                        relative_to_robot = self.calculate_robot_pos_from_contour_center(contour["center"], lower["center"], center)

                        self.result_points.append({
                            "center": center,
                            "id": payload_value,
                            "relative_to_robot": relative_to_robot
                        })

    def calculate_robot_pos_from_contour_center(self, upper_center, lower_center, center):
        camera_plane_height = self.calculate_camera_plane_height(upper_center, lower_center)

        d = 3.56 / camera_plane_height

        x_prime, y_prime = image_coords_to_image_plane_coords(
            center, self.cameraMatrix, self.distortionMatrix)
        ax = math.atan2(x_prime, 1.0)

        relative_to_camera = np.array(
            [math.sin(ax) * d, math.cos(ax) * d])
        relative_to_robot = (np.concatenate(
            [relative_to_camera, [1]]) @ self.camera_transform_inv)[:2]
        return relative_to_robot

    def add_markers_image(self, camera_frame):
        image_height, image_width, _ = camera_frame.shape

        hsv_frame = cv2.cvtColor(camera_frame, cv2.COLOR_BGR2HSV)
        #v_frame = hsv_frame[:,:,2]

        threshold_frame = cv2.inRange(
            hsv_frame, self.low_limit_hsv, self.high_limit_hsv)
        #threshold_frame = cv2.adaptiveThreshold(v_frame,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 111,29)

        self.raw_contours, hierarchy = cv2.findContours(
            threshold_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        self.filter_raw_contours(image_width, image_height)
        self.pair_contours()


        self.target_points = []
        self.grid_points = []
        self.result_points = []
        self.scan_lines = []

        for contour in self.contour_list:
            if not contour["lower"]:
                continue
            #if(np.linalg.norm(contour["center"]-contour["lower"]["center"])>30):
            self.process_large_contour_pair(contour, threshold_frame)
            #else:
            #    self.process_small_contour_pair(contour, camera_frame)

        for pt in self.result_points:
            markerfinder_position_solver.solver.add_marker(pt["relative_to_robot"], pt["id"])


    def extract_grid(self, threshold_frame, grid_pts):
        image_height, image_width = threshold_frame.shape

        grid_colors = np.empty((4 * 6))
        for i in range(len(grid_pts)):
            point = (grid_pts[i] + .5).astype(np.int)

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
            p2 = tuple((contour["center"] + contour["direction"]
                        * contour["major"]).astype(np.int))
            cv2.line(output_frame, p1, p2, (0, 255, 0), 1)

        upper_contours = [contour["contour"]
                          for contour in self.contour_list if contour["lower"]]
        lower_contours = [contour["contour"]
                          for contour in self.contour_list if contour["islower"]]
        other_contours = [contour["contour"] for contour in self.contour_list if not(
            contour["islower"] or contour["lower"])]

        #cv2.drawContours(output_frame, self.raw_contours, -1, (255, 0, 255), 1)
        #cv2.drawContours(output_frame, other_contours, -1, (128, 128, 128), 1)
        #cv2.drawContours(output_frame, upper_contours, -1, (0, 0, 255), 1)
        #cv2.drawContours(output_frame, lower_contours, -1, (255, 0, 0), 1)

        for line in self.scan_lines:
            p1 = tuple(line[0].astype(np.int))
            p2 = tuple(line[1].astype(np.int))
            cv2.line(output_frame, p1, p2, (0, 255, 255), 1)

        for point in self.target_points:
            if point[0] >= 0 and point[1] >= 0 and point[0] < output_frame.shape[1] and point[1] < output_frame.shape[0]:
                cv2.circle(output_frame, tuple(
                    point.astype(np.int)), 0, (255, 255, 255), -1)
                pass

        for point in self.grid_points:
            if point[0] >= 0 and point[1] >= 0 and point[0] < output_frame.shape[1] and point[1] < output_frame.shape[0]:
                #cv2.circle(output_frame, tuple(
                #    (point + .5).astype(np.int)), 0, (255, 0, 255), -1)
                pass

        for result_point in self.result_points:
            center_tup = tuple(result_point["center"].astype(np.int))

            cv2.circle(output_frame, center_tup, 3, (0, 255, 255), -1)

            cv2.putText(output_frame, str(result_point["id"]),
                        center_tup,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        .4,
                        (255, 255, 0),
                        1)


        cv2.putText(output_frame, self.camera_name,
                    (10,200),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    .4,
                    (255, 255, 0),
                    1)
        markerfinder_position_solver.solver.draw(output_frame)
        return output_frame
