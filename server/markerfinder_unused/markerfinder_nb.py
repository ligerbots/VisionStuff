# don't use!

import cv2
import numpy as np
import math

from genericfinder import GenericFinder, main

import numba as nb
from numba.np.extensions import cross2d
from numba.experimental import jitclass
import layout

@nb.njit(nb.float64(nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:]))
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

@nb.njit(nb.float64(nb.float64[:,:], nb.float64[:], nb.float64[:]))
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




@jitclass([("center", nb.float64[:]),
           ("id", nb.int64),
           ("real_position", nb.float64[:])])

class ResultPointInfo:
    def __init__(self,center,id,real_position):
        self.center=center
        self.id=id
        self.real_position=self.real_position

contour_type=nb.types.Array(nb.int32, 3, "C")

@jitclass([("contour", contour_type),
           ("center", nb.float64[:]),
           ("direction", nb.float64[:]),
           ("major", nb.float64),
           ("minor", nb.float64),
           ("lower_index", nb.int64),
           ("islower", nb.boolean)])
class ContourInfo:
    def __init__(self, contour, center, direction, major, minor, lower_index, islower):
        self.contour=contour
        self.center=center
        self.direction=direction
        self.major=major
        self.minor=minor
        self.lower_index=lower_index
        self.islower=islower

debug_annotation_type = nb.types.Tuple((nb.int64[:], nb.types.unicode_type))
debug_point_type = nb.types.Tuple((nb.int64[:], nb.int64[:]))

contour_info_type=ContourInfo.class_type.instance_type
result_point_info_type = ResultPointInfo.class_type.instance_type

@jitclass([
           ("debug_annotations", nb.types.ListType(debug_annotation_type)),
           ("debug_points", nb.types.ListType(debug_point_type)),

           ("raw_contours", nb.types.ListType(contour_type)),
           ("contour_list", nb.types.ListType(contour_info_type)),
           ("result_points", nb.types.ListType(result_point_info_type)),

           ("tilt_angle", nb.float64),
           ("camera_height", nb.float64),
           ("camera_matrix", nb.float64[:,:]),
           ("distortion_matrix", nb.float64[:,:])
          ])

class MarkerFinderContext:
    def __init__(self, tilt_angle, camera_height, camera_matrix, distortion_matrix):
        self.debug_annotations=nb.typed.List.empty_list(debug_annotation_type)
        self.debug_points=nb.typed.List.empty_list(debug_point_type)

        self.raw_contours=nb.typed.List.empty_list(contour_type)
        self.contour_list=nb.typed.List.empty_list(contour_info_type)
        self.result_points=nb.typed.List.empty_list(result_point_info_type)

        self.tilt_angle=tilt_angle
        self.camera_height=camera_height
        self.camera_matrix=camera_matrix
        self.distortion_matrix=distortion_matrix


@nb.njit(nb.float64[:](MarkerFinderContext.class_type.instance_type, nb.float64[:]))
def undo_tilt_on_image_plane_coords(ctx,point):
    theta = ctx.tilt_angle
    x, y = point
    return(np.array([
        x/(math.cos(theta)-y*math.sin(theta)),
        (math.sin(theta)+y*math.cos(theta))/(math.cos(theta)-y*math.sin(theta))
    ]))

@nb.njit(nb.float64[:](MarkerFinderContext.class_type.instance_type, nb.float64[:]))
def image_coords_to_image_plane_coords(ctx,point):
    x, y = point
    x_prime = (x - ctx.camera_matrix[0, 2]) / ctx.camera_matrix[0, 0]
    y_prime = -(y - ctx.camera_matrix[1, 2]) / ctx.camera_matrix[1, 1]
    return(np.array([x_prime, y_prime]))


@nb.njit(nb.float64(MarkerFinderContext.class_type.instance_type, nb.float64[:], nb.float64[:]))
def calculate_image_plane_height(ctx, upper_center_image,lower_center_image):
    upper_center = undo_tilt_on_image_plane_coords(ctx, image_coords_to_image_plane_coords(ctx, upper_center_image))

    lower_center = undo_tilt_on_image_plane_coords(ctx, image_coords_to_image_plane_coords(ctx, lower_center_image))

    return(upper_center[1]-lower_center[1])

@nb.njit(nb.types.UniTuple(nb.float64,3)(nb.float64,nb.float64,nb.float64,nb.float64))
def major_minor_axes(moments_m00, moments_mu20, moments_mu02, moments_mu11):
    '''Compute the major/minor axes and orientation of an object from the moments'''

    # See https://en.wikipedia.org/wiki/Image_moment
    # Be careful, different sites define the normalized central moments differently
    # See also http://raphael.candelier.fr/?blog=Image%20Moments

    m00 = moments_m00
    mu20 = moments_mu20 / m00
    mu02 = moments_mu02 / m00
    mu11 = moments_mu11 / m00

    descr = math.sqrt(4.0 * mu11*mu11 + (mu20 - mu02)**2)

    major = math.sqrt(2.0 * (mu20 + mu02 + descr))
    minor = math.sqrt(2.0 * (mu20 + mu02 - descr))

    # note this does not use atan2.
    angle = 0.5 * math.atan(2*mu11 / (mu20-mu02))
    if mu20 < mu02:
        angle += math.pi/2

    return major, minor, angle

def sort_by_y(contour_list):
    contour_list.sort(key=lambda c: c.center[1])



@nb.njit(nb.void(MarkerFinderContext.class_type.instance_type, nb.uint8[:,:,:]))
def process_image_contours(ctx, camera_frame):
    ctx.debug_annotations.clear()
    ctx.debug_points.clear()
    image_height, image_width, _ = camera_frame.shape

    with nb.objmode(hsv_frame="uint8[:,:,:]"):
        hsv_frame = cv2.cvtColor(camera_frame, cv2.COLOR_BGR2HSV)

    with nb.objmode(threshold_frame="uint8[:,:]"):
        threshold_frame = cv2.inRange(
            hsv_frame,
            np.array((0, 0, 0), dtype=np.uint8),
            np.array((255, 255, 100), dtype=np.uint8)
        )

    with nb.objmode():
        python_contours, hierarchy = cv2.findContours(
            threshold_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        ctx.raw_contours = nb.typed.List(python_contours)

    ctx.contour_list.clear()

    for contour in ctx.raw_contours:

        with nb.objmode(w="float64", h="float64",
                        real_area="float64",
                        moments_m00="float64", moments_m10="float64", moments_m01="float64",
                        moments_mu20="float64", moments_mu02="float64", moments_mu11="float64", ):
            x, y, w, h = cv2.boundingRect(contour)
            real_area = cv2.contourArea(contour)

            python_img_moments = cv2.moments(contour)

            moments_m00 = python_img_moments["m00"]
            moments_m10 = python_img_moments["m10"]
            moments_m01 = python_img_moments["m01"]
            moments_mu20 = python_img_moments["m20"]
            moments_mu02 = python_img_moments["m02"]
            moments_mu11 = python_img_moments["m11"]


        if(w > image_width/2 or h > image_height/2):
            continue

        if (real_area < 5):
            continue

        #if(real_area < w*h*.5):
        #    continue

        if(w==h or w<=0 or h <=0 or real_area==0):
            continue

        center = np.array((moments_m10/moments_m00,
                           moments_m01/moments_m00), dtype=np.float64)

        if(moments_mu20==moments_mu02):
            continue

        major, minor, angle = major_minor_axes(moments_m00=moments_m00, moments_mu20=moments_mu20, moments_mu02=moments_mu02, moments_mu11=moments_mu11)

        if(major < minor*2):
             continue

        direction = np.array((math.cos(angle), math.sin(angle)))

        ctx.contour_list.append(ContourInfo(
            contour,
            center,
            direction,
            major,
            minor,
            -1, # lower_index
            False # islower
        ))
    with nb.objmode():
        sort_by_y(ctx.contour_list)

    for i in range(len(ctx.contour_list)):
        contour = ctx.contour_list[i]

        if(contour.islower):
            continue

        center = contour.center
        lower = center[1] + contour.minor*18
        left = center[0] - contour.major
        right = center[0] + contour.major

        for j in range(i+1, len(ctx.contour_list)):
            potential_pair_contour = ctx.contour_list[j]
            if(potential_pair_contour.center[1] > lower):
                break
            if(potential_pair_contour.lower_index>=0 or potential_pair_contour.islower):
                continue
            if(potential_pair_contour.center[0] < left):
                continue
            if(potential_pair_contour.center[0] > right):
                continue
            if(max(contour.major, potential_pair_contour.major)/min(contour.major, potential_pair_contour.major) > 2):
                continue
            if(max(contour.minor, potential_pair_contour.minor)/min(contour.minor, potential_pair_contour.minor) > 2):
                continue
            contour.lower_index = j
            potential_pair_contour.islower = True

            break



class MarkerFinder2021(GenericFinder):
    '''Marker finder for Infinite Recharge at home'''

    def __init__(self, calib_matrix, dist_matrix, camera_name="shooter"):
        super().__init__('markerfinder', camera=camera_name, finder_id=2.0, exposure=0)

        self.ctx = MarkerFinderContext(
            tilt_angle=layout.camera_pos[camera_name]["tilt"],
            camera_height=layout.camera_pos[camera_name]["height"],
            camera_matrix=calib_matrix,
            distortion_matrix=dist_matrix
        )

    def process_image(self, camera_frame):
        process_image(self.ctx, camera_frame)
        return (1.0, self.finder_id, 0, 0, 0.0, 0, 0)

    def prepare_output_image(self, input_frame):
        output_frame = input_frame.copy()

        cv2.drawContours(output_frame, self.ctx.raw_contours, -1, (255, 0, 255), 1)

        for (pt, msg) in self.ctx.debug_annotations:
            cv2.putText(output_frame, msg,
                tuple(pt),
                cv2.FONT_HERSHEY_SIMPLEX,
                .4,
                (255,255,0),
                1)

        for (pt, col) in self.ctx.debug_points:
            cv2.circle(output_frame,tuple(pt), 1, (int(col[0]),int(col[1]),int(col[2])), -1)

        return output_frame


# Main routine
# This is for development/testing
if __name__ == '__main__':
    main(MarkerFinder2021)
