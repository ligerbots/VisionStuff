#!/usr/bin/env python3

import cv2
import numpy
import json
import math


class GoalFinder2020(object):
    '''Find high goal target for Infinite Recharge 2020'''

    # real world dimensions of the goal target
    # These are the full dimensions around both strips
    TARGET_STRIP_LENGTH = 19.625    # inches
    TARGET_HEIGHT = 17.0            # inches
    TARGET_TOP_WIDTH = 39.25        # inches
    TARGET_BOTTOM_WIDTH = math.acos(TARGET_HEIGHT / TARGET_STRIP_LENGTH)

    # [0, 0] is center of the quadrilateral drawn around the high goal target
    # [top_left, bottom_left, bottom_right, top_right]
    real_world_coordinates = [
        [-TARGET_TOP_WIDTH / 2, TARGET_HEIGHT / 2],
        [-TARGET_BOTTOM_WIDTH / 2, -TARGET_HEIGHT / 2],
        [TARGET_BOTTOM_WIDTH / 2, -TARGET_HEIGHT / 2],
        [TARGET_TOP_WIDTH / 2, TARGET_HEIGHT / 2]
    ]

    def __init__(self, calib_file):
        self.name = 'goalfinder'
        self.finder_id = 1.0
        self.camera = 'driver'      # TODO: change this
        self.exposure = 1

        # Color threshold values, in HSV space
        self.low_limit_hsv = numpy.array((65, 75, 135), dtype=numpy.uint8)
        self.high_limit_hsv = numpy.array((100, 255, 255), dtype=numpy.uint8)

        # pixel area of the bounding rectangle - just used to remove stupidly small regions
        self.contour_min_area = 80

        # Allowed "error" in the perimeter when fitting using approxPolyDP (in quad_fit)
        self.approx_polydp_error = 0.06     # TODO: experiment with this starting with very small and going larger
#0.06
        self.hull = None
        # ratio of height to width of one retroreflective strip
        self.height_width_ratio = GoalFinder2020.TARGET_HEIGHT / GoalFinder2020.TARGET_TOP_WIDTH

        # camera mount angle (radians)
        # NOTE: not sure if this should be positive or negative
        self.tilt_angle = math.radians(-7.5)

        self.hsv_frame = None
        self.threshold_frame = None

        # DEBUG values
        self.top_contours = None
        self.target_locations = None

        # output results
        self.target_contours = None

        if calib_file:
            with open(calib_file) as f:
                json_data = json.load(f)
                self.cameraMatrix = numpy.array(json_data["camera_matrix"])
                self.distortionMatrix = numpy.array(json_data["distortion"])

        self.outer_corners = []
        self.test_outer_corners=[];
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
    def dot_product(ax, ay, bx, by):

        return (ax * bx) + (ay * by)

    def get_outer_corners(self, hull):
        '''Return the four true corners of a contour'''
        if(len(hull)<4):
            return None

        dps = []    #dot products
        for i in range(len(hull)):#for each hull point,
            prev_pt = hull[i - 1][0]
            pt = hull[i][0]
            next_pt = hull[(i + 1) % len(hull)][0]
            ax = (pt[0] - prev_pt[0])
            ay = (pt[1] - prev_pt[1])

            da=math.hypot(ax,ay)
            bx = (pt[0] - next_pt[0])
            by = (pt[1] - next_pt[1])
            db=math.hypot(bx,by)

            if(da>0):#Make sure non 0
                dps.append([i,#keep index for sorting later
                        GoalFinder2020.dot_product( ax,ay,bx,by) / da / db,#calculate dot prod with last
                        da,db#also get dist info
                       ])
        if(len(dps)<4):
            return None

        self.test_outer_corners=[]

        #group close points together
        max_group_dist=7

        nddps=[]
        curr_max_group=-1
        test_color=0;
        for i in range(len(dps)):
            if(dps[i][2]<max_group_dist or curr_max_group==-1):#
                if(dps[curr_max_group][1]<dps[i][1] or curr_max_group==-1):
                    curr_max_group=i
            else:
                nddps.append(dps[curr_max_group])
                curr_max_group=i
                test_color=i
            self.test_outer_corners.append([hull[dps[i][0]][0],test_color]);
        nddps.append(dps[curr_max_group])
        if(len(nddps)<4):
            return None
        if(nddps[0][2]<max_group_dist):#special case for first and last points
            if(nddps[0][1]<nddps[-1][1]):
                del nddps[0]
            else:
                del nddps[-1]
        if(len(nddps)<4):
            return None
        for i in range(len(nddps)):#redo dot prod calcs without close points
            prev_pt = hull[nddps[i - 1][0]][0]
            pt = hull[nddps[i][0]][0]
            next_pt = hull[nddps[(i + 1) % len(nddps)][0]][0]
            ax = (pt[0] - prev_pt[0])
            ay = (pt[1] - prev_pt[1])

            da=math.hypot(ax,ay)
            bx = (pt[0] - next_pt[0])
            by = (pt[1] - next_pt[1])
            db=math.hypot(bx,by)
            nddps[i][1]=GoalFinder2020.dot_product( ax,ay,bx,by) / da / db

        print("nddps",len(nddps),nddps)

        nddps.sort(key=lambda pt: pt[1], reverse=True) #sort it by max dot prod(=min angle)
        print("selected",nddps)

        pts = [numpy.concatenate([hull[pt[0]][0],[int(math.degrees(math.acos(pt[1]))),pt[0]]]) for pt in nddps[0:4]]

        print("pts",len(pts),pts)

        return pts

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
        self.hull = None
        self.outer_corners = None

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
            center, widths = GoalFinder2020.contour_center_width(c)
            area = widths[0] * widths[1]
            if area > self.contour_min_area:        # area cut
                contour_list.append({'contour': c, 'center': center, 'widths': widths, 'area': area})

        # Sort the list of contours from biggest area to smallest
        contour_list.sort(key=lambda c: c['area'], reverse=True)

        # DEBUG
        self.top_contours = [x['contour'] for x in contour_list]

        # try only the 5 biggest regions at most
        for candidate_index in range(min(5, len(contour_list))):
            self.target_contour = self.test_candidate_contour(contour_list[candidate_index])
            if self.target_contour is not None:
                break

        # if self.target_contour is not None:
        #     # The target was found. Convert to real world co-ordinates.

        #     cnt = numpy.squeeze(self.target_contour).tolist()

        #     # Need to convert the contour (integer) into a matrix of corners (float; all 4 outside cnrs)

        #     # Important to get the corners in the right order, ***matching the real world ones***
        #     # Remember that y in the image increases *down*
        #     self.outer_corners = GoalFinder2020.get_outer_corners(cnt)

        #     print("Outside corners: ", self.outer_corners)
        #     print("Real World target_coords: ", self.real_world_coordinates)

        #     retval, rvec, tvec = cv2.solvePnP(self.real_world_coordinates, self.outer_corners,
        #                                       self.cameraMatrix, self.distortionMatrix)
        #     if retval:
        #         result = [1.0, self.finder_id, ]
        #         result.extend(self.compute_output_values(rvec, tvec))
        #         return result

        # no target found. Return "failure"
        return [0.0, self.finder_id, 0.0, 0.0, 0.0]

    def prepare_output_image(self, input_frame):
        '''Prepare output image for drive station. Draw the found target contour.'''

        output_frame = input_frame.copy()

        #if self.top_contours:
        #    cv2.drawContours(output_frame, self.top_contours, -1, (0, 0, 255), 2)

        #if self.target_contour is not None:
        #    cv2.drawContours(output_frame, [self.target_contour], -1, (255, 0, 0), 2)

        if self.hull is not None:
            cv2.drawContours(output_frame, [self.hull], -1, (0, 255, 0), 2)

            for cnr in self.hull:
                cv2.circle(output_frame, (cnr[0][0], cnr[0][1]), 0, (0, 0, 255), -1, lineType=8, shift=0)
        if self.test_outer_corners is not None:
            for e in self.test_outer_corners:
                col=cv2.cvtColor(numpy.uint8([[[e[1]*12,255,255 ]]]), cv2.COLOR_HSV2RGB);
                cv2.circle(output_frame, (e[0][0], e[0][1]), 4, (int(col[0][0][0]),int(col[0][0][1]),int(col[0][0][2])), -1, lineType=8, shift=0)

        if self.outer_corners is not None:
            for i in range(len(self.outer_corners)):
                cv2.circle(output_frame, (self.outer_corners[i][0], self.outer_corners[i][1]), 3, (255, 255, 255), -1, lineType=8, shift=0)
                #cv2.putText(output_frame,str(self.outer_corners[i][2]), (self.outer_corners[i][0]-30, self.outer_corners[i][1]), 0, .4, (255,255,255))
                #cv2.putText(output_frame,str(self.outer_corners[i][3]), (self.outer_corners[i][0], self.outer_corners[i][1]), 0, .4, (200,200,200))


        # for loc in self.target_locations:
        #     cv2.drawMarker(output_frame, loc, (0, 255, 255), cv2.MARKER_TILTED_CROSS, 15, 3)

        return output_frame

    def test_candidate_contour(self, candidate):
        '''Determine the true target contour out of potential candidates'''

        # cand_width = candidate['widths'][0]
        # cand_height = candidate['widths'][1]

        # TODO: make addition cuts here
        self.hull = cv2.convexHull(candidate['contour'])
        #print("Hull fit: " + str(self.hull))
        self.outer_corners = self.get_outer_corners(self.hull)
        contour = cv2.approxPolyDP(candidate['contour'], self.approx_polydp_error, True)
        #contour = self.quad_fit(self.hull, self.approx_polydp_error)

        print('found', len(contour), 'sides')
        if len(contour) <= 4:
            return contour

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


def process_files(line_finder, input_files, output_dir):
    '''Process the files and output the marked up image'''
    import os.path

    for image_file in input_files:
        # print()
        # print(image_file)
        bgr_frame = cv2.imread(image_file)
        result = line_finder.process_image(bgr_frame)
        print(image_file, result[0], result[1], result[2], math.degrees(result[3]), math.degrees(result[4]))

        bgr_frame = line_finder.prepare_output_image(bgr_frame)

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

    parser = argparse.ArgumentParser(description='2019 rrtarget finder')
    parser.add_argument('--output_dir', help='Output directory for processed images')
    parser.add_argument('--time', action='store_true', help='Loop over files and time it')
    parser.add_argument('--calib_file', help='Calibration file')
    parser.add_argument('input_files', nargs='+', help='input files')

    args = parser.parse_args()

    rrtarget_finder = GoalFinder2020(args.calib_file)

    if args.output_dir is not None:
        process_files(rrtarget_finder, args.input_files, args.output_dir)
    elif args.time:
        time_processing(rrtarget_finder, args.input_files)

    return


# Main routine
# This is for development/testing
if __name__ == '__main__':
    main()
