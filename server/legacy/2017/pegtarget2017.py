#!/usr/bin/env python3

# Find the peg target for 2017 STEAMWORKS
# This routine should not use any WPILib classes. That keeps it portable, making testing simpler.

import cv2
import numpy
import json


class PegTarget2017(object):
    '''Find peg target for Steamworks 2017'''

    # real world dimensions of the peg target
    TARGET_WIDTH = 10.25        # inches
    TARGET_HEIGHT = 5.0         # inches

    def __init__(self, calib_file):
        # Color threshold values, in HSV space
        self.low_limit_hsv = numpy.array((70, 60, 30), dtype=numpy.uint8)
        self.high_limit_hsv = numpy.array((100, 255, 255), dtype=numpy.uint8)

        # distance between the two target bars, in units of the width of a bar
        self.peg_target_separation = 3.5

        # max distance in pixels that a contour can from the guessed location
        self.peg_max_target_dist = 50

        # pixel area of the bounding rectangle - just used to remove stupidly small regions
        self.peg_contour_min_area = 100

        self.peg_approx_polydp_error = 0.06

        # Probably should use morphologyEx if needed
        # self.erodeKernel = numpy.ones((3,3), numpy.uint8)
        # self.erodeIterations = 0

        self.width_separation_ratio_max = 0.6

        self.hsv_frame = None
        self.threshold_frame = None

        # output results
        self.target_contour = None

        with open(calib_file) as f:
            json_data = json.load(f)
            self.cameraMatrix = numpy.array(json_data["camera_matrix"])
            self.distortionMatrix = numpy.array(json_data["distortion"])

        # Corners of the peg target in real world dimensions
        self.peg_target_coords = numpy.array([[-PegTarget2017.TARGET_WIDTH/2.0, -PegTarget2017.TARGET_HEIGHT/2.0, 0.0],
                                              [-PegTarget2017.TARGET_WIDTH/2.0,  PegTarget2017.TARGET_HEIGHT/2.0, 0.0],
                                              [ PegTarget2017.TARGET_WIDTH/2.0,  PegTarget2017.TARGET_HEIGHT/2.0, 0.0],
                                              [ PegTarget2017.TARGET_WIDTH/2.0, -PegTarget2017.TARGET_HEIGHT/2.0, 0.0]])

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
            center, widths = PegTarget2017.contour_center_width(c)
            area = widths[0] * widths[1]
            if area > self.peg_contour_min_area:
                # TODO: use a simple class? Maybe use "attrs" package?
                contour_list.append({'contour': c, 'center': center, 'widths': widths, 'area': area})

        # Sort the list of contours from biggest area to smallest
        contour_list.sort(key=lambda c: c['area'], reverse=True)

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
            PegTarget2017.sort_corners(cnrlist)   # in place sort
            image_corners = numpy.array(cnrlist)

            retval, rvec, tvec = cv2.solvePnP(self.peg_target_coords, image_corners,
                                              self.cameraMatrix, self.distortionMatrix)
            if retval:
                # Return values are 3x1 matrices. Convert to Python lists
                return rvec.flatten().tolist(), tvec.flatten().tolist()

        # no target found. Return "error"
        return None, None

    def prepare_output_image(self, output_frame):
        '''Prepare output image for drive station. Draw the found target contour.'''

        if self.target_contour is not None:
            cv2.drawContours(output_frame, [self.target_contour], -1, (255, 255, 255), 1)

        return

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

        # print('ratio:', cand_width / candidate['widths'][1])
        if cand_width / candidate['widths'][1] > self.width_separation_ratio_max:
            return None

        # Based on the candidate location and x-width, compute guesses where the other bar should be
        test_locations = []
        cy = candidate['center'][1]
        dx = int(self.peg_target_separation * cand_width)
        x = cand_x + dx
        if x < width:
            test_locations.append((x, cy))
        x = cand_x - dx
        if x > 0:
            test_locations.append((x, cy))

        # if neither location is inside the image, reject
        if not test_locations:
            return None

        # find the closest contour to either of the guessed locations
        second_cont_index = None
        distance = self.peg_max_target_dist
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

        # see if there is a second contour below. This happens if the peg obscures part of it.
        third_cont_index = None

        second_cont_x = contour_list[second_cont_index]['center'][0]
        second_cont_y = contour_list[second_cont_index]['center'][1]
        second_cont_width = contour_list[second_cont_index]['widths'][0]

        for cont3 in range(cand_index+1, len(contour_list)):
            if cont3 == second_cont_index:
                continue

            center3 = contour_list[cont3]['center']
            width3 = contour_list[cont3]['widths']

            # distance between 2nd and 3rd contours should be less than height of main contour
            delta_y = abs(second_cont_y - center3[1])
            if delta_y > cand_height:
                # too tall
                continue

            # 2nd and 3rd contour need to have almost the same X and almost the same width (correct??)
            if abs(second_cont_x - center3[0]) < 10 and abs(second_cont_width - width3[0]) < 10:
                third_cont_index = cont3
                break

        # We now have all the pieces for this candidate

        # Important cut: the actual distance between the two bars in porportion to the actual
        #    average width should be similar to the peg_target_separation variable
        ave_second_bar_x = second_cont_x
        ave_second_bar_width = second_cont_width
        if third_cont_index is not None:
            ave_second_bar_x = (ave_second_bar_x + contour_list[third_cont_index]['center'][0]) / 2
            ave_second_bar_width = (ave_second_bar_width + contour_list[third_cont_index]['widths'][0]) / 2

        delta_x = abs(cand_x - ave_second_bar_x)
        ave_width = (cand_width + ave_second_bar_width) / 2
        ratio = delta_x / (ave_width * self.peg_target_separation)
        # print('deltaX', deltaX, aveW, ratio)
        if ratio > 1.3 or ratio < 0.7:
            # not close enough to 1
            return None

        # DONE! We have a winner! Maybe!
        # Now form a quadrilateral around all the contours.
        # First, combine them using convexHull and the fit a polygon to it. If we get a 4-sided shape we are set.

        all_contours = [candidate['contour'], contour_list[second_cont_index]['contour']]
        if third_cont_index is not None:
            all_contours.append(contour_list[third_cont_index]['contour'])

        combined = numpy.vstack(all_contours)
        hull = cv2.convexHull(combined)

        target_contour = PegTarget2017.quad_fit(hull, self.peg_approx_polydp_error)
        if len(target_contour) == 4:
            return target_contour

        return None


# --------------------------------------------------------------------------------

def process_files(peg_processor, input_files, output_dir):
    '''Process the files and output the marked up image'''
    import os.path

    for image_file in input_files:
        print()
        print(image_file)
        bgr_frame = cv2.imread(image_file)
        rvec, tvec = peg_processor.process_image(bgr_frame)
        print('rvec:', rvec)
        print('tvec:', tvec)

        peg_processor.prepare_output_image(bgr_frame)
        outfile = os.path.join(output_dir, os.path.basename(image_file))
        # print('{} -> {}'.format(image_file, outfile))
        cv2.imwrite(outfile, bgr_frame)
    return


def time_processing(peg_processor, input_files):
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
                peg_processor.process_image(bgr_frame)

            cnt += 1

    deltat = time() - startt

    print("{0} frames in {1:.3f} seconds = {2:.2f} msec/call, {3:.2f} FPS".format(
        cnt, deltat, 1000.0 * deltat / cnt, cnt / deltat))
    CodeTimer.outputTimers()
    return


def main():
    '''Main routine, for testing'''
    import argparse

    parser = argparse.ArgumentParser(description='2017 peg target')
    parser.add_argument('--output_dir', help='Output directory for processed images')
    parser.add_argument('--time', action='store_true', help='Loop over files and time it')
    parser.add_argument('--calib', help='Calibration file')
    parser.add_argument('input_files', nargs='+', help='input files')

    args = parser.parse_args()

    peg_processor = PegTarget2017(args.calib)
    if args.output_dir is not None:
        process_files(peg_processor, args.input_files, args.output_dir)
    elif args.time:
        time_processing(peg_processor, args.input_files)

    return


# Main routine
# This is for development/testing
if __name__ == '__main__':
    main()
