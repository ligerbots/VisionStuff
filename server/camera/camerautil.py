#!/usr/bin/python3

# Utility routines for cameras.
# Keep these separate from the Camera classes so that we don't import cscore

import logging
import json
from numpy import array


def load_calibration_file(calib_file, rotation):
    '''Load calibration information from the specified file'''

    logging.info(f'Loading calibration from {calib_file}')
    try:
        with open(calib_file) as f:
            json_data = json.load(f)
            cal_matrix = array(json_data["camera_matrix"])
            dist_matrix = array(json_data["distortion"])

        if abs(rotation) == 90:
            # swap the x,y values
            cal_matrix[1, 1], cal_matrix[0, 0] = cal_matrix[0, 0], cal_matrix[1, 1]
            cal_matrix[1, 2], cal_matrix[0, 2] = cal_matrix[0, 2], cal_matrix[1, 2]
            dist_matrix[0, 3], dist_matrix[0, 2] = dist_matrix[0, 2], dist_matrix[0, 3]

        return cal_matrix, dist_matrix
    except Exception as e:
        logging.warn("Error loading calibration file:", e)
    return None, None
