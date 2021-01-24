#!/usr/bin/env python3

'''Vision server for 2019 Deep Space'''

import cv2
from networktables.util import ntproperty

from visionserver import VisionServer, main
from genericfinder import GenericFinder
from rrtargetfinder2019 import RRTargetFinder2019


class VisionServer2019(VisionServer):

    # Retro-reflective targer finding parameters

    # Color threshold values, in HSV space
    rrtarget_hue_low_limit = ntproperty('/SmartDashboard/vision/rrtarget/hue_low_limit', 65,
                                        doc='Hue low limit for thresholding (rrtarget mode)')
    rrtarget_hue_high_limit = ntproperty('/SmartDashboard/vision/rrtarget/hue_high_limit', 100,
                                         doc='Hue high limit for thresholding (rrtarget mode)')

    rrtarget_saturation_low_limit = ntproperty('/SmartDashboard/vision/rrtarget/saturation_low_limit', 75,
                                               doc='Saturation low limit for thresholding (rrtarget mode)')
    rrtarget_saturation_high_limit = ntproperty('/SmartDashboard/vision/rrtarget/saturation_high_limit', 255,
                                                doc='Saturation high limit for thresholding (rrtarget mode)')

    rrtarget_value_low_limit = ntproperty('/SmartDashboard/vision/rrtarget/value_low_limit', 135,
                                          doc='Value low limit for thresholding (rrtarget mode)')
    rrtarget_value_high_limit = ntproperty('/SmartDashboard/vision/rrtarget/value_high_limit', 255,
                                           doc='Value high limit for thresholding (rrtarget mode)')

    rrtarget_exposure = ntproperty('/SmartDashboard/vision/rrtarget/exposure', 0, doc='Camera exposure for rrtarget (0=auto)')

    intake_center_line = ((126, 0), (99, 424))
    # intake_center_line = None

    def __init__(self, calib_file, test_mode=False):
        super().__init__(initial_mode='driver_target', test_mode=test_mode)

        self.camera_device_intake = '/dev/v4l/by-id/usb-046d_Logitech_Webcam_C930e_70E19A9E-video-index0'    # for driver and rrtarget processing
        self.camera_device_target = '/dev/v4l/by-id/usb-046d_Logitech_Webcam_C930e_DF7AF0BE-video-index0'    # for line and hatch processing

        self.add_cameras()

        self.generic_finder_target = GenericFinder("driver_target", "target")      # finder_id=1.0
        self.add_target_finder(self.generic_finder_target)                         # TODO make it a seperate finder to draw a line down the screen where
                                                                                   # the line at the bottom of the rocket will be

        self.generic_finder_intake = GenericFinder("driver_intake", "intake", finder_id=2.0, rotation=cv2.ROTATE_90_COUNTERCLOCKWISE, line_coords=self.intake_center_line)
        self.add_target_finder(self.generic_finder_intake)

        self.rrtarget_finder = RRTargetFinder2019(calib_file, intake_finder=self.generic_finder_intake)  # finder_id=3.0
        self.add_target_finder(self.rrtarget_finder)

        self.rrtarget_finder_plain = RRTargetFinder2019(calib_file, name='rrtarget_plain', finder_id=4.0)
        self.add_target_finder(self.rrtarget_finder_plain)

        self.update_parameters()

        # start in driver_intake mode to get cameras going. Will switch to 'driver_target' after 1 sec.
        self.switch_mode('driver_intake')
        return

    def update_parameters(self):
        '''Update processing parameters from NetworkTables values.
        Only do this on startup or if "tuning" is on, for efficiency'''

        # Make sure to add any additional created properties which should be changeable

        self.rrtarget_finder.set_color_thresholds(self.rrtarget_hue_low_limit, self.rrtarget_hue_high_limit,
                                                  self.rrtarget_saturation_low_limit, self.rrtarget_saturation_high_limit,
                                                  self.rrtarget_value_low_limit, self.rrtarget_value_high_limit)
        self.rrtarget_finder_plain.set_color_thresholds(self.rrtarget_hue_low_limit, self.rrtarget_hue_high_limit,
                                                        self.rrtarget_saturation_low_limit, self.rrtarget_saturation_high_limit,
                                                        self.rrtarget_value_low_limit, self.rrtarget_value_high_limit)
        return

    def add_cameras(self):
        '''Add the cameras'''

        self.add_camera('intake', self.camera_device_intake, True)
        self.add_camera('target', self.camera_device_target, False)
        return

    def mode_after_error(self):
        if self.active_mode == 'driver_intake':
            return 'driver_target'
        return 'driver_intake'


# Main routine
if __name__ == '__main__':
    main(VisionServer2019)
