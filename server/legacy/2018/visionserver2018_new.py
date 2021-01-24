#!/usr/bin/env python3

'''Vision server for 2018 Power Up -- updated to conform with VisionServer superclass'''

import cv2
# import logging

from networktables.util import ntproperty
# from networktables import NetworkTables

from visionserver import VisionServer, main
from switchtarget2018 import SwitchTarget2018
from cubefinder2018 import CubeFinder2018
from genericfinder import GenericFinder


class VisionServer2018_new(VisionServer):

    # Cube finding parameters

    # Color threshold values, in HSV space
    cube_hue_low_limit = ntproperty('/SmartDashboard/vision/cube/hue_low_limit', 25,
                                    doc='Hue low limit for thresholding (cube mode)')
    cube_hue_high_limit = ntproperty('/SmartDashboard/vision/cube/hue_high_limit', 75,
                                     doc='Hue high limit for thresholding (cube mode)')

    cube_saturation_low_limit = ntproperty('/SmartDashboard/vision/cube/saturation_low_limit', 95,
                                           doc='Saturation low limit for thresholding (cube mode)')
    cube_saturation_high_limit = ntproperty('/SmartDashboard/vision/cube/saturation_high_limit', 255,
                                            doc='Saturation high limit for thresholding (cube mode)')

    cube_value_low_limit = ntproperty('/SmartDashboard/vision/cube/value_low_limit', 95,
                                      doc='Value low limit for thresholding (cube mode)')
    cube_value_high_limit = ntproperty('/SmartDashboard/vision/cube/value_high_limit', 255,
                                       doc='Value high limit for thresholding (cube mode)')

    cube_exposure = ntproperty('/SmartDashboard/vision/cube/exposure', 0, doc='Camera exposure for cube (0=auto)')

    # Switch target parameters

    switch_hue_low_limit = ntproperty('/SmartDashboard/vision/switch/hue_low_limit', 70,
                                      doc='Hue low limit for thresholding (switch mode)')
    switch_hue_high_limit = ntproperty('/SmartDashboard/vision/switch/hue_high_limit', 100,
                                       doc='Hue high limit for thresholding (switch mode)')

    switch_saturation_low_limit = ntproperty('/SmartDashboard/vision/switch/saturation_low_limit', 100,
                                             doc='Saturation low limit for thresholding (switch mode)')
    switch_saturation_high_limit = ntproperty('/SmartDashboard/vision/switch/saturation_high_limit', 255,
                                              doc='Saturation high limit for thresholding (switch mode)')

    switch_value_low_limit = ntproperty('/SmartDashboard/vision/switch/value_low_limit', 130,
                                        doc='Value low limit for thresholding (switch mode)')
    switch_value_high_limit = ntproperty('/SmartDashboard/vision/switch/value_high_limit', 255,
                                         doc='Value high limit for thresholding (switch mode)')

    switch_exposure = ntproperty('/SmartDashboard/vision/switch/exposure', 6, doc='Camera exposure for switch (0=auto)')

    camera_height = ntproperty('/SmartDashboard/vision/camera_height', 23.0, doc='Camera height (inches)')

    def __init__(self, calib_file, test_mode=False):
        super().__init__(initial_mode='switch', test_mode=test_mode)

        self.camera_device_driver = '/dev/v4l/by-id/usb-046d_Logitech_Webcam_C930e_DF7AF0BE-video-index0'
        self.camera_device_vision = '/dev/v4l/by-id/usb-046d_Logitech_Webcam_C930e_70E19A9E-video-index0'

        self.add_cameras()

        self.add_target_finder(SwitchTarget2018(calib_file))
        self.add_target_finder(CubeFinder2018(calib_file))
        # driver camera with no processing, but rotate 90 deg
        self.add_target_finder(GenericFinder(name='driver', camera='driver', rotation=cv2.ROTATE_90_COUNTERCLOCKWISE))
        # intake camera with no processing
        self.add_target_finder(GenericFinder(name='intake', camera='intake'))

        self.update_parameters()

        # Start in cube mode, then then switch to initial_mode after camera is fully initialized
        self.switch_mode('cube')
        return

    def update_parameters(self):
        '''Update processing parameters from NetworkTables values.
        Only do this on startup or if "tuning" is on, for efficiency'''

        # Make sure to add any additional created properties which should be changeable

        finder = self.target_finders['switch']
        finder.set_color_thresholds(self.switch_hue_low_limit, self.switch_hue_high_limit,
                                    self.switch_saturation_low_limit, self.switch_saturation_high_limit,
                                    self.switch_value_low_limit, self.switch_value_high_limit)
        finder.exposure = self.switch_exposure

        finder = self.target_finders['cube']
        finder.set_color_thresholds(self.cube_hue_low_limit, self.cube_hue_high_limit,
                                    self.cube_saturation_low_limit, self.cube_saturation_high_limit,
                                    self.cube_value_low_limit, self.cube_value_high_limit)
        finder.camera_height = self.camera_height
        finder.exposure = self.cube_exposure

        return

    def add_cameras(self):
        '''add a single camera at /dev/videoN, N=camera_device'''

        self.add_camera('intake', self.camera_device_vision, True)
        self.add_camera('driver', self.camera_device_driver, False)
        return


# Main routine
if __name__ == '__main__':
    # call the standard main routine, passing in the class type we want created, ie VisionServer2018_new
    main(VisionServer2018_new)
