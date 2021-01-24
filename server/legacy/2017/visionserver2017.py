#!/usr/bin/env python3

# Base camera server
# Create a subclass with this years image processing to actually use

import time
import cv2
import numpy
import logging

import cscore
from cscore.imagewriter import ImageWriter
from networktables.util import ntproperty
from networktables import NetworkTables

from pegtarget2017 import PegTarget2017


class VisionServer2017(object):
    '''Vision server for 2017 SteamWorks'''

    # NetworkTable parameters
    output_fps_limit = ntproperty('/vision/output_fps_limit', 15,
                                  doc='FPS limit of frames sent to MJPEG server')

    # fix the TCP port for the main video, so it does not change with multiple cameras
    output_port = ntproperty('/vision/output_port', 1190,
                             doc='TCP port for main image output')

    # Operation modes. Force the value on startup.
    tuning = ntproperty('/vision/tuning', False, writeDefault=True,
                        doc='Tuning mode. Reads processing parameters each time.')
    restart = ntproperty('/vision/restart', False, writeDefault=True,
                         doc='Restart algorithm. Needed after certain param changes.')

    image_width = ntproperty('/vision/width', 640, writeDefault=False,
                             doc='Image width')
    image_height = ntproperty('/vision/height', 480, writeDefault=False,
                              doc='Image height')
    camera_fps = ntproperty('/vision/fps', 30, writeDefault=False,
                            doc='FPS from camera')

    # Color threshold values, in HSV space
    hue_low_limit = ntproperty('/vision/hue_low_limit', 70,
                               doc='Hue low limit for thresholding')
    hue_high_limit = ntproperty('/vision/hue_high_limit', 100,
                                doc='Hue high limit for thresholding')

    saturation_low_limit = ntproperty('/vision/saturation_low_limit', 60,
                                      doc='Saturation low limit for thresholding')
    saturation_high_limit = ntproperty('/vision/saturation_high_limit', 255,
                                       doc='Saturation high limit for thresholding')

    value_low_limit = ntproperty('/vision/value_low_limit', 30,
                                 doc='Value low limit for thresholding')
    value_high_limit = ntproperty('/vision/value_high_limit', 255,
                                  doc='Value high limit for thresholding')

    # distance between the two target bars, in units of the width of a bar
    peg_target_separation = ntproperty('/vision/peg/target_separation', 3.5,
                                       doc='Peg target horizontal separation, in widths')

    # max distance in pixels that a contour can from the guessed location
    peg_max_target_dist = ntproperty('/vision/peg/max_target_dist', 50,
                                     doc='Peg max distance between contour and expected location (pixels)')

    # pixel area of the bounding rectangle - just used to remove stupidly small regions
    peg_contour_min_area = ntproperty('/vision/peg/contour_min_area', 100,
                                      doc='Peg min area of an interesting contour (sq. pixels)')

    peg_approx_polydp_error = ntproperty('/vision/peg/approx_polydp_error', 0.06,
                                         doc='Peg approxPolyDP error')

    image_writer_state = ntproperty('/vision/write_images', False, writeDefault=True,
                                    doc='Turn on saving of images')

    # This ought to be a Choosable, but the Python implementation is lame. Use a string for now.
    # This is the NT variable, which can be set from the Driver's station
    nt_active_camera = ntproperty('/vision/active_camera', 'main', doc='Active camera')

    # Targeting info sent to RoboRio
    # Send the results as one big array in order to guarantee that the results
    #  all arrive at the RoboRio at the same time
    # Value is (Found, tvec, rvec) as a flat array. All values are floating point (required by NT).
    target_info = ntproperty('/vision/target_info', 7 * [0.0, ], doc='Packed array of target info: found, tvec, rvec')

    def __init__(self, calib_file):
        # for processing stored files and no camera
        self.file_mode = False
        self.camera_device = 0

        self.camera_server = cscore.CameraServer.getInstance()
        self.camera_server.enableLogging()

        self.camera_feeds = {}
        self.current_camera = None
        # active camera name. To be compared to nt_active_camera to see if it has changed
        self.active_camera = None
        self.add_cameras()

        self.create_output_stream()

        self.peg_processor = PegTarget2017(calib_file)
        # TODO: set all the parameters from NT

        # rate limit parameters
        self.previous_output_time = time.time()
        self.camera_frame = None
        self.output_frame = None

        # if asked, save image every 1/2 second
        # images are saved under the directory 'saved_images' in the current directory
        #  (ie current directory when the server is started)
        self.image_writer = ImageWriter(location_root='./saved_images',
                                        capture_period=0.5, image_format='jpg')

        return

    # --------------------------------------------------------------------------------
    # Methods generally customized each year

    def update_parameters(self, table, key, value, isNew):
        '''Update processing parameters from NetworkTables values.
        Only do this on startup or if "tuning" is on, for efficiency
        - Only use if the ntproperty values are individual variables rather than links'''
        # Make sure to add any additional created properties which should be changeable down below in addition to above

        print("valueChanged: key: '%s'; value: %s; newValue: %s" % (key, value, isNew))

        if self.tuning:
            self.output_fps_limit = table.getInteger('/vision/output_fps_limit')
            self.camera_fps = table.getInteger('/vision/fps')

            self.tuning = table.getBoolean('/vision/tuning')
            self.restart = table.getBoolean('/vision/restart')

            self.image_width = table.getInteger('/vision/width')
            self.image_height = table.getInteger('/vision/height')
            self.image_writer_state = table.getBoolean('/vision/write_images')

            self.hue_low_limit = table.getInteger('/vision/hue_low_limit')
            self.hue_high_limit = table.getInteger('/vision/hue_high_limit')
            self.saturation_low_limit = table.getInteger('/vision/saturation_low_limit')
            self.saturation_high_limit = table.getInteger('/vision/saturation_high_limit')
            self.value_low_limit = table.getInteger('/vision/value_low_limit')
            self.value_high_limit = table.getInteger('/vision/value_high_limit')
        return

    def connectionListener(connected, info):
        print(info, '; Connected=%s' % connected)

    def add_cameras(self):
        '''add a single camera at /dev/videoN, N=camera_device'''

        self.add_camera('main', self.camera_device, True)

        # you can load a camera by ID (or path). This will help with getting the correct camera
        #  for front and back, especially if they are the same model.

        # Paul's old Logitech
        # self.add_camera('front', '/dev/v4l/by-id/usb-046d_0809_69B5F86C-video-index0', False)

        # The Spinel camera board
        # self.add_camera('rear', '/dev/v4l/by-id/usb-HD_Camera_Manufacturer_USB_2.0_Camera-video-index0', True)

        return

    def preallocate_arrays(self):
        '''Preallocate the intermediate result image arrays'''

        # NOTE: shape is (height, width, #bytes)
        self.camera_frame = numpy.zeros(shape=(self.image_height, self.image_width, 3),
                                        dtype=numpy.uint8)
        self.output_frame = self.camera_frame
        return

    def process_image(self):
        '''Run the processor on the image to find the target'''

        # rvec, tvec return as None if no target found
        rvec, tvec = self.peg_processor.process_image(self.camera_frame)

        # Send the results as one big array in order to guarantee that the results
        #  all arrive at the RoboRio at the same time
        # Value is (Found, tvec, rvec) as a flat array. All values are floating point (required by NT).

        if rvec is None or tvec is None:
            res = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        else:
            res = [1.0, ]       # Found
            res.extend(tvec)
            res.extend(rvec)
        self.target_info = res

        # Try to force an update of NT to the RoboRio. Docs say this may be rate-limited,
        #  so it might not happen every call.
        NetworkTables.flush()

        return

    def prepare_output_image(self):
        '''Prepare an image to send to the drivers station'''

        self.peg_processor.prepare_output_image(self.output_frame)
        return

    # ----------------------------------------------------------------------------
    # Methods which hopefully don't need to be updated

    def create_output_stream(self):
        '''Create the main image MJPEG server'''

        # Output server
        # Need to do this the hard way to set the TCP port
        self.output_stream = cscore.CvSource('camera', cscore.VideoMode.PixelFormat.kMJPEG,
                                             self.image_width, self.image_height,
                                             min(self.camera_fps, self.output_fps_limit))
        self.camera_server.addCamera(self.output_stream)
        server = self.camera_server.addServer(name='camera',
                                              port=self.output_port)
        server.setSource(self.output_stream)

        return

    def add_camera(self, name, device, active=True):
        '''Add a single camera and set it to active/disabled as indicated.
        Cameras are referenced by their name, so pick something unique'''

        camera = cscore.UsbCamera(name, device)
        self.camera_server.startAutomaticCapture(camera=camera)
        camera.setResolution(self.image_width, self.image_height)
        camera.setFPS(self.camera_fps)

        sink = self.camera_server.getVideo(camera=camera)
        self.camera_feeds[name] = sink
        if active:
            self.current_camera = sink
            self.active_camera = name
            self.nt_active_camera = name
        else:
            # if not active, disable it to save CPU
            sink.setEnabled(False)

        return

    def switch_camera(self, name):
        '''Switch the active camera, and disable the previously active one'''

        new_cam = self.camera_feeds.get(name, None)
        if new_cam is not None:
            # disable the old camera feed
            self.current_camera.setEnabled(False)
            # enable the new one. Do this 2nd in case it is the same as the old one.
            new_cam.setEnabled(True)
            self.current_camera = new_cam
            self.active_camera = name
        else:
            logging.warning('Unknown camera %s' % name)
            self.nt_active_camera = self.active_camera

        return

    def run(self):
        '''Main loop. Read camera, process the image, send to the MJPEG server'''

        while True:

            # Check whether DS has asked for a different camera
            ntcam = self.nt_active_camera  # temp, for efficiency
            if ntcam != self.active_camera:
                self.switch_camera(ntcam)
                # TODO: switch image processing algorithm??

            if self.camera_frame is None:
                self.preallocate_arrays()

            # Tell the CvSink to grab a frame from the camera and put it
            # in the source image.  If there is an error notify the output.
            frametime, self.camera_frame = self.current_camera.grabFrame(self.camera_frame)
            if frametime == 0:
                # Send the output the error.
                self.output_stream.notifyError(self.current_camera.getError())
                # skip the rest of the current iteration
                # TODO: do we need to indicate error to RoboRio?
                continue

            if self.image_writer_state:
                self.image_writer.setImage(self.camera_frame)

            self.process_image()

            # Done. Output the marked up image, if needed
            now = time.time()
            deltat = now - self.previous_output_time
            min_deltat = 1.0 / self.output_fps_limit
            if deltat >= min_deltat:
                self.prepare_output_image()

                self.output_stream.putFrame(self.output_frame)
                self.previous_output_time = now

        return

    def run_files(self, file_list):
        '''Run routine to loop through a set of files and process each.
        Waits a couple seconds between each, and loops forever'''

        self.file_mode = True
        file_index = 0
        while True:
            if self.camera_frame is None:
                self.preallocate_arrays()

            image_file = file_list[file_index]
            print('Processing', image_file)
            file_frame = cv2.imread(image_file)
            numpy.copyto(self.camera_frame, file_frame)

            self.process_image()

            self.prepare_output_image()

            self.output_stream.putFrame(self.output_frame)
            # probably don't want to use sleep. Want something thread-compatible
            for _ in range(4):
                time.sleep(0.5)

            file_index = (file_index + 1) % len(file_list)
        return

# -----------------------------------------------------------------------------


# syntax checkers don't like global variables, so use a simple function
def main():
    '''Main routine'''

    import argparse
    parser = argparse.ArgumentParser(description='2017 Vision Server')
    parser.add_argument('--test', action='store_true', help='Run in local test mode')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose. Turn up debug messages')
    parser.add_argument('--files', action='store_true', help='Process input files instead of camera')
    parser.add_argument('--calib', required=True, help='Calibration file for camera')
    parser.add_argument('input_files', nargs='*', help='input files')

    args = parser.parse_args()

    # To see messages from networktables, you must setup logging
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    if args.test:
        # FOR TESTING, set this box as the server
        NetworkTables.enableVerboseLogging()
        NetworkTables.initialize()
    else:
        NetworkTables.initialize(server='10.28.77.2')

    server = VisionServer2017(args.calib)

    if args.files:
        if not args.input_files:
            parser.usage()

        server.run_files(args.input_files)
    else:
        server.run()
    return


# Main routine
if __name__ == '__main__':
    main()
