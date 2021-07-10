#!/usr/bin/env python3

'''Defines a class for which each year's subclass vision server inherits from'''

# import sys
from time import time, sleep
import cv2
import numpy
import logging

import cscore
from wpilib import SmartDashboard, SendableChooser
from cscore.imagewriter import ImageWriter
from networktables.util import ntproperty, ChooserControl
from networktables import NetworkTables


class VisionServer:
    '''Base class for the VisionServer'''

    # NetworkTable parameters

    # this will be under /SmartDashboard, but the SendableChooser code does not allow full paths
    ACTIVE_MODE_KEY = "vision/active_mode"

    # frame rate is pretty variable, so set this a fair bit higher than what you really want.
    # using a large number for no limit
    output_fps_limit = ntproperty('/SmartDashboard/vision/output_fps_limit', 60,
                                  doc='FPS limit of frames sent to MJPEG server')

    # default "compression" on output stream. This is actually quality, so low is high compression, poor picture
    # NOTE: this should be about "30" for competition on a real field
    # Setting this high (good quality) for At Home runs in the shed
    default_compression = ntproperty('/SmartDashboard/vision/default_compression', 30,
                                     doc='Default compression of output stream')

    # fix the TCP port for the main video, so it does not change with multiple cameras
    output_port = ntproperty('/SmartDashboard/vision/output_port', 1190,
                             doc='TCP port for main image output')

    # Operation modes. Force the value on startup.
    tuning = ntproperty('/SmartDashboard/vision/tuning', False, writeDefault=True,
                        doc='Tuning mode. Reads processing parameters each time.')

    # Logitech c930 are wide-screen cameras, so 424x240 is the correct proportion
    image_width = ntproperty('/SmartDashboard/vision/width', 424, writeDefault=False, doc='Image width')
    image_height = ntproperty('/SmartDashboard/vision/height', 240, writeDefault=False, doc='Image height')
    camera_fps = ntproperty('/SmartDashboard/vision/fps', 30, writeDefault=False, doc='FPS from camera')

    image_writer_state = ntproperty('/SmartDashboard/vision/write_images', False, writeDefault=True,
                                    doc='Turn on saving of images')

    # Targeting info sent to RoboRio
    # Send the results as one big array in order to guarantee that the results
    #  all arrive at the RoboRio at the same time.
    # Value is (time, success, finder_id, distance, angle1, angle2) as a flat array.
    # All values are floating point (required by NT).
    target_info = ntproperty('/SmartDashboard/vision/target_info', 6 * [0.0, ],
                             doc='Packed array of target info: time, success, finder_id, distance, angle1, angle2')

    def __init__(self, initial_mode, test_mode=False):
        self.test_mode = test_mode
        # for processing stored files and no camera
        self.file_mode = False

        # time of each frame. Sent to the RoboRio as a heartbeat
        self.image_time = 0

        self.camera_server = cscore.CameraServer.getInstance()
        self.camera_server.enableLogging()

        self.cameras = {}
        self.active_camera = None

        self.create_output_stream()

        # Dictionary of finders. The key is the string "name" of the finder.
        self.target_finders = {}

        # Initial mode for start of match.
        # VisionServer switches to this mode after a second, to get the cameras initialized
        self.initial_mode = initial_mode
        self.nt_active_mode = self.initial_mode

        # SendableChooser creates a dropdown chooser in ShuffleBoard
        self.mode_chooser = SendableChooser()
        SmartDashboard.putData(self.ACTIVE_MODE_KEY, self.mode_chooser)
        self.mode_chooser_ctrl = ChooserControl(self.ACTIVE_MODE_KEY)

        # active mode. To be compared to the value from mode_chooser to see if it has changed
        self.active_mode = None

        self.curr_finder = None

        # rate limit parameters
        self.previous_output_time = time()
        self.camera_frame = None
        self.output_frame = None

        # Last error message (from cscore)
        self.error_msg = None

        # if asked, save image every 1/2 second
        # images are saved under the directory 'saved_images' in the current directory
        #  (ie current directory when the server is started)
        self.image_writer = ImageWriter(location_root='./saved_images',
                                        capture_period=0.5, image_format='png')

        return

    # --------------------------------------------------------------------------------
    # Methods generally customized each year

    """Methods you should/must include"""

    def update_parameters(self):
        '''Update processing parameters from NetworkTables values.
        Only do this on startup or if "tuning" is on, for efficiency'''

        # Make sure to add any additional created properties which should be changeable
        raise NotImplementedError

    # --------------------------------------------------------------------------------
    # Methods which hopefully don't need to be updated

    def preallocate_arrays(self):
        '''Preallocate the intermediate result image arrays'''

        # NOTE: shape is (height, width, #bytes)
        # self.camera_frame = numpy.zeros(shape=(int(self.image_height), int(self.image_width), 3),
        #                                 dtype=numpy.uint8)

        return

    def create_output_stream(self):
        '''Create the main image MJPEG server'''

        # Output server
        # Need to do this the hard way to set the TCP port
        self.output_stream = cscore.CvSource('camera', cscore.VideoMode.PixelFormat.kMJPEG,
                                             int(self.image_width), int(self.image_height),
                                             int(min(self.camera_fps, self.output_fps_limit)))
        self.camera_server.addCamera(self.output_stream)
        server = self.camera_server.addServer(name='camera', port=int(self.output_port))

        # set the default compression in case driver does not
        # this is actually "quality", so low means low bandwidth, poor picture
        def_comp = int(self.default_compression)
        logging.info("Set default compression to %d", def_comp)
        server.setCompression(def_comp)
        server.setDefaultCompression(def_comp)

        server.setSource(self.output_stream)

        return

    def add_camera(self, camera, active=True):
        '''Add a single camera and set it to active/disabled as indicated.
        Cameras are referenced by their name, so pick something unique'''

        self.cameras[camera.get_name()] = camera
        camera.start()          # start read thread
        if active:
            self.active_camera = camera

        return

    def switch_camera(self, name):
        '''Switch the active camera, and disable the previously active one'''

        new_camera = self.cameras.get(name, None)
        if new_camera is not None:
            self.active_camera = new_camera
        else:
            logging.error('Unknown camera %s' % name)

        return

    def add_target_finder(self, finder):
        n = finder.name
        logging.info("Adding target finder '{}' id {}".format(n, finder.finder_id))
        self.target_finders[n] = finder
        NetworkTables.getEntry('/SmartDashboard/' + self.ACTIVE_MODE_KEY + '/options').setStringArray(list(self.target_finders.keys()))

        if n == self.initial_mode:
            NetworkTables.getEntry('/SmartDashboard/' + self.ACTIVE_MODE_KEY + '/default').setString(n)
            self.mode_chooser_ctrl.setSelected(n)
        return

    def switch_mode(self, new_mode):
        '''Switch processing mode. new_mode is the string name'''

        try:
            logging.info("Switching mode to '%s'" % new_mode)
            finder = self.target_finders.get(new_mode, None)
            if finder is not None:
                if self.active_camera.get_name() != finder.camera:
                    self.switch_camera(finder.camera)

                self.curr_finder = finder
                self.active_camera.set_exposure(finder.exposure)
                self.active_mode = new_mode
            else:
                logging.error("Unknown mode '%s'" % new_mode)

            self.mode_chooser_ctrl.setSelected(self.active_mode)  # make sure they are in sync
        except Exception as e:
            logging.error('Exception when switching mode: %s', e)

        return

    def process_image(self):
        # rvec, tvec return as None if no target found
        try:
            result = self.curr_finder.process_image(self.camera_frame)
        except Exception as e:
            logging.error("Exception caught in process_image(): %s", e)
            result = (0.0, 0.0, 0.0, 0.0, 0.0)

        return result

    def prepare_output_image(self):
        '''Create the image to send to the Driver station.
        Finder is expected to *copy* the input image, as needed'''

        if self.camera_frame is None:
            return

        try:
            if self.curr_finder is None:
                self.output_frame = self.camera_frame.copy()
            else:
                base_frame = self.camera_frame

                # Finders are allowed to designate a different image to stream to the DS
                cam = self.curr_finder.stream_camera
                if cam is not None:
                    rdr = self.video_readers.get(cam, None)
                    if rdr is not None:
                        _, base_frame = rdr.get_frame()  # does not wait

                self.output_frame = self.curr_finder.prepare_output_image(base_frame)

            min_dim = min(self.output_frame.shape[0:2])

            # Rescale if needed
            if min_dim > 400:
                # downscale by 2x
                self.output_frame = cv2.resize(self.output_frame, dsize=(0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)
                min_dim //= 2

            if min_dim < 400:  # test on height
                dotrad = 3
                fontscale = 0.4
                fontthick = 1
            else:
                dotrad = 5
                fontscale = 0.75
                fontthick = 2

            # If saving images, add a little red "Recording" dot in upper left
            if self.image_writer_state:
                cv2.circle(self.output_frame, (20, 20), dotrad, (0, 0, 255), thickness=2*dotrad, lineType=8, shift=0)

            # If tuning mode is on, add text to the upper left corner saying "Tuning On"
            if self.tuning:
                cv2.putText(self.output_frame, "TUNING ON", (60, 30), cv2.FONT_HERSHEY_SIMPLEX, fontscale, (255, 0, 0), thickness=fontthick)

            # If test mode (ie running the NT server), give a warning
            if self.test_mode:
                cv2.putText(self.output_frame, "TEST MODE", (5, self.output_frame.shape[0]-5), cv2.FONT_HERSHEY_SIMPLEX,
                            fontscale, (0, 255, 255), thickness=fontthick)

        except Exception as e:
            logging.error("Exception caught in prepare_output_image(): %s", e)

        return

    def run(self):
        '''Main loop. Read camera, process the image, send to the MJPEG server'''

        frame_num = 0
        errors = 0

        fps_count = 0
        fps_startt = time()
        imgproc_nettime = 0

        while True:
            try:
                # Check whether DS has asked for a different camera
                # ntmode = self.nt_active_mode  # temp, for efficiency
                ntmode = self.mode_chooser_ctrl.getSelected()
                if ntmode != self.active_mode:
                    self.switch_mode(ntmode)

                # if self.camera_frame is None:
                #     self.preallocate_arrays()

                if self.tuning:
                    self.update_parameters()

                # Tell the CvReader to grab a frame from the camera and put it
                # in the source image.  Frametime==0 on error
                frametime, self.camera_frame = self.active_camera.next_frame()
                frame_num += 1

                imgproc_startt = time()

                if frametime == 0:
                    # ERROR!!
                    self.error_msg = self.active_camera.sink.getError()

                    if errors < 10:
                        errors += 1
                    else:   # if 10 or more iterations without any stream, switch cameras
                        logging.warning(self.active_camera.get_name() + " camera is no longer streaming. Switching cameras...")
                        self.switch_mode(self.mode_after_error())
                        errors = 0

                    target_res = [time(), ]
                    target_res.extend(5*[0.0, ])
                else:
                    self.error_msg = None
                    errors = 0

                    if self.image_writer_state:
                        self.image_writer.setImage(self.camera_frame)

                    # frametime = time() * 1e6  (ie microseconds)
                    # convert frametime to seconds to use as the heartbeat sent to the RoboRio
                    target_res = [1e-6 * frametime, ]

                    proc_result = self.process_image()
                    target_res.extend(proc_result)

                # Send the results as one big array in order to guarantee that the results
                #  all arrive at the RoboRio at the same time
                # Value is (Timestamp, Found, Mode, distance, angle1, angle2) as a flat array.
                #  All values are floating point (required by NT).
                self.target_info = target_res

                # Try to force an update of NT to the RoboRio. Docs say this may be rate-limited,
                #  so it might not happen every call.
                NetworkTables.flush()

                # Done. Output the marked up image, if needed
                # Note this rate limiting can also be done via the URL, but this is more efficient
                now = time()
                deltat = now - self.previous_output_time
                min_deltat = 1.0 / self.output_fps_limit
                if deltat >= min_deltat:
                    self.prepare_output_image()
                    # TODO: this is pretty slow. Why?
                    self.output_stream.putFrame(self.output_frame)
                    self.previous_output_time = now

                if frame_num == 30:
                    # This is a bit stupid, but you need to poke the camera *after* the first
                    #  bunch of frames has been collected.
                    self.switch_mode(self.initial_mode)

                fps_count += 1
                imgproc_nettime += time() - imgproc_startt
                if fps_count >= 150:
                    endt = time()
                    dt = endt - fps_startt
                    logging.info("{0} frames in {1:.3f} seconds = {2:.2f} FPS".format(fps_count, dt, fps_count/dt))
                    logging.info("Image processing time = {0:.2f} msec/frame".format(1000.0 * imgproc_nettime / fps_count))
                    fps_count = 0
                    fps_startt = endt
                    imgproc_nettime = 0

            except Exception as e:
                # major exception. Try to keep going
                logging.error('Caught general exception: %s', e)

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
            # for _ in range(4):
            sleep(0.5)

            file_index = (file_index + 1) % len(file_list)
        return

# -----------------------------------------------------------------------------


def wait_on_nt_connect(max_delay=10):
    cnt = 0
    while True:
        if NetworkTables.isConnected():
            logging.info('Connect to NetworkTables after %d seconds', cnt)
            return

        if cnt >= max_delay:
            break

        if cnt > 0 and cnt % 5 == 0:
            logging.warning("Still waiting to connect to NT (%d sec)", cnt)
        sleep(1)
        cnt += 1

    logging.warning("Failed to connect to NetworkTables after %d seconds. Continuing", cnt)
    return


# syntax checkers don't like global variables, so use a simple function
def main(server_type):
    '''Main routine'''

    import argparse
    import signal

    parser = argparse.ArgumentParser(description='LigerBots Vision Server')
    parser.add_argument('--calib_dir', required=True, help='Directory for calibration files')
    parser.add_argument('--test', action='store_true', help='Run in local test mode')
    parser.add_argument('--delay', type=int, default=0, help='Max delay trying to connect to NT server (seconds)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose. Turn up debug messages')
    parser.add_argument('--files', action='store_true', help='Process input files instead of camera')
    parser.add_argument('input_files', nargs='*', help='input files')

    args = parser.parse_args()

    # To see messages from networktables, you must setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format='%(asctime)s %(levelname)s: %(message)s')

    logging.info("cscore version '%s'" % cscore.__version__)
    logging.info("OpenCV version '%s'" % cv2.__version__)

    if args.test:
        # FOR TESTING, set this box as the server
        # NetworkTables.enableVerboseLogging()
        NetworkTables.startServer()
    else:
        if args.verbose:
            # Turn up the noise from NetworkTables. VERY noisy!
            # DO NOT do this during competition, unless you are really sure
            NetworkTables.enableVerboseLogging()
        # NetworkTables.startClient('10.28.77.2')
        # Try startClientTeam() method; it auto tries a whole bunch of standard addresses
        NetworkTables.startClientTeam(2877)
        if args.delay > 0:
            wait_on_nt_connect(args.delay)

    # Initialization sometimes locks up. Set an alarm timer and bomb if it runs out
    # Don't set a handler. It will crash and systemd will restart it automatically.
    logging.info('Setting alarm for 5 seconds')
    signal.alarm(5)             # 5 seconds

    server = server_type(calib_dir=args.calib_dir, test_mode=args.test)

    # We got through. Cancel the alarm
    signal.alarm(0)
    logging.info('Alarm canceled')

    if args.files:
        if not args.input_files:
            parser.usage()

        server.run_files(args.input_files)
    else:
        server.run()
    return
