#!/usr/bin/python3

# Wrap the CSCore camera
# Allows for easier handling of different models

import cscore
import logging
from numpy import rot90
import os.path
from threading import Thread
from time import time, sleep
from camerautil import load_calibration_file


class Camera:
    '''Wrapper for camera related functionality.
    Makes handling different camera models easier
    Includes a threaded reader, so you can grab a frame without waiting, if needed'''

    def __init__(self, camera_server, name, device, height=240, fps=30, width=320, rotation=0,
                 threaded=False, simulation_image = None):
        '''Create a USB camera and configure it.
        Note: rotation is an angle: 0, 90, 180, -90'''

        self.width = int(width)
        self.height = int(height)
        self.rot90_count = (rotation // 90) % 4  # integer division

        if simulation_image is None:
            self.simulation_image = None

            self.camera = cscore.UsbCamera(name, device)
            camera_server.startAutomaticCapture(camera=self.camera)
            # keep the camera open for faster switching
            self.camera.setConnectionStrategy(cscore.VideoSource.ConnectionStrategy.kKeepOpen)

            self.camera.setResolution(self.width, self.height)
            self.camera.setFPS(int(fps))

            # set the camera for no auto focus, focus at infinity
            # NOTE: order does matter
            self.set_property('focus_auto', 0)
            self.set_property('focus_absolute', 0)

            mode = self.camera.getVideoMode()
            logging.info("camera '%s' pixel format = %s, %dx%d, %dFPS", name,
                         mode.pixelFormat, mode.width, mode.height, mode.fps)

            # Variables for the threaded read loop
            self.sink = camera_server.getVideo(camera=self.camera)

        else:
            if self.rot90_count%2 == 0:
                assert self.width == simulation_image.shape[1]
                assert self.height == simulation_image.shape[0]
            else:
                assert self.height == simulation_image.shape[1]
                assert self.width == simulation_image.shape[0]
            self.simulation_image = simulation_image
            self.simulation_name = name

        self.calibration_matrix = None
        self.distortion_matrix = None

        self.threaded = threaded
        self.frametime = None
        self.camera_frame = None
        self.stopped = False
        self.frame_number = 0
        self.last_read = 0

        return

    def get_name(self):
        if self.simulation_image is None:
            return self.camera.getName()
        else:
            return self.simulation_name
    def set_exposure(self, value):
        '''Set the camera exposure. 0 means auto exposure'''
        if self.simulation_image is None:
            logging.info(f"Setting camera exposure to '{value}'")
            if value == 0:
                self.camera.setExposureAuto()
                # Logitech does not like having exposure_auto_priority on when the light is poor
                #  slows down the frame rate
                # camera.getProperty('exposure_auto_priority').set(1)
            else:
                self.camera.setExposureManual(int(value))
                # camera.getProperty('exposure_auto_priority').set(0)
        else:
            logging.info(f"Attempted to set camera exposure to '{value}', but using simulation image")

        return

    def set_property(self, name, value):
        '''Set a camera property, such as auto_focus'''
        if self.simulation_image is None:
            logging.info(f"Setting camera property '{name}' to '{value}'")
            try:
                try:
                    propVal = int(value)
                except ValueError:
                    self.camera.getProperty(name).setString(value)
                else:
                    self.camera.getProperty(name).set(propVal)
            except Exception as e:
                logging.warn("Unable to set property '{}': {}".format(name, e))
        else:
            logging.info(f"Attempted to set camera property '{name}' to '{value}', but using simulation image")

        return

    def start(self):
        '''Start the thread to read frames from the video stream'''

        if self.threaded:
            t = Thread(target=self.update, args=())
            t.daemon = True
            t.start()
        return

    def update(self):
        '''Threaded read loop'''

        fps_startt = time()

        while True:
            # if the thread indicator variable is set, stop the thread
            if self.stopped:
                return

            # otherwise, read the next frame from the stream
            self._read_one_frame()

            if self.frame_number % 150 == 0:
                endt = time()
                dt = endt - fps_startt
                logging.info("threadedcamera: 150 frames in {0:.3f} seconds = {1:.2f} FPS".format(dt, 150.0 / dt))
                fps_startt = endt
        return

    def next_frame(self):
        '''Wait for a new frame'''

        if self.threaded:
            while self.last_read == self.frame_number:
                sleep(0.001)
            self.last_read = self.frame_number
        else:
            self._read_one_frame()

        return self.frametime, self.camera_frame

    def get_frame(self):
        '''Return the frame most recently read, no waiting. This may be a repeat of the previous image.'''

        if not self.threaded:
            raise Exception("Called get_frame on a non-threaded reader")

        return self.frametime, self.camera_frame

    def _read_one_frame(self):
        if self.simulation_image is None:
            self.frametime, self.camera_frame = self.sink.grabFrame(self.camera_frame)
            self.frame_number += 1

            if self.rot90_count and self.frametime > 0:
                # Numpy is *much* faster than the OpenCV routine
                self.camera_frame = rot90(self.camera_frame, self.rot90_count)
        else:
            self.frametime = time() * 1e8
            self.camera_frame = self.simulation_image
        return

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True
        sleep(0.3)         # time for thread to stop (proper way???)
        return


class LogitechC930e(Camera):
    def __init__(self, camera_server, name, device, height=240, fps=30, width=None, rotation=0, threaded=False, simulation_image = None):
        if not width:
            width = 424 if height == 240 else 848

        super().__init__(camera_server, name, device, height=height, fps=fps, width=width, rotation=rotation, threaded=threaded, simulation_image=simulation_image)

        # Logitech does not like having exposure_auto_priority on when the light is poor
        #  slows down the frame rate
        self.set_property('exposure_auto_priority', 0)

        return

    def load_calibration(self, calibration_dir):
        '''Find the correct calibration file and load it'''

        filename = f'c930e_{self.width}x{self.height}_calib.json'
        fullname = os.path.join(calibration_dir, filename)
        self.calibration_matrix, self.distortion_matrix = load_calibration_file(fullname, self.rot90_count*90)
        return


class PSEye(Camera):
    def __init__(self, camera_server, name, device, height=240, fps=30, width=None, rotation=0, threaded=False, simulation_image = None):
        if not width:
            width = 320 if height == 240 else 640

        super().__init__(camera_server, name, device, height=height, fps=fps, width=width, rotation=rotation, threaded=threaded, simulation_image=simulation_image)

        # PS Eye camera needs to have its pixelformat set
        # Not tested yet. Does this need to happen earlier?
        self.camera.setPixelFormat(cscore.VideoMode.PixelFormat.kYUYV)

        return

    def load_calibration(self, calibration_dir):
        '''Find the correct calibration file and load it'''

        # nothing yet
        return
