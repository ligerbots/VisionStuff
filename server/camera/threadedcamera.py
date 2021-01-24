#!/usr/bin/env python3

# import the necessary packages
from threading import Thread
import time
import logging


class ThreadedCamera:
    '''Threaded camera reader. For now, a thin wrapper around the CSCore classes.'''

    def __init__(self, sink):
        '''Remember the source and set up the reading loop'''

        self.sink = sink
        self.timer = None

        self.frametime = None
        self.camera_frame = None

        # initialize the variable used to indicate if the thread should
        # be stopped
        self.stopped = False
        self.frame_number = 0
        self.last_read = 0
        return

    def start(self):
        '''Start the thread to read frames from the video stream'''

        t = Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        '''Threaded read loop'''

        fps_startt = time.time()

        while True:
            # if the thread indicator variable is set, stop the thread
            if self.stopped:
                return

            # otherwise, read the next frame from the stream
            self.frametime, self.camera_frame = self.sink.grabFrame(self.camera_frame)
            self.frame_number += 1

            if self.frame_number % 150 == 0:
                endt = time.time()
                dt = endt - fps_startt
                logging.info("threadedcamera: 150 frames in {0:.3f} seconds = {1:.2f} FPS".format(dt, 150.0 / dt))
                fps_startt = endt
        return

    def next_frame(self):
        '''Wait for a new frame'''

        while self.last_read == self.frame_number:
            time.sleep(0.001)
        self.last_read = self.frame_number
        return self.frametime, self.camera_frame

    def get_frame(self):
        '''Return the frame most recently read, no waiting. This may be a repeat of the previous image.'''

        return self.frametime, self.camera_frame

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True
        self.timer.output()
        time.sleep(0.3)         # time for thread to stop (proper way???)
        return
