#!/usr/bin/env python3

# Simple timer for a block of code
# use with a "with" statement

import time


class CodeTimer:
    timerDict = {}

    def __init__(self, name):
        self.name = name
        self.startT = None
        return

    def __enter__(self):
        self.startT = time.time()
        return

    def __exit__(self, exc_type, exc_value, traceback):
        dt = time.time() - self.startT
        entry = CodeTimer.timerDict.get(self.name, None)
        if entry is None:
            CodeTimer.timerDict[self.name] = [self.name, 1, dt]
        else:
            entry[1] += 1
            entry[2] += dt

        return

    @staticmethod
    def output_timers():
        for v in sorted(CodeTimer.timerDict.values(), key=lambda s: -s[2]):
            print("{0}: {1} frames in {2:.3f} sec: {3:.3f} ms/call, {4:.2f} calls/sec".format(v[0], v[1], v[2], 1000.0 * v[2]/float(v[1]), v[1]/v[2]))
        return

    @staticmethod
    def clear_timers():
        CodeTimer.timerDict.clear()
        return
