import numpy as np
import math


def top_coords(xi):
    return {
        "position": np.array([30*xi,30*5+50]),
        "height": 21.5
    }
def left_coords(yi):
    return {
        "position": np.array([-19,30*(6-yi)]),
        "height": 21
    }
def right_coords(yi):
    return {
        "position": np.array([30+30*11,30*(6-yi)]),
        "height": 19
    }

markers = {
    139: left_coords(1.5),
    145: left_coords(2),
    156: left_coords(2.5),
    163: left_coords(3),
    183: left_coords(3.5),
    87: left_coords(4),
    88: left_coords(4.5),
    89: left_coords(5),
    97: left_coords(5.5),

    110: top_coords(1),
    105: top_coords(1.5),
    254: top_coords(2),
    251: top_coords(2.5),
    239: top_coords(3),
    234: top_coords(3.5),
    230: top_coords(4),
    229: top_coords(4.5),
    226: top_coords(5),
    222: top_coords(5.5),
    216: top_coords(6),
    215: top_coords(6.5),
    211: top_coords(7),
    86: top_coords(7.5),
    79: top_coords(8),
    76: top_coords(8.5),
    61: top_coords(9),
    59: top_coords(9.5),
    203: top_coords(10),
    201: top_coords(10.5),
    53: top_coords(11),
    48: top_coords(11.5),

    44: right_coords(.5),
    41: right_coords(1),
    39: right_coords(1.5),
    36: right_coords(2),
    20: right_coords(2.5),
    12: right_coords(3),
    3: right_coords(3.5),
    255: right_coords(4),
    199: right_coords(4.5),
    198: right_coords(5),
    194: right_coords(5.5),
}

# based on cad
# (x,y), height, angle, tilt
camera_pos = {
    "intake": {
        "position": np.array([0,-8]),
        "height": 20,
        "angle": math.radians(180.0),
        "tilt": math.radians(-15.0)
    },
    "shooter": {
        "position": np.array([7.5,0]),
        "height": 24,
        "angle": math.radians(0.0),
        "tilt": math.radians(30.0)
    }
}

for camera_name in camera_pos:
    pos_info = camera_pos[camera_name]
    rot_mat = np.array([
        [ np.cos(pos_info["angle"]), -np.sin(pos_info["angle"]), 0.],
        [ np.sin(pos_info["angle"]),  np.cos(pos_info["angle"]), 0.],
        [                         0,                          0, 1.],
    ])
    trans_mat = np.array([
        [                       1,                       0, 0.],
        [                       0,                       1, 0.],
        [ pos_info["position"][0], pos_info["position"][1], 1.],
    ])
    pos_info["transform"] = rot_mat @ trans_mat
    pos_info["transform_inv"] = np.linalg.inv(pos_info["transform"])
