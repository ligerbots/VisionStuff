import numpy as np
import math

import layout
import cv2



# nx2, nx2
def find_rigid_transform(x, y):
    # https://scicomp.stackexchange.com/a/6901
    assert len(x) == len(y)
    x_center = np.mean(x, axis=0)
    y_center = np.mean(y, axis=0)
    A = x - x_center
    B = y - y_center
    C = B.T @ A
    U, S_diag, V_T = np.linalg.svd(C)
    R = U @ V_T
    if(np.linalg.det(R) < 0):
        R = U @ np.diag([1, -1]) @ V_T
    d = y_center - R @ x_center
    return R, d


def find_rigid_transform_ransac(x, y, cutoff_error, iters=20, sample_size=3):
    assert len(x) >= sample_size

    best = np.zeros((x.shape[0],), dtype=bool)
    best_count = 0
    for i in range(iters):
        sample_indexes = np.random.choice(
            x.shape[0], sample_size, replace=False)
        sample_x = x[sample_indexes]
        sample_y = y[sample_indexes]
        R, d = find_rigid_transform(sample_x, sample_y)
        transformed_x = x @ R.T + d

        error = np.linalg.norm(transformed_x - y, axis=1)
        success = error < cutoff_error

        count = np.count_nonzero(success)
        if(count > best_count):
            best = success
            best_count = count

    R, d = find_rigid_transform(x[best], y[best])
    return R, d

class PositionSolver:
    def __init__(self):
        self.prev_solution_pts = []
        self.prev_solution = np.array([0,0,0],dtype=np.float64)
        self.solve_successful = False
        self.marker_pts = []
        self.marker_ids = []
        self.absolute_pts = []
    def add_marker(self, relative_to_robot_position, marker_id):
        self.marker_pts.append(relative_to_robot_position)
        self.marker_ids.append(marker_id)
        self.absolute_pts.append(layout.markers[marker_id]["position"])
    def solve(self):
        if(len(self.marker_pts)>=3):

            relative_to_robot_pts = np.array(self.marker_pts)
            absolute_pts = np.array(self.absolute_pts)
            R, d = find_rigid_transform_ransac(relative_to_robot_pts, absolute_pts, 20)

            self.prev_solution_pts = [{
                "id": self.marker_ids[i],
                "pos": R @ self.marker_pts[i] + d,
                "realpos": self.absolute_pts[i]
            } for i in range(len(self.marker_pts))]

            self.prev_solution[:2] = d

            dir_vec = R @ np.array([0, 1])
            dir = math.atan2(dir_vec[1],dir_vec[0])
            self.prev_solution[2]=dir

            self.solve_successful=True
        else:
            self.solve_successful=False
    def draw(self,draw_frame):
        box_topright = np.array([10,10+180], dtype=np.float64)
        box_bottomleft = np.array([10+360,10], dtype=np.float64)

        def to_imagespace(pt):
            x=np.interp([pt[0]], [0, 360], [box_topright[0], box_bottomleft[0]])[0]
            y=np.interp([pt[1]], [0, 180], [box_topright[1], box_bottomleft[1]])[0]
            print(pt,[x,y])

            return((int(x),int(y)))
        cv2.rectangle(draw_frame, tuple(box_topright.astype(np.int)), tuple(box_bottomleft.astype(np.int)), (0,255,0), 2)
        for sol_pt in self.prev_solution_pts:
            pos_image = to_imagespace(sol_pt["pos"])
            realpos_image = to_imagespace(sol_pt["realpos"])

            cv2.circle(draw_frame, realpos_image, 2, (0, 0, 255), -1)

            cv2.putText(draw_frame, str(sol_pt["id"]),
                        realpos_image,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        .4,
                        (0, 0, 255),
                        1)
            cv2.circle(draw_frame, pos_image, 2, (255, 0, 0), -1)

            cv2.putText(draw_frame, str(sol_pt["id"]),
                        pos_image,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        .4,
                        (255, 0, 0),
                        1)
        robotpos_image = to_imagespace(self.prev_solution[:2])
        cv2.circle(draw_frame, robotpos_image, 5, (0, 255, 255), -1)
        cv2.arrowedLine(draw_frame, robotpos_image,
                        (int(robotpos_image[0]+math.cos(self.prev_solution[2])*20), int(robotpos_image[1]+math.sin(self.prev_solution[2])*20)),
                        (255,255,0), 2)

    def get_vision_status(self, finder_id):
        return (1. if self.solve_successful else 0., self.prev_solution[0], self.prev_solution[1], self.prev_solution[2], 0.0, 0.0, 0.0)

solver = PositionSolver()
