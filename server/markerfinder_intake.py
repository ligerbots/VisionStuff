from markerfinder import MarkerFinder2021
import markerfinder_position_solver
from genericfinder import main

class MarkerFinder2021Intake(MarkerFinder2021):
    def __init__(self, calib_matrix, dist_matrix, set_finder_mode=lambda x: print("test: set mode to", x)):
        super().__init__(calib_matrix, dist_matrix, "intake")
        self.set_finder_mode = set_finder_mode

    def process_image(self, camera_frame):
        self.add_markers_image(camera_frame)
        self.set_finder_mode("markerfinder_shooter")
        return(markerfinder_position_solver.solver.get_vision_status(self.finder_id))
if __name__ == '__main__':
    main(MarkerFinder2021Intake)
