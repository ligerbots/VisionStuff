import json
import numpy as np
import math
import cv2
import random
import pyopencl as cl
import numba as nb
import time
from numba.np.extensions import cross2d


@nb.njit(nb.float32(nb.float32))
def make_nonzero(n):
    if(n == 0):
        return(np.finfo(np.float32).eps)
    return(n)


@nb.njit(nb.float32[:](nb.float32, nb.float32, nb.float32))
def get_direction_at(x, y, tilt):
    return(np.array([
        x * math.sin(tilt),
        y * math.sin(tilt) + math.cos(tilt)
    ]))


@nb.njit(nb.float32[:](nb.float32, nb.float32, nb.float32, nb.float32[:, :]))
def get_direction_at_imagespace(x, y, tilt, camera_matrix):
    x_prime = (x - camera_matrix[0, 2]) / camera_matrix[0, 0]
    y_prime = (y - camera_matrix[1, 2]) / camera_matrix[1, 1]
    slope_prime = get_direction_at(x_prime, y_prime, tilt)
    return(np.array([
        slope_prime[0] * camera_matrix[0, 0],
        slope_prime[1] * camera_matrix[1, 1]
    ]))


@nb.njit(nb.int32[:, :](nb.float32, nb.float32, nb.float32,
                        nb.float32[:, :],
                        nb.float32, nb.float32, nb.float32, nb.float32))
def get_segment_at_imagespace(x, y, tilt,
                              camera_matrix,
                              top, bottom, left, right):
    slope = get_direction_at_imagespace(x, y, tilt, camera_matrix)
    slope[0] = make_nonzero(slope[0])
    slope[1] = make_nonzero(slope[1])

    coeff_top = make_nonzero(np.float32((top - y) / slope[1]))
    coeff_left = make_nonzero(np.float32((left - x) / slope[0]))
    coeff_bottom = make_nonzero(np.float32((bottom - y) / slope[1]))
    coeff_right = make_nonzero(np.float32((right - x) / slope[0]))
    top_d = coeff_top
    if(top_d / coeff_left > 1.):
        top_d = coeff_left
    if(top_d / coeff_right > 1.):
        top_d = coeff_right

    bottom_d = coeff_bottom
    if(bottom_d / coeff_left > 1.):
        bottom_d = coeff_left
    if(bottom_d / coeff_right > 1.):
        bottom_d = coeff_right

    start = np.array([x, y])
    return(np.vstack((
        start + slope * top_d,
        start + slope * bottom_d
    )).astype(np.int32))

# generates line segments left to right (lower index = lefter line)


@nb.njit(nb.int32[:, :, :](nb.int32, nb.float32,
                           nb.float32[:, :],
                           nb.float32, nb.float32, nb.float32, nb.float32))
def gen_segments_nb(num_lines, tilt,
                    camera_matrix,
                    top, bottom, left, right):
    if(tilt > 0):
        begin_y = top
    else:
        begin_y = bottom

    lines = np.empty((num_lines, 2, 2), dtype=np.int32)
    for i in range(num_lines):
        x = left + (i + 1.) * (right - left) / (num_lines + 1)
        lines[i] = get_segment_at_imagespace(x, begin_y, tilt,
                                             camera_matrix,
                                             top, bottom, left, right)
    return lines


@nb.njit(nb.int32[:, :, :](nb.int32, nb.float32, nb.float32, nb.float32, nb.float32,
                           nb.float32[:, :],
                           nb.float32, nb.float32, nb.float32, nb.float32))
# start_x inclusive, end_x inclusive
def gen_segments_subset_nb(num_lines, tilt, start_x, end_x, center_y,
                           camera_matrix,
                           top, bottom, left, right):

    lines = np.empty((num_lines, 2, 2), dtype=np.int32)
    for i in range(num_lines):
        x = i / (num_lines - 1) * (end_x - start_x) + start_x
        lines[i] = get_segment_at_imagespace(x, center_y, tilt,
                                             camera_matrix,
                                             top, bottom, left, right)
    return lines

# finds ranges in const lines


@nb.njit(nb.types.ListType(nb.types.Array(nb.int32, 1, "C"))(nb.int32[:, :], nb.int32[:, :], nb.int32[:, :]))
def find_ranges_const_lines(const_starts_np, const_ends_np, const_dest_np):
    res = nb.typed.List()
    range_start = -1
    range_end = -1
    last_y = -1
    last_total = -1
    num_fails = 0

    i = 0
    while i < len(const_dest_np):
        # only valid if const_dest_np[const_dest_np,0]>=0
        this_y = const_starts_np[i, 1] + const_dest_np[i, 0]
        this_total = const_dest_np[i, 1]  # same here

        if (range_start == -1):
            if(const_dest_np[i, 0] >= 0):
                num_fails = 0
                range_start = i
                range_end = i
                last_y = this_y
                last_total = this_total
        else:
            if(const_dest_np[i, 0] >= 0):
                if(np.abs(this_y - last_y) < last_total):
                    range_end = i
                    last_y = this_y
                    last_total = this_total
                    num_fails = 0
                else:
                    num_fails += 2
            else:
                num_fails += 1
            if(num_fails >= 5):
                res.append(np.array([range_start, range_end], dtype=np.int32))
                range_start = -1
                i = range_end
        i += 1
    if(range_start != -1):
        res.append(np.array([range_start, range_end], dtype=np.int32))
    return(res)


@nb.njit(nb.boolean[:](nb.float32[:, :], nb.float32, nb.int32))
def ransac(points, cutoff_d, iters):
    best = np.zeros((points.shape[0],), dtype=nb.boolean)
    best_count = 0
    for i in range(iters):
        indexes = np.random.choice(points.shape[0], 2, replace=False)
        p1, p2 = points[indexes]

        diff = p2 - p1
        norm = np.linalg.norm(diff)

        distances = np.abs(cross2d(p2 - p1, p1 - points)) / \
            np.linalg.norm(p2 - p1)

        inliers = distances < cutoff_d
        count = np.count_nonzero(inliers)
        if(count > best_count):
            best = inliers
            best_count = count
    return(best)


@nb.njit(nb.int32[:](nb.int32[:], nb.int32[:], nb.int32))
def calculate_found_pos(start, end, indx):
    return(np.array([
        start[0] + (end[0] - start[0]) * indx // (end[1] - start[1]),
        start[1] + indx
    ], dtype=np.int32))


@nb.njit(nb.int32[:, :](nb.int32[:, :, :], nb.int32[:, :]))
def calculate_poses(segments, indxes):
    res = np.empty((segments.shape[0], 2), dtype=np.int32)
    for i in range(segments.shape[0]):
        res[i] = calculate_found_pos(
            segments[i, 0], segments[i, 1], indxes[i, 0])

    return(res)


class MarkerFinder:
    OCL_SOURCE = """
    __kernel void processline(read_only image2d_t image, __global const int2 *starts, __global const int2 *ends, __global int2 *dest){
        int2 image_dim=get_image_dim(image);
        const sampler_t sampler =  CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;


        int gid=get_global_id(0);
        int swap_i=0;
        int swap_t[3]={-1000,-1000,-1000};
        int prev_swap=-1000;

        dest[gid]=(int2)(-1,-1);
        int2 start=starts[gid];
        int2 offset=ends[gid]-start;

        uint4 prev_col_even=read_imageui(image, sampler, (int2)(start.x,start.y));

        uint4 prev_col_odd=read_imageui(image, sampler, (int2)(start.x+offset.x/offset.y,start.y+1));
        for(int i=2;i<=offset.y;i++){
            uint4 this_col = read_imageui(image, sampler, (int2)(start.x+offset.x*i/offset.y,start.y+i));
            uint4 prev_col;
            if(i%2==0){
                prev_col=prev_col_even;
                prev_col_even=this_col;
            }else{
                prev_col=prev_col_odd;
                prev_col_odd=this_col;
            }

            if(
                //(this_col.x>prev_col.x+REQUIRED_DIFF && this_col.y>prev_col.y+REQUIRED_DIFF && this_col.z>prev_col.z+REQUIRED_DIFF)||
                //(this_col.x+REQUIRED_DIFF<prev_col.x && this_col.y+REQUIRED_DIFF<prev_col.y && this_col.z+REQUIRED_DIFF<prev_col.z)
                distance(convert_float4(this_col),convert_float4(prev_col)) > REQUIRED_DIFF
            ){
                swap_t[swap_i%3]=i-prev_swap;
                swap_i++;
                prev_swap=i;

                int total=swap_t[0]+swap_t[1]+swap_t[2];
                int max_swap_t=max(max(swap_t[0],swap_t[1]),swap_t[2]);
                int min_swap_t=min(min(swap_t[0],swap_t[1]),swap_t[2]);
                if(max_swap_t-min_swap_t<total/3*MAX_TOLERANCE+3 && total>6){
                    dest[gid].x=i-1;
                    dest[gid].y=total;
                    break;
                }
                i++;

                uint4 this_col = read_imageui(image, sampler, (int2)(start.x+offset.x*i/offset.y,start.y+i));
                if(i%2==0){
                    prev_col_even=this_col;
                }else{
                    prev_col_odd=this_col;
                }
            }
        }
    }
    """.replace("REQUIRED_DIFF", "150").replace("MAX_TOLERANCE", ".1")
    MAX_DYN_LINES = 2000

    def __init__(self, tilt, image_width, image_height, camera_matrix, const_scan_stride=5, ):
        self.tilt = tilt
        self.image_width = image_width
        self.image_height = image_height
        self.camera_matrix = camera_matrix.astype(np.float32)
        self.const_scan_stride = const_scan_stride

        self.gen_const_arrays()
        self.init_ocl()

    def gen_segments(self, num_lines,
                     top=0, bottom=None, left=0, right=None):
        if bottom is None:
            bottom = self.image_height
        if right is None:
            right = self.image_width
        return(gen_segments_nb(num_lines, self.tilt,
                               self.camera_matrix,
                               top, bottom, left, right))

    def gen_segments_subset(self, num_lines, start_x, end_x, center_y,
                            top=0, bottom=None, left=0, right=None):
        if bottom is None:
            bottom = self.image_height
        if right is None:
            right = self.image_width
        return(gen_segments_subset_nb(num_lines, self.tilt, start_x, end_x, center_y,
                                      self.camera_matrix,
                                      top, bottom, left, right))

    def gen_const_arrays(self):
        self.const_line_count = self.image_width // self.const_scan_stride

        const_lines = self.gen_segments(self.const_line_count)

        self.const_starts_np = np.ascontiguousarray(
            const_lines[:, 0, :], dtype=np.int32)
        self.const_ends_np = np.ascontiguousarray(
            const_lines[:, 1, :], dtype=np.int32)

        self.const_dest_np = np.zeros(
            (self.const_line_count, 2), dtype=np.int32)

        self.dyn_dest_np = np.zeros(
            (MarkerFinder.MAX_DYN_LINES, 2), dtype=np.int32)

    def init_ocl(self):
        self.ctx = cl.create_some_context()
        self.queue = cl.CommandQueue(self.ctx)
        self.prg = cl.Program(self.ctx, MarkerFinder.OCL_SOURCE).build()
        self.processline = self.prg.processline

        fmt = cl.ImageFormat(cl.channel_order.RGBA,
                             cl.channel_type.UNSIGNED_INT8)
        self.image_buf = cl.Image(self.ctx, cl.mem_flags.READ_ONLY, fmt, shape=(
            self.image_width, self.image_height))

        self.const_starts_buf = cl.Buffer(
            self.ctx, cl.mem_flags.READ_ONLY, self.const_starts_np.nbytes)
        self.const_ends_buf = cl.Buffer(
            self.ctx, cl.mem_flags.READ_ONLY, self.const_ends_np.nbytes)

        cl.enqueue_copy(self.queue, self.const_starts_buf,
                        self.const_starts_np).wait()
        cl.enqueue_copy(self.queue, self.const_ends_buf,
                        self.const_ends_np).wait()

        self.const_dest_buf = cl.Buffer(
            self.ctx, cl.mem_flags.WRITE_ONLY, self.const_dest_np.nbytes)

        assert self.dyn_dest_np.nbytes == 4 * 2 * \
            MarkerFinder.MAX_DYN_LINES  # make sure math is right

        self.dyn_starts_buf = cl.Buffer(
            self.ctx, cl.mem_flags.READ_ONLY, 4 * 2 * MarkerFinder.MAX_DYN_LINES)
        self.dyn_ends_buf = cl.Buffer(
            self.ctx, cl.mem_flags.READ_ONLY, 4 * 2 * MarkerFinder.MAX_DYN_LINES)

        self.dyn_dest_buf = cl.Buffer(
            self.ctx, cl.mem_flags.WRITE_ONLY, self.dyn_dest_np.nbytes)

    def calculate_const_found_pos(self, i):
        start = self.const_starts_np[i]
        end = self.const_ends_np[i]
        indx = self.const_dest_np[i][0]

        return(calculate_found_pos(start, end, indx))

    def gen_segments_from_range(self, range_start, range_end):
        left = self.calculate_const_found_pos(range_start)
        right = self.calculate_const_found_pos(range_end)

        center = (left + right) / 2
        height = max(self.const_dest_np[range_start, 1],
                     self.const_dest_np[range_end, 1]) * 2
        width = right[0] - left[0]
        width += 2 * self.const_scan_stride
        width *= 1.5
        height *= 1.5
        return(
            self.gen_segments_subset(
                50,
                start_x=center[0] - width * .5, end_x=center[0] + width * .5, center_y=center[1],
                top=max(0, center[1] - height * .5),
                bottom=min(self.image_height - 1, center[1] + height * .5)
            )
        )

    def process_image(self, image, debug=False):

        assert image.shape[1] == self.image_width
        assert image.shape[0] == self.image_height

        if(debug):
            self.debug_timer()
        # copy image to ocl device
        self.image_np = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)
        cl.enqueue_copy(self.queue, self.image_buf, self.image_np, origin=(
            0, 0), region=(self.image_width, self.image_height)).wait()
        if(debug):
            self.debug_timer("load image")
        # find trackers via const lines in image
        self.processline(self.queue, (self.const_line_count,), None, self.image_buf,
                         self.const_starts_buf, self.const_ends_buf, self.const_dest_buf).wait()

        # copy results back to host
        cl.enqueue_copy(self.queue, self.const_dest_np,
                        self.const_dest_buf).wait()
        if(debug):
            self.debug_timer("find const lines")

        ranges = list(find_ranges_const_lines(
            self.const_starts_np, self.const_ends_np, self.const_dest_np))

        if(debug):
            self.debug_dyn_segments = []
            self.debug_dyn_points = []
            self.debug_dyn_bounds = []
            self.debug_point_val = []

        for (range_start, range_end) in ranges:
            dyn_segments = self.gen_segments_from_range(range_start, range_end)
            if(debug):
                self.debug_timer("gen dyn lines")

            if(debug):
                self.debug_dyn_segments.append(dyn_segments)
            cl.enqueue_copy(self.queue, self.dyn_starts_buf,
                            np.ascontiguousarray(dyn_segments[:, 0, :])).wait()
            cl.enqueue_copy(self.queue, self.dyn_ends_buf,
                            np.ascontiguousarray(dyn_segments[:, 1, :])).wait()
            if(debug):
                self.debug_timer("load dyn lines")

            self.processline(self.queue, (len(dyn_segments),), None, self.image_buf,
                             self.dyn_starts_buf, self.dyn_ends_buf, self.dyn_dest_buf).wait()
            cl.enqueue_copy(self.queue, self.dyn_dest_np,
                            self.dyn_dest_buf).wait()
            if(debug):
                self.debug_timer("find dyn lines")

            dyn_dest_np_trunc = self.dyn_dest_np[:len(dyn_segments)]
            filtered_segments = dyn_segments[dyn_dest_np_trunc[:, 0] >= 0]
            filtered_dest = dyn_dest_np_trunc[dyn_dest_np_trunc[:, 0] >= 0]
            if(len(filtered_segments)<2):
                continue

            poses = calculate_poses(filtered_segments, filtered_dest)
            if(debug):
                self.debug_dyn_points.extend(poses)

            if(debug):
                self.debug_timer("calc dyn poses")

            bound = self.find_marker_bounds(
                filtered_segments, poses, filtered_dest)

            if(debug):
                self.debug_timer("find marker bounds")

            if(bound is None):
                continue

            if(debug):
                self.debug_dyn_bounds.append(bound)
            res = self.get_value_and_position(bound)

            if(debug):
                self.debug_timer("calc value & center")

            if(res is None):
                continue

            if(debug):
                self.debug_point_val.append(res)

    # assumes that filtered_segments is from left to right

    def find_marker_bounds(self, filtered_segments, poses, filtered_dest):
        inliers = ransac(poses.astype(np.float32), 5., 50)

        inlier_poses = poses[inliers]
        inlier_totals = filtered_dest[inliers, 1]

        if(len(inlier_poses) < 3):
            return(None)

        point_m, point_b = np.polyfit(
            inlier_poses[:, 0], inlier_poses[:, 1], 1)

        # relates x-pos to y offset
        total_m, total_b = np.polyfit(inlier_poses[:, 0], inlier_totals, 1)

        def get_pos(x, total_scale):
            y = x * point_m + point_b
            dir = get_direction_at_imagespace(
                x, y, self.tilt, self.camera_matrix)

            total = (x * total_m + total_b) * total_scale

            return(np.array([x, y]) + dir * (total / dir[1]))

        left = inlier_poses[0]
        right = inlier_poses[-1]

        return(np.array([
            get_pos(left[0], 1),  # top left
            get_pos(right[0], 1),  # top right
            get_pos(left[0], -1),  # bottom left
            get_pos(right[0], -1),  # bottom right
        ]))

    # 0<=x<4
    # 0<=y<6
    def get_point_in_grid(self, bound, xi, yi):
        top_left, top_right, bottom_left, bottom_right = bound

        bord_hor = .5
        bord_vert = .5

        right = (xi + bord_hor) / (3 + 2 * bord_hor)
        left = 1 - right

        bottom = (yi + bord_vert) / (5 + 2 * bord_vert)
        top = 1 - bottom
        return(
            top_left * (top * left) +
            top_right * (top * right) +
            bottom_left * (bottom * left) +
            bottom_right * (bottom * right)
        )

    def get_value_and_position(self, bound):
        colors = np.zeros((6, 4, 3), dtype=np.uint8)
        for yi in range(6):
            for xi in range(4):
                x, y = self.get_point_in_grid(bound, xi, yi).astype(np.int32)
                if(x < 0 or y < 0 or x >= self.image_width or y >= self.image_height):
                    return None
                colors[yi, xi] = self.image_np[y, x, :3]
        should_be_white = np.concatenate([colors[2, :], colors[4, :]])
        should_be_black = np.concatenate([colors[3, :], colors[5, :]])

        data_color = np.concatenate([colors[1, :], colors[0, :]])

        white_min = np.min(should_be_white, axis=0)
        black_max = np.max(should_be_black, axis=0)

        if(np.all(white_min > black_max)):
            white_avg = np.mean(should_be_white, axis=0)
            black_avg = np.mean(should_be_black, axis=0)
            data_distance_to_white = np.linalg.norm(
                data_color - white_avg, axis=1)
            data_distance_to_black = np.linalg.norm(
                data_color - black_avg, axis=1)

            data = data_distance_to_white > data_distance_to_black  # which colors are black?
            data_value = np.packbits(data)[0]
            center = np.mean(bound, axis=0)
            return((center, data_value))
        else:
            return None

    def debug_timer(self, name=None):
        if name is None:
            self.debug_times = {}
            self.debug_last_time = time.time()
        else:
            if(name in self.debug_times):
                self.debug_times[name] += time.time() - self.debug_last_time
            else:
                self.debug_times[name] = time.time() - self.debug_last_time
            self.debug_last_time = time.time()

    def print_debug_timer(self):
        total = 0
        for name in self.debug_times:
            total += self.debug_times[name]
            print(name + ": " + str(self.debug_times[name] * 1000) + "ms")
        print("Total time: " + str(total * 1000) + "ms")

    def process_image_debug(self, image, plt):
        self.process_image(image, debug=True)
        self.print_debug_timer()

        plt.figure()
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        for i in range(0, self.const_line_count, 1):
            p1 = self.const_starts_np[i]
            p2 = self.const_ends_np[i]
            plt.arrow(*p1,*(p2-p1))

        const_found_points = []
        for i in range(self.const_line_count):
            if(self.const_dest_np[i][0] >= 0):
                const_found_points.append(self.calculate_const_found_pos(i))

        const_found_points_np = np.array(const_found_points).T

        if(len(const_found_points_np)):
            plt.scatter(
                const_found_points_np[0], const_found_points_np[1], c='green', s=3, linewidths=0, marker='o')

        cmap = plt.cm.get_cmap("hsv", len(self.debug_dyn_segments))

        for i, segments in enumerate(self.debug_dyn_segments):
            color = cmap(i)
            for i, line in enumerate(segments):
                if(i % 10 == 0):
                    p1, p2 = line

                    plt.arrow(*p1,*(p2-p1),color=color)
            p1, p2 = segments[-1]
            plt.arrow(*p1,*(p2-p1),color=color)
        # print(self.debug_dyn_points)
        dyn_found_points_t = np.array(self.debug_dyn_points).T
        if(len(dyn_found_points_t)):
            plt.scatter(dyn_found_points_t[0], dyn_found_points_t[1],
                        c='purple', s=3, linewidths=0, marker='o')

        for bound in self.debug_dyn_bounds:
            top_left, top_right, bottom_left, bottom_right = bound
            plt.plot(*zip(
                top_left,
                top_right,
                bottom_right,
                bottom_left,
                top_left,
            ), color="orange", linewidth=2)

            pts = []
            for xi in range(4):
                for yi in range(6):
                    pts.append(self.get_point_in_grid(bound, xi, yi))
            plt.scatter(*zip(*pts), c='orange', s=3, linewidths=0, marker='o')

        for (center, val) in self.debug_point_val:
            plt.scatter(*center)
            plt.annotate(str(val), center, color="red")

        plt.show()



if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import glob

    with open("../data/calibration/c930e_848x480_calib.json") as f:
        json_data = json.load(f)
        camera_matrix = np.array(json_data["camera_matrix"])
        distortion_matrix = np.array(json_data["distortion"])

    camera_matrix[1, 1], camera_matrix[0, 0] = camera_matrix[0, 0], camera_matrix[1, 1]
    camera_matrix[1, 2], camera_matrix[0, 2] = camera_matrix[0, 2], camera_matrix[1, 2]
    distortion_matrix[0, 3], distortion_matrix[0, 2] = distortion_matrix[0, 2], distortion_matrix[0, 3]


    image = cv2.imread("../data/test_images/2021/marker-finder/shed/2021-02-17/intake/1613603018.70.png")
    tilt = math.radians(-15)
    image_height, image_width, _ = image.shape

    #tilt, image_width, image_height, camera_matrix
    marker_finder = MarkerFinder(
        tilt, image_width, image_height, camera_matrix)
    marker_finder.process_image_debug(image, plt)
