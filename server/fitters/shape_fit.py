#!/usr/bin/python3

# Use standard fitting techniques to fit a mathematical shape to a contour
# Unfortunately, it is pretty slow, and does not seem to be that much better

import cv2
import numpy
from scipy.optimize import minimize
from util.codetimer import CodeTimer
from scipy.spatial.distance import cdist


class PolygonFitter():
    def __init__(self):
        self._call_count = 0
        self._cached_results = {}
        return

    def fit(self, contour, shape_function, start_fit):

        self._call_count = 0
        self._prev_results = {}

        # print('starting fit', start_fit)
        # x, y, w, h = cv2.boundingRect(contour)

        # contour_plot = zeros(shape=(y + h, x + w), dtype=uint8)
        # cv2.drawContours(contour_plot, [contour, ], -1, 255, 1)
        # contour_pts = numpy.flip(numpy.argwhere(contour_plot > 0), axis=1)

        contour_pts = contour.reshape((-1, 2))
        start_fit = start_fit.reshape(-1).tolist()

        if True:
            initial_simplex = [start_fit, ]
            for i in range(len(start_fit)):
                x = start_fit.copy()
                x[i] += 1.0
                initial_simplex.append(x)

            opt_result = minimize(self._shape_measure_opencv, start_fit, args=(shape_function, contour_pts), method='Nelder-Mead',
                                  options={'xatol': 0.5, 'fatol': 5.0, 'initial_simplex': initial_simplex})
        else:
            opt_result = minimize(self._shape_measure_opencv, start_fit, args=(shape_function, contour_pts), method='Powell',
                                  options={'xtol': 0.5, 'ftol': 5.0})
        # print(opt_result)

        if opt_result.success:
            # print(f'Successful fit: {opt_result.nit} iterations, {self._call_count} calls, Chi^2 {opt_result.fun}')
            return numpy.array(opt_result.x).reshape((-1, 2))
        return None

    def _shape_measure_opencv(self, x, shape_function, ptlist):
        xtuple = tuple(x)

        chi2 = self._cached_results.get(xtuple, None)
        if chi2 is not None:
            # print('** duplicate call **', xtuple)
            return chi2

        with CodeTimer("_shape_measure_opencv"):
            # minimize flattens the array, so put it back as a contour
            inshape = numpy.array(x).reshape((-1, 2))

            contour = numpy.around(shape_function(inshape)).astype(int)

            chi2 = 0.0
            for pt in ptlist:
                d = cv2.pointPolygonTest(contour, tuple(pt), measureDist=True)
                chi2 += d*d

            # print(f'_shape_measure {call_count} {x} {chi2}')
            self._call_count += 1
            self._cached_results[xtuple] = chi2

        return chi2

    def _shape_measure_numpy(self, x, shape_function, ptlist):

        with CodeTimer("_shape_measure_numpy"):
            # minimize flattens the array, so put it back as a list of points
            inshape = numpy.array(x).reshape((-1, 2))

            # contour = numpy.around(shape_function(inshape)).astype(int)
            contour = shape_function(inshape)

            # compute the distances to each segment
            min_dist = None
            nsides = len(contour)
            for i0 in range(nsides):
                pt0 = contour[i0]
                pt1 = contour[(i0 + 1) % nsides]

                delta_line = pt1 - pt0
                delta_pts = ptlist - pt0

                # distance along the line to the point of closest approach
                along_line = delta_pts.dot(delta_line) / delta_line.dot(delta_line)
                # clip to be on the contour line segment
                # reshape to allow fast multiple. Basically a transpose.
                along_line = numpy.clip(along_line, 0.0, 1.0).reshape((-1, 1))

                pt_to_line = ptlist - (pt0 + along_line * delta_line)
                distances = numpy.linalg.norm(pt_to_line, axis=1)  # not squared
                if min_dist is None:
                    min_dist = distances
                else:
                    min_dist = numpy.minimum(min_dist, distances)

            chi2 = numpy.sum(min_dist**2)

        return chi2

# https://docs.scipy.org/doc/scipy-0.18.1/reference/generated/scipy.spatial.distance.cdist.html
# Y = cdist(XA, XB, 'sqeuclidean')

# https://maprantala.com/2010/05/16/measuring-distance-from-a-point-to-a-line-segment/
