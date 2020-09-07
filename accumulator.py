"""
Diamond space transformation for vanishing point detection
This package implements the method from
    Dubska et al, Real Projective Plane Mapping for Detection of Orthogonal Vanishing Points, BMVC 2013
Module
------
The module provides a high-level function for vanishing point detection in image
and also low-level functions for
* accumulation of observations to Diamond space,
* point mapping from/to Diamond space
that can be used to construct a custom transform of user-defined lines.
See also
--------
* diamondspace.accumulate
* diamondspace.find_peaks
* diamondspace.line_parameters
References
----------
[1] Dubska et al, Real Projective Plane Mapping for Detection of Orthogonal Vanishing Points, BMVC 2013
"""

import numpy as np
from skimage.feature import peak_local_max
from skimage.morphology.grey import erosion, dilation


def _validate_lines(x:np.ndarray):
    if not isinstance(x, np.ndarray):
        raise TypeError("Lines must be numpy array")
    if x.ndim != 2 or x.shape[1] != 3:
        raise ValueError("Lines must be 2D array with 3 columns (lines in homogeneous coordinates)")


def _validate_points(x: np.ndarray):
    if not isinstance(x, np.ndarray):
        raise TypeError("Points must be numpy array")
    if x.ndim != 2 or x.shape[1] != 3:
        raise ValueError("Points must be 2D array with 3 columns (points in homogeneous coordinates)")

def _validate_points2D(x: np.ndarray):
    if not isinstance(x, np.ndarray):
        raise TypeError("Points must be numpy array")
    if x.ndim != 2 or x.shape[1] != 2:
        raise ValueError("Points must be 2D array with 2 columns")

def _generate_subspaces(D, D_I, A, A_I):
    idx_d = [0, 1, 2, 3]
    idx_a = [(0, 3), (1, 3), (1, 2), (0, 2)]
    for d, a, in zip(idx_d, idx_a):
        yield D[:, d, :], D_I[:, d], A[:, a, :], A_I[:, a]


def _generate_lines(d, d_i, a, a_i):
    for i in range(d.shape[0]):
        if a_i[i, 0] and a_i[i, 1] and d_i[i]:
            if np.linalg.norm(a[i, 1, :] - a[i, 0, :]) < 0.0001:
                yield d[i, :], a[i, 0, :], True
            else:
                yield a[i, 1, :], a[i, 0, :], False
        elif a_i[i, 0] and a_i[i, 1]:
            yield a[i, 1, :], a[i, 0, :], False
        elif d_i[i] and a_i[i, 0]:
            yield d[i, :], a[i, 0, :], True
        elif d_i[i] and a_i[i, 1]:
            yield d[i, :], a[i, 1, :], True


def _accumulate_line(accumulator, end_points, d_flag, weight=1):
    """ Accumulate one line segment into accumulator """
    space_size = accumulator.shape[0]
    x0, y0, x1, y1 = end_points
    dx = x0 - x1
    dy = y0 - y1

    if abs(dx) < 0.001 and abs(dy) < 0.001:
        X = x0
        Y = y0
    elif abs(dx) > abs(dy):
        if d_flag:
            steps = np.round(x0).astype(np.int32) + 1
            X = np.linspace(x0, 0, steps)
            dif_y = (y0 - y1) / (x1 - x0)
            Y = np.arange(0, steps) * dif_y + y0
        else:
            if x1 < x0:
                x0, y0, x1, y1 = x1, y1, x0, y0
            steps = space_size
            X = np.arange(0, steps)
            dif_y = (y0 - y1) / (x0 - x1)
            Y = np.arange(0, steps) * dif_y + y0
    else:
        if d_flag:
            steps = np.round(y0).astype(np.int32) + 1
            Y = np.linspace(y0, 0, steps)
            dif_x = (x0 - x1) / (y1 - y0)
            X = np.arange(0, steps) * dif_x + x0
        else:
            if y1 < y0:
                x0, y0, x1, y1 = x1, y1, x0, y0
            steps = space_size
            Y = np.arange(0, steps)
            dif_x = (x0 - x1) / (y0 - y1)
            X = np.arange(0, steps) * dif_x + x0

    X = np.round(X).astype(np.int32)
    Y = np.round(Y).astype(np.int32)

    iX = np.logical_and(X >= 0, X <= space_size)
    iY = np.logical_and(Y >= 0, Y <= space_size)
    iXY = np.logical_and(iX, iY)

    accumulator[Y[iXY], X[iXY]] += weight

def _relevant_intersections(points, d):
    """ Check if intersections lie in diamond space """
    # normalize points - be careful with ideal points
    regular_points = np.copy(points[:, :, 2])
    regular_points[abs(regular_points) < 0.0001] = 1

    points = points / np.repeat(regular_points[:, :, np.newaxis], 3, axis=2)

    # check x coordinate
    X = np.logical_and(points[:, :, 0] >= 0, points[:, :, 0] <= d)

    # check y coordinate
    Y = np.logical_and(points[:, :, 1] >= 0, points[:, :, 1] <= d)

    return points, np.logical_and(np.logical_and(X, Y), abs(points[:, :, 2]) > 0.0001)

def _line_intersections(l, d):
    """ Return all intersections of projected lines with space border and axes """
    # intersections with diagonal in ST, SS, TS, TT spaces
    diagonals = np.array([[-d ** 2, 0, d, -d ** 2, 0, -d, d ** 2, 0, d,
                           d ** 2, 0, -d],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, d, 1, 0, d, 1, 0, -d, 1, 0, -d, 1]])

    # flip matrix for ST, SS, TS, TT spaces
    f = np.array([[-1, 1, 1, 1, 1, 1, 1, -1, 1, -1, -1, 1],
                  [-1, 1, 1, 1, 1, 1, 1, -1, 1, -1, -1, 1],
                  [-1, 1, 1, 1, 1, 1, 1, -1, 1, -1, -1, 1]])

    p_diagonals = np.matmul(l, diagonals * f)

    # intersections with -X,X,-Y,Y axes
    axes = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1],
                     [d ** 2, 0, -d, d ** 2, 0, d, 0, d, -1, 0, d, 1],
                     [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0]])

    # flip matrix for -X,X,-Y,Y spaces
    f = np.array([[-1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1],
                  [-1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1],
                  [-1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1]])

    p_axes = np.matmul(l, axes * f)

    return p_diagonals.reshape(-1, 4, 3), p_axes.reshape(-1, 4, 3)

class DiamondSpace:
    """
    Wrapper for DiamondSpace accumulator of certain size
    """

    def __init__(self, d=1, size=256):

        # Init accumulator
        self.d = d
        self.size = size
        self.A = []
        for i in range(4):
            self.A.append(np.zeros([self.size, self.size]))

    def clear(self):
        for i in range(4):
            self.A[i][:] = 0

    def lines_to_subspaces(self, l):
        """
        Transform lines l to the diamond space and for each return 4 lines in diamond spaces (one for each subspace).
        Input
        -----
        l : ndarray
            Nx3 array with lines in homogeneous coordinates
        Output
        ------
        l_transform : ndarray
            4x3xN array with four lines (transformation to ST, SS, TS and TT space) for each input line
        """
        _validate_lines(l)

        # ST, SS, TS, TT projection matrices
        m = np.array([[0, -self.d, 0, 0, -self.d, 0, 0, -self.d, 0, 0, -self.d, 0],
                      [self.d, -self.d, self.d ** 2, -self.d, -self.d, self.d ** 2, -self.d, self.d,
                       self.d ** 2, self.d, self.d, self.d ** 2],
                      [-1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0]])

        l_transform = np.matmul(l, m)

        return l_transform.reshape(-1, 4, 3)

    def points_to_subspaces(self, p):
        """
        Transform points p to the diamond space and for each return 4 points in diamond spaces (one for each subspace).
        Input
        -----
        p : ndarray
            Nx3 array with points in homogeneous coordinates
        Output
        ------
        p_transform : ndarray
           4x3xN array with four points (transformation to ST, SS, TS and TT space) for each input point
        """
        _validate_points(p)

        # ST, SS, TS, TT projection matrices
        m = np.array([[0, -self.d, -1, 0, -self.d, -1, 0, -self.d, 1, 0, -self.d, 1],
                      [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1],
                      [-self.d ** 2, 0, self.d, -self.d ** 2, 0, -self.d, -self.d ** 2, 0, -self.d, -self.d ** 2,
                       0, self.d]])

        p_transform = np.matmul(p, m)

        return p_transform.reshape(-1, 4, 3)

    def points_transform(self, point):
        """
        Transform point p to the diamond space
        -----
        p : Nx3 array with point in homogeneous coordinates
        Output
        ------
        p_transform :Nx3 array with point coordinates in Diamond space
        """
        s0 = 1
        s1 = 1

        if point[0] < 0:
            s0 = -1

        if point[1] < 0:
            s1 = -1

        m = np.array([[0, -self.d, s0 * s1],
                      [0, 0, 1],
                      [-self.d ** 2, 0, self.d * s1]])

        p = np.matmul(point, m)
        if p[2] != 0:
            return p / p[2]
        else:
            return p

    def points_inverse(self, p):
        """
        Transform point p from the diamond space to original coordinate system
        -----
        p : Nx3 array with point in homogeneous coordinates in Diamond space
        Output
        ------
        p_inverse :Nx3 array with coordinates of the point in the original coordinate system
        """
        _validate_points2D(p)

        x = p[:,1]*self.d
        y = self.d*(np.abs(p[:,0]) + np.abs(p[:,1]) - self.d)
        w = p[:,0]

        p_i = np.vstack((x,y,w)).T

        p_regular = w != 0
        p_i[p_regular, :] = p_i[p_regular, :]/p_i[p_regular, 2].reshape(-1, 1)

        return p_i

    def get_intersection(self,l):
        """ Return relevant intersections of projected lines with space border and axes """

        l = l / np.linalg.norm(l, axis=1).reshape(-1, 1)

        D, A = _line_intersections(l, self.d)
        D, D_I = _relevant_intersections(D, self.d + 0.0001)
        A, A_I = _relevant_intersections(A, self.d + 0.0001)

        return D, D_I, A, A_I


    def accumulate_subspaces(self, D, D_I, A, A_I):
        """ Accumulate all line segments """

        self.clear()

        scale = (self.size - 1) / self.d
        D_s = D
        D_s[:, :, 0:2] = D_s[:, :, 0:2] * scale
        A_s = A
        A_s[:, :, 0:2] = A_s[:, :, 0:2] * scale

        for i, data in enumerate(_generate_subspaces(D_s, D_I, A_s, A_I)):
            d, d_i, a, a_i = data
            for x, y, d_flag in _generate_lines(d, d_i, a, a_i):
                _accumulate_line(self.A[i],[x[0], x[1], y[0], y[1]], d_flag)

    def insert(self, l, weight=None):
        """
        Insert and accumulate lines l (in homogeneous coordinate) to diamond space
        """
        _validate_lines(l)
        D, D_I, A, A_I = self.get_intersection(l)

        self.accumulate_subspaces(D, D_I, A, A_I)


    def find_peaks_in_subspace(self, subspace, t_abs, prominence, min_dist):
        """
        Retrieve locations with prominent local maxima from one part of the accumulator
        """
        de = dilation(subspace + 1, ) / erosion(subspace + 1)
        p = peak_local_max(subspace, threshold_abs=t_abs, min_distance=min_dist, exclude_border=False)
        r,c = p[:,0], p[:,1]
        v = subspace[r,c]
        valid = de[r,c] > prominence

        peaks = np.ndarray([np.sum(valid),2])
        values = v[valid]

        # Subpixel correction of the peak
        s = np.pad(subspace, ((1, 1), (1, 1)), constant_values=((0, 0), (0, 0)))

        for i,pi in enumerate(p[valid]):
            neighborhood = s[pi[0]:pi[0] + 3, pi[1]:pi[1] + 3]
            neighborhood = neighborhood / np.sum(neighborhood)
            weight_r = np.tile(np.array([-1.0, 0, 1]), (3, 1)).T + pi[0]
            weight_c = np.tile(np.array([-1.0, 0, 1]), (3, 1)) + pi[1]

            peaks[i,1] = np.sum(neighborhood * weight_r).astype(float)
            peaks[i,0] = np.sum(neighborhood * weight_c).astype(float)

        return peaks,values


    def find_peaks(self, t=0.8, prominence=2, min_dist=1):
        space_flip = [(-1, 1), (1, 1), (1, -1), (-1, -1)]
        scale = (self.size - 1) / self.d

        peaks = []
        peaks_ds = []
        values = []

        threshold_abs = np.max(self.A) * t
        
        for i, (s, f) in enumerate(zip(self.A, space_flip)):
             p,v = self.find_peaks_in_subspace(s, threshold_abs, prominence, min_dist)

             p_ds =  p*f/scale
             p_i = self.points_inverse(p_ds)

             peaks.append(p_i)
             values.append(v)
             peaks_ds.append(p*f)

        return np.vstack(peaks), np.hstack(values), np.vstack(peaks_ds)

    def attach_spaces(self):
        accumulator = np.zeros([self.size * 2 - 1, self.size * 2 - 1])
        accumulator[0: self.size, 0: self.size] = np.flipud(np.fliplr(self.A[3]))
        accumulator[0: self.size, self.size-1:] = np.flipud(self.A[2])
        accumulator[self.size-1:, self.size-1:] = self.A[1]
        accumulator[self.size-1:, 0: self.size] = np.fliplr(self.A[0])

        return accumulator
