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
* diamondspace.insert
* diamondspace.find_peaks

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

def _accumulate_line(accumulator, end_points, d_flag, weight):
    """ Accumulate one line segment into accumulator """

    size = accumulator.shape[0]
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
            steps = size
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
            steps = size
            Y = np.arange(0, steps)
            dif_x = (x0 - x1) / (y0 - y1)
            X = np.arange(0, steps) * dif_x + x0

    X = np.round(X).astype(np.int32)
    Y = np.round(Y).astype(np.int32)

    iX = np.logical_and(X >= 0, X <= size)
    iY = np.logical_and(Y >= 0, Y <= size)
    iXY = np.logical_and(iX, iY)

    accumulator[Y[iXY], X[iXY]] += weight

class DiamondSpace:
    """
    Wrapper for DiamondSpace accumulator of certain size
    """

    def __init__(self, d=1, size=256):

        # Init accumulator
        self.d = d
        self.size = size
        self.scale = (self.size - 1) / self.d
        self.A = []
        for i in range(4):
            self.A.append(np.zeros([self.size, self.size]))

    def clear(self):
        for i in range(4):
            self.A[i][:] = 0

    def lines_to_subspaces(self, l):
        """
        Transform lines l to the cascaded hough spaces (ST, SS, TS and TT space) using parallel coordinate.
        Input
        -----
        l : ndarray
            Nx3 array with lines in homogeneous coordinates
        Output
        ------
        l_transform : ndarray
            Nx4x3 array with transformation coordinates to ST, SS, TS and TT space for each input line
        """
        _validate_lines(l)
        d = self.d

        # ST, SS, TS, TT projection matrices
        m_ST = np.array([[0, -1, 0], [1, -1, d], [-1/d, 0, 0]])
        m_SS = np.array([[0, -1, 0], [-1, -1, d],[-1/d, 0, 0]])
        m_TS = np.array([[0, -1, 0], [-1, 1, d], [-1/d, 0, 0]])
        m_TT = np.array([[0, -1, 0], [1, 1, d],  [-1/d, 0, 0]])

        l_projections = np.stack([l @ M_i for M_i in (m_ST, m_SS, m_TS, m_TT)], axis=1)

        return l_projections

    def points_to_subspaces(self, p):
        """
        Transform points p to the cascaded hough spaces (ST, SS, TS and TT space) using parallel coordinate.
        Input
        -----
        p : ndarray
            Nx3 array with points in homogeneous coordinates
        Output
        ------
        p_transform : ndarray
            Nx4x3 array with transformation coordinates to ST, SS, TS and TT space for each input point
        """
        _validate_points(p)
        d = self.d

        # ST, SS, TS, TT projection matrices
        m_ST = np.array([[0, -1, -1/d], [0, 0, 1/d], [-d, 0, 1]])
        m_SS = np.array([[0, -1, -1/d], [0, 0, 1/d], [-d, 0, -1]])
        m_TS = np.array([[0, -1, 1/d],  [0, 0, 1/d], [-d, 0, -1]])
        m_TT = np.array([[0, -1, 1/d],  [0, 0, 1/d], [-d, 0, 1]])

        p_projections = np.stack([p @ M_i for M_i in (m_ST, m_SS, m_TS, m_TT)], axis=1)

        return p_projections

    def points_to_ds(self, p):
        """
        Transform point p to the diamond space
        Input
        -----
        p : Nx3 array with point in homogeneous coordinates
        Output
        ------
        p_transform :Nx3 array with point coordinates in Diamond space
        """

        _validate_points(p)
        d = self.d

        p_transform = np.empty(p.shape, np.float64)

        s0 = (p[:, 0] >= 0) * 1 + (p[:, 0] < 0) * -1
        s1 = (p[:, 1] >= 0) * 1 + (p[:, 1] < 0) * -1

        p_transform[:, 0] = -p[:,2]*d
        p_transform[:, 1] = -p[:,0]
        p_transform[:, 2] = (s0*s1*p[:,0] + p[:,1])/d + p[:,2]*s1

        return p_transform[:,0:2]/p_transform[:,2].reshape(-1,1)

    def points_from_ds(self, p):
        """
        Transform points p from the diamond space to original coordinate system
        -----
        p : Nx2 array with point in coordinates of the Diamond space
        Output
        ------
        p_inverse :Nx3 array with homogeneous coordinates of the point in the original coordinate system
        """
        _validate_points2D(p)

        d = self.d

        p_inverse = np.empty([p.shape[0],3], np.float64)

        p_inverse[:,0] = p[:,1]
        p_inverse[:,1] = np.abs(p[:,0]) + np.abs(p[:,1]) - d
        p_inverse[:,2] = p[:,0]/d

        eps = 1e-12

        regular_points = ~np.isclose(p_inverse[:,2:], 0, atol=eps)
        p_inverse = np.divide(p_inverse, p_inverse[:, 2:], where=regular_points, out=p_inverse)

        return p_inverse

    def line_intersections(self, lines):
        """ Return all intersections of projected lines with space border and axes  used in rasterization"""
        d = self.d

        # intersections with diagonal in flipped ST, SS, TS, TT spaces (all subspaces are flipped to positive quadrant)
        m_ST = np.array([[d, 0, 1],[0, 0, 0],[0, 1, 1/d]])
        m_SS = np.array([[-d, 0, -1], [0, 0, 0], [0, 1, 1 / d]])
        m_TS = np.array([[d, 0, 1],[0, 0, 0],[0, 1, 1/d]])
        m_TT = np.array([[-d, 0, -1],[0, 0, 0],[0, 1, 1/d]])

        p_diagonals = np.stack([lines @ M_i for M_i in (m_ST, m_SS, m_TS, m_TT)], axis=1)

        # intersections with -X,X,-Y,Y axes (all subspaces are flipped to positive quadrant)
        m_nX = np.array([[0, 0, 0],[-d, 0, -1],[0, 0, 1/d]])
        m_pX = np.array([[0, 0, 0],[d, 0, 1],[0, 0, 1/d]])
        m_nY = np.array([[0, 0, 1],[0, -d, -1],[0, 0, 0]])
        m_pY = np.array([[0, 0, 1],[0, d, 1],[0, 0, 0]])

        p_axes = np.stack([lines @ M_i for M_i in (m_nX, m_pX, m_nY, m_pY)], axis=1)

        return p_diagonals, p_axes

    def relevant_intersections(self, points):
        """ Check if intersections lie in diamond space """
        # normalize points - be careful with ideal points

        eps = 1e-12

        regular_points = ~np.isclose(points[:,:,2:],0,atol=eps)
        points = np.divide(points,points[:,:,2:], where=regular_points, out=points)

        # check x coordinate
        X = np.logical_and(points[:, :, 0] >= 0 - eps, points[:, :, 0] <= self.d + eps)

        # check y coordinate
        Y = np.logical_and(points[:, :, 1] >= 0 - eps, points[:, :, 1] <= self.d + eps)

        return points,  np.logical_and(np.logical_and(X, Y), np.logical_and(np.logical_and(X, Y), regular_points[:,:,0]))

    def get_intersection(self,l):
        """ Return relevant intersections of projected lines with space border and axes """

        l /= np.linalg.norm(l, keepdims=True, axis=1)

        D, A = self.line_intersections(l)
        D, D_I = self.relevant_intersections(D)
        A, A_I = self.relevant_intersections(A)

        return D, D_I, A, A_I


    def accumulate_subspaces(self, D, D_I, A, A_I, weights):
        """ Accumulate all line segments """

        D_s = D
        D_s[:, :, 0:2] = D_s[:, :, 0:2] * self.scale
        A_s = A
        A_s[:, :, 0:2] = A_s[:, :, 0:2] * self.scale

        for i, data in enumerate(_generate_subspaces(D_s, D_I, A_s, A_I)):
            d, d_i, a, a_i = data
            for j, data_lines in enumerate(_generate_lines(d, d_i, a, a_i)):
                x, y, d_flag = data_lines
                _accumulate_line(self.A[i],[x[0], x[1], y[0], y[1]], d_flag, weights[j])

    def insert(self, lines, weights=None):
        """
        Insert and accumulate lines 'lines' (in homogeneous coordinate) to diamond space
        """
        _validate_lines(lines)
        lines = lines.astype(np.double)
        D, D_I, A, A_I = self.get_intersection(lines)

        if weights is None:
            weights = np.ones(lines.shape[0])

        self.accumulate_subspaces(D, D_I, A, A_I, weights)


    def find_peaks_in_subspace(self, subspace, t_abs, prominence, min_dist):
        """
        Retrieve locations with prominent local maxima from one part of the accumulator
        """
        de = dilation(subspace + 1, ) / erosion(subspace + 1)
        p = peak_local_max(subspace, threshold_abs=t_abs, min_distance=min_dist, exclude_border=False)
        r,c = p[:,0], p[:,1]
        v = subspace[r,c]
        valid = de[r,c] > prominence

        peaks = np.empty([np.sum(valid),2], np.float64)
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

        peaks = []
        peaks_ds = []
        values = []

        threshold_abs = np.max(self.A) * t
        
        for i, (s, f) in enumerate(zip(self.A, space_flip)):
             p,v = self.find_peaks_in_subspace(s, threshold_abs, prominence, min_dist)

             p_ds =  p*f/self.scale
             p_i = self.points_from_ds(p_ds)

             peaks.append(p_i)
             values.append(v)
             peaks_ds.append(p*f)

        return np.vstack(peaks), np.hstack(values), np.vstack(peaks_ds)

    def attach_spaces(self):
        accumulator = np.zeros([self.size * 2 - 1, self.size * 2 - 1])
        accumulator[0: self.size, 0: self.size] = np.flipud(np.fliplr(self.A[3]))
        accumulator[0: self.size, self.size-1:] = np.maximum(np.flipud(self.A[2]), accumulator[0: self.size, self.size-1:])
        accumulator[self.size-1:, self.size-1:] = np.maximum(self.A[1], accumulator[self.size-1:, self.size-1:])
        accumulator[self.size-1:, 0: self.size] = np.maximum(np.fliplr(self.A[0]),accumulator[self.size-1:, 0: self.size])

        return accumulator
