import logging
import functools
import numpy as np
import operator
import warnings
import copy
import time 

import scipy
from scipy.linalg import LinAlgWarning

from . import dn, d3, ease, errors, util

# Constants

logger = logging.getLogger(__name__)    

ORIGIN = np.array((0, 0))

X = np.array((1, 0))
Y = np.array((0, 1))

UP = Y

# SDF Class

_ops = {}


class SDF2:
    def __init__(self, f):
        self.f = f

    def __call__(self, p):
        return self.f(p).reshape((-1, 1))

    def __getattr__(self, name):
        if name in _ops:
            f = _ops[name]
            return functools.partial(f, self)
        raise AttributeError

    def __or__(self, other):
        return union(self, other)

    def __and__(self, other):
        return intersection(self, other)

    def __sub__(self, other):
        return difference(self, other)

    def k(self, k=None):
        newSelf = copy.deepcopy(self)
        newSelf._k = k
        return newSelf
    
    @errors.alpha_quality
    def closest_surface_point(self, point):
        def distance(p):
            # root() wants same input/output dims (yeah...)
            return np.repeat(self.f(np.expand_dims(p, axis=0)).ravel()[0], 2)

        dist = self.f(np.expand_dims(point, axis=0)).ravel()[0]
        optima = dict()
        with warnings.catch_warnings():
            warnings.simplefilter(
                "ignore", (LinAlgWarning, RuntimeWarning, UserWarning)
            )
            for method in (
                # loosely sorted by speed
                "lm",
                "broyden2",
                "df-sane",
                "hybr",
                "broyden1",
                "anderson",
                "linearmixing",
                "diagbroyden",
                "excitingmixing",
                "krylov",
            ):
                try:
                    optima[method] = (
                        opt := scipy.optimize.root(
                            distance, x0=np.array(point), method=method
                        )
                    )
                except Exception as e:
                    pass
                opt.zero_error = abs(opt.fun[0])
                opt.zero_error_rel = opt.zero_error / dist
                opt.dist_error = np.linalg.norm(opt.x - point) - dist
                opt.dist_error_rel = opt.dist_error / dist
                logger.debug(f"{method = }, {opt = }")
                # shortcut if fit is good
                if (
                    np.allclose(opt.fun, 0)
                    and abs(opt.dist_error / dist - 1) < 0.01
                    and opt.dist_error < 0.01
                ):
                    break

            def cost(m):
                penalty = (
                    # unsuccessfulness is penaltied
                    (not optima[m].success)
                    # a higher status normally means something bad
                    + abs(getattr(optima[m], "status", 1))
                    # the more we're away from zero, the worse it is
                    # „1mm of away from boundary is as bad as one status or success step”
                    + optima[m].zero_error
                    # the distance error can be quite large e.g. for non-uniform scaling,
                    # and methods often find weird points, it makes sense to compare to the SDF
                    + optima[m].dist_error
                )
                logger.debug(f"{m = :20s}: {penalty = }")
                return penalty

            best_root = optima[best_method := min(optima, key=cost)]
            closest_point = best_root.x
            if (
                best_root.zero_error > 1
                or best_root.zero_error_rel > 0.01
                or best_root.dist_error > 1
                or best_root.dist_error_rel > 0.01
            ):
                warnings.warn(
                    f"Closest surface point to {point} acc. to method {best_method!r} seems to be {closest_point}. "
                    f"The SDF there is {best_root.fun[0]} (should be 0, that's {best_root.zero_error} or {best_root.zero_error_rel*100:.2f}% off).\n"
                    f"Distance between {closest_point} and {point} is {np.linalg.norm(point - closest_point)}, "
                    f"SDF says it should be {dist} (that's {best_root.dist_error} or {best_root.dist_error_rel*100:.2f}% off)).\n"
                    f"The root finding algorithms seem to have a problem with your SDF, "
                    f"this might be caused due to operations breaking the metric like non-uniform scaling.",
                    errors.SDFCADWarning,
                )
        return closest_point

    @errors.alpha_quality
    def surface_intersection(self, start, direction=None):
        """
        ``start`` at a point, move (back or forth) along a line following a
        ``direction`` and return surface intersection coordinates.

        .. note::

            In case there is no intersection, the result *might* (not sure
            about that) return the point on the line that's closest to the
            surface 🤔.

        Args:
            start (2d vector): starting point
            direction (2d vector or None): direction to move into, defaults to
            ``-start`` (”move to origin”).

        Returns:
            2d vector: the optimized surface intersection
        """
        if direction is None:
            direction = -start

        def transform(t):
            return start + t * direction

        def distance(t):
            # root() wants same input/output dims (yeah...)
            return np.repeat(self.f(np.expand_dims(transform(t), axis=0)).ravel()[0], 3)
            # return self.f(np.expand_dims(transform(t), axis=0)).ravel()[0]

        dist = self.f(np.expand_dims(start, axis=0)).ravel()[0]
        optima = dict()
        with warnings.catch_warnings():
            warnings.simplefilter(
                "ignore", (LinAlgWarning, RuntimeWarning, UserWarning)
            )
            for method in (
                # loosely sorted by speed
                "lm",
                "broyden2",
                "df-sane",
                "hybr",
                "broyden1",
                "anderson",
                "linearmixing",
                "diagbroyden",
                "excitingmixing",
                "krylov",
            ):
                try:
                    optima[method] = (
                        opt := scipy.optimize.root(distance, x0=[0], method=method)
                    )
                except Exception as e:
                    pass
                opt.zero_error = abs(opt.fun[0])
                opt.zero_error_rel = opt.zero_error / dist
                opt.point = transform(opt.x[0])
                logger.debug(f"{method = }, {opt = }")
                # shortcut if fit is good
                if np.allclose(opt.fun, 0):
                    break

            def cost(m):
                penalty = (
                    # unsuccessfulness is penaltied
                    (not optima[m].success)
                    # a higher status normally means something bad
                    + abs(getattr(optima[m], "status", 1))
                    # the more we're away from zero, the worse it is
                    # „1mm of away from boundary is as bad as one status or success step”
                    + optima[m].zero_error
                )
                logger.debug(f"{m = :20s}: {penalty = }")
                return penalty

            best_root = optima[best_method := min(optima, key=cost)]
            closest_point = transform(best_root.x[0])
            if best_root.zero_error > 1 or best_root.zero_error_rel > 0.01:
                warnings.warn(
                    f"Surface intersection point from {start = } to {direction = }, acc. to method {best_method!r} seems to be {closest_point}. "
                    f"The SDF there is {best_root.fun[0]} (should be 0, that's {best_root.zero_error} or {best_root.zero_error_rel*100:.2f}% off).\n"
                    f"The root finding algorithms seem to have a problem with your SDF, "
                    f"this might be caused due to operations breaking the metric like non-uniform scaling "
                    f"or just because there is no intersection...",
                    errors.SDFCADWarning,
                )
        return closest_point

    @errors.alpha_quality
    def minimum_sdf_on_plane(self, origin, normal, return_point=False):
        """
        Find the minimum SDF distance (not necessarily the real distance if you
        have non-uniform scaling!) on a plane around an ``origin`` that points
        into the ``normal`` direction.

        Args:
            origin (2d vector): a point on the plane
            normal (2d vector): normal vector of the plane
            return_point (bool): whether to also return the closest point (on
                the plane!)

        Returns:
            float: the (minimum) distance to the plane
            float, 2d vector : distance and closest point (on the plane!) if
                ``return_point=True``
        """
        basemat = np.array(
            (e1 := _perpendicular(normal))
        )

        def transform(t):
            return origin + basemat * t

        def distance(t):
            return self.f(np.expand_dims(transform(t), axis=0)).ravel()[0]

        optima = dict()
        with warnings.catch_warnings():
            warnings.simplefilter(
                "ignore", (LinAlgWarning, RuntimeWarning, UserWarning)
            )
            for method in (
                "Nelder-Mead",
                "Powell",
                "CG",
                "BFGS",
                "L-BFGS-B",
                "TNC",
                "COBYLA",
                "SLSQP",
                "trust-constr",
            ):
                try:
                    optima[method] = (
                        opt := scipy.optimize.minimize(
                            distance, x0=0, method=method
                        )
                    )
                    opt.point = transform(opt.x[0])
                    logger.debug(f"{method = }, {opt = }")
                except Exception as e:
                    logger.error(f"{method = } error {e!r}")

        best_min = optima[min(optima, key=lambda m: optima[m].fun)]
        if return_point:
            return best_min.fun, best_min.point
        else:
            return best_min.fun

    @errors.alpha_quality
    def extent_in(self, direction):
        """
        Determine the largest distance from the origin in a given ``direction``
        that's still within the object.

        Args:
            direction (3d vector): the direction to check

        Returns:
            float: distance from origin
        """
        # create probing points along direction to check where object ends roughly
        # object ends when SDF is only increasing (not the case for infinite repetitions e.g.)
        probing_points = np.expand_dims(np.logspace(-5, 9, 30), axis=1) * direction
        d = self.f(probing_points)  # get SDF value at probing points
        n_trailing_ascending = util.n_trailing_ascending_positive(d)
        if not n_trailing_ascending:
            return np.inf
        if (ratio := n_trailing_ascending / d.size) < 0.5:
            warnings.warn(
                f"extent_in({direction = !r}): "
                f"Only {n_trailing_ascending}/{d.size} ({ratio*100:.1f}%) of probed points in "
                f"{direction = } have ascending positive SDF distance values. "
                f"This can be caused by infinite objects. "
                f"Result of might be wrong. ",
                errors.SDFCADWarning,
            )
        faraway = probing_points[
            -n_trailing_ascending + 1
        ]  # choose first point after which SDF only increases
        closest_surface_point = self.closest_surface_point_to_plane(
            origin=faraway, normal=direction
        )
        extent = np.linalg.norm(closest_surface_point)
        return extent

    @property
    @errors.alpha_quality
    def bounds(self):
        """
        Return X Y bounds based on :any:`extent_in`.

        Return:
            6-sequence: lower X, upper X, lower Y, upper Y
        """
        return tuple(
            np.sign(d.sum()) * self.extent_in(d) for d in [-X, X, -Y, Y]
        )

    @errors.alpha_quality
    def closest_surface_point_to_plane(self, origin, normal):
        """
        Find the closest surface point to a plane around an ``origin`` that points
        into the ``normal`` direction.

        Args:
            origin (3d vector): a point on the plane
            normal (3d vector): normal vector of the plane

        Returns:
            3d vector : closest surface point
        """
        distance, plane_point = self.minimum_sdf_on_plane(
            origin=origin, normal=normal, return_point=True
        )
        return self.surface_intersection(start=plane_point, direction=normal)

    @errors.alpha_quality
    def move_to_positive(self, direction=Y):
        return self.translate(self.extent_in(-direction) * direction)


def sdf2(f):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        return SDF2(f(*args, **kwargs))

    return wrapper


def op2(f):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        return SDF2(f(*args, **kwargs))

    _ops[f.__name__] = wrapper
    return wrapper


def op23(f):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        return d3.SDF3(f(*args, **kwargs))

    _ops[f.__name__] = wrapper
    return wrapper


# Helpers


def _length(a):
    return np.linalg.norm(a, axis=1)


def _normalize(a):
    return a / np.linalg.norm(a)


def _dot(a, b):
    return np.sum(a * b, axis=1)


def _vec(*arrs):
    return np.stack(arrs, axis=-1)

def _perpendicular(v):
    if all(v == 0):
        raise ValueError("zero vector")
    return np.array([-v[1], v[0]])


_min = np.minimum
_max = np.maximum

# Primitives


@sdf2
def circle(radius=None, diameter=None, center=ORIGIN):
    if (radius is not None) == (diameter is not None):
        raise ValueError(f"Specify either radius or diameter")
    if radius is None:
        radius = diameter / 2

    def f(p):
        return _length(p - center) - radius

    return f


@sdf2
def line(normal=UP, point=ORIGIN):
    normal = _normalize(normal)

    def f(p):
        return np.dot(point - p, normal)

    return f


@sdf2
def slab(x0=None, y0=None, x1=None, y1=None, k=None):
    fs = []
    if x0 is not None:
        fs.append(line(X, (x0, 0)))
    if x1 is not None:
        fs.append(line(-X, (x1, 0)))
    if y0 is not None:
        fs.append(line(Y, (0, y0)))
    if y1 is not None:
        fs.append(line(-Y, (0, y1)))
    return intersection(*fs, k=k)


@sdf2
def rectangle(size=1, center=ORIGIN, a=None, b=None):
    if a is not None and b is not None:
        a = np.array(a)
        b = np.array(b)
        size = b - a
        center = a + size / 2
        return rectangle(size, center)
    size = np.array(size)

    def f(p):
        q = np.abs(p - center) - size / 2
        return _length(_max(q, 0)) + _min(np.amax(q, axis=1), 0)

    return f


@sdf2
def rounded_rectangle(size, radius, center=ORIGIN):
    try:
        r0, r1, r2, r3 = radius
    except TypeError:
        r0 = r1 = r2 = r3 = radius

    def f(p):
        x = p[:, 0]
        y = p[:, 1]
        r = np.zeros(len(p)).reshape((-1, 1))
        r[np.logical_and(x > 0, y > 0)] = r0
        r[np.logical_and(x > 0, y <= 0)] = r1
        r[np.logical_and(x <= 0, y <= 0)] = r2
        r[np.logical_and(x <= 0, y > 0)] = r3
        q = np.abs(p) - size / 2 + r
        return (
            _min(_max(q[:, 0], q[:, 1]), 0).reshape((-1, 1))
            + _length(_max(q, 0)).reshape((-1, 1))
            - r
        )

    return f


@sdf2
def equilateral_triangle():
    def f(p):
        k = 3**0.5
        p = _vec(np.abs(p[:, 0]) - 1, p[:, 1] + 1 / k)
        w = p[:, 0] + k * p[:, 1] > 0
        q = _vec(p[:, 0] - k * p[:, 1], -k * p[:, 0] - p[:, 1]) / 2
        p = np.where(w.reshape((-1, 1)), q, p)
        p = _vec(p[:, 0] - np.clip(p[:, 0], -2, 0), p[:, 1])
        return -_length(p) * np.sign(p[:, 1])

    return f


@sdf2
def hexagon(radius=None, diameter=None):
    if (radius is not None) == (diameter is not None):
        raise ValueError(f"Specify either radius or diameter")
    if radius is None:
        radius = diameter / 2
    radius *= 3**0.5 / 2

    def f(p):
        k = np.array((3**0.5 / -2, 0.5, np.tan(np.pi / 6)))
        p = np.abs(p)
        p -= 2 * k[:2] * _min(_dot(k[:2], p), 0).reshape((-1, 1))
        p -= _vec(
            np.clip(p[:, 0], -k[2] * radius, k[2] * radius), np.zeros(len(p)) + radius
        )
        return _length(p) * np.sign(p[:, 1])

    return f


@sdf2
def rounded_x(w, r):
    def f(p):
        p = np.abs(p)
        q = (_min(p[:, 0] + p[:, 1], w) * 0.5).reshape((-1, 1))
        return _length(p - q) - r

    return f


def RegularPolygon(n, r=1):
    ri = r * np.cos(np.pi / n)
    return intersection(
        *[slab(y0=-ri).rotate(a) for a in np.arange(0, 2 * np.pi, 2 * np.pi / n)]
    )


@sdf2
def polygon(points):
    points = [np.array(p) for p in points]

    def f(p):
        n = len(points)
        d = _dot(p - points[0], p - points[0])
        s = np.ones(len(p))
        for i in range(n):
            j = (i + n - 1) % n
            vi = points[i]
            vj = points[j]
            e = vj - vi
            w = p - vi
            b = w - e * np.clip(np.dot(w, e) / np.dot(e, e), 0, 1).reshape((-1, 1))
            d = _min(d, _dot(b, b))
            c1 = p[:, 1] >= vi[1]
            c2 = p[:, 1] < vj[1]
            c3 = e[0] * w[:, 1] > e[1] * w[:, 0]
            c = _vec(c1, c2, c3)
            s = np.where(np.all(c, axis=1) | np.all(~c, axis=1), -s, s)
        return s * np.sqrt(d)

    return f


# Positioning


@op2
def translate(other, offset):
    def f(p):
        return other(p - offset)

    return f


@op2
def scale(other, factor):
    try:
        x, y = factor
    except TypeError:
        x = y = factor
    s = (x, y)
    m = min(x, y)

    def f(p):
        return other(p / s) * m

    return f


@op2
def rotate(other, angle):
    s = np.sin(angle)
    c = np.cos(angle)
    m = 1 - c
    matrix = np.array(
        [
            [c, -s],
            [s, c],
        ]
    ).T

    def f(p):
        return other(np.dot(p, matrix))

    return f


@op2
def circular_array(other, count):
    angles = [i / count * 2 * np.pi for i in range(count)]
    return union(*[other.rotate(a) for a in angles])


# Alterations


@op2
def elongate(other, size):
    def f(p):
        q = np.abs(p) - size
        x = q[:, 0].reshape((-1, 1))
        y = q[:, 1].reshape((-1, 1))
        w = _min(_max(x, y), 0)
        return other(_max(q, 0)) + w

    return f


# 2D => 3D Operations


@op23
def extrude(other, h=np.inf):
    def f(p):
        d = other(p[:, [0, 1]])
        w = _vec(d.reshape(-1), np.abs(p[:, 2]) - h / 2)
        return _min(_max(w[:, 0], w[:, 1]), 0) + _length(_max(w, 0))

    return f


@op23
def extrude_to(a, b, h, e=ease.linear):
    def f(p):
        d1 = a(p[:, [0, 1]])
        d2 = b(p[:, [0, 1]])
        t = e(np.clip(p[:, 2] / h, -0.5, 0.5) + 0.5)
        d = d1 + (d2 - d1) * t.reshape((-1, 1))
        w = _vec(d.reshape(-1), np.abs(p[:, 2]) - h / 2)
        return _min(_max(w[:, 0], w[:, 1]), 0) + _length(_max(w, 0))

    return f


@op23
def revolve(other, offset=0):
    def f(p):
        xy = p[:, [0, 1]]
        # use horizontal distance to Z axis as X coordinate in 2D shape
        # use Z coordinate as Y coordinate in 2D shape
        q = _vec(_length(xy) - offset, p[:, 2])
        return other(q)

    return f


# Common

union = op2(dn.union)
difference = op2(dn.difference)
intersection = op2(dn.intersection)
blend = op2(dn.blend)
negate = op2(dn.negate)
dilate = op2(dn.dilate)
erode = op2(dn.erode)
shell = op2(dn.shell)
repeat = op2(dn.repeat)
mirror = op2(dn.mirror)
modulate_between = op2(dn.modulate_between)
stretch = op2(dn.stretch)
