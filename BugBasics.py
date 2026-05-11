from __future__ import annotations
import numpy as np
from numpy import array as ar, pi, cos, sin, sqrt
from ladybug_geometry.geometry2d import Point2D, LineSegment2D, Polyline2D, Ray2D, Vector2D, Polygon2D
from functools import cached_property
import matplotlib.pyplot as plt
from numbers import Number
from typing import overload, Sequence, Mapping, Any

# ----------------------------- HELPERS -----------------------------
def _fmt(arr, dtype=float):
    return ar(arr, dtype=dtype)
    
def coerce_coords(
    obj: Any, *, dtype=float,
    elem_pairs: tuple[tuple[str, ...], ...] = (("x", "y"), ("X", "Y")),
    pair_accessors: tuple[str, ...] = (
        "xy", "coords",
        "to_xy", "as_xy",
        "to_array", "as_array",
        "to_list", "as_list", "tolist",
        "to_tuple", "as_tuple", "astuple", "arr"
    )
):
    if isinstance(obj, (str, bytes)):
        raise TypeError("Cannot coerce string to coordinates")
    
    # 1) Array-like check
    try: return _fmt(obj, dtype)
    except: pass

    # 2) Shared coord refs (property or zero-arg method returning an array-like)
    def val(a): return a() if callable(a) else a
    
    for name in pair_accessors:
        attr = getattr(obj, name, None)
        if attr is not None:
            try:
                coords = val(attr)
                return _fmt(coords, dtype)
            except Exception:
                pass
                
    # 3) Individual coord refs (properties or zero-arg methods)
    # 3.1) Direct attribute extraction
    for pair in elem_pairs:
        if np.all([hasattr(obj, name) for name in pair]):
            try: return _fmt([val(getattr(obj, name)) for name in pair], dtype)
            except Exception: pass

    # 3.2) Mapping with per-element keys (case-insensitive)
    if isinstance(obj, Mapping):
        keys = {str(k).lower(): k for k in obj.keys()}
        for pair in elem_pairs:
            pair_low = [name.lower() for name in pair]
            if np.all([name in keys for name in pair_low]):
                return _fmt([obj[keys[name]] for name in pair_low], dtype)

    raise TypeError(f"Cannot coerce {type(obj)} to coordinates")

# ----------------------------- BUG DELEGATION -----------------------------
# Map bug types -> your wrapper classes
_BUG_WRAP_MAP = {}

def _register_bug_map(d):
    _BUG_WRAP_MAP.update(d)

class BugDelegateMixin:
    """Fallback to self._bug for missing attributes/methods.
    - Coerces args/kwargs to their _bug equivalents (for your types).
    - Wraps return values from ladybug back into your types.
    """

    # --- conversion helpers ---
    @staticmethod
    def _is_primitive(x):
        return isinstance(x, (Number, str, bytes, np.ndarray, slice, range, type(None)))

    @staticmethod
    def _bug_to_array(obj):
        # Robustly convert ladybug objects to np array
        if hasattr(obj, "to_array"):
            return ar(obj.to_array(), dtype=float)
        for names in (("x","y"), ("X","Y")):
            if all(hasattr(obj, n) for n in names):
                return ar([getattr(obj, names[0]), getattr(obj, names[1])], dtype=float)
        # Fallback (may raise)
        return ar(obj, dtype=float)

    def _to_bug(self, x):
        # Your wrappers → their ._bug; containers → recurse; primitives → pass
        if isinstance(x, BaseShape):
            return x._bug
        if isinstance(x, (list, tuple)):
            return type(x)(self._to_bug(e) for e in x)
        if isinstance(x, dict):
            return {k: self._to_bug(v) for k, v in x.items()}
        return x  # primitives unchanged

    def _from_bug(self, x):
        # ladybug → your wrappers; containers → recurse; primitives → pass
        for bug_t, wrap_cls in _BUG_WRAP_MAP.items():
            if isinstance(x, bug_t):
                return wrap_cls(self._bug_to_array(x))
        if isinstance(x, (list, tuple)):
            return type(x)(self._from_bug(e) for e in x)
        if isinstance(x, dict):
            return {k: self._from_bug(v) for k, v in x.items()}
        return x
    
    

    # --- the delegation hook ---
    def __getattr__(self, name):
        # Called only if normal lookup fails
        bug = getattr(self, "_bug", None)
        if bug is None:
            raise AttributeError(f"{type(self).__name__!s} has no attribute {name!r}")
        attr = getattr(bug, name)  # may raise AttributeError

        if callable(attr):
            def wrapper(*args, **kwargs):
                bargs  = tuple(self._to_bug(a) for a in args)
                bkwargs = {k: self._to_bug(v) for k, v in kwargs.items()}
                out = attr(*bargs, **bkwargs)
                return self._from_bug(out)
            # Preserve some metadata if you want
            wrapper.__name__ = getattr(attr, "__name__", name)
            return wrapper
        else:
            return self._from_bug(attr)


# ----------------------------- VIRTUALS -----------------------------
class BaseShape(BugDelegateMixin):
    __slots__ = ("arr", "tol")
    ndim = 2
    def __init__(self, arr, tol: float | None = None) -> None:
        arr = coerce_coords(arr)
        arr = np.atleast_2d(arr)
        assert arr.ndim == 2 and arr.shape[1] == 2, f"{type(self).__name__} input must be of shape [n,2], not {arr.shape}"
        self.arr = arr
        self.tol = float(tol) if tol is not None else 1e-6
    
    # initiation
    def copy(self):
        return type(self)(self.arr, tol=self.tol)
    
    # Helpers
    def _coerce_other(self, other):
        T = type(self)
        if isinstance(other, T): return other
        try: return T(other)  # allow e.g. Point((x,y)) -> Point
        except: return other

    # Access
    def __getitem__(self, key): return self.arr[key]
    @property
    def xs(self): return self[:,0]
    @property
    def ys(self): return self[:,1]
    @property
    def x(self): return sum(self[:,0])/self.shape[0]
    @property
    def y(self): return sum(self[:,1])/self.shape[0]
    @property
    def centroid(self): return Point(np.sum(self.arr, axis=0)/self.shape[0], tol=self.tol)#
    @property
    def bounds(self): return ar([min(self[:,0]), min(self[:,1]), max(self[:,0]), max(self[:,1])])
    @property
    def bounding_points(self): return ar([Point([self.bounds[0], self.bounds[1]], tol=self.tol),Point([self.bounds[2], self.bounds[3]], tol=self.tol)])
    @cached_property
    def perimeter_segments(self):
        return [Line(self[i], self[i+1]) for i in range(self.shape[0]-1)]
    @property
    def x_min(self): return min(self[:,0])
    @property
    def x_max(self): return max(self[:,0])
    @property
    def y_min(self): return min(self[:,1])
    @property
    def y_max(self): return max(self[:,1])
    
    # Array properties
    def __len__(self): return len(self.arr)
    @property
    def shape(self): return self.arr.shape
    
    # Boolean operations
    def _same_shape(self, other) -> bool | NotImplemented:
        other = self._coerce_other(other)
        sa = getattr(self, "shape", None)
        sb = getattr(other, "shape", None)
        if sa is None or sb is None:
            return NotImplemented
        return sa == sb
        
    def __eq__(self, other):
        b = self._coerce_other(other)
        # Shape guard (if relevant)
        same = self._same_shape(b)
        if same is NotImplemented: return NotImplemented
        elif same is False: return False
        # Numeric equality (override arr accessor if your subclass differs)
        a_arr = getattr(self, "arr", None)
        b_arr = getattr(b, "arr", None)
        if a_arr is None or b_arr is None: return NotImplemented
        return bool(np.allclose(a_arr, b_arr, rtol=0.0, atol=self.tol))
    
    def __neq__(self, other): return not self.__eq__(other)
    
    # Maths operations
    def __add__(self, other):
        #other = self._coerce_other(other)
        arr2 = coerce_coords(other)#getattr(other, "arr", None)
        if arr2 is None:
            raise TypeError(f"Cannot add obj of type {type(other).__name__} to {type(self).__name__}")
        arr2 = np.broadcast_to(arr2, self.shape)
        
        result = self.copy()
        result.arr += arr2
        return type(self)(result, tol=self.tol)
    
    def __radd__(self, other): return self.__add__(other)
            
    def __neg__(self):
        result = self.copy()
        result.arr = -result.arr
        return result
    
    def __sub__(self, other):
        return self.__add__(-other)
    def __rsub__(self, other):
        return (-self).__add__(other)
    
    def __mul__(self, other):
        other = coerce_coords(other)
        other = np.broadcast(other, self.shape)
        result = self.copy()
        result.arr = result.arr*other
        return result
    
    def __rmul__(self, other): return self.__mul__(other)
    
    def __truediv__(self, other):
        other = coerce_coords(other)
        other = np.broadcast(other, self.shape)
        result = self.copy()
        result.arr = result.arr/other
        return result
    
    def __rtruediv__(self, other):
        other = coerce_coords(other)
        other = np.broadcast(other, self.shape)
        result = self.copy()
        result.arr = other/result.arr
        return result
    
    def plot(self, show=False, ax=None, point_color="black", line_color=None):
        pts = self.arr
        
        if ax is None:
            fig, ax = plt.subplots()
        if line_color is not None:
            ax.plot(pts[:, 0], pts[:, 1], color=line_color, linewidth=1.0)
        if point_color is not None:
            ax.scatter(pts[:, 0], pts[:, 1], color=point_color, s=15, zorder=3)

        ax.set_aspect("equal", adjustable="box")
        ax.autoscale_view()
        if show:
            plt.show()
        return ax
    
    @cached_property
    def direction_vectors(self):
        if self.shape[0] == 1: return np.empty([0,2])
        else: return self[1:,:]-self[:-1,:]
    @cached_property
    def lengths(self):
        if self.shape[0] == 1: return ar([])
        else:
            return np.linalg.norm(self.direction_vectors, axis=1)
    @cached_property
    def direction_unit_vectors(self):
        if self.shape[0] == 1: return np.empty([0,2])
        else: return self.direction_vectors/self.lengths
    @cached_property
    def length(self): return sum(self.lengths)
    
    def move(self, vector):
        if not isinstance(vector, Vector):
            vector = Vector(vector, tol=self.tol)
        boundary = self.arr + vector.arr
        return type(self)(boundary, tol=self.tol)
    
    def rotate(self, angle_ccw, centroid=None):
        if centroid is None:
            centroid = self.centroid
        elif not isinstance(centroid, Point):
            centroid = Point(centroid, tol=self.tol)
    
        theta = float(angle_ccw)
        c, s = cos(theta), sin(theta)
        R = ar([[c, -s],
                [s,  c]])                # CCW rotation
    
        boundary = self.arr.copy()   # (N,2)
        rel = boundary - centroid.arr[None, ...]  # (N,2)
        rot = rel @ R                     # (N,2)
        new_boundary = rot + centroid.arr[None, ...]
        return type(self)(new_boundary, tol=self.tol)
    
    def scale(self, factor, centroid=None):
        if centroid is None:
            centroid = self.centroid
        elif not isinstance(centroid, Point):
            centroid = Point(centroid, tol=self.tol)
        
        factor = float(factor)
        boundary = self.arr.copy()   # (N,2)
        rel = boundary - centroid.arr[None, ...]  # (N,2)
        scl = rel*factor
        new_boundary = scl + centroid.arr[None, ...]
        return type(self)(new_boundary, tol=self.tol)
    
    def transform(self, vector=None, angle_ccw=0, scale_factor=1, centroid=None):
        poly = self
        if angle_ccw:
            poly = poly.rotate(angle_ccw, centroid)
        if not np.isclose(scale_factor, 1.0):
            poly = poly.scale(scale_factor, centroid)
        return poly if vector is None else poly.move(vector)
    
    def __str__(self): return self.arr.__str__()
    def __repr__(self): return self.arr.__repr__()
    

# ----------------------------- CLASSES -----------------------------
class Point(BaseShape):
    __slots__ = ("arr", "tol")

    @overload
    def __init__(self, x: float, y: float, /, *, tol: float | None = ...) -> None: ...
    @overload
    def __init__(self, pts: Sequence[Number] | Mapping[str, Number], /, *, tol: float | None = ...) -> None: ...
    @overload
    def __init__(self, *, x: float, y: float, tol: float | None = ...) -> None: ...

    def __init__(self, *args, **kw) -> None:
        tol = kw.pop("tol", None)
        if "x" in kw or "y" in kw:
            x, y = kw.get("x", 0.0), kw.get("y", 0.0)
        elif len(args) >= 2 and all(isinstance(a, Number) for a in args[:2]):
            x, y = float(args[0]), float(args[1])
        elif len(args) == 1:
            try:
                x, y = coerce_coords(args[0])
            except:
                x, y = coerce_coords(args[0][0])
        else:
            print(args)
            raise TypeError("Provide (x,y), pts=[x,y], mapping {'x','y'}, or x=...,y=...")
        arr = ar([float(x), float(y)], dtype=float)
        super().__init__(arr, tol)
    
    # initiation
    def copy(self):
        return type(self)(self.arr[0], tol=self.tol)

    def __getitem__(self, key):
        try:
            return super().__getitem__((0, key))
        except:
            return super().__getitem__(key)
    
    def __add__(self, other):
        if isinstance(other, Point):
            return Point(self.arr[0]+other.arr[0], tol=self.tol)
        elif isinstance(other, Vector):
            return type(self)(self.arr[0]+other.arr[0], tol=self.tol)
        elif isinstance(other, (list, tuple, np.ndarray)):
            return type(self)(self.arr[0]+ar(other, dtype=float), tol=self.tol)
        elif isinstance(other, Number):
            return type(self)([v+float(other) for v in self.arr[0]], tol=self.tol)
        try:
            return type(self)(self.arr[0]+(other.arr[0] if isinstance(other, BaseShape) else other), tol=self.tol)
        except:
            result = self.copy()
            try:
                result.arr = ar([result.arr[0]+other], dtype=float)
            except:
                result.arr = ar([result.arr[0]+type(self)(other).arr[0]])
            return type(self)(result, tol=self.tol)

    def __sub__(self, other):
        if isinstance(other, Point):
            return Vector(self.arr[0]-other.arr[0], tol=self.tol)
        elif isinstance(other, Vector):
            return type(self)(self.arr[0]-other.arr[0], tol=self.tol)
        elif isinstance(other, (list, tuple, np.ndarray)):
            return Vector(self.arr[0]-ar(other, dtype=float), tol=self.tol)
        elif isinstance(other, Number):
            return type(self)([v-float(other) for v in self.arr[0]], tol=self.tol)
        try:
            return type(self)(self.arr[0]-(other.arr[0] if isinstance(other, BaseShape) else other), tol=self.tol)
        except:
            result = self.copy()
            try:
                result.arr = ar([result.arr[0]-other], dtype=float)
            except:
                result.arr = ar([result.arr[0]-type(self)(other).arr[0]])
            return type(self)(result, tol=self.tol)
    
    def __mul__(self, other):
        if isinstance(other, Point):
            return Point(self.arr[0]*other.arr[0], tol=self.tol)
        elif isinstance(other, Vector):
            return type(self)(self.arr[0]*other.arr[0], tol=self.tol)
        elif isinstance(other, (list, tuple, np.ndarray)):
            return type(self)(self.arr[0]*ar(other, dtype=float), tol=self.tol)
        elif isinstance(other, Number):
            return type(self)(self.arr[0]*float(other), tol=self.tol)
        try:
            return type(self)(self.arr[0]*(other.arr[0] if isinstance(other, BaseShape) else other), tol=self.tol)
        except:
            result = self.copy()
            try:
                result.arr = ar([result.arr[0]*other], dtype=float)
            except:
                result.arr = ar([result.arr[0]*type(self)(other).arr[0]])
            return type(self)(result, tol=self.tol)

    def __truediv__(self, other):
        if isinstance(other, (Point, Vector)):
            return type(self)(self.arr[0]/other.arr[0], tol=self.tol)
        elif isinstance(other, (list, tuple, np.ndarray)):
            return type(self)(self.arr[0]/ar(other, dtype=float), tol=self.tol)
        elif isinstance(other, Number):
            return type(self)(self.arr[0]/float(other), tol=self.tol)
        try:
            return type(self)(self.arr[0]/(other.arr[0] if isinstance(other, BaseShape) else other), tol=self.tol)
        except:
            result = self.copy()
            try:
                result.arr = ar([result.arr[0]/other], dtype=float)
            except:
                result.arr = ar([result.arr[0]/type(self)(other).arr[0]])
            return type(self)(result, tol=self.tol)
    
    def distance(self, other):
        if not isinstance(other, type(self)):
            other = type(self)(other)
        return Vector(self-other, tol=self.tol).norm
    
    def isclose(self, other, atol=1e-6, rtol=0):
        return np.isclose(self.distance(other), 0, atol=atol, rtol=rtol)
    
    @cached_property
    def _bug(self): return Point2D.from_array(self.arr[0])

class Vector(Point):
    __slots__ = ("arr", "tol")

    def __init__(self, *args, **kw) -> None:
        super().__init__(*args, **kw)
        
    def isclose(self, a, b, tol=None):
        if tol is None: tol = self.tol
        if isinstance(a, (list, tuple, np.ndarray)):
            return np.allclose(a,b,atol=tol,rtol=0)
        else:
            return np.isclose(a,b,atol=tol,rtol=0)
    
    @property
    def is_zero(self): return self.isclose(self.norm, 0)
    
    def is_parallel(self, other):
        if not isinstance(other, Vector):
            other = Vector(other, tol=self.tol).unit
        return self.isclose(self.unit.det(other), 0)
    
    def is_perpendicular(self, other):
        if not isinstance(other, Vector):
            other = Vector(other, tol=self.tol)
        return self.isclose(self.unit.dot(other.unit), 0)

    @cached_property
    def norm(self): return np.linalg.norm(self.arr)
    @cached_property
    def unit(self): return None if self.is_zero else Vector(self.arr/self.norm, tol=self.tol)
    @cached_property
    def normal(self): return Vector(self[1], -self[0], tol=self.tol) # points right
    
    def dot(self, other): return np.dot(self.arr[0], Vector(other, tol=self.tol).arr[0])
    def cross(self, other):
        other = Vector(other, tol=self.tol)
        return self[0]*other[1] - self[1]*other[0]
    def det(self, other): return self.cross(other)
    def cos(self, other): return self.unit.dot(Vector(other.unit))
    def sin(self, other): return self.unit.cross(Vector(other.unit))
    def angle(self, other): return np.arctan2(self.sin(other), self.cos(other))
    @cached_property
    def angle_from_x(self): return self.angle(Vector(1,0))
    
    def rotate(self, angle_ccw):
        """Rotate the 2D vector counterclockwise by angle_ccw (radians)."""
        c, s = np.cos(angle_ccw), np.sin(angle_ccw)
        R = ar([[c, -s],
                [s,  c]])
        return Vector(self.arr @ R, tol=self.tol)

    
    @cached_property
    def _bug(self): return Vector2D.from_array(self.arr[0])
    
class BaseLine(BaseShape):
    __slots__ = ("arr", "tol")
    def __init__(self, arr, tol=None):
        super().__init__(arr, tol=tol)
    @property
    def p1(self): return Point(self[0], tol=self.tol)
    @property
    def p2(self): return Point(self[1], tol=self.tol)
    @cached_property
    def vector(self): return Vector(self.p2-self.p1, tol=self.tol)
    @property
    def v(self): return self.vector
    @cached_property
    def normal(self): return self.vector.normal

class Line(BaseLine):
    __slots__ = ("arr", "tol")

    @overload
    def __init__(self, p1, p2, /, *, tol: float | None = ...) -> None: ...
    @overload
    def __init__(self, pts: Sequence[Number] | Mapping[str, Number], /, *, tol: float | None = ...) -> None: ...
    @overload
    def __init__(self, *, p1, p2, tol: float | None = ...) -> None: ...

    def __init__(self, *args, **kw) -> None:
        tol = kw.pop("tol", None)
        if "p1" in kw or "p2" in kw:
            p1, p2 = coerce_coords(kw.get("p1", 0.0)), coerce_coords(kw.get("p2", 0.0))
            
        elif len(args) >= 2:
            p1, p2 = coerce_coords(args[0]), coerce_coords(args[1])
        elif len(args) == 1:
            p1, p2 = coerce_coords(args[0])
        else:
            raise TypeError("Provide (p1,p2), pts=[p1,p2], mapping {'p1','p2'}, or p1=...,p2=...")
        arr = ar([p1, p2], dtype=float)
        super().__init__(arr, tol)
    
    @cached_property
    def length(self): return self.vector.norm
    
    def to_polygon(self, thickness):
        thickness = float(thickness)
        l_left = self.move(self.vector.unit.normal*thickness/2)
        l_left = Line(l_left.arr[::-1], tol=self.tol)
        l_right = self.move(-self.vector.unit.normal*thickness/2)
        return np.concatenate([l_left.arr, l_right.arr], dtype=float)
    
    @cached_property
    def to_ray(self):
        return Ray(self.p1, self.v, tol=self.tol)
    
    def does_point_touch(self, point, incl_boundary=True):
        if not isinstance(point, Point):
            point = Point(point, tol=self.tol)
        if np.allclose(self.p1.arr, point.arr, atol=self.tol, rtol=0) or np.allclose(self.p2.arr, point.arr, atol=self.tol, rtol=0):
            return incl_boundary
        return self._bug.distance_to_point(point) < 2*self.tol
    
    def does_line_touch(self, line, incl_boundary=True):
        if not isinstance(line, Line):
            line = Line(line, tol=self.tol)
        if line.does_point_touch(self.p1, True) or line.does_point_touch(self.p2, True) or self.does_point_touch(line.p1, True) or self.does_point_touch(line.p2, True):
            return incl_boundary
        return self._bug.distance_to_line(line) < 2*self.tol
    
    def remove_colinear_vertices(self, tol): return self
    
    @property
    def segments(self): return [self]
    
    @property
    def is_self_intersecting(self): return False
    
    @cached_property
    def _bug(self): return LineSegment2D.from_array(self.arr)
    
    def linspace(self, n, scale=1.0, half_shift=False, **kwargs):
        line = self.scale(scale) if not np.isclose(scale, 1.0) else self
        ts = np.linspace(0.0, 1.0, n, **kwargs)
        v = line.p2 - line.p1
        result = ar([line.p1 + t*v for t in ts], float)
        if half_shift:
            result = (result[1:] + result[:-1])/2
        return result
    
    def arange(self, d, scale=1.0, half_shift=False, **kwargs):
        L = self.length
        if L == 0:
            return [self.p1.copy()]
        n = int(L//d+1)
        return self.linspace(n, scale=scale, half_shift=half_shift, **kwargs)
    
    def scale(self, factor):
        c = (self.p1 + self.p2) * 0.5
        v = (self.p2 - self.p1) * 0.5 * factor
        return Line(c - v, c + v, tol=self.tol)


class Ray(BaseLine):
    __slots__ = ("arr", "tol")

    @overload
    def __init__(self, p1, v, /, *, tol: float | None = ...) -> None: ...
    @overload
    def __init__(self, pts: Sequence[Number] | Mapping[str, Number], /, *, tol: float | None = ...) -> None: ...
    @overload
    def __init__(self, *, p1, v, tol: float | None = ...) -> None: ...

    def __init__(self, *args, **kw) -> None:
        tol = kw.pop("tol", None)
        if "p1" in kw or "v" in kw:
            p1, v = coerce_coords(kw.get("p1", 0.0)), coerce_coords(kw.get("v", 0.0))
            
        elif len(args) >= 2:
            p1, v = coerce_coords(args[0]), coerce_coords(args[1])
        elif len(args) == 1:
            p1, v = coerce_coords(args[0]) 
        else:
            raise TypeError(f"Provide (p1,v), pts=[p1,v], mapping {'p1','v'}, or p1=...,v=...\n{args}")
        arr = ar([p1, p1+v], dtype=float)
        super().__init__(arr, tol)
        self.ray_arr = ar([p1, v], dtype=float)
    
    def point_is_on_right(self, point):
        point = Point(point)
        v1 = self.unit_vector
        v2 = Vector(point-self.p1)
        return v1.det(v2) < 0
    
    def is_parallel(self, other):
        if isinstance(other, Vector):
            return self.vector.is_parallel(other)
        elif not isinstance(other, Ray):
            other = Ray(other, tol=self.tol)
        return self.vector.is_parallel(other.vector)
    
    def is_perpendicular(self, other):
        if isinstance(other, Vector):
            return self.vector.is_perpendicular(other)
        elif not isinstance(other, Ray):
            other = Ray(other, tol=self.tol)
        return self.vector.is_perpendicular(other.vector)
    
    def intersection(self, other):
        if not isinstance(other, Ray):
            other = Ray(other, tol=self.tol)
        if self.is_parallel(other): return None
        p1, v1 = self.p1, self.vector.unit
        p2, v2 = other.p1, other.vector.unit
        alpha = (p2-p1).cross(v2)/(v1.cross(v2))
        return p1 + v1*alpha
    
    def __getitem__(self, key):
        return self.ray_arr[key]
    
    @cached_property
    def _bug(self): return Ray2D.from_array(self.ray_arr)

class PolyLine(BaseShape):
    
    def __new__(cls, *args, **kw):
        # Try to peek points; if we can't, fall back to normal construction
        try:
            tol = kw.get("tol", None)
            if "pts" in kw:
                pts = coerce_coords(kw["pts"])
            elif len(args) >= 2:
                pts = coerce_coords(args)
            elif len(args) == 1:
                pts = coerce_coords(args[0])
            else:
                return super().__new__(cls)
        except Exception:
            # If probing fails for any reason, construct as PolyLine
            return super().__new__(cls)
        if len(pts) < 3:
            try:
                return Line(*args, **kw)
            except:
                raise ValueError(f"{cls.__name__} cannot be initiated with fewer than 3 vertices")
            
        atol = 1e-6 if tol is None else float(tol)
        is_closed = pts.shape[0] >= 3 and np.allclose(pts[0], pts[-1], rtol=0.0, atol=atol)
        
        if is_closed and cls is PolyLine:
            if len(pts) < 4:
                raise ValueError("Ring cannot be initiated with fewer than 3 unique vertices")
            cls = Ring

        return super().__new__(cls)
    
    @overload
    def __init__(self, *pts, tol: float | None = ...) -> None: ...
    
    @overload
    def __init__(self, pts, /, *, tol: float | None = ...) -> None: ...
    
    def __init__(self, *args, **kw) -> None:
        tol = kw.pop("tol", None)
        if "pts" in kw:
            pts = coerce_coords(kw.get("pts", 0.0))
        elif len(args) >= 2:
            pts = coerce_coords(args)
        elif len(args) == 1:
            pts = coerce_coords(args[0])
        else:
            raise TypeError("Provide pts, pts=pts, mapping {'pts'}, or pts=...")
        arr = ar(pts, dtype=float)
        if np.allclose(arr[0], arr[-1], rtol=0, atol=tol or 1e-6):
            arr = arr[:-1]
        super().__init__(arr, tol)
    
    @cached_property
    def segments(self):
        return [Line(self[i], self[i+1]) for i in range(self.shape[0]-1)]
    @cached_property
    def direction_vectors(self):
        return [s.vector for s in self.segments]
    @cached_property
    def length(self): return sum([seg.vector.norm for seg in self.segments])
    
    @property
    def is_self_intersecting(self):
        """
        Checks if the polygon is self-intersecting (complex) using a vectorized 
        orientation test. Returns True if any non-adjacent edges intersect or touch.
        """
        # 1. Setup Vertices
        # P1 = start points, P2 = end points of all segments
        P1 = self.arr[:-1]
        P2 = self.arr[1:]
        N = len(P1)
        
        # A triangle cannot self-intersect
        if N < 4: 
            return False
        
        # 2. Iterate segments and verify against non-adjacent candidates
        # We loop 'i' (segment 1) and vector-check 'j' (segment 2 candidates)
        # Loop runs N-2 times, inner operations are vectorized.
        for i in range(N - 2):
            
            # --- Define Segment 1 (Scalar) ---
            A = P1[i]
            B = P2[i]
            
            # --- Define Segment 2 Candidates (Vector) ---
            # We must skip adjacent segments.
            # Adjacent to 'i' are 'i-1' (wraps) and 'i+1'.
            # So candidates start at i+2.
            # Special Case: If i=0, the last segment (N-1) connects back to 0, 
            # so it is adjacent and must be excluded.
            j_start = i + 2
            j_end = N - 1 if i == 0 else N
            
            if j_start >= j_end:
                continue
                
            C = P1[j_start:j_end]
            D = P2[j_start:j_end]
            
            # --- Optimization: Bounding Box Pruning ---
            # If bounding boxes don't overlap, intersection is impossible.
            # This filters out ~95%+ of cases in typical polygons cheaply.
            
            # Bounds of Segment 1
            min_s1 = np.minimum(A, B)
            max_s1 = np.maximum(A, B)
            
            # Bounds of Candidate Segments
            min_s2 = np.minimum(C, D)
            max_s2 = np.maximum(C, D)
            
            # Strict tolerance for floating point comparisons
            tol = 1e-9
            
            # Disjoint condition: Max < Min (with tolerance)
            disjoint = (
                (max_s1[0] < min_s2[:, 0] - tol) | 
                (min_s1[0] > max_s2[:, 0] + tol) |
                (max_s1[1] < min_s2[:, 1] - tol) | 
                (min_s1[1] > max_s2[:, 1] + tol)
            )
            
            # Keep only potential intersections
            mask = ~disjoint
            if not np.any(mask):
                continue
                
            C_test = C[mask]
            D_test = D[mask]
            
            # --- Intersection Test (Cross Product) ---
            # Two segments AB and CD intersect if and only if:
            # 1. C and D lie on opposite sides of line AB
            # 2. A and B lie on opposite sides of line CD
            
            # Test 1: CD relative to AB
            AB = B - A
            AC = C_test - A
            AD = D_test - A
            
            # 2D Cross Product (Determinant): x1*y2 - y1*x2
            cross_1 = AB[0] * AC[:, 1] - AB[1] * AC[:, 0]
            cross_2 = AB[0] * AD[:, 1] - AB[1] * AD[:, 0]
            
            # If signs differ (product <= 0), they span the line.
            # We use <= tol to catch touching edges (which counts as self-intersection)
            spans_1 = (cross_1 * cross_2) <= tol
            
            # If no candidates span line AB, we can skip Test 2
            if not np.any(spans_1):
                continue
            
            # Filter candidates for Test 2
            C_final = C_test[spans_1]
            D_final = D_test[spans_1]
            
            # Test 2: AB relative to CD
            CD = D_final - C_final
            CA = A - C_final
            CB = B - C_final
            
            cross_3 = CD[:, 0] * CA[:, 1] - CD[:, 1] * CA[:, 0]
            cross_4 = CD[:, 0] * CB[:, 1] - CD[:, 1] * CB[:, 0]
            
            spans_2 = (cross_3 * cross_4) <= tol
            
            # If any candidate passes both tests, we have an intersection
            if np.any(spans_2):
                return True
                
        return False
    
    def does_point_touch(self, point, incl_boundary=True):
        if not isinstance(point, Point):
            point = Point(point, tol=self.tol)
        if np.allclose(point.arr, self[0].arr, atol=self.tol, rtol=0) or np.allclose(point.arr, self[-1].arr, atol=self.tol, rtol=0):
            return incl_boundary
        return np.any([seg.does_point_touch(point, True) for seg in self.segments])
    
    def does_line_touch(self, line, incl_boundary=True):
        if not isinstance(line, Line):
            line = Line(line, tol=self.tol)
        if np.allclose(line.p1.arr, self[0].arr, atol=self.tol, rtol=0) or np.allclose(line.p1.arr, self[-1].arr, atol=self.tol, rtol=0):
            return incl_boundary
        if np.allclose(line.p2.arr, self[0].arr, atol=self.tol, rtol=0) or np.allclose(line.p2.arr, self[-1].arr, atol=self.tol, rtol=0):
            return incl_boundary
        
        return np.any([seg.does_line_touch(line, True) for seg in self.segments])
    
    @cached_property
    def non_intersecting_polylines_and_rings(self):
        if not self.is_self_intersecting:
            return [self]
        
        tol = self.tol
        polys = Polygon2D.from_array(self.arr).split_through_self_intersection
        results = []
        for poly in polys:
            results.append(Ring(ar(poly.to_array()), tol=tol))
        if not isinstance(self, Ring):
            for i, result in enumerate(results):
                ind1 = next((ind for ind, p in enumerate(result.arr) if np.allclose(p, self.p1, atol=tol, rtol=0)), None)
                if ind1 is None: continue
                ind2 = next((ind for ind, p in enumerate(result.arr) if np.allclose(p, self.p2, atol=tol, rtol=0)), None)
                if ind2 is None: continue
                start_ind = ind2 if (ind2 - ind1)%result.shape[0] == 1 else ind1
                results[i] = PolyLine(np.roll(result.arr, -start_ind, axis=0), tol=tol)
                break
        return results
    
    @classmethod
    def join_segments(cls, segs):
        arr = [seg.p1 for seg in segs]
        arr += segs[-1].p2
        return PolyLine(arr)
    
    def arange(self, d, scale=1.0, half_shift=False, **kwargs):
        result = [line.arange(d,scale,half_shift,**kwargs) for line in self.segments]
        result = [group if (i == 0 or half_shift) else group[1:] for i, group in enumerate(result)]
        result = [point for group in result for point in group]
        return ar(result)
    
    def arange_grouped(self, d, scale=1.0, half_shift=False, **kwargs):
        result = [line.arange(d,scale,half_shift,**kwargs) for line in self.segments]
        result = [group if (i == 0 or half_shift) else group[1:] for i, group in enumerate(result)]
        return result
    
    def plot(self, show=False, ax=None, point_color="black", line_color="black"):
        super().plot(show, ax, point_color, line_color)
    
    @cached_property
    def _bug(self): return Polyline2D.from_array(self.arr)

class Ring(PolyLine):
    
    def __init__(self, *args, **kw) -> None:
        super().__init__(*args, **kw)
    
    @cached_property
    def segments(self):
        return [Line(self[i], self[(i+1)%self.shape[0]], tol=self.tol) for i in range(self.shape[0])]
    @property
    def perimeter_segments(self): return self.segments
    @cached_property
    def _arr(self): return np.concatenate([self.arr, [self.arr[0]]], axis=0)
    
    def __getitem__(self, key): return self._arr[key]
    
    def does_point_touch(self, point, incl_boundary=True):
        if not isinstance(point, Point):
            point = Point(point, tol=self.tol)
        return np.any([seg.does_point_touch(point, incl_boundary) for seg in self.segments])
    
    def does_line_touch(self, line, incl_boundary=True):
        if not isinstance(line, Line):
            line = Line(line, tol=self.tol)
        
        return np.any([seg.does_line_touch(line, incl_boundary) for seg in self.segments])
    
    @property
    def length(self): return sum([seg.length for seg in self.segments])
    
    @cached_property
    def _bug(self): return Polyline2D.from_array(self._arr)

_register_bug_map({
    Point2D:    Point,
    Vector2D:   Vector,
    LineSegment2D: Line,
    Ray2D:      Ray,
    Polyline2D: PolyLine,
})    

    
if __name__ == "__main__":
    a = BaseShape([[0,0],[1,1], [5,3]])
    #print(a.bounding_points)
    a.plot(True, line_color="k")
    b = Point([0,6])
    c = Line([[0,0.5],[1,-3]])
    d = PolyLine([0,0],[1,1],[5,3])
    print(d.intersect_line_ray(c))