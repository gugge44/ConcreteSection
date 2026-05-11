import numpy as np
from numpy import array as ar, cos, sin, pi
from ladybug_geometry.geometry2d import Point2D, LineSegment2D, Polyline2D, Polygon2D, Ray2D
from functools import cached_property
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from sys import exit

from BugBasics import Point, Line, PolyLine, Ray, Vector, BaseShape, coerce_coords, Ring
from Polynomial import Polynomial

def remove_colinear(arr, tol=1e-6):
    single_vectors = np.roll(arr, -1, axis=0) - arr
    double_vectors = np.roll(arr, -2, axis=0) - arr
    norms = np.linalg.norm(double_vectors, axis=-1, keepdims=True)
    norms[norms == 0] = 1.0
    double_unit = double_vectors/norms
    single_vectors_projected = np.sum(single_vectors * double_unit, axis=-1, keepdims=True) * double_unit
    distance = np.linalg.norm(single_vectors_projected - single_vectors, axis=-1)
    mask = np.roll(distance >= tol, 1)
    return arr[mask]


def despike_polygon(verts, tol=50):
    if isinstance(verts, Polygon2D):
        verts = ar(verts.to_array(), float)

    V = ar(verts, float)
    n = len(V)
    if n <= 5:
        return V

    out = []
    for j in range(n):
        i = (j - 1) % n
        k = (j + 1) % n
        l = (j + 2) % n

        # skip if neighbours collapse (i ≈ k) → j is a spike tip between duplicates
        if np.allclose(V[i], V[k], atol=tol, rtol=0):
            continue
        # skip if this vertex duplicates either neighbour
        if np.allclose(V[j], V[l], atol=tol, rtol=0) or np.allclose(V[j], V[k], atol=tol, rtol=0):
            continue

        out.append(V[j])
    
    """if as_bug:
        if len(out) >= 3: return Polygon2D.from_array(out)
        else: return Polygon2D.from_array(V.tolist())"""
    # if everything got removed (shouldn't now), fall back to original
    return V if len(out) < 3 else ar(out, float)



def unique_points_from_array(arr):
    arr_unique = []
    for i in arr:
        unique = True
        for j in arr_unique:
            if np.allclose(i, j):
                unique = False
                break
        if unique:
            arr_unique.append(i)
    return ar(arr_unique, float)

def ensure_ccw(coords, tol=0.0):
    """Return coords ordered CCW. Accepts (n,2); closed or open ring."""
    a = np.asarray(coords, float)
    if a.ndim != 2 or a.shape[1] != 2 or a.shape[0] < 3:
        raise ValueError("coords must be (n,2) with n>=3")
    closed = np.allclose(a[0], a[-1])
    b = a[:-1] if closed else a  # compute area on unique vertices
    x, y = b[:, 0], b[:, 1]
    x2, y2 = np.roll(x, -1), np.roll(y, -1)
    area2 = np.dot(x, y2) - np.dot(y, x2)   # = 2 * signed area
    if area2 < -tol:                        # CW → flip
        a = a[::-1]
    return a

class SimplePolygon(BaseShape):
    def __init__(self, boundary, tol=1e-6, despike=False):
        if isinstance(boundary, Polygon):
            boundary = boundary.boundary[0]
        """new_boundary = []
        for i, b in enumerate(boundary):
            if i == 0:
                new_boundary.append(b)
            else:
                if not np.allclose(b, new_boundary[-1]):
                    new_boundary.append(b)"""
        
        #boundary = ar(new_boundary, float)  
        boundary = remove_colinear(ar(Polygon2D.from_array(boundary).remove_duplicate_vertices(tol*2).to_array(), float), tol=tol*2)
        if despike:
            boundary = despike_polygon(boundary, tol=tol*50)
        if np.allclose(boundary[0], boundary[-1]): boundary = boundary[:-1]
        try:
            super().__init__(ensure_ccw(coerce_coords(boundary)), tol=tol)
        except:
            try:
                super().__init__(ensure_ccw(coerce_coords(boundary[0])), tol=tol)
            except:
                raise ValueError(f"Cannot innitiate SimplePolygon with {boundary}")
        if np.allclose(self.arr[0], self.arr[-1], atol=tol, rtol=0):
            self.arr = self.arr[:-1]
        self._bug = Polygon2D.from_array(self.boundary)
    
    def polygon_relationship(self, other, tol=None):
        tol = self.tol if tol is None else float(tol)
        if not isinstance(other, SimplePolygon):
            other = SimplePolygon(other, tol=tol)
        return self._bug.polygon_relationship(other._bug, tolerance=tol)
    
    # Basic properties
    def __getitem__(self, key): return self.arr[key]
    
    def copy(self):
        return SimplePolygon(self.arr, tol=self.tol)
    
    @cached_property
    def simplified(self):
        new_arr = despike_polygon(self.boundary, tol=self.tol*50)
        return SimplePolygon(new_arr, tol=self.tol)
    
    @cached_property
    def bounding_rectangle(self):
        x1, y1, x2, y2 = self.bounds
        return SimplePolygon([[x1,y1],[x2,y1],[x2,y2],[x1,y2]], tol=self.tol)
    @property
    def x_min(self): return self.bounds[0]
    @property
    def y_min(self): return self.bounds[1]
    @property
    def x_max(self): return self.bounds[2]
    @property
    def y_max(self): return self.bounds[3]
    @property
    def H(self): return self.y_max-self.y_min
    @property
    def B(self): return self.x_max-self.x_min
    
    @cached_property
    def bl(self): return ar([self.bounds[0], self.bounds[1]], dtype=float)
    @cached_property
    def tr(self): return ar([self.bounds[2], self.bounds[3]], dtype=float)
    
    @cached_property
    def boundary(self): return self.arr
    
    @cached_property
    def closed_boundary(self):
        return np.concatenate([self.boundary, [self.boundary[0]]])
    
    @cached_property
    def perimeter_segments(self):
        res = np.stack([self.boundary, np.roll(self.boundary, -1, axis=0)], axis=1)
        return [Line(r, tol=self.tol) for r in res]
    
    @cached_property
    def perimeter_vectors(self):
        return self.perimeter_segments[:,1,:]-self.perimeter_segments[:,0,:]
    
    @cached_property
    def perimeter_unit_vectors(self):
        v = self.perimeter_vectors
        lengths = np.linalg.norm(v, axis=1)
        return np.divide(v, lengths[:, None], out=np.zeros_like(v), where=lengths[:, None]!=0)
    
    # Basic tests
    def is_point_inside(self, point, incl_boundary=True):
        if not isinstance(point, Point):
            point = Point(point, tol=self.tol)
        if self._bug.is_point_on_edge(point._bug, self.tol): return incl_boundary
        return self._bug.is_point_inside_check(point._bug)
    
    def is_line_inside(self, line, incl_boundary=True):
        if isinstance(line, (PolyLine, Ring)): return self.is_polyline_inside(line, incl_boundary)
        if not isinstance(line, Line):
            line = Line(line, tol=self.tol)
        
        if (not self.is_point_inside(line.p1, incl_boundary)) or (not self.is_point_inside(line.p2, incl_boundary)):
            return False
        if np.any([seg.does_line_touch(line, True) for seg in self.perimeter_segments]):
            return incl_boundary
        
        for seg in self.perimeter_segments:
            L2 = Line(seg, tol=self.tol)
            intersect = line.intersect_line_ray(L2)
            if intersect: return False
        return True
    
    def does_line_touch(self, line, incl_boundary=True, print_info=False, incl_split_test=True):
        if isinstance(line, (PolyLine, Ring)): return self.does_polyline_touch(line, incl_boundary, print_info)

        if not isinstance(line, Line):
            line = Line(line, tol=self.tol)
        if print_info:
            print("does_line_touch")
            #print(self.boundary)
            #print(line.arr)
        if self.is_point_inside(line.p1, incl_boundary) or self.is_point_inside(line.p2, incl_boundary) or self.is_point_inside(line.centroid, incl_boundary):
            if print_info: print("end point is inside")
            return True
        if incl_split_test:
            xs = np.linspace(line.p1[0], line.p2[0], 10)
            ys = np.linspace(line.p1[1], line.p2[1], 10)
            pts = [Point(x, y, tol=self.tol) for x, y in zip(xs, ys)]
            if np.any([self.is_point_inside(pt, incl_boundary) for pt in pts]):
                return True
        if self.is_line_inside(line, True):
            if print_info: print("whole line is inside")
            return True
        if len(self.intersect_line_ray(line))>0:
            if print_info: print("line intersects")
            return True
        
        
        return False
    
    def is_polyline_inside(self, polyline, incl_boundary=True):
        if isinstance(polyline, Line): return self.is_line_inside(polyline, incl_boundary)
        if not isinstance(polyline, (PolyLine, Ring)):
            polyline = PolyLine(polyline)
        return np.all([self.is_line_inside(line, incl_boundary) for line in polyline.segments])
    
    def does_polyline_touch(self, polyline, incl_boundary=True, print_info=False):
        if isinstance(polyline, Line): return self.does_line_touch(polyline, incl_boundary, print_info)
        if print_info: print("does_polyline_touch", type(polyline))
        if not isinstance(polyline, (PolyLine, Ring)):
            if print_info: print("changing type")
            polyline = PolyLine(polyline, tol=self.tol)
        if (not isinstance(polyline, Ring)) and (self.is_point_inside(polyline[0], incl_boundary) or self.is_point_inside(polyline[0], incl_boundary)):
            if print_info: print("checking end points")
            return True
        return np.any([self.does_line_touch(line, True, print_info) for line in polyline.segments])
    
    def line_intersect(self, line, include_boundary=True, tol=None):
        tol = self.tol if tol is None else tol
    
        # Polyline: concatenate per segment
        if isinstance(line, PolyLine):
            out = []
            for seg in line.segments:
                out.extend(self.line_intersect(seg, include_boundary=include_boundary, tol=tol))
            return out
    
        # Quick accepts/rejects
        if self.is_line_inside(line):
            return [line]
        if not self.does_line_touch(line):
            return []
    
        # Endpoints + boundary intersections
        pts = [line.p1, line.p2] + list(self.intersect_line_ray(line))
    
        # Dedup near-duplicates
        uniq = []
        for p in pts:
            if all(not np.allclose(p, q, atol=self.tol, rtol=0) for q in uniq):
                uniq.append(p)
        pts = uniq
        
    
        # Sort along the line using projected parameter t
        v = line.vector  # assume non-zero
        def t_of(p): return v.dot(Vector(p - line.p1, tol=tol))
        pts.sort(key=t_of)
        
    
        # Build candidate sub-spans
        spans = [(pts[i], pts[i+1]) for i in range(len(pts)-1)]
    
        # Keep only spans whose midpoint is inside (or on boundary if requested)
        kept = []
        for a, b in spans:
            if a.distance_to_point(b) <= tol:
                continue
            m = (a + b) * 0.5
            inside = self.is_point_inside(m)
            if inside:
                kept.append((a, b))
    
        if not kept:
            return []
        # Merge contiguous kept spans (touching endpoints)
        merged = []
        cur_a, cur_b = kept[0]
        for a, b in kept[1:]:
            if np.allclose(cur_b, a, atol=self.tol, rtol=0):  # contiguous
                cur_b = b
            else:
                merged.append((cur_a, cur_b))
                cur_a, cur_b = a, b
        merged.append((cur_a, cur_b))
    
        # Build Line objects
        return [Line([a, b], tol=tol) for a, b in merged]
    
    def line_difference(self, line, include_boundary=True, tol=None):
        tol = self.tol if tol is None else tol
    
        # Polyline: concatenate per segment
        if isinstance(line, PolyLine):
            out = []
            for seg in line.segments:
                out.extend(self.line_difference(seg, include_boundary=include_boundary, tol=tol))
            return out
    
        # Quick accepts/rejects
        if self.is_line_inside(line):
            return []  # fully inside -> nothing outside
        if not self.does_line_touch(line):
            return [line]  # fully outside -> keep entire line
    
        # Endpoints + boundary intersections
        pts = [line.p1, line.p2] + list(self.intersect_line_ray(line))
    
        # Dedup near-duplicates
        uniq = []
        for p in pts:
            if all(not np.allclose(p, q, atol=tol, rtol=0) for q in uniq):
                uniq.append(p)
        pts = uniq
    
        # Sort along the line using projected parameter t
        v = line.vector
        def t_of(p): return v.dot(Vector(p - line.p1, tol=tol))
        pts.sort(key=t_of)
    
        # Build candidate sub-spans
        spans = [(pts[i], pts[i+1]) for i in range(len(pts)-1)]
    
        # Keep only spans whose midpoint is outside (or boundary-excluded if requested)
        kept = []
        for a, b in spans:
            if a.distance_to_point(b) <= tol:
                continue
            m = (a + b) * 0.5
            inside = self.is_point_inside(m)
            if not inside:
                kept.append((a, b))
    
        if not kept:
            return []
    
        # Merge contiguous kept spans
        merged = []
        cur_a, cur_b = kept[0]
        for a, b in kept[1:]:
            if np.allclose(cur_b, a, atol=tol, rtol=0):
                cur_b = b
            else:
                merged.append((cur_a, cur_b))
                cur_a, cur_b = a, b
        merged.append((cur_a, cur_b))
    
        return [Line([a, b], tol=tol) for a, b in merged]

    
    # Geometric properties
    
    def area_integral(self, poly, centroid=None):
        if centroid is None: centroid = self.centroid
        #else: centroid = np.asarray(centroid)
        lines = ar([[b[0]-centroid, b[1]-centroid] for b in self.perimeter_segments])
        poly = Polynomial(poly).integrate('x')
        
        result = 0
        for line in lines:
            p1, p2 = line
            x_fun = Polynomial({(0,): p1[0], (1,): p2[0]-p1[0]})
            if poly.ndim != 1:
                y_fun = Polynomial({(0,): p1[1], (1,): p2[1]-p1[1]})
                local_poly = poly.transform([y_fun, x_fun]).integrate('x')
            else: local_poly = poly.transform([x_fun]).integrate('x')
            result += local_poly.evaluate([1])*(p2[1]-p1[1])
        return result
    
    @cached_property
    def signed_area(self):
        #return self.area_integral([1])
        x = self.arr[:, 0]
        y = self.arr[:, 1]
        return 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)
    
       
    @property
    def area(self): return abs(self.signed_area)
    
    @cached_property
    def centroid(self):
        A = self.signed_area
        if abs(A) < 1e-12:
            return np.mean(self.arr, axis=0)
        
        x = self.arr[:, 0]
        y = self.arr[:, 1]
        x_next = np.roll(x, -1)
        y_next = np.roll(y, -1)
        
        cross = x * y_next - x_next * y
        
        cx = np.sum((x + x_next) * cross) / (6 * A)
        cy = np.sum((y + y_next) * cross) / (6 * A)
        
        return Point([cx, cy], tol=self.tol)
    
    def second_moment(self, centroid=None):
        I_yy = self.area_integral([[0],[0],[1]], centroid)
        I_zz = self.area_integral([[0,0,1]], centroid)
        I_zy = self.area_integral([[0, 0],[0, 1]], centroid)
        return ar([[I_zz, I_zy],[I_zy, I_yy]])
    
    def radial_moment(self, centroid=None):
        return self.area_integral([[0,0,1],[0,0,0],[1,0,0]], centroid)
    
    def first_moment_base(self):
        i_yy = self.area_integral([[0],[1]], self.bl)
        i_zz = self.area_integral([[0,1]], self.bl)
        return ar([i_zz, i_yy])
        
    def first_moment_top(self):
        i_yy = self.area_integral([[0],[1]], self.tr)
        i_zz = self.area_integral([[0,1]], self.tr)
        return ar([i_zz, i_yy])
    
    def first_moment(self, centroid=None):
        if centroid is None: centroid = self.centroid
        i_yy = self.area_integral([[0],[1]], centroid)
        i_zz = self.area_integral([[0,1]], centroid)
        return ar([i_zz, i_yy])
    
    def elastic_modulus(self, centroid=None):
        if centroid is None: centroid = self.centroid
        I = self.second_moment(centroid)
        W_zz = I[0,0]/(max(self.x_max-centroid[0], centroid[0]-self.x_min))
        W_yy = I[1,1]/(max(self.y_max-centroid[1], centroid[1]-self.y_min))
        return ar([W_zz, W_yy])
    
    def plastic_modulus(self, centroid=None, compression_only = False):
        if centroid is None: centroid = self.centroid
        elif not isinstance(centroid, Point):
            centroid = Point(centroid)
        poly = self.split_by_ray(Ray([centroid, [1, 0]]))[0] if compression_only else self.copy()
        if compression_only or centroid[1] >= poly.y_max or centroid[1] <= poly.y_min:
            return abs(poly.first_moment(centroid)[1])
        top, btm = poly.split_by_ray(Ray(centroid, Vector([1,0])))
        return abs(btm.first_moment(centroid)[1]) + abs(top.first_moment(centroid)[1])
    
    def elasto_plastic_modulus(self, centroid=None, yield_distance=0, compression_only=False):
        if centroid is None: centroid = self.centroid
        elif not isinstance(centroid, Point):
            centroid = Point(centroid)
        poly = self.split_by_ray(Ray([centroid, [1, 0]]))[0] if compression_only else self.copy()
        z_top = centroid[1] + yield_distance
        z_btm = centroid[1] - yield_distance
        if z_top >= poly.y_max:
            if z_btm <= poly.y_min:
                return poly.elastic_modulus(centroid)
            else:
                elastic, plastic = poly.split_by_ray(Ray([0, z_top], [1,0]))
                return abs(elastic.elastic_modulus(centroid)[1]) + abs(plastic.first_moment(centroid)[1])
        else:
            if z_btm <= poly.y_min:
                plastic, elastic = poly.split_by_ray(Ray([0, z_btm], [1,0]))
                return abs(elastic.elastic_modulus(centroid)[1]) + abs(plastic.first_moment(centroid)[1])
            else:
                base, plastic1 = poly.split_by_ray(Ray([0, z_top], [1,0]))
                plastic2, elastic = base.split_by_ray(Ray([0, z_btm], [1,0]))
                return abs(elastic.elastic_modulus(centroid)[1]) + abs(plastic1.first_moment(centroid)[1]) + abs(plastic2.first_moment(centroid)[1])
        
    # Geometric operations
    
    def split_by_ray(self, ray):
        if not isinstance(ray, Ray):
            ray = Ray(ray, tol=self.tol)
        rect = self.bounding_rectangle
        intersects = rect.intersect_line_infinite(ray)

        intersects = unique_points_from_array(intersects)
        
        if intersects is None or len(intersects) < 2:
            v1 = ray.vector.unit
            v2 = self.centroid - ray.p1
            return (self.copy(), None) if v1.det(v2) > 0 else (None, self.copy())
        
        # create a max-length line from the intersection points
        dists = [np.linalg.norm(p-intersects[0]) for p in intersects]
        ind1 = np.argmin(dists)
        ind2 = np.argmax(dists)
        intersects = [intersects[ind1], intersects[ind2]]
        intersects = Line(intersects)
        
        # create a max dimension to ensure that the cutting rectangle covers the shape
        max_dist = max(rect.x_max - rect.x_min, rect.y_max - rect.y_min)*10
        # create outer points for the cutting rectangle based on a normal distance from the cutting plane
        # the unit normal is (y, -x) which always points right from the perspective of the original vector
        outer = intersects + (float(max_dist)*intersects.normal.unit)
        side_rect = SimplePolygon([intersects[0], intersects[1], outer[1], outer[0]])
        # the left side is the difference with the right side rectangle and the right is the intersect
        return self.boolean_difference(side_rect), self.boolean_intersect(side_rect)
    
    def move(self, vector):
        if not isinstance(vector, Vector):
            vector = Vector(vector, tol=self.tol)
        boundary = self.boundary + vector.arr[None, ...]
        return SimplePolygon(boundary[0], tol=self.tol)
    
    def rotate(self, angle_ccw, centroid=None):
        if centroid is None:
            centroid = self.centroid
        elif not isinstance(centroid, Point):
            centroid = Point(centroid, tol=self.tol)
    
        theta = float(angle_ccw)*pi/180
        c, s = cos(theta), sin(theta)
        R = ar([[c, -s],
                [s,  c]])                # CCW rotation
    
        boundary = self.boundary.copy()   # (N,2)
        rel = boundary - centroid.arr[None, ...]  # (N,2)
        rot = rel @ R                     # (N,2)
        new_boundary = rot + centroid.arr[None, ...]
        return SimplePolygon(new_boundary[0], tol=self.tol)
    
    def scale(self, factor, centroid=None):
        if centroid is None:
            centroid = self.centroid
        elif not isinstance(centroid, Point):
            centroid = Point(centroid, tol=self.tol)
        
        factor = float(factor)
        boundary = ar(self.boundary, float)   # (N,2)
        rel = boundary - centroid[None, ...]  # (N,2)
        scl = rel*factor
        new_boundary = scl + centroid.arr[None, ...]

        try:
            return SimplePolygon(new_boundary[0], tol=self.tol)
        except:
            return None
    
    def scale_per_segment(self, outward_distance, miter_limit=10.0):
        segments = self.perimeter_segments
        pts = self.arr.copy()
        d = -float(outward_distance)
        eps = self.tol
    
        # unit outward normals for each edge
        n = [(-seg.vector.normal.unit) for seg in segments]  # keep your sign convention consistent
        
        out = []
        for i, p in enumerate(pts):
            n_prev = n[(i - 1) % len(pts)]
            n_next = n[i]
    
            b = n_prev + n_next
            denom = 1.0 + float(n_prev.dot(n_next))  # == b·n_prev == b·n_next
    
            # near-180° (or degenerate) -> fall back
            if abs(denom) < eps or b.norm < eps:
                #print(f"{abs(denom) = }, {b.length = }, {b = }")
                dp = d * n_next
            else:
                dp = d * b / denom
    
                # optional: cap extreme miters at very acute angles
                if dp.length > miter_limit * abs(d):
                    dp = (dp.unit * (miter_limit * abs(d))) if dp.length > eps else (d * n_next)
    
            out.append(p + dp)
    
        return SimplePolygon(out, tol=self.tol)

    
    @staticmethod
    def create_circle(centre, radius, n=10, tol=None):
        if not isinstance(centre, Point):
            centre = Point(centre, tol=tol)
        thetas = np.linspace(0, 2*pi, n, endpoint=False)
        pts = ar([centre.x, centre.y], float)[None, ...] + radius * ar([[cos(theta), sin(theta)] for theta in thetas])
        return SimplePolygon(pts, tol=tol)

    
    def expand_with_rounded_corners(self, radius):
        radius = float(radius)
        circles = [self.create_circle(p, radius, 10, self.tol) for p in self.arr]
        normals = [-seg.normal.unit*radius for seg in self.segments]
        polys = [self.move(n) for n in normals]
        return self.boolean_union_all([self.scale(1+radius/10)]+polys+circles)
    
    def transform(self, vector=None, angle_ccw=0, scale_factor=1, centroid=None):
        poly = self
        if angle_ccw:
            poly = poly.rotate(angle_ccw, centroid)
        if not np.isclose(scale_factor, 1.0):
            poly = poly.scale(scale_factor, centroid)
        return poly if vector is None else poly.move(vector)

  
    # Boolean ops
    def _boolean_op(self, other, method_name: str, tol=None):
        tol_eff = max(self.tol, getattr(other, "tol", self.tol)) if tol is None else float(tol)
        if not isinstance(other, SimplePolygon):
            other = SimplePolygon(other, tol=self.tol)
    
        bug_method = getattr(self._bug, method_name)  # no pre-simplify here
        try:
            op = bug_method(other._bug, tol_eff)       # let the kernel do its work
        except Exception:
            ax = other.plot(False); ax = self.plot(False, ax=ax); plt.show()
            raise
    
        if not op:
            return None
        if len(op) == 1:
            return SimplePolygon(op[0], tol=self.tol).simplified  # simplify AFTER
    
        groups = Polygon2D.group_boundaries_and_holes(op, tol_eff)
        bounds = [g[0] for g in groups if g]
        holes  = [h for g in groups for h in (g[1:] if len(g) > 1 else [])]
        return Polygon(bounds, holes, tol=self.tol)

    """def _boolean_op(self, other, method_name: str, tol=None):
        # normalize 'other' to SimplePolygon
        if tol is None:
            tol = self.tol*2
        else:
            tol = float(tol)
        if not isinstance(other, SimplePolygon):
            other = SimplePolygon(other, tol=self.tol)
    
        # call the instance method on the underlying ladybug object
        bug_method = getattr(self._bug.remove_colinear_vertices(10), method_name)          # e.g. self._bug.boolean_union
        try:
            op = bug_method(other._bug.remove_colinear_vertices(10), tol)                  # expected signature: (other, tol)
        except:
            ax = None
            ax = other.plot(False)
            ax = self.plot(False, ax=ax)
            plt.show()
            print(self)
            print(other)
            print(tol)
            exit()
        # normalize outputs
        if not op:                                            # None or []
            return None
        if len(op) == 1:
            return SimplePolygon(op[0], tol=self.tol)
    
        groups = Polygon2D.group_boundaries_and_holes(op, tol)
        bounds = [g[0] for g in groups if g]
        holes  = [h for g in groups for h in (g[1:] if len(g) > 1 else [])]
        return Polygon(bounds, holes, tol=self.tol)"""
    
    def boolean_union(self, other, tol=None):
        return self._boolean_op(other, "boolean_union", tol=tol)
    def boolean_intersect(self, other, tol=None):
        return self._boolean_op(other, "boolean_intersect", tol=tol)
    def boolean_difference(self, other, tol=None):
        return self._boolean_op(other, "boolean_difference", tol=tol)
    
    @staticmethod
    def boolean_union_all(arr, tol=None):
        
        lst = [val if isinstance(val, SimplePolygon) else SimplePolygon(val) for val in arr]
        bug_lst = [val._bug for val in lst]
        if tol is None:
            tol = lst[0].tol*2
        else:
            tol = float(tol)
        bug_union = Polygon2D.boolean_union_all(bug_lst, tol)
        bug_group = Polygon2D.group_boundaries_and_holes(bug_union, tol)
        bounds = [g[0] for g in bug_group if g]
        holes  = [h for g in bug_group for h in (g[1:] if len(g) > 1 else [])]
        return Polygon(bounds, holes, tol=lst[0].tol)
        
            
    # creation
    @classmethod
    def from_polyline(cls, polyline, thickness, align="c"):
        tol = polyline.tol
        polyline = polyline.remove_colinear_vertices(tol)
        if isinstance(polyline, Ring):
            return cls.from_ring(polyline, thickness, align)
        elif not isinstance(polyline, PolyLine) or len(polyline) == 2:
            if isinstance(polyline, Line):
                norm = polyline.vector.unit.normal
                t = thickness
                if align in ["r", "right"]:
                    p1, p2 = polyline.p1, polyline.p2
                    p3, p4 = p2 - norm*t, p1 - norm*t
                elif align in ["l", "left"]:
                    p1, p2 = polyline.p1, polyline.p2
                    p3, p4 = p2 + norm*t, p1 + norm*t
                else:
                    p1 = polyline.p1 + norm*t/2
                    p2 = polyline.p2 + norm*t/2
                    p3 = polyline.p2 - norm*t/2
                    p4 = polyline.p1 - norm*t/2

                return SimplePolygon([p1, p2, p3, p4])
            polyline = PolyLine(polyline)
        
        
        
        thickness = float(thickness)
        if align in ["r", "right"]:
            left_side = polyline.offset(thickness).remove_colinear_vertices(tol)
            right_side = polyline.copy()
        elif align in ["l", "left"]:
            left_side = polyline.copy()
            right_side = polyline.offset(-thickness).remove_colinear_vertices(tol)
        else:
            left_side = polyline.offset(thickness/2).remove_colinear_vertices(tol)
            right_side = polyline.offset(-thickness/2).remove_colinear_vertices(tol)
        sides = [left_side, right_side]
        for s, side in enumerate(sides):
            if side.is_self_intersecting:
                print("inter")
                print(side)
                print(type(side))
                exit()
                segs = side.segments
                for i in range(len(segs)):
                    seg1, seg2 = segs[i], segs[(i+1)%len(segs)]
                    intersect = seg1.intersect_line_ray(seg2)
                    if intersect is None: continue
                    if intersect != seg1.p2:
                        segs[i] = Line(seg1.p1, intersect, tol=tol)
                    if intersect != seg2.p1:
                        segs[(i+1)%len(segs)] = Line(intersect, seg2.p2, tol=tol)
                sides[s] = PolyLine.join_segments(segs)
        
        left_side, right_side = sides
        is_ring = [isinstance(side, Ring) for side in sides]
        if np.any(is_ring):
            return Polygon.from_ring(Ring(polyline, tol=tol), thickness)
        else:
            arr = np.concatenate([left_side.arr[::-1], right_side.arr], axis=0).astype(float)
            try:
                return SimplePolygon(arr, tol=tol)
            except:
                print(arr)
                print(polyline)
                exit()
    
    @classmethod
    def from_ring(cls, polyline, thickness, align="c"):
        if not isinstance(polyline, Ring):
            return cls.from_polyline(polyline, thickness, align)

        tol = polyline.tol
        thickness = float(thickness)
        if align in ["r", "right"]:
            left_side = polyline.offset(thickness).remove_colinear_vertices(tol)
            right_side = polyline.copy()
        elif align in ["l", "left"]:
            left_side = polyline.copy()
            right_side = polyline.offset(-thickness).remove_colinear_vertices(tol)
        else:
            left_side = polyline.offset(thickness/2).remove_colinear_vertices(tol)
            right_side = polyline.offset(-thickness/2).remove_colinear_vertices(tol)
        #left_side = polyline.offset(thickness/2).remove_colinear_vertices(tol)
        #right_side = polyline.offset(-thickness/2).remove_colinear_vertices(tol)
        sides = [left_side, right_side]
        for s, side in enumerate(sides):
            if side.is_self_intersecting:
                segs = side.segments
                for i in range(len(segs)):
                    seg1, seg2 = segs[i], segs[(i+1)%len(segs)]
                    intersect = seg1.intersect_line_ray(seg2)
                    if intersect is None: continue
                    if intersect != seg1.p2:
                        segs[i] = Line(seg1.p1, intersect, tol=tol)
                    if intersect != seg2.p1:
                        segs[(i+1)%len(segs)] = Line(intersect, seg2.p2, tol=tol)
                sides[s] = PolyLine.join_segments(segs)

        right = SimplePolygon(right_side, tol=tol)
        left = SimplePolygon(left_side, tol=tol)
        outer = left if left.area > right.area else right
        inner = left if left.area < right.area else right
        return outer.boolean_difference(inner)
    
    # Plotting
    
    def plot(self, show=True, ax=None, boundary_color="black", fill_color="#cce6ff"):
        
        if ax is None:
            fig, ax = plt.subplots()
    
        if fill_color is not None or boundary_color is not None:
            patch = MplPolygon(
                self.boundary,
                closed=True,
                facecolor=fill_color if fill_color is not None else "none",
                edgecolor=boundary_color if boundary_color is not None else "none",
                linewidth=1.0,
            )
            ax.add_patch(patch)
    
        ax.set_aspect("equal", adjustable="box")
        ax.autoscale_view()
        if show:
            plt.show()
        return ax

class Polygon:
    def __init__(self, boundaries, holes=None, tol=1e-6, repair=True):
        self.tol = float(tol)
        
        try:
            self.boundary = [SimplePolygon(p, tol=self.tol) for p in boundaries]
        except:
            try:
                self.boundary = [SimplePolygon(boundaries, tol=self.tol)]
            except:
                raise ValueError(f"{boundaries} {tol}")
        if holes:
            try:
                self.holes = [SimplePolygon(p, tol=self.tol) for p in holes]
            except:
                self.holes = [SimplePolygon(holes, tol=self.tol)]
        else:
            self.holes = []
        # Repair once to handle overlaps, new islands/holes, etc.
        self._is_repaired = False
        if repair:
            self._repair()
            self._is_repaired = True
    
    def ensure_repaired(self):
        if not getattr(self, "_is_repaired", False):
            self._repair()
            self._is_repaired = True
        return self
    
    def copy(self):
        return Polygon([SimplePolygon(p, tol=self.tol) for p in self.boundary], [SimplePolygon(h, tol=self.tol) for h in self.holes], tol=self.tol, repair=False)
    
    @property
    def H(self): return self.bounds[3]-self.bounds[1]
    @property
    def B(self): return self.bounds[2]-self.bounds[0]
    
    def add_hole_fast(self, hole):
        if not isinstance(hole, SimplePolygon):
            hole = SimplePolygon(hole, tol=self.tol)
        self.holes.append(hole)
    
    @cached_property
    def simplified(self):
        new_item = self.copy()
        new_item.boundary = [b.simplified for b in new_item.boundary]
        new_item.holes = [b.simplified for b in new_item.holes]
        return new_item
    # Basic Properties
    @cached_property
    def bounds(self):
        bounds_list = ar([p.bounds for p in self.boundary])
        min_b = np.min(bounds_list[:,:2], axis=0)
        max_b = np.max(bounds_list[:,2:], axis=0)
        return ar([min_b[0], min_b[1], max_b[0], max_b[1]])
    @cached_property
    def bounding_rectangle(self):
        x1, y1, x2, y2 = self.bounds
        return SimplePolygon([[x1,y1],[x2,y1],[x2,y2],[x1,y2]], tol=self.tol)
    @property
    def x_min(self):
        return min([p.x_min for p in self.boundary])
    @property
    def y_min(self):
        return min([p.y_min for p in self.boundary])
    @property
    def x_max(self):
        return max([p.x_max for p in self.boundary])
    @property
    def y_max(self):
        return max([p.y_max for p in self.boundary])
    
    @property
    def perimeter_segments(self):
        seg_groups = [b.perimeter_segments for b in self.boundary]
        seg_groups.extend([h.perimeter_segments for h in self.holes])
        return [seg for segs in seg_groups for seg in segs]
    
    # Intersection tests
        
    def is_point_inside(self, point):
        if not np.any([b.is_point_inside(point) for b in self.boundary]):
            return False
        for hole in self.holes:
            if hole.is_point_inside(point):
                return False
        return True
    
    def area_integral(self, poly, centroid=None):
        if centroid is None:
            centroid = self.centroid
        return sum([b.area_integral(poly, centroid) for b in self.boundary]) - sum([h.area_integral(poly, centroid) for h in self.holes])
    
    def is_line_inside(self, line, incl_boundary=True):
        if not np.any([b.is_line_inside(line, incl_boundary) for b in self.boundary]):
            return False
        for hole in self.holes:
            if hole.does_line_touch(line, ~incl_boundary):
                return False
        return True
    
    def does_line_touch(self, line, incl_boundary=True, print_info=False):
        if print_info: print(self)
        if not np.any([b.does_line_touch(line, incl_boundary, print_info) for b in self.boundary]):
            return False
        for hole in self.holes:
            if hole.is_line_inside(line, ~incl_boundary):
                return False
        return True
    
    def does_polyline_touch(self, polyline, incl_boundary=True, print_info=False):
        if isinstance(polyline, Line): return self.does_line_touch(polyline, incl_boundary, print_info)
        if print_info: print("does_polyline_touch", type(polyline))
        if not isinstance(polyline, (PolyLine, Ring)):
            if print_info: print("changing type")
            polyline = PolyLine(polyline, tol=self.tol)
        if (not isinstance(polyline, Ring)) and (self.is_point_inside(polyline[0]) or self.is_point_inside(polyline[0])):
            if print_info: print("checking end points")
            return True
        return np.any([self.does_line_touch(line, True, print_info) for line in polyline.segments])
    
    def polygon_relationship_inv(self, other, tol=None):
        if tol is None: tol = self.tol
        if isinstance(other, Polygon): raise ValueError("cannot carry out polygon_relationship_inv on a Polygon, it must be SimplePolygon")
        if not isinstance(other, SimplePolygon):
            other = SimplePolygon(other, tol=tol)
        outer_rel = ar([other.polygon_relationship(b, tol) for b in self.boundary], int)
        if np.any(outer_rel == 0): return 0
        if np.all(outer_rel == -1): return -1
        #inner_rel = ar([b.polygon_relationship(other, tol) for b in self.holes], int)
        #if np.any(inner_rel == 1): return -1
        #if np.any(inner_rel == 0): return 0
        if np.all(outer_rel == 1): return 1
        return -1
    
    def polygon_relationship(self, other, tol=None):
        if tol is None: tol = self.tol
        if not isinstance(other, Polygon):
            if not isinstance(other, SimplePolygon):
                other = SimplePolygon(other, tol=tol)
            outer_rel = ar([b.polygon_relationship(other, tol) for b in self.boundary], int)
            if np.any(outer_rel == 0): return 0
            if np.all(outer_rel == -1): return -1
            inner_rel = ar([b.polygon_relationship(other, tol) for b in self.holes], int)
            if np.any(inner_rel == 1): return -1
            if np.any(inner_rel == 0): return 0
            if np.all(outer_rel == 1): return 1
            if np.any(outer_rel == 1): return 0
            return -1
        else:
            outer_rel = ar([self.polygon_relationship(b, tol) for b in other.boundary], int)
            if np.any(outer_rel == 0): return 0
            if np.all(outer_rel == -1): return -1
            inner_rel = ar([self.polygon_relationship_inv(b, tol) for b in other.holes], int)
            if np.any(inner_rel == 0): return 0
            if np.any(inner_rel == 1): return -1
            if np.any(outer_rel == 1): return 1
            return -1
        
    def line_intersect(self, line):
        # --- Handle PolyLine recursively ---
        if isinstance(line, PolyLine):
            out = []
            for seg in line.segments:
                out.extend(self.line_intersect(seg))
            return out
    
        # --- Intersect with outer boundaries ---
        lines = []
        for b in self.boundary:
            lines.extend(b.line_intersect(line))
    
        # --- Subtract holes from each intersected line segment ---
        for hole in self.holes:
            new_lines = []
            for l in lines:
                # line_difference returns list of remaining (non-intersecting) segments
                new_lines.extend(hole.line_difference(l))
            lines = new_lines
    
        return lines
    
    def line_difference(self, line, include_boundary=True, tol=None):
        tol = self.tol if tol is None else tol
    
        # --- Handle PolyLine recursively ---
        if isinstance(line, PolyLine):
            out = []
            for seg in line.segments:
                out.extend(self.line_difference(seg, include_boundary=include_boundary, tol=tol))
            return out
    
        # --- Start with the full line as "outside" candidate ---
        lines = [line]
    
        # --- Subtract intersections with outer boundaries (clip out interior parts) ---
        for b in self.boundary:
            new_lines = []
            for l in lines:
                # Remove parts inside boundary
                new_lines.extend(b.line_difference(l, include_boundary=include_boundary, tol=tol))
            lines = new_lines
    
        # --- Add back the regions that lie inside any holes ---
        for hole in self.holes:
            new_lines = []
            for l in lines:
                # Inside a hole = outside of polygon -> keep intersection part
                new_lines.extend(hole.line_intersect(l, tol=tol))
            lines.extend(new_lines)
    
        return lines

    
    
    # Geometric properties
    @cached_property
    def area(self):
        return sum([p.area for p in self.boundary]) - sum([h.area for h in self.holes])
    
    @cached_property
    def centroid(self):
        prod = np.sum([p.area*p.centroid for p in self.boundary], axis=0)
        prod -= np.sum([h.area*h.centroid for h in self.holes], axis=0)
        mass = sum([p.area for p in self.boundary])
        mass -= sum([h.area for h in self.holes])
        return Point(prod/mass, tol=self.tol)
    
    def second_moment(self, centroid=None):
        if centroid is None:
            centroid = self.centroid
        Is = np.sum([p.second_moment(centroid) for p in self.boundary], axis=0)
        Is -= np.sum([h.second_moment(centroid) for h in self.holes], axis=0)
        return Is
    
    def first_moment(self, centroid=None):
        if centroid is None:
            centroid = self.centroid
        Is = np.sum([p.first_moment(centroid) for p in self.boundary], axis=0)
        Is -= np.sum([h.first_moment(centroid) for h in self.holes], axis=0)
        return Is
    
    def first_moment_base(self):
        return self.first_moment([self.x_min, self.y_min])
    
    def first_moment_top(self):
        return self.first_moment([self.x_max, self.y_max])
    
    def elastic_modulus(self, centroid=None):
        if centroid is None:
            centroid = self.centroid
        I = self.second_moment(centroid)
        W_zz = I[0,0]/(max(self.x_max-centroid[0], centroid[0]-self.x_min))
        W_yy = I[1,1]/(max(self.y_max-centroid[1], centroid[1]-self.y_min))
        return ar([W_zz, W_yy])
    
    # Boolean ops
    def _boolean_union_simple(self, other, tol=None):
        tol = self.tol if tol is None else float(tol)
        if not isinstance(other, SimplePolygon):
            other = SimplePolygon(other, tol=self.tol)
        
        union_poly = SimplePolygon.boolean_union_all(self.boundary+[other], tol=tol)
        holes = union_poly.holes
        boundary = union_poly.boundary
        for hole in self.holes:
            rel = other._bug.polygon_relationship(hole._bug, tol)
            if rel == 1:
                continue
            elif rel == -1:
                holes.append(hole)
            else:
                new_holes = hole.boolean_difference(other, tol=tol)
                holes.extend(new_holes.boundary)
        return Polygon(boundary, holes, tol=self.tol)

    
    def _boolean_intersect_simple(self, other, tol=None):
        tol = self.tol if tol is None else float(tol)
        if not isinstance(other, SimplePolygon):
            other = SimplePolygon(other, tol=self.tol)
        bounds = [b.copy() for b in self.boundary]
        new_bounds = []
        new_holes = [h.copy() for h in self.holes]
        for b in bounds:
            new_bound = b._bug.boolean_intersect(other._bug, tolerance=tol)
            if new_bound is None or len(new_bound) == 0: continue
            
            groups = Polygon2D.group_boundaries_and_holes(new_bound, tol)
            
            #if isinstance(new_bound, SimplePolygon):
            #    new_bounds.append(new_bound)
            #elif new_bound is None:
            #    continue
            #else:
            new_bounds.extend([ar(g[0].to_array(), float) for g in groups])
            new_holes.extend([g_ for g in groups for g_ in g[1:] if len(g) > 1])

        return Polygon(new_bounds, new_holes, tol=self.tol) if len(new_bounds) > 0 else None
    
    def _boolean_difference_simple(self, other, tol=None):
        tol = self.tol if tol is None else float(tol)
        if not isinstance(other, SimplePolygon):
            other = SimplePolygon(other, tol=self.tol)
        bounds = self.boundary
        new_bounds = []
        new_holes = self.holes
        for b in bounds:
            new_bound = b.boolean_difference(other, tol=tol)
            if isinstance(new_bound, SimplePolygon):
                new_bounds.append(new_bound)
            elif new_bound is None:
                continue
            else:
                new_bounds.extend(new_bound.boundary)
                new_holes.extend(new_bound.holes)
        return Polygon(new_bounds, new_holes, tol=self.tol)
    
    def boolean_union(self, other, tol=None):
        tol = self.tol if tol is None else float(tol)
        if not isinstance(other, Polygon): return self._boolean_union_simple(other, tol=tol)
        new_shape = self.copy()
        for b in other.boundary:
            new_shape = new_shape._boolean_union_simple(b, tol=tol)
        for h in other.holes:
            new_hole = h.copy()
            for b in self.boundary:
                new_hole = new_hole.boolean_difference(b, tol=tol)
            new_shape = new_shape.boolean_difference(new_hole, tol=tol)
        return new_shape
    
    def boolean_intersect(self, other, tol=None):
        tol = self.tol if tol is None else float(tol)
        if not isinstance(other, Polygon): return self._boolean_intersect_simple(other, tol=tol)
        if len(other.boundary) == 1 and len(other.holes) == 0: return self._boolean_intersect_simple(other.boundary[0], tol=tol)
        new_shape = self.copy()
        boundary_group = []
        hole_group = self.holes + other.holes
        for b in other.boundary:
            intersect = new_shape._boolean_intersect_simple(b, tol=tol)
            if isinstance(intersect, SimplePolygon):
                boundary_group.append(intersect)
            else:
                boundary_group.extend(intersect.boundary)
                hole_group.extend(intersect.holes)
        
        boundary_poly = SimplePolygon.boolean_union_all(boundary_group, tol=tol)
        if isinstance(boundary_poly, Polygon):
            boundary = boundary_poly.boundary
            hole_group.extend(boundary_poly.holes)
        else:
            boundary = [boundary_poly]
        
        return Polygon(boundary, hole_group)

    
    def boolean_difference(self, other, tol=None):
        tol = self.tol if tol is None else float(tol)
        if not isinstance(other, Polygon): return self._boolean_difference_simple(other, tol=tol)
        new_shape = self.copy()
        for b in other.boundary:
            new_shape = new_shape._boolean_difference_simple(b, tol=tol)
        for h in other.holes:
            new_shape = new_shape.boolean_union(self._boolean_intersect_simple(h, tol=tol))
        return new_shape
    
    # Geometric operations
    def move(self, vector):
        v = vector if isinstance(vector, Vector) else Vector(vector, tol=self.tol)
        if np.allclose(v.arr, 0.0):  # early return
            return self
        return type(self)([b.move(v) for b in self.boundary],
                          [h.move(v) for h in self.holes], tol=self.tol)
    
    def rotate(self, angle_ccw, centroid=None):
        if not angle_ccw:
            return self
        c = self.centroid if centroid is None else (centroid if isinstance(centroid, Point) else Point(centroid, tol=self.tol))
        return type(self)([b.rotate(angle_ccw, c) for b in self.boundary],
                          [h.rotate(angle_ccw, c) for h in self.holes], tol=self.tol, repair=False)
    
    def scale(self, factor, centroid=None):
        f = float(factor)
        if np.isclose(f, 1.0):
            return self
        c = self.centroid if centroid is None else (centroid if isinstance(centroid, Point) else Point(centroid, tol=self.tol))
        poly = type(self)([b.scale(f, c) for b in self.boundary if b.scale(f, c) is not None],
                          [h.scale(-f, c) for h in self.holes if h.scale(-f, c) is not None], tol=self.tol)
        
        # Optional: enforce winding convention after negative scale (outer CCW, holes CW)
        if f < 0:
            poly = poly.ensure_winding(outer_ccw=True)  # if you have such a helper
        return poly
    
    def transform(self, vector=None, angle_ccw=0.0, scale_factor=1.0, centroid=None):
        poly = self
        if angle_ccw:
            poly = poly.rotate(angle_ccw, centroid)
        if not np.isclose(scale_factor, 1.0):
            poly = poly.scale(scale_factor, centroid)
        if vector is not None:
            poly = poly.move(vector)
        return poly
    
    def scale_per_segment(self, outward_distance, miter_limit=10.0):
        boundary = [p.copy().scale_per_segment(outward_distance, miter_limit) for p in self.boundary]
        holes = [h.copy().scale_per_segment(-outward_distance, miter_limit) for h in self.holes]
        return Polygon(boundaries=boundary, holes=holes, tol=self.tol, repair=False)
    
    # Helpers
    def _repair(self):
        tol = self.tol
        bool_tol = 2 * tol
    
        # ---- trivial cases ----
        if len(self.boundary) == 1 and not self.holes:
            # Already canonical: one outer, no holes
            self.boundary = [self.boundary[0]]
            self.holes = []
            return
        # Union on the boundary will make sure that there are no overlapping boundaries
        # Where boundaries overlap they could feasibly form new openings
        bug_holes = [h._bug for h in self.holes]
        added_holes = False
        bug_bounds = [b._bug for b in self.boundary]
        if len(self.boundary) > 1:
            if np.any([np.any([self.boundary[i].polygon_relationship(self.boundary[j]) != -1 for j in range(i+1, len(self.boundary))]) for i in range(len(self.boundary)-1)]):
                bug_boundary = Polygon2D.boolean_union_all([b._bug for b in self.boundary], bool_tol)
                bug_grouping = Polygon2D.group_boundaries_and_holes(bug_boundary, bool_tol)
                bug_bounds = [b[0] for b in bug_grouping]
                
                bug_hole_groups = [b[1:] if len(b) > 1 else [] for b in bug_grouping]
                bug_holes_ = [h for g in bug_hole_groups for h in g]
                if len(bug_holes_) != 0:
                    added_holes = True
                bug_holes = bug_holes + bug_holes_
        if len(bug_holes) == 0:
            self.boundary = [SimplePolygon(p, tol=tol) for p in bug_bounds]
            self.holes = []
            return
        
        if np.any([np.any([bug_holes[i].polygon_relationship(bug_holes[j], tolerance=self.tol) != -1 for j in range(i+1, len(bug_holes))]) for i in range(len(bug_holes)-1)]):
            bug_holes = Polygon2D.boolean_union_all(bug_holes, bool_tol) # no allowance for formation of new boundaries where holes overlap
        
        rels = ar([[b.polygon_relationship(h, tol) for b in bug_bounds] for h in bug_holes], dtype=int)
        all_contained = (rels == 1).any(axis=1).all()          # every hole contained in at least one bound
        no_overlaps   = not (rels == 0).any()                  # no hole overlaps any bound
        if all_contained:
            self.boundary = [SimplePolygon(p, tol=tol) for p in bug_bounds]
            self.holes = [SimplePolygon(h, tol=tol) for h in bug_holes]
            return
        elif no_overlaps:
            self.boundary = [SimplePolygon(p, tol=tol) for p in bug_bounds]
            keep = (rels == 1).any(axis=1)
            self.holes = [SimplePolygon(h, tol=tol) for h, k in zip(bug_holes, keep) if k]
            return
        else:
            new_holes = []
            for hole in bug_holes:
                rel = ar([b.polygon_relationship(hole, tol) for b in bug_bounds], dtype=int)
                if np.all(rel == -1):
                    continue
                if not np.any(rel == 0):
                    new_holes.append(hole)
                    continue
                new_bounds = [b for b, r in zip(bug_bounds, rel) if r != 0]
                overlap_bounds = [b for b, r in zip(bug_bounds, rel) if r == 0]
                for b in overlap_bounds:
                    new_bound = b.boolean_difference(hole, tol)
                    new_bound = [p[0] for p in Polygon2D.group_boundaries_and_holes(new_bound, tol)]
                    new_bounds.extend(new_bound)
                bug_bounds = new_bounds
            bug_holes = new_holes
            
            self.boundary = [SimplePolygon(p, tol=tol) for p in bug_bounds]
            self.holes = [SimplePolygon(p, tol=tol) for p in bug_holes]
            return

    
    @cached_property
    def group_boundaries_and_holes(self):
        return Polygon2D.group_boundaries_and_holes([p._bug for p in self.boundary]+[h._bug for h in self.holes], self.tol)
    
    @staticmethod
    def to_poly_list(obj):
        """Accept a single polygon (Polygon2D/SimplePolygon/(n,2) array)
        or an iterable of such; return list[Polygon2D]."""
        def as_poly(x):
            if isinstance(x, Polygon2D):
                return x
            if isinstance(x, SimplePolygon):
                return x._bug
            a = np.asarray(x, float)
            if a.ndim != 2 or a.shape[1] != 2 or a.shape[0] < 3:
                raise ValueError("Each boundary/hole must be (n,2) with n≥3.")
            return Polygon2D.from_array(a)
    
        # Handle single objects first
        if isinstance(obj, (Polygon2D, SimplePolygon)):
            return [as_poly(obj)]
        try:
            a = np.asarray(obj, float)
            if a.ndim == 2 and a.shape[1] == 2 and a.shape[0] >= 3:
                return [Polygon2D.from_array(a)]
        except Exception:
            pass
    
        # Otherwise, assume iterable of items
        return [as_poly(x) for x in obj]
    
    def split_by_ray(self, ray):
        b = []
        for boundary in self.boundary:
            b.append(boundary.split_by_ray(ray))
        if not np.all([b_[0] is None for b_ in b]):
            left = Polygon([b_[0] for b_ in b if b_[0] is not None], [h for h in self.holes], tol=self.tol)
        else:
            left = None
        if not np.all([b_[1] is None for b_ in b]):
            right = Polygon([b_[1] for b_ in b if b_[1] is not None], [h for h in self.holes], tol=self.tol)
        else:
            right = None
        return left, right
    
    def _to_simple_polys(self):
        bounds = [p._bug for p in self.boundary]
        holes = [h._bug for h in self.holes]
        groups = Polygon2D.group_boundaries_and_holes(bounds+holes, self.tol)
        groups = [[[Point2D.from_array(q) for q in p.to_array()] for p in group] for group in groups]
        polys = [Polygon2D.from_shape_with_holes(p[0], [h for h in p[1:]]) if len(p) > 1 else p[0] for p in groups]
        return [SimplePolygon(p, tol=self.tol) for p in polys]
    
    # creation
    @classmethod
    def from_polyline(cls, polyline, thickness, align="c"):
        return SimplePolygon.from_polyline(polyline, thickness, align=align)
    @classmethod
    def from_ring(cls, polyline, thickness):
        return SimplePolygon.from_ring(polyline, thickness)
        
    def plot(self, show=False, ax=None, boundary_color="black", fill_color="#cce6ff", lw=1.0, dpi=150):
        if ax is None:
            _, ax = plt.subplots(dpi=dpi)
        
        # 3) fill the stitched solids
        if fill_color is not None:
            areas = self._to_simple_polys()
            for a in areas:
                a.plot(False, ax=ax, boundary_color=None, fill_color=fill_color)
            """for arr in solids:
                patch = MplPolygon(arr, closed=True, facecolor=fill_color, edgecolor="none", linewidth=0)
                ax.add_patch(patch)"""
    
        # 4) draw original outer and hole boundaries
        if boundary_color is not None:
            for ring in self.boundary+self.holes:
                arr = ar(ring.to_array(), dtype=float)
                arr = np.vstack([arr, arr[0]])  # close for plotting
                ax.plot(arr[:, 0], arr[:, 1], color=boundary_color, linewidth=lw)
    
        ax.set_aspect("equal", adjustable="box")
        ax.autoscale_view()
        if show:
            plt.show()
        return ax
    
    def __str__(self):
        return f"Polygon:\n\tBoundary:{[p.__str__() for p in self.boundary]}\n\tHoles:{[h.__str__() for h in self.holes]}"
    def __repr__(self):
        return f"Polygon:\n\tBoundary:{[p.__str__() for p in self.boundary]}\n\tHoles:{[h.__str__() for h in self.holes]}"
    

        
if __name__ == "__main__":
    sp = SimplePolygon([[0,0], [7,0], [7,5],[13,5],[13,0],[20,0],[20,10],[0,10]])
    sp1, sp2 = sp.split_by_ray([[10,0],[0,1]])
    sp = SimplePolygon([[0,0],[10,0],[10,10],[0,10]])
    print(sp.elastic_modulus()[1])
    print(sp.plastic_modulus())
    print(sp.elasto_plastic_modulus(yield_distance=2.5))
    exit()
    #print(sp1, type(sp1))
    #ax=sp2.plot()
    #sp1.plot(True, ax=ax)
    
    op = SimplePolygon([[4,-1],[6,-1],[6,20],[4,20]])
    ap = Polygon(sp, op)#[[[-1,-1],[3,-1],[3,3],[-1,3]], [[4,4],[4,7],[7,8],[4.5,4]]])
    sp = Polygon([[2,-1],[17,-1],[17,6],[2,6]], [[3,0],[16,0],[16,5],[3,5]])
    qp = ap.boolean_union(sp)
    a, b = qp.split_by_ray([[0,0],[1,1]])
    print(b)
    a.plot(True)
    exit()
    ax = ap.plot(show=True)
    line = [[5,1],[6,1]]
    lines = [[[5,1],[6,1]],[[3,3],[5,7]]]
    print(sp.elastic_modulus())