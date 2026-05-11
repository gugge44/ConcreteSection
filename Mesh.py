"""Triangular mesh for cross-section analysis.

Generates a constrained Delaunay triangulation from a Polygon boundary,
with optional internal constraint lines and hole openings. Provides
vectorized geometric integrals (area, first/second moments) and
post-process splitting by lines for piecewise material models.

Design: Mesh objects are treated as immutable after construction.
Operations like split_by_y return new TriElements rather than mutating.
"""

from meshpy import triangle
from dataclasses import dataclass, field
from functools import cached_property
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from BugPoly import Polygon, SimplePolygon
import numpy as np
from numpy import array as ar


# ---------------------------------------------------------------------------
# Core splitting engine (operates on signed distances)
# ---------------------------------------------------------------------------

def _split_by_dist(verts, d):
    """Split triangles where d is a per-vertex signed distance to a line.

    Parameters
    ----------
    verts : ndarray (N, 3, 2)
    d : ndarray (N, 3) - signed distance of each vertex to the cut line.
        Positive = one side, negative = the other. Zero = on the line.

    Returns
    -------
    new_verts : ndarray (M, 3, 2)
    parent : ndarray (M,) - index into original array.
    """
    above = d > 0
    below = d < 0
    on = ~above & ~below

    n_above = above.sum(axis=1)
    n_below = below.sum(axis=1)
    n_on = on.sum(axis=1)

    uncut = (n_above == 0) | (n_below == 0)
    touch1 = (n_on == 1) & (n_above >= 1) & (n_below >= 1)
    cut3 = ~uncut & ~touch1

    result = [verts[uncut]]
    parent = [np.where(uncut)[0]]

    if touch1.any():
        idx_t = np.where(touch1)[0]
        tv = verts[touch1]
        td = d[touch1]
        ton = on[touch1]
        tidx = np.arange(tv.shape[0])

        hi = ton.argmax(axis=1)
        o1i = (hi + 1) % 3
        o2i = (hi + 2) % 3

        H  = tv[tidx, hi]
        O1 = tv[tidx, o1i]
        O2 = tv[tidx, o2i]

        # Interpolation: find where d crosses zero on edge O1-O2
        d1 = td[tidx, o1i]
        d2 = td[tidx, o2i]
        dd = d2 - d1
        t = -d1 / np.where(np.abs(dd) < 1e-15, 1e-15, dd)
        M = O1 + t[:, None] * (O2 - O1)

        result.append(np.stack([H, O1, M], axis=1))
        result.append(np.stack([H, M, O2], axis=1))
        parent.extend([idx_t, idx_t])

    if cut3.any():
        idx_c = np.where(cut3)[0]
        cv = verts[cut3]
        ca = above[cut3]
        cd = d[cut3]
        cn = n_above[cut3]
        cidx = np.arange(cv.shape[0])

        lone_mask = ca == (cn == 1)[:, None]
        li = lone_mask.argmax(axis=1)
        i1 = (li + 1) % 3
        i2 = (li + 2) % 3

        P0 = cv[cidx, li]
        P1 = cv[cidx, i1]
        P2 = cv[cidx, i2]

        d0 = cd[cidx, li]
        dP1 = cd[cidx, i1]
        dP2 = cd[cidx, i2]

        # Interpolate on edges P0-P1 and P0-P2
        dd01 = dP1 - d0
        dd02 = dP2 - d0
        t1 = -d0 / np.where(np.abs(dd01) < 1e-15, 1e-15, dd01)
        t2 = -d0 / np.where(np.abs(dd02) < 1e-15, 1e-15, dd02)

        A = P0 + t1[:, None] * (P1 - P0)
        B = P0 + t2[:, None] * (P2 - P0)

        result.append(np.stack([P0, A, B], axis=1))
        result.append(np.stack([A, P1, P2], axis=1))
        result.append(np.stack([A, P2, B], axis=1))
        parent.extend([idx_c, idx_c, idx_c])

    return np.concatenate(result, axis=0), np.concatenate(parent)


# ---------------------------------------------------------------------------
# Public splitting functions
# ---------------------------------------------------------------------------

def split_triangles(verts, y_cut):
    """Split triangles by a horizontal line y = y_cut.

    Parameters
    ----------
    verts : ndarray (N, 3, 2)
    y_cut : float

    Returns
    -------
    new_verts : ndarray (M, 3, 2)
    parent : ndarray (M,) - index into original array.
    """
    verts = np.asarray(verts, dtype=float)
    d = verts[:, :, 1] - y_cut  # signed distance to y = y_cut
    return _split_by_dist(verts, d)


def split_triangles_line(verts, p1, v):
    """Split triangles by an arbitrary line through p1 with direction v.

    The line is defined as {p1 + t*v : t in R}. The "positive" side is
    to the left of v (i.e. in the direction of the normal (-vy, vx)).

    Parameters
    ----------
    verts : ndarray (N, 3, 2)
    p1 : array-like (2,) - point on the line
    v : array-like (2,) - direction vector (need not be unit length)

    Returns
    -------
    new_verts : ndarray (M, 3, 2)
    parent : ndarray (M,) - index into original array.
    """
    verts = np.asarray(verts, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    v = np.asarray(v, dtype=float)

    # Normal (left of v): n = (-vy, vx)
    n = ar([-v[1], v[0]])
    n = n / np.linalg.norm(n)  # unit normal for consistent distance

    # Signed distance of each vertex to the line
    # d = n . (P - p1) for each vertex
    rel = verts - p1[None, None, :]  # (N, 3, 2)
    d = rel[:, :, 0] * n[0] + rel[:, :, 1] * n[1]  # (N, 3)

    return _split_by_dist(verts, d)


# ---------------------------------------------------------------------------
# Exact integrals of monomials over triangles (vectorized)
# ---------------------------------------------------------------------------

def tri_integrals(v):
    """Exact integrals of 1, x, y, x^2, y^2, xy over each triangle.

    Parameters
    ----------
    v : ndarray (N, 3, 2)

    Returns
    -------
    dict with keys '1','x','y','x2','y2','xy', each ndarray (N,).
    Values are signed integrals (positive for CCW winding).

    Formulae
    --------
    integral(1 dA) = A
    integral(x dA) = A/3 * (x1+x2+x3)
    integral(y dA) = A/3 * (y1+y2+y3)
    integral(x^2 dA) = A/6 * (x1^2+x2^2+x3^2 + x1*x2+x2*x3+x1*x3)
    integral(y^2 dA) = A/6 * (y1^2+y2^2+y3^2 + y1*y2+y2*y3+y1*y3)
    integral(xy dA) = A/12 * (2*x1*y1 + 2*x2*y2 + 2*x3*y3
                              + x1*y2 + x1*y3 + x2*y1 + x2*y3
                              + x3*y1 + x3*y2)
    """
    d1 = v[:, 1] - v[:, 0]
    d2 = v[:, 2] - v[:, 0]
    A = 0.5 * (d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0])

    x, y = v[:, :, 0], v[:, :, 1]
    x1, x2, x3 = x[:, 0], x[:, 1], x[:, 2]
    y1, y2, y3 = y[:, 0], y[:, 1], y[:, 2]

    return {
        '1':  A,
        'x':  A / 3 * (x1 + x2 + x3),
        'y':  A / 3 * (y1 + y2 + y3),
        'x2': A / 6 * (x1**2 + x2**2 + x3**2 + x1*x2 + x2*x3 + x1*x3),
        'y2': A / 6 * (y1**2 + y2**2 + y3**2 + y1*y2 + y2*y3 + y1*y3),
        'xy': A / 12 * (2*x1*y1 + 2*x2*y2 + 2*x3*y3
                        + x1*y2 + x1*y3 + x2*y1 + x2*y3 + x3*y1 + x3*y2),
    }


# ---------------------------------------------------------------------------
# TriElements - lightweight immutable view of a triangle array
# ---------------------------------------------------------------------------

class TriElements:
    """Immutable collection of triangles with cached geometric properties.

    Wraps an (N, 3, 2) vertex array and optional parent index mapping.
    All properties are computed once on first access.
    """

    def __init__(self, verts, parent_map=None):
        self._verts = np.asarray(verts, dtype=float)
        self._parent_map = parent_map

    @property
    def verts(self):
        """Triangle vertices, shape (N, 3, 2). Read-only."""
        return self._verts

    @property
    def parent_map(self):
        """Index into original mesh elements, or None."""
        return self._parent_map

    def __len__(self):
        return self._verts.shape[0]

    # -- Integrals (cached) --

    @cached_property
    def integrals(self):
        """Dict of per-element exact integrals: '1','x','y','x2','y2','xy'."""
        return tri_integrals(self._verts)

    @cached_property
    def area(self):
        """Per-element unsigned area, shape (N,)."""
        return np.abs(self.integrals['1'])

    @cached_property
    def total_area(self):
        return self.area.sum()
    
    @cached_property
    def centroids(self):
        """Per-element centroid, shape (N, 2)."""
        return self._verts.mean(axis=1)

    @cached_property
    def centroid(self):
        """Section centroid (x_c, y_c)."""
        I = self.integrals
        A = I['1'].sum()
        return ar([I['x'].sum() / A, I['y'].sum() / A])

    def Ixx(self, about=None):
        """Second moment of area about a horizontal axis.
        
        Parameters
        ----------
        about : float, array-like, or None
            y-coordinate of the axis, or [x, y] point (uses y).
            None returns I about the shape's own centroid.
        """
        c_y = self.centroid[1]
        I0 = self.integrals['y2'].sum() - self.total_area * c_y ** 2
        if about is None:
            return I0
        y = about[1] if hasattr(about, '__len__') else about
        return I0 + self.total_area * (c_y - y) ** 2
    
    def Iyy(self, about=None):
        """Second moment of area about a vertical axis."""
        c_x = self.centroid[0]
        I0 = self.integrals['x2'].sum() - self.total_area * c_x ** 2
        if about is None:
            return I0
        x = about[0] if hasattr(about, '__len__') else about
        return I0 + self.total_area * (c_x - x) ** 2
    
    def Ixy(self, about=None):
        """Product moment of area about shifted axes."""
        c_x, c_y = self.centroid
        I0 = self.integrals['xy'].sum() - self.total_area * c_x * c_y
        if about is None:
            return I0
        if hasattr(about, '__len__'):
            x, y = about[0], about[1]
        else:
            return I0  # scalar doesn't make sense for product moment
        return I0 + self.total_area * (c_x - x) * (c_y - y)
        # -- Masking (returns new TriElements) --

    def masked(self, mask):
        """Return a new TriElements containing only selected elements."""
        pm = self._parent_map[mask] if self._parent_map is not None else None
        return TriElements(self._verts[mask], pm)

    # -- Splitting (returns new TriElements) --

    def _chain_parent(self, local_parent):
        """Chain local_parent through existing parent_map."""
        if self._parent_map is not None:
            return self._parent_map[local_parent]
        return local_parent

    def split_by_y(self, y_cut):
        """Split by horizontal line, return new TriElements."""
        new_verts, local_parent = split_triangles(self._verts, y_cut)
        return TriElements(new_verts, self._chain_parent(local_parent))

    def split_by_line(self, p1, v):
        """Split by arbitrary line through p1 with direction v.

        Returns new TriElements with no element crossing the line.
        """
        new_verts, local_parent = split_triangles_line(self._verts, p1, v)
        return TriElements(new_verts, self._chain_parent(local_parent))

    def split_by_y_multi(self, y_cuts):
        """Split by multiple horizontal lines, return new TriElements."""
        result = self
        for yc in sorted(np.atleast_1d(y_cuts)):
            result = result.split_by_y(yc)
        return result

    def split_by_lines(self, lines):
        """Split by multiple arbitrary lines.

        Parameters
        ----------
        lines : list of (p1, v) tuples
        """
        result = self
        for p1, v in lines:
            result = result.split_by_line(p1, v)
        return result

    # -- Stress integration helpers --

    def axial_force(self, E, epsilon_0, phi, na):
        """F = integral(sigma dA) for sigma = E*(epsilon_0 + phi*(y - na)).

        Parameters
        ----------
        E : float or ndarray (N,) - Young's modulus per element
        epsilon_0 : float - strain at reference axis
        phi : float - curvature
        na : float - neutral axis y-coordinate
        """
        I = self.integrals
        return (E * (epsilon_0 * I['1'] + phi * (I['y'] - na * I['1']))).sum()

    def bending_moment(self, E, epsilon_0, phi, na):
        """M = integral(sigma * y dA) for sigma = E*(epsilon_0 + phi*(y - na))."""
        I = self.integrals
        return (E * (epsilon_0 * I['y'] + phi * (I['y2'] - na * I['y']))).sum()
    
    # -- Geometric tests --
    
    def is_point_inside(self, pts):
        """Test if each point is inside any triangle.
        
        Parameters
        ----------
        pts : array-like (2,) or (M, 2)
        
        Returns
        -------
        bool or ndarray (M,) of bool
        """
        pts = np.asarray(pts, float)
        scalar = pts.ndim == 1
        if scalar:
            pts = pts[None, :]
        # d: (M, N, 3, 2) = verts[None,:,:,:] - pts[:,None,None,:]
        d = self._verts[None, :, :, :] - pts[:, None, None, :]  # (M, N, 3, 2)
        i = ar([0, 1, 2])
        j = ar([1, 2, 0])
        c = d[:, :, i, 0] * d[:, :, j, 1] - d[:, :, i, 1] * d[:, :, j, 0]  # (M, N, 3)
        inside_tri = (c >= 0).all(axis=2) | (c <= 0).all(axis=2)  # (M, N)
        result = inside_tri.any(axis=1)  # (M,)
        return bool(result[0]) if scalar else result
        
    # -- Plotting --

    def plot(self, ax=None, show=False, color='#ADD8E6', edgecolor='blue',
             lw=0.5, values=None, cmap='viridis', alpha=0.5):
        """Plot the triangles.

        Parameters
        ----------
        values : ndarray (N,), optional
            Per-element scalar for colour-mapped plot.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 6))

        from matplotlib.collections import PolyCollection
        if values is not None:
            pc = PolyCollection(self._verts, array=values, cmap=cmap,
                                edgecolors=edgecolor, linewidths=lw)
            ax.add_collection(pc)
            plt.colorbar(pc, ax=ax)
        else:
            pc = PolyCollection(self._verts, edgecolors=edgecolor,
                                facecolors=color, linewidths=lw, alpha=alpha)
            ax.add_collection(pc)

        ax.set_aspect('equal')
        ax.autoscale_view()
        if show:
            plt.show()
        return ax


# ---------------------------------------------------------------------------
# Hole-boundary proximity detection and repair
# ---------------------------------------------------------------------------

def _point_to_segment_dist(p, a, b):
    """Min distance from point p to segment a-b."""
    ab = b - a
    ab2 = ab @ ab
    if ab2 < 1e-30:
        return np.linalg.norm(p - a)
    t = np.clip((p - a) @ ab / ab2, 0.0, 1.0)
    return np.linalg.norm(p - (a + t * ab))


def _hole_touches_boundary(hole_arr, bdy_arr, tol):
    """Check if any hole vertex is within tol of any boundary edge or vice versa."""
    n_b = len(bdy_arr)
    for hp in hole_arr:
        for i in range(n_b):
            if _point_to_segment_dist(hp, bdy_arr[i], bdy_arr[(i + 1) % n_b]) < tol:
                return True
    n_h = len(hole_arr)
    for bp in bdy_arr:
        for i in range(n_h):
            if _point_to_segment_dist(bp, hole_arr[i], hole_arr[(i + 1) % n_h]) < tol:
                return True
    return False


def _fix_dangerous_hole(bdy_poly, hole_poly, tol):
    """Fix a hole that touches/overlaps the boundary so meshpy won't crash.

    Two-pass approach:

    Pass 1 - Parallel edges:
        For each hole edge parallel and close to a boundary edge, move its
        two vertices by 2*dist along the boundary edge's outward normal
        (pointing away from the boundary interior, i.e. outside).
        If any edges were fixed, boolean-diff the expanded hole from
        the boundary and remove the hole.

    Pass 2 - Vertex contact:
        For any hole vertex still within tol of the boundary, create a
        small rectangular tube: 1*dist into the hole, 2*dist past the
        boundary, width 4*tol, oriented perpendicular to the nearest
        boundary edge. Boolean-union the tube onto the hole, then
        boolean-diff from the boundary and remove the hole.

    Returns
    -------
    result : SimplePolygon or Polygon
        The new boundary with the hole subtracted. The caller should
        extract .boundary[0] and .holes if it's a Polygon.
    was_fixed : bool
        True if any repair was done (hole should not be passed to Triangle
        separately). False if no contact was found (shouldn't happen if
        _hole_touches_boundary was True, but defensive).
    """
    hole_arr = hole_poly.arr.copy()
    bdy_arr = bdy_poly.arr
    n_h = len(hole_arr)
    n_b = len(bdy_arr)
    dist = tol  # base distance unit

    # ------------------------------------------------------------------
    # Pass 1: fix parallel edges
    # ------------------------------------------------------------------
    any_parallel = False
    new_hole_pts = hole_arr.copy()

    for hi in range(n_h):
        ha, hb = hole_arr[hi], hole_arr[(hi + 1) % n_h]
        dh = hb - ha
        lh = np.linalg.norm(dh)
        if lh < 1e-15:
            continue

        for bi in range(n_b):
            ba, bb = bdy_arr[bi], bdy_arr[(bi + 1) % n_b]
            db = bb - ba
            lb = np.linalg.norm(db)
            if lb < 1e-15:
                continue
            ub = db / lb

            # Check parallel
            cross = abs(dh[0] * ub[1] - dh[1] * ub[0]) / lh
            if cross > tol / max(lh, lb):
                continue

            # Check close
            perp_vec = ha - ba - ((ha - ba) @ ub) * ub
            perp_dist = np.linalg.norm(perp_vec)
            if perp_dist > tol:
                continue

            # Parallel and close. Compute the boundary edge outward normal.
            # For a CCW boundary, the outward normal of edge ba->bb is
            # (dy, -dx) normalised (pointing right of the edge direction).
            outward_n = ar([db[1], -db[0]]) / lb

            # Check the normal points outward (away from boundary interior).
            # Test: midpoint of boundary edge + small step along normal
            # should be outside the boundary.
            mid_bdy = (ba + bb) / 2
            test_pt = mid_bdy + outward_n * lb * 0.01
            if bdy_poly.is_point_inside(test_pt):
                outward_n = -outward_n

            # Move both hole vertices 2*dist along this outward normal
            new_hole_pts[hi] = new_hole_pts[hi] + outward_n * 2 * dist
            new_hole_pts[(hi + 1) % n_h] = new_hole_pts[(hi + 1) % n_h] + outward_n * 2 * dist
            any_parallel = True
            break  # one boundary edge match per hole edge

    if any_parallel:
        expanded_hole = SimplePolygon(new_hole_pts, tol=bdy_poly.tol)
        return bdy_poly.boolean_difference(expanded_hole, tol=bdy_poly.tol), True

    # ------------------------------------------------------------------
    # Pass 2: fix vertex-only contact with tubes
    # ------------------------------------------------------------------
    # Rebuild hole_poly in case pass 1 modified nothing
    any_vertex = False
    tube_polys = []
    tube_width = 4 * tol
    tube_into_hole = 1 * dist
    tube_past_bdy = 2 * dist

    for hi in range(n_h):
        hp = hole_arr[hi]
        # Find nearest boundary edge
        best_d = np.inf
        best_bi = 0
        for bi in range(n_b):
            d = _point_to_segment_dist(hp, bdy_arr[bi], bdy_arr[(bi + 1) % n_b])
            if d < best_d:
                best_d = d
                best_bi = bi

        if best_d >= tol:
            continue

        # Build tube perpendicular to boundary edge at contact point
        ba, bb = bdy_arr[best_bi], bdy_arr[(best_bi + 1) % n_b]
        db = bb - ba
        lb = np.linalg.norm(db)
        if lb < 1e-15:
            continue
        ub = db / lb

        # Outward normal of boundary edge
        outward_n = ar([db[1], -db[0]]) / lb
        mid_bdy = (ba + bb) / 2
        test_pt = mid_bdy + outward_n * lb * 0.01
        if bdy_poly.is_point_inside(test_pt):
            outward_n = -outward_n

        # Inward = into the hole = opposite of outward
        inward_n = -outward_n

        # Tangent along boundary edge (for tube width)
        tangent = ub

        # Tube corners: centred on hp,
        # extends tube_past_bdy in outward direction,
        # extends tube_into_hole in inward direction,
        # half-width tube_width/2 in tangent direction.
        hw = tube_width / 2
        p_out = hp + outward_n * tube_past_bdy
        p_in = hp + inward_n * tube_into_hole

        tube_pts = [
            p_out - tangent * hw,
            p_out + tangent * hw,
            p_in + tangent * hw,
            p_in - tangent * hw,
        ]
        tube_polys.append(SimplePolygon(tube_pts, tol=bdy_poly.tol))
        any_vertex = True

    if any_vertex:
        # Union all tubes onto the hole
        merged_hole = Polygon(hole_poly, tol=bdy_poly.tol)
        for tube in tube_polys:
            merged_hole = merged_hole.boolean_union(tube, tol=bdy_poly.tol)
        # Diff from boundary
        return bdy_poly.boolean_difference(merged_hole, tol=bdy_poly.tol), True

    # Nothing to fix (shouldn't reach here if _hole_touches_boundary was True)
    return None, False


# ---------------------------------------------------------------------------
# Mesh class - generates triangulation, then exposes as TriElements
# ---------------------------------------------------------------------------

@dataclass
class Mesh:
    """Constrained Delaunay triangulation of a polygonal cross-section.

    Parameters
    ----------
    boundary : SimplePolygon, Polygon, or array-like
        Outer boundary. Accepts a SimplePolygon (single region),
        a Polygon (multiple boundaries + holes), or raw (N,2) coords.
    spacing : float, optional
        Target element edge length (max triangle area = spacing^2/2).
        None (default) for coarsest possible mesh (no area refinement).
    min_angle : float, optional
        Minimum interior angle in degrees. Triggers Steiner point
        insertion to eliminate skinny triangles. None for no constraint.
        Typical values: 20-33 (Triangle guarantees termination up to
        ~33 degrees; default in meshpy is 20 when refinement is active).
    internal_points : ndarray (M, 2), optional
        Points forced into the triangulation.
    internal_lines : list of (2, 2) arrays, optional
        Line segments forced as constraint edges.

    After construction, the Mesh is immutable. Access elements via the
    .tri property which returns a TriElements object, or use .split_by_y()
    to get a new TriElements split at a given y-value.
    """
    boundary: Polygon
    spacing: float = None
    min_angle: float = 20
    internal_points: np.ndarray = field(default=None)
    internal_lines: list = field(default=None)

    def __post_init__(self):
        # Normalise input to a Polygon (handles SimplePolygon, Polygon, raw coords)
        bdy = self.boundary
        if isinstance(bdy, SimplePolygon):
            poly = Polygon(bdy)
        elif isinstance(bdy, Polygon):
            poly = bdy
        else:
            poly = Polygon(bdy)

        max_area = None if self.spacing is None else self.spacing ** 2 / 2
        build_kwargs = {}
        if max_area is not None:
            build_kwargs['max_volume'] = max_area
        if self.min_angle is not None:
            build_kwargs['min_angle'] = self.min_angle

        # Group boundaries with their holes so each disjoint region
        # is meshed independently (avoids Triangle filling gaps between them).
        # For the common single-boundary case, skip the grouping overhead.
        if len(poly.boundary) == 1:
            groups = [(poly.boundary[0], poly.holes)]
        else:
            raw_groups = poly.group_boundaries_and_holes  # list of [bdy, hole1, ...]
            groups = [
                (SimplePolygon(g[0], tol=poly.tol),
                 [SimplePolygon(h, tol=poly.tol) for h in g[1:]])
                for g in raw_groups
            ]

        all_points = []
        all_elements = []
        all_boundary_facets = []
        all_hole_facets = []
        all_line_facets = []
        pt_offset = 0

        for bdy_poly, hole_polys in groups:

            points = bdy_poly.arr.tolist()
            n_bdy = len(points)
            facets = [[i, (i + 1) % n_bdy] for i in range(n_bdy)]
            facet_markers = [1] * n_bdy
            marker_id = 2

            # Internal constraint lines (added to every group - lines outside
            # this group's boundary won't affect the mesh, but the endpoints
            # must lie inside for Triangle to use them as constraints)
            line_marker_ids = []
            if self.internal_lines is not None:
                for line in self.internal_lines:
                    # Only include lines whose midpoint is inside this boundary
                    mid = (ar(line[0]) + ar(line[1])) / 2
                    if not bdy_poly.is_point_inside(mid):
                        continue
                    n = len(points)
                    points.extend(line)
                    facets.append([n, n + 1])
                    facet_markers.append(marker_id)
                    line_marker_ids.append(marker_id)
                    marker_id += 1

            # Holes for this group
            # Check each hole for proximity to boundary. If a hole touches
            # or nearly touches the boundary, fix it via boolean operations
            # (meshpy crashes on coincident/near-coincident edges).
            # Otherwise pass it normally as a hole with seed point.
            hole_marker_ids = []
            hole_seeds = []
            safe_holes = []
            tol_proximity = poly.tol * 10  # detection tolerance

            for hole in hole_polys:
                touches = _hole_touches_boundary(hole.arr, bdy_poly.arr, tol_proximity)
                if touches:
                    result, was_fixed = _fix_dangerous_hole(bdy_poly, hole, tol_proximity)
                    if was_fixed and result is not None:
                        if isinstance(result, Polygon):
                            bdy_poly = result.boundary[0]
                            safe_holes.extend(result.holes)
                        elif isinstance(result, SimplePolygon):
                            bdy_poly = result
                        # Hole has been diffed into boundary - don't add as separate hole
                    else:
                        safe_holes.append(hole)  # fallback: pass as-is
                else:
                    safe_holes.append(hole)

            # Rebuild boundary facets from (possibly modified) bdy_poly
            points = bdy_poly.arr.tolist()
            n_bdy = len(points)
            facets = [[i, (i + 1) % n_bdy] for i in range(n_bdy)]
            facet_markers = [1] * n_bdy
            marker_id = 2

            # Internal constraint lines
            line_marker_ids = []
            if self.internal_lines is not None:
                for line in self.internal_lines:
                    mid = (ar(line[0]) + ar(line[1])) / 2
                    if not bdy_poly.is_point_inside(mid):
                        continue
                    n = len(points)
                    points.extend(line)
                    facets.append([n, n + 1])
                    facet_markers.append(marker_id)
                    line_marker_ids.append(marker_id)
                    marker_id += 1

            # Add safe holes normally
            for hole in safe_holes:
                n = len(points)
                pts = hole.arr
                points.extend(pts.tolist())
                n_h = len(pts)
                for i in range(n_h):
                    facets.append([n + i, n + (i + 1) % n_h])
                    facet_markers.append(marker_id)
                hole_marker_ids.append(marker_id)
                marker_id += 1
                seed = pts.mean(axis=0).tolist()
                if not hole.is_point_inside(seed):
                    seed = list(hole.pole_of_inaccessibility(1e-6))
                hole_seeds.append(seed)

            # Internal seed points inside this boundary
            if self.internal_points is not None:
                inside = [bdy_poly.is_point_inside(p) for p in self.internal_points]
                if any(inside):
                    pts_inside = self.internal_points[inside]
                    points = np.concatenate((points, pts_inside)).tolist()

            # -- Run meshpy.triangle for this group --
            info = triangle.MeshInfo()
            info.set_points(points)
            info.set_facets(facets, facet_markers=facet_markers)
            if hole_seeds:
                info.set_holes(hole_seeds)

            mesh = triangle.build(info, **build_kwargs)
            markers = list(mesh.facet_markers)

            # Collect results with global point offset
            mesh_pts = ar(mesh.points)
            mesh_elems = ar(mesh.elements) + pt_offset

            all_points.append(mesh_pts)
            all_elements.append(mesh_elems)

            all_boundary_facets.append(
                ar([f for f, m in zip(mesh.facets, markers) if m == 1]) + pt_offset
            )
            all_line_facets.extend([
                ar([f for f, m in zip(mesh.facets, markers) if m == mid]) + pt_offset
                for mid in line_marker_ids
            ])
            all_hole_facets.extend([
                ar([f for f, m in zip(mesh.facets, markers) if m == mid]) + pt_offset
                for mid in hole_marker_ids
            ])

            pt_offset += len(mesh_pts)

        # -- Store immutable results --
        self._points = np.concatenate(all_points, axis=0)
        self._element_inds = np.concatenate(all_elements, axis=0)
        self._boundary = poly

        self._boundary_facets = all_boundary_facets
        self._line_facets = all_line_facets
        self._hole_facets = all_hole_facets

    # -- Public read-only access --

    @property
    def points(self):
        return self._points

    @property
    def element_inds(self):
        return self._element_inds

    @property
    def boundary_facets(self):
        return self._boundary_facets

    @property
    def line_facets(self):
        return self._line_facets

    @property
    def hole_facets(self):
        return self._hole_facets

    @cached_property
    def elements(self):
        """Triangle vertex coordinates, shape (N, 3, 2)."""
        return self._points[self._element_inds]

    @property
    def n_elements(self):
        return self._element_inds.shape[0]

    # -- TriElements view --

    @cached_property
    def tri(self):
        """TriElements view of the mesh - all geometric properties via this."""
        return TriElements(self.elements)

    def split_by_y(self, y_cut):
        """Return a new TriElements split at y = y_cut."""
        return self.tri.split_by_y(y_cut)

    def split_by_y_multi(self, y_cuts):
        """Return a new TriElements split at multiple y-values."""
        return self.tri.split_by_y_multi(y_cuts)

    def split_by_line(self, p1, v):
        """Return a new TriElements split by line through p1 with direction v."""
        return self.tri.split_by_line(p1, v)

    # -- Edge / adjacency --

    @cached_property
    def edges(self):
        """Unique edges as (M, 2) index array, sorted per row."""
        ei = self._element_inds
        e0 = np.stack([ei[:, 0], ei[:, 1]], axis=1)
        e1 = np.stack([ei[:, 1], ei[:, 2]], axis=1)
        e2 = np.stack([ei[:, 2], ei[:, 0]], axis=1)
        all_edges = np.concatenate([e0, e1, e2], axis=0)
        all_edges.sort(axis=1)
        return np.unique(all_edges, axis=0)

    # -- Plotting --

    def plot(self, show=False, color='#ADD8E6', outline=False, ax=None,
             alt_mask=None, alt_color='r', edgecolor='blue', lw=1.5,
             values=None, cmap='viridis'):
        """Plot the mesh."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 6))

        if values is not None:
            from matplotlib.collections import PolyCollection
            pc = PolyCollection(self.elements, array=values, cmap=cmap,
                                edgecolors=edgecolor, linewidths=lw)
            ax.add_collection(pc)
            plt.colorbar(pc, ax=ax)
        else:
            for e, inds in enumerate(self._element_inds):
                coords = self._points[inds]
                if outline:
                    coords = np.concatenate((coords, [coords[0]]))
                    c = 'r' if (alt_mask is not None and alt_mask[e]) else 'b'
                    ax.plot(coords[:, 0], coords[:, 1], c)
                else:
                    fc = alt_color if (alt_mask is not None and alt_mask[e]) else color
                    patch = patches.Polygon(
                        coords, closed=True, edgecolor=edgecolor,
                        facecolor=fc, alpha=0.5, lw=lw, zorder=1)
                    ax.add_patch(patch)

        for bf in self._boundary_facets:
            for facet in bf:
                p = self._points[facet]
                ax.plot(p[:, 0], p[:, 1], 'k', lw=1)
        for hf in self._hole_facets:
            for facet in hf:
                p = self._points[facet]
                ax.plot(p[:, 0], p[:, 1], 'k', lw=1)

        ax.set_aspect('equal')
        ax.autoscale_view()
        if show:
            plt.show()
        return ax


# ---------------------------------------------------------------------------
# __main__ demo / tests
# ---------------------------------------------------------------------------

if __name__ == '__main__':

    # ---- Helper ----
    def check(label, got, expect, tol=1e-4):
        ok = np.isclose(got, expect, atol=tol)
        tag = 'PASS' if (ok if np.isscalar(ok) else ok.all()) else 'FAIL'
        print(f'  {tag} {label}: {got} (expect {expect})')

    # ================================================================
    # 1. SimplePolygon - rectangle 300x500
    # ================================================================
    print('=== SimplePolygon: 300x500 rectangle ===')
    rect = SimplePolygon([[0,0],[300,0],[300,500],[0,500]])
    m1 = Mesh(rect, spacing=50)
    check('area', m1.tri.total_area, 300*500)
    check('centroid', m1.tri.centroid, ar([150, 250]))
    check('Ixx', m1.tri.Ixx, 300*500**3/12)
    check('Iyy', m1.tri.Iyy, 500*300**3/12)
    print(f'  {m1.n_elements} elements')

    # Coarsest mesh (no spacing)
    m1c = Mesh(rect)
    print(f'  Coarsest: {m1c.n_elements} elements, area={m1c.tri.total_area:.1f}')

    # ================================================================
    # 2. SimplePolygon - T-section as single boundary
    # ================================================================
    print('\n=== SimplePolygon: T-section ===')
    # Flange 400x100 on top, web 200x400 below
    t_section = SimplePolygon([
        [0, 400], [400, 400], [400, 500], [0, 500],  # flange
        [0, 400], [100, 400], [100, 0], [300, 0],     # web
        [300, 400], [400, 400],
    ])
    # Actually that won't work as a simple polygon - let's do the outline properly
    t_section = SimplePolygon([
        [100, 0], [300, 0], [300, 400], [400, 400],
        [400, 500], [0, 500], [0, 400], [100, 400],
    ])
    t_area = 200*400 + 400*100  # web + flange = 120000
    m2 = Mesh(t_section, spacing=30)
    check('area', m2.tri.total_area, t_area)
    print(f'  {m2.n_elements} elements')

    # ================================================================
    # 3. Polygon with hole - hollow rectangle
    # ================================================================
    print('\n=== Polygon: hollow rectangle ===')
    hollow = Polygon(
        [[0,0],[400,0],[400,300],[0,300]],
        [[50,50],[350,50],[350,250],[50,250]]
    )
    hollow_area = 400*300 - 300*200
    m3 = Mesh(hollow, spacing=30)
    check('area', m3.tri.total_area, hollow_area)
    print(f'  {m3.n_elements} elements')

    # ================================================================
    # 4. Polygon with multiple disjoint boundaries - two rectangles
    # ================================================================
    print('\n=== Polygon: two disjoint rectangles ===')
    two_rects = Polygon([
        SimplePolygon([[0,0],[100,0],[100,200],[0,200]]),
        SimplePolygon([[200,0],[400,0],[400,200],[200,200]]),
    ], repair=False)
    two_area = 100*200 + 200*200
    m4 = Mesh(two_rects, spacing=30)
    check('area', m4.tri.total_area, two_area)
    check('n_boundaries', len(m4.boundary_facets), 2)
    print(f'  {m4.n_elements} elements')

    # ================================================================
    # 5. Complex Polygon: I-beam with service hole
    # ================================================================
    print('\n=== Complex Polygon: I-beam with hole ===')
    # I-beam: flanges 400x40, web 40x320, total height 400
    # Plus a circular-ish service hole in the web
    flange_b, flange_t = 400, 40
    web_t, web_h = 40, 320
    H = 2*flange_t + web_h  # 400

    i_beam = SimplePolygon([
        [0, 0], [flange_b, 0], [flange_b, flange_t],                   # bottom flange right
        [(flange_b+web_t)/2, flange_t],                                  # web right bottom
        [(flange_b+web_t)/2, flange_t+web_h],                           # web right top
        [flange_b, flange_t+web_h],                                      # top flange right
        [flange_b, H], [0, H],                                           # top flange
        [0, flange_t+web_h],                                             # top flange left
        [(flange_b-web_t)/2, flange_t+web_h],                           # web left top
        [(flange_b-web_t)/2, flange_t],                                  # web left bottom
        [0, flange_t],                                                    # bottom flange left
    ])

    # Circular-ish hole in web centre (octagon approximation)
    hole_r = 12
    hole_cx, hole_cy = flange_b/2, H/2
    n_pts = 12
    angles = np.linspace(0, 2*np.pi, n_pts, endpoint=False)
    hole_pts = [[hole_cx + hole_r*np.cos(a), hole_cy + hole_r*np.sin(a)] for a in angles]

    i_beam_with_hole = Polygon(i_beam, hole_pts)
    i_area = 2*flange_b*flange_t + web_t*web_h - np.pi*hole_r**2

    m5 = Mesh(i_beam_with_hole, spacing=8)
    check('area (approx)', m5.tri.total_area, i_area, tol=50)  # octagon approx of circle
    print(f'  {m5.n_elements} elements')

    # ================================================================
    # 6. Complex Polygon: two channels with holes (disjoint + holes)
    # ================================================================
    print('\n=== Complex: two disjoint C-channels ===')
    def make_channel(x_off):
        """C-channel: 150 wide, 300 tall, 20mm flanges, 15mm web."""
        return SimplePolygon([
            [x_off, 0], [x_off+150, 0], [x_off+150, 20],
            [x_off+15, 20], [x_off+15, 280],
            [x_off+150, 280], [x_off+150, 300], [x_off, 300],
        ])

    ch1 = make_channel(0)
    ch2 = make_channel(250)
    two_channels = Polygon([ch1, ch2], repair=False)
    ch_area = 2 * (150*20*2 + 15*260)  # two channels
    m6 = Mesh(two_channels, spacing=10)
    check('area', m6.tri.total_area, ch_area)
    check('n_boundaries', len(m6.boundary_facets), 2)
    print(f'  {m6.n_elements} elements')

    # ================================================================
    # 7. Split test on meshed Polygon
    # ================================================================
    print('\n=== Split test on I-beam mesh ===')
    split = m5.split_by_y(H/2)
    check('area conserved', split.total_area, m5.tri.total_area, tol=1)
    I_orig = tri_integrals(m5.tri.verts)
    I_split = tri_integrals(split.verts)
    for k in I_orig:
        ok = np.isclose(I_orig[k].sum(), I_split[k].sum(), atol=1e-3)
        print(f'  integral({k}): {"PASS" if ok else "FAIL"}')

    # ================================================================
    # Plot: 2x3 grid showing all test cases
    # ================================================================
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # 1) T-section
    m2.tri.plot(ax=axes[0, 0], edgecolor='steelblue', lw=0.4)
    axes[0, 0].set_title(f'T-section ({m2.n_elements} tris)')

    # 2) Hollow rectangle
    m3.tri.plot(ax=axes[0, 1], edgecolor='steelblue', lw=0.4)
    axes[0, 1].set_title(f'Hollow rect ({m3.n_elements} tris)')

    # 3) Two disjoint rectangles
    m4.tri.plot(ax=axes[0, 2], edgecolor='steelblue', lw=0.4)
    axes[0, 2].set_title(f'Two disjoint ({m4.n_elements} tris)')

    # 4) I-beam with hole
    m5.tri.plot(ax=axes[1, 0], edgecolor='steelblue', lw=0.3)
    axes[1, 0].set_title(f'I-beam + hole ({m5.n_elements} tris)')

    # 5) Two C-channels
    m6.tri.plot(ax=axes[1, 1], edgecolor='steelblue', lw=0.3)
    axes[1, 1].set_title(f'Two C-channels ({m6.n_elements} tris)')

    # 6) I-beam split at mid-height, coloured by side
    y_mid = H / 2
    centroids = split.verts.mean(axis=1)
    side = (centroids[:, 1] > y_mid).astype(float)
    split.plot(ax=axes[1, 2], values=side, cmap='coolwarm', edgecolor='grey', lw=0.2)
    axes[1, 2].axhline(y_mid, color='k', ls='--', lw=1.5)
    axes[1, 2].set_title(f'I-beam split y={y_mid:.0f} ({len(split)} tris)')

    for ax in axes.flat:
        ax.set_aspect('equal')
        ax.autoscale_view()

    plt.tight_layout()
    plt.savefig('mesh_tests.png', dpi=150)
    print('\nSaved mesh_tests.png')
    plt.show()
