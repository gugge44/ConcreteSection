import numpy as np
from numpy import array as ar
from BugBasics import Point, Line, PolyLine, Vector, Ray
from BugPoly import SimplePolygon, Polygon
from RC_beam_mesh import RCSection, DesignConstants


def _find_subsequence_placements(needle, haystack):
    results = []
    def recurse(ni, hi, chosen):
        if ni == len(needle):
            results.append(tuple(chosen))
            return
        for h in range(hi, len(haystack) - (len(needle) - ni) + 1):
            if haystack[h] == needle[ni]:
                recurse(ni + 1, h + 1, chosen + [h])
    recurse(0, 0, [])
    return results


def broadcast_to_shape(obj, shape):
    arr = np.asarray(obj)
    target = tuple(shape)
    nd = len(target)

    if arr.ndim == 0:
        return np.full(target, arr.item())

    if arr.ndim > nd:
        raise ValueError(f"input ndim {arr.ndim} exceeds target ndim {nd}")

    if arr.ndim == nd:
        for i, (s, t) in enumerate(zip(arr.shape, target)):
            if s != 1 and s != t:
                raise ValueError(f"axis {i}: input dim {s} does not match target {t}")
        return np.broadcast_to(arr, target).copy()

    in_shape = arr.shape
    nonone_axes = [k for k, s in enumerate(in_shape) if s != 1]
    nonone_dims = [in_shape[k] for k in nonone_axes]

    placements = _find_subsequence_placements(nonone_dims, target)
    if not placements:
        raise ValueError(
            f"input shape {in_shape} cannot be placed into target {target}: "
            f"non-size-1 dims {nonone_dims} are not an ordered subsequence"
        )
    if len(placements) > 1:
        raise ValueError(
            f"input shape {in_shape} is ambiguous for target {target}: "
            f"non-size-1 dims could be placed at {placements}"
        )

    nonone_positions = placements[0]
    target_positions = [None] * arr.ndim
    for k, p in zip(nonone_axes, nonone_positions):
        target_positions[k] = p

    free_target = set(range(nd)) - set(nonone_positions)
    i = 0
    prev_tp = -1
    while i < arr.ndim:
        if in_shape[i] != 1:
            prev_tp = target_positions[i]
            i += 1
            continue
        j = i
        while j < arr.ndim and in_shape[j] == 1:
            j += 1
        next_tp = nd
        for k2 in range(j, arr.ndim):
            if in_shape[k2] != 1:
                next_tp = target_positions[k2]
                break
        gap = [p for p in range(prev_tp + 1, next_tp) if p in free_target]
        run_len = j - i
        chosen = gap[-run_len:]
        for k2, p in zip(range(i, j), chosen):
            target_positions[k2] = p
            free_target.discard(p)
        i = j

    new_shape = [1] * nd
    for k, p in enumerate(target_positions):
        new_shape[p] = in_shape[k]
    return np.broadcast_to(arr.reshape(new_shape), target).copy()

class CoreSection:
    _instance_counter = 0
    
    def __init__(
            self, f_ck, wall_lines, wall_thk, v_dias, v_spacing,
            h_dias, h_spacing, cover=30, c_class='CN',
            t_ref=28, k_E=9500, f_y=500,
            gamma_s=DesignConstants.GAMMA_S_DEFAULT,
            gamma_c: float = DesignConstants.GAMMA_C_DEFAULT,
            verts_outer_layer:bool = False
        ):
        self.global_index = CoreSection._instance_counter
        CoreSection._instance_counter += 1
        self.local_index = None  # Set by CoreLevel
        self.level = None  # Set by CoreLevel
        
        N_walls = len(wall_lines)
        self.f_ck = float(f_ck)
        self.wall_lines = []
        self.ts = []
        self.v_centres = []
        self.v_spacing = broadcast_to_shape(v_spacing, [N_walls,2]).astype(float)#np.atleast_1d(v_spacing).astype(float)
        self.h_spacing = broadcast_to_shape(h_spacing, [N_walls,2]).astype(float)#np.atleast_1d(h_spacing).astype(float)
        self.v_dias = broadcast_to_shape(v_dias, [N_walls,2]).astype(float)#np.atleast_1d(v_dias).astype(float)
        self.h_dias = broadcast_to_shape(h_dias, [N_walls,2]).astype(float)#np.atleast_1d(h_dias).astype(float)
        self.cover = broadcast_to_shape(cover, [N_walls,2]).astype(float)#np.atleast_2d(cover).astype(float)
        
        if not isinstance(wall_thk, (list, tuple, np.ndarray)):
            wall_thk = wall_thk * np.ones(len(wall_lines), float)
        
        all_wall_lines = []
        all_wall_polys = []
        all_ts = []
        adders = {}
        subbers = {}
        
        # Separate lines and polylines into individual line segments, populating the relevant lists
        for line, t in zip(wall_lines, wall_thk):
            self.wall_lines.append(line)
            self.ts.append(t)
            lines = [line] if isinstance(line, Line) else line.segments
            ts = [t]*len(lines)
            all_wall_lines.extend(lines)
            all_ts.extend(ts)
            all_wall_polys.extend([SimplePolygon.from_polyline(l, t_) for l, t_ in zip(lines, ts)])
        
        # number of individual wall segments
        N_segs = len(all_wall_lines)
        
        # repair intersections between connecting walls
        for i, (line1, poly1, thk1) in enumerate(zip(all_wall_lines, all_wall_polys, all_ts)):
            if i == N_segs-1: continue
            for j in range(i+1, N_segs):
                # set up variables incl vectors
                line2, poly2, thk2 = all_wall_lines[j], all_wall_polys[j], all_ts[j]
                t1, t2 = ts = line1.vector.unit, line2.vector.unit
                n1, n2 = ns = t1.normal, t2.normal
                cs = [line1.centroid, line2.centroid]
                
                lines, polys, thks = [line1, line2], [poly1, poly2], [thk1, thk2]
                max_length = max([line1.length, line2.length, thk1, thk2])*2 # this is a tolerance variable for extrema
                
                # if the polygons don't touch at all continue to the next connection
                if not poly1.does_polygon_touch(poly2, tolerance=1e-6):
                    continue
                
                # if the walls are parallel but touching just merge them
                if np.isclose(t1.cross(t2),0): # parallel lines
                    #all_wall_polys[i] = all_wall_polys[j] = poly1.boolean_union(poly2)
                    continue
                
                # set up a point for each line and use those to find the point of intersection
                p1, p2 = Vector(line1[0]), Vector(line2[0])
                α = (p2.dot(t1)-p1.dot(t1) + (t1.dot(t2))*(p1.dot(t2)-p2.dot(t2)))/(1-t1.dot(t2)**2)
                pc = p1 + t1*α # point of intersection
                
                sqs = []
                for m in [0, 1]:
                    n = 1-m
                    ind_global = i if m == 0 else j
                    
                    # === Fill in where polygons fall short ===
                    
                    # create an infill rectangle for each wall
                    # the height of the rectangle is the possible intersection area including the tₘ distance of tₙ across nₘ and the tₘ distance of thkₙ across nₘ
                    # thks[n] represents the projected thickness of n
                    # thks[m]*... represents the length of n across the width of m (pythagoras)
                    h = (thks[n] + thks[m]*abs(ts[m].dot(ts[n])))/abs(ns[m].dot(ts[n]))
                    sq = SimplePolygon([
                        pc + ns[m]*thks[m]/2 + ts[m]*h/2,
                        pc + ns[m]*thks[m]/2 - ts[m]*h/2,
                        pc - ns[m]*thks[m]/2 - ts[m]*h/2,
                        pc - ns[m]*thks[m]/2 + ts[m]*h/2
                    ])
                    if ind_global not in adders.keys():
                        adders[ind_global] = sq
                    else:
                        adders[ind_global] = adders[ind_global].boolean_union(sq)
                    
                    # === Cut back where polygons over-extend ===
                    if not np.any([sq.is_point_inside(lines[m][q]) for q in [0,1]]):
                        continue # wall doesn't end at the intersect, no risk of over-extending
                    
                    ind_local = 0 if sq.is_point_inside(lines[m][0]) else 1
                    # pick the other wall normal that points away from the considered wall
                    n_ef = ns[n] if Vector(cs[n]+ns[n]-cs[m]).magnitude > Vector(cs[n]-cs[m]).magnitude else -ns[n]
                    # pick two points on the other wall centre which fully cover the space
                    p1_ef = lines[n][0] - ts[n]*max_length
                    p2_ef = lines[n][1] + ts[n]*max_length
                    # create a rectangle which sits just beyond the other wall to use for trimming
                    cut_shape = SimplePolygon([
                        p1_ef + n_ef*thks[n]/2,
                        p1_ef + n_ef*max_length,
                        p2_ef + n_ef*max_length,
                        p2_ef + n_ef*thks[n]/2
                    ])
                    if ind_global not in subbers.keys():
                        subbers[ind_global] = {0:None, 1:None}
                    if subbers[ind_global][ind_local] is None:
                        subbers[ind_global][ind_local] = cut_shape
                    else:
                        subbers[ind_global][ind_local] = subbers[ind_global][ind_local].boolean_intersect(cut_shape)
                    
        for k, wall in enumerate(all_wall_polys):
            if k in adders.keys():
                wall = wall.boolean_union(adders[k])
            if k in subbers.keys():
                if subbers[k][0] is not None:
                    wall = wall.boolean_difference(subbers[k][0])
                if subbers[k][1] is not None:
                    wall = wall.boolean_difference(subbers[k][1])
            all_wall_polys[k] = wall
        
        wall_section = SimplePolygon.boolean_union_all(all_wall_polys)
        
        # === Distribute reinforcement ===
        
        vds = []
        vss = []
        for wall, thk, d, s, c, dh in zip(self.wall_lines, self.ts, self.v_dias, self.v_spacing, self.cover, self.h_dias):
            c_add = (c[0] + d[0]/2 + (0 if verts_outer_layer else dh[0]),
                            c[1] + d[1]/2 + (0 if verts_outer_layer else dh[1]))
            thk_ef = ar([-thk/2+c_add[0], thk/2-c_add[1]], float)
            t = wall.vector.unit
            n = t.normal
            bar_lines = [wall.move(thk_ef[k]*n) for k in [0,1]]
            bar_crs = []
            bar_dias = []
            for k in [0, 1]:
                bc = bar_lines[k].arange(s[k], half_shift=True)
                bd = broadcast_to_shape(d[k], bc.shape[:-1])
                bar_crs.extend(bc)
                bar_dias.extend(bd)
            
            vds.extend(bar_dias)
            vss.extend(bar_crs)
        
        
        self.v_section = RCSection(
            f_ck, wall_section, vss, vds,
            c_class, t_ref, k_E, f_y, gamma_s, gamma_c, True, False
        )
    
    def plot(self, **kw):
        self.v_section.plot(**kw)


class CoreBeam(RCSection):
    _instance_counter = 0
    
    def __init__(
            self,
            f_ck_28,
            concrete_poly,
            rebar_centers,
            rebar_diameters,
            c_class: str = 'CN',
            t_ref: float = 28,
            k_E: float = 9500,
            f_y: float = 500,
            gamma_s: float = DesignConstants.GAMMA_S_DEFAULT,
            gamma_c: float = DesignConstants.GAMMA_C_DEFAULT,
            as_square=False,
            remove_conc_to_rebar=False
        ):
        self.global_index = CoreBeam._instance_counter
        CoreBeam._instance_counter += 1
        self.local_index = None  # Set by CoreLevel
        self.level = None  # Set by CoreLevel
        
        super().__init__(
            f_ck_28, concrete_poly, rebar_centers, rebar_diameters,
            c_class, t_ref, k_E, f_y, gamma_s, gamma_c,
            as_square, remove_conc_to_rebar
        )


class _Connection:
    """Internal connection data - managed by CoreLevel."""
    
    def __init__(self, core1, core2, beam, beam_length, fixed_side_1, fixed_side_2):
        self.core1 = core1
        self.core2 = core2
        self.beam = beam
        self.beam_length = float(beam_length)
        self.fixed_side_1 = bool(fixed_side_1)
        self.fixed_side_2 = bool(fixed_side_2)
        if (not self.fixed_side_1) and (not self.fixed_side_2):
            raise ValueError("At least one side of a connecting beam must be fixed")
    
    def other_core(self, core):
        """Return the core on the other side of this connection."""
        if core is self.core1:
            return self.core2
        elif core is self.core2:
            return self.core1
        else:
            raise ValueError("Given core is not part of this connection")
    
    def is_fixed_at(self, core):
        """Return whether the connection is fixed at the given core."""
        if core is self.core1:
            return self.fixed_side_1
        elif core is self.core2:
            return self.fixed_side_2
        else:
            raise ValueError("Given core is not part of this connection")


class CoreLevel:
    _instance_counter = 0
    
    def __init__(self, height, core_sections, core_centroids=None, F_ext=0, e_ext=0):
        self.height = float(height)
        self.index = CoreLevel._instance_counter
        CoreLevel._instance_counter += 1
        
        self.core_sections = []
        if not isinstance(core_centroids, (tuple, list, np.ndarray)):
            core_centroids = np.zeros([len(core_sections), 2], float)
        self.core_centroids = ar(core_centroids)
        self.beams = []
        self.connections = []
        self.F_ext = float(F_ext) # N compression
        self.e_mid = float(e_ext) # mm eccentricity from geometric centroid
        
        for i, (core, centroid) in enumerate(zip(core_sections, core_centroids)):
            if not isinstance(core, CoreSection):
                raise TypeError(f"Expected CoreSection, got {type(core).__name__}")
            core = core.copy().move(centroid - core.v_section.concrete_poly.centroid)
            core.local_index = i
            core.level = self
            self.core_sections.append(core)
            
        self.f_ck = self.core_sections[0].f_ck
        self.c_class = self.core_sections[0].c_class
        self.t_ref = self.core_sections[0].t_ref
        self.k_E = self.core_sections[0].k_E
        self.f_y = self.core_sections[0].f_y
        self.gamma_s = self.core_sections[0].gamma_s
        self.gamma_c = self.core_sections[0].gamma_c
        
        if len(self.core_sections) > 1:
            for section in self.core_sections[1:]:
                s = section.v_section
                if (not np.isclose(self.f_ck , s.f_ck)) or (not np.isclose(self.t_ref , s.t_ref)) or (not np.isclose(self.f_y , s.f_y)) or (not np.isclose(self.gamma_s , s.gamma_s)) or (not np.isclose(self.gamma_c , s.gamma_c) or self.c_class != s.c_class):
                    print("Warning - core sections with varying material properties present. The material properties of the first core section will be used")
    
    
    def y_e(self, t=None, creep=0):
        return self.e(t, creep) + self.composite_section().concrete_poly.elastic_na(t, creep)
    
    def e_ext(self, t=None, creep=0):
        comp = self.composite_section()
        na = comp.elastic_na(t, creep)
        na_geom = (comp.concrete_poly.y_max + comp.concrete_poly.y_min)/2
        return self.e_mid + na_geom - na
    
    def M_c(self, t=None, creep=0, F_c=None):
        if F_c is None: F_c = self.F_c
        return self.F_c*self.e(t, creep)
    
    def add_connection(self, core1, core2, beam, fixed_side_1=True, fixed_side_2=True):
        """
        Add a connection between two cores via a beam.
        
        Parameters
        ----------
        core1, core2 : CoreSection
            The two cores to connect. Must belong to this level.
        beam : CoreBeam
            The beam forming the connection.
        fixed_side_1 : bool
            Whether the connection is fixed (moment-resisting) at core1.
        fixed_side_2 : bool
            Whether the connection is fixed (moment-resisting) at core2.
        
        Returns
        -------
        _Connection
            The created connection object.
        """
        if core1 not in self.core_sections:
            raise ValueError(f"core1 (index {core1.global_index}) does not belong to this level")
        if core2 not in self.core_sections:
            raise ValueError(f"core2 (index {core2.global_index}) does not belong to this level")
        if not fixed_side_1 and not fixed_side_2:
            raise ValueError("At least one side must be fixed")
        if not isinstance(beam, CoreBeam):
            raise TypeError(f"Expected CoreBeam, got {type(beam).__name__}")
        
        conn = _Connection(core1, core2, beam, fixed_side_1, fixed_side_2)
        self.connections.append(conn)
        
        if beam not in self.beams:
            beam.local_index = len(self.beams)
            beam.level = self
            self.beams.append(beam)
        
        return conn
    
    def get_connections_for(self, core):
        """Return all connections involving the given core."""
        if core not in self.core_sections:
            raise ValueError("Core does not belong to this level")
        return [c for c in self.connections if c.core1 is core or c.core2 is core]
    
    def get_adjacent_cores(self, core):
        """Return all cores directly connected to the given core."""
        return [c.other_core(core) for c in self.get_connections_for(core)]
    
    def _build_adjacency(self):
        """Build adjacency dict from current connections."""
        adj = {core: set() for core in self.core_sections}
        for conn in self.connections:
            adj[conn.core1].add(conn.core2)
            adj[conn.core2].add(conn.core1)
        return adj
    
    def _find_connected_components(self):
        """Return list of sets, each set being a connected component."""
        adj = self._build_adjacency()
        unvisited = set(self.core_sections)
        components = []
        
        while unvisited:
            start = next(iter(unvisited))
            component = set()
            stack = [start]
            
            while stack:
                current = stack.pop()
                if current in component:
                    continue
                component.add(current)
                unvisited.discard(current)
                stack.extend(adj[current] - component)
            
            components.append(component)
        
        return components
    
    def check_fully_connected(self):
        """
        Check connectivity and print diagnostic info.
        
        Returns
        -------
        bool
            True if all core sections are connected, False otherwise.
        """
        if len(self.core_sections) == 0:
            print(f"Level {self.index}: No core sections.")
            return True
        
        if len(self.core_sections) == 1:
            print(f"Level {self.index}: Single core section (trivially connected).")
            return True
        
        components = self._find_connected_components()
        
        if len(components) == 1:
            print(f"Level {self.index}: All {len(self.core_sections)} core sections are connected.")
            return True
        
        print(f"Level {self.index}: Found {len(components)} isolated groups:")
        for i, comp in enumerate(components):
            indices = sorted(c.local_index for c in comp)
            print(f"  Group {i + 1}: core sections {indices}")
        
        return False
    
    def require_fully_connected(self):
        """
        Raise exception if not fully connected.
        
        Call this at the start of any analysis method to ensure
        valid topology before proceeding.
        
        Raises
        ------
        ValueError
            If the core sections are not fully connected.
        """
        if len(self.core_sections) <= 1:
            return
        
        components = self._find_connected_components()
        if len(components) > 1:
            isolated_groups = [sorted(c.local_index for c in comp) for comp in components]
            raise ValueError(
                f"Level {self.index} has {len(components)} disconnected groups: {isolated_groups}. "
                "All core sections must be connected before analysis."
            )
    
    @property
    def n_cores(self):
        return len(self.core_sections)
    
    @property
    def n_beams(self):
        return len(self.beams)
    
    @property
    def n_connections(self):
        return len(self.connections)
    
    def composite_section(self):
        sections = [s.v_section for s in self.core_sections]
        bar_dias = [b for s in self.core_sections for b in s.v_dias]
        bar_centres = [b for s in self.core_sections for b in s.v_centres]
        section = SimplePolygon.boolean_union_all(sections)
        return RCSection(self.f_ck, section, bar_centres, bar_dias, self.c_class, self.t_ref, self.k_E, self.f_y, self.gamma_s, self.gamma_c, True, False)
    
    def non_composite_uncracked_stiffness(self, t=None, creep=0):
        result = 0
        for core in self.core_sections:
            result += core.I_u(t, creep)
        return result
    
    def non_composite_cracked_stiffness(self, t=None, creep=0, e=None):
        if e is None: e = self.e_ext(t, creep)
        result = 0
        for core in self.core_sections:
            result += core.I_c(t, creep, e=e)
        return result
    
    def full_composite_uncracked_stiffness(self, t=None, creep=0):
        self.require_fully_connected()
        return self.composite_section().I_u(t, creep)
    
    def full_composite_cracked_stiffness(self, t=None, creep=0, e=None):
        if e is None: e = self.e_ext(t, creep)
        self.require_fully_connected()
        return self.composite_section().I_c(t, creep, e=e)
    
    def non_composite_M_cr(self, t=None, creep=0, F_c=None):
        if F_c is None: F_c = self.F_c
        result = 0
        for core in self.core_sections:
            result += core.M_cr(t, creep, F_c=F_c)
        return result
    
    def full_composite_M_cr(self, t=None, creep=0, F_c=None):
        if F_c is None: F_c = self.F_c
        self.require_fully_connected()
        return self.composite_section().M_cr(t, creep, F_c=F_c)
    
    def weight_properties(self):
        comp = self.composite_section()
        A_c = comp.A_c
        h = self.height
        wall_vol = A_c*h/1e6
        
        beam_vol = 0
        for connection in self.connections:
            beam = connection.beam
            beam_vol += connection.length*beam.concrete_poly.A_c/1e6
        return (wall_vol + beam_vol)*25

class Core:
    def __init__(self, core_levels, mesh_spacing=100):
        self.core_levels = core_levels
        for lev in self.core_levels:
            assert isinstance(lev, CoreLevel)
            lev.require_fully_connected()
        
        self.N_levels = len(self.core_levels)
        self.V_g_ext = np.zeros(self.n_levels, float)
        self.V_q_ext = np.zeros(self.n_levels, float)
        self.H_g_ext = np.zeros(self.n_levels, float)
        self.H_q_ext = np.zeros(self.n_levels, float)
        self.H_w = np.zeros(self.n_levels, float)
        self.e_ext_mid = np.zeros(self.n_levels, float)
        self.N_mesh = int(self.height//(float(mesh_spacing)+1))
        self.mesh_spacing = float(self.height/(self.N_mesh-1))
        self.z = np.linspace(0, self.height, self.N_mesh)
    
    def level_of_node(self, node_ind):
        h = node_ind*self.mesh_spacing
        btm_levels = self.base_levels
        top_levels = self.top_levels
        return next(i for i, (btm, top) in enumerate(zip(btm_levels, top_levels)) if h >= btm and h <= top)
    
    def btm_node_of_level(self, lvl_ind):
        if lvl_ind == 0: return 0
        btm_level = self.base_levels[lvl_ind]
        return int(btm_level/self.mesh_spacing)
    
    def top_node_of_level(self, lvl_ind):
        if lvl_ind == self.N_levels-1: return self.N_mesh-1
        top_level = self.top_levels[lvl_ind]
        return int(top_level/self.mesh_spacing)
    
    @property
    def heights(self):
        return ar([fl.height for fl in self.core_levels], float)
    
    @property
    def top_levels(self):
        return np.cumsum(self.heights)
    
    @property
    def base_levels(self):
        return self.top_levels - self.heights
    
    @property
    def mid_heights(self):
        h = self.heights
        # At junction i: half of panel i + half of panel i+1
        influence = np.zeros(len(h) + 1)
        influence[:-1] += h / 2  # Contribution from below
        influence[1:] += h / 2   # Contribution from above
        return influence
    
    @property
    def height(self): return sum(self.heights)
    
    def add_wind_load_to_all(self, W_per_metre_height):
        W_per_metre_height  = float(W_per_metre_height)
        W = self.mid_heights*W_per_metre_height
        self.H_w += W
    
    def add_wind_load(self, W_per_metre_height, floor_ind):
        W_per_metre_height  = float(W_per_metre_height)
        W = self.mid_heights[floor_ind]*W_per_metre_height
        self.H_w[floor_ind] += W
    
    def add_lat_load_to_all(self, g, q):
        self.H_g_ext += float(g)
        self.H_q_ext += float(q)
    
    def add_lat_load(self, g, q, floor_ind):
        self.H_g_ext[floor_ind] += float(g)
        self.H_q_ext[floor_ind] += float(q)
    
    def add_vert_load_to_all(self, g, q):
        self.V_g_ext += float(g)
        self.V_q_ext += float(q)
    
    def add_vert_load(self, g, q, floor_ind):
        self.V_g_ext[floor_ind] += float(g)
        self.V_q_ext[floor_ind] += float(q)
        

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    t1, t2 = 600.0, 200.0
    L = 3000.0
    
    cases = []
    
    # 1. L-corner at 90 deg
    w1 = Line([[0, 0], [L, 0]])
    w2 = Line([[L, 0], [L, L]])
    cases.append(("L 90 deg", [w1, w2], [t1, t2]))
    
    # 2. L-corner at 60 deg (acute)
    a = np.deg2rad(60)
    w1 = Line([[0, 0], [L, 0]])
    w2 = Line([[L, 0], [L + L*np.cos(a), L*np.sin(a)]])
    cases.append(("L 60 deg", [w1, w2], [t1, t2]))
    
    # 3. L-corner at 120 deg (obtuse)
    a = np.deg2rad(120)
    w1 = Line([[0, 0], [L, 0]])
    w2 = Line([[L, 0], [L + L*np.cos(a), L*np.sin(a)]])
    cases.append(("L 120 deg", [w1, w2], [t1, t2]))
    
    # 4. T-junction: 200mm stem into middle of 600mm flange
    w1 = Line([[0, 0], [L, 0]])
    w2 = Line([[L/2, 0], [L/2, L]])
    cases.append(("T thin-stem", [w1, w2], [t1, t2]))
    
    # 5. T-junction: 600mm stem into middle of 200mm flange
    w1 = Line([[0, 0], [L, 0]])
    w2 = Line([[L/2, 0], [L/2, L]])
    cases.append(("T thick-stem", [w1, w2], [t2, t1]))
    
    # 6. Cross at 90 deg
    w1 = Line([[0, 0], [L, 0]])
    w2 = Line([[L/2, -L/2], [L/2, L/2]])
    cases.append(("Cross 90 deg", [w1, w2], [t1, t2]))
    
    # 7. Y-junction at 45 deg
    a = np.deg2rad(45)
    w1 = Line([[0, 0], [0, L]])
    w2 = Line([[0, L], [L*np.sin(a), L + L*np.cos(a)]])
    w3 = Line([[0, L], [-L*np.sin(a), L + L*np.cos(a)]])
    cases.append(("Y 45 deg", [w1, w2, w3], [t1, t2, t2]))
    
    ind = 6
    for i, (name, wall_lines, thicknesses) in enumerate(cases):
        if i != ind: continue
        cr = CoreSection(
            f_ck=40, wall_lines=wall_lines, wall_thk=thicknesses,
            v_dias=16, v_spacing=200, h_dias=16, h_spacing=200,
        )
        cr.v_section.plot(True)
        plt.title(f"{name}  (thicknesses: {thicknesses})")
        plt.show()

exit()

if __name__ == "__main__":
    w1 = PolyLine([[0,0],[2000,0],[2000,3000]])
    w2 = Line([[2000,500],[1000,500]])
    cr = CoreSection(40, [w1, w2], 200, 16, 200, 16, 200)
    cr.v_section.plot(True)
    print(f"M_Rd = {cr.v_section.M_Rd()/1e6:.0f}")
    print(f"I_u = {cr.v_section.I_u()/1e9:.0f}")
    print(f"I_c = {cr.v_section.I_c()/1e9:.0f}")
    print(f"M_cr = {cr.v_section.M_cr()/1e6:.0f}")
                
            