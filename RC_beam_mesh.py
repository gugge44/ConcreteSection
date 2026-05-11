"""
Reinforced Concrete Beam Analysis Tool - BugPoly Version

A comprehensive tool for analyzing reinforced concrete sections according to Eurocode 2
(EN 1992-1-1). Uses BugPoly geometry classes instead of ladybug_geometry.

Author: Gustaf
Version: 2.1 - Enhanced Optimization & Fixed M_cr
"""

import numpy as np
from numpy import array as ar
from numpy import exp, sqrt, log as ln, pi, cos, sin
from scipy.optimize import minimize_scalar, OptimizeResult
from typing import Optional, Tuple, Dict, List, Union
from dataclasses import dataclass
from functools import lru_cache
import warnings
from scipy.optimize import brentq
from scipy.optimize import root_scalar, minimize
from scipy.interpolate import make_interp_spline
import matplotlib.pyplot as plt
from sys import exit
from functools import cached_property

# Import BugPoly geometry classes
from BugPoly import SimplePolygon, Polygon
from BugBasics import Point, Line, PolyLine, Ray
from Mesh import Mesh
from Notation import eng_format as ef


# =============================================================================
# CONSTANTS AND CONFIGURATION
# =============================================================================

class DesignConstants:
    """Eurocode 2 design constants and factors"""
    # Material factors
    GAMMA_C_DEFAULT = 1.5  # Partial factor for concrete
    GAMMA_S_DEFAULT = 1.15  # Partial factor for steel
    
    # Concrete stress-strain parameters
    ALPHA_CC = 0.85  # Long-term strength reduction factor
    EPS_C2 = 0.002  # Strain at peak stress (parabola-rectangle)
    EPS_CU = 0.0035  # Ultimate concrete strain
    
    # Steel parameters
    E_S = 200000  # MPa - Steel Young's modulus
    K_STEEL = 1.08  # Characteristic strain hardening ratio
    EPS_UK = 0.05  # Characteristic ultimate strain
    
    # Numerical tolerances
    TOLERANCE = 1e-6
    MAX_ITERATIONS = 100
    
    # Creep limits
    CREEP_STRESS_LIMIT_WARN = 0.4  # Non-linear creep starts
    CREEP_STRESS_LIMIT_MAX = 0.6  # Maximum allowable stress ratio


# =============================================================================
# CONCRETE MATERIAL MODEL
# =============================================================================

@dataclass
class ConcreteProperties:
    """Time-dependent concrete properties at a specific age"""
    f_ck: float  # Characteristic cylinder strength [MPa]
    f_cm: float  # Mean cylinder strength [MPa]
    f_cd: float  # Design compressive strength [MPa]
    f_ctm: float  # Mean tensile strength [MPa]
    E_cm: float  # Secant modulus [MPa]
    E_c: float  # Tangent modulus [MPa]
    beta_cc: float  # Time-dependent strength factor
    age: Optional[float]  # Age in days


class ConcreteMaterial:
    """
    Concrete material model following Eurocode 2.
    
    Handles time-dependent strength development, elastic properties,
    and design stress-strain relationships.
    """
    
    CEMENT_CLASSES = {'CR': 0.0, 'CN': 0.2, 'CS': 0.3}
    STRENGTH_CLASSES = {
        (float('-inf'), 35): 0.3,
        (35, 60): 0.2,
        (60, float('inf')): 0.1
    }
    
    def __init__(
        self,
        f_ck_28: float,
        c_class: str = 'CN',
        t_ref: float = 28,
        k_E: float = 9500,
        gamma_c: float = DesignConstants.GAMMA_C_DEFAULT,
        gamma_s: float = DesignConstants.GAMMA_S_DEFAULT
    ):
        """
        Initialize concrete material.
        
        Parameters
        ----------
        f_ck_28 : float
            Characteristic cylinder strength at 28 days [MPa]
        c_class : str
            Cement class: 'CR' (rapid), 'CN' (normal), 'CS' (slow)
        t_ref : float
            Reference age for strength calculations [days]
        k_E : float
            Modulus coefficient (typically 9500)
        gamma_c : float
            Partial factor for concrete
        gamma_s : float
            Partial factor for steel
        """
        # Validate inputs
        if c_class not in self.CEMENT_CLASSES:
            raise ValueError(f"Invalid cement class '{c_class}'. Must be CR, CN, or CS.")
        if f_ck_28 <= 0:
            raise ValueError(f"f_ck_28 must be positive, got {f_ck_28}")
        if t_ref <= 0:
            raise ValueError(f"t_ref must be positive, got {t_ref}")
            
        self.f_ck_28 = float(f_ck_28)
        self.c_class = c_class
        self.t_ref = float(t_ref)
        self.k_E = float(k_E)
        self.gamma_c = gamma_c
        self.gamma_s = gamma_s
        
        # Calculate 28-day properties
        self.f_cm_28 = self.f_ck_28 + 8
        self.f_ctm_28 = self._calculate_tensile_strength(self.f_ck_28)
        self.E_cm_28 = self.k_E * self.f_cm_28 ** (1/3)
        self.E_c_28 = self._calculate_tangent_modulus(self.E_cm_28, self.f_cm_28)
        
        # Calculate strength development parameters
        self.s_c = self._calculate_strength_coefficient()
        
        # Calculate reference properties (at t_ref)
        self.f_ck_ref = self.f_ck_28 * exp(self.s_c * (1 - sqrt(28/self.t_ref)))
        self.f_cm_ref = self.f_ck_ref + 8
        self.f_ctm_ref = self._calculate_tensile_strength(self.f_ck_ref)
        self.E_cm_ref = self.k_E * self.f_cm_ref ** (1/3)
        self.E_c_ref = self._calculate_tangent_modulus(self.E_cm_ref, self.f_cm_ref)
        
        # Strain parameters
        self.eps_c2 = DesignConstants.EPS_C2
        self.eps_cu = DesignConstants.EPS_CU
        
        self.alpha_bs = 800 if self.c_class == "CS" else 700 if self.c_class == "CN" else 600
        self.alpha_ds = 3 if self.c_class == "CS" else 4 if self.c_class == "CN" else 6
    
    def _calculate_tensile_strength(self, f_ck: float) -> float:
        """Calculate mean tensile strength from characteristic strength"""
        if f_ck <= 50:
            return 0.3 * f_ck ** (2/3)
        else:
            return 1.1 * f_ck ** (1/3)
    
    def _calculate_tangent_modulus(self, E_cm: float, f_cm: float) -> float:
        """Calculate tangent modulus from secant modulus"""
        return E_cm * max(1.0, 1 / (0.8 + 0.2 * f_cm / 88))
    
    def _calculate_strength_coefficient(self) -> float:
        """Calculate strength development coefficient s_c"""
        # Cement class contribution
        s_c1 = self.CEMENT_CLASSES[self.c_class]
        
        # Strength class contribution
        for (lower, upper), value in self.STRENGTH_CLASSES.items():
            if lower < self.f_ck_28 <= upper:
                s_c2 = value
                break
        
        return s_c1 + s_c2
    
    def beta_cc(self, t: Optional[float]) -> float:
        """
        Calculate time-dependent strength factor.
        
        Parameters
        ----------
        t : float or None
            Age of concrete [days]. None returns 1.0 (reference age)
            
        Returns
        -------
        float
            Strength factor β_cc(t)
        """
        if t is None or t >= self.t_ref or np.isclose(t, 0):
            return 1.0
        
        exponent = self.s_c * sqrt(28) / sqrt(self.t_ref) * (1 - sqrt(self.t_ref / t))
        return exp(exponent)
    
    def get_properties(self, t: Optional[float] = None) -> ConcreteProperties:
        """
        Get all concrete properties at specified age.
        
        Parameters
        ----------
        t : float or None
            Age of concrete [days]. None uses reference age
            
        Returns
        -------
        ConcreteProperties
            Complete set of material properties
        """
        beta = self.beta_cc(t)
        f_cm = beta * self.f_cm_ref
        
        return ConcreteProperties(
            f_ck=f_cm - 8,
            f_cm=f_cm,
            f_cd=DesignConstants.ALPHA_CC * (f_cm - 8) / self.gamma_c,
            f_ctm=beta ** 0.6 * self.f_ctm_ref,
            E_cm=beta ** (1/3) * self.E_cm_ref,
            E_c=self._calculate_tangent_modulus(
                beta ** (1/3) * self.E_cm_ref, f_cm
            ),
            beta_cc=beta,
            age=t
        )
    
    @property
    def n(self):
        if self.f_ck_28 <= 50: return 2.0
        else: return 1.4 + 23.4*((90-self.f_ck_28)/100)**4
    
    # Convenience methods for individual properties
    def f_cm(self, t: Optional[float] = None) -> float:
        """Mean cylinder strength at age t [MPa]"""
        return self.beta_cc(t) * self.f_cm_ref
    
    def f_ck(self, t: Optional[float] = None) -> float:
        """Characteristic cylinder strength at age t [MPa]"""
        return self.f_cm(t) - 8
    
    def f_ctm(self, t: Optional[float] = None) -> float:
        """Mean tensile strength at age t [MPa]"""
        return self.beta_cc(t) ** 0.6 * self.f_ctm_ref
    
    def E_cm(self, t: Optional[float] = None) -> float:
        """Secant modulus at age t [MPa]"""
        return self.beta_cc(t) ** (1/3) * self.E_cm_ref
    
    def E_c(self, t: Optional[float] = None) -> float:
        """Tangent modulus at age t [MPa]"""
        f_cm = self.f_cm(t)
        E_cm = self.E_cm(t)
        return self._calculate_tangent_modulus(E_cm, f_cm)
    
    def f_cd(self, t: Optional[float] = None) -> float:
        """Design compressive strength at age t [MPa]"""
        return DesignConstants.ALPHA_CC * self.f_ck(t) / self.gamma_c
    
    def poisson_ratio(self, cracked: bool) -> float:
        """Poisson's ratio (0.0 if cracked, 0.2 if uncracked)"""
        return 0.0 if cracked else 0.2
    
    def design_stress(self, eps: float, t: Optional[float] = None) -> float:
        """
        Calculate design stress from strain (parabola-rectangle diagram).
        
        Parameters
        ----------
        eps : float
            Concrete compressive strain (positive in compression)
        t : float or None
            Age of concrete [days]
            
        Returns
        -------
        float
            Design stress [MPa]
        """
        if eps <= 0:
            return 0.0
        
        f_cd = self.f_cd(t)
        
        if eps <= self.eps_c2:
            # Parabolic branch
            return f_cd * eps/self.eps_c2#(1 - (1 - eps / self.eps_c2) ** 2)
        elif eps <= self.eps_cu:
            # Constant stress branch
            return f_cd
        else:
            # Beyond ultimate strain
            return 0.0
    
    def __repr__(self) -> str:
        return (f"ConcreteMaterial(f_ck={self.f_ck_28:.0f} MPa, "
                f"class={self.c_class}, age_ref={self.t_ref:.0f} days)")


# =============================================================================
# REINFORCED CONCRETE SECTION
# =============================================================================

class RCSection:
    """
    Reinforced concrete section analysis using BugPoly geometry.
    
    Handles geometric properties, neutral axis calculations, moment capacity,
    and cracking analysis for RC sections.
    """
    
    def __init__(
        self,
        f_ck_28: float,
        concrete_poly: Union[SimplePolygon, Polygon, np.ndarray, list],
        rebar_centers: np.ndarray,
        rebar_diameters: np.ndarray,
        c_class: str = 'CN',
        t_ref: float = 28,
        k_E: float = 9500,
        f_y: float = 500,
        gamma_s: float = DesignConstants.GAMMA_S_DEFAULT,
        gamma_c: float = DesignConstants.GAMMA_C_DEFAULT,
        as_square = False,
        remove_conc_to_rebar = True,
        no_checks=False
    ):
        """
        Initialize RC section with BugPoly geometry.
        
        Parameters
        ----------
        f_ck_28 : float
            Concrete characteristic strength at 28 days [MPa]
        concrete_poly : SimplePolygon, Polygon, array, or list
            Concrete section geometry
        rebar_centers : ndarray
            Rebar center coordinates, shape (n_bars, 2) [mm]
        rebar_diameters : ndarray
            Rebar diameters [mm]
        c_class : str
            Cement class
        t_ref : float
            Reference age [days]
        k_E : float
            Modulus coefficient
        f_y : float
            Reinforcement yield strength [MPa]
        gamma_s : float
            Partial factor for steel
        gamma_c : float
            Partial factor for concrete
        """
        if not isinstance(rebar_diameters, (tuple, list, np.ndarray)):
            rebar_diameters = np.ones(len(rebar_centers), float)*rebar_diameters
        else:
            rebar_diameters = ar(rebar_diameters, float)
        self._rebar_centers = ar(rebar_centers, float)
        self._rebar_diameters = rebar_diameters
        # Initialize material
        self.concrete = ConcreteMaterial(f_ck_28, c_class, t_ref, k_E, gamma_c, gamma_s)
        self.gamma_s = gamma_s
        self.gamma_c = gamma_c
        # Steel properties
        self.E_s = DesignConstants.E_S
        self.f_y = float(f_y)
        self.f_yd = self.f_y / gamma_s
        self.eps_yk = self.f_y / self.E_s
        self.eps_yd = self.f_yd / self.E_s
        self.eps_uk = DesignConstants.EPS_UK
        self.eps_ud = self.eps_uk / gamma_s
        self.k_steel = DesignConstants.K_STEEL
        self.f_uk = self.f_y * self.k_steel
        self.f_ud = self.f_yd * self.k_steel
        # Process geometry - convert to Polygon if needed
        if isinstance(concrete_poly, Polygon):
            concrete_poly = concrete_poly
        elif isinstance(concrete_poly, SimplePolygon):
            concrete_poly = Polygon(concrete_poly)
        else:
            # Array-like input
            coords = ar(concrete_poly, dtype=float)
            simple = SimplePolygon(coords)
            concrete_poly = Polygon(simple)
        
        # Store rebar information
        self.as_square = as_square
        self.remove_conc_to_rebar = remove_conc_to_rebar
        self.square_dias = sqrt((rebar_diameters/2)**2*pi)
        if as_square:
            rebar_polys = ar([[cos(ang), sin(ang)] for ang in np.linspace(pi/4, 2*pi+pi/4, 4, endpoint=False)], float)[None,...]*self.square_dias[:,None,None]/sqrt(2) + self._rebar_centers[:,None,:]
        else:
            rebar_polys = ar([[cos(ang), sin(ang)] for ang in np.linspace(0, 2*pi, 8, endpoint=False)], float)[None,...]*rebar_diameters[:,None,None]*1.0522/2 + self._rebar_centers[:,None,:]
        rebar_poly = Polygon(rebar_polys, tol=concrete_poly.tol, repair=False)
        self.rebar_poly = rebar_poly
        self.full_poly = concrete_poly.copy()
        if remove_conc_to_rebar:
            concrete_poly = concrete_poly.boolean_difference(rebar_poly)
        self.concrete_mesh = Mesh(boundary=concrete_poly)
        
        # Effective depth (distance from top to bottom reinforcement)
        self.bounds = self.full_poly.bounds
        self.x_min, self.y_min, self.x_max, self.y_max = self.bounds
        # Initialize caches
        self._clear_cache()
    
    def _clear_cache(self):
        """Clear cached property calculations"""
        self._cached_properties = {}
    
    def _get_cache(self, name, t, creep):
        return self._cached_properties.get((name, int(t) if t is not None else t, int(creep*1e3)), None)
    
    def _set_cache(self, name, t, creep, value):
        self._cached_properties[(name, int(t) if t is not None else t, int(creep*1e3))] = value
    
    def _has_cache(self, name, t, creep):
        return self._get_cache(name, t, creep) is not None
    
    @cached_property
    def rebar_mask(self):
        if not self.remove_conc_to_rebar:
            return self.concrete_mesh.tri.is_point_inside(self.rebar_centers)
        else:
            return np.zeros(len(self.rebar_diameters), bool)
        
    @cached_property
    def A_s_arr(self): return ar([(d/2)**2*pi for d in self.rebar_diameters], float)
    @property
    def A_s(self): return sum(self.A_s_arr)
    @property
    def A_full(self): return self.full_poly.area
    @cached_property
    def A_c(self): return sum(self.triangle_areas) - sum(self.A_s_arr*self.rebar_mask)
    @property
    def rebar_centers(self): return self._rebar_centers
    @property
    def rebar_diameters(self): return self._rebar_diameters
    @property
    def triangle_centers(self): return self.concrete_mesh.tri.centroids
    @property
    def triangle_areas(self): return self.concrete_mesh.tri.area
    @cached_property
    def c_s(self): return np.sum(self.rebar_centers*self.A_s_arr[:,None], axis=0)/self.A_s
    @property
    def c_c(self): return self.concrete_mesh.tri.centroid
    @property
    def B(self): return self.bounds[2]-self.bounds[0]
    @property
    def H(self): return self.bounds[3]-self.bounds[1]
    @cached_property
    def na_ref(self): return self.full_poly.centroid
    @cached_property
    def na_ref_y(self): return self.na_ref[1]
    
    @staticmethod
    def _Ixx_about(tri, y):
        """Second moment of TriElements about horizontal axis at y.
        
        Uses parallel axis theorem: I_y = I_centroid + A*(c_y - y)^2.
        TriElements.Ixx(centroid) only works when centroid IS the centroid
        of the shape. This helper works for any y.
        """
        return tri.Ixx() + tri.total_area * (tri.centroid[1] - y)**2
    
    @property
    def d_ef(self):
        y_s = self.rebar_centers[:,1]
        y_max = self.bounds[3]
        return y_max - np.min(y_s)
        
    @property
    def d_min(self):
        y_s = self.rebar_centers[:,1]
        y_max = self.bounds[3]
        return y_max - np.max(y_s[y_s<y_max])
    
    def alpha_e(self, t, creep):
        E_c = self.concrete.E_c(t)
        alpha_e = np.ones(len(self.rebar_diameters), float)*self.E_s * (1 + creep) / E_c
        alpha_e[self.rebar_mask] -= 1
        return alpha_e
    
    def A_ef(self, t, creep):
        return self.A_c + sum(self.A_s_arr*self.alpha_e(t, creep))
    
    def c_ef(self, t, creep):
        alpha_A = self.A_s_arr * self.alpha_e(t, creep)
        return (self.A_c * self.c_c + np.sum(alpha_A[:, None] * self.rebar_centers, axis=0)) / self.A_ef(t, creep)
        
    def d_ef_A_y_cut(self, y_cut):
        mask = self.rebar_centers[:,1] < y_cut
        return self.bounds[3] - self.rebar_centers[mask, 1], self.A_s_arr[mask]
    
    def d_ef_A_x_cut(self, x_cut):
        mask = self.rebar_centers[:,0] < x_cut
        return self.bounds[2] - self.rebar_centers[mask, 0], self.A_s_arr[mask]
    
    def concrete_stresses(self, eps_na, phi, t=None):
        na_y = self.na_ref_y
        f_cd = self.concrete.f_cd(t)
        eps_c2 = self.concrete.eps_c2
        eps_tri = (self.concrete_mesh.tri.centroids[:,1] - na_y)*phi + eps_na
        sig_tri = np.minimum(np.maximum(eps_tri*f_cd/eps_c2, 0), f_cd)
        return sig_tri
    
    def concrete_forces(self, eps_na, phi, t=None):
        return self.concrete_stresses(eps_na, phi, t)*self.concrete_mesh.tri.area
    
    def rebar_stresses(self, eps_na, phi, t=None):
        na_y = self.na_ref_y
        f_yd = self.f_yd
        f_ud = self.f_ud
        eps_yd = self.eps_yd
        eps_ud = self.eps_ud
        E_s = self.E_s
        f_cd = self.concrete.f_cd(t)
        eps_c2 = self.concrete.eps_c2
        eps = (self.rebar_centers[:,1] - na_y)*phi + eps_na
        eps_abs = np.abs(eps)
        elastic_mask = eps_abs < eps_yd
        plastic_mask = (eps_abs <= eps_ud)&~elastic_mask
        result = np.zeros_like(eps)
        result[elastic_mask] = eps[elastic_mask]*E_s
        result[plastic_mask] = (f_yd + (f_ud-f_yd)*(eps_abs[plastic_mask]-eps_yd)/(eps_ud-eps_yd))*np.sign(eps[plastic_mask])
        
        if not self.remove_conc_to_rebar:
            c_stress = np.minimum(np.maximum(eps[self.rebar_mask]*f_cd/eps_c2,0),f_cd)
            result[self.rebar_mask] -= c_stress
        
        return result
    
    def rebar_forces(self, eps_na, phi, t=None):
        return self.rebar_stresses(eps_na, phi, t)*self.A_s_arr
    
    def copy(self):
        """Create a deep copy of this section"""
        return RCSection(
            self.concrete.f_ck_28,
            self.full_poly.copy(),
            self.rebar_centers.copy(),
            self.rebar_diameters.copy(),
            self.concrete.c_class,
            self.concrete.t_ref,
            self.concrete.k_E,
            self.f_y,
            self.gamma_s,
            self.gamma_c,
            as_square=self.as_square,
            remove_conc_to_rebar=self.remove_conc_to_rebar,
            no_checks=True
        )
    
    def rotate(self, angle_ccw, centroid=None):
        if centroid is None:
            centroid = self.full_poly.centroid
        new_conc_poly = self.full_poly.copy().rotate(angle_ccw, centroid=centroid)
        # Negate angle to match BugPoly's effective rotation direction
        c, s = np.cos(np.deg2rad(-angle_ccw)), np.sin(np.deg2rad(-angle_ccw))
        p = np.asarray(self.rebar_centers, float) - centroid
        new_rebar_centres = np.column_stack([c*p[:,0] - s*p[:,1], s*p[:,0] + c*p[:,1]]) + centroid
        return RCSection(
            self.concrete.f_ck_28,
            new_conc_poly,
            new_rebar_centres,
            self.rebar_diameters.copy(),
            self.concrete.c_class,
            self.concrete.t_ref,
            self.concrete.k_E,
            self.f_y,
            self.gamma_s,
            self.gamma_c,
            as_square=self.as_square,
            remove_conc_to_rebar=self.remove_conc_to_rebar,
            no_checks=True
        )
    
    def move(self, vector):
        return RCSection(
            self.concrete.f_ck_28,
            self.full_poly.move(vector),
            self.rebar_centers + vector[None,...],
            self.rebar_diameters.copy(),
            self.concrete.c_class,
            self.concrete.t_ref,
            self.concrete.k_E,
            self.f_y,
            self.gamma_s,
            self.gamma_c,
            as_square=self.as_square,
            remove_conc_to_rebar=self.remove_conc_to_rebar
        )
    
    # =========================================================================
    # NEUTRAL AXIS CALCULATION (ENHANCED WITH STRICT OPTIMIZATION)
    # =========================================================================
    
    def elastic_na(self, t=None, creep=0.0):
        return self.c_ef(t,creep)[1]
    
    def W_uc(self, t=None, creep=0.0, na=None):
        if na is None:
            na = self.elastic_na(t, creep)
        I_u = self.I_u(t, creep)
        z = na - self.y_min
        return I_u/z

    def elastic_cracking_depth(self, t=None, creep=0.0, e=None):
        """
        Cracked elastic transformed section.
        
        Returns y_cut (zero strain line / cracking boundary)
        """
        if e is None or not np.isfinite(e):
            cached = self._get_cache("elastic_cracking_depth", t, creep)
            if cached is not None:
                return cached
        y_max = self.y_max
        y_min = self.y_min
        A_full = self.A_ef(t, creep)
        
        alpha_e = self.alpha_e(t, creep)
        rebar_z = self.rebar_centers[:, 1]
        rebar_A = self.A_s_arr
        A_s_ef = alpha_e*rebar_A
        Az_s_ef = A_s_ef*rebar_z
        
        na = self.elastic_na(t, creep)
        
        clip_cache = {}
        conc_tri = self.concrete_mesh.tri
        
        def clip_above(y):
            key = round(y, 4)
            if key in clip_cache:
                return clip_cache[key]
            
            if y >= y_max:
                clip_cache[key] = None
            elif y <= y_min:
                clip_cache[key] = conc_tri
            else:
                try:
                    split = self.concrete_mesh.split_by_y(y)
                    mask = split.centroids[:,1] > y
                    above = split.masked(mask)
                    clip_cache[key] = above if (len(above) > 0 and above.total_area > 1e-6) else None
                except:
                    clip_cache[key] = None
            return clip_cache[key]
        
        def calc_W(y_cut):
            """Section modulus of cracked section about na to top fibre"""
            conc_eff = clip_above(y_cut)
            
            if conc_eff is None:
                I_c = 0.0
            else:
                I_c = conc_eff.Ixx() + conc_eff.total_area * (conc_eff.centroid[1] - na)**2
            
            I_s = np.sum(A_s_ef * (rebar_z - na)**2)
            I_t = I_c + I_s
            
            return I_t / (y_max - na)
        
        def calc_na_cracked(y_cut):
            """NA of cracked section"""
            conc_eff = clip_above(y_cut)
            
            if conc_eff is None:
                return np.sum(Az_s_ef) / np.sum(A_s_ef)
        
            else:
                A_c = conc_eff.total_area
                S_c = A_c * conc_eff.centroid[1]
                return (S_c + np.sum(Az_s_ef)) / (A_c + np.sum(A_s_ef))
        
        if e is None or not np.isfinite(e):
            # Pure bending
            y_cut = na
            for _ in range(50):
                new_y_cut = calc_na_cracked(y_cut)
                if abs(new_y_cut - y_cut) < 1e-3:
                    break
                y_cut = new_y_cut
            self._set_cache("elastic_cracking_depth", t, creep, y_cut)
            return y_cut
        
        # With axial: y_cut = na - W*(y_max - na)/(e*A_full)
        y_cut = (na + y_min) / 2
        
        for _ in range(50):
            W = calc_W(y_cut)
            new_y_cut = na - W * (y_max - na) / (e * A_full)
            
            if abs(new_y_cut - y_cut) < 1e-3:
                break
            y_cut = new_y_cut
        
        return y_cut
    
    def F_integral(self, eps_na, phi, t=None, creep=0.0):
        na_y = self.na_ref_y
        eps_na = float(eps_na)
        phi = float(phi)
        f_cd = self.concrete.f_cd(t)
        eps_c2 = self.concrete.eps_c2
        if np.isclose(phi, 0):
            F_s = self.rebar_forces(eps_na, phi, t)
            return np.sum(F_s)
        y_0 = na_y + (0 - eps_na)/phi
        y_y = na_y + (eps_c2 - eps_na)/phi
        tri = self.concrete_mesh.tri
        if y_0 > self.y_min and y_0 < self.y_max:
            tri = tri.split_by_y(y_0)
        if y_y > self.y_min and y_y < self.y_max:
            tri = tri.split_by_y(y_y)
        c = tri.centroids
        pl_mask = c[:,1] > y_y
        el_mask = (c[:,1] <= y_y) & (c[:,1] > y_0)
        tri_pl = tri.masked(pl_mask)
        tri_el = tri.masked(el_mask)
        F_c_el = np.sum(tri_el.area * (tri_el.centroids[:,1] - y_0) * phi * f_cd / eps_c2)
        F_c_pl = tri_pl.area * f_cd
        F_s = self.rebar_forces(eps_na, phi, t)
        return F_c_el + F_c_pl + np.sum(F_s)
    
    def M_integral(self, eps_na, phi, t=None, creep=0.0, incl_F=False):
        na_y = self.na_ref_y
        eps_na = float(eps_na)
        phi = float(phi)
        f_cd = self.concrete.f_cd(t)
        eps_c2 = self.concrete.eps_c2
        if np.isclose(phi, 0):
            F = self.F_integral(eps_na, phi, t, creep)
            return (F, 0.0) if incl_F else 0.0
        y_0 = na_y + (0 - eps_na)/phi
        y_y = na_y + (eps_c2 - eps_na)/phi
        tri = self.concrete_mesh.tri
        if y_0 > self.y_min and y_0 < self.y_max:
            tri = tri.split_by_y(y_0)
        if y_y > self.y_min and y_y < self.y_max:
            tri = tri.split_by_y(y_y)
        c = tri.centroids
        pl_mask = c[:,1] > y_y
        el_mask = (c[:,1] <= y_y) & (c[:,1] > y_0)
        tri_pl = tri.masked(pl_mask)
        tri_el = tri.masked(el_mask)
        # Per-element forces
        F_c_el = tri_el.area * (tri_el.centroids[:,1] - y_0) * phi * f_cd / eps_c2
        F_c_pl = tri_pl.area * f_cd
        F_s = self.rebar_forces(eps_na, phi, t)
        # Moments about na
        M_c_el = np.sum(F_c_el * (tri_el.centroids[:,1] - na_y))
        M_c_pl = np.sum(F_c_pl * (tri_pl.centroids[:,1] - na_y))
        M_s = np.sum(F_s * (self.rebar_centers[:,1] - na_y))
        M_total = M_c_el + M_c_pl + M_s
        if incl_F:
            return (np.sum(F_c_el) + np.sum(F_c_pl) + np.sum(F_s), M_total)
        return M_total
   
    def M_Rd_direct(self, F_target=0.0, t=None, creep=0.0, return_F=False, ignore_boundary=False, x_d_max=None):
        #x_d_max = 0.25 for plastic
        # --- 1. Constants & Geometry ---
        y_top = self.y_max
        d_ef = self.d_ef
        
        y_btm = y_top - d_ef
        na_y = self.na_ref_y
        
        eps_cu = self.concrete.eps_cu
        eps_ud = self.eps_ud
        eps_c2 = self.concrete.eps_c2
        
        dy_mid = na_y - y_btm
        dy_top = y_top - y_btm
        # Ratio for mid-strain interpolation: eps_mid = e_b + (e_t - e_b) * ratio
        ratio = dy_mid / dy_top
        
        ys_set = self.rebar_centers[:,1]
        
        if x_d_max is not None:
            x_max = d_ef*x_d_max
        else:
            x_max = d_ef
        y_cut_min = y_top-x_max
        if y_cut_min == y_btm:
            y_cut_min += 1e-9
            
        # --- 2. Helper Functions ---
        def get_state(e_top, e_btm):
            """Converts top/btm strains to (eps_na, curvature)"""
            # Curvature calculation based on effective depth d_ef
            curvature = (e_top - e_btm) / d_ef
            e_na = e_btm + curvature * dy_mid
            return e_na, curvature
        
        def get_force(e_top, e_btm):
            e_na, phi = get_state(e_top, e_btm)
            return self.F_integral(e_na, phi, t, creep)
        
        def get_force_err(e_top, e_btm):
            return get_force(e_top, e_btm) - F_target
        
        def get_moment(e_top, e_btm):
            e_na, phi = get_state(e_top, e_btm)
            return self.M_integral(e_na, phi, t, creep)
        
        def get_btm_from_mid(e_t):
            # Solves for e_btm given e_top and fixed eps_mid=eps_c2
            # eps_c2 = e_b * (1-ratio) + e_t * ratio
            return (eps_c2 - e_t * ratio) / (1 - ratio)
                
        if not ignore_boundary:
            # =========================================================
            # PART A: BOUNDARY SEARCH (Fast Path using brentq)
            # =========================================================
            
            # Pre-calculate intersection points to define search limits
            
            # Limit for Pivot 1: The e_btm where eps_mid hits eps_c2 while e_top=eps_cu
            e_btm_max_p1 = (eps_c2 - eps_cu * ratio) / (1 - ratio)
            
            # Limit for Pivot 2: The e_top where eps_mid hits eps_c2 while e_btm=-eps_ud
            e_top_max_p2 = (eps_c2 + eps_ud * (1 - ratio)) / ratio
            
            # Limit for x_d_ratio
            e_btm_max_p12 = eps_cu + eps_cu*(y_btm-y_top)/(y_top-y_cut_min)
            e_top_min_p2 = -eps_ud + eps_ud*(y_top-y_btm)/(y_cut_min-y_btm)
            
            # --- Pivot 1: Top Controlled (Fix Top = eps_cu) ---
            bracket_1 = [-eps_ud, min(e_btm_max_p1, e_btm_max_p12)]
            
            try:
                def res_1(e_b): return get_force_err(eps_cu, e_b)
                if np.sign(res_1(bracket_1[0])) != np.sign(res_1(bracket_1[1])):
                    sol = root_scalar(res_1, bracket=bracket_1, method='brentq')
                    if sol.converged:
                        M = get_moment(eps_cu, sol.root)
                        y_na = y_btm + (y_top-y_btm)*(0-sol.root)/(eps_cu-sol.root)
                    
                        return (M, F_target, y_na) if return_F else M
            except (ValueError, ArithmeticError): pass
            
            # --- Pivot 2: Bottom Controlled (Fix Btm = -eps_ud) ---
            bracket_2 = [0.0, min(eps_cu, e_top_max_p2)]
            
            if bracket_2[1] > bracket_2[0]:
                try:
                    def res_2(e_t): return get_force_err(e_t, -eps_ud)
                    if np.sign(res_2(bracket_2[0])) != np.sign(res_2(bracket_2[1])):
                        sol = root_scalar(res_2, bracket=bracket_2, method='brentq')
                        if sol.converged:
                            M = get_moment(sol.root, -eps_ud)
                            y_na = y_btm + (y_top-y_btm)*(0+eps_ud)/(sol.root+eps_ud)
                    
                            return (M, F_target, y_na) if return_F else M
                except (ValueError, ArithmeticError): pass
            
            # --- Pivot 3: Midpoint Controlled (Fix Mid = eps_c2) ---
            upper_bound_3 = min(eps_cu, e_top_max_p2)
            bracket_3 = [eps_c2, upper_bound_3]
            
            if bracket_3[1] > bracket_3[0]:
                try:
                    def res_3(e_t): 
                        return get_force_err(e_t, get_btm_from_mid(e_t))
                    
                    if np.sign(res_3(bracket_3[0])) != np.sign(res_3(bracket_3[1])):
                        sol = root_scalar(res_3, bracket=bracket_3, method='brentq')
                        if sol.converged:
                            e_t_sol = sol.root
                            e_b_sol = get_btm_from_mid(e_t_sol)
                            M = get_moment(e_t_sol, e_b_sol)
                            y_na = y_btm + (y_top-y_btm)*(0-e_b_sol)/(e_t_sol-e_b_sol)
                    
                            return (M, F_target, y_na) if return_F else M
                except (ValueError, ArithmeticError): pass
            
        # =========================================================
        # PART B: SCALED INTERNAL SEARCH (Robust Fallback)
        # =========================================================
        # Used when solution lies strictly inside the domain boundaries.
        
        phi_max = (eps_cu-eps_c2)/(y_top-na_y)
        eps_top_max = min(eps_cu, -eps_ud + (y_top-y_btm)*phi_max)
        F_ex = get_force(eps_c2, eps_c2)
        M_ex = get_moment(eps_top_max, -eps_ud)
        
        SCALE_EPS = 1000.0  
        SCALE_MOMENT = 10/abs(M_ex)#(abs(F_target * d_ef * 0.9) + 1e6) #1.0 / M_ex#
        SCALE_FORCE = 10/abs(F_ex)#(abs(F_target) + 1000.0) #1.0 / F_ex#
        
        bounds_scaled = [
            (0.0, eps_cu * SCALE_EPS),                 # Top Strain
            (-eps_ud * SCALE_EPS, eps_ud * SCALE_EPS)  # Bottom Strain
        ]
        
        # 1. Scaled Objective: Minimize Negative Moment
        def objective(x_scaled):
            e_top = x_scaled[0] / SCALE_EPS
            e_btm = x_scaled[1] / SCALE_EPS
            M_real = get_moment(e_top, e_btm)
            return -M_real * SCALE_MOMENT
        
        # 2. Scaled Constraint: Force Equilibrium
        def const_force(x_scaled):
            e_top = x_scaled[0] / SCALE_EPS
            e_btm = x_scaled[1] / SCALE_EPS
            F_err = get_force_err(e_top, e_btm)
            return F_err * SCALE_FORCE
            
        # 3. Scaled Constraint: Midpoint Check (eps_mid <= eps_c2)
        def const_mid(x_scaled):
            e_top = x_scaled[0] / SCALE_EPS
            e_btm = x_scaled[1] / SCALE_EPS
            e_mid = e_btm + (e_top - e_btm) * ratio
            return (eps_c2 - e_mid) * SCALE_EPS
    
        constraints = [
            {'type': 'ineq', 'fun': const_force},
            {'type': 'ineq', 'fun': lambda x: -const_force(x)},
            {'type': 'ineq', 'fun': const_mid},
            # Bounds as constraints:
            # e_top >= 0
            {'type': 'ineq', 'fun': lambda x: x[0]},
            # e_top <= eps_cu
            {'type': 'ineq', 'fun': lambda x: eps_cu * SCALE_EPS - x[0]},
            # e_btm >= -eps_ud
            {'type': 'ineq', 'fun': lambda x: x[1] + eps_ud * SCALE_EPS},
            # e_btm <= eps_ud
            {'type': 'ineq', 'fun': lambda x: eps_ud * SCALE_EPS - x[1]},
        ]
        print("minimize")
        # Initial guess (Standard balanced section)
        x0 = [eps_cu * 0.8 * SCALE_EPS, -eps_ud * 0.5 * SCALE_EPS]
        
        # Run Optimization with normalized tolerances
        res = minimize(objective, x0, method="COBYLA", constraints=constraints, options={'rhobeg': 0.5, 'maxiter': 1000, 'tol': 1e-5})#tol=1e-5)
        
        if res.success:
            M_final = -res.fun / SCALE_MOMENT
            e_top_res = res.x[0] / SCALE_EPS
            e_btm_res = res.x[1] / SCALE_EPS
            print(f"Minimization terminated successfully\n\t\tM = {M_final/1e6:.1f} kNm\n\t\tε_top = {e_top_res*1000:.2f} MRad ({e_top_res/eps_cu*100:.0f}%)\n\t\tε_btm = {e_btm_res*1000:.2f} MRad ({-e_btm_res/eps_ud*100:.0f}%)")
            
            if return_F:
                # Recompute exact force for return
                F_final = F_target + get_force_err(e_top_res, e_btm_res)
                y_na = y_btm + (y_top-y_btm)*(0-e_btm_res)/(e_top_res-e_btm_res)
                return M_final, F_final, y_na
                
            return M_final
            
        raise ValueError(f"M_Rd calculation failed. Force {F_target} unreachable.")

    
    # =========================================================================
    # SECTION PROPERTIES
    # =========================================================================
    
    def I_u(
        self,
        t: Optional[float] = None,
        creep: float = 0,
        na=None
    ) -> np.ndarray:
        """
        Calculate uncracked second moment of area for transformed section.
        
        Parameters
        ----------
        t : float or None
            Concrete age [days]
        creep : float
            Creep coefficient
            
        Returns
        -------
        I : ndarray
            Second moments as 2x2 matrix:
            [[I_zz, I_zy],
             [I_zy, I_yy]]
            where I_zz is for bending about x-axis (strong axis for typical beam)
            and I_yy is for bending about y-axis
        """
        if na is None:
            cache = self._get_cache("I_u", t, creep)
            if cache is not None: return cache
        
        # Get effective modulus (per-bar, accounts for remove_conc_to_rebar)
        alpha_e = self.alpha_e(t, creep)
        
        # Get neutral axis of uncracked section
        if na is None:
            na = self.elastic_na(t, creep)
        
        I_c = self._Ixx_about(self.concrete_mesh.tri, na)
        I_s = alpha_e * (np.pi*self.rebar_diameters**4/64 + self.A_s_arr*(self.rebar_centers[:,1] - na)**2)
        
        result = I_c + np.sum(I_s)
        if na is not None:
            self._set_cache("I_u", t, creep, result)
        
        return result
        
        
    def I_c(
        self,
        t: Optional[float] = None,
        creep: float = 0,
        e=None
    ) -> np.ndarray:
        """
        Calculate cracked second moment of area.
        
        Uses transformed section method with iterative neutral axis location.
        Note: This is simplified - assumes crack direction perpendicular to bending axis.
        
        Parameters
        ----------
        t : float or None
            Concrete age [days]
        creep : float
            Creep coefficient
            
        Returns
        -------
        I : ndarray
            Second moments as 2x2 matrix (same format as I_u)
        """
        if e is None:
            cache = self._get_cache("I_c", t, creep)
            if cache is not None: return cache
        
        # Get effective modulus ratio (per-bar)
        alpha_e = self.alpha_e(t, creep)
        
        y_crack = self.elastic_cracking_depth(t, creep, e=e)
        
        split = self.concrete_mesh.split_by_y(y_crack)
        top_section = split.masked(split.centroids[:,1] > y_crack)
        
        # Compute cracked transformed NA (centroid of concrete above crack + steel)
        A_c_cr = top_section.total_area
        S_c_cr = A_c_cr * top_section.centroid[1] if A_c_cr > 0 else 0.0
        A_s_ef = alpha_e * self.A_s_arr
        na_cr = (S_c_cr + np.sum(A_s_ef * self.rebar_centers[:,1])) / (A_c_cr + np.sum(A_s_ef))
        
        I_c_val = self._Ixx_about(top_section, na_cr) if A_c_cr > 0 else 0.0
        I_s = alpha_e * (np.pi*self.rebar_diameters**4/64 + self.A_s_arr*(self.rebar_centers[:,1] - na_cr)**2)
        
        result = I_c_val + np.sum(I_s)
        if e is None:
            self._set_cache("I_c", t, creep, result)
        return result
    
    def f_ctm_fl(self, t: Optional[float] = None) -> float:
        """Mean flexural tensile strength at age t [MPa]"""
        return self.concrete.beta_cc(t) ** 0.6 * self.concrete.f_ctm_ref * max(1.6-self.H/1000, 1.0)
        
    def M_cr(
        self,
        t: Optional[float] = None,
        creep=0.0,
        use_f_ctm_fl: bool = False,
        F_c=0.0
    ) -> Tuple[float, float]:
        """
        Calculate cracking moment using section modulus.
        
        Returns cracking moments for bending about both principal axes.
        
        Parameters
        ----------
        t : float or None
            Concrete age [days]
            
        Returns
        -------
        M_cr_x : float
            Cracking moment for bending about x-axis [N·mm]
            (vertical load on typical beam - strong axis)
        M_cr_y : float
            Cracking moment for bending about y-axis [N·mm]
            (lateral load - weak axis)
        """
        
        f_ctm = self.f_ctm_fl(t) if use_f_ctm_fl else self.concrete.f_ctm(t)
        sigma_c = F_c/self.A_c if self.A_c > 0 else 0.0
        W_uc = self.W_uc(t, creep)
        return np.inf if W_uc <= 0 else W_uc*(f_ctm+sigma_c)
        
    def crack_factor(
        self,
        M: float,
        t: Optional[float] = None,
        creep: float = 0,
        use_LT_factor=False
    ) -> float:
        """
        Calculate crack state factor (0 = uncracked, 1 = fully cracked).
        
        Uses linear interpolation between cracking moment and a factor of 2.
        
        Parameters
        ----------
        M : float
            Applied moment [N·mm]
        t : float or None
            Concrete age [days]
        creep : float
            Creep coefficient
        axis : str
            'x' for bending about x-axis (strong), 'y' for y-axis (weak)
            
        Returns
        -------
        zeta : float
            Crack factor [0-1]
        """
        M_cr = self.M_cr(t, creep)
        M_abs = abs(M)
        
        if M_abs <= M_cr:
            return 0.0
        
        factor = 0.5 if use_LT_factor else 1.0
        
        c = 1-factor*(M_cr/M_abs)**2
        return min(1.0, c)
        
    def effective_I(
        self,
        M: float = 0.0,
        t: Optional[float] = None,
        creep: float = 0,
        use_LT_factor: bool = False
    ) -> float:
        """
        Calculate effective moment of inertia considering cracking.
        
        Uses interpolation between uncracked and cracked properties based
        on crack factor.
        
        Parameters
        ----------
        M : float
            Applied moment [N·mm]
        t : float or None
            Concrete age [days]
        creep : float
            Creep coefficient
        use_LT_factor : bool
            Use long-term beta factor (0.5) for crack interpolation
            
        Returns
        -------
        I_eff : float
            Effective moment of inertia [mm⁴]
        """
        
        I_u = self.I_u(t, creep)
        I_c = self.I_c(t, creep)
        zeta = self.crack_factor(M, t, creep, use_LT_factor)
        
        # Interpolate: I_eff = I_u*(1-ζ) + I_c*ζ
        return I_u*I_c/(I_u * zeta + I_c * (1-zeta))
    
    # =========================================================================
    # UTILITY METHODS
    # =========================================================================
    
    def plot(self, show: bool = False, dpi=150, incl_uls=False, incl_stiffness=False, incl_dims=False, creep=2.2, t=50*365, fill_color="#cce6ff"):
        """
        Plot the section with table below, absolutely fixed row heights.
        
        Parameters:
            incl_dims: bool - Include blue dimension annotations (default True)
        
        Returns:
            fig: matplotlib Figure object
        """
        na = self.elastic_na()
    
        # Build table data first
        try:
            rho_str = f"{self.A_s/self.A_c*100:.2f}%"
        except ZeroDivisionError:
            rho_str = "0%"
        
        try:
            rho_t = f"{sum(self.A_s_arr[self.rebar_centers[:,1]<na])/self.A_c*100:.2f}%"
        except ZeroDivisionError:
            rho_t = "0%"
    
        table_data = [
            ["Dimensions", f"{ef(self.B)} × {ef(self.H)} mm"],
            ["Concrete Area ($A_c$)", f"{ef(self.A_c)} mm²"],
            ["Steel Area ($A_s$)", f"{ef(self.A_s)} mm²"],
            ["Reinf. Ratio ($\\rho$)", rho_str],
            ["Steel below NA ($A_{st}$)", f"{ef(sum(self.A_s_arr[self.rebar_centers[:,1]<na]))} mm²"],
            ["Reinf. Ratio below NA ($\\rho_t$)", rho_t],
            ["Concrete cylinder strength ($f_{ck}$)", f"{ef(self.concrete.f_ck_28)} MPa"],
            ["Steel yield strength ($f_y$)", f"{ef(self.f_y)} MPa"],
            ["Uncracked NA depth ($x_{u}$)", f"{ef(self.y_max-na)} mm"],
            ["Cracked NA depth ($x_{c}$)", f"{ef(self.elastic_cracking_depth())} mm"],
        ]
        
        if incl_stiffness:
            Ic, Ic_lg = self.I_c(), self.I_c(t=t, creep=creep)
            Iu, Iu_lg = self.I_u(), self.I_u(t=t, creep=creep)
            Mcr, Mcr_lg = self.M_cr(), self.M_cr(t=t, creep=creep)
            table_data.extend([
                ["uncracked, short ($I_{u,st}$)", f"{ef(Iu)} mm⁴"],
                ["cracked, short ($I_{c,st}$)", f"{ef(Ic)} mm⁴"],
                ["cracking, short ($M_{cr,st}$)", f"{ef(Mcr/1e6)} kNm"],
                ["Long-term params", f"t={ef(t/365)}yrs, φ={ef(creep)}"],
                ["uncracked, long ($I_{u,lt}$)", f"{ef(Iu_lg)} mm⁴"],
                ["cracked, long ($I_{c,lt}$)", f"{ef(Ic_lg)} mm⁴"],
                ["cracking, long ($M_{cr,lt}$)", f"{ef(Mcr_lg/1e6)} kNm"],
            ])
        if incl_uls:
            M, _, na_uls = self.M_Rd(return_F=True, incl_x=True)
            table_data.append(["Bending capacity ($M_{Rd}$)", f"{ef(M/1e6)} kNm"])
            if na_uls is not None:
                x_uls = self.y_max - na_uls
                table_data.extend([
                    ["Design NA depth ($x_{d}$)", f"{ef(x_uls)} mm"],
                    ["Depth ratio ($x_{d}/d_{ef}$)", f"{ef(x_uls/self.d_ef)}"],
                ])
        
    
        n_rows = len(table_data) + 1  # +1 for header
        
        # Fixed dimensions in inches
        row_height_in = 0.28
        gap_rows = 1  # Gap between plot and table
        table_height_in = (n_rows + gap_rows) * row_height_in
        plot_width_in = 8
        
        # Plot height based on geometry aspect ratio
        geom_aspect = self.B / self.H
        plot_height_in = plot_width_in / max(geom_aspect, 0.5)
        plot_height_in = max(2.5, min(plot_height_in, 7))
        
        total_height_in = plot_height_in + table_height_in + 1.0
        
        # Always create our own figure for proper layout control
        fig, ax = plt.subplots(figsize=(plot_width_in, total_height_in), dpi=dpi)
    
        # Plot geometry
        self.concrete_mesh.tri.plot(ax=ax, color=fill_color, edgecolor='#99bbdd', lw=0.3, alpha=0.6)
        self.rebar_poly.plot(show=False, ax=ax, fill_color='red', boundary_color='darkred', dpi=dpi)
        
        # Rebar labels
        for center, diameter in zip(self.rebar_centers, self.rebar_diameters):
            radius = diameter / 2
            ax.text(center[0] + radius + 5, center[1] + radius + 5, 
                    f"ø{diameter:.0f}", fontsize=9, color='darkred', ha='left', va='bottom')
    
        # ============================================================
        # DIMENSION ANNOTATIONS WITH BANDING
        # ============================================================
        # Get geometry bounds (needed for axis limits regardless of dims)
        x_min, x_max = self.x_min, self.x_max
        y_min, y_max = self.y_min, self.y_max
        
        # Absolute offsets in mm (not scaled to section size)
        dim_gap = 30  # Gap from section edge to first dim line
        dim_spacing = 40  # Spacing between dim lines
        text_offset = 8  # Text offset from dim line
        
        if incl_dims:
            def band_values(values, tol=5.0):
                """Group values within tolerance, return list of (representative_value, [original_indices])"""
                if len(values) == 0:
                    return []
                sorted_idx = np.argsort(values)
                sorted_vals = values[sorted_idx]
                
                bands = []
                current_band_vals = [sorted_vals[0]]
                current_band_idx = [sorted_idx[0]]
                
                for i in range(1, len(sorted_vals)):
                    if sorted_vals[i] - sorted_vals[i-1] <= tol:
                        current_band_vals.append(sorted_vals[i])
                        current_band_idx.append(sorted_idx[i])
                    else:
                        bands.append((np.mean(current_band_vals), current_band_idx.copy()))
                        current_band_vals = [sorted_vals[i]]
                        current_band_idx = [sorted_idx[i]]
                bands.append((np.mean(current_band_vals), current_band_idx.copy()))
                return bands
            
            # Get bar centre coordinates
            bar_x = self.rebar_centers[:, 0]
            bar_y = self.rebar_centers[:, 1]
            
            # Band the x and y coordinates
            x_bands = band_values(bar_x, tol=5.0)
            y_bands = band_values(bar_y, tol=5.0)
            
            # Get unique banded x positions (sorted)
            banded_x = sorted([b[0] for b in x_bands])
            banded_y = sorted([b[0] for b in y_bands])
            
            # Build dimension chain for X: x_min -> bars -> x_max
            x_dim_points = [x_min] + banded_x + [x_max]
            y_dim_points = [y_min] + banded_y + [y_max]
            
            # Dimension styling
            dim_color = '#0066cc'
            dim_fontsize = 8
            arrow_props = dict(arrowstyle='<->', color=dim_color, lw=0.8)
            
            # Horizontal dimensions (below the section)
            dim_y_offset = y_min - dim_gap  # First row
            dim_y_offset2 = y_min - dim_gap - dim_spacing  # Second row for totals
            
            for i in range(len(x_dim_points) - 1):
                x1, x2 = x_dim_points[i], x_dim_points[i+1]
                dist = x2 - x1
                if dist < 1:  # Skip negligible dimensions
                    continue
                
                # Draw dimension line
                ax.annotate('', xy=(x2, dim_y_offset), xytext=(x1, dim_y_offset),
                           arrowprops=arrow_props)
                # Dimension text
                ax.text((x1 + x2) / 2, dim_y_offset - text_offset,
                       f"{dist:.0f}", fontsize=dim_fontsize, ha='center', va='top', color=dim_color)
                # Extension lines
                ax.plot([x1, x1], [y_min, dim_y_offset], color=dim_color, lw=0.5, ls=':')
                ax.plot([x2, x2], [y_min, dim_y_offset], color=dim_color, lw=0.5, ls=':')
            
            # Total width dimension
            ax.annotate('', xy=(x_max, dim_y_offset2), xytext=(x_min, dim_y_offset2),
                       arrowprops=arrow_props)
            ax.text((x_min + x_max) / 2, dim_y_offset2 - text_offset,
                   f"{x_max - x_min:.0f}", fontsize=dim_fontsize, ha='center', va='top', 
                   color=dim_color, weight='bold')
            
            # Vertical dimensions (to the right of section)
            dim_x_offset = x_max + dim_gap  # First column
            dim_x_offset2 = x_max + dim_gap + dim_spacing  # Second column for totals
            
            for i in range(len(y_dim_points) - 1):
                y1, y2 = y_dim_points[i], y_dim_points[i+1]
                dist = y2 - y1
                if dist < 1:  # Skip negligible dimensions
                    continue
                
                # Draw dimension line
                ax.annotate('', xy=(dim_x_offset, y2), xytext=(dim_x_offset, y1),
                           arrowprops=arrow_props)
                # Dimension text
                ax.text(dim_x_offset + text_offset, (y1 + y2) / 2,
                       f"{dist:.0f}", fontsize=dim_fontsize, ha='left', va='center', color=dim_color)
                # Extension lines
                ax.plot([x_max, dim_x_offset], [y1, y1], color=dim_color, lw=0.5, ls=':')
                ax.plot([x_max, dim_x_offset], [y2, y2], color=dim_color, lw=0.5, ls=':')
            
            # Total height dimension
            ax.annotate('', xy=(dim_x_offset2, y_max), xytext=(dim_x_offset2, y_min),
                       arrowprops=arrow_props)
            ax.text(dim_x_offset2 + text_offset, (y_min + y_max) / 2,
                   f"{y_max - y_min:.0f}", fontsize=dim_fontsize, ha='left', va='center',
                   color=dim_color, weight='bold')
        
        # ============================================================
        # END DIMENSION ANNOTATIONS
        # ============================================================
    
        ax.set_title("RC Section Analysis", pad=10)
        ax.set_aspect('equal')
        
        # Force axis limits AFTER set_aspect to include dimension annotations
        if incl_dims:
            x_pad_left = 20
            x_pad_right = dim_gap + dim_spacing + 50  # Space for vertical dims + text
            y_pad_bottom = dim_gap + dim_spacing + 30  # Space for horizontal dims + text
            y_pad_top = 20
        else:
            x_pad_left = 20
            x_pad_right = 20
            y_pad_bottom = 20
            y_pad_top = 20
        ax.set_xlim(x_min - x_pad_left, x_max + x_pad_right)
        ax.set_ylim(y_min - y_pad_bottom, y_max + y_pad_top)
        
        # Set up layout for table below plot
        bottom_margin = (table_height_in + 0.2) / total_height_in
        plt.subplots_adjust(left=0.08, right=0.95, top=0.93, bottom=bottom_margin)
        
        fig.canvas.draw()
        fig_height_in = fig.get_figheight()
        fig_width_in = fig.get_figwidth()
        
        # Fixed table dimensions in figure coordinates
        table_width_in = 6.0
        table_width_fig = min(table_width_in / fig_width_in, 0.9)
        table_left_fig = (1.0 - table_width_fig) / 2
        
        row_height_fig = row_height_in / fig_height_in
        table_total_height_fig = n_rows * row_height_fig
        
        # Position table at bottom with small margin
        table_top_fig = bottom_margin - 0.02
        table_bottom_fig = table_top_fig - table_total_height_fig
        table_bottom_fig = max(0.01, table_bottom_fig)
        
        # Create dedicated axes for table
        table_ax = fig.add_axes([table_left_fig, table_bottom_fig, table_width_fig, table_total_height_fig])
        table_ax.axis('off')
        
        table = table_ax.table(
            cellText=table_data,
            colLabels=["Property", "Value"],
            cellLoc='center',
            colWidths=[0.55, 0.45],
            bbox=[0.0, 0.0, 1.0, 1.0]
        )
        
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        
        for (row, col), cell in table.get_celld().items():
            cell.set_height(1.0 / n_rows)
            if row == 0:
                cell.get_text().set_weight('bold')
                cell.set_facecolor('#e6e6e6')
            cell.set_edgecolor('#cccccc')
    
        if show:
            plt.show()
        
        return fig
    
    def __repr__(self) -> str:
        return (f"RCSection(f_ck={self.concrete.f_ck_28:.0f} MPa, "
                f"n_bars={len(self.rebar_centers)}, "
                f"A_s={self.A_s:.0f} mm²)")
                
    
    def FM_graph(self, steps=100, t=None, creep=0, incl_bounds=False, incl_x=False):
        """Build F-M interaction diagram following EC2 strain limits.
        
        Three EC2 limits on the linear strain profile:
          1. eps_top <= eps_cu  (concrete crushing at top face)
          2. eps_mid <= eps_c2  (strain at section centroid)
          3. eps_s   >= -eps_ud (steel strain at lowest rebar)
        
        Branch 1 (high moment): eps_top = eps_cu, eps_s from -eps_ud
            up to where eps_mid = eps_c2. Covers the bending-dominated region.
        Branch 2 (high compression): eps_mid = eps_c2, phi from Branch 1
            endpoint down to 0 (pure compression). Covers the squash region.
        Branch 3 (low F): eps_s = -eps_ud, eps_top from eps_cu downward.
            Only used if Branch 1 didn't reach F=0. Covers the tension region.
        
        Parameters
        ----------
        incl_x : bool
            If True, also tracks the neutral-axis depth x = y_top - y_NA at
            each strain state and returns an x(F) spline alongside the M(F)
            spline. Pure-axial states (phi=0, NA at infinity) carry NaN, which
            is linearly filled before the x spline is built.
        
        Returns
        -------
        Depending on flags:
          fun                            (default)
          fun, F_min, F_max              (incl_bounds=True)
          fun, fun_x                     (incl_x=True)
          fun, fun_x, F_min, F_max       (incl_bounds=True, incl_x=True)
        """
        cache_key = ("FM_graph", steps, int(t) if t is not None else t, int(creep*1e3))
        cached = self._cached_properties.get(cache_key)
        if cached is not None:
            fun, fun_x, F_min, F_max = cached
            if incl_bounds and incl_x:
                return fun, fun_x, F_min, F_max
            if incl_bounds:
                return fun, F_min, F_max
            if incl_x:
                return fun, fun_x
            return fun
        
        eps_cu = self.concrete.eps_cu   # 0.0035
        eps_c2 = self.concrete.eps_c2   # 0.002
        eps_ud = self.eps_ud
        y_top = self.y_max
        y_btm = self.y_min
        y_s_min = np.min(self.rebar_centers[:,1])  # lowest rebar centre
        na_ref = self.na_ref_y
        
        # d_s = distance from top to lowest rebar (for strain profile)
        d_s = y_top - y_s_min
        # r = centroid position as ratio within top-to-rebar depth
        r_s = (na_ref - y_s_min) / d_s if d_s > 1e-6 else 0.5
        # H_full for full section depth (used in add_point)
        H = y_top - y_btm
        
        FM_1 = []
        FM_2 = []
        FM_3 = []
        
        def add_point(store, eps_top, eps_s):
            """Convert strains at top face and lowest rebar to (eps_na, phi).
            
            eps_top: strain at y_top
            eps_s: strain at y_s_min (lowest rebar)
            phi is defined over the full section depth for M_integral.
            Also stores x = y_top - y_NA (NaN for pure-axial phi=0 case).
            """
            if abs(d_s) < 1e-6:
                return
            phi = (eps_top - eps_s) / d_s
            eps_na = eps_s + phi * (na_ref - y_s_min)  # strain at centroid
            f, m = self.M_integral(eps_na, phi, t, creep, True)
            if abs(phi) > 1e-12:
                x = y_top - (na_ref - eps_na / phi)
            else:
                x = float('nan')  # pure axial: NA at infinity
            if np.isfinite(f) and np.isfinite(m):
                store.append([f, m, x])
        
        # --- Branch 1: eps_top = eps_cu, sweep eps_s ---
        # Start: eps_s = -eps_ud (max bending, steel at limit)
        # End:   eps_s where eps_mid = eps_c2
        #   eps_mid = eps_s + (eps_cu - eps_s) * r_s = eps_s*(1-r_s) + eps_cu*r_s
        #   eps_s_end = (eps_c2 - eps_cu*r_s) / (1-r_s)
        
        eps_s_start = -eps_ud
        eps_s_end = (eps_c2 - eps_cu * r_s) / (1 - r_s) if abs(1 - r_s) > 1e-9 else eps_c2
        
        for eps_s in np.linspace(eps_s_start, eps_s_end, steps):
            add_point(FM_1, eps_cu, eps_s)
        
        # --- Branch 2: eps_mid = eps_c2, sweep phi from B1 end to 0 ---
        # eps_top = eps_c2 + phi*(y_top - na_ref)
        # eps_s   = eps_c2 + phi*(y_s_min - na_ref)
        # At B1 end: phi_start = (eps_cu - eps_s_end) / d_s
        # At end:    phi = 0 (pure compression at eps_c2)
        
        phi_start_b2 = (eps_cu - eps_s_end) / d_s
        
        for phi in np.linspace(phi_start_b2, 0, steps // 2, endpoint=True):
            eps_top_b2 = eps_c2 + phi * (y_top - na_ref)
            eps_s_b2 = eps_c2 + phi * (y_s_min - na_ref)
            # Only hard limit is top face crushing
            if eps_top_b2 > eps_cu * 1.001:
                continue
            add_point(FM_2, eps_top_b2, eps_s_b2)
        
        # --- Branch 3: eps_s = -eps_ud, sweep eps_top from just below eps_cu ---
        # Only needed if Branch 1 didn't reach F <= 0.
        # Starts just below eps_cu (Branch 1's first point is at eps_cu)
        # to avoid duplicating it.
        
        FM_1_arr = ar(FM_1) if FM_1 else np.empty((0, 3))
        b1_reached_zero = len(FM_1_arr) > 0 and FM_1_arr[:,0].min() <= 0
        
        if not b1_reached_zero:
            # Sweep from just below eps_cu down to -eps_ud
            eps_top_values = np.linspace(eps_cu, 0, steps + 1)[1:]  # skip first (=eps_cu)
            for eps_top in eps_top_values:
                add_point(FM_3, eps_top, -eps_ud)
        
        # --- Per-branch F=0 interpolation on Branch 3 ---
        FM_3_arr = ar(FM_3) if FM_3 else np.empty((0, 3))
        if len(FM_3_arr) > 1:
            order_3 = np.argsort(FM_3_arr[:,0])
            FM_3_arr = FM_3_arr[order_3]
            if FM_3_arr[0, 0] < 0 and FM_3_arr[-1, 0] > 0:
                neg_mask = FM_3_arr[:,0] < 0
                i = np.where(neg_mask)[0][-1]
                F_lo, M_lo, x_lo = FM_3_arr[i]
                F_hi, M_hi, x_hi = FM_3_arr[i + 1]
                t_i = -F_lo / (F_hi - F_lo)
                row = [0.0, M_lo + t_i * (M_hi - M_lo), x_lo + t_i * (x_hi - x_lo)]
                FM_3_arr = np.insert(FM_3_arr, i + 1, row, axis=0)
            FM_3_arr = FM_3_arr[FM_3_arr[:,0] >= -1e-6]
            if len(FM_3_arr) > 0:
                FM_3_arr[FM_3_arr[:,0] < 0, 0] = 0.0
        
        # --- Per-branch F=0 interpolation on Branch 1 ---
        FM_1_arr = ar(FM_1) if FM_1 else np.empty((0, 3))
        if len(FM_1_arr) > 1:
            order_1 = np.argsort(FM_1_arr[:,0])
            FM_1_arr = FM_1_arr[order_1]
            if FM_1_arr[0, 0] < 0 and FM_1_arr[-1, 0] > 0:
                neg_mask = FM_1_arr[:,0] < 0
                i = np.where(neg_mask)[0][-1]
                F_lo, M_lo, x_lo = FM_1_arr[i]
                F_hi, M_hi, x_hi = FM_1_arr[i + 1]
                t_i = -F_lo / (F_hi - F_lo)
                row = [0.0, M_lo + t_i * (M_hi - M_lo), x_lo + t_i * (x_hi - x_lo)]
                FM_1_arr = np.insert(FM_1_arr, i + 1, row, axis=0)
            FM_1_arr = FM_1_arr[FM_1_arr[:,0] >= -1e-6]
            if len(FM_1_arr) > 0:
                FM_1_arr[FM_1_arr[:,0] < 0, 0] = 0.0
        
        # --- Merge ---
        FM_2_arr = ar(FM_2) if FM_2 else np.empty((0, 3))
        
        # B1 is authoritative in its F range
        # B2 only above B1's max F
        # B3 only below B1's min F (skipped entirely if B1 reached F=0)
        if len(FM_1_arr) > 0:
            F_min_b1 = FM_1_arr[:,0].min()
            F_max_b1 = FM_1_arr[:,0].max()
            if len(FM_2_arr) > 0:
                FM_2_arr = FM_2_arr[FM_2_arr[:,0] > F_max_b1]
            if len(FM_3_arr) > 0:
                FM_3_arr = FM_3_arr[FM_3_arr[:,0] < F_min_b1]
        
        FM = np.concatenate([p for p in [FM_1_arr, FM_2_arr, FM_3_arr] if len(p) > 0], axis=0)
        if len(FM) < 4:
            raise ValueError("FM_graph: fewer than 4 valid strain states found")
        
        # Sort by force (no further interpolation - each branch is clean)
        order = np.argsort(FM[:,0])
        FM = FM[order]
        
        if len(FM) < 4:
            raise ValueError("FM_graph: fewer than 4 points after removing F<0")
        
        # Envelope extraction: bin by F, keep max M in each bin.
        # x in column 2 follows automatically since the whole row is kept.
        n_bins = min(len(FM), 4 * steps)
        F_edges = np.linspace(FM[0,0] - 1, FM[-1,0] + 1, n_bins + 1)
        bin_idx = np.digitize(FM[:,0], F_edges) - 1
        FM_env = []
        for b in range(n_bins):
            mask = bin_idx == b
            if mask.any():
                group = FM[mask]
                best = group[np.argmax(group[:,1])]
                FM_env.append(best)
        FM = ar(FM_env)
        
        # Remove non-strictly-increasing F values
        mask = np.concatenate(([True], np.diff(FM[:,0]) > 1e-3))
        FM = FM[mask]
        
        if len(FM) < 4:
            raise ValueError("FM_graph: fewer than 4 points after envelope")
        
        # M(F) spline. x(F) spline built on the same F nodes; NaNs from the
        # pure-axial squash-load tail are linearly interpolated from
        # neighbouring finite values so the spline is well-defined everywhere.
        fun = make_interp_spline(FM[:,0], FM[:,1], k=min(3, len(FM)-1))
        x_vals = FM[:,2].copy()
        finite = np.isfinite(x_vals)
        if finite.any() and not finite.all():
            x_vals = np.interp(FM[:,0], FM[finite,0], x_vals[finite])
        fun_x = make_interp_spline(FM[:,0], x_vals, k=min(3, len(FM)-1))
        
        F_max = float(FM[-1, 0])
        F_min = float(FM[0, 0])
        
        self._cached_properties[cache_key] = (fun, fun_x, F_min, F_max)
        
        if incl_bounds and incl_x:
            return fun, fun_x, F_min, F_max
        if incl_bounds:
            return fun, F_min, F_max
        if incl_x:
            return fun, fun_x
        return fun
    
    def M_Rd(self, F_target=0.0, t=None, creep=0.0, return_F=False, steps=100, incl_x=False):
        """Ultimate moment capacity at a given axial load via FM spline.
        
        Builds an FM interaction spline (cached), then evaluates at F_target.
        Faster and more robust than M_Rd_direct across the full axial range.
        
        Parameters
        ----------
        F_target : float
            Applied axial force [N] (positive = compression)
        return_F : bool
            If True, return (M, F_target, None) by default for backwards
            compatibility, or (M, F_target, na_uls) if incl_x=True. na_uls
            is the y-coordinate of the neutral axis at F_target, recovered
            from the paired x(F) spline built alongside M(F) in FM_graph.
        steps : int
            Number of strain states for the FM sweep (higher = more accurate)
        incl_x : bool
            If True (and return_F=True), populate the third return element
            with the neutral-axis y-coordinate instead of None.
        """
        fun, fun_x, F_min, F_max = self.FM_graph(steps, t, creep,
                                                  incl_bounds=True, incl_x=True)
        F_clipped = np.clip(F_target, F_min, F_max)
        M = float(fun(F_clipped))
        if not return_F:
            return M
        if incl_x:
            x = float(fun_x(F_clipped))
            na_uls = self.y_max - x
            return M, F_target, na_uls
        return M, F_target, None
        
    def plot_FM_graph(self, steps=50, t=None, creep=0, ax=None, show=False):
        fun, F_min, F_max = self.FM_graph(steps, t, creep, incl_bounds=True)
        fs_raw = np.linspace(F_min, F_max, 3*steps)
        ms_raw = fun(fs_raw)

        fs = fs_raw / 1e3
        ms = ms_raw / 1e6

        # Clip to M >= 0 for plotting. Find the F at which M crosses zero on
        # the upper (high-F) end and on the lower (low-F) end if those
        # crossings exist, then linearly interpolate exact endpoints.
        i_pos = np.where(ms >= 0)[0]
        if len(i_pos) == 0:
            # No positive-M region. Nothing to plot meaningfully, but keep
            # the original behaviour rather than blowing up.
            fs_plot = fs
            ms_plot = ms
            f_top_callout = fs[-1]
            m_top_callout = ms[-1]
        else:
            i_lo = i_pos[0]
            i_hi = i_pos[-1]

            # Upper crossing (between i_hi and i_hi+1, if it exists)
            if i_hi < len(ms) - 1:
                f_a, m_a = fs[i_hi], ms[i_hi]
                f_b, m_b = fs[i_hi + 1], ms[i_hi + 1]
                if m_a != m_b:
                    t_i = m_a / (m_a - m_b)  # m goes from +ve to -ve
                    f_top = f_a + t_i * (f_b - f_a)
                else:
                    f_top = f_a
                m_top = 0.0
            else:
                f_top = fs[i_hi]
                m_top = ms[i_hi]

            # Lower crossing (between i_lo-1 and i_lo, if it exists)
            if i_lo > 0:
                f_a, m_a = fs[i_lo - 1], ms[i_lo - 1]
                f_b, m_b = fs[i_lo], ms[i_lo]
                if m_a != m_b:
                    t_i = m_a / (m_a - m_b)
                    f_bot = f_a + t_i * (f_b - f_a)
                else:
                    f_bot = f_b
                m_bot = 0.0
                fs_plot = np.concatenate([[f_bot], fs[i_lo:i_hi + 1], [f_top]])
                ms_plot = np.concatenate([[m_bot], ms[i_lo:i_hi + 1], [m_top]])
            else:
                fs_plot = np.concatenate([fs[i_lo:i_hi + 1], [f_top]])
                ms_plot = np.concatenate([ms[i_lo:i_hi + 1], [m_top]])

            f_top_callout = f_top
            m_top_callout = m_top

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 7))

        ax.plot(ms_plot, fs_plot, c='r', lw=1.5)
        ax.fill_betweenx(fs_plot, ms_plot, 0, color='r', alpha=0.1)

        def plot_point(m_val, f_val, color='k'):
            ax.axhline(f_val, linestyle="--", c=color, lw=0.7)
            ax.axvline(m_val, linestyle="--", c=color, lw=0.7)
            ax.scatter(m_val, f_val, c=color, s=30, zorder=5)
            label = f"M: {m_val:.1f}\nN: {f_val:.1f}"
            ax.annotate(label, xy=(m_val, f_val), xytext=(10, 0),
                        textcoords="offset points", fontsize=9,
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="k", alpha=0.9))

        # Bottom callout: M_Rd at F=0 (pure bending). M_Rd handles clipping
        # internally so this works whether F=0 is inside or just outside the
        # envelope. Label F as 0.0 rather than the clipped query value.
        M_at_zero = self.M_Rd(F_target=0.0, t=t, creep=creep) / 1e6
        plot_point(M_at_zero, 0.0)
        # Top callout: the clipped M=0 endpoint (or curve end if no crossing)
        #plot_point(m_top_callout, f_top_callout)
        # Top callout: F at which M = 0. Extrapolate the spline past F_max
        # if necessary. M is decreasing on the high-F side, so search outward
        # from F_max until M goes negative, then brentq for the exact root.
        #from scipy.optimize import brentq
        F_peak = fs_raw[int(np.argmax(ms_raw))]
        M_at_Fmax = float(fun(F_max))
        if M_at_Fmax <= 0:
            # Root is inside the envelope, bracket between peak and F_max
            F_lo, M_lo = F_peak, float(fun(F_peak))
            F_hi, M_hi = F_max, M_at_Fmax
        else:
            # Root is past F_max, walk outward to find a negative M
            F_lo, M_lo = F_max, M_at_Fmax
            F_hi = F_max
            M_hi = M_at_Fmax
            step = max(abs(F_max), 1.0) * 0.25  # 25% expansion per step
            for _ in range(20):
                F_hi += step
                M_hi = float(fun(F_hi))
                if M_hi < 0:
                    break
                step *= 1.5  # accelerate if we don't find it fast
        try:
            if M_lo > 0 > M_hi:
                F_top_callout = brentq(
                    lambda F: float(fun(F)), F_lo, F_hi,
                    xtol=1.0, maxiter=80,
                )
            else:
                # Couldn't bracket - extrapolation never went negative within
                # 20 steps. Highly unusual; fall back to last evaluated point.
                F_top_callout = F_hi
        except Exception:
            F_top_callout = F_hi
        
        plot_point(0.0, F_top_callout / 1e3)
        # Peak moment callout
        i_peak = int(np.argmax(ms_plot))
        plot_point(ms_plot[i_peak], fs_plot[i_peak])

        ax.set_xlabel('Moment (kNm)')
        ax.set_ylabel('Axial Force (kN)')
        ax.grid(True, alpha=0.3)

        if show:
            plt.show()
        return ax
    
    """
    def plot_FM_graph(self, steps=50, t=None, creep=0, ax=None, show=False):
        fun, F_min, F_max = self.FM_graph(steps, t, creep, incl_bounds=True)
        fs_raw = np.linspace(F_min, F_max, 3*steps)
        ms_raw = fun(fs_raw)

        fs = fs_raw / 1e3
        ms = ms_raw / 1e6

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 7))

        ax.plot(ms, fs, c='r', lw=1.5)
        ax.fill_betweenx(fs, ms, 0, color='r', alpha=0.1)

        def plot_point(idx, color='k'):
            m_val, f_val = ms[idx], fs[idx]
            ax.axhline(f_val, linestyle="--", c=color, lw=0.7)
            ax.axvline(m_val, linestyle="--", c=color, lw=0.7)
            ax.scatter(m_val, f_val, c=color, s=30, zorder=5)
            label = f"M: {m_val:.1f}\nN: {f_val:.1f}"
            ax.annotate(label, xy=(m_val, f_val), xytext=(10, 0),
                        textcoords="offset points", fontsize=9,
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="k", alpha=0.9))

        plot_point(0)
        plot_point(-1)
        plot_point(np.argmax(ms))

        ax.set_xlabel('Moment (kNm)')
        ax.set_ylabel('Axial Force (kN)')
        ax.grid(True, alpha=0.3)

        if show:
            plt.show()
        return ax
    """
    
    def plot_F_vs_x(self, F_target=0.0, t=None, creep=0.0):
        y_top = self.y_max
        d_ef = self.d_ef
        fcd = self.concrete.f_cd(t)
        y_s = self.rebar_centers[:,1]
        A_s = self.A_s_arr
        eps_cu = self.concrete.eps_cu
        na = self.elastic_na(t, creep)
        
        x_vals = np.linspace(0.01, d_ef, 100)
        F_vals = []
        M_vals = []
        
        for x in x_vals:
            y_cut = y_top - 0.8*x
            y_pl = y_top - x
            split = self.concrete_mesh.split_by_y(y_cut)
            top = split.masked(split.centroids[:,1] > y_cut)
            F_c = fcd * top.total_area if len(top) > 0 else 0.0
            
            e_s = eps_cu*(y_s - y_pl)/(y_top - y_pl)
            s_s = self.rebar_stresses(0.0, 0.0, t)  # placeholder; compute manually
            # Manual steel stress from strain
            eps_abs = np.abs(e_s)
            s_s = np.where(eps_abs < self.eps_yd,
                           e_s * self.E_s,
                           np.where(eps_abs <= self.eps_ud,
                                    (self.f_yd + (self.f_ud - self.f_yd)*(eps_abs - self.eps_yd)/(self.eps_ud - self.eps_yd)) * np.sign(e_s),
                                    0.0))
            
            if not self.remove_conc_to_rebar:
                inside_mask = self.rebar_mask
                c_stress = np.minimum(np.maximum(e_s[inside_mask]*fcd/self.concrete.eps_c2, 0), fcd)
                s_s[inside_mask] -= c_stress
            
            F_s = np.sum(A_s * s_s)
            F_total = F_c + F_s
            
            top_cy = top.centroid[1] if len(top) > 0 else y_top
            M = F_c * (top_cy - na) + np.sum(A_s * s_s * (y_s - na))
            
            F_vals.append(F_total)
            M_vals.append(M)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax1.plot(x_vals, F_vals)
        ax1.axhline(F_target, color='r', linestyle='--')
        ax1.set_xlabel('x')
        ax1.set_ylabel('F')
        
        ax2.plot(x_vals, M_vals)
        ax2.set_xlabel('x')
        ax2.set_ylabel('M')
        
        plt.tight_layout()
        plt.show()


# =============================================================================
# HELPERS
# =============================================================================

def simple_beam(B, H, f_ck, d_t, n_t, d_c=None, n_c=None, c=30, outer=False, s_outer=None, d_perp=None, **kw):
    """Quick rectangular section builder.

    Places n_t tension bars of diameter d_t near the bottom face and
    optionally n_c compression bars of diameter d_c near the top.
    """
    if outer:
        f_t = f_c = 0.5
    else:
        if d_perp is not None:
            f_t = (d_t/2+d_perp)/d_t
            f_c = (d_c/2+d_perp)/d_c
        else:
            f_t = f_c = 1.5

    if isinstance(c, (tuple, list, np.ndarray)):
        c_t, c_c = c
    else:
        c_t = c_c = c

    y_tens = c_t + f_t*d_t
    if s_outer is not None:
        s_o = s_outer + d_t/2
    else:
        s_o = y_tens

    x_tens = np.linspace(s_o, B-s_o, n_t)
    y_tens = np.ones_like(x_tens)*y_tens
    crs = np.column_stack([x_tens, y_tens])
    ds = np.ones_like(x_tens)*d_t

    if (d_c is not None) and (n_c is not None):
        z_comp = c_c + f_c*d_c
        y_comp = H-z_comp
        if s_outer is None:
            s_o = z_comp
        x_comp = np.linspace(s_o, B-s_o, n_c)
        y_comp = np.ones_like(x_comp)*y_comp
        crs = np.concatenate([crs, np.column_stack([x_comp, y_comp])])
        ds = np.concatenate([ds, np.ones_like(x_comp)*d_c])

    concrete_poly = ar([
        [0, 0], [B, 0],
        [B, H], [0, H]
    ], float)

    return RCSection(f_ck, concrete_poly, crs, ds, **kw)


# =============================================================================
# TESTS
# =============================================================================

import time

def _check(label, got, expect, tol=None, rtol=1e-3):
    """Print PASS/FAIL for a single value check."""
    if tol is not None:
        ok = abs(got - expect) < tol
    else:
        ok = np.isclose(got, expect, rtol=rtol, atol=1e-6)
    tag = 'PASS' if ok else 'FAIL'
    err = (got - expect) / expect * 100 if expect != 0 else got
    print(f'  {tag}  {label}: {got:.6g}  (expect {expect:.6g}, err {err:+.2f}%)')
    return ok


def _timed(fn, label, *args, **kwargs):
    """Run fn, print elapsed time, return result."""
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    dt = time.perf_counter() - t0
    print(f'  TIME  {label}: {dt*1000:.1f} ms')
    return result


def test_rectangle_no_rebar():
    """Plain rectangle: mesh area and I_xx vs analytical bh^3/12."""
    print('\n=== Plain rectangle (no rebar) ===')
    b, h = 300.0, 600.0
    sec = RCSection(30, ar([[0,0],[b,0],[b,h],[0,h]]),
                    ar([[b/2, h/2]]), ar([0.001]),  # dummy bar
                    remove_conc_to_rebar=False)
    passed = True
    passed &= _check('A_c', sec.A_c, b*h, rtol=1e-3)
    passed &= _check('centroid_y', sec.c_c[1], h/2, rtol=1e-3)
    I_analytical = b*h**3/12
    I_mesh = sec.concrete_mesh.tri.Ixx()
    passed &= _check('Ixx (mesh)', I_mesh, I_analytical, rtol=1e-3)
    return passed


def test_transformed_section():
    """300x600 beam, 3T25 bottom, 2T16 top. Transformed section vs hand calc.

    Uses remove_conc_to_rebar=False so the mesh is the full rectangle.
    The code's convention:
        A_c = mesh_area - sum(A_s * rebar_mask)   [net concrete]
        alpha_e = E_s/E_c for outside bars, E_s/E_c - 1 for inside bars
        A_ef = A_c + sum(alpha_e * A_s)
        c_ef = (A_c * c_c + sum(alpha_e * A_s * y_s)) / A_ef

    We replicate this analytically on the rectangle to get the expected
    NA and I_u, then check the mesh-based result matches.
    """
    print('\n=== Transformed section (300x600, C30, 3T25+2T16) ===')
    b, h = 300.0, 600.0
    cover = 35.0
    d_t, d_c = 25.0, 16.0
    y_t = cover + d_t/2
    y_c = h - cover - d_c/2
    centers = ar([
        [b/4, y_t], [b/2, y_t], [3*b/4, y_t],
        [b/3, y_c], [2*b/3, y_c],
    ])
    dias = ar([d_t]*3 + [d_c]*2)
    A_bars = pi*(dias/2)**2

    sec = _timed(
        lambda: RCSection(30, ar([[0,0],[b,0],[b,h],[0,h]]),
                          centers, dias, remove_conc_to_rebar=False),
        'RCSection init')

    # Replicate the code's own convention exactly
    E_c = sec.concrete.E_c()               # tangent modulus at ref age
    alpha_raw = sec.E_s / E_c
    alpha_e = np.ones(len(dias)) * alpha_raw
    alpha_e -= 1.0                          # all bars inside rectangle

    A_c_hand = b*h - np.sum(A_bars)        # net concrete (code subtracts bar areas)
    c_c_y = h/2                             # rectangle centroid
    A_ef_hand = A_c_hand + np.sum(alpha_e * A_bars)
    y_ef_hand = (A_c_hand * c_c_y + np.sum(alpha_e * A_bars * centers[:,1])) / A_ef_hand

    # I_u about the transformed NA
    I_c_hand = b*h**3/12 + b*h*(c_c_y - y_ef_hand)**2  # concrete about NA
    # subtract self-I of bar holes (code doesn't, but A_c is net, so parallel axis from gross is approximate)
    I_s_hand = np.sum(alpha_e * (pi*dias**4/64 + A_bars*(centers[:,1] - y_ef_hand)**2))
    I_u_hand = I_c_hand + I_s_hand

    na = _timed(sec.elastic_na, 'elastic_na')
    I_u = _timed(sec.I_u, 'I_u')

    passed = True
    passed &= _check('elastic_na', na, y_ef_hand, rtol=5e-3)
    passed &= _check('I_u', I_u, I_u_hand, rtol=1e-2)  # 1% tolerance (mesh vs analytical)
    # Also cross-check: code's own A_c matches our hand calc
    passed &= _check('A_c', sec.A_c, A_c_hand, rtol=1e-3)
    return passed


def test_cracking_and_stiffness():
    """Check cracking depth, I_c, M_cr, effective_I for consistency.

    Conventions (tension at bottom, compression at top):
        elastic_cracking_depth returns y_cut: the y-coordinate of the
        zero-strain line in the cracked section. For bottom-tension bending
        this is ABOVE the uncracked NA (concrete below y_cut is cracked).
        I_u is about the uncracked transformed NA.
        I_c is about the cracked transformed NA.
    """
    print('\n=== Cracking, I_c, M_cr, effective_I ===')
    sec = simple_beam(300, 600, 30, 25, 3, 16, 2, c=35,
                      remove_conc_to_rebar=False)
    passed = True

    na_u = sec.elastic_na()
    na_c = _timed(sec.elastic_cracking_depth, 'elastic_cracking_depth')
    I_u = _timed(sec.I_u, 'I_u')
    I_c = _timed(sec.I_c, 'I_c')
    M_cr = _timed(sec.M_cr, 'M_cr')

    # Cracked NA should be above uncracked NA for bottom-tension bending
    if na_c <= na_u:
        print(f'  FAIL  na_c={na_c:.1f} should be > na_u={na_u:.1f} (tension at bottom)')
        passed = False
    else:
        print(f'  PASS  na_c={na_c:.1f} > na_u={na_u:.1f} (tension at bottom)')

    if not (sec.y_min < na_c < sec.y_max):
        print(f'  FAIL  na_c={na_c:.1f} outside section [{sec.y_min:.0f}, {sec.y_max:.0f}]')
        passed = False
    else:
        print(f'  PASS  na_c={na_c:.1f} within section')

    # I_c < I_u (both now about their own respective NAs)
    if I_c >= I_u:
        print(f'  FAIL  I_c={I_c:.3e} should be < I_u={I_u:.3e}')
        passed = False
    else:
        print(f'  PASS  I_c={I_c:.3e} < I_u={I_u:.3e}')

    # M_cr > 0
    passed &= _check('M_cr > 0', float(M_cr > 0), 1.0, tol=0.1)

    # effective_I: at M=0 should equal I_u (zeta=0)
    I_eff_zero = sec.effective_I(0.0)
    passed &= _check('I_eff(M=0) == I_u', I_eff_zero, I_u, rtol=1e-6)

    # effective_I at large M should converge toward I_c
    I_eff_big = sec.effective_I(M_cr * 10)
    if not (I_c * 0.95 < I_eff_big < I_c * 1.05):
        print(f'  FAIL  I_eff(10*M_cr)={I_eff_big:.3e} not near I_c={I_c:.3e}')
        passed = False
    else:
        print(f'  PASS  I_eff(10*M_cr)={I_eff_big:.3e} near I_c={I_c:.3e}')

    # Long-term creep info
    I_u_lt = sec.I_u(creep=2.2)
    I_c_lt = sec.I_c(creep=2.2)
    na_lt = sec.elastic_na(creep=2.2)
    print(f'  INFO  I_u(st)={I_u:.3e}, I_u(lt)={I_u_lt:.3e}, '
          f'na(st)={na_u:.1f}, na(lt)={na_lt:.1f}')
    print(f'  INFO  I_c(st)={I_c:.3e}, I_c(lt)={I_c_lt:.3e}')

    return passed


def test_force_moment_equilibrium():
    """Check F_integral returns ~0 at balanced strain, and M_integral > 0."""
    print('\n=== F/M integral equilibrium ===')
    sec = simple_beam(300, 600, 30, 25, 3, 16, 2, c=35)
    passed = True

    # At zero strain everywhere, F and M should be zero
    F0 = sec.F_integral(0.0, 0.0)
    passed &= _check('F(eps=0, phi=0)', F0, 0.0, tol=1.0)

    # At a small positive curvature with mid-strain at eps_c2/2,
    # check that F_integral and M_integral return finite values
    phi_test = 1e-5
    eps_test = 0.001
    F_test = sec.F_integral(eps_test, phi_test)
    M_test = sec.M_integral(eps_test, phi_test)
    if not np.isfinite(F_test):
        print(f'  FAIL  F_integral returned {F_test}')
        passed = False
    else:
        print(f'  PASS  F_integral = {F_test:.0f} N (finite)')

    if not np.isfinite(M_test):
        print(f'  FAIL  M_integral returned {M_test}')
        passed = False
    else:
        print(f'  PASS  M_integral = {M_test/1e6:.1f} kNm (finite)')

    return passed


def test_M_Rd():
    """Check M_Rd solves, returns positive moment, and satisfies equilibrium."""
    print('\n=== M_Rd (ULS capacity) ===')
    sec = simple_beam(300, 600, 30, 25, 3, 16, 2, c=35)
    passed = True

    M_Rd, F_out, na_d = _timed(
        lambda: sec.M_Rd(F_target=0.0, return_F=True), 'M_Rd(F=0)')

    passed &= _check('M_Rd > 0', float(M_Rd > 0), 1.0, tol=0.1)
    passed &= _check('F passthrough', F_out, 0.0, tol=1.0)

    # Sanity: M_Rd should be larger than M_cr
    M_cr = sec.M_cr()
    if M_Rd <= M_cr:
        print(f'  FAIL  M_Rd={M_Rd/1e6:.0f} kNm <= M_cr={M_cr/1e6:.0f} kNm')
        passed = False
    else:
        print(f'  PASS  M_Rd={M_Rd/1e6:.0f} kNm > M_cr={M_cr/1e6:.0f} kNm')

    # na is None from spline-based M_Rd (no strain state returned)
    print(f'  INFO  na_d={na_d} (spline M_Rd does not compute NA)')

    # With axial compression, M_Rd should change
    M_Rd_N, _, _ = _timed(
        lambda: sec.M_Rd(F_target=500e3, return_F=True), 'M_Rd(F=500kN)')
    print(f'  INFO  M_Rd(F=0)={M_Rd/1e6:.0f} kNm, M_Rd(F=500kN)={M_Rd_N/1e6:.0f} kNm')

    return passed


def test_copy_and_rotate():
    """Check copy preserves properties, rotate preserves area and I magnitude."""
    print('\n=== copy / rotate ===')
    sec = simple_beam(300, 600, 30, 25, 3, 16, 2, c=35)
    passed = True

    sec2 = _timed(sec.copy, 'copy')
    passed &= _check('copy A_c', sec2.A_c, sec.A_c, rtol=1e-6)
    passed &= _check('copy A_s', sec2.A_s, sec.A_s, rtol=1e-6)
    passed &= _check('copy I_u', sec2.I_u(), sec.I_u(), rtol=1e-3)

    sec90 = _timed(lambda: sec.rotate(90), 'rotate(90)')
    passed &= _check('rotate A_c', sec90.A_c, sec.A_c, rtol=1e-2)
    passed &= _check('rotate A_s', sec90.A_s, sec.A_s, rtol=1e-6)
    # B and H should swap (approximately)
    passed &= _check('rotate B~H', sec90.B, sec.H, rtol=5e-2)
    passed &= _check('rotate H~B', sec90.H, sec.B, rtol=5e-2)

    return passed


def test_t_section():
    """T-section: check area matches analytical."""
    print('\n=== T-section geometry ===')
    # Flange 600x150 on top, web 250x450 below. Total h=600.
    t_poly = SimplePolygon([
        [0, 0], [250, 0], [250, 450], [600, 450],
        [600, 600], [0, 600], [0, 450], [0, 0],
    ])
    # Correct outline for T
    t_poly = SimplePolygon([
        [175, 0], [425, 0], [425, 450], [600, 450],
        [600, 600], [0, 600], [0, 450], [175, 450],
    ])
    A_expected = 250*450 + 600*150  # web + flange = 202500
    centers = ar([[300, 50], [200, 50], [400, 50]])  # 3 bars near bottom
    dias = ar([25, 25, 25])

    sec = RCSection(30, t_poly, centers, dias, remove_conc_to_rebar=False)
    passed = True
    passed &= _check('T-section A_c', sec.A_c, A_expected, rtol=1e-2)
    # NA should be above geometric centre (flange pulls it up)
    geom_cy = (250*450*225 + 600*150*525) / A_expected
    na = sec.elastic_na()
    print(f'  INFO  geometric cy={geom_cy:.1f}, elastic_na={na:.1f}')
    return passed


def test_M_Rd_vs_FM_graph():
    """Compare M_Rd (spline) against M_Rd_direct (optimiser) at several axial loads.

    M_Rd_direct uses a brentq boundary search that works well up to moderate
    axial loads, falling back to COBYLA near the squash load. M_Rd sweeps
    the full strain space via FM_graph so covers the entire range.

    Checks value agreement within the reliable M_Rd_direct range and reports
    high-axial-load behaviour as INFO.
    """
    print('\n=== M_Rd (spline) vs M_Rd_direct (optimiser) ===')
    sec = simple_beam(300, 600, 30, 25, 3, 16, 2, c=35)
    passed = True

    # Build the spline via M_Rd, timing it
    sec._clear_cache()
    t0 = time.perf_counter()
    M_first = sec.M_Rd(F_target=0.0)  # triggers FM_graph build
    dt_build = time.perf_counter() - t0
    _, F_min, F_max = sec.FM_graph(incl_bounds=True)  # cached, near-free
    print(f'  M_Rd spline build: {dt_build*1000:.1f} ms')
    print(f'  F range: {F_min/1e3:.0f} to {F_max/1e3:.0f} kN')

    # Checked range: 0 to 50% of F_max (where M_Rd_direct is reliable)
    F_checked = np.linspace(F_min + 1e3, F_max * 0.50, 5)
    # Info range: 60-95% (M_Rd_direct may fall back to COBYLA)
    F_info = np.array([F_max * r for r in [0.6, 0.75, 0.9]])

    t_direct_total = 0.0
    t_spline_total = 0.0
    n_checked = 0

    print(f'\n  {"F (kN)":>10s}  {"Direct (kNm)":>12s}  {"Spline (kNm)":>12s}  '
          f'{"err %":>8s}  {"t_dir ms":>10s}  {"t_spl ms":>10s}')
    print(f'  {"-"*10}  {"-"*12}  {"-"*12}  {"-"*8}  {"-"*10}  {"-"*10}')

    for F in F_checked:
        # Direct optimiser
        sec._clear_cache()
        t0 = time.perf_counter()
        M_dir, _, _ = sec.M_Rd_direct(F_target=F, return_F=True)
        dt_dir = time.perf_counter() - t0
        t_direct_total += dt_dir

        # Spline (already built)
        t0 = time.perf_counter()
        M_spl = sec.M_Rd(F_target=F)
        dt_spl = time.perf_counter() - t0
        t_spline_total += dt_spl
        n_checked += 1

        err = (M_spl - M_dir) / M_dir * 100 if abs(M_dir) > 1 else 0.0
        ok = abs(err) < 5.0
        if not ok:
            passed = False
        tag = '    ' if ok else 'FAIL'
        print(f'  {F/1e3:10.0f}  {M_dir/1e6:12.1f}  {M_spl/1e6:12.1f}  '
              f'{err:+7.2f}%  {dt_dir*1000:10.2f}  {dt_spl*1000:10.4f}  {tag}')

    print(f'\n  High-axial info (M_Rd_direct boundary search may not bracket):')
    for F in F_info:
        sec._clear_cache()
        t0 = time.perf_counter()
        try:
            M_dir, _, _ = sec.M_Rd_direct(F_target=F, return_F=True)
        except (ValueError, ArithmeticError):
            M_dir = float('nan')
        dt_dir = time.perf_counter() - t0

        M_spl = sec.M_Rd(F_target=F)
        err_str = f'{(M_spl - M_dir)/M_dir*100:+.1f}%' if np.isfinite(M_dir) and abs(M_dir) > 1 else 'n/a'
        print(f'  {F/1e3:10.0f}  {M_dir/1e6:12.1f}  {M_spl/1e6:12.1f}  '
              f'{err_str:>8s}  {dt_dir*1000:10.2f}       INFO')

    print(f'\n  Direct total:     {t_direct_total*1000:.1f} ms  '
          f'({t_direct_total/n_checked*1000:.1f} ms/call avg)')
    print(f'  Spline build:     {dt_build*1000:.1f} ms  (one-off)')
    print(f'  Spline lookups:   {t_spline_total*1000:.3f} ms  '
          f'({t_spline_total/n_checked*1000:.4f} ms/call avg)')
    speedup = t_direct_total / (dt_build + t_spline_total) if (dt_build + t_spline_total) > 0 else 0
    print(f'  Speedup:          {speedup:.1f}x (spline amortised over {n_checked} evals)')

    return passed



def test_timing_suite():
    """Benchmark key methods on a moderately complex section."""
    print('\n=== Timing suite (1000x250 slab, 10T20+5T12) ===')
    sec = simple_beam(1000, 250, 40, 20, 10, 12, 5, c=35, outer=True)
    print(f'  Section: {sec.B:.0f}x{sec.H:.0f}, '
          f'{len(sec.rebar_centers)} bars, '
          f'{sec.concrete_mesh.n_elements} elements')

    N = 20
    methods = [
        ('elastic_na',              lambda: sec.elastic_na()),
        ('elastic_cracking_depth',  lambda: sec.elastic_cracking_depth()),
        ('I_u',                     lambda: sec.I_u()),
        ('I_c',                     lambda: sec.I_c()),
        ('M_cr',                    lambda: sec.M_cr()),
        ('effective_I(50kNm)',      lambda: sec.effective_I(50e6)),
        ('F_integral',              lambda: sec.F_integral(0.001, 1e-5)),
        ('M_integral',              lambda: sec.M_integral(0.001, 1e-5)),
        ('M_Rd(F=0)',               lambda: sec.M_Rd(F_target=0, return_F=True)),
        ('M_Rd_direct(F=0)',        lambda: sec.M_Rd_direct(F_target=0, return_F=True)),
    ]

    for name, fn in methods:
        # Clear caches before timing
        sec._clear_cache()
        times = []
        for _ in range(N):
            sec._clear_cache()
            t0 = time.perf_counter()
            fn()
            times.append(time.perf_counter() - t0)
        med = sorted(times)[N//2]
        mn = min(times)
        print(f'  {name:30s}  median {med*1000:7.2f} ms  min {mn*1000:7.2f} ms')


def main():
    print('=' * 70)
    print('RC_beam_mesh test suite')
    print('=' * 70)

    results = {}
    for test_fn in [
        test_rectangle_no_rebar,
        test_transformed_section,
        test_cracking_and_stiffness,
        test_force_moment_equilibrium,
        test_M_Rd,
        test_copy_and_rotate,
        test_t_section,
        test_M_Rd_vs_FM_graph,
    ]:
        try:
            results[test_fn.__name__] = test_fn()
        except Exception as e:
            print(f'  ERROR  {test_fn.__name__}: {e}')
            results[test_fn.__name__] = False

    test_timing_suite()

    print('\n' + '=' * 70)
    n_pass = sum(results.values())
    n_total = len(results)
    print(f'Results: {n_pass}/{n_total} test groups passed')
    for name, ok in results.items():
        print(f'  {"PASS" if ok else "FAIL"}  {name}')
    print('=' * 70)

    # Plot tests - outside try/except so errors are visible
    print('\n=== Plot tests (outside try block) ===')
    sec = simple_beam(300, 600, 30, 25, 3, 16, 2, c=35)
    print('  Generating FM graph...')
    sec.plot_FM_graph(show=True)
    print('  Generating section plot...')
    sec.plot(show=True, incl_stiffness=True, incl_uls=True)


if __name__ == "__main__":
    main()