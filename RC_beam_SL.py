"""
RC Section Analysis Streamlit App
=================================
Wraps RC_beam_mesh.RCSection in an interactive UI.

Features
--------
1. Material property inputs in the sidebar (concrete, steel, factors).
2. Editable table of section vertices (xy) with dynamic add/delete rows.
3. Editable table of rebar (x, y, dia) with dynamic add/delete rows.
4. Live preview of the section for double-checking geometry before solving.
5. "Run Analysis" produces:
   - sec.plot(incl_uls=True, incl_stiffness=True, incl_dims=False)
   - sec.plot_FM_graph()
6. "Export to DOCX" writes a CalcDoc report containing both plots and
   a tabulated section property summary.

Usage:
    streamlit run RC_section_app.py
"""

from __future__ import annotations

import io
import traceback
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MPLPolygon, Circle

import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Module imports (RC_beam_mesh + CalcDoc). Imports fail loud and early.
# ---------------------------------------------------------------------------
try:
    from RC_beam_mesh import RCSection
except ImportError as e:
    st.set_page_config(page_title="RC Section", layout="wide")
    st.error(f"Could not import RC_beam_mesh.RCSection: {e}")
    st.info(
        "Place RC_beam_mesh.py (and its BugPoly/BugBasics/Mesh/Notation "
        "dependencies) on the Python path next to this script."
    )
    st.stop()

try:
    from CalcDoc import CalcDoc, Cm, Pt
    _CALC_DOC_OK = True
    _CALC_DOC_ERR = None
except ImportError as e:
    _CALC_DOC_OK = False
    _CALC_DOC_ERR = str(e)

from Notation import eng_format as ef


# ---------------------------------------------------------------------------
# Page config and defaults
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="RC Section Analysis",
    page_icon="🧱",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _default_vertices() -> np.ndarray:
    """300 x 600 rectangle, origin at bottom-left."""
    B, H = 300.0, 600.0
    return np.array([[0, 0], [B, 0], [B, H], [0, H]], float)


def _default_rebar() -> np.ndarray:
    """3T25 tension bars 40 mm from bottom on a 300 mm wide section."""
    B = 300.0
    c = 40.0
    xs = np.linspace(c, B - c, 3)
    return np.column_stack([xs, np.full(3, c), np.full(3, 25.0)])


def _init_state():
    if "vertices" not in st.session_state:
        st.session_state.vertices = _default_vertices()
    if "rebar" not in st.session_state:
        st.session_state.rebar = _default_rebar()
    if "section" not in st.session_state:
        st.session_state.section = None
    if "plot_fig" not in st.session_state:
        st.session_state.plot_fig = None
    if "fm_fig" not in st.session_state:
        st.session_state.fm_fig = None
    if "summary_rows" not in st.session_state:
        st.session_state.summary_rows = None


_init_state()


# ---------------------------------------------------------------------------
# Preview plot (cheap, no FE solve, used live as inputs change)
# ---------------------------------------------------------------------------
def plot_section_preview(vertices: np.ndarray, rebar: np.ndarray) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 6), dpi=90)

    if len(vertices) >= 3:
        # Concrete polygon
        poly = MPLPolygon(
            vertices,
            closed=True,
            facecolor="#cce6ff",
            edgecolor="#003366",
            linewidth=1.8,
            alpha=0.7,
        )
        ax.add_patch(poly)

        # Vertex numbering
        ax.plot(vertices[:, 0], vertices[:, 1], "o", color="#003366", markersize=5)
        for i, (x, y) in enumerate(vertices):
            ax.text(
                x, y, f"  {i + 1}",
                fontsize=8, color="#003366",
                ha="left", va="bottom", weight="bold",
            )

    # Rebar
    for i, (x, y, d) in enumerate(rebar):
        ax.add_patch(
            Circle((x, y), d / 2, facecolor="red", edgecolor="darkred",
                   linewidth=1.2, alpha=0.85)
        )
        ax.text(
            x + d / 2 + 4, y + d / 2 + 4,
            f"\u00f8{d:.0f}",
            fontsize=8, color="darkred", ha="left", va="bottom",
        )

    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_title("Live Section Preview")

    if len(vertices) >= 3:
        x_min, y_min = vertices.min(axis=0)
        x_max, y_max = vertices.max(axis=0)
        pad = max(x_max - x_min, y_max - y_min) * 0.12 + 5
        ax.set_xlim(x_min - pad, x_max + pad)
        ax.set_ylim(y_min - pad, y_max + pad)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Section builder (single source of truth for RCSection construction)
# ---------------------------------------------------------------------------
def build_section(vertices: np.ndarray, rebar: np.ndarray, mat: dict) -> RCSection:
    centers = rebar[:, :2].astype(float)
    dias = rebar[:, 2].astype(float)
    return RCSection(
        f_ck_28=float(mat["f_ck"]),
        concrete_poly=vertices.astype(float),
        rebar_centers=centers,
        rebar_diameters=dias,
        c_class=mat["c_class"],
        t_ref=float(mat["t_ref"]),
        k_E=float(mat["k_E"]),
        f_y=float(mat["f_y"]),
        gamma_c=float(mat["gamma_c"]),
        gamma_s=float(mat["gamma_s"]),
        as_square=bool(mat["as_square"]),
        remove_conc_to_rebar=bool(mat["remove_conc_to_rebar"]),
    )


# ---------------------------------------------------------------------------
# 90-degree clockwise rotation of vertices and rebar
# ---------------------------------------------------------------------------
def rotate_90_cw(vertices: np.ndarray, rebar: np.ndarray):
    """Rotate vertices and rebar 90 degrees clockwise about the section
    centroid (area centroid of the vertex polygon), then translate so the
    bounding box minimum sits at the origin.

    Mirrors the convention of RCSection.rotate (rotation about polygon
    centroid) but operates directly on the input arrays so it works before
    the section has been built.
    """
    if len(vertices) < 3:
        return vertices.copy(), rebar.copy()

    # Polygon centroid by the shoelace formula
    x = vertices[:, 0]
    y = vertices[:, 1]
    x_next = np.roll(x, -1)
    y_next = np.roll(y, -1)
    cross = x * y_next - x_next * y
    A2 = np.sum(cross)
    if abs(A2) < 1e-9:
        # Degenerate polygon, fall back to bbox centre
        cx = 0.5 * (x.min() + x.max())
        cy = 0.5 * (y.min() + y.max())
    else:
        cx = np.sum((x + x_next) * cross) / (3.0 * A2)
        cy = np.sum((y + y_next) * cross) / (3.0 * A2)

    # 90 deg CW: (x, y) -> (cx + (y - cy), cy - (x - cx))
    def rot(px, py):
        return cx + (py - cy), cy - (px - cx)

    new_vx, new_vy = rot(vertices[:, 0], vertices[:, 1])
    new_vertices = np.column_stack([new_vx, new_vy])

    if len(rebar) > 0:
        new_rx, new_ry = rot(rebar[:, 0], rebar[:, 1])
        new_rebar = np.column_stack([new_rx, new_ry, rebar[:, 2]])
    else:
        new_rebar = rebar.copy()

    # Translate so the rotated section sits at >= 0, matching the bottom-left
    # origin convention used by default inputs.
    x_min = new_vertices[:, 0].min()
    y_min = new_vertices[:, 1].min()
    new_vertices[:, 0] -= x_min
    new_vertices[:, 1] -= y_min
    if len(new_rebar) > 0:
        new_rebar[:, 0] -= x_min
        new_rebar[:, 1] -= y_min

    return new_vertices, new_rebar


# ---------------------------------------------------------------------------
# Section property summary (used by the docx export and the in-page table)
# ---------------------------------------------------------------------------
def section_summary(sec: RCSection, mat: dict, t_long: float, creep: float):
    """Return a list of (label, value) tuples covering all key properties.

    Robust to None / NaN / inf coming out of the solver, since `ef` (via
    math.log10) blows up on non-finite values.
    """

    def _safe_ef(value, sig_figs=3):
        try:
            if value is None:
                return "n/a"
            v = float(value)
            if not np.isfinite(v):
                return "n/a"
            if v == 0.0:
                return "0"
            return ef(v, sig_figs)
        except Exception:
            return "n/a"

    def _safe_div(a, b):
        try:
            if a is None or b is None:
                return float("nan")
            a = float(a); b = float(b)
            if b == 0 or not np.isfinite(a) or not np.isfinite(b):
                return float("nan")
            return a / b
        except Exception:
            return float("nan")

    na = sec.elastic_na()
    x_c = sec.elastic_cracking_depth()
    I_u_st = sec.I_u()
    I_c_st = sec.I_c()
    M_cr_st = sec.M_cr()
    I_u_lt = sec.I_u(t=t_long, creep=creep)
    I_c_lt = sec.I_c(t=t_long, creep=creep)
    M_cr_lt = sec.M_cr(t=t_long, creep=creep)

    # M_Rd (spline) with incl_x=True returns the y-coordinate of the ULS
    # neutral axis as the third element, recovered from the paired x(F)
    # spline built alongside M(F) in FM_graph. x_d = y_top - na_uls.
    try:
        M_Rd, _F, na_uls = sec.M_Rd(F_target=0.0, return_F=True, incl_x=True)
    except Exception:
        M_Rd, na_uls = float("nan"), None

    if na_uls is None:
        x_d = float("nan")
    else:
        try:
            na_f = float(na_uls)
            x_d = sec.y_max - na_f if np.isfinite(na_f) else float("nan")
        except Exception:
            x_d = float("nan")

    A_s = sec.A_s
    A_c = sec.A_c
    rho = (A_s / A_c * 100) if A_c > 0 else 0.0

    mask_below = sec.rebar_centers[:, 1] < na
    A_st = float(np.sum(sec.A_s_arr[mask_below])) if mask_below.any() else 0.0
    rho_t = (A_st / A_c * 100) if A_c > 0 else 0.0

    M_Rd_kNm = _safe_div(M_Rd, 1e6)
    M_cr_st_kNm = _safe_div(M_cr_st, 1e6)
    M_cr_lt_kNm = _safe_div(M_cr_lt, 1e6)
    x_d_ratio = _safe_div(x_d, sec.d_ef)

    rows = [
        ("**Material**", ""),
        ("Concrete strength f_{ck,28}", f"{_safe_ef(mat['f_ck'])} MPa"),
        ("Cement class", mat["c_class"]),
        ("Reference age t_{ref}", f"{_safe_ef(mat['t_ref'])} days"),
        ("Modulus coefficient k_{E}", f"{_safe_ef(mat['k_E'])}"),
        ("Steel yield f_{y}", f"{_safe_ef(mat['f_y'])} MPa"),
        ("Partial factor γ_{c}", f"{mat['gamma_c']:.2f}"),
        ("Partial factor γ_{s}", f"{mat['gamma_s']:.2f}"),
        ("**Geometry**", ""),
        ("Bounding box B × H", f"{_safe_ef(sec.B)} × {_safe_ef(sec.H)} mm"),
        ("Concrete area A_{c}", f"{_safe_ef(A_c)} mm^{{2}}"),
        ("Steel area A_{s}", f"{_safe_ef(A_s)} mm^{{2}}"),
        ("Reinforcement ratio ρ", f"{rho:.2f}%"),
        ("Steel area below NA A_{st}", f"{_safe_ef(A_st)} mm^{{2}}"),
        ("Tension ratio ρ_{t}", f"{rho_t:.2f}%"),
        ("Effective depth d_{ef}", f"{_safe_ef(sec.d_ef)} mm"),
        ("**Elastic (uncracked) properties**", ""),
        ("Elastic NA y", f"{_safe_ef(na)} mm"),
        ("Uncracked NA depth x_{u}", f"{_safe_ef(sec.y_max - na)} mm"),
        ("Cracked NA depth x_{c}", f"{_safe_ef(x_c)} mm"),
        ("Short term I_{u}", f"{_safe_ef(I_u_st)} mm^{{4}}"),
        ("Short term I_{c}", f"{_safe_ef(I_c_st)} mm^{{4}}"),
        ("Short term M_{cr}", f"{_safe_ef(M_cr_st_kNm)} kNm"),
        ("Long term (t, φ)", f"t = {_safe_ef(t_long / 365)} yrs, φ = {_safe_ef(creep)}"),
        ("Long term I_{u}", f"{_safe_ef(I_u_lt)} mm^{{4}}"),
        ("Long term I_{c}", f"{_safe_ef(I_c_lt)} mm^{{4}}"),
        ("Long term M_{cr}", f"{_safe_ef(M_cr_lt_kNm)} kNm"),
        ("**Ultimate limit state**", ""),
        ("Bending capacity M_{Rd} (F = 0)", f"{_safe_ef(M_Rd_kNm)} kNm"),
        ("Design NA depth x_{d}", f"{_safe_ef(x_d)} mm"),
        ("Depth ratio x_{d} / d_{ef}", f"{_safe_ef(x_d_ratio)}"),
    ]
    return rows


# ---------------------------------------------------------------------------
# Validation helpers: closed-form benchmarks for a rectangular RC section
# ---------------------------------------------------------------------------
def _is_axis_aligned_rectangle(vertices: np.ndarray, tol: float = 1e-3) -> bool:
    """True if the polygon is a 4-vertex axis-aligned rectangle.

    Tolerance is in millimetres, which is the input unit used throughout.
    """
    if vertices.shape[0] != 4:
        return False
    xs = np.sort(np.unique(np.round(vertices[:, 0] / tol) * tol))
    ys = np.sort(np.unique(np.round(vertices[:, 1] / tol) * tol))
    if xs.size != 2 or ys.size != 2:
        return False
    # All four corners must be one of the (x_i, y_j) combinations.
    corners = {(round(x, 6), round(y, 6)) for x in xs for y in ys}
    poly_pts = {(round(v[0], 6), round(v[1], 6)) for v in vertices}
    return corners == poly_pts


def _ec2_fctm(f_ck: float) -> float:
    """Mean axial tensile strength per EC2 Table 3.1 (MPa).

    f_ctm = 0.30 * f_ck^(2/3)   for f_ck <= 50 MPa
    f_ctm = 2.12 * ln(1 + (f_ck + 8) / 10)   for f_ck > 50 MPa
    """
    if f_ck <= 50.0:
        return 0.30 * f_ck ** (2.0 / 3.0)
    return 2.12 * np.log(1.0 + (f_ck + 8.0) / 10.0)


def _ec2_Ecm_kE(f_ck: float, k_E: float) -> float:
    """Mean secant modulus using the k_E formulation (MPa).

    E_cm = k_E * (f_ck + 8)^(1/3)

    This matches the form used by RCSection. EC2 Table 3.1 gives an
    equivalent value via E_cm = 22000 * ((f_ck+8)/10)^0.3 (MPa); both
    yield very similar results for normal-weight concrete.
    """
    return k_E * (f_ck + 8.0) ** (1.0 / 3.0)


def benchmark_rectangle(sec: RCSection, mat: dict, t_long: float, creep: float):
    """Closed-form benchmark values for a singly reinforced rectangular section.

    Returns a dict with the benchmark values. Assumes:
    - Concrete polygon is the bounding box B x H (a note flags this if not).
    - All rebar is treated as a single tension layer at the rebar area
      centroid (effective depth d). For cracked / ULS benchmarks this matches
      standard textbook practice (Mosley, Bungey & Hulse; Bhatt, MacGinley & Choo).
    - alpha_cc is 1.0 (UK NA value for EC2 used by D4S as default).
    """
    B = float(sec.B)
    H = float(sec.H)

    A_s_arr = np.asarray(sec.A_s_arr, dtype=float)
    y_s = np.asarray(sec.rebar_centers[:, 1], dtype=float)
    A_s_total = float(A_s_arr.sum())

    # Effective depth as area-weighted centroid of all rebar (matches d_ef).
    if A_s_total > 0:
        y_bar_s = float((A_s_arr * y_s).sum() / A_s_total)
    else:
        y_bar_s = 0.0
    d = H - y_bar_s  # distance from top fibre to rebar centroid

    f_ck = float(mat["f_ck"])
    f_y = float(mat["f_y"])
    gamma_c = float(mat["gamma_c"])
    gamma_s = float(mat["gamma_s"])
    k_E = float(mat["k_E"])
    E_s = 200_000.0  # MPa, EC2 §3.2.7(4)

    # --- Elastic moduli (short and long term) -------------------------------
    E_cm = _ec2_Ecm_kE(f_ck, k_E)
    E_c_eff = E_cm / (1.0 + creep)  # EC2 §7.4.3, §5.8.4
    n_st = E_s / E_cm
    n_lt = E_s / E_c_eff

    # --- Uncracked transformed section (short term) -------------------------
    # Take top fibre as y = H, bottom fibre as y = 0 (matches input convention).
    # Transformed area = B*H + (n - 1) * A_s   (concrete still present at steel locations)
    def _uncracked(n):
        A_t = B * H + (n - 1.0) * A_s_total
        # First moment about y = 0
        S_t = B * H * (H / 2.0) + (n - 1.0) * float((A_s_arr * y_s).sum())
        y_NA = S_t / A_t
        # Second moment about y_NA
        I_t = (
            B * H ** 3 / 12.0
            + B * H * (y_NA - H / 2.0) ** 2
            + (n - 1.0) * float((A_s_arr * (y_s - y_NA) ** 2).sum())
        )
        return y_NA, I_t

    y_NA_st, I_u_st = _uncracked(n_st)
    y_NA_lt, I_u_lt = _uncracked(n_lt)

    # --- Cracked transformed section: singly reinforced rectangle -----------
    # Take moments of transformed area about the NA. With NA depth x from top:
    #   B * x * (x / 2) = n * A_s * (d - x)
    # which gives x^2 + (2 n A_s / B) x - (2 n A_s d / B) = 0.
    def _cracked(n):
        if A_s_total <= 0 or d <= 0:
            return float("nan"), float("nan")
        a = 2.0 * n * A_s_total / B
        # x = (-a + sqrt(a^2 + 4 a d)) / 2
        x = 0.5 * (-a + np.sqrt(a * a + 4.0 * a * d))
        I_c = B * x ** 3 / 3.0 + n * A_s_total * (d - x) ** 2
        return x, I_c

    x_cr_st, I_c_st = _cracked(n_st)
    x_cr_lt, I_c_lt = _cracked(n_lt)

    # --- Cracking moment ----------------------------------------------------
    # M_cr = f_ctm * I_u / y_t, with y_t = distance from NA to extreme tension
    # fibre (bottom fibre for positive sagging).
    f_ctm = _ec2_fctm(f_ck)
    M_cr_st = f_ctm * I_u_st / y_NA_st if y_NA_st > 0 else float("nan")
    M_cr_lt = f_ctm * I_u_lt / y_NA_lt if y_NA_lt > 0 else float("nan")

    # --- ULS bending capacity (rectangular stress block, EC2 §3.1.7) --------
    # lambda = 0.8, eta = 1.0 for f_ck <= 50 MPa; otherwise:
    # lambda = 0.8 - (f_ck - 50)/400, eta = 1.0 - (f_ck - 50)/200
    if f_ck <= 50.0:
        lam = 0.8
        eta = 1.0
    else:
        lam = 0.8 - (f_ck - 50.0) / 400.0
        eta = 1.0 - (f_ck - 50.0) / 200.0
    alpha_cc = 1.0  # UK NA
    f_cd = alpha_cc * f_ck / gamma_c
    f_yd = f_y / gamma_s

    if A_s_total > 0 and d > 0:
        # Force equilibrium: lam * x * B * eta * f_cd = A_s * f_yd
        x_uls = A_s_total * f_yd / (lam * B * eta * f_cd)
        # Lever arm z = d - lam*x/2
        z = d - lam * x_uls / 2.0
        M_Rd = A_s_total * f_yd * z
        xd_ratio = x_uls / d
    else:
        x_uls = float("nan")
        M_Rd = float("nan")
        xd_ratio = float("nan")

    return dict(
        B=B, H=H, d=d, A_s=A_s_total,
        f_ck=f_ck, f_ctm=f_ctm, f_cd=f_cd, f_yd=f_yd,
        E_cm=E_cm, E_c_eff=E_c_eff, n_st=n_st, n_lt=n_lt,
        lam=lam, eta=eta, alpha_cc=alpha_cc,
        y_NA_st=y_NA_st, I_u_st=I_u_st,
        y_NA_lt=y_NA_lt, I_u_lt=I_u_lt,
        x_cr_st=x_cr_st, I_c_st=I_c_st,
        x_cr_lt=x_cr_lt, I_c_lt=I_c_lt,
        M_cr_st=M_cr_st, M_cr_lt=M_cr_lt,
        x_uls=x_uls, M_Rd=M_Rd, xd_ratio=xd_ratio,
    )


def validation_rows(sec: RCSection, mat: dict, t_long: float, creep: float):
    """Build the validation comparison table rows.

    Returns (header_note, rows) where rows is a list of tuples
        (quantity, units, computed, benchmark, abs_diff, pct_diff, notes)
    All numeric columns are pre-formatted strings.
    """
    # Try a few likely attribute names that RCSection might use to expose
    # the original polygon; if none are present, fall back to the bounding box
    # (in which case the section is by definition rectangular).
    poly = None
    for _attr in ("concrete_poly", "_concrete_poly", "poly", "vertices"):
        if hasattr(sec, _attr):
            try:
                _candidate = np.asarray(getattr(sec, _attr), dtype=float)
                if _candidate.ndim == 2 and _candidate.shape[1] == 2:
                    poly = _candidate
                    break
            except Exception:
                continue
    if poly is None:
        poly = np.array([[0, 0], [sec.B, 0], [sec.B, sec.H], [0, sec.H]], float)
    is_rect = _is_axis_aligned_rectangle(poly)

    if is_rect:
        header_note = (
            "The section is a single axis-aligned rectangle, so all closed-form "
            "benchmarks below apply directly. Differences arise from the mesh "
            "based numerical integration replacing the closed-form integrals, "
            "from the parabola-rectangle stress block (EC2 §3.1.7(1)) used by "
            "the solver versus the equivalent rectangular block (EC2 §3.1.7(3)) "
            "used here as a benchmark, and from rebar holes being subtracted "
            "from the concrete area if that option is enabled."
        )
    else:
        header_note = (
            "The section is not a simple rectangle. The benchmarks below use "
            "the bounding box B x H as the comparison rectangle with the same "
            "total rebar area concentrated at the rebar centroid. They serve "
            "as approximate sanity checks rather than strict closed-form "
            "validations, and material differences are expected in proportion "
            "to how far the real geometry departs from the bounding rectangle."
        )

    bm = benchmark_rectangle(sec, mat, t_long, creep)

    def _fmt(v, sig=4):
        try:
            v = float(v)
            if not np.isfinite(v):
                return "n/a"
            if v == 0:
                return "0"
            return ef(v, sig)
        except Exception:
            return "n/a"

    def _pct(a, b):
        try:
            a = float(a); b = float(b)
            if not np.isfinite(a) or not np.isfinite(b) or b == 0:
                return "n/a"
            return f"{100.0 * (a - b) / b:+.2f}%"
        except Exception:
            return "n/a"

    # Pull computed values from the solver.
    na_st = sec.elastic_na()
    try:
        na_lt = sec.elastic_na(t=t_long, creep=creep)
    except TypeError:
        na_lt = na_st
    x_c_st = sec.elastic_cracking_depth()
    try:
        x_c_lt = sec.elastic_cracking_depth(t=t_long, creep=creep)
    except TypeError:
        x_c_lt = x_c_st
    I_u_st_c = sec.I_u()
    I_c_st_c = sec.I_c()
    M_cr_st_c = sec.M_cr() / 1e6 if np.isfinite(sec.M_cr()) else float("nan")
    I_u_lt_c = sec.I_u(t=t_long, creep=creep)
    I_c_lt_c = sec.I_c(t=t_long, creep=creep)
    try:
        M_cr_lt_c = sec.M_cr(t=t_long, creep=creep) / 1e6
    except Exception:
        M_cr_lt_c = float("nan")

    try:
        M_Rd_c, _F_c, na_uls_c = sec.M_Rd(F_target=0.0, return_F=True, incl_x=True)
        M_Rd_c = M_Rd_c / 1e6
        x_uls_c = float(sec.y_max - float(na_uls_c)) if na_uls_c is not None else float("nan")
    except Exception:
        M_Rd_c = float("nan")
        x_uls_c = float("nan")

    # Solver NA is measured from the bottom fibre (y origin); benchmark y_NA is
    # also from the bottom fibre, so they are directly comparable.
    # Solver cracking depth x_c is from the top fibre, so is the benchmark x_cr.
    rows = [
        ("Quantity", "Units", "Computed", "Benchmark", "Difference", "Notes"),
        ("Elastic NA (uncracked, short term)", "mm",
         _fmt(na_st), _fmt(bm["y_NA_st"]),
         _pct(na_st, bm["y_NA_st"]),
         "Transformed area, n = E_s / E_cm"),
        ("Uncracked I_u (short term)", "mm^4",
         _fmt(I_u_st_c), _fmt(bm["I_u_st"]),
         _pct(I_u_st_c, bm["I_u_st"]),
         "Transformed second moment about elastic NA"),
        # Solver returns the cracked-NA y-coordinate measured from the
        # bottom fibre; benchmark "x_cr_st" is the compression-zone depth
        # measured from the top. Compare in the solver's convention so the
        # numbers line up: y_NA_bottom = H - x_cr.
        ("Cracked NA y (short term, from bottom)", "mm",
         _fmt(x_c_st), _fmt(sec.H - bm["x_cr_st"]),
         _pct(x_c_st, sec.H - bm["x_cr_st"]),
         "Quadratic in x, tension steel only (y = H - x)"),
        ("Cracked I_c (short term)", "mm^4",
         _fmt(I_c_st_c), _fmt(bm["I_c_st"]),
         _pct(I_c_st_c, bm["I_c_st"]),
         "B x^3 / 3 + n A_s (d - x)^2"),
        ("Cracking moment M_cr (short term)", "kNm",
         _fmt(M_cr_st_c), _fmt(bm["M_cr_st"] / 1e6),
         _pct(M_cr_st_c, bm["M_cr_st"] / 1e6),
         "f_ctm I_u / y_t, EC2 Table 3.1"),
        ("Elastic NA (uncracked, long term)", "mm",
         _fmt(na_lt), _fmt(bm["y_NA_lt"]),
         _pct(na_lt, bm["y_NA_lt"]),
         "Effective modulus E_c,eff = E_cm / (1 + phi)"),
        ("Uncracked I_u (long term)", "mm^4",
         _fmt(I_u_lt_c), _fmt(bm["I_u_lt"]),
         _pct(I_u_lt_c, bm["I_u_lt"]),
         "Creep applied via effective modulus, EC2 §7.4.3"),
        ("Cracked I_c (long term)", "mm^4",
         _fmt(I_c_lt_c), _fmt(bm["I_c_lt"]),
         _pct(I_c_lt_c, bm["I_c_lt"]),
         "Using long-term modular ratio"),
        ("Cracking moment M_cr (long term)", "kNm",
         _fmt(M_cr_lt_c), _fmt(bm["M_cr_lt"] / 1e6),
         _pct(M_cr_lt_c, bm["M_cr_lt"] / 1e6),
         "Same f_ctm, long term I_u and y_t"),
        ("ULS bending capacity M_Rd (N=0)", "kNm",
         _fmt(M_Rd_c), _fmt(bm["M_Rd"] / 1e6),
         _pct(M_Rd_c, bm["M_Rd"] / 1e6),
         "Solver uses parabola-rectangle, benchmark uses rectangular block"),
    ]
    return header_note, rows, bm, is_rect


# ---------------------------------------------------------------------------
# DOCX rich-text helpers
# ---------------------------------------------------------------------------
def _rich_cell(cell, text, bold=False, font_size=None):
    """Render text with CalcDoc's _{sub}, ^{sup}, **bold** markup into a
    docx table cell. Replaces any existing cell content with one paragraph
    of rich-formatted runs.
    """
    if not _CALC_DOC_OK:
        cell.text = str(text)
        return

    paragraphs = list(cell.paragraphs)
    for extra in paragraphs[1:]:
        p_el = extra._element
        p_el.getparent().remove(p_el)
    para = cell.paragraphs[0]
    for run in list(para.runs):
        r_el = run._element
        r_el.getparent().remove(r_el)

    fs = font_size if font_size is not None else Pt(10)
    tokens = CalcDoc.parse_rich_tokens(str(text))
    if bold:
        for tok in tokens:
            tok["bold"] = True
    CalcDoc.apply_tokens_to_paragraph(para, tokens, font_size=fs)


def _eq_safe(cd, latex_str, fallback_text=""):
    """Try to render a displayed equation. If mathtext rejects the LaTeX,
    fall back to a rich-text paragraph using the provided fallback (which
    uses CalcDoc markup syntax).
    """
    try:
        cd.add_equation(latex_str)
    except Exception:
        if fallback_text:
            cd.add_rich_paragraph(fallback_text)
        else:
            cd.add_rich_paragraph(latex_str)


# ---------------------------------------------------------------------------
# DOCX export via CalcDoc
# ---------------------------------------------------------------------------
def build_calcdoc_bytes(
    sec: RCSection,
    mat: dict,
    t_long: float,
    creep: float,
    plot_fig: plt.Figure,
    fm_fig: plt.Figure,
    project_name: str,
    project_number: str,
    design_element: str,
    calc_title: str,
    calc_by: str,
    checked_by: str,
    template_path: str | None,
) -> bytes:
    if not _CALC_DOC_OK:
        raise RuntimeError(
            f"CalcDoc is unavailable: {_CALC_DOC_ERR}. "
            "Install python-docx and ensure CalcDoc.py + CalcTemplate.docx are on the path."
        )

    cd = CalcDoc(
        project_name=project_name,
        project_number=project_number,
        design_element=design_element,
        calc_title=calc_title,
        calc_by=calc_by,
        checked_by=checked_by,
        date=datetime.now().strftime("%d/%m/%Y"),
        template_path=template_path,
    )

    # =======================================================================
    # 1. Scope and overview
    # =======================================================================
    cd.add_heading("Scope and Overview", level=1)
    cd.add_rich_paragraph(
        "This calculation reports the cross-section properties and ULS bending "
        "capacity of a reinforced concrete section defined by an arbitrary "
        "concrete polygon and a list of circular reinforcing bars. The "
        "analysis covers, in order:"
    )
    cd.add_rich_paragraph(
        "1. Elastic (uncracked) transformed section properties, short term and "
        "long term, using the effective modulus method to account for creep."
    )
    cd.add_rich_paragraph(
        "2. Cracked transformed section properties under pure bending, with "
        "the neutral axis depth and second moment of area computed from "
        "force and moment equilibrium of a transformed section."
    )
    cd.add_rich_paragraph(
        "3. Cracking moment M_{cr} based on the EC2 mean axial tensile "
        "strength f_{ctm}."
    )
    cd.add_rich_paragraph(
        "4. ULS bending capacity through a fibre-style integration of the "
        "parabola-rectangle concrete stress block and elastic-perfectly-plastic "
        "steel stress-strain law, sweeping the neutral axis position and "
        "producing the full force-moment (F-M) interaction diagram."
    )
    cd.add_rich_paragraph(
        "5. A validation section comparing the solver outputs against "
        "closed-form benchmarks computed from standard rectangular RC theory."
    )
    cd.add_rich_paragraph(
        "The geometry is treated as a closed polygon with rebar areas modelled "
        "either as circles or as equivalent squares (user option). Concrete "
        "area at the rebar locations may optionally be subtracted, which "
        "matters at high reinforcement ratios."
    )

    # =======================================================================
    # 2. Inputs and section property summary
    # =======================================================================
    cd.add_heading("Inputs and Section Properties", level=1)
    cd.add_rich_paragraph(
        "The table below lists the material inputs and the resulting section "
        "properties. Section headers (Material, Geometry, Elastic, ULS) are "
        "shown in bold."
    )

    rows = section_summary(sec, mat, t_long, creep)
    table = cd.add_table(rows=len(rows), cols=2)
    for i, (k, v) in enumerate(rows):
        is_header = k.startswith("**") and v == ""
        _rich_cell(table.rows[i].cells[0], k, bold=is_header)
        _rich_cell(table.rows[i].cells[1], v, bold=is_header)
    
    # =======================================================================
    # 6. Section and FM plots
    # =======================================================================
    cd.add_heading("Plots", level=1)

    cd.add_heading("Section Plot", level=2)
    cd.add_rich_paragraph(
        "Concrete polygon, rebar layout, elastic neutral axis, ULS strain "
        "diagram and stiffness annotations as enabled in the analysis "
        "options. Coordinates are in mm."
    )
    buf_plot = io.BytesIO()
    plot_fig.savefig(buf_plot, format="png", dpi=180, bbox_inches="tight",
                     facecolor="white")
    buf_plot.seek(0)
    cd.add_picture(buf_plot, width=Cm(16))

    cd.add_heading("Force-Moment Interaction (F-M Diagram)", level=2)
    cd.add_rich_paragraph(
        "Full ULS F-M envelope of the section. M_{Rd} at any axial force is "
        "obtained by intersecting a horizontal line at that F with the "
        "envelope. The pure bending capacity reported in the summary "
        "corresponds to F = 0."
    )
    buf_fm = io.BytesIO()
    fm_fig.savefig(buf_fm, format="png", dpi=180, bbox_inches="tight",
                   facecolor="white")
    buf_fm.seek(0)
    cd.add_picture(buf_fm, width=Cm(16))
    
    # =======================================================================
    # 3. Methodology
    # =======================================================================
    cd.add_heading("Methodology", level=1)

    cd.add_heading("Material Models", level=2)
    cd.add_rich_paragraph(
        "Concrete and steel are modelled to BS EN 1992-1-1 (Eurocode 2). "
        "Compression is positive throughout. The mean cylinder strength "
        "f_{cm} = f_{ck} + 8 MPa is taken per EC2 \u00a73.1.2(3) and "
        "Table 3.1. The mean secant modulus of concrete uses the k_{E} "
        "power-law form:"
    )
    _eq_safe(
        cd,
        r"E_{\mathrm{cm}} = k_E \, (f_{\mathrm{ck}} + 8)^{1/3} \quad \mathrm{[MPa]}",
        "E_{cm} = k_E (f_{ck} + 8)^{1/3} [MPa]",
    )
    cd.add_rich_paragraph(
        "with k_{E} entered as an input. This produces values consistent with "
        "the EC2 Table 3.1 expression E_{cm} = 22000 ((f_{ck} + 8) / 10)^{0.3} "
        "(MPa) for normal-weight concrete."
    )
    cd.add_rich_paragraph(
        "The mean axial tensile strength is taken from EC2 Table 3.1:"
    )
    _eq_safe(
        cd,
        r"f_{\mathrm{ctm}} = 0.30 \, f_{\mathrm{ck}}^{\,2/3} \quad \mathrm{for}\ f_{\mathrm{ck}} \le 50\ \mathrm{MPa}",
        "f_{ctm} = 0.30 f_{ck}^{2/3} for f_{ck} <= 50 MPa",
    )
    _eq_safe(
        cd,
        r"f_{\mathrm{ctm}} = 2.12 \, \ln\!\left(1 + \frac{f_{\mathrm{ck}} + 8}{10}\right) \quad \mathrm{for}\ f_{\mathrm{ck}} > 50\ \mathrm{MPa}",
        "f_{ctm} = 2.12 ln(1 + (f_{ck} + 8) / 10) for f_{ck} > 50 MPa",
    )
    cd.add_rich_paragraph(
        "f_{ctm} is used for the cracking moment only and is never relied on "
        "for shear or anchorage."
    )
    cd.add_rich_paragraph(
        "Steel is linear elastic up to f_{yd} = f_{yk} / \u03b3_{s}, then "
        "perfectly plastic, with E_{s} = 200 GPa per EC2 \u00a73.2.7(4). The "
        "horizontal yield plateau (Figure 3.8, idealised) is used because it "
        "is conservative for ductility class B and C bars and is the standard "
        "simplification for section analysis."
    )
    cd.add_rich_paragraph("The design strengths are:")
    _eq_safe(
        cd,
        r"f_{\mathrm{cd}} = \alpha_{\mathrm{cc}} \, \frac{f_{\mathrm{ck}}}{\gamma_c}, \quad f_{\mathrm{yd}} = \frac{f_{\mathrm{yk}}}{\gamma_s}",
        "f_{cd} = \u03b1_{cc} f_{ck} / \u03b3_{c},   f_{yd} = f_{yk} / \u03b3_{s}",
    )
    cd.add_rich_paragraph(
        "with \u03b1_{cc} = 1.0 (UK National Annex value) and \u03b3_{c}, "
        "\u03b3_{s} entered as inputs (defaults 1.5 and 1.15 per EC2 "
        "Table 2.1N)."
    )

    cd.add_heading("Creep and Long-Term Stiffness", level=2)
    cd.add_rich_paragraph(
        "Creep is applied through an effective modulus, per EC2 "
        "\u00a77.4.3(5) and \u00a75.8.4:"
    )
    _eq_safe(
        cd,
        r"E_{\mathrm{c,eff}} = \frac{E_{\mathrm{cm}}}{1 + \varphi(t,\,t_0)}",
        "E_{c,eff} = E_{cm} / (1 + \u03c6(t, t_{0}))",
    )
    cd.add_rich_paragraph(
        "The creep coefficient \u03c6 is entered directly and is intended to "
        "represent the value at the loading age and duration of interest, "
        "e.g. \u03c6(\u221e, t_{0}) for permanent loads, derived from EC2 "
        "Annex B or Figure 3.1 if needed. The same E_{c,eff} is applied "
        "uniformly to all concrete fibres for the long-term properties; no "
        "separate shrinkage contribution is included."
    )

    cd.add_heading("Elastic (Uncracked) Section Properties", level=2)
    cd.add_rich_paragraph(
        "The uncracked transformed section is built by replacing each steel "
        "bar of area A_{si} with an equivalent extra concrete area "
        "(n \u2212 1) A_{si} at the bar centroid, where n is the modular "
        "ratio:"
    )
    _eq_safe(
        cd,
        r"n = \frac{E_s}{E_c}, \quad E_c = E_{\mathrm{cm}}\;(\mathrm{short\;term}),\; E_{\mathrm{c,eff}}\;(\mathrm{long\;term})",
        "n = E_{s} / E_{c}  (E_{c} = E_{cm} short term, E_{c,eff} long term)",
    )
    cd.add_rich_paragraph(
        "The elastic neutral axis y_{NA} is the centroid of the transformed "
        "area:"
    )
    _eq_safe(
        cd,
        r"y_{\mathrm{NA}} = \frac{\int_{A_t} y \, dA_t}{\int_{A_t} dA_t}",
        "y_{NA} = (\u222b y dA_{t}) / (\u222b dA_{t})",
    )
    cd.add_rich_paragraph(
        "and I_{u} is the transformed second moment about that NA. For an "
        "arbitrary polygon this is done by numerical integration over the "
        "meshed concrete area; for the validation in Section 5 the same "
        "result is obtained in closed form for the bounding rectangle."
    )

    cd.add_heading("Cracked Section Properties", level=2)
    cd.add_rich_paragraph(
        "Under pure bending the tension concrete is assumed cracked and "
        "discounted, per EC2 \u00a77.1(2). The cracked NA depth x_{c} is "
        "found from force equilibrium of the transformed section, with the "
        "concrete in compression contributing a linear stress distribution "
        "(elastic, since this is a serviceability state) and the steel "
        "transformed by the modular ratio n."
    )
    cd.add_rich_paragraph(
        "For the standard rectangular case used in validation, this reduces "
        "to the textbook quadratic:"
    )
    _eq_safe(
        cd,
        r"\frac{B \, x^2}{2} = n \, A_s \, (d - x)",
        "B x^{2} / 2 = n A_{s} (d \u2212 x)",
    )
    cd.add_rich_paragraph("which solves to:")
    _eq_safe(
        cd,
        r"x = \frac{-a + \sqrt{a^2 + 4 a d}}{2}, \quad a = \frac{2 \, n \, A_s}{B}",
        "x = (\u2212a + sqrt(a^{2} + 4ad)) / 2,   a = 2 n A_{s} / B",
    )
    cd.add_rich_paragraph("with the cracked second moment about the NA:")
    _eq_safe(
        cd,
        r"I_c = \frac{B \, x^3}{3} + n \, A_s \, (d - x)^2",
        "I_{c} = B x^{3} / 3 + n A_{s} (d \u2212 x)^{2}",
    )

    cd.add_heading("Cracking Moment", level=2)
    cd.add_rich_paragraph("The cracking moment is taken as:")
    _eq_safe(
        cd,
        r"M_{\mathrm{cr}} = \frac{f_{\mathrm{ctm}} \, I_u}{y_t}",
        "M_{cr} = f_{ctm} I_{u} / y_{t}",
    )
    cd.add_rich_paragraph(
        "where y_{t} is the distance from the elastic NA to the extreme "
        "tension fibre. This is the EC2 \u00a77.1 definition used as the "
        "threshold between the uncracked and cracked stiffness branches in "
        "deflection calculations to EC2 \u00a77.4.3."
    )

    cd.add_heading("ULS Bending Capacity", level=2)
    cd.add_rich_paragraph(
        "The ULS capacity is computed by sweeping the neutral axis position "
        "and integrating the concrete and steel stresses over the section. "
        "Concrete in compression follows the parabola-rectangle stress-"
        "strain law of EC2 \u00a73.1.7(1), Figure 3.3, with peak stress "
        "\u03b7 f_{cd} and limit strain \u03b5_{cu2} = 3.5 \u2030 for "
        "f_{ck} \u2264 50 MPa (modified per EC2 Table 3.1 for higher "
        "strengths). Concrete in tension is ignored. Steel follows the "
        "elastic-perfectly-plastic law described above."
    )
    cd.add_rich_paragraph(
        "At each NA position the section is in equilibrium with some axial "
        "force F and bending moment M, both recorded and joined by a spline "
        "to form the F-M interaction diagram. M_{Rd} at a given axial force "
        "is read off the spline by inversion. With F = 0 this is the pure "
        "bending capacity reported in the summary."
    )
    cd.add_rich_paragraph(
        "The validation in Section 5 compares this to the simplified "
        "rectangular stress block of EC2 \u00a73.1.7(3), Figure 3.5: depth "
        "\u03bb x with stress \u03b7 f_{cd}. The stress block parameters "
        "are:"
    )
    _eq_safe(
        cd,
        r"\lambda = 0.8, \quad \eta = 1.0 \quad \mathrm{for}\ f_{\mathrm{ck}} \le 50\ \mathrm{MPa}",
        "\u03bb = 0.8,  \u03b7 = 1.0   for f_{ck} <= 50 MPa",
    )
    _eq_safe(
        cd,
        r"\lambda = 0.8 - \frac{f_{\mathrm{ck}} - 50}{400}, \quad \eta = 1.0 - \frac{f_{\mathrm{ck}} - 50}{200} \quad \mathrm{for}\ f_{\mathrm{ck}} > 50\ \mathrm{MPa}",
        "\u03bb = 0.8 \u2212 (f_{ck} \u2212 50) / 400,  \u03b7 = 1.0 \u2212 (f_{ck} \u2212 50) / 200   for f_{ck} > 50 MPa",
    )
    cd.add_rich_paragraph(
        "Force equilibrium with no axial load and the rectangular block "
        "gives:"
    )
    _eq_safe(
        cd,
        r"\lambda \, x \, B \, \eta \, f_{\mathrm{cd}} = A_s \, f_{\mathrm{yd}} \;\;\Rightarrow\;\; x = \frac{A_s \, f_{\mathrm{yd}}}{\lambda \, \eta \, f_{\mathrm{cd}} \, B}",
        "\u03bb x B \u03b7 f_{cd} = A_{s} f_{yd}  =>  x = A_{s} f_{yd} / (\u03bb \u03b7 f_{cd} B)",
    )
    cd.add_rich_paragraph("The lever arm and bending capacity then follow as:")
    _eq_safe(
        cd,
        r"z = d - \frac{\lambda \, x}{2}, \quad M_{\mathrm{Rd}} = A_s \, f_{\mathrm{yd}} \, z",
        "z = d \u2212 \u03bb x / 2,  M_{Rd} = A_{s} f_{yd} z",
    )
    cd.add_rich_paragraph(
        "The two approaches give values within typically 1 to 3 percent for "
        "normal rectangular beams; the parabola-rectangle block tends to "
        "give a very slightly larger lever arm and therefore slightly "
        "higher M_{Rd}."
    )

    # =======================================================================
    # 4. Compliance with Eurocodes
    # =======================================================================
    cd.add_heading("Compliance with Eurocodes", level=1)
    cd.add_rich_paragraph(
        "The methodology above implements the following clauses of "
        "BS EN 1992-1-1:2004 + A1:2014 (Eurocode 2, Part 1-1), with the UK "
        "National Annex values where listed:"
    )
    compliance_rows = [
        ("**Clause / Reference**", "**Aspect**", "**Implementation**"),
        ("\u00a73.1.2(3), Table 3.1", "f_{cm}, f_{ctm}, E_{cm}",
         "f_{cm} = f_{ck} + 8; f_{ctm} and E_{cm} computed as documented "
         "above."),
        ("\u00a73.1.6(1), \u00a73.1.6(2)", "Design strengths f_{cd}, f_{yd}",
         "f_{cd} = \u03b1_{cc} f_{ck} / \u03b3_{c} with \u03b1_{cc} = 1.0 "
         "(UK NA); f_{yd} = f_{yk} / \u03b3_{s}."),
        ("\u00a73.1.7(1), Figure 3.3", "Parabola-rectangle stress-strain (ULS)",
         "Used directly in the fibre integration that builds the F-M "
         "diagram."),
        ("\u00a73.1.7(3), Figure 3.5", "Rectangular stress block (ULS)",
         "Used as the closed-form validation benchmark, with \u03bb and "
         "\u03b7 per Table 3.1 for the entered f_{ck}."),
        ("Table 3.1", "\u03b5_{cu2} limit strain",
         "\u03b5_{cu2} = 3.5 \u2030 for f_{ck} \u2264 50 MPa, reduced per "
         "Table 3.1 for higher strengths."),
        ("\u00a73.2.7(4), Figure 3.8", "Steel stress-strain (ULS)",
         "Elastic-perfectly-plastic with E_{s} = 200 GPa and yield plateau "
         "at f_{yd}."),
        ("Table 2.1N (UK NA)", "Partial factors \u03b3_{c}, \u03b3_{s}",
         "Defaults 1.5 and 1.15 respectively, both user-overridable."),
        ("\u00a75.8.4, \u00a77.4.3(5)", "Effective modulus for creep",
         "E_{c,eff} = E_{cm} / (1 + \u03c6) applied uniformly to compute "
         "long-term elastic and cracked properties."),
        ("\u00a76.1", "Bending and axial force",
         "Full F-M interaction by NA sweep, with M_{Rd} at any F "
         "recoverable from the spline."),
        ("\u00a77.1(2)", "Cracked section behaviour",
         "Tension concrete discounted below the cracked NA."),
        ("Table 3.1", "f_{ctm} for cracking moment",
         "Used in M_{cr} = f_{ctm} I_{u} / y_{t}."),
    ]
    table = cd.add_table(rows=len(compliance_rows), cols=3)
    for i, row in enumerate(compliance_rows):
        is_header = (i == 0)
        for j, txt in enumerate(row):
            _rich_cell(table.rows[i].cells[j], txt, bold=is_header)

    cd.add_rich_paragraph(
        "Items not included in this calculation, and which must be checked "
        "separately if relevant: shear (EC2 \u00a76.2), torsion "
        "(\u00a76.3), punching (\u00a76.4), serviceability deflections "
        "(\u00a77.4), crack widths (\u00a77.3), minimum reinforcement "
        "(\u00a79.2.1.1 for beams, \u00a79.3.1.1 for slabs, \u00a79.5.2 "
        "for columns), and detailing (Section 9)."
    )

    # =======================================================================
    # 5. Validation against closed-form benchmarks
    # =======================================================================
    cd.add_heading("Validation", level=1)
    header_note, val_rows, bm, is_rect = validation_rows(sec, mat, t_long, creep)
    cd.add_rich_paragraph(header_note)

    cd.add_heading("Benchmark intermediate values", level=2)
    cd.add_rich_paragraph(
        "The benchmark uses the following derived values for the comparison "
        "rectangle (B \u00d7 H) with a single tension steel layer at the "
        "area centroid of all bars."
    )
    bm_summary = [
        ("Bounding B \u00d7 H",
         f"{ef(bm['B'], 4)} \u00d7 {ef(bm['H'], 4)} mm"),
        ("Total tension area A_{s,eff}",
         f"{ef(bm['A_s'], 4)} mm^{{2}}"),
        ("Effective depth d",
         f"{ef(bm['d'], 4)} mm"),
        ("E_{cm} (k_{E} formulation)",
         f"{ef(bm['E_cm'], 4)} MPa"),
        ("E_{c,eff} = E_{cm} / (1 + \u03c6)",
         f"{ef(bm['E_c_eff'], 4)} MPa"),
        ("Modular ratio n, short term",
         f"{ef(bm['n_st'], 4)}"),
        ("Modular ratio n, long term",
         f"{ef(bm['n_lt'], 4)}"),
        ("f_{ctm}",
         f"{ef(bm['f_ctm'], 4)} MPa"),
        ("f_{cd} = \u03b1_{cc} f_{ck} / \u03b3_{c}",
         f"{ef(bm['f_cd'], 4)} MPa  (\u03b1_{{cc}} = {bm['alpha_cc']:.2f})"),
        ("f_{yd} = f_{yk} / \u03b3_{s}",
         f"{ef(bm['f_yd'], 4)} MPa"),
        ("Stress block \u03bb, \u03b7",
         f"\u03bb = {bm['lam']:.3f}, \u03b7 = {bm['eta']:.3f}"),
    ]
    table = cd.add_table(rows=len(bm_summary), cols=2)
    for i, (k, v) in enumerate(bm_summary):
        _rich_cell(table.rows[i].cells[0], k)
        _rich_cell(table.rows[i].cells[1], v)

    cd.add_heading("Comparison table", level=2)
    cd.add_rich_paragraph(
        "Each row reports the value produced by the numerical solver, the "
        "closed-form benchmark value, and the percentage difference "
        "(computed minus benchmark) divided by the benchmark."
    )
    label_map = {
        "Uncracked I_u (short term)": "Uncracked I_{u} (short term)",
        "Cracked NA y (short term, from bottom)":
            "Cracked NA y (short term, from bottom)",
        "Cracked I_c (short term)": "Cracked I_{c} (short term)",
        "Cracking moment M_cr (short term)":
            "Cracking moment M_{cr} (short term)",
        "Uncracked I_u (long term)": "Uncracked I_{u} (long term)",
        "Cracked I_c (long term)": "Cracked I_{c} (long term)",
        "Cracking moment M_cr (long term)":
            "Cracking moment M_{cr} (long term)",
        "ULS bending capacity M_Rd (N=0)":
            "ULS bending capacity M_{Rd} (N = 0)",
    }
    unit_map = {"mm^4": "mm^{4}", "mm^2": "mm^{2}"}
    notes_map = {
        "Transformed area, n = E_s / E_cm":
            "Transformed area, n = E_{s} / E_{cm}",
        "Quadratic in x, tension steel only (y = H - x)":
            "Quadratic in x, tension steel only (y = H \u2212 x)",
        "B x^3 / 3 + n A_s (d - x)^2":
            "B x^{3} / 3 + n A_{s} (d \u2212 x)^{2}",
        "f_ctm I_u / y_t, EC2 Table 3.1":
            "f_{ctm} I_{u} / y_{t}, EC2 Table 3.1",
        "Effective modulus E_c,eff = E_cm / (1 + phi)":
            "Effective modulus E_{c,eff} = E_{cm} / (1 + \u03c6)",
        "Same f_ctm, long term I_u and y_t":
            "Same f_{ctm}, long term I_{u} and y_{t}",
    }
    val_rows_rich = [
        ("**Quantity**", "**Units**", "**Computed**", "**Benchmark**",
         "**Difference**", "**Notes**"),
    ]
    for row in val_rows[1:]:
        q, u, comp, bench, diff, notes = row
        q_rich = label_map.get(q, q)
        u_rich = unit_map.get(u, u)
        notes_rich = notes_map.get(notes, notes)
        val_rows_rich.append((q_rich, u_rich, comp, bench, diff, notes_rich))

    table = cd.add_table(rows=len(val_rows_rich), cols=6)
    # Disable autofit and set explicit column widths. The Quantity (col 0) and
    # Notes (col 5) columns hold the longest text and need most of the width;
    # Units / Computed / Benchmark / Difference (cols 1-4) are short numbers.
    # Total target width = 16 cm, matching the picture widths used elsewhere.
    table.autofit = False
    col_widths_cm = [4.2, 1.4, 2.0, 2.0, 1.8, 4.6]
    from docx.shared import Cm as _Cm
    for col_idx, w_cm in enumerate(col_widths_cm):
        for row in table.rows:
            row.cells[col_idx].width = _Cm(w_cm)
    for i, row in enumerate(val_rows_rich):
        is_header = (i == 0)
        for j, txt in enumerate(row):
            _rich_cell(table.rows[i].cells[j], str(txt), bold=is_header)

    cd.add_heading("Expected agreement", level=2)
    if is_rect:
        cd.add_rich_paragraph(
            "For a singly reinforced axis-aligned rectangle the elastic "
            "quantities (NA position, I_{u}, I_{c}, M_{cr}) should agree to "
            "within approximately 0.5 percent. Residual differences come "
            "from the mesh used by the solver replacing the closed-form "
            "integrals and, if enabled, from concrete area being subtracted "
            "at the rebar locations (which the benchmark does not subtract, "
            "hence benchmarks may sit slightly above the solver values when "
            "rebar area is removed)."
        )
        cd.add_rich_paragraph(
            "For the ULS capacity, the solver uses the parabola-rectangle "
            "stress block (EC2 \u00a73.1.7(1)) and the benchmark uses the "
            "rectangular block (\u00a73.1.7(3)). Both are accepted by EC2 "
            "and give M_{Rd} values typically within 1 to 3 percent of each "
            "other, with the parabola-rectangle block giving a slightly "
            "larger lever arm. Differences outside that range warrant "
            "investigation."
        )
    else:
        cd.add_rich_paragraph(
            "Because the actual section is not a simple rectangle, larger "
            "deviations are expected. The elastic NA depth and the cracked "
            "NA depth will diverge most for sections whose effective width "
            "in compression differs materially from the bounding box width, "
            "for example flanged sections, tapered sections or sections "
            "with voids. The comparison still gives a useful "
            "order-of-magnitude sanity check, and gross errors (orders of "
            "magnitude, wrong sign) should be picked up immediately."
        )

    

    # =======================================================================
    # 7. References
    # =======================================================================
    cd.add_heading("References", level=1)
    cd.add_rich_paragraph(
        "BS EN 1992-1-1:2004 + A1:2014, Eurocode 2: Design of concrete "
        "structures, Part 1-1: General rules and rules for buildings. BSI."
    )
    cd.add_rich_paragraph(
        "NA to BS EN 1992-1-1:2004 + A1:2014, UK National Annex to "
        "Eurocode 2, Part 1-1. BSI."
    )
    cd.add_rich_paragraph(
        "Mosley, W. H., Bungey, J. H., and Hulse, R. Reinforced Concrete "
        "Design to Eurocode 2 (7th ed.). Palgrave Macmillan. Used for the "
        "rectangular benchmark formulae (cracked NA quadratic, cracked I, "
        "rectangular stress block lever-arm form)."
    )
    cd.add_rich_paragraph(
        "Bhatt, P., MacGinley, T. J., and Choo, B. S. Reinforced Concrete "
        "Design to Eurocodes (4th ed.), CRC Press. Reference for the "
        "transformed section approach to uncracked and cracked elastic "
        "properties."
    )

    out = io.BytesIO()
    cd.save(out)
    out.seek(0)
    return out.getvalue()



# ===========================================================================
# Sidebar - Material properties and analysis settings
# ===========================================================================
with st.sidebar:
    st.header("Material Properties")

    f_ck = st.number_input(
        "Concrete f_ck_28 [MPa]", min_value=12.0, max_value=120.0,
        value=35.0, step=1.0,
    )
    c_class = st.selectbox(
        "Cement class", ["CR", "CN", "CS"], index=1,
        help="CR rapid, CN normal, CS slow",
    )
    t_ref = st.number_input(
        "Reference age t_ref [days]", min_value=1.0, max_value=10000.0,
        value=28.0, step=1.0,
    )
    k_E = st.number_input(
        "Modulus coefficient k_E", min_value=5000.0, max_value=15000.0,
        value=9500.0, step=100.0,
    )
    f_y = st.number_input(
        "Steel f_y [MPa]", min_value=200.0, max_value=900.0,
        value=500.0, step=10.0,
    )

    st.markdown("**Partial factors**")
    col1, col2 = st.columns(2)
    with col1:
        gamma_c = st.number_input("gamma_c", min_value=1.0, max_value=2.0,
                                  value=1.50, step=0.05)
    with col2:
        gamma_s = st.number_input("gamma_s", min_value=1.0, max_value=2.0,
                                  value=1.15, step=0.05)

    st.markdown("**Long-term parameters** (creep)")
    col1, col2 = st.columns(2)
    with col1:
        t_long_yrs = st.number_input("t [yrs]", min_value=0.1, max_value=200.0,
                                     value=50.0, step=1.0)
    with col2:
        creep = st.number_input("phi (creep)", min_value=0.0, max_value=5.0,
                                value=2.2, step=0.1)
    t_long = t_long_yrs * 365.0

    st.divider()
    st.header("Plot Options")
    col1, col2 = st.columns(2)
    with col1:
        plot_incl_uls = st.checkbox("ULS strains", value=True,
                                   help="Show ULS strain diagram and neutral axis.")
        plot_incl_stiffness = st.checkbox("Stiffness", value=True,
                                        help="Show short/long-term stiffness annotations.")
    with col2:
        plot_incl_dims = st.checkbox("Dimensions", value=False,
                                    help="Show section dimensions.")

    with st.expander("Advanced"):
        as_square = st.checkbox(
            "Model rebar as square", value=False,
            help="If checked, rebar is approximated as squares of equivalent area.",
        )
        remove_conc_to_rebar = st.checkbox(
            "Subtract rebar area from concrete", value=True,
        )

    mat = dict(
        f_ck=f_ck, c_class=c_class, t_ref=t_ref, k_E=k_E, f_y=f_y,
        gamma_c=gamma_c, gamma_s=gamma_s,
        as_square=as_square, remove_conc_to_rebar=remove_conc_to_rebar,
    )

    st.divider()
    st.header("DOCX Export Settings")
    project_name = st.text_input("Project name", value="Project")
    project_number = st.text_input("Project number", value="0000")
    design_element = st.text_input("Design element", value="RC Section")
    calc_title = st.text_input("Calc title", value="Section Properties")
    calc_by = st.text_input("Calc by", value="GGS")
    checked_by = st.text_input("Checked by", value="")
    template_path_str = st.text_input(
        "CalcTemplate.docx path",
        value=str(Path(__file__).parent / "CalcTemplate.docx") if "__file__" in globals()
              else "CalcTemplate.docx",
        help="Path to the D4S CalcTemplate.docx file used by CalcDoc.",
    )


# ===========================================================================
# Main area
# ===========================================================================
st.title("Reinforced Concrete Section Analysis")
st.caption(
    "Edit the section vertices and rebar tables, then run analysis. "
    "Both plot() and plot_FM_graph() will be produced."
)

col_inputs, col_preview = st.columns([1.1, 1.0])

with col_inputs:
    tab_vertices, tab_rebar = st.tabs(["Section Vertices", "Rebar"])

    with tab_vertices:
        st.markdown(
            "**Concrete section outline** (min 3 vertices, ordered around the perimeter)."
        )
        vert_df = pd.DataFrame(st.session_state.vertices, columns=["x (mm)", "y (mm)"])
        edited_vert = st.data_editor(
            vert_df,
            num_rows="dynamic",
            use_container_width=True,
            key="vertices_editor",
            height=300,
            column_config={
                "x (mm)": st.column_config.NumberColumn(format="%.2f"),
                "y (mm)": st.column_config.NumberColumn(format="%.2f"),
            },
        )

        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("Reset rectangle", use_container_width=True):
                st.session_state.vertices = _default_vertices()
                st.rerun()
        with c2:
            B_new = st.number_input("B [mm]", 50.0, 5000.0, 300.0, 50.0,
                                    key="quick_B", label_visibility="collapsed")
        with c3:
            H_new = st.number_input("H [mm]", 50.0, 5000.0, 600.0, 50.0,
                                    key="quick_H", label_visibility="collapsed")
        if st.button("Build BxH rectangle from above", use_container_width=True):
            st.session_state.vertices = np.array(
                [[0, 0], [B_new, 0], [B_new, H_new], [0, H_new]], float
            )
            st.rerun()

        # Validate and commit the edit
        try:
            cleaned = edited_vert.dropna()
            if len(cleaned) >= 3:
                st.session_state.vertices = cleaned.to_numpy(dtype=float)
                st.success(f"\u2713 {len(cleaned)} vertices")
            else:
                st.error("Need at least 3 vertices.")
        except Exception as ex:
            st.warning(f"Vertex parse issue: {ex}")

    with tab_rebar:
        st.markdown("**Rebar positions** (x, y in mm, diameter in mm).")
        rebar_df = pd.DataFrame(
            st.session_state.rebar,
            columns=["x (mm)", "y (mm)", "diameter (mm)"],
        )
        edited_rebar = st.data_editor(
            rebar_df,
            num_rows="dynamic",
            use_container_width=True,
            key="rebar_editor",
            height=300,
            column_config={
                "x (mm)": st.column_config.NumberColumn(format="%.2f"),
                "y (mm)": st.column_config.NumberColumn(format="%.2f"),
                "diameter (mm)": st.column_config.NumberColumn(format="%.1f"),
            },
        )

        c1, c2 = st.columns(2)
        with c1:
            if st.button("Clear all rebar", use_container_width=True):
                st.session_state.rebar = np.empty((0, 3), float)
                st.rerun()
        with c2:
            if st.button("Default 3T25 bottom", use_container_width=True):
                st.session_state.rebar = _default_rebar()
                st.rerun()

        try:
            cleaned = edited_rebar.dropna()
            if len(cleaned) >= 1:
                st.session_state.rebar = cleaned.to_numpy(dtype=float)
                st.success(f"\u2713 {len(cleaned)} bars")
            else:
                st.session_state.rebar = np.empty((0, 3), float)
                st.warning("No rebar defined.")
        except Exception as ex:
            st.warning(f"Rebar parse issue: {ex}")

with col_preview:
    st.markdown("**Live Preview**")
    try:
        prev_fig = plot_section_preview(
            st.session_state.vertices, st.session_state.rebar
        )
        st.pyplot(prev_fig, use_container_width=True)
        plt.close(prev_fig)
    except Exception as ex:
        st.error(f"Preview error: {ex}")

    if st.button("\u21bb Rotate 90\u00b0 CW", use_container_width=True,
                 help="Rotate vertices and rebar 90 degrees clockwise about "
                      "the section centroid, then translate to keep the "
                      "section in the positive quadrant."):
        try:
            new_v, new_r = rotate_90_cw(
                st.session_state.vertices, st.session_state.rebar
            )
            st.session_state.vertices = new_v
            st.session_state.rebar = new_r
            st.rerun()
        except Exception as ex:
            st.error(f"Rotate failed: {ex}")

st.divider()

# ===========================================================================
# Run + export buttons
# ===========================================================================
run_col, export_col, _ = st.columns([1, 1, 2])
with run_col:
    run_btn = st.button("Run Analysis", type="primary", use_container_width=True)
with export_col:
    export_btn = st.button("Export to DOCX", use_container_width=True,
                           disabled=(st.session_state.section is None))

if run_btn:
    with st.spinner("Building section and solving..."):
        try:
            sec = build_section(
                st.session_state.vertices,
                st.session_state.rebar,
                mat,
            )

            # All options on except incl_dims, per spec.
            plot_fig = sec.plot(
                show=False,
                incl_uls=plot_incl_uls,
                incl_stiffness=plot_incl_stiffness,
                incl_dims=plot_incl_dims,
                creep=creep,
                t=t_long,
            )

            # plot_FM_graph creates its own figure internally
            fm_fig, fm_ax = plt.subplots(figsize=(10, 7))
            sec.plot_FM_graph(ax=fm_ax, show=False)
            fm_fig.tight_layout()

            st.session_state.section = sec
            st.session_state.plot_fig = plot_fig
            st.session_state.fm_fig = fm_fig
            st.session_state.summary_rows = section_summary(sec, mat, t_long, creep)
            st.success("Analysis complete.")
        except Exception as ex:
            st.session_state.section = None
            st.session_state.plot_fig = None
            st.session_state.fm_fig = None
            st.session_state.summary_rows = None
            st.error(f"Analysis failed: {ex}")
            with st.expander("Traceback"):
                st.code(traceback.format_exc())

# Display results if available
if st.session_state.plot_fig is not None and st.session_state.fm_fig is not None:
    res_left, res_right = st.columns([1.05, 1.0])
    with res_left:
        st.subheader("Section Plot")
        st.pyplot(st.session_state.plot_fig, use_container_width=True)
    with res_right:
        st.subheader("Force-Moment Interaction")
        st.pyplot(st.session_state.fm_fig, use_container_width=True)

    if st.session_state.summary_rows is not None:
        with st.expander("Section property summary", expanded=False):
            # Strip CalcDoc rich-text markup so labels display cleanly in
            # Streamlit. _{ck} -> _ck, ^{2} -> ^2, **x** -> x.
            import re as _re_strip
            def _strip_markup(s):
                s = str(s)
                s = _re_strip.sub(r"\*\*(.*?)\*\*", r"\1", s)
                s = _re_strip.sub(r"_\{([^}]*)\}", r"_\1", s)
                s = _re_strip.sub(r"\^\{([^}]*)\}", r"^\1", s)
                return s
            clean_rows = [
                (_strip_markup(k), _strip_markup(v))
                for k, v in st.session_state.summary_rows
            ]
            df = pd.DataFrame(clean_rows, columns=["Property", "Value"])
            st.dataframe(df, use_container_width=True, hide_index=True)

# Handle export
if export_btn:
    if st.session_state.section is None:
        st.error("Run analysis first.")
    else:
        try:
            with st.spinner("Building DOCX..."):
                docx_bytes = build_calcdoc_bytes(
                    sec=st.session_state.section,
                    mat=mat,
                    t_long=t_long,
                    creep=creep,
                    plot_fig=st.session_state.plot_fig,
                    fm_fig=st.session_state.fm_fig,
                    project_name=project_name,
                    project_number=project_number,
                    design_element=design_element,
                    calc_title=calc_title,
                    calc_by=calc_by,
                    checked_by=checked_by,
                    template_path=template_path_str or None,
                )
            fname = (
                f"{project_number}_{design_element.replace(' ', '_')}"
                f"_{datetime.now().strftime('%Y%m%d_%H%M')}.docx"
            )
            st.success("DOCX built. Use the button below to download.")
            st.download_button(
                label="Download DOCX",
                data=docx_bytes,
                file_name=fname,
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                use_container_width=True,
            )
        except Exception as ex:
            st.error(f"Export failed: {ex}")
            with st.expander("Traceback"):
                st.code(traceback.format_exc())