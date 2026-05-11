"""
RC Core Section Analysis Streamlit App
======================================
Wraps CoreDesigner.CoreSection in an interactive UI.

Defaults vs the beam app:
- as_square = True   (rebar modelled as equivalent squares, default for cores)
- remove_conc_to_rebar = False  (concrete area not netted off for rebar)

Features
--------
1. Material property inputs in the sidebar (concrete, steel, factors).
2. A single editable table of walls with dynamic add/delete rows:
   x1, y1, x2, y2, thk, cover, v_dia_in, v_dia_out, v_spc_in, v_spc_out,
   h_dia_in, h_dia_out, h_spc_in, h_spc_out
3. Single section-level toggle for verts_outer_layer (vertical rebar
   placed outside the horizontals).
4. Live preview of wall polygons and distributed verticals.
5. "Run Analysis" produces:
   - sec.plot(incl_uls=True, incl_stiffness=True, incl_dims=False)
   - sec.plot_FM_graph()
6. "Export to DOCX" writes a CalcDoc report containing both plots and
   a tabulated section property summary.

Usage:
    streamlit run RC_core_section_app.py
"""

from __future__ import annotations

import io
import traceback
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MPLPolygon, Circle, Rectangle

import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Module imports (CoreDesigner + CalcDoc). Imports fail loud and early.
# ---------------------------------------------------------------------------
try:
    from CoreDesigner import CoreSection
    from BugBasics import Point, Line
except ImportError as e:
    st.set_page_config(page_title="RC Core Section", layout="wide")
    st.error(f"Could not import CoreDesigner / BugBasics: {e}")
    st.info(
        "Place CoreDesigner.py, RC_beam_mesh.py, BugBasics.py, BugPoly.py, "
        "Mesh.py, Notation.py on the Python path next to this script."
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
    page_title="RC Core Section Analysis",
    page_icon="\U0001f3db",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Column names for the walls table. Kept short to fit on screen.
WALL_COLS = [
    "x1", "y1", "x2", "y2",
    "thk", "cover",
    "v_dia_in", "v_dia_out",
    "v_spc_in", "v_spc_out",
    "h_dia_in", "h_dia_out",
    "h_spc_in", "h_spc_out",
]


def _default_walls() -> pd.DataFrame:
    """Default geometry: a 3000 x 3000 C-shaped core with 250 mm walls.

    Three walls forming a 'C' open on the right:
      - bottom: (0,0) -> (3000,0)
      - left:   (0,0) -> (0,3000)
      - top:    (0,3000) -> (3000,3000)

    Verticals T16 @ 200 ctrs each face, horizontals T12 @ 200 ctrs each face,
    cover 40 mm.
    """
    rows = [
        # x1, y1, x2,   y2,   thk, cover, vDi, vDo, vSi, vSo, hDi, hDo, hSi, hSo
        [   0,  0, 3000,    0, 250,    40,  16,  16, 200, 200,  12,  12, 200, 200],
        [   0,  0,    0, 3000, 250,    40,  16,  16, 200, 200,  12,  12, 200, 200],
        [   0,3000, 3000, 3000, 250,    40,  16,  16, 200, 200,  12,  12, 200, 200],
    ]
    return pd.DataFrame(rows, columns=WALL_COLS, dtype=float)


def _init_state():
    if "walls" not in st.session_state:
        st.session_state.walls = _default_walls()
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
# Preview plot helpers
# ---------------------------------------------------------------------------
def _wall_polygon_corners(x1, y1, x2, y2, thk):
    """Return the 4 corner points of a wall extruded perpendicular to its
    centreline by +/- thk/2. Matches CoreDesigner's SimplePolygon.from_polyline
    convention (offset along the line normal)."""
    dx = x2 - x1
    dy = y2 - y1
    L = float(np.hypot(dx, dy))
    if L < 1e-9:
        return None
    # Unit normal (rotate tangent 90 CCW)
    nx = -dy / L
    ny = dx / L
    half = thk / 2.0
    return np.array([
        [x1 + nx * half, y1 + ny * half],
        [x2 + nx * half, y2 + ny * half],
        [x2 - nx * half, y2 - ny * half],
        [x1 - nx * half, y1 - ny * half],
    ])


def _wall_bar_positions(x1, y1, x2, y2, thk, cover,
                        v_dia_in, v_dia_out, v_spc_in, v_spc_out,
                        h_dia_in, h_dia_out,
                        verts_outer_layer):
    """Distribute vertical bars along both faces of a wall, replicating
    CoreDesigner's logic. Used for live preview only; the analysis itself
    goes through CoreSection which does the same thing internally.

    Returns (centres_inner, dia_inner, centres_outer, dia_outer) where
    "inner" is the face on the negative-normal side and "outer" the
    positive-normal side (signs match the n-vector convention).
    """
    dx = x2 - x1
    dy = y2 - y1
    L = float(np.hypot(dx, dy))
    if L < 1e-9:
        return np.empty((0, 2)), [], np.empty((0, 2)), []
    tx = dx / L
    ty = dy / L
    # Normal (90 CCW)
    nx = -ty
    ny = tx

    half = thk / 2.0
    # Effective offset from centreline of each face
    # Inner = side 0 (positive normal in CoreDesigner indexing of bar_lines):
    # bar_lines[k] = wall.move(thk_ef[k] * n) where thk_ef = [-thk/2+c_add[0], thk/2-c_add[1]]
    # So face 0 sits at -half + c_add_0 along n, face 1 at +half - c_add_1.
    c_add_0 = cover + v_dia_in / 2 + (0 if verts_outer_layer else h_dia_in)
    c_add_1 = cover + v_dia_out / 2 + (0 if verts_outer_layer else h_dia_out)
    off_0 = -half + c_add_0
    off_1 = half - c_add_1

    def _along_line(spc, off):
        # half_shift=True: bars are placed at half-spacing from each end,
        # one bar at (s/2, 3s/2, ...) along the line, mirroring arange(half_shift=True)
        if spc <= 0:
            return np.empty((0, 2))
        n_bars = int(np.floor(L / spc))
        # If n_bars*spc < L, distribute remaining gap as a half-shift each end.
        # CoreDesigner's PolyLine.arange(s, half_shift=True) typically places
        # bars at s/2, 3s/2, ... up to L - s/2. We replicate that.
        if n_bars < 1:
            return np.empty((0, 2))
        # bar positions from line start
        ss = (np.arange(n_bars) + 0.5) * spc
        # if last bar would overshoot L - small tol, ok; if there's a remainder
        # gap at both ends, half_shift means start offset = (L - (n_bars-1)*spc) / 2
        # Use that more accurate form:
        start = (L - (n_bars - 1) * spc) / 2.0
        ss = start + np.arange(n_bars) * spc
        pts = np.column_stack([
            x1 + tx * ss + nx * off,
            y1 + ty * ss + ny * off,
        ])
        return pts

    pts_in = _along_line(v_spc_in, off_0)
    pts_out = _along_line(v_spc_out, off_1)
    dia_in = np.full(len(pts_in), v_dia_in)
    dia_out = np.full(len(pts_out), v_dia_out)
    return pts_in, dia_in, pts_out, dia_out


def plot_section_preview(walls_df: pd.DataFrame,
                          verts_outer_layer: bool) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 7), dpi=90)

    if len(walls_df) == 0:
        ax.set_title("Add walls to begin")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        return fig

    all_xy = []  # for setting axis limits

    for i, row in walls_df.iterrows():
        try:
            x1, y1, x2, y2 = float(row.x1), float(row.y1), float(row.x2), float(row.y2)
            thk = float(row.thk)
            cover = float(row.cover)
            v_dia_in, v_dia_out = float(row.v_dia_in), float(row.v_dia_out)
            v_spc_in, v_spc_out = float(row.v_spc_in), float(row.v_spc_out)
            h_dia_in, h_dia_out = float(row.h_dia_in), float(row.h_dia_out)
        except (ValueError, TypeError):
            continue

        corners = _wall_polygon_corners(x1, y1, x2, y2, thk)
        if corners is None:
            continue
        all_xy.append(corners)

        poly = MPLPolygon(
            corners, closed=True,
            facecolor="#cce6ff", edgecolor="#003366",
            linewidth=1.5, alpha=0.6,
        )
        ax.add_patch(poly)

        # Centreline
        ax.plot([x1, x2], [y1, y2], "--", color="#003366", lw=0.8, alpha=0.6)

        # Wall index label at midpoint
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mx, my, f"W{i + 1}", fontsize=9, color="#003366",
                ha="center", va="center", weight="bold",
                bbox=dict(boxstyle="round,pad=0.15", fc="white",
                          ec="#003366", alpha=0.8))

        # Verticals
        pts_in, dia_in, pts_out, dia_out = _wall_bar_positions(
            x1, y1, x2, y2, thk, cover,
            v_dia_in, v_dia_out, v_spc_in, v_spc_out,
            h_dia_in, h_dia_out, verts_outer_layer,
        )
        for pts, dias, label_color in [
            (pts_in, dia_in, "red"),
            (pts_out, dia_out, "darkred"),
        ]:
            for (bx, by), bd in zip(pts, dias):
                ax.add_patch(Rectangle(
                    (bx - bd / 2, by - bd / 2), bd, bd,
                    facecolor=label_color, edgecolor="black",
                    linewidth=0.5, alpha=0.85,
                ))

    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_title("Live Section Preview")

    if all_xy:
        all_xy = np.vstack(all_xy)
        x_min, y_min = all_xy.min(axis=0)
        x_max, y_max = all_xy.max(axis=0)
        pad = max(x_max - x_min, y_max - y_min) * 0.10 + 50
        ax.set_xlim(x_min - pad, x_max + pad)
        ax.set_ylim(y_min - pad, y_max + pad)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Section builder
# ---------------------------------------------------------------------------
def build_section(walls_df: pd.DataFrame, mat: dict) -> CoreSection:
    """Build a CoreSection from the walls dataframe and material dict.

    CoreDesigner.CoreSection broadcasts per-wall, per-side arrays from the
    inputs, so we assemble per-wall lists of [inner, outer] pairs.
    """
    walls_df = walls_df.dropna().reset_index(drop=True)
    if len(walls_df) == 0:
        raise ValueError("No walls defined.")

    wall_lines = []
    wall_thk = []
    v_dias = []
    v_spacing = []
    h_dias = []
    h_spacing = []
    cover = []

    for _, row in walls_df.iterrows():
        p1 = Point(float(row.x1), float(row.y1))
        p2 = Point(float(row.x2), float(row.y2))
        if (p1 - p2).magnitude < 1e-6:
            continue
        wall_lines.append(Line(p1, p2))
        wall_thk.append(float(row.thk))
        v_dias.append([float(row.v_dia_in), float(row.v_dia_out)])
        v_spacing.append([float(row.v_spc_in), float(row.v_spc_out)])
        h_dias.append([float(row.h_dia_in), float(row.h_dia_out)])
        h_spacing.append([float(row.h_spc_in), float(row.h_spc_out)])
        cover.append([float(row.cover), float(row.cover)])

    if not wall_lines:
        raise ValueError("No valid walls (all had zero length?).")

    return CoreSection(
        f_ck=float(mat["f_ck"]),
        wall_lines=wall_lines,
        wall_thk=wall_thk,
        v_dias=v_dias,
        v_spacing=v_spacing,
        h_dias=h_dias,
        h_spacing=h_spacing,
        cover=cover,
        c_class=mat["c_class"],
        t_ref=float(mat["t_ref"]),
        k_E=float(mat["k_E"]),
        f_y=float(mat["f_y"]),
        gamma_s=float(mat["gamma_s"]),
        gamma_c=float(mat["gamma_c"]),
        verts_outer_layer=bool(mat["verts_outer_layer"]),
    )


# ---------------------------------------------------------------------------
# 90-degree clockwise rotation of wall endpoints
# ---------------------------------------------------------------------------
def rotate_90_cw(walls_df: pd.DataFrame) -> pd.DataFrame:
    """Rotate all wall endpoints 90 degrees clockwise about the bounding-box
    centre of the wall endpoints, then translate so the bounding box minimum
    sits at the origin. Wall properties (thk, cover, rebar) are unchanged.
    """
    df = walls_df.dropna().copy().reset_index(drop=True)
    if len(df) == 0:
        return walls_df.copy()

    xs = np.concatenate([df.x1.to_numpy(float), df.x2.to_numpy(float)])
    ys = np.concatenate([df.y1.to_numpy(float), df.y2.to_numpy(float)])
    cx = 0.5 * (xs.min() + xs.max())
    cy = 0.5 * (ys.min() + ys.max())

    # 90 deg CW: (x, y) -> (cx + (y - cy), cy - (x - cx))
    def rot(px, py):
        return cx + (py - cy), cy - (px - cx)

    new_x1, new_y1 = rot(df.x1.to_numpy(float), df.y1.to_numpy(float))
    new_x2, new_y2 = rot(df.x2.to_numpy(float), df.y2.to_numpy(float))

    all_x = np.concatenate([new_x1, new_x2])
    all_y = np.concatenate([new_y1, new_y2])
    x_min = all_x.min()
    y_min = all_y.min()

    df.x1 = new_x1 - x_min
    df.y1 = new_y1 - y_min
    df.x2 = new_x2 - x_min
    df.y2 = new_y2 - y_min
    return df


# ---------------------------------------------------------------------------
# Section property summary (used by the docx export and the in-page table)
# ---------------------------------------------------------------------------
def section_summary(core_sec: CoreSection, mat: dict,
                     t_long: float, creep: float):
    """Return a list of (label, value) tuples covering all key properties.

    Properties are read from CoreSection.v_section (the wrapped RCSection).
    Robust to None / NaN / inf coming out of the solver, since `ef` (via
    math.log10) blows up on non-finite values.
    """
    sec = core_sec.v_section

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
    _x_d_err = None
    try:
        M_Rd, _F, na_uls = sec.M_Rd(F_target=0.0, return_F=True, incl_x=True)
    except TypeError as e:
        # incl_x kwarg not present - falls back to old signature and surfaces
        # the issue so it's obvious what needs updating.
        _x_d_err = (
            f"M_Rd does not accept incl_x: {e}. RC_beam_mesh.py probably "
            "wasn't updated with the FM_graph/M_Rd patch. Falling back to "
            "the old signature (na_uls will be None)."
        )
        try:
            M_Rd, _F, na_uls = sec.M_Rd(F_target=0.0, return_F=True)
        except Exception as e2:
            M_Rd, na_uls = float("nan"), None
            _x_d_err = f"M_Rd failed: {e2}"
    except Exception as e:
        M_Rd, na_uls = float("nan"), None
        _x_d_err = f"M_Rd failed: {e}"

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
        ("Material", ""),
        ("Concrete strength f_ck_28", f"{_safe_ef(mat['f_ck'])} MPa"),
        ("Cement class", mat["c_class"]),
        ("Reference age t_ref", f"{_safe_ef(mat['t_ref'])} days"),
        ("Modulus coefficient k_E", f"{_safe_ef(mat['k_E'])}"),
        ("Steel yield f_y", f"{_safe_ef(mat['f_y'])} MPa"),
        ("Partial factor gamma_c", f"{mat['gamma_c']:.2f}"),
        ("Partial factor gamma_s", f"{mat['gamma_s']:.2f}"),
        ("Verticals outer layer",
         "yes" if mat["verts_outer_layer"] else "no"),
        ("Geometry", ""),
        ("Number of walls", str(len(core_sec.wall_lines))),
        ("Bounding box B x H", f"{_safe_ef(sec.B)} x {_safe_ef(sec.H)} mm"),
        ("Concrete area A_c", f"{_safe_ef(A_c)} mm^2"),
        ("Steel area A_s", f"{_safe_ef(A_s)} mm^2"),
        ("Reinforcement ratio rho", f"{rho:.2f}%"),
        ("Steel area below NA A_st", f"{_safe_ef(A_st)} mm^2"),
        ("Tension ratio rho_t", f"{rho_t:.2f}%"),
        ("Effective depth d_ef", f"{_safe_ef(sec.d_ef)} mm"),
        ("Elastic (uncracked) properties", ""),
        ("Elastic NA y", f"{_safe_ef(na)} mm"),
        ("Uncracked NA depth x_u", f"{_safe_ef(sec.y_max - na)} mm"),
        ("Cracked NA depth x_c", f"{_safe_ef(x_c)} mm"),
        ("Short term I_u", f"{_safe_ef(I_u_st)} mm^4"),
        ("Short term I_c", f"{_safe_ef(I_c_st)} mm^4"),
        ("Short term M_cr", f"{_safe_ef(M_cr_st_kNm)} kNm"),
        ("Long term (t, phi)", f"t={_safe_ef(t_long / 365)} yrs, phi={_safe_ef(creep)}"),
        ("Long term I_u", f"{_safe_ef(I_u_lt)} mm^4"),
        ("Long term I_c", f"{_safe_ef(I_c_lt)} mm^4"),
        ("Long term M_cr", f"{_safe_ef(M_cr_lt_kNm)} kNm"),
        ("Ultimate limit state", ""),
        ("Bending capacity M_Rd (F=0)", f"{_safe_ef(M_Rd_kNm)} kNm"),
        ("Design NA depth x_d", f"{_safe_ef(x_d)} mm"),
        ("Depth ratio x_d / d_ef", f"{_safe_ef(x_d_ratio)}"),
    ]
    return rows


# ---------------------------------------------------------------------------
# DOCX export via CalcDoc
# ---------------------------------------------------------------------------
def build_calcdoc_bytes(
    sec: CoreSection,
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

    cd.add_heading("Section Properties", level=1)

    cd.add_heading("Inputs", level=2)
    rows = section_summary(sec, mat, t_long, creep)

    # Render the summary as a 2-column table.
    table = cd.add_table(rows=len(rows), cols=2)
    for i, (k, v) in enumerate(rows):
        cells = table.rows[i].cells
        cells[0].text = k
        cells[1].text = str(v)
        # Make section headers bold (rows where value is empty).
        if v == "":
            for cell in cells:
                for para in cell.paragraphs:
                    for run in para.runs:
                        run.bold = True

    cd.add_heading("Section Plot", level=2)
    buf_plot = io.BytesIO()
    plot_fig.savefig(buf_plot, format="png", dpi=180, bbox_inches="tight",
                     facecolor="white")
    buf_plot.seek(0)
    cd.add_picture(buf_plot, width=Cm(16))

    cd.add_heading("Force-Moment Interaction (FM Graph)", level=2)
    buf_fm = io.BytesIO()
    fm_fig.savefig(buf_fm, format="png", dpi=180, bbox_inches="tight",
                   facecolor="white")
    buf_fm.seek(0)
    cd.add_picture(buf_fm, width=Cm(16))

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

    st.markdown("**Reinforcement layout**")
    verts_outer_layer = st.checkbox(
        "Verticals as outer layer", value=False,
        help="If checked, vertical bars sit outside the horizontals "
             "(cover offset = cover + v_dia/2). Otherwise verticals sit "
             "inside the horizontals (cover offset = cover + v_dia/2 + h_dia).",
    )

    mat = dict(
        f_ck=f_ck, c_class=c_class, t_ref=t_ref, k_E=k_E, f_y=f_y,
        gamma_c=gamma_c, gamma_s=gamma_s,
        verts_outer_layer=verts_outer_layer,
    )

    st.divider()
    st.header("DOCX Export Settings")
    project_name = st.text_input("Project name", value="Project")
    project_number = st.text_input("Project number", value="0000")
    design_element = st.text_input("Design element", value="RC Core Section")
    calc_title = st.text_input("Calc title", value="Core Section Properties")
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
st.title("RC Core Section Analysis")
st.caption(
    "Edit the walls table (one row per wall, all rebar dia/spacing per face), "
    "then run analysis. Both plot() and plot_FM_graph() will be produced."
)

col_inputs, col_preview = st.columns([1.4, 1.0])

with col_inputs:
    st.markdown(
        "**Walls table.** Each row defines a wall by centreline endpoints "
        "(x1,y1) to (x2,y2), thickness, cover, and per-face vertical and "
        "horizontal rebar (in = side 0, out = side 1). Add or delete rows "
        "directly in the table."
    )

    edited_walls = st.data_editor(
        st.session_state.walls,
        num_rows="dynamic",
        use_container_width=True,
        key="walls_editor",
        height=320,
        column_config={
            "x1": st.column_config.NumberColumn("x1", format="%.1f", help="Start x [mm]"),
            "y1": st.column_config.NumberColumn("y1", format="%.1f", help="Start y [mm]"),
            "x2": st.column_config.NumberColumn("x2", format="%.1f", help="End x [mm]"),
            "y2": st.column_config.NumberColumn("y2", format="%.1f", help="End y [mm]"),
            "thk": st.column_config.NumberColumn("thk", format="%.0f", help="Wall thickness [mm]"),
            "cover": st.column_config.NumberColumn("cover", format="%.0f", help="Cover to outermost steel [mm]"),
            "v_dia_in": st.column_config.NumberColumn("v\u00f8 in", format="%.0f", help="Vertical bar dia, inner face [mm]"),
            "v_dia_out": st.column_config.NumberColumn("v\u00f8 out", format="%.0f", help="Vertical bar dia, outer face [mm]"),
            "v_spc_in": st.column_config.NumberColumn("v sp in", format="%.0f", help="Vertical bar spacing, inner face [mm]"),
            "v_spc_out": st.column_config.NumberColumn("v sp out", format="%.0f", help="Vertical bar spacing, outer face [mm]"),
            "h_dia_in": st.column_config.NumberColumn("h\u00f8 in", format="%.0f", help="Horizontal bar dia, inner face [mm]"),
            "h_dia_out": st.column_config.NumberColumn("h\u00f8 out", format="%.0f", help="Horizontal bar dia, outer face [mm]"),
            "h_spc_in": st.column_config.NumberColumn("h sp in", format="%.0f", help="Horizontal bar spacing, inner face [mm]"),
            "h_spc_out": st.column_config.NumberColumn("h sp out", format="%.0f", help="Horizontal bar spacing, outer face [mm]"),
        },
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("Reset C-core", use_container_width=True):
            st.session_state.walls = _default_walls()
            st.rerun()
    with c2:
        if st.button("Clear all walls", use_container_width=True):
            st.session_state.walls = pd.DataFrame(columns=WALL_COLS, dtype=float)
            st.rerun()
    with c3:
        if st.button("Mirror inner -> outer", use_container_width=True,
                     help="Copy inner-face dia/spacing values to the outer "
                          "face for every wall."):
            df = edited_walls.copy()
            df["v_dia_out"] = df["v_dia_in"]
            df["v_spc_out"] = df["v_spc_in"]
            df["h_dia_out"] = df["h_dia_in"]
            df["h_spc_out"] = df["h_spc_in"]
            st.session_state.walls = df
            st.rerun()

    # Commit edits
    try:
        cleaned = edited_walls.dropna()
        if len(cleaned) >= 1:
            st.session_state.walls = cleaned.reset_index(drop=True).astype(float)
            st.success(f"\u2713 {len(cleaned)} walls")
        else:
            st.session_state.walls = pd.DataFrame(columns=WALL_COLS, dtype=float)
            st.warning("No walls defined.")
    except Exception as ex:
        st.warning(f"Walls parse issue: {ex}")

with col_preview:
    st.markdown("**Live Preview**")
    try:
        prev_fig = plot_section_preview(
            st.session_state.walls, verts_outer_layer,
        )
        st.pyplot(prev_fig, use_container_width=True)
        plt.close(prev_fig)
    except Exception as ex:
        st.error(f"Preview error: {ex}")

    if st.button("\u21bb Rotate 90\u00b0 CW", use_container_width=True,
                 help="Rotate all wall endpoints 90 degrees clockwise about "
                      "the section bounding-box centre, then translate to "
                      "keep the section in the positive quadrant. Wall "
                      "thicknesses, covers, and rebar are unchanged."):
        try:
            st.session_state.walls = rotate_90_cw(st.session_state.walls)
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
    with st.spinner("Building core section and solving..."):
        try:
            core_sec = build_section(st.session_state.walls, mat)
            sec_v = core_sec.v_section  # underlying RCSection

            # All options on except incl_dims, per spec.
            plot_fig = sec_v.plot(
                show=False,
                incl_uls=True,
                incl_stiffness=True,
                incl_dims=False,
                creep=creep,
                t=t_long,
            )

            # plot_FM_graph creates its own figure internally
            fm_fig, fm_ax = plt.subplots(figsize=(10, 7))
            sec_v.plot_FM_graph(ax=fm_ax, show=False)
            fm_fig.tight_layout()

            st.session_state.section = core_sec
            st.session_state.plot_fig = plot_fig
            st.session_state.fm_fig = fm_fig
            st.session_state.summary_rows = section_summary(
                core_sec, mat, t_long, creep
            )
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
            df = pd.DataFrame(
                st.session_state.summary_rows, columns=["Property", "Value"]
            )
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
