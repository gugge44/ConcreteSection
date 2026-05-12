"""
RC Core Section Analysis Streamlit App
======================================
Wraps CoreDesigner.CoreSection in an interactive UI.

A core is defined by a list of straight wall centre-lines. Each wall row
in the input table carries its own thickness, cover and reinforcement
detailing. Per-face values (face 0 / face 1, i.e. the two sides of the
wall along its centre-line normal) can be left blank to fall back to the
sidebar defaults, which is how you'd typically detail a core in practice.

Features
--------
1. Material property inputs in the sidebar (concrete, steel, factors).
2. Per-wall detailing defaults in the sidebar, used when table cells are blank.
3. Global "verticals as outer layer" toggle (passes through to verts_outer_layer).
4. Editable table of wall lines: x1, y1, x2, y2, thk and the per-face
   reinforcement inputs CoreSection accepts (v_dia_0/1, v_s_0/1, h_dia_0/1,
   h_s_0/1, cover_0/1).
5. Live geometry preview that draws the wall centre-lines and their
   projected thicknesses, plus a small face-0 marker so the user can see
   which side is which.
6. "Run Analysis" builds CoreSection and produces:
   - core.v_section.plot(incl_uls=True, incl_stiffness=True, incl_dims=False)
   - core.v_section.plot_FM_graph()
7. "Export to DOCX" writes a CalcDoc report with both plots and a
   tabulated section property summary.

Usage:
    streamlit run Core_SL.py
"""

from __future__ import annotations

import io
import traceback
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Module imports. Fail loud and early.
# ---------------------------------------------------------------------------
try:
    from CoreDesigner import CoreSection
    from BugBasics import Line
except ImportError as e:
    st.set_page_config(page_title="Core Section", layout="wide")
    st.error(f"Could not import CoreDesigner / BugBasics: {e}")
    st.info(
        "Place CoreDesigner.py and its dependencies (BugBasics, BugPoly, "
        "RC_beam_mesh, Mesh, Notation) on the Python path next to this script."
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
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="RC Core Section Analysis",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Wall table schema
# ---------------------------------------------------------------------------
# Each wall row is fully defined by these columns. Per-face inputs may be
# left blank, in which case the sidebar defaults are substituted at build
# time. Geometry columns (x1,y1,x2,y2,thk) must be present.
WALL_COLS = [
    "x1 (mm)", "y1 (mm)", "x2 (mm)", "y2 (mm)",
    "thk (mm)",
    "v_dia_0 (mm)", "v_dia_1 (mm)",
    "v_s_0 (mm)",   "v_s_1 (mm)",
    "h_dia_0 (mm)", "h_dia_1 (mm)",
    "h_s_0 (mm)",   "h_s_1 (mm)",
    "cover_0 (mm)", "cover_1 (mm)",
]

GEOM_COLS = ["x1 (mm)", "y1 (mm)", "x2 (mm)", "y2 (mm)", "thk (mm)"]


def _wall_row_from_defaults(x1, y1, x2, y2, thk, defaults) -> list:
    """Build a single fully-populated wall row from the geometry plus the
    sidebar detailing defaults. No NaNs in the resulting row."""
    return [
        x1, y1, x2, y2, thk,
        defaults["v_dia"], defaults["v_dia"],
        defaults["v_s"],   defaults["v_s"],
        defaults["h_dia"], defaults["h_dia"],
        defaults["h_s"],   defaults["h_s"],
        defaults["cover"], defaults["cover"],
    ]


def _default_walls(defaults: dict) -> np.ndarray:
    """A simple 3000 x 3000 C-shape: three walls forming an open channel.

    Wall 1: left flange, vertical
    Wall 2: web, horizontal at the bottom
    Wall 3: right flange, vertical
    All walls are 300 mm thick. Every cell is populated with the supplied
    detailing defaults. Returns an (N_walls, 15) float ndarray.

    State is stored as ndarray rather than DataFrame on purpose: feeding a
    DataFrame into st.data_editor and writing the editor's return value
    back into the same session_state slot creates a reactive identity
    loop that re-renders the widget on every rerun. The fix (mirroring
    RC_beam_SL.py) is to store an ndarray, build a fresh DataFrame each
    rerun, and convert the editor's return back to ndarray via .to_numpy
    before stashing -- that type round-trip breaks the loop.
    """
    geom_rows = [
        # x1,    y1,    x2,    y2,   thk
        (   0.0,    0.0,    0.0, 3000.0, 300.0),
        (   0.0,    0.0, 3000.0,    0.0, 300.0),
        (3000.0,    0.0, 3000.0, 3000.0, 300.0),
    ]
    data = [_wall_row_from_defaults(*r, defaults) for r in geom_rows]
    return np.array(data, dtype=float)


# Defaults used the very first time the page loads, before the sidebar
# widgets have rendered. The sidebar can later re-seed the table on demand.
_BOOT_DEFAULTS = dict(v_dia=16.0, v_s=200.0, h_dia=12.0, h_s=200.0, cover=30.0)


def _init_state():
    if "walls" not in st.session_state:
        st.session_state.walls = _default_walls(_BOOT_DEFAULTS)
    if "core" not in st.session_state:
        st.session_state.core = None
    if "preview_fig" not in st.session_state:
        st.session_state.preview_fig = None
    if "plot_fig" not in st.session_state:
        st.session_state.plot_fig = None
    if "fm_fig" not in st.session_state:
        st.session_state.fm_fig = None
    if "summary_rows" not in st.session_state:
        st.session_state.summary_rows = None


_init_state()


# ---------------------------------------------------------------------------
# Wall row -> CoreSection inputs
# ---------------------------------------------------------------------------
def _fill_with_default(val, default):
    """Treat NaN / None / blank as 'use the default'."""
    if val is None:
        return float(default)
    try:
        f = float(val)
        if not np.isfinite(f):
            return float(default)
        return f
    except (TypeError, ValueError):
        return float(default)


def df_to_core_inputs(df: pd.DataFrame, defaults: dict):
    """Convert the wall table to the per-wall arrays CoreSection expects.

    Returns a dict ready to splat into CoreSection(...).
    """
    df = df.dropna(subset=GEOM_COLS).reset_index(drop=True)
    if len(df) == 0:
        raise ValueError("No valid walls defined. Each wall needs x1, y1, x2, y2 and thk.")

    wall_lines = []
    wall_thk = []
    v_dias = []
    v_spacing = []
    h_dias = []
    h_spacing = []
    cover = []

    for i, row in df.iterrows():
        x1, y1, x2, y2 = float(row["x1 (mm)"]), float(row["y1 (mm)"]), float(row["x2 (mm)"]), float(row["y2 (mm)"])
        if np.hypot(x2 - x1, y2 - y1) < 1e-6:
            raise ValueError(f"Wall {i+1}: zero-length wall (endpoints coincide).")
        thk = float(row["thk (mm)"])
        if thk <= 0:
            raise ValueError(f"Wall {i+1}: thickness must be > 0.")

        wall_lines.append(Line([[x1, y1], [x2, y2]]))
        wall_thk.append(thk)

        v_dias.append([
            _fill_with_default(row["v_dia_0 (mm)"], defaults["v_dia"]),
            _fill_with_default(row["v_dia_1 (mm)"], defaults["v_dia"]),
        ])
        v_spacing.append([
            _fill_with_default(row["v_s_0 (mm)"], defaults["v_s"]),
            _fill_with_default(row["v_s_1 (mm)"], defaults["v_s"]),
        ])
        h_dias.append([
            _fill_with_default(row["h_dia_0 (mm)"], defaults["h_dia"]),
            _fill_with_default(row["h_dia_1 (mm)"], defaults["h_dia"]),
        ])
        h_spacing.append([
            _fill_with_default(row["h_s_0 (mm)"], defaults["h_s"]),
            _fill_with_default(row["h_s_1 (mm)"], defaults["h_s"]),
        ])
        cover.append([
            _fill_with_default(row["cover_0 (mm)"], defaults["cover"]),
            _fill_with_default(row["cover_1 (mm)"], defaults["cover"]),
        ])

    return dict(
        wall_lines=wall_lines,
        wall_thk=wall_thk,
        v_dias=np.array(v_dias, float),
        v_spacing=np.array(v_spacing, float),
        h_dias=np.array(h_dias, float),
        h_spacing=np.array(h_spacing, float),
        cover=np.array(cover, float),
    )


def build_core(walls: np.ndarray, mat: dict, defaults: dict) -> CoreSection:
    """Build a CoreSection from the wall ndarray. The ndarray has shape
    (N_walls, 15) and columns in WALL_COLS order."""
    df = pd.DataFrame(walls, columns=WALL_COLS)
    inputs = df_to_core_inputs(df, defaults)
    return CoreSection(
        f_ck=float(mat["f_ck"]),
        wall_lines=inputs["wall_lines"],
        wall_thk=inputs["wall_thk"],
        v_dias=inputs["v_dias"],
        v_spacing=inputs["v_spacing"],
        h_dias=inputs["h_dias"],
        h_spacing=inputs["h_spacing"],
        cover=inputs["cover"],
        c_class=mat["c_class"],
        t_ref=float(mat["t_ref"]),
        k_E=float(mat["k_E"]),
        f_y=float(mat["f_y"]),
        gamma_s=float(mat["gamma_s"]),
        gamma_c=float(mat["gamma_c"]),
        verts_outer_layer=bool(mat["verts_outer_layer"]),
    )


# ---------------------------------------------------------------------------
# (No custom preview function — we use core.v_section.plot() directly,
# which renders the actual merged wall geometry and rebar layout after
# CoreSection has run its polygon-intersection repair. Building the
# CoreSection is cheap; only M_Rd / I_u / M_cr etc. trigger FE work.)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Section property summary
# ---------------------------------------------------------------------------
def core_summary(core: CoreSection, mat: dict, t_long: float, creep: float):
    """List of (label, value) rows describing the core section. Mirrors the
    RC beam summary but uses core.v_section as the underlying RCSection.
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

    sec = core.v_section

    na = sec.elastic_na()
    x_c = sec.elastic_cracking_depth()
    I_u_st = sec.I_u()
    I_c_st = sec.I_c()
    M_cr_st = sec.M_cr()
    I_u_lt = sec.I_u(t=t_long, creep=creep)
    I_c_lt = sec.I_c(t=t_long, creep=creep)
    M_cr_lt = sec.M_cr(t=t_long, creep=creep)

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
        ("Partial factor \u03b3_{c}", f"{mat['gamma_c']:.2f}"),
        ("Partial factor \u03b3_{s}", f"{mat['gamma_s']:.2f}"),
        ("Verticals as outer layer", "Yes" if mat["verts_outer_layer"] else "No"),
        ("**Geometry**", ""),
        ("Number of walls (input)",
         f"{core.N_walls if hasattr(core, 'N_walls') else len(core.wall_lines)}"),
        ("Bounding box B \u00d7 H",
         f"{_safe_ef(sec.B)} \u00d7 {_safe_ef(sec.H)} mm"),
        ("Concrete area A_{c}", f"{_safe_ef(A_c)} mm^{{2}}"),
        ("Steel area A_{s}", f"{_safe_ef(A_s)} mm^{{2}}"),
        ("Reinforcement ratio \u03c1", f"{rho:.2f}%"),
        ("Effective depth d_{ef}", f"{_safe_ef(sec.d_ef)} mm"),
        ("**Elastic (uncracked) properties**", ""),
        ("Elastic NA y", f"{_safe_ef(na)} mm"),
        ("Uncracked NA depth x_{u}", f"{_safe_ef(sec.y_max - na)} mm"),
        ("Cracked NA depth x_{c}", f"{_safe_ef(x_c)} mm"),
        ("Short term I_{u}", f"{_safe_ef(I_u_st)} mm^{{4}}"),
        ("Short term I_{c}", f"{_safe_ef(I_c_st)} mm^{{4}}"),
        ("Short term M_{cr}", f"{_safe_ef(M_cr_st_kNm)} kNm"),
        ("Long term (t, \u03c6)",
         f"t = {_safe_ef(t_long / 365)} yrs, \u03c6 = {_safe_ef(creep)}"),
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
    core: CoreSection,
    walls_df: pd.DataFrame,
    mat: dict,
    defaults: dict,
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
        "This calculation reports the cross-section properties and ULS "
        "bending capacity of a reinforced concrete core. The core is "
        "defined as an assembly of straight walls, each carrying its own "
        "thickness, cover and per-face reinforcement (vertical and "
        "horizontal bars). CoreDesigner merges the walls into a single "
        "composite polygon, lays out the rebar grids on both faces of each "
        "wall, and exposes the merged geometry as a standard "
        "RCSection (core.v_section) for analysis. The analysis then covers, "
        "in order:"
    )
    cd.add_rich_paragraph(
        "1. Elastic (uncracked) transformed section properties of the "
        "merged core, short term and long term, using the effective "
        "modulus method to account for creep."
    )
    cd.add_rich_paragraph(
        "2. Cracked transformed section properties under pure bending, "
        "with the neutral axis depth and second moment of area computed "
        "from force and moment equilibrium of a transformed section."
    )
    cd.add_rich_paragraph(
        "3. Cracking moment M_{cr} based on the EC2 mean axial tensile "
        "strength f_{ctm}."
    )
    cd.add_rich_paragraph(
        "4. ULS bending capacity through a fibre-style integration of the "
        "parabola-rectangle concrete stress block and elastic-perfectly-"
        "plastic steel stress-strain law, sweeping the neutral axis "
        "position and producing the full force-moment (F-M) interaction "
        "diagram."
    )
    cd.add_rich_paragraph(
        "All numerical work is carried out on the merged section. The "
        "wall layout and detailing inputs are reported separately below "
        "for traceability."
    )

    # =======================================================================
    # 2. Wall geometry and detailing inputs
    # =======================================================================
    cd.add_heading("Inputs: Wall geometry and detailing", level=1)
    cd.add_rich_paragraph(
        "Each row of the table below defines one wall centre-line with its "
        "thickness and per-face detailing. Face 0 lies on the +n side of "
        "the wall direction (x_{1}, y_{1}) to (x_{2}, y_{2}); face 1 lies "
        "on the opposite side. A cell shown as \"default\" means the wall "
        "row was left blank for that field and the sidebar default was "
        "used."
    )
    cleaned = walls_df.dropna(subset=GEOM_COLS).reset_index(drop=True)
    table = cd.add_table(rows=len(cleaned) + 1, cols=len(WALL_COLS) + 1)
    header = ["Wall"] + WALL_COLS
    for j, h in enumerate(header):
        _rich_cell(table.rows[0].cells[j], h, bold=True)
    for i, row in cleaned.iterrows():
        _rich_cell(table.rows[i + 1].cells[0], f"W{i + 1}")
        for j, col in enumerate(WALL_COLS):
            v = row[col]
            if v is None or (isinstance(v, float) and not np.isfinite(v)):
                _rich_cell(table.rows[i + 1].cells[j + 1], "default")
            else:
                _rich_cell(table.rows[i + 1].cells[j + 1], f"{float(v):g}")

    cd.add_heading("Detailing defaults", level=2)
    cd.add_rich_paragraph(
        "These values are substituted wherever a wall row leaves a "
        "detailing cell blank."
    )
    drows = [
        ("Vertical bar diameter v_{dia}", f"{defaults['v_dia']} mm"),
        ("Vertical bar spacing v_{s}", f"{defaults['v_s']} mm"),
        ("Horizontal bar diameter h_{dia}", f"{defaults['h_dia']} mm"),
        ("Horizontal bar spacing h_{s}", f"{defaults['h_s']} mm"),
        ("Cover", f"{defaults['cover']} mm"),
        ("Verticals as outer layer",
         "Yes" if mat["verts_outer_layer"] else "No"),
    ]
    dtable = cd.add_table(rows=len(drows), cols=2)
    for i, (k, v) in enumerate(drows):
        _rich_cell(dtable.rows[i].cells[0], k)
        _rich_cell(dtable.rows[i].cells[1], v)

    # =======================================================================
    # 3. Section property summary (merged section)
    # =======================================================================
    cd.add_heading("Merged section property summary", level=1)
    cd.add_rich_paragraph(
        "Properties of the merged core section. Section headers (Material, "
        "Geometry, Elastic, ULS) are shown in bold."
    )
    rows = core_summary(core, mat, t_long, creep)
    table = cd.add_table(rows=len(rows), cols=2)
    for i, (k, v) in enumerate(rows):
        is_header = k.startswith("**") and v == ""
        _rich_cell(table.rows[i].cells[0], k, bold=is_header)
        _rich_cell(table.rows[i].cells[1], v, bold=is_header)

    # =======================================================================
    # 4. Methodology
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
        "with k_{E} entered as an input. This produces values consistent "
        "with the EC2 Table 3.1 expression E_{cm} = 22000 ((f_{ck} + 8) / "
        "10)^{0.3} (MPa) for normal-weight concrete."
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
        "f_{ctm} is used for the cracking moment only and is never relied "
        "on for shear or anchorage."
    )
    cd.add_rich_paragraph(
        "Steel is linear elastic up to f_{yd} = f_{yk} / \u03b3_{s}, then "
        "perfectly plastic, with E_{s} = 200 GPa per EC2 \u00a73.2.7(4). "
        "The horizontal yield plateau (Figure 3.8, idealised) is used "
        "because it is conservative for ductility class B and C bars and "
        "is the standard simplification for section analysis."
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

    cd.add_heading("Core Geometry Assembly", level=2)
    cd.add_rich_paragraph(
        "Each wall is defined by its centre-line and a uniform thickness. "
        "CoreDesigner generates the wall polygon as a constant-offset "
        "buffer around the centre-line, then merges all wall polygons "
        "into a single composite concrete region by polygon union. The "
        "merged outline is what the section analysis sees; junctions "
        "between walls become continuous regions of concrete with no "
        "double-counting at overlaps."
    )
    cd.add_rich_paragraph(
        "Rebar is laid out independently on each face of each wall, "
        "running along the wall centre-line. The vertical and horizontal "
        "bar grids honour the entered diameters, spacings and covers. "
        "The verts_outer_layer flag controls layer order at the face: "
        "if true, vertical bars sit at depth cover + v_{dia} / 2 from "
        "the face and horizontals sit one bar-diameter deeper; if false, "
        "the order is swapped. This matches typical site practice for "
        "either bar-first detail. Where two walls meet at an inside "
        "corner, the duplicated rebar near the junction is removed to "
        "avoid counting the same area twice."
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
        "The creep coefficient \u03c6 is entered directly and is intended "
        "to represent the value at the loading age and duration of "
        "interest, e.g. \u03c6(\u221e, t_{0}) for permanent loads, "
        "derived from EC2 Annex B or Figure 3.1 if needed. The same "
        "E_{c,eff} is applied uniformly to all concrete fibres for the "
        "long-term properties; no separate shrinkage contribution is "
        "included."
    )

    cd.add_heading("Elastic (Uncracked) Section Properties", level=2)
    cd.add_rich_paragraph(
        "The uncracked transformed section is built by replacing each "
        "steel bar of area A_{si} with an equivalent extra concrete area "
        "(n \u2212 1) A_{si} at the bar centroid, where n is the modular "
        "ratio:"
    )
    _eq_safe(
        cd,
        r"n = \frac{E_s}{E_c}, \quad E_c = E_{\mathrm{cm}}\;(\mathrm{short\;term}),\; E_{\mathrm{c,eff}}\;(\mathrm{long\;term})",
        "n = E_{s} / E_{c}  (E_{c} = E_{cm} short term, E_{c,eff} long term)",
    )
    cd.add_rich_paragraph(
        "The elastic neutral axis y_{NA} is the centroid of the "
        "transformed area:"
    )
    _eq_safe(
        cd,
        r"y_{\mathrm{NA}} = \frac{\int_{A_t} y \, dA_t}{\int_{A_t} dA_t}",
        "y_{NA} = (\u222b y dA_{t}) / (\u222b dA_{t})",
    )
    cd.add_rich_paragraph(
        "and I_{u} is the transformed second moment about that NA. For "
        "the merged core polygon (which is generally non-rectangular) "
        "the integrals are evaluated by numerical integration over the "
        "meshed concrete area plus a discrete sum over the rebar bars."
    )

    cd.add_heading("Cracked Section Properties", level=2)
    cd.add_rich_paragraph(
        "Under pure bending the tension concrete is assumed cracked and "
        "discounted, per EC2 \u00a77.1(2). The cracked NA depth x_{c} is "
        "found from force equilibrium of the transformed section, with "
        "the concrete in compression contributing a linear stress "
        "distribution (elastic, since this is a serviceability state) "
        "and the steel transformed by the modular ratio n. For a core "
        "with steel distributed along multiple walls the equilibrium is "
        "solved numerically rather than via the textbook quadratic that "
        "applies to a singly-reinforced rectangle. The cracked second "
        "moment is then taken about that NA."
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
        "threshold between the uncracked and cracked stiffness branches "
        "in deflection calculations to EC2 \u00a77.4.3."
    )

    cd.add_heading("ULS Bending Capacity", level=2)
    cd.add_rich_paragraph(
        "The ULS capacity is computed by sweeping the neutral axis "
        "position and integrating the concrete and steel stresses over "
        "the merged core section. Concrete in compression follows the "
        "parabola-rectangle stress-strain law of EC2 \u00a73.1.7(1), "
        "Figure 3.3, with peak stress \u03b7 f_{cd} and limit strain "
        "\u03b5_{cu2} = 3.5 \u2030 for f_{ck} \u2264 50 MPa (modified per "
        "EC2 Table 3.1 for higher strengths). Concrete in tension is "
        "ignored. Steel follows the elastic-perfectly-plastic law "
        "described above."
    )
    cd.add_rich_paragraph(
        "The parabola-rectangle stress block parameters \u03bb and "
        "\u03b7 are:"
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
        "At each NA position the section is in equilibrium with some "
        "axial force F and bending moment M, both recorded and joined by "
        "a spline to form the F-M interaction diagram. M_{Rd} at a given "
        "axial force is read off the spline by inversion. With F = 0 "
        "this is the pure bending capacity reported in the summary. The "
        "F-M envelope captures the interaction useful for combined "
        "axial-bending check of the core lift, e.g. where wind moments "
        "act simultaneously with gravity axial load."
    )
    cd.add_rich_paragraph(
        "Bending is reported about the analysis axis used by "
        "CoreDesigner. To capture biaxial bending and the resistance "
        "about both principal axes of the core, the section can be "
        "rotated and re-run; only the single-axis F-M is included here."
    )

    # =======================================================================
    # 5. Compliance with Eurocodes
    # =======================================================================
    cd.add_heading("Compliance with Eurocodes", level=1)
    cd.add_rich_paragraph(
        "The methodology above implements the following clauses of "
        "BS EN 1992-1-1:2004 + A1:2014 (Eurocode 2, Part 1-1), with the "
        "UK National Annex values where listed:"
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
        ("Table 3.1", "\u03b5_{cu2} limit strain",
         "\u03b5_{cu2} = 3.5 \u2030 for f_{ck} \u2264 50 MPa, reduced per "
         "Table 3.1 for higher strengths."),
        ("\u00a73.2.7(4), Figure 3.8", "Steel stress-strain (ULS)",
         "Elastic-perfectly-plastic with E_{s} = 200 GPa and yield "
         "plateau at f_{yd}."),
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
        ("\u00a79.6", "Reinforced concrete walls, minimum reinforcement",
         "Not checked by this calculation. The user must verify minimum "
         "v_{s,min}, h_{s,min} and minimum reinforcement ratios "
         "separately against \u00a79.6.2 to \u00a79.6.4."),
    ]
    table = cd.add_table(rows=len(compliance_rows), cols=3)
    # Set explicit column widths so the long Implementation column is
    # readable. Total target width is 16 cm (matches the picture widths).
    table.autofit = False
    col_widths_cm = [3.0, 4.5, 8.5]
    from docx.shared import Cm as _Cm
    for col_idx, w_cm in enumerate(col_widths_cm):
        for row in table.rows:
            row.cells[col_idx].width = _Cm(w_cm)
    for i, row in enumerate(compliance_rows):
        is_header = (i == 0)
        for j, txt in enumerate(row):
            _rich_cell(table.rows[i].cells[j], txt, bold=is_header)

    cd.add_rich_paragraph(
        "Items not included in this calculation, and which must be "
        "checked separately if relevant: shear (EC2 \u00a76.2), torsion "
        "(\u00a76.3), serviceability deflections (\u00a77.4), crack "
        "widths (\u00a77.3), wall slenderness and second-order effects "
        "(\u00a75.8), buckling of the compression edge, minimum "
        "reinforcement and detailing (\u00a79.6), in-plane shear and "
        "boundary element checks where the core acts as a shear wall "
        "system, and any biaxial bending interaction beyond the single "
        "F-M plane reported here."
    )

    # =======================================================================
    # 6. Section and FM plots
    # =======================================================================
    cd.add_heading("Plots", level=1)

    cd.add_heading("Section Plot", level=2)
    cd.add_rich_paragraph(
        "Merged core polygon, rebar layout, elastic neutral axis, ULS "
        "strain diagram and stiffness annotations as enabled in the "
        "analysis options. Coordinates are in mm."
    )
    buf_plot = io.BytesIO()
    plot_fig.savefig(buf_plot, format="png", dpi=180, bbox_inches="tight",
                     facecolor="white")
    buf_plot.seek(0)
    cd.add_picture(buf_plot, width=Cm(16))

    cd.add_heading("Force-Moment Interaction (F-M Diagram)", level=2)
    cd.add_rich_paragraph(
        "Full ULS F-M envelope of the core section. M_{Rd} at any axial "
        "force is obtained by intersecting a horizontal line at that F "
        "with the envelope. The pure bending capacity reported in the "
        "summary corresponds to F = 0."
    )
    buf_fm = io.BytesIO()
    fm_fig.savefig(buf_fm, format="png", dpi=180, bbox_inches="tight",
                   facecolor="white")
    buf_fm.seek(0)
    cd.add_picture(buf_fm, width=Cm(16))

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
        "Design to Eurocode 2 (7th ed.). Palgrave Macmillan. General "
        "reference for the rectangular and parabola-rectangle stress "
        "blocks and the transformed section approach."
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
# Sidebar - Material properties, detailing defaults, analysis settings
# ===========================================================================
with st.sidebar:
    st.header("Material Properties")

    f_ck = st.number_input(
        "Concrete f_ck_28 [MPa]", min_value=12.0, max_value=120.0,
        value=40.0, step=1.0,
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

    st.divider()
    st.header("Detailing")
    verts_outer_layer = st.checkbox(
        "Verticals as outer layer", value=False,
        help=(
            "If checked, vertical bars sit on the outside (cover + dia/2) and "
            "horizontals sit inboard. If unchecked, horizontals are on the "
            "outside and verticals sit one bar-diameter deeper. "
            "Maps to CoreSection(verts_outer_layer=...)."
        ),
    )

    st.markdown("**Per-face defaults** (used where wall row is blank)")
    col1, col2 = st.columns(2)
    with col1:
        v_dia_def = st.number_input("v_dia [mm]", min_value=6.0, max_value=50.0,
                                    value=16.0, step=1.0)
        h_dia_def = st.number_input("h_dia [mm]", min_value=6.0, max_value=50.0,
                                    value=12.0, step=1.0)
    with col2:
        v_s_def = st.number_input("v_s [mm]", min_value=50.0, max_value=600.0,
                                  value=200.0, step=10.0)
        h_s_def = st.number_input("h_s [mm]", min_value=50.0, max_value=600.0,
                                  value=200.0, step=10.0)
    cover_def = st.number_input("cover [mm]", min_value=10.0, max_value=100.0,
                                value=30.0, step=5.0)

    mat = dict(
        f_ck=f_ck, c_class=c_class, t_ref=t_ref, k_E=k_E, f_y=f_y,
        gamma_c=gamma_c, gamma_s=gamma_s,
        verts_outer_layer=verts_outer_layer,
    )
    defaults = dict(
        v_dia=v_dia_def, v_s=v_s_def,
        h_dia=h_dia_def, h_s=h_s_def,
        cover=cover_def,
    )

    st.divider()
    st.header("DOCX Export Settings")
    project_name = st.text_input("Project name", value="Project")
    project_number = st.text_input("Project number", value="0000")
    design_element = st.text_input("Design element", value="RC Core")
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
st.title("Reinforced Concrete Core Section Analysis")
st.caption(
    "Define the core's wall centre-lines in the table below. Every cell "
    "is pre-populated from the sidebar detailing defaults; edit any cell "
    "to override per wall or per face. Face 0 is on the +n (left-hand) "
    "side of the wall direction (x1,y1) to (x2,y2), marked red in the "
    "preview. Edits are not committed live: click **Refresh preview** to "
    "redraw, or **Run Analysis** to solve."
)

col_inputs, col_preview = st.columns([1.4, 1.0])

with col_inputs:
    st.markdown("**Wall centre-lines and per-wall detailing**")

    # The data_editor lives inside an st.form so cell edits are batched
    # locally and do not trigger a rerun until the submit button is clicked.
    # Without this, every cell edit fires a full page rerun and the user
    # loses focus mid-typing (Streamlit docs: st.form; discuss.streamlit.io
    # threads 42886, 52793, 91924). Edits stay in the editor's local state
    # until "Apply wall edits" is pressed.
    with st.form("walls_form", clear_on_submit=False):
        walls_df = pd.DataFrame(st.session_state.walls, columns=WALL_COLS)
        edited = st.data_editor(
            walls_df,
            num_rows="dynamic",
            use_container_width=True,
            key="walls_editor",
            height=380,
            column_config={
                c: st.column_config.NumberColumn(format="%.1f")
                for c in WALL_COLS
            },
        )
        walls_submitted = st.form_submit_button(
            "Apply wall edits", type="primary",
            use_container_width=True,
        )

    if walls_submitted:
        # Commit: convert back to ndarray. The type round-trip
        # (ndarray -> DataFrame -> editor -> DataFrame -> ndarray) is what
        # breaks the reactive identity loop.
        try:
            cleaned = edited.dropna(how="all")
            if len(cleaned) > 0:
                st.session_state.walls = cleaned.to_numpy(dtype=float)
                st.success(f"\u2713 {len(cleaned)} walls applied.")
            else:
                st.session_state.walls = np.empty((0, len(WALL_COLS)), float)
                st.warning("No walls defined.")
        except Exception as ex:
            st.warning(f"Wall parse issue: {ex}")

    # Geometry mutation buttons live OUTSIDE the form (st.button is not
    # allowed inside a form). They overwrite st.session_state.walls
    # directly and rerun, which redraws the editor from the new state.
    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("Reset C-shape", use_container_width=True,
                     key="btn_reset_cshape",
                     help="Re-seed the table with the default 3 m C-shape "
                          "using current sidebar detailing defaults."):
            st.session_state.walls = _default_walls(defaults)
            st.rerun()
    with c2:
        if st.button("Add blank row", use_container_width=True,
                     key="btn_add_row",
                     help="Append a new wall row, pre-filled with current "
                          "sidebar detailing defaults and zero geometry."):
            new_row = np.array(
                [_wall_row_from_defaults(0.0, 0.0, 0.0, 0.0, 200.0, defaults)],
                dtype=float,
            )
            st.session_state.walls = np.vstack([st.session_state.walls, new_row])
            st.rerun()
    with c3:
        if st.button("Clear all walls", use_container_width=True,
                     key="btn_clear_walls"):
            st.session_state.walls = np.empty((0, len(WALL_COLS)), float)
            st.rerun()

with col_preview:
    st.markdown("**Section Preview**")
    if st.button("\u21bb Refresh preview", use_container_width=True,
                 help="Build the CoreSection and render its geometry "
                      "(merged walls + rebar). No analysis is run."):
        try:
            if st.session_state.preview_fig is not None:
                plt.close(st.session_state.preview_fig)
            core = build_core(st.session_state.walls, mat, defaults)
            # Geometry-only plot: no ULS, no stiffness, no dims. This avoids
            # any FE work and matches the behaviour of the existing tools.
            st.session_state.preview_fig = core.v_section.plot(
                show=False,
                incl_uls=False,
                incl_stiffness=False,
                incl_dims=False,
            )
        except Exception as ex:
            st.session_state.preview_fig = None
            st.error(f"Preview error: {ex}")

    if st.session_state.preview_fig is not None:
        st.pyplot(st.session_state.preview_fig, use_container_width=True)
    else:
        st.caption("Click **Refresh preview** to draw the current section.")

st.divider()

# ===========================================================================
# Run + export buttons
# ===========================================================================
run_col, export_col, _ = st.columns([1, 1, 2])
with run_col:
    run_btn = st.button("Run Analysis", type="primary", use_container_width=True)
with export_col:
    export_btn = st.button("Export to DOCX", use_container_width=True,
                           disabled=(st.session_state.core is None))

if run_btn:
    with st.spinner("Building core section and solving..."):
        try:
            core = build_core(st.session_state.walls, mat, defaults)

            plot_fig = core.v_section.plot(
                show=False,
                incl_uls=plot_incl_uls,
                incl_stiffness=plot_incl_stiffness,
                incl_dims=plot_incl_dims,
                creep=creep,
                t=t_long,
            )

            fm_fig, fm_ax = plt.subplots(figsize=(10, 7))
            core.v_section.plot_FM_graph(ax=fm_ax, show=False)
            fm_fig.tight_layout()

            st.session_state.core = core
            st.session_state.plot_fig = plot_fig
            st.session_state.fm_fig = fm_fig
            st.session_state.summary_rows = core_summary(core, mat, t_long, creep)
            st.success("Analysis complete.")
        except Exception as ex:
            st.session_state.core = None
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
    if st.session_state.core is None:
        st.error("Run analysis first.")
    else:
        try:
            with st.spinner("Building DOCX..."):
                docx_bytes = build_calcdoc_bytes(
                    core=st.session_state.core,
                    walls_df=pd.DataFrame(st.session_state.walls, columns=WALL_COLS),
                    mat=mat,
                    defaults=defaults,
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