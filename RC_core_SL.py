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
        ("Material", ""),
        ("Concrete strength f_ck_28", f"{_safe_ef(mat['f_ck'])} MPa"),
        ("Cement class", mat["c_class"]),
        ("Reference age t_ref", f"{_safe_ef(mat['t_ref'])} days"),
        ("Modulus coefficient k_E", f"{_safe_ef(mat['k_E'])}"),
        ("Steel yield f_y", f"{_safe_ef(mat['f_y'])} MPa"),
        ("Partial factor gamma_c", f"{mat['gamma_c']:.2f}"),
        ("Partial factor gamma_s", f"{mat['gamma_s']:.2f}"),
        ("Verticals as outer layer", "Yes" if mat["verts_outer_layer"] else "No"),
        ("Geometry", ""),
        ("Number of walls (input)", f"{core.N_walls if hasattr(core, 'N_walls') else len(core.wall_lines)}"),
        ("Bounding box B x H", f"{_safe_ef(sec.B)} x {_safe_ef(sec.H)} mm"),
        ("Concrete area A_c", f"{_safe_ef(A_c)} mm^2"),
        ("Steel area A_s", f"{_safe_ef(A_s)} mm^2"),
        ("Reinforcement ratio rho", f"{rho:.2f}%"),
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

    cd.add_heading("Core Section Properties", level=1)

    # --- Wall table ---
    cd.add_heading("Wall geometry and detailing", level=2)
    cleaned = walls_df.dropna(subset=GEOM_COLS).reset_index(drop=True)
    table = cd.add_table(rows=len(cleaned) + 1, cols=len(WALL_COLS) + 1)
    header = ["Wall"] + WALL_COLS
    for j, h in enumerate(header):
        cell = table.rows[0].cells[j]
        cell.text = h
        for para in cell.paragraphs:
            for run in para.runs:
                run.bold = True
    for i, row in cleaned.iterrows():
        cells = table.rows[i + 1].cells
        cells[0].text = f"W{i + 1}"
        for j, col in enumerate(WALL_COLS):
            v = row[col]
            if v is None or (isinstance(v, float) and not np.isfinite(v)):
                cells[j + 1].text = "default"
            else:
                cells[j + 1].text = f"{float(v):g}"

    # --- Per-face defaults ---
    cd.add_heading("Detailing defaults (used when re-seeding the wall table)", level=2)
    drows = [
        ("Vertical bar diameter v_dia", f"{defaults['v_dia']} mm"),
        ("Vertical bar spacing v_s", f"{defaults['v_s']} mm"),
        ("Horizontal bar diameter h_dia", f"{defaults['h_dia']} mm"),
        ("Horizontal bar spacing h_s", f"{defaults['h_s']} mm"),
        ("Cover", f"{defaults['cover']} mm"),
    ]
    dtable = cd.add_table(rows=len(drows), cols=2)
    for i, (k, v) in enumerate(drows):
        dtable.rows[i].cells[0].text = k
        dtable.rows[i].cells[1].text = str(v)

    # --- Property summary ---
    cd.add_heading("Section property summary", level=2)
    rows = core_summary(core, mat, t_long, creep)
    table = cd.add_table(rows=len(rows), cols=2)
    for i, (k, v) in enumerate(rows):
        cells = table.rows[i].cells
        cells[0].text = k
        cells[1].text = str(v)
        if v == "":
            for cell in cells:
                for para in cell.paragraphs:
                    for run in para.runs:
                        run.bold = True

    # --- Plots ---
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

    # Build a FRESH DataFrame each rerun from the ndarray in state. This
    # is the same pattern the working RC_beam_SL.py uses. Do NOT pass
    # st.session_state.walls directly into data_editor and write back to
    # the same slot: that creates a reactive identity loop.
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

    # Commit: convert back to ndarray. The type round-trip
    # (ndarray -> DataFrame -> editor -> DataFrame -> ndarray) is what
    # breaks the reactive loop.
    try:
        cleaned = edited.dropna(how="all")
        if len(cleaned) > 0:
            st.session_state.walls = cleaned.to_numpy(dtype=float)
        else:
            st.session_state.walls = np.empty((0, len(WALL_COLS)), float)
    except Exception:
        # Don't blow up on a transient mid-edit state; keep the previous
        # ndarray and let the user finish typing.
        pass

    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("Reset C-shape", use_container_width=True,
                     help="Re-seed the table with the default 3 m C-shape "
                          "using current sidebar detailing defaults."):
            st.session_state.walls = _default_walls(defaults)
            st.rerun()
    with c2:
        if st.button("Add blank row", use_container_width=True,
                     help="Append a new wall row, pre-filled with current "
                          "sidebar detailing defaults and zero geometry."):
            new_row = np.array(
                [_wall_row_from_defaults(0.0, 0.0, 0.0, 0.0, 200.0, defaults)],
                dtype=float,
            )
            st.session_state.walls = np.vstack([st.session_state.walls, new_row])
            st.rerun()
    with c3:
        if st.button("Clear all walls", use_container_width=True):
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
            df = pd.DataFrame(
                st.session_state.summary_rows, columns=["Property", "Value"]
            )
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