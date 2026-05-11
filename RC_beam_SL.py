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
        ("Material", ""),
        ("Concrete strength f_ck_28", f"{_safe_ef(mat['f_ck'])} MPa"),
        ("Cement class", mat["c_class"]),
        ("Reference age t_ref", f"{_safe_ef(mat['t_ref'])} days"),
        ("Modulus coefficient k_E", f"{_safe_ef(mat['k_E'])}"),
        ("Steel yield f_y", f"{_safe_ef(mat['f_y'])} MPa"),
        ("Partial factor gamma_c", f"{mat['gamma_c']:.2f}"),
        ("Partial factor gamma_s", f"{mat['gamma_s']:.2f}"),
        ("Geometry", ""),
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
                incl_uls=True,
                incl_stiffness=True,
                incl_dims=False,
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
