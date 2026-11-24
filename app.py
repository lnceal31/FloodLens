"""FloodLens - AI-powered flood intelligence platform (demonstration build)."""

from __future__ import annotations

import html
import io
import os
import zlib

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pydeck as pdk
import streamlit as st
from PIL import Image

from data import (
    AREAS,
    DUBAI_CENTER,
    SEVERITIES,
    STATUSES,
    dubai_now,
    make_demo_reports,
    make_history,
    priority_table,
)
from vision import analyze_image

st.set_page_config(page_title="FloodLens", layout="wide", initial_sidebar_state="expanded")

# API key for the AI engine: environment variable, or .streamlit/secrets.toml
if not os.environ.get("ANTHROPIC_API_KEY"):
    try:
        if "ANTHROPIC_API_KEY" in st.secrets:
            os.environ["ANTHROPIC_API_KEY"] = str(st.secrets["ANTHROPIC_API_KEY"])
    except Exception:
        pass  # no secrets file: the app uses the local prototype analyser

# ----------------------------------------------------------------------------
# Palette
# ----------------------------------------------------------------------------
BG, NAVY, DEEP, BLUE = "#F5F1E8", "#19324A", "#173D57", "#397DA8"
LIGHT, CYAN, CARD, BORDER = "#EAF0F2", "#20C5D8", "#FFFEFB", "#DCE4E8"
ORANGE, RED, MUTED = "#F4A340", "#F15B5B", "#3F5568"
SEV_COLOR = {"High": RED, "Medium": ORANGE, "Low": BLUE}
SEV_RGB = {"High": [241, 91, 91], "Medium": [244, 163, 64], "Low": [57, 125, 168]}
STATUS_COLOR = {"Active": NAVY, "Monitoring": BLUE, "Resolved": "#9ED9E0"}
PAGES = [
    "Overview",
    "Live Flood Map",
    "Reports & Analysis",
    "AI Image Analysis",
    "Historical Insights",
    "Submit a Report",
]

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Manrope:wght@600;700;800&display=swap');
.stApp{background:#F5F1E8;font-family:'Inter',sans-serif;color:#19324A;}
.stApp input,.stApp textarea,.stApp button,.stApp [data-baseweb="select"],.stApp [data-testid="stMarkdownContainer"] *{font-family:'Inter',sans-serif;}
#MainMenu,footer,.stDeployButton{visibility:hidden;}
[data-testid="stHeader"]{background:transparent;}
.block-container{padding:2rem 2.5rem 3rem;max-width:1440px;}
h1,h2,h3,h4,.fl-title,.card-title,.fl-metric-val{font-family:'Manrope',sans-serif!important;color:#19324A;letter-spacing:-0.01em;}

/* Sidebar */
[data-testid="stSidebar"]{background:#19324A;border-right:none;}
[data-testid="stSidebar"] *{color:#E8EEF2;}
[data-testid="stSidebar"] [role="radiogroup"]{gap:4px;}
[data-testid="stSidebar"] [role="radiogroup"] label{padding:10px 14px;border-radius:10px;width:100%;cursor:pointer;transition:background .15s;}
[data-testid="stSidebar"] [role="radiogroup"] label:hover{background:rgba(255,255,255,.07);}
[data-testid="stSidebar"] [role="radiogroup"] label > div:first-child{display:none;}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked){background:rgba(32,197,216,.16);box-shadow:inset 3px 0 0 #20C5D8;}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) p{color:#fff;font-weight:600;}
[data-testid="stSidebar"] [role="radiogroup"] p{font-size:.93rem;}
.fl-logo{display:flex;align-items:center;gap:12px;padding:.3rem 0 .2rem;}
.fl-logo-name{font-family:'Manrope',sans-serif;font-weight:800;font-size:1.35rem;color:#fff;letter-spacing:-0.02em;}
.fl-logo-tag{font-size:.78rem;color:#D3E0EA;margin:.1rem 0 1.4rem;}
.fl-nav-label{font-size:.68rem;letter-spacing:.12em;text-transform:uppercase;color:#C3D3DF;margin:.4rem 0 .5rem .2rem;}
.fl-side-note{margin-top:2rem;padding:.9rem 1rem;border:1px solid rgba(255,255,255,.12);border-radius:12px;font-size:.78rem;line-height:1.5;color:#E1EAF1;}

/* Page header */
.fl-head{display:flex;justify-content:space-between;align-items:flex-end;margin-bottom:1.4rem;gap:1rem;flex-wrap:wrap;}
.fl-eyebrow{font-size:.72rem;letter-spacing:.12em;text-transform:uppercase;color:#1F5F88;font-weight:600;}
.fl-title{font-size:1.95rem;font-weight:800;margin:.1rem 0;}
.fl-sub{color:#3F5568;font-size:.95rem;margin:0;}
.pill-demo{border:1px solid #DCE4E8;background:#FFFEFB;color:#3F5568;border-radius:999px;padding:5px 13px;font-size:.74rem;font-weight:500;}

/* Cards */
[class*="st-key-card"]{background:#FFFEFB;border:1px solid #DCE4E8!important;border-radius:16px;padding:1.1rem 1.25rem;box-shadow:0 1px 2px rgba(25,50,74,.04);}
.card-title{font-size:1.05rem;font-weight:700;margin:0;}
.card-sub{font-size:.86rem;color:#3F5568;margin:.15rem 0 .8rem;}
.fl-metric{background:#FFFEFB;border:1px solid #DCE4E8;border-radius:16px;padding:1.1rem 1.25rem;box-shadow:0 1px 2px rgba(25,50,74,.04);}
.fl-metric-top{display:flex;align-items:center;gap:8px;font-size:.82rem;color:#3F5568;font-weight:500;}
.dot{width:9px;height:9px;border-radius:50%;display:inline-block;}
.fl-metric-val{font-size:2.2rem;font-weight:800;line-height:1.15;margin:.35rem 0 .1rem;}
.fl-metric-desc{font-size:.82rem;color:#3F5568;}

/* Pills */
.pill{display:inline-block;padding:2px 10px;border-radius:999px;font-size:.7rem;font-weight:600;white-space:nowrap;}
.pill-high{background:#FDE6E6;color:#B33A3A;}
.pill-medium{background:#FDEFD9;color:#9A5C0C;}
.pill-low{background:#E1EEF6;color:#26688F;}
.pill-st-active{background:#19324A;color:#fff;}
.pill-st-monitoring{background:#E1EEF6;color:#26688F;}
.pill-st-resolved{background:#E0F5F7;color:#0F7F8C;}

/* Alerts & detail */
.alert{border:1px solid #DCE4E8;border-left:4px solid #F15B5B;border-radius:12px;padding:.75rem .9rem;margin-bottom:.6rem;background:#FFFEFB;}
.alert-top{display:flex;justify-content:space-between;align-items:center;gap:8px;font-weight:600;font-size:.92rem;}
.alert-meta{font-size:.74rem;color:#3F5568;margin:.25rem 0;display:flex;gap:8px;align-items:center;flex-wrap:wrap;}
.alert-desc{font-size:.82rem;line-height:1.45;color:#2E4A61;}
.detail-grid{display:grid;grid-template-columns:1fr 1fr;gap:.7rem 1rem;margin:.6rem 0;}
.detail-k{font-size:.72rem;text-transform:uppercase;letter-spacing:.08em;color:#4F657A;}
.detail-v{font-size:.9rem;font-weight:500;}
.legend{display:flex;gap:16px;font-size:.82rem;color:#3F5568;margin:.4rem 0 .2rem;flex-wrap:wrap;}
.legend span{display:inline-flex;align-items:center;gap:6px;}
.chip-row{display:flex;gap:8px;flex-wrap:wrap;margin:.7rem 0 .2rem;}
.chip{background:#EAF0F2;color:#173D57;border-radius:8px;padding:4px 10px;font-size:.74rem;font-weight:500;}
.note{background:#EAF0F2;border-radius:12px;padding:.8rem 1rem;font-size:.8rem;line-height:1.5;color:#2E4A61;}
.conf-track{height:8px;border-radius:99px;background:#EAF0F2;overflow:hidden;margin-top:6px;}
.conf-fill{height:100%;border-radius:99px;background:#20C5D8;}
.big-class{font-family:'Manrope',sans-serif;font-weight:800;font-size:1.5rem;margin:.2rem 0;}

/* Inputs & buttons */
.stApp [data-baseweb="select"]>div,.stApp .stTextInput input,.stApp .stNumberInput input,.stApp .stTextArea textarea{background:#FFFFFF;border-color:#DCE4E8;border-radius:10px;}
button[kind="primary"]{background:#19324A;border:none;color:#fff;border-radius:10px;font-weight:600;}
button[kind="primary"]:hover{background:#173D57;color:#fff;}
button[kind="secondary"]{border-radius:10px;border-color:#DCE4E8;}
[data-testid="stFileUploaderDropzone"]{background:#F8FAFA;border:1.5px dashed #BFCFD7;border-radius:14px;}

/* Contrast fixes for Streamlit's own text */
[data-testid="stMain"] [data-testid="stWidgetLabel"] p,[data-testid="stMain"] label p{color:#19324A!important;font-weight:500;}
[data-testid="stCaptionContainer"],[data-testid="stCaptionContainer"] *,.stApp small{color:#3F5568!important;}
.stApp input::placeholder,.stApp textarea::placeholder{color:#4F657A!important;opacity:1;}
.stApp input,.stApp textarea{color:#19324A!important;}
[data-testid="stFileUploaderDropzone"] *{color:#2E4A61;}
[data-testid="stFileUploaderDropzone"] button{color:#19324A;border-color:#8FA5B5;}
[data-baseweb="tag"]{background:#19324A!important;}
[data-baseweb="tag"] *{color:#FFFFFF!important;}
[data-testid="stDataFrame"]{color:#19324A;}

/* File upload areas */
[data-testid="stFileUploaderDropzone"] {
    background-color: #FFFFFF !important;
    border: 1.5px dashed #397DA8 !important;
    border-radius: 12px !important;
}
[data-testid="stFileUploaderDropzone"] * {
    color: #19324A !important;
}
[data-testid="stFileUploaderDropzone"] button {
    background-color: #EAF0F2 !important;
    color: #19324A !important;
    border: 1px solid #397DA8 !important;
    font-weight: 600 !important;
}
[data-testid="stFileUploaderDropzone"] button:hover {
    background-color: #DCEAF1 !important;
    color: #173D57 !important;
}
[data-testid="stFileUploaderDropzoneInstructions"],
[data-testid="stFileUploaderDropzoneInstructions"] span,
[data-testid="stFileUploaderDropzoneInstructions"] small {
    color: #19324A !important;
}

/* ================================================================
   HIGH-CONTRAST FIX FOR AREA SELECTBOX & PORTALLED POPOVER MENUS
   ================================================================ */
.st-key-sb_loc, .st-key-sb_cond, .st-key-sb_map,
.st-key-sub_area, .st-key-sub_landmark, .st-key-sub_use_location {
    color: #19324A !important;
}

.st-key-sb_loc [data-testid="stWidgetLabel"] *,
.st-key-sb_cond [data-testid="stWidgetLabel"] *,
.st-key-sb_map [data-testid="stWidgetLabel"] *,
.st-key-sb_loc [data-testid="stMarkdownContainer"] *,
.st-key-sb_cond [data-testid="stMarkdownContainer"] *,
.st-key-sb_map [data-testid="stMarkdownContainer"] *,
.st-key-sb_loc [data-testid="stCaptionContainer"] *,
.st-key-sb_cond [data-testid="stCaptionContainer"] *,
.st-key-sb_map [data-testid="stCaptionContainer"] *,
.st-key-sub_area label *,
.st-key-sub_landmark label * {
    color: #19324A !important;
    -webkit-text-fill-color: #19324A !important;
}

/* Force white field background on Area selectbox and all inner wrappers */
.st-key-sub_area [data-testid="stSelectbox"] > div,
.st-key-sub_area [data-baseweb="select"],
.st-key-sub_area [data-baseweb="select"] > div,
.st-key-sub_area [data-baseweb="select"] > div > div,
.st-key-sub_area [data-baseweb="select"] [class*="control"],
.st-key-sub_area [data-baseweb="select"] [class*="valueContainer"],
.st-key-sub_area [data-baseweb="select"] [class*="singleValue"],
.st-key-sub_area [data-baseweb="select"] [class*="inputContainer"],
.st-key-sb_loc [data-baseweb="select"] > div,
.st-key-sb_loc input,
.st-key-sb_loc textarea,
.st-key-sb_cond input,
.st-key-sb_cond textarea,
.st-key-rp_sort [data-baseweb="select"] > div,
.st-key-lm_area [data-baseweb="select"] > div,
.st-key-map_area [data-baseweb="select"] > div {
    background-color: #FFFFFF !important;
    background: #FFFFFF !important;
    color: #19324A !important;
    -webkit-text-fill-color: #19324A !important;
}

.st-key-sub_area [data-baseweb="select"] > div,
.st-key-sb_loc [data-baseweb="select"] > div,
.st-key-sb_loc input,
.st-key-sb_loc textarea,
.st-key-sb_cond input,
.st-key-sb_cond textarea,
.st-key-rp_sort [data-baseweb="select"] > div,
.st-key-lm_area [data-baseweb="select"] > div,
.st-key-map_area [data-baseweb="select"] > div {
    border: 1.5px solid #6E8A9E !important;
    box-shadow: 0 1px 2px rgba(25, 50, 74, 0.06) !important;
}

.st-key-sub_area [data-baseweb="select"]:focus-within > div,
.st-key-sb_loc input:focus,
.st-key-sb_cond textarea:focus {
    border-color: #19324A !important;
    box-shadow: 0 0 0 2px rgba(32, 197, 216, 0.35) !important;
}

/* Force all text, search input, and chevron arrow inside the selectbox to dark navy */
.st-key-sub_area [data-baseweb="select"] *,
.st-key-sub_area [data-baseweb="select"] input,
.st-key-sub_area [data-baseweb="select"] span,
.st-key-sub_area [data-baseweb="select"] div,
.st-key-rp_sort [data-baseweb="select"] *,
.st-key-lm_area [data-baseweb="select"] *,
.st-key-map_area [data-baseweb="select"] * {
    color: #19324A !important;
    -webkit-text-fill-color: #19324A !important;
    opacity: 1 !important;
}

.st-key-sub_area [data-baseweb="select"] svg,
.st-key-rp_sort [data-baseweb="select"] svg,
.st-key-lm_area [data-baseweb="select"] svg {
    fill: #19324A !important;
    color: #19324A !important;
}

/* Portalled Dropdown Popover & Option List (rendered at document body level) */
div[data-baseweb="popover"],
div[data-baseweb="popover"] > div,
div[data-baseweb="menu"],
div[data-baseweb="menu"] > div,
ul[role="listbox"],
div[role="listbox"] {
    background-color: #FFFFFF !important;
    color: #19324A !important;
    border: 1px solid #8FA5B5 !important;
    box-shadow: 0 10px 25px rgba(25, 50, 74, 0.16) !important;
}

ul[role="listbox"] li[role="option"],
div[role="listbox"] [role="option"],
div[data-baseweb="popover"] [role="option"],
div[data-baseweb="menu"] [role="option"] {
    background-color: #FFFFFF !important;
    color: #19324A !important;
    -webkit-text-fill-color: #19324A !important;
    font-weight: 500 !important;
    opacity: 1 !important;
}

ul[role="listbox"] li[role="option"] *,
div[role="listbox"] [role="option"] *,
div[data-baseweb="popover"] [role="option"] *,
div[data-baseweb="menu"] [role="option"] * {
    background-color: transparent !important;
    color: #19324A !important;
    -webkit-text-fill-color: #19324A !important;
    opacity: 1 !important;
}

ul[role="listbox"] li[role="option"]:hover,
div[role="listbox"] [role="option"]:hover,
div[data-baseweb="popover"] [role="option"]:hover,
div[data-baseweb="menu"] [role="option"]:hover {
    background-color: #E3EFF5 !important;
    color: #173D57 !important;
    -webkit-text-fill-color: #173D57 !important;
}

ul[role="listbox"] li[role="option"][aria-selected="true"],
div[role="listbox"] [role="option"][aria-selected="true"],
div[data-baseweb="popover"] [role="option"][aria-selected="true"],
div[data-baseweb="menu"] [role="option"][aria-selected="true"] {
    background-color: #D4E8F2 !important;
    color: #19324A !important;
    -webkit-text-fill-color: #19324A !important;
    font-weight: 600 !important;
}

/* Segmented control, slider, and buttons on Submit Report */
.st-key-sb_cond [data-testid="stSegmentedControl"] button {
    background: #EAF0F2 !important;
    color: #19324A !important;
    border: 1px solid #9CB3C3 !important;
}
.st-key-sb_cond [data-testid="stSegmentedControl"] button[aria-pressed="true"] {
    background: #19324A !important;
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
}
.st-key-sb_cond [data-testid="stSlider"] * {
    color: #19324A !important;
}
.st-key-sub_use_location button,
.st-key-sub_use_location button * {
    background-color: #EAF0F2 !important;
    color: #19324A !important;
    -webkit-text-fill-color: #19324A !important;
    border-color: #8FA5B5 !important;
}
.st-key-sub_go button,
.st-key-sub_go button * {
    background: #19324A !important;
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# State
# ----------------------------------------------------------------------------
NOW = dubai_now()


def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add depth and street/landmark columns if missing (demo rows get illustrative depths)."""
    if "depth_cm" in df.columns and "landmark" in df.columns:
        return df
    df = df.copy()
    if "depth_cm" not in df.columns:
        rng = np.random.default_rng(5)
        ranges = {"High": (40, 100), "Medium": (15, 40), "Low": (3, 15)}
        df["depth_cm"] = [int(rng.integers(*ranges[s])) for s in df["severity"]]
    if "landmark" not in df.columns:
        df["landmark"] = ""
    return df


def parse_coords(text: str):
    """Read 'lat, lon' pasted from Google Maps or Apple Maps. Returns (lat, lon) or None."""
    import re

    nums = re.findall(r"-?\d+(?:\.\d+)?", text or "")
    if len(nums) != 2:
        return None
    lat, lon = float(nums[0]), float(nums[1])
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        return None
    return lat, lon


def depth_reference(cm: float) -> str:
    if cm < 10:
        return "about ankle height"
    if cm < 25:
        return "about shin height"
    if cm < 45:
        return "about knee height"
    if cm < 80:
        return "about thigh height"
    return "waist height or deeper"


@st.cache_data(show_spinner=False)
def load_reports(anchor: str) -> pd.DataFrame:
    return make_demo_reports(pd.Timestamp(anchor).to_pydatetime())


@st.cache_data(show_spinner=False)
def load_history(anchor: str) -> pd.DataFrame:
    return make_history(pd.Timestamp(anchor).to_pydatetime())


@st.cache_data(show_spinner=False)
def run_analysis(data: bytes) -> dict:
    return analyze_image(Image.open(io.BytesIO(data)))


_anchor = NOW.strftime("%Y-%m-%dT%H:00:00")
if "reports" not in st.session_state:
    st.session_state.reports = load_reports(_anchor).copy()
st.session_state.setdefault("nav", "Overview")
st.session_state.setdefault("sub_up_n", 0)
st.session_state.setdefault("sub_sev", "Medium")
st.session_state.setdefault("sub_desc", "")
st.session_state.setdefault("sub_img_bytes", None)
st.session_state.setdefault("sub_msg", None)
st.session_state.setdefault("sub_landmark", "")
st.session_state.setdefault("sub_paste", "")
st.session_state.setdefault("sub_depth", 15)
st.session_state.reports = ensure_columns(st.session_state.reports)


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def esc(x) -> str:
    return html.escape(str(x))


def card(key: str):
    return st.container(border=True, key=f"card_{key}")


def header(eyebrow: str, title: str, sub: str):
    st.markdown(
        f'<div class="fl-head"><div><div class="fl-eyebrow">{esc(eyebrow)}</div>'
        f'<div class="fl-title">{esc(title)}</div><p class="fl-sub">{esc(sub)}</p></div>'
        f'<span class="pill-demo">Demonstration data</span></div>',
        unsafe_allow_html=True,
    )


def card_title(title: str, sub: str = ""):
    st.markdown(
        f'<div class="card-title">{esc(title)}</div>' + (f'<div class="card-sub">{esc(sub)}</div>' if sub else ""),
        unsafe_allow_html=True,
    )


def sev_pill(s: str) -> str:
    return f'<span class="pill pill-{esc(s).lower()}">{esc(s)}</span>'


def status_pill(s: str) -> str:
    return f'<span class="pill pill-st-{esc(s).lower()}">{esc(s)}</span>'


def ago(ts: pd.Timestamp) -> str:
    mins = max(int((NOW - ts.to_pydatetime()).total_seconds() // 60), 0)
    if mins < 60:
        return f"{mins} min ago"
    if mins < 48 * 60:
        return f"{mins // 60} h ago"
    return f"{mins // 1440} days ago"


def metric_card(col, label: str, value, desc: str, accent: str):
    col.markdown(
        f'<div class="fl-metric"><div class="fl-metric-top"><span class="dot" style="background:{accent}"></span>'
        f'{esc(label)}</div><div class="fl-metric-val">{esc(value)}</div>'
        f'<div class="fl-metric-desc">{esc(desc)}</div></div>',
        unsafe_allow_html=True,
    )


def legend():
    st.markdown(
        '<div class="legend">'
        f'<span><i class="dot" style="background:{RED}"></i>High</span>'
        f'<span><i class="dot" style="background:{ORANGE}"></i>Medium</span>'
        f'<span><i class="dot" style="background:{BLUE}"></i>Low</span></div>',
        unsafe_allow_html=True,
    )


def table_frame(d: pd.DataFrame) -> pd.DataFrame:
    t = d.copy()
    t["Time"] = t["ts"].dt.strftime("%d %b %Y, %H:%M")
    t["Depth (cm)"] = t["depth_cm"]
    t = t[["id", "area", "severity", "status", "Time", "Depth (cm)", "description"]]
    return t.rename(
        columns={"id": "Report ID", "area": "Location", "severity": "Severity", "status": "Status", "description": "Observation"}
    )


def render_detail(row: pd.Series):
    img = row.get("image")
    has_img = isinstance(img, (bytes, bytearray))
    depth = row.get("depth_cm")
    depth_txt = f"~{int(depth)} cm ({depth_reference(depth)})" if pd.notna(depth) else "Not provided"
    spot = str(row.get("landmark") or "").strip() or "Not provided"
    st.markdown(
        f'<div class="card-title">{esc(row["area"])}</div>'
        f'<div class="card-sub">{esc(row["id"])} &middot; reported {esc(ago(row["ts"]))}</div>'
        f'<div>{sev_pill(row["severity"])} {status_pill(row["status"])}</div>'
        '<div class="detail-grid">'
        f'<div><div class="detail-k">Reported</div><div class="detail-v">{esc(row["ts"].strftime("%d %b %Y, %H:%M"))}</div></div>'
        f'<div><div class="detail-k">Coordinates</div><div class="detail-v">{row["lat"]:.5f}, {row["lon"]:.5f}</div></div>'
        f'<div><div class="detail-k">Street / landmark</div><div class="detail-v">{esc(spot)}</div></div>'
        f'<div><div class="detail-k">Approx. depth</div><div class="detail-v">{esc(depth_txt)}</div></div>'
        '</div>'
        f'<div class="detail-k">Observation</div><div class="detail-v" style="line-height:1.5">{esc(row["description"])}</div>',
        unsafe_allow_html=True,
    )
    if has_img:
        st.image(bytes(img), width="stretch")
    else:
        st.markdown('<div class="detail-k" style="margin-top:.7rem">Photo</div><div class="detail-v">Not attached</div>', unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# Map
# ----------------------------------------------------------------------------
TOOLTIP = {
    "html": (
        "<div style='font-family:Inter,sans-serif'><b>{area}</b> &middot; {id}<br/>"
        "{severity} &middot; {status} &middot; ~{depth_str}<br/><span style='opacity:.75'>{time_str}</span><br/>{desc_short}</div>"
    ),
    "style": {
        "backgroundColor": NAVY,
        "color": "#FFFEFB",
        "fontSize": "12px",
        "borderRadius": "10px",
        "padding": "10px 12px",
        "maxWidth": "260px",
    },
}


def map_frame(d: pd.DataFrame) -> pd.DataFrame:
    m = d.copy()
    m["color"] = m["severity"].map(SEV_RGB)
    m["radius"] = m["severity"].map({"High": 260, "Medium": 210, "Low": 170})
    m["time_str"] = m["ts"].dt.strftime("%d %b %Y, %H:%M")
    m["desc_short"] = m["description"].str.slice(0, 90)
    m["depth_str"] = m["depth_cm"].astype(int).astype(str) + " cm"
    return m[["id", "area", "severity", "status", "time_str", "depth_str", "desc_short", "lat", "lon", "color", "radius"]]


def render_map(d: pd.DataFrame, key: str, height: int = 560, center=None, zoom: float = 10.4):
    """Render the flood map; returns the id of the selected report, if any."""
    m = map_frame(d)
    lat, lon = center or DUBAI_CENTER
    layers = [
        pdk.Layer(
            "ScatterplotLayer",
            data=m[m["severity"] == "High"],
            id="halo",
            get_position="[lon, lat]",
            get_radius="radius * 2.4",
            get_fill_color=[241, 91, 91, 40],
            stroked=False,
            pickable=False,
        ),
        pdk.Layer(
            "ScatterplotLayer",
            data=m,
            id="reports",
            get_position="[lon, lat]",
            get_radius="radius",
            get_fill_color="[color[0], color[1], color[2], 210]",
            get_line_color=[255, 255, 255, 255],
            stroked=True,
            line_width_min_pixels=2,
            radius_min_pixels=7,
            radius_max_pixels=26,
            pickable=True,
            auto_highlight=True,
        ),
    ]
    deck = pdk.Deck(
        layers=layers,
        initial_view_state=pdk.ViewState(latitude=lat, longitude=lon, zoom=zoom, pitch=0),
        map_style="light",
        tooltip=TOOLTIP,
    )
    event = st.pydeck_chart(
        deck, height=height, width="stretch", on_select="rerun", selection_mode="single-object", key=key
    )
    try:
        objs = event.selection.objects.get("reports") or []
        return objs[0].get("id") if objs else None
    except Exception:
        return None


def get_row(df: pd.DataFrame, rid):
    if not rid:
        return None
    hit = df[df["id"] == rid]
    return hit.iloc[0] if len(hit) else None


# ----------------------------------------------------------------------------
# Plotly styling
# ----------------------------------------------------------------------------
def style_fig(fig: go.Figure, height: int = 300) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=8, t=8, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color=NAVY, size=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, title=None),
        hoverlabel=dict(bgcolor=NAVY, font_color="#FFFEFB"),
    )
    fig.update_xaxes(showgrid=False, linecolor=BORDER, tickfont=dict(color=MUTED))
    fig.update_yaxes(gridcolor="#E8EDF0", zeroline=False, tickfont=dict(color=MUTED))
    return fig


def show(fig: go.Figure):
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


def chart_severity(d: pd.DataFrame) -> go.Figure:
    counts = d["severity"].value_counts().reindex(SEVERITIES, fill_value=0)
    fig = go.Figure(
        go.Bar(
            x=counts.index,
            y=counts.values,
            marker_color=[SEV_COLOR[s] for s in counts.index],
            text=counts.values,
            textposition="outside",
            cliponaxis=False,
            width=0.5,
        )
    )
    fig.update_yaxes(showticklabels=False, showgrid=False)
    return style_fig(fig)


def chart_location(d: pd.DataFrame) -> go.Figure:
    ct = pd.crosstab(d["area"], d["severity"]).reindex(columns=SEVERITIES, fill_value=0)
    ct = ct.loc[ct.sum(axis=1).sort_values().index]
    fig = go.Figure()
    for s in SEVERITIES:
        fig.add_trace(go.Bar(y=ct.index, x=ct[s], name=s, orientation="h", marker_color=SEV_COLOR[s]))
    fig.update_layout(barmode="stack")
    fig.update_yaxes(showgrid=False)
    fig.update_xaxes(showgrid=True, gridcolor="#E8EDF0")
    return style_fig(fig, 330)


def chart_status(d: pd.DataFrame) -> go.Figure:
    counts = d["status"].value_counts().reindex(STATUSES, fill_value=0)
    fig = go.Figure(
        go.Pie(
            labels=counts.index,
            values=counts.values,
            hole=0.64,
            marker=dict(colors=[STATUS_COLOR[s] for s in counts.index], line=dict(color=CARD, width=3)),
            textinfo="value",
            sort=False,
        )
    )
    fig.update_layout(annotations=[dict(text=f"<b>{len(d)}</b><br>reports", x=0.5, y=0.5, showarrow=False, font=dict(size=15))])
    return style_fig(fig, 300)


def chart_trend(d: pd.DataFrame) -> go.Figure:
    days = d["ts"].dt.floor("D")
    idx = pd.date_range(days.min(), days.max(), freq="D")
    counts = days.value_counts().reindex(idx, fill_value=0).sort_index()
    fig = go.Figure(
        go.Scatter(
            x=counts.index,
            y=counts.values,
            mode="lines+markers",
            line=dict(color=BLUE, width=2.5, shape="spline"),
            marker=dict(size=6, color=NAVY),
            fill="tozeroy",
            fillcolor="rgba(57,125,168,0.12)",
            hovertemplate="%{x|%d %b}: %{y} reports<extra></extra>",
        )
    )
    return style_fig(fig, 300)


# ----------------------------------------------------------------------------
# Pages
# ----------------------------------------------------------------------------
def page_overview(df: pd.DataFrame):
    header("Overview", "Flood activity across Dubai", "A live snapshot of crowdsourced flood intelligence.")

    total = len(df)
    active = int((df["status"] == "Active").sum())
    high = int((df["severity"] == "High").sum())
    high_open = int(((df["severity"] == "High") & (df["status"] != "Resolved")).sum())
    resolved = int((df["status"] == "Resolved").sum())

    c = st.columns(4, gap="medium")
    metric_card(c[0], "Total reports", total, "Submitted across all monitored areas", BLUE)
    metric_card(c[1], "Active flood reports", active, "Incidents currently unresolved", CYAN)
    metric_card(c[2], "High-severity reports", high, f"{high_open} still unresolved", RED)
    metric_card(c[3], "Resolved reports", resolved, f"{resolved / max(total, 1):.0%} resolution rate", NAVY)
    st.write("")

    left, right = st.columns([2.15, 1], gap="medium")
    with left:
        with card("ov_map"):
            card_title("Flood map", "Hover or click a marker to inspect the report.")
            f1, f2 = st.columns(2)
            sev = f1.multiselect("Severity", SEVERITIES, default=SEVERITIES, key="ov_sev")
            stat = f2.multiselect("Status", STATUSES, default=STATUSES, key="ov_stat")
            view = df[df["severity"].isin(sev) & df["status"].isin(stat)]
            legend()
            rid = render_map(view, key="ov_map_chart", height=470)
            row = get_row(df, rid)
            if row is not None:
                st.markdown(
                    f'<div class="note"><b>{esc(row["area"])}</b> &middot; {esc(row["id"])} &middot; '
                    f'{sev_pill(row["severity"])} {status_pill(row["status"])}<br>{esc(row["description"])}</div>',
                    unsafe_allow_html=True,
                )
    with right:
        with card("ov_alerts"):
            card_title("Priority alerts", "Unresolved high-severity reports")
            alerts = df[(df["severity"] == "High") & (df["status"] != "Resolved")].sort_values("ts", ascending=False).head(4)
            if alerts.empty:
                st.caption("No unresolved high-severity reports.")
            for _, a in alerts.iterrows():
                st.markdown(
                    '<div class="alert">'
                    f'<div class="alert-top"><span>{esc(a["area"])}</span>{sev_pill(a["severity"])}</div>'
                    f'<div class="alert-meta"><span>{esc(ago(a["ts"]))}</span><span>~{int(a["depth_cm"])} cm deep</span>{status_pill(a["status"])}</div>'
                    f'<div class="alert-desc">{esc(a["description"])}</div></div>',
                    unsafe_allow_html=True,
                )
    st.write("")
    with card("ov_recent"):
        card_title("Recent reports", "The ten most recent submissions")
        st.dataframe(table_frame(df.sort_values("ts", ascending=False).head(10)), hide_index=True, width="stretch")


def page_live_map(df: pd.DataFrame):
    header("Live Flood Map", "Geographical flood monitoring", "Filter, search and inspect reports across the city.")

    with card("lm_filters"):
        c1, c2, c3, c4 = st.columns([1.4, 1, 1, 1.2])
        q = c1.text_input("Search", placeholder="Location, report ID or keyword", key="lm_q")
        sev = c2.multiselect("Severity", SEVERITIES, default=SEVERITIES, key="lm_sev")
        stat = c3.multiselect("Status", STATUSES, default=STATUSES, key="lm_stat")
        areas = c4.multiselect("Location", list(AREAS), default=[], placeholder="All locations", key="lm_area")

    view = df[df["severity"].isin(sev) & df["status"].isin(stat)]
    if areas:
        view = view[view["area"].isin(areas)]
    if q.strip():
        needle = q.strip().lower()
        view = view[
            view["area"].str.lower().str.contains(needle, regex=False)
            | view["id"].str.lower().str.contains(needle, regex=False)
            | view["description"].str.lower().str.contains(needle, regex=False)
        ]

    center, zoom = DUBAI_CENTER, 10.4
    if 0 < len(view) and (q.strip() or areas):
        center = (float(view["lat"].mean()), float(view["lon"].mean()))
        zoom = 12.2 if view["area"].nunique() == 1 else 11.0

    st.write("")
    m_col, d_col = st.columns([2.3, 1], gap="medium")
    with m_col:
        with card("lm_map"):
            top_l, top_r = st.columns([2, 1])
            top_l.markdown(f'<div class="card-title">{len(view)} of {len(df)} reports shown</div>', unsafe_allow_html=True)
            top_r.markdown(
                f'<div style="text-align:right"><span class="pill-demo">Illustrative coordinates</span></div>', unsafe_allow_html=True
            )
            legend()
            key = f"lm_map_{zlib.crc32((q.strip() + ','.join(areas)).encode())}"
            rid = render_map(view, key=key, height=600, center=center, zoom=zoom)
    with d_col:
        with card("lm_detail"):
            row = get_row(df, rid)
            if row is not None:
                card_title("Report details")
                render_detail(row)
            else:
                card_title("Report details", "Click a marker on the map to inspect it.")
                st.markdown('<div class="detail-k" style="margin:.4rem 0">Latest in view</div>', unsafe_allow_html=True)
                for _, r in view.sort_values("ts", ascending=False).head(5).iterrows():
                    st.markdown(
                        f'<div class="alert" style="border-left-color:{SEV_COLOR[r["severity"]]}">'
                        f'<div class="alert-top"><span>{esc(r["area"])}</span>{sev_pill(r["severity"])}</div>'
                        f'<div class="alert-meta"><span>{esc(r["id"])}</span><span>{esc(ago(r["ts"]))}</span></div></div>',
                        unsafe_allow_html=True,
                    )
                if view.empty:
                    st.caption("No reports match the current filters.")


def page_reports(df: pd.DataFrame):
    header("Reports & Analysis", "Report management", "Search, filter and inspect every submitted report.")

    with card("rp_filters"):
        c1, c2, c3, c4 = st.columns([1.5, 1, 1, 1])
        q = c1.text_input("Search", placeholder="Location or report ID", key="rp_q")
        sev = c2.multiselect("Severity", SEVERITIES, default=SEVERITIES, key="rp_sev")
        stat = c3.multiselect("Status", STATUSES, default=STATUSES, key="rp_stat")
        order = c4.selectbox("Sort by", ["Newest first", "Oldest first", "Severity (high to low)", "Location (A-Z)"], key="rp_sort")

    view = df[df["severity"].isin(sev) & df["status"].isin(stat)]
    if q.strip():
        n = q.strip().lower()
        view = view[view["area"].str.lower().str.contains(n, regex=False) | view["id"].str.lower().str.contains(n, regex=False)]
    rank = {"High": 0, "Medium": 1, "Low": 2}
    if order == "Newest first":
        view = view.sort_values("ts", ascending=False)
    elif order == "Oldest first":
        view = view.sort_values("ts")
    elif order == "Severity (high to low)":
        view = view.assign(_r=view["severity"].map(rank)).sort_values(["_r", "ts"], ascending=[True, False]).drop(columns="_r")
    else:
        view = view.sort_values(["area", "ts"], ascending=[True, False])
    view = view.reset_index(drop=True)

    st.write("")
    t_col, i_col = st.columns([1.9, 1], gap="medium")
    with t_col:
        with card("rp_table"):
            card_title(f"{len(view)} reports", "Select a row to inspect it.")
            ev = st.dataframe(
                table_frame(view),
                hide_index=True,
                width="stretch",
                height=400,
                on_select="rerun",
                selection_mode="single-row",
                key="rp_table_sel",
            )
    with i_col:
        with card("rp_inspect"):
            rows = ev.selection.rows if ev is not None else []
            if rows and rows[0] < len(view):
                card_title("Report details")
                render_detail(view.iloc[rows[0]])
            else:
                card_title("Report details", "Select a report from the table to see its full record.")

    st.write("")
    if view.empty:
        st.info("No reports match the current filters, so there is nothing to chart.")
        return
    a, b = st.columns(2, gap="medium")
    with a:
        with card("rp_c1"):
            card_title("Reports by severity")
            show(chart_severity(view))
    with b:
        with card("rp_c2"):
            card_title("Reports by location", "Stacked by severity")
            show(chart_location(view))
    c, d = st.columns(2, gap="medium")
    with c:
        with card("rp_c3"):
            card_title("Active versus resolved")
            show(chart_status(view))
    with d:
        with card("rp_c4"):
            card_title("Report trend", "Reports per day")
            show(chart_trend(view))


def go_to_submit(severity: str | None, description: str, img: bytes):
    st.session_state.sub_sev = severity or "Low"
    st.session_state.sub_desc = description
    st.session_state.sub_img_bytes = img
    st.session_state.nav = "Submit a Report"


def page_ai():
    header("AI Image Analysis", "Flood detection from photographs", "Upload a road or street photo and get a structured assessment.")
    left, right = st.columns([1, 1.25], gap="medium")

    with left:
        with card("ai_up"):
            card_title("Upload a photograph", "JPG, PNG or WebP of a road, underpass or street.")
            up = st.file_uploader("Photograph", type=["jpg", "jpeg", "png", "webp"], label_visibility="collapsed", key="ai_upload")
            st.markdown(
                '<div class="chip-row"><span class="chip">1. Upload</span><span class="chip">2. Segment water surfaces</span>'
                '<span class="chip">3. Classify severity</span><span class="chip">4. Explain</span></div>',
                unsafe_allow_html=True,
            )
        st.write("")
        if os.environ.get("ANTHROPIC_API_KEY"):
            note = (
                "<b>About this analysis.</b> Photos are assessed by a vision AI model that returns a severity "
                "category, a confidence estimate and an explanation. The confidence is the model's own estimate, "
                "not a calibrated probability, and results are indicative. Always verify on site."
            )
        else:
            note = (
                "<b>About this analysis.</b> No AI key is configured, so this uses a local colour- and "
                "texture-based prototype. It can be wrong on unusual scenes (sand, wet pavement, glare). "
                "Results are indicative. Always verify on site."
            )
        st.markdown(f'<div class="note">{note}</div>', unsafe_allow_html=True)

    with right:
        if up is None:
            with card("ai_wait"):
                card_title("Awaiting a photograph", "Each analysis returns:")
                st.markdown(
                    "- Whether visible flooding is present\n- A visual severity category\n- A confidence score\n"
                    "- A highlighted view of the detected water\n- A short written explanation"
                )
            return
        data = up.getvalue()
        try:
            res = run_analysis(data)
        except Exception as exc:  # unreadable or too-small image
            with card("ai_err"):
                st.error(f"This image could not be analysed: {exc}")
            return

        sev = res["severity"]
        accent = {"none": CYAN, "low": BLUE, "moderate": ORANGE, "severe": RED}[res["key"]]
        with card("ai_result"):
            st.markdown(
                '<div class="detail-k">Classification</div>'
                f'<div class="big-class" style="color:{accent}">{esc(res["label"])}</div>',
                unsafe_allow_html=True,
            )
            m1, m2, m3 = st.columns(3)
            m1.markdown(
                f'<div class="detail-k">Confidence</div><div class="detail-v">{res["confidence"]:.0%}</div>'
                f'<div class="conf-track"><div class="conf-fill" style="width:{res["confidence"] * 100:.0f}%"></div></div>',
                unsafe_allow_html=True,
            )
            if res["coverage"] is not None:
                m2.markdown(f'<div class="detail-k">Water-like coverage</div><div class="detail-v">{res["coverage"]:.0%} of ground</div>', unsafe_allow_html=True)
            else:
                m2.markdown(f'<div class="detail-k">Engine</div><div class="detail-v">{esc(res["engine"])}</div>', unsafe_allow_html=True)
            m3.markdown(f'<div class="detail-k">Report severity</div><div class="detail-v">{esc(sev) if sev else "None"}</div>', unsafe_allow_html=True)
            st.markdown(f'<p style="margin:.9rem 0 .2rem;line-height:1.55;font-size:.92rem">{esc(res["explanation"])}</p>', unsafe_allow_html=True)

    st.write("")
    with card("ai_images"):
        if res["overlay"] is not None:
            card_title("Visual breakdown", "Detected water-like surfaces are highlighted in cyan.")
            i1, i2 = st.columns(2)
            i1.image(res["original"], caption="Original", width="stretch")
            i2.image(res["overlay"], caption="Detected regions", width="stretch")
        else:
            card_title("Analysed photograph")
            st.image(res["original"], width="stretch")
        if sev:
            desc = (
                f"Visible flooding assessed as {res['label'].lower()} by FloodLens Vision "
                f"({res['confidence']:.0%} confidence). {res['explanation']}"
            )
            st.button(
                "Create a report from this analysis",
                type="primary",
                key="ai_to_report",
                on_click=go_to_submit,
                args=(sev, desc, data),
            )


def page_history():
    header("Historical Insights", "Recurring flood patterns", "Where flooding repeats, and where inspections matter most.")
    hist = load_history(_anchor)

    period = st.segmented_control("Period", ["Last 12 months", "Last 24 months", "All time"], default="Last 12 months", key="hi_period")
    months = sorted(hist["month"].unique())
    n = {"Last 12 months": 12, "Last 24 months": 24}.get(period or "Last 12 months", len(months))
    keep = months[-n:]
    h = hist[hist["month"].isin(keep)]

    monthly = h.groupby("month")["reports"].sum()
    by_area = h.groupby("area")["reports"].sum().sort_values(ascending=False)
    c = st.columns(4, gap="medium")
    metric_card(c[0], "Reports in period", int(monthly.sum()), f"{len(keep)} months of observations", BLUE)
    metric_card(c[1], "Most affected area", by_area.index[0], f"{int(by_area.iloc[0])} reports", RED)
    metric_card(c[2], "Peak month", monthly.idxmax().strftime("%b %Y"), f"{int(monthly.max())} reports", ORANGE)
    metric_card(c[3], "Monthly average", f"{monthly.mean():.0f}", "Reports per month", NAVY)
    st.write("")

    a, b = st.columns([1.7, 1], gap="medium")
    with a:
        with card("hi_trend"):
            card_title("Monthly flood reports", "Seasonal peaks follow the winter rainfall period.")
            fig = go.Figure(
                go.Scatter(
                    x=monthly.index,
                    y=monthly.values,
                    mode="lines+markers",
                    line=dict(color=BLUE, width=2.5),
                    marker=dict(size=6, color=NAVY),
                    fill="tozeroy",
                    fillcolor="rgba(57,125,168,0.12)",
                    hovertemplate="%{x|%b %Y}: %{y} reports<extra></extra>",
                )
            )
            if pd.Timestamp("2024-04-01") in monthly.index:
                fig.add_annotation(
                    x=pd.Timestamp("2024-04-01"),
                    y=float(monthly.loc[pd.Timestamp("2024-04-01")]),
                    text="April 2024 rainfall event",
                    showarrow=True,
                    arrowhead=0,
                    ax=60,
                    ay=-10,
                    font=dict(size=11, color=MUTED),
                )
            show(style_fig(fig, 320))
    with b:
        with card("hi_season"):
            card_title("Seasonality", "Average reports by calendar month")
            totals = hist.groupby("month")["reports"].sum()
            avg = totals.groupby(totals.index.month).mean().reindex(range(1, 13))
            labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
            colors = [NAVY if v >= avg.quantile(0.75) else "#9CBFD6" for v in avg.values]
            fig = go.Figure(go.Bar(x=labels, y=avg.values, marker_color=colors, hovertemplate="%{x}: %{y:.0f} avg<extra></extra>"))
            show(style_fig(fig, 320))

    st.write("")
    with card("hi_heat"):
        card_title("Recurrence heatmap", "Reports per location per month, last 12 months")
        recent = hist[hist["month"].isin(months[-12:])]
        pv = recent.pivot(index="area", columns="month", values="reports").fillna(0)
        pv = pv.loc[pv.sum(axis=1).sort_values().index]
        fig = go.Figure(
            go.Heatmap(
                z=pv.values,
                x=[m.strftime("%b %y") for m in pv.columns],
                y=list(pv.index),
                colorscale=[[0, "#EAF0F2"], [0.5, "#7FB2D1"], [1, "#19324A"]],
                xgap=3,
                ygap=3,
                colorbar=dict(thickness=10, title=None),
                hovertemplate="%{y}, %{x}: %{z} reports<extra></extra>",
            )
        )
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=False)
        show(style_fig(fig, 340))

    st.write("")
    with card("hi_priority"):
        card_title("Inspection priority", "Ranked by volume, months with repeat flooding and high-severity share (full history).")
        pt = priority_table(hist)
        st.dataframe(
            pt.rename(
                columns={
                    "rank": "Rank",
                    "location": "Location",
                    "total": "Reports",
                    "recurring": "Repeat months",
                    "high_share": "High-severity share",
                    "score": "Priority score",
                    "action": "Suggested action",
                }
            ),
            hide_index=True,
            width="stretch",
            column_config={
                "Priority score": st.column_config.ProgressColumn("Priority score", min_value=0, max_value=100, format="%d"),
                "High-severity share": st.column_config.NumberColumn("High-severity share", format="percent"),
            },
        )


def resolve_coords(area: str):
    """Return the demo coordinates assigned to the selected area."""
    return AREAS[area], None


def submit_report():
    ss = st.session_state
    area = ss.get("sub_area") or list(AREAS)[0]
    landmark = (ss.get("sub_landmark") or "").strip()
    desc = (ss.get("sub_desc") or "").strip()
    if len(landmark) < 3:
        ss.sub_msg = ("error", "Please give the street name or nearest landmark so responders can find the exact spot.")
        return
    coords, err = resolve_coords(area)
    if err:
        ss.sub_msg = ("error", err)
        return
    lat, lon = coords
    if not (22.5 <= lat <= 26.5 and 51.0 <= lon <= 56.5):
        ss.sub_msg = ("error", "Those coordinates fall outside the UAE. Please check latitude and longitude.")
        return
    if len(desc) < 10:
        ss.sub_msg = ("error", "Please describe what you can see in at least 10 characters.")
        return
    photo = ss.get(f"sub_photo_{ss.sub_up_n}")
    img = photo.getvalue() if photo is not None else ss.get("sub_img_bytes")

    df = ss.reports
    nums = df["id"].str.slice(3).astype(int)
    rid = f"FL-{int(nums.max()) + 1}"
    row = {
        "id": rid,
        "area": area,
        "lat": float(lat),
        "lon": float(lon),
        "ts": pd.Timestamp(dubai_now()).floor("min"),
        "severity": ss.get("sub_sev") or "Medium",
        "status": "Active",
        "description": desc,
        "image": img,
        "depth_cm": int(ss.get("sub_depth", 15)),
        "landmark": landmark,
    }
    ss.reports = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    ss.sub_desc = ""
    ss.sub_landmark = ""
    ss.sub_depth = 15
    ss.sub_img_bytes = None
    ss.sub_up_n += 1
    ss.sub_msg = ("success", f"Report {rid} submitted. It now appears on the Overview and Live Flood Map pages.")


def page_submit(df: pd.DataFrame):
    header("Submit a Report", "Report flooding", "Turn what you see into structured, location-specific intelligence.")
    msg = st.session_state.sub_msg
    if msg:
        (st.success if msg[0] == "success" else st.error)(msg[1])
        st.session_state.sub_msg = None

    form, side = st.columns([1.5, 1], gap="medium")
    with form:
        with card("sb_loc"):
            card_title("1. Exact location", "Help responders find the precise spot.")
            area = st.selectbox("Area", list(AREAS), key="sub_area")
            st.text_input("Street or nearest landmark", placeholder="e.g. Underpass on the main road near the metro station", key="sub_landmark")
            st.button(
                "📍 Use my location",
                key="sub_use_location",
                use_container_width=True,
                help="Demo button only. Location detection is not enabled in this prototype.",
            )
        st.write("")
        with card("sb_cond"):
            card_title("2. Flood conditions")
            st.segmented_control("Flood severity", SEVERITIES, key="sub_sev")
            st.slider("Approximate water depth (cm)", 1, 150, key="sub_depth")
            d = int(st.session_state.sub_depth)
            note = f"About {d} cm, {depth_reference(d)}."
            if d >= 30:
                note += " Water at this depth can stall vehicles."
            st.caption(note)
            st.text_area("What can you see?", placeholder="Describe affected lanes, traffic and anything blocked.", height=100, key="sub_desc")
            st.file_uploader("Photo (optional)", type=["jpg", "jpeg", "png", "webp"], key=f"sub_photo_{st.session_state.sub_up_n}")
            if st.session_state.sub_img_bytes and st.session_state[f"sub_photo_{st.session_state.sub_up_n}"] is None:
                st.caption("The photo from your AI analysis will be attached.")
            st.button("Submit report", type="primary", on_click=submit_report, key="sub_go")
    with side:
        with card("sb_map"):
            card_title("Location preview")
            coords, err = resolve_coords(area)
            if err:
                st.warning(err)
                coords = AREAS[area]
            lat, lon = float(coords[0]), float(coords[1])
            pt = pd.DataFrame([{"lat": lat, "lon": lon}])
            st.pydeck_chart(
                pdk.Deck(
                    layers=[
                        pdk.Layer(
                            "ScatterplotLayer",
                            data=pt,
                            get_position="[lon, lat]",
                            get_radius=220,
                            get_fill_color=[32, 197, 216, 220],
                            get_line_color=[255, 255, 255],
                            stroked=True,
                            line_width_min_pixels=2,
                            radius_min_pixels=8,
                        )
                    ],
                    initial_view_state=pdk.ViewState(latitude=lat, longitude=lon, zoom=12.5),
                    map_style="light",
                ),
                height=260,
                width="stretch",
            )
            st.caption(f"Reporting at {lat:.5f}, {lon:.5f}")
        st.write("")
        st.markdown(
            '<div class="note"><b>How reports are used.</b> Each submission becomes a time-stamped, geo-located record '
            'with depth and severity. Reports feed the live map, recurrence analysis and inspection prioritisation. '
            'This demo stores reports for your current browser session only.</div>',
            unsafe_allow_html=True,
        )


# ----------------------------------------------------------------------------
# Sidebar + routing
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        '<div class="fl-logo"><svg width="36" height="36" viewBox="0 0 34 34" fill="none">'
        '<circle cx="17" cy="17" r="15" stroke="#20C5D8" stroke-width="2.5"/>'
        '<path d="M8 19c3-4 6 4 9 0s6 4 9 0" stroke="#fff" stroke-width="2.4" stroke-linecap="round"/>'
        '<circle cx="17" cy="11" r="2" fill="#20C5D8"/></svg><span class="fl-logo-name">FloodLens</span></div>'
        '<div class="fl-logo-tag">See the danger before it spreads.</div>'
        '<div class="fl-nav-label">Platform</div>',
        unsafe_allow_html=True,
    )
    st.radio("Navigation", PAGES, key="nav", label_visibility="collapsed")
    st.markdown(
        '<div class="fl-side-note"><b>Demonstration build</b><br>Reports, coordinates and history are '
        'illustrative sample data for Dubai, not real observations.</div>',
        unsafe_allow_html=True,
    )

reports = st.session_state.reports
page = st.session_state.nav
if page == "Overview":
    page_overview(reports)
elif page == "Live Flood Map":
    page_live_map(reports)
elif page == "Reports & Analysis":
    page_reports(reports)
elif page == "AI Image Analysis":
    page_ai()
elif page == "Historical Insights":
    page_history()
else:
    page_submit(reports)