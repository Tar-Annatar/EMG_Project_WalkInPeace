"""
EMG Pathology & Freeze-of-Gait Monitor — Streamlit App
=======================================================
Public-facing build with real-time ML inference, synthetic playback,
and an explicit Arduino IoT Cloud connector card with connection testing.

Run locally:
    streamlit run app.py
"""

import os
import time
import hashlib
from io import BytesIO

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from signal_utils import (
    FS, N_CHANNELS, CONDITION_NAMES, TTF_MAX,
    StreamingFeatureExtractor, RunningNormalizer, SyntheticEMGStreamer,
)
from arduino_cloud import ArduinoCloudClient, ArduinoCloudError, decode_emg_batch

MAX_LOG_LEN = 1000       # Cap prediction history for smooth browser performance
DEFAULT_SPEED = 1.0       # Fixed playback speed
DEFAULT_CHUNK_MS = 150    # Fixed update interval

# ─────────────────────────────────────────────────────────────────────────
# LOGO
# ─────────────────────────────────────────────────────────────────────────
def find_logo_path():
    for name in ["logo.png", "logo.jpeg", "logo.jpg"]:
        if os.path.exists(name):
            return name
    return None

LOGO_PATH = find_logo_path()

# ─────────────────────────────────────────────────────────────────────────
# PAGE CONFIG + THEME
# ─────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="WalkInPeace EMG Monitor",
    page_icon=LOGO_PATH or "🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

THEME_CSS = """
<style>
:root {
    --bg:        #f4f6fb;
    --surface:   #ffffff;
    --border:    #e4e8f1;
    --ink:       #0f172a;
    --ink-soft:  #56607a;
    --accent:    #4f46e5;
    --accent-2:  #14b8a6;
    --accent-soft: #eef0fe;
    --success:   #10b981;
    --warning:   #f59e0b;
    --danger:    #ef4444;
    --sidebar-1: #0b1120;
    --sidebar-2: #172038;
}
html, body, [class*="css"] {
    font-family: "Inter", "Segoe UI", -apple-system, BlinkMacSystemFont, sans-serif;
}
.stApp { background: var(--bg); }
h1, h2, h3, h4 { color: var(--ink) !important; letter-spacing: -0.01em; }
p, span, label, div { color: var(--ink); }

/* App header */
.app-header { display:flex; align-items:center; gap:14px; padding: 4px 0 18px 0; }
.app-header .kicker {
    color: var(--accent); font-weight:700; font-size:0.78em;
    letter-spacing:0.08em; text-transform:uppercase; margin:0;
}
.app-header h1 { margin:0; font-size:1.7em; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, var(--sidebar-1) 0%, var(--sidebar-2) 100%);
    border-right: 1px solid rgba(255,255,255,0.06);
}
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] h4,
section[data-testid="stSidebar"] .stMarkdown,
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"],
section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] {
    color: #e7eaf6 !important;
}
section[data-testid="stSidebar"] input,
section[data-testid="stSidebar"] textarea,
section[data-testid="stSidebar"] [data-baseweb="select"] *,
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"],
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] *,
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzoneInstructions"] * {
    color: var(--ink) !important;
}
section[data-testid="stSidebar"] input::placeholder { color: #7c8aa5 !important; opacity:1; }
section[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.08); }
.sidebar-section-title {
    font-size:0.72em; font-weight:700; letter-spacing:0.09em; text-transform:uppercase;
    color:#8b96b8 !important; margin: 14px 0 4px 0;
}

/* Buttons */
div.stButton > button, div.stDownloadButton > button {
    background: var(--accent); color:white; border:none; border-radius:9px;
    padding:0.5em 1.1em; font-weight:600; transition: all 0.15s ease;
    box-shadow: 0 1px 2px rgba(79,70,229,0.25);
}
div.stButton > button:hover { background:#4338ca; transform: translateY(-1px); }
div.stButton > button:active { transform: translateY(0); }
div.stDownloadButton > button { background: var(--accent-2); box-shadow: 0 1px 2px rgba(20,184,166,0.25); }
div.stDownloadButton > button:hover { background:#0d9488; transform: translateY(-1px); }

/* Metric cards */
[data-testid="stMetric"] {
    background: var(--surface); border: 1px solid var(--border); border-radius: 14px;
    padding: 14px 18px; box-shadow: 0 1px 3px rgba(15,23,42,0.04);
}
[data-testid="stMetricLabel"] { color: var(--ink-soft) !important; font-weight:600; }
[data-testid="stMetricValue"] { color: var(--ink) !important; }

/* Login card */
.login-box {
    max-width: 430px; margin: 8vh auto 0 auto; background: var(--surface);
    padding: 2.6em 2.6em 2em 2.6em; border-radius: 18px;
    box-shadow: 0 20px 60px rgba(15,23,42,0.12);
    border-top: 5px solid var(--accent);
}
.login-title { text-align:center; color: var(--ink); margin-bottom:0.15em; }
.login-sub { text-align:center; color: var(--ink-soft); margin-bottom:1.6em; font-size:0.92em; }

/* Status badges */
.badge { display:inline-block; padding:4px 12px; border-radius:20px; font-size:0.78em; font-weight:700; letter-spacing:0.02em; }
.badge-live   { background:#dcfce7; color:#15803d; }
.badge-paused { background:#f1f5f9; color:#475569; }
.badge-demo   { background:#fef3c7; color:#92400e; }
.badge-error  { background:#fee2e2; color:#b91c1c; }
.badge-iot    { background:#e0e7ff; color:#3730a3; }

/* Panel container around charts */
.panel {
    background: var(--surface); border:1px solid var(--border);
    border-radius:16px; padding: 6px 6px 2px 6px; margin-bottom:16px;
    box-shadow: 0 1px 3px rgba(15,23,42,0.04);
}
</style>
"""
st.markdown(THEME_CSS, unsafe_allow_html=True)

PLOTLY_COLORS = ["#4f46e5", "#14b8a6", "#f59e0b", "#ef4444",
                  "#0ea5e9", "#8b5cf6", "#22c55e", "#ec4899"]
CHART_TEMPLATE = dict(plot_bgcolor="#fbfbfe", paper_bgcolor="white")

# ─────────────────────────────────────────────────────────────────────────
# AUTH
# ─────────────────────────────────────────────────────────────────────────
def _hash(pw: str) -> str:
    return hashlib.sha256(pw.encode()).hexdigest()

DEFAULT_USERS = {
    "demo": _hash("emg2026"),
    "clinician": _hash("neurogait"),
}

def get_user_db():
    try:
        if "credentials" in st.secrets:
            return {u: _hash(p) for u, p in st.secrets["credentials"].items()}
    except Exception:
        pass
    return DEFAULT_USERS


def login_page():
    st.markdown('<div class="login-box">', unsafe_allow_html=True)
    if LOGO_PATH:
        lcol1, lcol2, lcol3 = st.columns([1, 1, 1])
        with lcol2:
            st.image(LOGO_PATH, width="stretch")
        st.markdown('<h2 class="login-title">WalkInPeace EMG Monitor</h2>', unsafe_allow_html=True)
    else:
        st.markdown('<h2 class="login-title">🧠 WalkInPeace EMG Monitor</h2>', unsafe_allow_html=True)
    st.markdown('<div class="login-sub">Sign in to access the live EMG dashboard</div>', unsafe_allow_html=True)

    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Sign In", width="stretch")

    if submitted:
        users = get_user_db()
        if username in users and users[username] == _hash(password):
            st.session_state.authenticated = True
            st.session_state.username = username
            st.rerun()
        else:
            st.error("Invalid username or password.")

    with st.expander("Demo credentials"):
        st.code("username: demo\npassword: emg2026", language="text")
    st.markdown("</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────
# AUTOMATED MODEL & STATS LOADING
# ─────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model_cached(_path: str, content_hash: str):
    import tensorflow as tf
    return tf.keras.models.load_model(_path, compile=False)


def find_default_model_path():
    for name in ["fog_pathology_model.h5", "model.h5"]:
        if os.path.exists(name):
            return name
    return None


@st.cache_data(show_spinner=False)
def load_norm_stats_cached(content_hash: str, raw_bytes: bytes):
    data = np.load(BytesIO(raw_bytes))
    return data["mean"], data["std"]


def load_bundled_norm_stats():
    for name in ["norm_stats.npz"]:
        if os.path.exists(name):
            with open(name, "rb") as f:
                raw = f.read()
            return load_norm_stats_cached(hashlib.md5(raw).hexdigest(), raw)
    return None, None


# ─────────────────────────────────────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────────────────────────────────────
def init_state():
    defaults = dict(
        authenticated=False,
        username=None,
        extractor=None,
        normalizer=None,
        pred_log=[],
        session_t=0.0,
        running=False,
        streamer=None,
        norm_mean=None,
        norm_std=None,
        example_active=False,
        example_stop_at=None,
        arduino_client=None,
        arduino_status="Disconnected",
        last_poll_time=None,
        source_mode="Synthetic Demo",
    )
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# ─────────────────────────────────────────────────────────────────────────
# LOGIN GATE
# ─────────────────────────────────────────────────────────────────────────
if not st.session_state.authenticated:
    login_page()
    st.stop()

# ─────────────────────────────────────────────────────────────────────────
# ARDUINO HELPER
# ─────────────────────────────────────────────────────────────────────────
def get_arduino_credentials():
    cid, csec = "", ""
    try:
        if "arduino_cloud" in st.secrets:
            cid = st.secrets["arduino_cloud"].get("client_id", "")
            csec = st.secrets["arduino_cloud"].get("client_secret", "")
    except Exception:
        pass
    return cid, csec

# ─────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────
with st.sidebar:
    if LOGO_PATH:
        st.image(LOGO_PATH, width="stretch")
        st.markdown(f"### Welcome, {st.session_state.username}")
    else:
        st.markdown(f"### 🧠 Welcome, {st.session_state.username}")
    if st.button("Log out", width="stretch"):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()

    st.markdown('<div class="sidebar-section-title">Data Source</div>', unsafe_allow_html=True)
    
    source = st.radio(
        "Source Mode",
        ["Synthetic Demo", "Arduino Cloud (IoT)"],
        key="source_mode",
        label_visibility="collapsed",
    )

    scenario, inject_freeze, freeze_at = CONDITION_NAMES[0], False, None
    ard_client_id, ard_client_secret, ard_thing_id, ard_var_name, ard_poll_s = "", "", "", "emgBatch", 0.5

    if source == "Synthetic Demo":
        st.markdown('<div class="sidebar-section-title">Try a scenario</div>', unsafe_allow_html=True)
        scenario = st.selectbox("Simulated condition", CONDITION_NAMES)
        if scenario == "Parkinson's / FOG risk":
            inject_freeze = st.checkbox("Include a freeze-of-gait event", value=True)
            if inject_freeze:
                freeze_at = st.slider("Freeze occurs at (s)", 10, 60, 25)
    else:
        # ─────────────────────────────────────────────────────────────
        # ALWAYS-VISIBLE ARDUINO CONNECTOR CARD
        # ─────────────────────────────────────────────────────────────
        st.markdown('<div class="sidebar-section-title">📡 Arduino IoT Connector</div>', unsafe_allow_html=True)
        
        secret_cid, secret_csec = get_arduino_credentials()
        ard_client_id = st.text_input("Client ID", value=secret_cid, help="From Arduino Cloud -> API Keys")
        ard_client_secret = st.text_input("Client Secret", value=secret_csec, type="password")
        ard_thing_id = st.text_input("Thing ID", help="Found on your Thing's dashboard page")
        ard_var_name = st.text_input("Batch Variable Name", value="emgBatch")
        ard_poll_s = st.slider("Poll Interval (s)", 0.3, 3.0, 0.5, step=0.1)

        # Connection Status Display Container
        with st.container():
            st.markdown("---")
            st.markdown("#### Hardware Status")
            if st.session_state.arduino_status == "Connected":
                st.success("🟢 Device Connected (Wi-Fi Active)")
            elif st.session_state.arduino_status == "Polling":
                st.warning("🟡 Polling Cloud API...")
            else:
                st.error("🔴 Device Disconnected")

            if st.session_state.last_poll_time:
                st.caption(f"Last Polled: {st.session_state.last_poll_time}")

            if st.button("⚡ Test Connection", width="stretch"):
                if not (ard_client_id and ard_client_secret and ard_thing_id):
                    st.error("Please enter Client ID, Secret, and Thing ID.")
                else:
                    with st.spinner("Ping Arduino IoT Cloud..."):
                        try:
                            client = ArduinoCloudClient(ard_client_id, ard_client_secret)
                            client.get_property_value(ard_thing_id, ard_var_name)
                            st.session_state.arduino_client = client
                            st.session_state.arduino_status = "Connected"
                            st.session_state.last_poll_time = time.strftime("%H:%M:%S")
                            st.success("Connection Successful!")
                        except Exception as e:
                            st.session_state.arduino_status = "Disconnected"
                            st.error(f"Connection Failed: {e}")

    st.markdown('<div class="sidebar-section-title">Controls</div>', unsafe_allow_html=True)
    col_a, col_b, col_c = st.columns(3)
    play_clicked = col_a.button("▶ Play", width="stretch")
    pause_clicked = col_b.button("⏸ Pause", width="stretch")
    stop_clicked = col_c.button("⏹ Stop", width="stretch")

    example_clicked = st.button(
        "🧪 Run Example (FOG, 15s)", width="stretch",
        help="A ready-made 15s demo of a Parkinson's/FOG scenario with a freeze event.",
    )

    st.markdown('<div class="sidebar-section-title">Report</div>', unsafe_allow_html=True)
    has_data = bool(st.session_state.pred_log)
    if has_data:
        log = st.session_state.pred_log
        report_df = pd.DataFrame({
            "time_s": [round(r["t"], 2) for r in log],
            **{CONDITION_NAMES[i]: [round(float(r["path_probs"][i]), 4) for r in log]
               for i in range(len(CONDITION_NAMES))},
            "time_to_freeze_s": [r["fog_seconds"] for r in log],
        })
        report_csv = report_df.to_csv(index=False).encode()
    else:
        report_csv = b""
    st.download_button(
        "⬇ Download report", data=report_csv,
        file_name="walkinpeace_report.csv", mime="text/csv",
        disabled=not has_data, width="stretch",
        help="Download this session's predictions as a CSV report.",
    )

# ─────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────
if LOGO_PATH:
    hcol1, hcol2 = st.columns([1, 10])
    with hcol1:
        st.image(LOGO_PATH, width="stretch")
    with hcol2:
        st.markdown(
            '<div class="app-header"><div><p class="kicker">Real-time neuromuscular monitoring</p>'
            '<h1>WalkInPeace — EMG Assister</h1></div></div>',
            unsafe_allow_html=True,
        )
else:
    st.markdown(
        '<div class="app-header"><div><p class="kicker">Real-time neuromuscular monitoring</p>'
        '<h1>🧠 WalkInPeace — EMG Assister</h1></div></div>',
        unsafe_allow_html=True,
    )
status_placeholder = st.empty()

# ─────────────────────────────────────────────────────────────────────────
# PLAY / PAUSE / STOP / EXAMPLE HANDLING
# ─────────────────────────────────────────────────────────────────────────
if stop_clicked:
    st.session_state.extractor = StreamingFeatureExtractor()
    st.session_state.normalizer = None
    st.session_state.pred_log = []
    st.session_state.session_t = 0.0
    st.session_state.streamer = None
    st.session_state.running = False
    st.session_state.example_active = False
    st.session_state.example_stop_at = None
    st.session_state.arduino_status = "Disconnected"

if pause_clicked:
    st.session_state.running = False
    st.session_state.example_active = False
    st.session_state.example_stop_at = None

if play_clicked:
    st.session_state.running = True
    st.session_state.example_active = False
    st.session_state.example_stop_at = None
    if st.session_state.extractor is None:
        st.session_state.extractor = StreamingFeatureExtractor()
    
    if source == "Synthetic Demo":
        if st.session_state.streamer is None:
            use_freeze = scenario == "Parkinson's / FOG risk" and inject_freeze
            st.session_state.streamer = SyntheticEMGStreamer(
                scenario=scenario, seed=42,
                freeze_at_s=freeze_at if use_freeze else None,
            )
    else:
        if not (ard_client_id and ard_client_secret and ard_thing_id):
            status_placeholder.error("Arduino Cloud needs Client ID, Client Secret, and Thing ID.")
            st.session_state.running = False
            st.session_state.arduino_status = "Disconnected"
        else:
            st.session_state.arduino_client = ArduinoCloudClient(ard_client_id, ard_client_secret)
            st.session_state.arduino_status = "Polling"

if example_clicked:
    st.session_state.extractor = StreamingFeatureExtractor()
    st.session_state.normalizer = None
    st.session_state.pred_log = []
    st.session_state.session_t = 0.0
    st.session_state.streamer = SyntheticEMGStreamer(
        scenario="Parkinson's / FOG risk", seed=42, freeze_at_s=8.0,
    )
    st.session_state.running = True
    st.session_state.example_active = True
    st.session_state.example_stop_at = 15.0

# ─────────────────────────────────────────────────────────────────────────
# LOAD MODEL & NORM STATS AUTOMATICALLY
# ─────────────────────────────────────────────────────────────────────────
model = None
model_err = None
default_path = find_default_model_path()
try:
    if default_path:
        content_hash = hashlib.md5(default_path.encode()).hexdigest()
        model = load_model_cached(default_path, content_hash)
    else:
        model_err = "Model file non-existent. Place fog_pathology_model.h5 next to app.py."
except Exception as e:
    model_err = f"Model loading failed: {e}"

if st.session_state.norm_mean is None:
    nm, ns = load_bundled_norm_stats()
    if nm is not None:
        st.session_state.norm_mean, st.session_state.norm_std = nm, ns

if model_err:
    status_placeholder.warning(model_err)
else:
    run_badge = (
        '<span class="badge badge-live">LIVE</span>' if st.session_state.running
        else '<span class="badge badge-paused">PAUSED</span>'
    )
    status_placeholder.markdown(run_badge, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────
# LAYOUT PLACEHOLDERS
# ─────────────────────────────────────────────────────────────────────────
metrics_row = st.empty()
range_choice = st.radio(
    "Time range for charts", ["Last 5s", "Last 15s", "Last 30s", "Full session"],
    horizontal=True, key="range_choice",
)
emg_chart_ph = st.empty()
pred_chart_ph = st.empty()
fog_chart_ph = st.empty()
table_ph = st.empty()


def range_seconds(label):
    return {"Last 5s": 5, "Last 15s": 15, "Last 30s": 30, "Full session": None}[label]


def render_dashboard():
    ext = st.session_state.extractor
    log = st.session_state.pred_log
    win_s = range_seconds(st.session_state.range_choice)

    with metrics_row.container():
        c1, c2, c3, c4 = st.columns(4)
        if log:
            last = log[-1]
            top_idx = int(np.argmax(last["path_probs"][1:])) + 1
            top_name = CONDITION_NAMES[top_idx]
            top_conf = last["path_probs"][top_idx]
            healthy_score = max(0.0, 1.0 - float(np.max(last["path_probs"][1:])))
            fog_s = last["fog_seconds"]
        else:
            top_name, top_conf, healthy_score, fog_s = "—", 0.0, 1.0, None

        c1.metric("Session time", f"{st.session_state.session_t:0.1f} s")
        c2.metric("Dominant signal", top_name, f"{top_conf*100:0.0f}% conf." if log else None)
        c3.metric("Healthy-pattern score", f"{healthy_score*100:0.0f}%")
        c4.metric("Est. time-to-freeze", f"{fog_s:0.1f} s" if fog_s is not None else "n/a")

    tail_n = FS * (win_s if win_s else 30)
    sig = ext.get_filtered_tail(tail_n) if ext else np.zeros((N_CHANNELS, 0))
    fig_emg = go.Figure()
    if sig.shape[1] > 0:
        tvec = (np.arange(sig.shape[1]) - sig.shape[1]) / FS + st.session_state.session_t
        for ch in range(N_CHANNELS):
            fig_emg.add_trace(go.Scatter(
                x=tvec, y=sig[ch] + ch * 3, mode="lines", name=f"Ch {ch+1}",
                line=dict(width=1.2, color=PLOTLY_COLORS[ch % len(PLOTLY_COLORS)]),
            ))
    fig_emg.update_layout(
        title="Live EMG — noise removed (10-200Hz band-pass + 50Hz notch)",
        height=380, **CHART_TEMPLATE,
        xaxis_title="time (s)", yaxis_title="channel (offset)",
        legend=dict(orientation="h", y=-0.2),
        margin=dict(l=40, r=20, t=50, b=40),
    )
    with emg_chart_ph.container():
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.plotly_chart(fig_emg, width="stretch", key="emg_chart")
        st.markdown('</div>', unsafe_allow_html=True)

    if log:
        df = pd.DataFrame({
            "t": [r["t"] for r in log],
            **{CONDITION_NAMES[i]: [r["path_probs"][i] for r in log] for i in range(1, 7)},
        })
        if win_s:
            df = df[df["t"] >= df["t"].max() - win_s]
        fig_pred = go.Figure()
        for i, cname in enumerate(CONDITION_NAMES[1:]):
            fig_pred.add_trace(go.Scatter(
                x=df["t"], y=df[cname], mode="lines", name=cname,
                line=dict(width=2, color=PLOTLY_COLORS[i % len(PLOTLY_COLORS)]),
            ))
        fig_pred.update_layout(
            title=f"Pathology-condition probability over time ({st.session_state.range_choice})",
            height=340, **CHART_TEMPLATE,
            xaxis_title="time (s)", yaxis_title="probability", yaxis_range=[0, 1],
            legend=dict(orientation="h", y=-0.25),
            margin=dict(l=40, r=20, t=50, b=40),
        )
        with pred_chart_ph.container():
            st.markdown('<div class="panel">', unsafe_allow_html=True)
            st.plotly_chart(fig_pred, width="stretch", key="pred_chart")
            st.markdown('</div>', unsafe_allow_html=True)

        fog_vals = [r["fog_seconds"] for r in log]
        latest_fog = fog_vals[-1]
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=float(np.clip(latest_fog, 0, TTF_MAX)),
            title={"text": "Estimated time-to-freeze (s)"},
            gauge={
                "axis": {"range": [0, TTF_MAX]},
                "bar": {"color": "#4f46e5"},
                "steps": [
                    {"range": [0, 3], "color": "#fecaca"},
                    {"range": [3, 6], "color": "#fde68a"},
                    {"range": [6, TTF_MAX], "color": "#bbf7d0"},
                ],
            },
        ))
        gauge.update_layout(height=280, margin=dict(l=20, r=20, t=50, b=10))

        fog_df = pd.DataFrame({"t": [r["t"] for r in log], "ttf": fog_vals})
        if win_s:
            fog_df = fog_df[fog_df["t"] >= fog_df["t"].max() - win_s]
        fog_trend = go.Figure()
        fog_trend.add_trace(go.Scatter(
            x=fog_df["t"], y=fog_df["ttf"], mode="lines",
            line=dict(width=2, color="#4f46e5"), fill="tozeroy",
            fillcolor="rgba(79,70,229,0.12)",
        ))
        fog_trend.update_layout(
            title="Time-to-freeze trend", height=280, **CHART_TEMPLATE,
            xaxis_title="time (s)", yaxis_title="seconds", yaxis_range=[0, TTF_MAX],
            margin=dict(l=40, r=20, t=50, b=40),
        )

        with fog_chart_ph.container():
            st.markdown('<div class="panel">', unsafe_allow_html=True)
            gcol, tcol = st.columns([1, 1.4])
            gcol.plotly_chart(gauge, width="stretch", key="fog_gauge")
            tcol.plotly_chart(fog_trend, width="stretch", key="fog_trend")
            st.markdown('</div>', unsafe_allow_html=True)

        with table_ph.container():
            st.markdown("##### Latest prediction snapshot")
            snap = pd.DataFrame({
                "Condition": CONDITION_NAMES[1:],
                "Probability": [f"{p*100:0.1f}%" for p in log[-1]["path_probs"][1:]],
            })
            st.dataframe(snap, width="stretch", hide_index=True, key="snap_table")
    else:
        # Guarantee initial run populates placeholders to prevent StreamlitAPIException
        with pred_chart_ph.container():
            st.info("Waiting for enough signal (needs 7s of context) before predictions begin...")
        
        with fog_chart_ph.container():
            st.empty()

        with table_ph.container():
            st.empty()


# ─────────────────────────────────────────────────────────────────────────
# LIVE TICK — FRAGMENT
# ─────────────────────────────────────────────────────────────────────────
def _live_tick():
    if not st.session_state.running or model is None:
        render_dashboard()
        return

    chunk = None
    if source == "Synthetic Demo" or st.session_state.example_active:
        chunk_samples = max(1, int(FS * (DEFAULT_CHUNK_MS / 1000.0) * DEFAULT_SPEED))
        chunk = st.session_state.streamer.generate_chunk(chunk_samples)
    else:  # Arduino Cloud
        client = st.session_state.arduino_client
        try:
            raw_val = client.get_property_value(ard_thing_id, ard_var_name)
            chunk = decode_emg_batch(raw_val, n_channels=N_CHANNELS)
            st.session_state.arduino_status = "Connected"
            st.session_state.last_poll_time = time.strftime("%H:%M:%S")

            if chunk is None:
                status_placeholder.markdown(
                    '<span class="badge badge-iot">ARDUINO CLOUD</span>&nbsp;'
                    '<span class="badge badge-paused">waiting for first batch…</span>',
                    unsafe_allow_html=True,
                )
                render_dashboard()
                return
        except ArduinoCloudError as e:
            status_placeholder.markdown(f'<span class="badge badge-error">{e}</span>', unsafe_allow_html=True)
            st.session_state.running = False
            st.session_state.arduino_status = "Disconnected"
            render_dashboard()
            return

    new_slices = st.session_state.extractor.push_samples(chunk)
    st.session_state.session_t += chunk.shape[1] / FS

    if new_slices and st.session_state.extractor.ready_for_inference():
        ctx = st.session_state.extractor.get_context_window()

        if st.session_state.norm_mean is not None:
            ctx_n = (ctx - st.session_state.norm_mean) / st.session_state.norm_std
        else:
            if st.session_state.normalizer is None:
                st.session_state.normalizer = RunningNormalizer(ctx.shape[1])
            for row in ctx:
                st.session_state.normalizer.update(row)
            ctx_n = st.session_state.normalizer.normalize(ctx)

        try:
            preds = model.predict(ctx_n[np.newaxis, ...], verbose=0)
            path_probs = np.asarray(preds[0]).flatten()
            fog_seconds = float(np.asarray(preds[1]).flatten()[0]) if len(preds) > 1 else None
        except Exception as e:
            status_placeholder.error(f"Prediction error: {e}")
            st.session_state.running = False
            render_dashboard()
            return

        st.session_state.pred_log.append(dict(
            t=st.session_state.session_t, path_probs=path_probs, fog_seconds=fog_seconds,
        ))
        if len(st.session_state.pred_log) > MAX_LOG_LEN:
            st.session_state.pred_log = st.session_state.pred_log[-MAX_LOG_LEN:]

    render_dashboard()
    if source == "Arduino Cloud (IoT)":
        badge = '<span class="badge badge-iot">ARDUINO CLOUD</span>'
    elif st.session_state.example_active:
        badge = '<span class="badge badge-demo">EXAMPLE</span>'
    else:
        badge = '<span class="badge badge-live">LIVE</span>'

    status_placeholder.markdown(
        f'{badge}&nbsp;&nbsp;t = {st.session_state.session_t:0.1f}s', unsafe_allow_html=True,
    )

    if (st.session_state.example_active and st.session_state.example_stop_at is not None
            and st.session_state.session_t >= st.session_state.example_stop_at):
        st.session_state.running = False
        st.session_state.example_active = False
        status_placeholder.info("Example run complete — 15s synthetic FOG demo finished.")


tick_interval = ard_poll_s if source == "Arduino Cloud (IoT)" else (DEFAULT_CHUNK_MS / 1000.0)
tick_interval = max(0.3, tick_interval)
st.fragment(run_every=tick_interval)(_live_tick)()