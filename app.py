"""
NeuroGait — Gait & Freeze Monitor — Streamlit App
==================================================
Patient-facing app. Two ways to see data:
  - Live: reads your connected monitor (via Arduino IoT Cloud in the
    background — no setup details are ever shown in the UI).
  - Example: a fixed, 30-second sample reading with a brief freeze
    episode in the middle, so you can see what the monitor looks like
    without a device connected.

There are no file uploads, model selection, or technical configuration
in the UI — the model and (optional) normalization stats are bundled
with the app and loaded automatically. Device credentials live in
st.secrets, never in the interface.

Run locally:
    streamlit run app.py

Deploy: push this folder (app.py, signal_utils.py, arduino_cloud.py,
requirements.txt, fog_pathology_model.h5) to a GitHub repo and deploy on
Streamlit Community Cloud (share.streamlit.io) or any host that runs
Streamlit. Put device credentials in `.streamlit/secrets.toml` under an
`[arduino_cloud]` table (client_id, client_secret, thing_id, var_name).
"""

import os
import hashlib

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from signal_utils import (
    FS, N_CHANNELS, WINDOW_SLICES, TTF_MAX, CONDITION_NAMES,
    StreamingFeatureExtractor, RunningNormalizer, SyntheticEMGStreamer,
)
from arduino_cloud import ArduinoCloudClient, ArduinoCloudError, decode_emg_batch

MAX_LOG_LEN = 20000  # cap prediction history so "Full session" can't grow unbounded

# Fixed, non-configurable timing (no playback/tech controls in the UI).
LIVE_POLL_S = 0.5
EXAMPLE_CHUNK_MS = 150
EXAMPLE_DURATION_S = 30.0
EXAMPLE_FOG_START_S, EXAMPLE_FOG_END_S = 12.0, 18.0

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
    page_title="NeuroGait Monitor",
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

/* Images never carry an opaque box behind them */
[data-testid="stImage"] { background: transparent !important; }

/* App header */
.app-header { display:flex; align-items:center; gap:14px; padding: 4px 0 18px 0; }
.app-header .kicker {
    color: var(--accent); font-weight:700; font-size:0.78em;
    letter-spacing:0.08em; text-transform:uppercase; margin:0;
}
.app-header h1 { margin:0; font-size:1.6em; }

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
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
    color: #e7eaf6 !important;
}
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
div.stButton > button:hover, div.stDownloadButton > button:hover { background:#4338ca; transform: translateY(-1px); }
div.stButton > button:active, div.stDownloadButton > button:active { transform: translateY(0); }
div.stButton > button:disabled { background:#c7cbe0; color:#7a819c; box-shadow:none; }

/* Metric cards */
[data-testid="stMetric"] {
    background: var(--surface); border: 1px solid var(--border); border-radius: 14px;
    padding: 14px 18px; box-shadow: 0 1px 3px rgba(15,23,42,0.04);
}
[data-testid="stMetricLabel"] { color: var(--ink-soft) !important; font-weight:600; }
[data-testid="stMetricValue"] { color: var(--ink) !important; }

/* Login card — sleek, tight, no dead space above the logo */
.login-box {
    max-width: 400px; margin: 9vh auto 0 auto; background: var(--surface);
    padding: 1.6em 2.4em 2em 2.4em; border-radius: 20px;
    box-shadow: 0 24px 60px rgba(15,23,42,0.14);
    border-top: 4px solid var(--accent);
}
.login-box [data-testid="stImage"] {
    display:flex; justify-content:center; margin: 0 0 0.5em 0;
}
.login-box [data-testid="stImage"] img { border-radius: 14px; }
.login-title { text-align:center; color: var(--ink); margin: 0 0 0.15em 0; font-size:1.3em; }
.login-sub { text-align:center; color: var(--ink-soft); margin-bottom:1.5em; font-size:0.9em; }

/* Status badges */
.badge { display:inline-block; padding:4px 12px; border-radius:20px; font-size:0.78em; font-weight:700; letter-spacing:0.02em; }
.badge-live   { background:#dcfce7; color:#15803d; }
.badge-paused { background:#f1f5f9; color:#475569; }
.badge-demo   { background:#92400e; color:#ffffff; }
.badge-error  { background:#fee2e2; color:#b91c1c; }

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
        st.image(LOGO_PATH, width=76)
        st.markdown('<h2 class="login-title">NeuroGait Monitor</h2>', unsafe_allow_html=True)
    else:
        st.markdown('<h2 class="login-title">🧠 NeuroGait Monitor</h2>', unsafe_allow_html=True)
    st.markdown('<div class="login-sub">Sign in to see your monitor</div>', unsafe_allow_html=True)

    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Sign In", use_container_width=True)

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
# MODEL + NORMALIZATION — bundled with the app, loaded automatically.
# No file uploads or paths are ever shown in the UI.
# ─────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Getting your monitor ready...")
def load_model_cached(_path: str, content_hash: str):
    import tensorflow as tf
    return tf.keras.models.load_model(_path, compile=False)


def find_default_model_path():
    for name in ["fog_pathology_model.h5", "model.h5"]:
        if os.path.exists(name):
            return name
    return None


def find_default_norm_path():
    for name in ["norm_stats.npz"]:
        if os.path.exists(name):
            return name
    return None


@st.cache_data(show_spinner=False)
def load_norm_stats_cached(path: str, mtime: float):
    data = np.load(path)
    return data["mean"], data["std"]


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
# DEVICE CONFIG — read only from st.secrets, never entered in the UI.
# ─────────────────────────────────────────────────────────────────────────
def get_device_config():
    try:
        cfg = st.secrets["arduino_cloud"]
        return (
            cfg.get("client_id", ""),
            cfg.get("client_secret", ""),
            cfg.get("thing_id", ""),
            cfg.get("var_name", "emgBatch"),
        )
    except Exception:
        return "", "", "", "emgBatch"

DEV_CLIENT_ID, DEV_CLIENT_SECRET, DEV_THING_ID, DEV_VAR_NAME = get_device_config()
DEVICE_READY = bool(DEV_CLIENT_ID and DEV_CLIENT_SECRET and DEV_THING_ID)

# ─────────────────────────────────────────────────────────────────────────
# REPORT CARD
# ─────────────────────────────────────────────────────────────────────────
def build_report_card(log, session_t, example_mode):
    lines = []
    lines.append("NeuroGait Monitor — Session Report Card")
    lines.append("=" * 40)
    lines.append(f"Session type: {'Example' if example_mode else 'Live'}")
    lines.append(f"Duration: {session_t:0.1f} seconds")
    if not log:
        lines.append("")
        lines.append("No readings were recorded during this session.")
        return "\n".join(lines)

    path_probs = np.array([r["path_probs"] for r in log])
    healthy_scores = np.clip(1.0 - path_probs[:, 1:].max(axis=1), 0.0, 1.0)
    avg_healthy = float(healthy_scores.mean())
    mean_by_condition = path_probs[:, 1:].mean(axis=0)
    top_idx = int(np.argmax(mean_by_condition)) + 1
    fog_vals = [r["fog_seconds"] for r in log if r["fog_seconds"] is not None]
    low_fog_count = sum(1 for v in fog_vals if v < 3.0)

    lines.append(f"Readings recorded: {len(log)}")
    lines.append(f"Average steadiness score: {avg_healthy*100:0.0f}%")
    lines.append(f"Most noticeable pattern: {CONDITION_NAMES[top_idx]}")
    if fog_vals:
        lines.append(f"Times a short freeze warning appeared: {low_fog_count}")
    lines.append("")
    lines.append("This report is a summary for your own reference and does not")
    lines.append("replace advice from your care team.")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────
# SIDEBAR — LIVE / EXAMPLE + REPORT CARD
# ─────────────────────────────────────────────────────────────────────────
with st.sidebar:
    if LOGO_PATH:
        st.image(LOGO_PATH, width=64)
    st.markdown(f"### Welcome, {st.session_state.username}")
    if st.button("Log out", use_container_width=True):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()

    st.markdown('<div class="sidebar-section-title">Live monitoring</div>', unsafe_allow_html=True)
    if not DEVICE_READY:
        st.caption("Your monitor isn't connected yet.")
    col_a, col_b = st.columns(2)
    start_clicked = col_a.button("▶ Start", use_container_width=True, disabled=not DEVICE_READY)
    stop_clicked = col_b.button("⏸ Stop", use_container_width=True)
    reset_clicked = st.button("↺ Reset session", use_container_width=True)

    st.markdown("---")
    example_clicked = st.button(
        "🧪 Try Example", use_container_width=True,
        help="A 30-second sample reading with a brief freeze episode in the middle.",
    )

    st.markdown('<div class="sidebar-section-title">Report</div>', unsafe_allow_html=True)
    _report_text = build_report_card(
        st.session_state.pred_log, st.session_state.session_t, st.session_state.example_active,
    )
    st.download_button(
        "⬇ Download report card", data=_report_text, file_name="gait_report_card.txt",
        mime="text/plain", use_container_width=True,
        disabled=len(st.session_state.pred_log) == 0,
    )

# ─────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────
if LOGO_PATH:
    hcol1, hcol2 = st.columns([1, 12])
    with hcol1:
        st.image(LOGO_PATH, width=52)
    with hcol2:
        st.markdown(
            '<div class="app-header"><div><p class="kicker">Real-time gait monitoring</p>'
            '<h1>NeuroGait — Gait &amp; Freeze Monitor</h1></div></div>',
            unsafe_allow_html=True,
        )
else:
    st.markdown(
        '<div class="app-header"><div><p class="kicker">Real-time gait monitoring</p>'
        '<h1>🧠 NeuroGait — Gait &amp; Freeze Monitor</h1></div></div>',
        unsafe_allow_html=True,
    )
status_placeholder = st.empty()

# ─────────────────────────────────────────────────────────────────────────
# RESET / START / STOP / EXAMPLE HANDLING
# ─────────────────────────────────────────────────────────────────────────
if reset_clicked:
    st.session_state.extractor = StreamingFeatureExtractor()
    st.session_state.normalizer = None
    st.session_state.pred_log = []
    st.session_state.session_t = 0.0
    st.session_state.streamer = None
    st.session_state.running = False
    st.session_state.example_active = False
    st.session_state.example_stop_at = None

if start_clicked and DEVICE_READY:
    st.session_state.running = True
    st.session_state.example_active = False
    st.session_state.example_stop_at = None
    if st.session_state.extractor is None:
        st.session_state.extractor = StreamingFeatureExtractor()
    st.session_state.streamer = None
    st.session_state.arduino_client = ArduinoCloudClient(DEV_CLIENT_ID, DEV_CLIENT_SECRET)

if stop_clicked:
    st.session_state.running = False
    st.session_state.example_active = False
    st.session_state.example_stop_at = None

if example_clicked:
    st.session_state.extractor = StreamingFeatureExtractor()
    st.session_state.normalizer = None
    st.session_state.pred_log = []
    st.session_state.session_t = 0.0
    st.session_state.streamer = SyntheticEMGStreamer(
        seed=42, freeze_window=(EXAMPLE_FOG_START_S, EXAMPLE_FOG_END_S),
    )
    st.session_state.running = True
    st.session_state.example_active = True
    st.session_state.example_stop_at = EXAMPLE_DURATION_S

# ─────────────────────────────────────────────────────────────────────────
# LOAD MODEL (bundled, automatic — no UI exposure)
# ─────────────────────────────────────────────────────────────────────────
model = None
model_err = None
default_model_path = find_default_model_path()
try:
    if default_model_path:
        content_hash = hashlib.md5(default_model_path.encode()).hexdigest()
        model = load_model_cached(default_model_path, content_hash)
    else:
        model_err = "Your monitor is still being set up. Please check back soon."
except Exception:
    model_err = "We're having trouble getting your monitor ready. Please try again shortly."

norm_path = find_default_norm_path()
if norm_path:
    nm, ns = load_norm_stats_cached(norm_path, os.path.getmtime(norm_path))
    st.session_state.norm_mean, st.session_state.norm_std = nm, ns

if model_err:
    status_placeholder.warning(model_err)
else:
    run_badge = (
        '<span class="badge badge-live">LIVE</span>' if st.session_state.running and not st.session_state.example_active
        else '<span class="badge badge-demo">EXAMPLE</span>' if st.session_state.running and st.session_state.example_active
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
        c3.metric("Steadiness score", f"{healthy_score*100:0.0f}%")
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
        title="Live signal — noise removed",
        height=380, **CHART_TEMPLATE,
        xaxis_title="time (s)", yaxis_title="channel (offset)",
        legend=dict(orientation="h", y=-0.2),
        margin=dict(l=40, r=20, t=50, b=40),
    )
    with emg_chart_ph.container():
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.plotly_chart(fig_emg, use_container_width=True, key="emg_chart")
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
            title=f"Pattern probability over time ({st.session_state.range_choice})",
            height=340, **CHART_TEMPLATE,
            xaxis_title="time (s)", yaxis_title="probability", yaxis_range=[0, 1],
            legend=dict(orientation="h", y=-0.25),
            margin=dict(l=40, r=20, t=50, b=40),
        )
        with pred_chart_ph.container():
            st.markdown('<div class="panel">', unsafe_allow_html=True)
            st.plotly_chart(fig_pred, use_container_width=True, key="pred_chart")
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
            gcol.plotly_chart(gauge, use_container_width=True, key="fog_gauge")
            tcol.plotly_chart(fog_trend, use_container_width=True, key="fog_trend")
            st.markdown('</div>', unsafe_allow_html=True)

        with table_ph.container():
            st.markdown("##### Latest reading")
            snap = pd.DataFrame({
                "Pattern": CONDITION_NAMES[1:],
                "Likelihood": [f"{p*100:0.1f}%" for p in log[-1]["path_probs"][1:]],
            })
            st.dataframe(snap, use_container_width=True, hide_index=True, key="snap_table")
    else:
        pred_chart_ph.info("Waiting for enough signal (about 7 seconds) before readings begin...")


# ─────────────────────────────────────────────────────────────────────────
# LIVE TICK — one polling/inference/redraw step, run on a timer via
# st.fragment. Non-blocking: Stop takes effect immediately, and only this
# fragment's slice of the page re-renders each tick.
# ─────────────────────────────────────────────────────────────────────────
def _live_tick():
    if not st.session_state.running or model is None:
        render_dashboard()
        return

    chunk = None
    if st.session_state.example_active:
        chunk_samples = max(1, int(FS * (EXAMPLE_CHUNK_MS / 1000.0)))
        chunk = st.session_state.streamer.generate_chunk(chunk_samples)
    else:
        client = st.session_state.arduino_client
        try:
            raw_val = client.get_property_value(DEV_THING_ID, DEV_VAR_NAME)
            chunk = decode_emg_batch(raw_val, n_channels=N_CHANNELS)
            if chunk is None:
                status_placeholder.markdown(
                    '<span class="badge badge-paused">Waiting for your monitor…</span>',
                    unsafe_allow_html=True,
                )
                render_dashboard()
                return
        except ArduinoCloudError:
            status_placeholder.markdown(
                '<span class="badge badge-error">We couldn\'t reach your monitor. Please try again.</span>',
                unsafe_allow_html=True,
            )
            st.session_state.running = False
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
            status_placeholder.error("Something went wrong while reading the signal. Please try again.")
            st.session_state.running = False
            render_dashboard()
            return

        st.session_state.pred_log.append(dict(
            t=st.session_state.session_t, path_probs=path_probs, fog_seconds=fog_seconds,
        ))
        if len(st.session_state.pred_log) > MAX_LOG_LEN:
            st.session_state.pred_log = st.session_state.pred_log[-MAX_LOG_LEN:]

    render_dashboard()
    if st.session_state.example_active:
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
        status_placeholder.info("Example finished — that's what a brief freeze episode looks like.")


tick_interval = EXAMPLE_CHUNK_MS / 1000.0 if st.session_state.example_active else LIVE_POLL_S
tick_interval = max(0.15, tick_interval)  # floor so the UI thread always stays responsive
st.fragment(run_every=tick_interval)(_live_tick)()
