import os
import time
import hashlib
from io import BytesIO

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from signal_utils import (
    FS, N_CHANNELS, STRIDE, WINDOW_SLICES, TTF_MAX, CONDITION_NAMES,
    StreamingFeatureExtractor, RunningNormalizer, SyntheticEMGStreamer,
)
from arduino_cloud import ArduinoCloudClient, ArduinoCloudError, decode_emg_batch

MAX_LOG_LEN = 20000

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
    page_title="NeuroGait EMG Monitor",
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
            st.image(LOGO_PATH, use_container_width=True)
        st.markdown('<h2 class="login-title">NeuroGait EMG Monitor</h2>', unsafe_allow_html=True)
    else:
        st.markdown('<h2 class="login-title">🧠 NeuroGait EMG Monitor</h2>', unsafe_allow_html=True)
    st.markdown('<div class="login-sub">Sign in to access the live EMG dashboard</div>', unsafe_allow_html=True)

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
# MODEL LOADING — cache keyed on content hash, not path, so re-uploading a
# different file actually invalidates the cache (the old version cached by
# tmp filename, which never changed between uploads — a real bug, fixed here).
# ─────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model...")
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


@st.cache_data(show_spinner=False)
def load_csv_cached(content_hash: str, raw_bytes: bytes):
    return pd.read_csv(BytesIO(raw_bytes))


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
        csv_cursor=0,
        arduino_client=None,
        arduino_error=None,
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
# SIDEBAR — DATA SOURCE + MODEL + CONTROLS
# ─────────────────────────────────────────────────────────────────────────
def get_arduino_credentials():
    """Prefer st.secrets (for deployed apps); sidebar fields override/fill gaps."""
    cid, csec = "", ""
    try:
        if "arduino_cloud" in st.secrets:
            cid = st.secrets["arduino_cloud"].get("client_id", "")
            csec = st.secrets["arduino_cloud"].get("client_secret", "")
    except Exception:
        pass
    return cid, csec

with st.sidebar:
    if LOGO_PATH:
        st.image(LOGO_PATH, use_container_width=True)
        st.markdown(f"### Welcome, {st.session_state.username}")
    else:
        st.markdown(f"### 🧠 Welcome, {st.session_state.username}")
    if st.button("Log out", use_container_width=True):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()

    st.markdown('<div class="sidebar-section-title">Model</div>', unsafe_allow_html=True)
    default_path = find_default_model_path()
    model_file = st.file_uploader("Upload .h5 model (optional)", type=["h5"])
    model_path_text = st.text_input(
        "...or path to model on disk",
        value=default_path or "fog_pathology_model.h5",
    )
    norm_file = st.file_uploader("Normalization stats (.npz, optional)", type=["npz"])

    st.markdown('<div class="sidebar-section-title">Data source</div>', unsafe_allow_html=True)
    source = st.radio(
        "Choose EMG input",
        ["Live Demo (synthetic)", "Upload CSV recording", "Arduino Cloud (IoT)"],
        label_visibility="collapsed",
    )

    # Always defined so later code never hits a NameError.
    scenario, inject_freeze, freeze_at, seed = CONDITION_NAMES[0], False, None, 42
    uploaded_csv, csv_fs = None, FS
    ard_client_id, ard_client_secret, ard_thing_id, ard_var_name, ard_poll_s = (
        "", "", "", "emgBatch", 0.5
    )

    if source == "Live Demo (synthetic)":
        scenario = st.selectbox("Simulated condition", CONDITION_NAMES)
        if scenario == "Parkinson's / FOG risk":
            inject_freeze = st.checkbox("Simulate a freeze-of-gait event", value=True)
            if inject_freeze:
                freeze_at = st.slider("Freeze occurs at (s)", 10, 60, 25)
        seed = st.number_input("Random seed", value=42, step=1)

    elif source == "Upload CSV recording":
        uploaded_csv = st.file_uploader("EMG CSV (>=8 numeric channel columns)", type=["csv"])
        st.caption(f"First 8 numeric columns are used as channels 1-8. If the CSV wasn't "
                   f"recorded at {FS} Hz, set its real rate below and it'll be resampled.")
        csv_fs = st.number_input("Sample rate of CSV (Hz)", value=FS, step=50)

    else:  # Arduino Cloud (IoT)
        secret_cid, secret_csec = get_arduino_credentials()
        st.caption("Reads live 8-channel batches pushed by your Arduino board via "
                   "Arduino IoT Cloud. See the setup guide below the dashboard.")
        ard_client_id = st.text_input(
            "Client ID", value=secret_cid,
            help="From Arduino Cloud → API Keys. Leave blank if set in st.secrets.",
        )
        ard_client_secret = st.text_input(
            "Client Secret", value=secret_csec, type="password",
            help="Shown once when the API key is created — store it in st.secrets for deployed apps.",
        )
        ard_thing_id = st.text_input("Thing ID", help="Found on the Thing's page in Arduino Cloud.")
        ard_var_name = st.text_input(
            "Batch variable name", value="emgBatch",
            help="The String Cloud Variable your sketch writes JSON batches into.",
        )
        ard_poll_s = st.slider(
            "Poll interval (s)", 0.2, 5.0, 0.5, step=0.1,
            help="How often to pull the latest batch from Arduino Cloud's REST API. "
                 "Keep this ≥0.2s to stay comfortably within API rate limits.",
        )

    st.markdown('<div class="sidebar-section-title">Playback</div>', unsafe_allow_html=True)
    speed = st.slider("Playback speed multiplier", 0.5, 10.0, 3.0, step=0.5,
                       disabled=(source == "Arduino Cloud (IoT)"))
    chunk_ms = st.slider("Update interval (ms)", 50, 500, 150, step=50,
                          disabled=(source == "Arduino Cloud (IoT)"))

    col_a, col_b = st.columns(2)
    start_clicked = col_a.button("▶ Start", use_container_width=True)
    stop_clicked = col_b.button("⏸ Stop", use_container_width=True)
    reset_clicked = st.button("↺ Reset session", use_container_width=True)

    st.markdown("---")
    example_clicked = st.button(
        "🧪 Run Example (FOG, 15s)", use_container_width=True,
        help="Synthetic 15s demo, Parkinson's/FOG scenario with a freeze event, light noise.",
    )

# ─────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────
if LOGO_PATH:
    hcol1, hcol2 = st.columns([1, 10])
    with hcol1:
        st.image(LOGO_PATH, use_container_width=True)
    with hcol2:
        st.markdown(
            '<div class="app-header"><div><p class="kicker">Real-time neuromuscular monitoring</p>'
            '<h1>NeuroGait — EMG Pathology &amp; Freeze-of-Gait Monitor</h1></div></div>',
            unsafe_allow_html=True,
        )
else:
    st.markdown(
        '<div class="app-header"><div><p class="kicker">Real-time neuromuscular monitoring</p>'
        '<h1>🧠 NeuroGait — EMG Pathology &amp; Freeze-of-Gait Monitor</h1></div></div>',
        unsafe_allow_html=True,
    )
status_placeholder = st.empty()

# ─────────────────────────────────────────────────────────────────────────
# RESET / START / STOP HANDLING
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
    st.session_state.csv_cursor = 0

if start_clicked:
    st.session_state.running = True
    st.session_state.example_active = False
    st.session_state.example_stop_at = None
    if st.session_state.extractor is None:
        st.session_state.extractor = StreamingFeatureExtractor()
    if source == "Live Demo (synthetic)":
        use_freeze = scenario == "Parkinson's / FOG risk" and inject_freeze
        st.session_state.streamer = SyntheticEMGStreamer(
            scenario=scenario, seed=int(seed),
            freeze_at_s=freeze_at if use_freeze else None,
        )
    elif source == "Arduino Cloud (IoT)":
        st.session_state.streamer = None
        if not (ard_client_id and ard_client_secret and ard_thing_id):
            status_placeholder.error("Arduino Cloud needs Client ID, Client Secret, and Thing ID.")
            st.session_state.running = False
        else:
            st.session_state.arduino_client = ArduinoCloudClient(ard_client_id, ard_client_secret)
    else:
        st.session_state.streamer = None  # CSV handled separately below

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
        scenario="Parkinson's / FOG risk", seed=42, freeze_at_s=8.0,
    )
    st.session_state.running = True
    st.session_state.example_active = True
    st.session_state.example_stop_at = 15.0

# ─────────────────────────────────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────────────────────────────────
model = None
model_err = None
try:
    if model_file is not None:
        raw = model_file.getbuffer()
        content_hash = hashlib.md5(raw).hexdigest()
        tmp_path = f"/tmp/_uploaded_model_{content_hash}.h5"
        if not os.path.exists(tmp_path):
            with open(tmp_path, "wb") as f:
                f.write(raw)
        model = load_model_cached(tmp_path, content_hash)
    elif model_path_text and os.path.exists(model_path_text):
        content_hash = hashlib.md5(model_path_text.encode()).hexdigest()
        model = load_model_cached(model_path_text, content_hash)
    else:
        model_err = ("No model found yet. Upload a `.h5` file in the sidebar, or place "
                      "`fog_pathology_model.h5` next to app.py.")
except Exception as e:
    model_err = f"Could not load model: {e}"

if norm_file is not None:
    raw_norm = bytes(norm_file.getbuffer())
    norm_hash = hashlib.md5(raw_norm).hexdigest()
    nm, ns = load_norm_stats_cached(norm_hash, raw_norm)
    st.session_state.norm_mean, st.session_state.norm_std = nm, ns

if model_err:
    status_placeholder.warning(model_err)
else:
    run_badge = (
        '<span class="badge badge-live">LIVE</span>' if st.session_state.running
        else '<span class="badge badge-paused">PAUSED</span>'
    )
    status_placeholder.markdown(
        f'<span class="badge badge-demo">MODEL READY</span>&nbsp;{run_badge}',
        unsafe_allow_html=True,
    )

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
            title=f"Pathology-condition probability over time ({st.session_state.range_choice})",
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
            st.markdown("##### Latest prediction snapshot")
            snap = pd.DataFrame({
                "Condition": CONDITION_NAMES[1:],
                "Probability": [f"{p*100:0.1f}%" for p in log[-1]["path_probs"][1:]],
            })
            st.dataframe(snap, use_container_width=True, hide_index=True, key="snap_table")
    else:
        pred_chart_ph.info("Waiting for enough signal (needs 7s of context) before predictions begin...")
        # Claim these slots on the initial full run too (even with no predictions
        # yet), otherwise the fragment can't safely write into them later once
        # `log` becomes non-empty on a fragment-only rerun.
        fog_chart_ph.empty()
        table_ph.empty()


# ─────────────────────────────────────────────────────────────────────────
# CSV PLAYBACK SETUP (resampled to FS if the CSV's real rate differs — the
# old version accepted a csv_fs input but silently ignored it, a real bug)
# ─────────────────────────────────────────────────────────────────────────
csv_data = None
if source == "Upload CSV recording" and uploaded_csv is not None:
    raw_bytes = bytes(uploaded_csv.getbuffer())
    csv_hash = hashlib.md5(raw_bytes).hexdigest()
    raw_df = load_csv_cached(csv_hash, raw_bytes)
    numeric_cols = raw_df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) >= N_CHANNELS:
        csv_data = raw_df[numeric_cols[:N_CHANNELS]].to_numpy().T  # (8, N)
        if csv_fs and int(csv_fs) != FS:
            from scipy.signal import resample_poly
            from math import gcd
            g = gcd(int(FS), int(csv_fs))
            csv_data = resample_poly(csv_data, int(FS) // g, int(csv_fs) // g, axis=1)
            st.caption(f"Resampled from {int(csv_fs)} Hz to {FS} Hz.")
        st.caption(f"Loaded {csv_data.shape[1]} samples across {N_CHANNELS} channels "
                   f"(~{csv_data.shape[1]/FS:0.1f}s at {FS} Hz).")
    else:
        st.error(f"CSV needs at least {N_CHANNELS} numeric columns; found {len(numeric_cols)}.")

# ─────────────────────────────────────────────────────────────────────────
# LIVE TICK — one polling/inference/redraw step, run on a timer via
# st.fragment. This replaces the old blocking while-loop + time.sleep +
# st.rerun pattern: it's non-blocking, Stop takes effect immediately, and
# only this fragment's slice of the page re-renders each tick.
# ─────────────────────────────────────────────────────────────────────────
def _live_tick():
    if not st.session_state.running or model is None:
        render_dashboard()
        return

    chunk = None
    if st.session_state.example_active or source == "Live Demo (synthetic)":
        chunk_samples = max(1, int(FS * (chunk_ms / 1000.0) * speed))
        chunk = st.session_state.streamer.generate_chunk(chunk_samples)

    elif source == "Arduino Cloud (IoT)":
        client = st.session_state.arduino_client
        try:
            raw_val = client.get_property_value(ard_thing_id, ard_var_name)
            chunk = decode_emg_batch(raw_val, n_channels=N_CHANNELS)
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
            render_dashboard()
            return

    else:  # CSV
        if csv_data is None:
            status_placeholder.error("Upload a CSV recording to begin, or switch to Live Demo.")
            st.session_state.running = False
            render_dashboard()
            return
        chunk_samples = max(1, int(FS * (chunk_ms / 1000.0) * speed))
        cur = st.session_state.csv_cursor
        end = cur + chunk_samples
        if cur >= csv_data.shape[1]:
            status_placeholder.info("End of recording reached.")
            st.session_state.running = False
            render_dashboard()
            return
        chunk = csv_data[:, cur:end]
        st.session_state.csv_cursor = end

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
            status_placeholder.error(f"Inference error: {e}")
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


tick_interval = ard_poll_s if source == "Arduino Cloud (IoT)" else (chunk_ms / 1000.0 / max(speed, 0.1))
tick_interval = max(0.15, tick_interval)  # floor so the UI thread always stays responsive
st.fragment(run_every=tick_interval)(_live_tick)()