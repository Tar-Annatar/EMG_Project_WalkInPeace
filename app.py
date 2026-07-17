"""
EMG Pathology & Freeze-of-Gait Monitor — Streamlit App
=======================================================
- Simple username/password gate.
- Live (streamed / simulated) 8-channel EMG view with noise removed
  (band-pass + 50Hz notch, matching the training pipeline).
- Runs the trained Keras model (fog_pathology_model.h5) in real time and
  shows pathology-condition probabilities and Parkinson's freeze-risk
  countdown over multiple time ranges.

Run locally:
    streamlit run app.py

Deploy: push this folder (app.py, signal_utils.py, requirements.txt,
fog_pathology_model.h5) to a GitHub repo and deploy on Streamlit
Community Cloud (share.streamlit.io) or any host that runs Streamlit.
"""

import os
import time
import hashlib
from collections import deque

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from signal_utils import (
    FS, N_CHANNELS, STRIDE, WINDOW_SLICES, TTF_MAX, CONDITION_NAMES,
    StreamingFeatureExtractor, RunningNormalizer, SyntheticEMGStreamer,
)

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
# PAGE CONFIG + GREEN THEME
# ─────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="WalkInPeace EMG Monitor",
    page_icon=LOGO_PATH or "🟢",
    layout="wide",
    initial_sidebar_state="expanded",
)

GREEN_CSS = """
<style>
:root {
    --g-dark:   #0b3d2e;
    --g-mid:    #1b5e3a;
    --g-main:   #2e7d32;
    --g-accent: #52b788;
    --g-light:  #b7e4c7;
    --g-bg:     #f1faf3;
}
html, body, [class*="css"]  { font-family: 'Segoe UI', sans-serif; }
.stApp { background-color: var(--g-bg); }
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, var(--g-dark) 0%, var(--g-mid) 100%);
}
section[data-testid="stSidebar"] { color: #eafaf0; }
/* Text sitting directly on the dark sidebar gradient: keep it light */
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
    color: #eafaf0 !important;
}
/* Native Streamlit controls keep their own light surface (white/light-gray
   box) — force dark text on THOSE instead of inheriting the light color
   above, or the value becomes invisible (near-white text on a light box). */
section[data-testid="stSidebar"] input,
section[data-testid="stSidebar"] textarea,
section[data-testid="stSidebar"] [data-baseweb="select"] *,
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"],
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] *,
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzoneInstructions"] * {
    color: var(--g-dark) !important;
}
section[data-testid="stSidebar"] input::placeholder {
    color: #4b6f5f !important;
    opacity: 1;
}
h1, h2, h3 { color: var(--g-dark) !important; }
div.stButton > button, div.stDownloadButton > button {
    background-color: var(--g-main);
    color: white;
    border-radius: 8px;
    border: none;
    padding: 0.5em 1.2em;
    font-weight: 600;
}
div.stButton > button:hover { background-color: var(--g-dark); color: white; }
[data-testid="stMetric"] {
    background-color: white;
    border: 1px solid var(--g-light);
    border-radius: 10px;
    padding: 10px 14px;
    box-shadow: 0 1px 4px rgba(11,61,46,0.08);
}
[data-testid="stMetricLabel"] { color: var(--g-mid) !important; }
.login-box {
    max-width: 420px;
    margin: 8vh auto 0 auto;
    background: white;
    padding: 2.5em 2.5em 2em 2.5em;
    border-radius: 16px;
    box-shadow: 0 6px 24px rgba(11,61,46,0.15);
    border-top: 6px solid var(--g-main);
}
.login-title { text-align:center; color: var(--g-dark); margin-bottom:0.2em;}
.login-sub { text-align:center; color: var(--g-mid); margin-bottom:1.5em; font-size:0.9em;}
.badge-live {
    display:inline-block; background:var(--g-accent); color:white;
    padding:3px 10px; border-radius:12px; font-size:0.8em; font-weight:600;
}
</style>
"""
st.markdown(GREEN_CSS, unsafe_allow_html=True)

PLOTLY_GREENS = ["#1b5e3a", "#2e7d32", "#40916c", "#52b788", "#74c69d",
                  "#95d5b2", "#b7e4c7", "#d8f3dc"]

# ─────────────────────────────────────────────────────────────────────────
# AUTH
# ─────────────────────────────────────────────────────────────────────────
def _hash(pw: str) -> str:
    return hashlib.sha256(pw.encode()).hexdigest()

# Default demo credentials. For real deployments, replace this dict with
# values pulled from st.secrets["credentials"] instead of hardcoding.
DEFAULT_USERS = {
    "demo": _hash("emg2026"),
    "clinician": _hash("WalkInPeace"),
}

def get_user_db():
    try:
        if "credentials" in st.secrets:
            return {u: _hash(p) for u, p in st.secrets["credentials"].items()}
    except Exception:
        pass  # no secrets.toml configured -> fall back to demo credentials
    return DEFAULT_USERS


def login_page():
    st.markdown('<div class="login-box">', unsafe_allow_html=True)
    if LOGO_PATH:
        lcol1, lcol2, lcol3 = st.columns([1, 1, 1])
        with lcol2:
            st.image(LOGO_PATH, use_container_width=True)
        st.markdown('<h2 class="login-title">WalkInPeace EMG Monitor</h2>', unsafe_allow_html=True)
    else:
        st.markdown('<h2 class="login-title">🟢 WalkInPeace EMG Monitor</h2>', unsafe_allow_html=True)
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
# MODEL LOADING
# ─────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model...")
def load_model(model_bytes_or_path):
    import tensorflow as tf
    return tf.keras.models.load_model(model_bytes_or_path, compile=False)


def find_default_model_path():
    for name in ["fog_pathology_model.h5", "model.h5"]:
        if os.path.exists(name):
            return name
    return None


@st.cache_resource(show_spinner=False)
def load_norm_stats(path):
    if path and os.path.exists(path):
        data = np.load(path)
        return data["mean"], data["std"]
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
        pred_log=[],  # list of dicts: t, path_probs(7,), fog_seconds
        session_t=0.0,
        running=False,
        streamer=None,
        norm_mean=None,
        norm_std=None,
        example_active=False,
        example_stop_at=None,
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
with st.sidebar:
    if LOGO_PATH:
        st.image(LOGO_PATH, use_container_width=True)
        st.markdown(f"### Welcome, {st.session_state.username}")
    else:
        st.markdown(f"### 🟢 Welcome, {st.session_state.username}")
    if st.button("Log out"):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()

    st.markdown("---")
    st.markdown("#### Model")
    default_path = find_default_model_path()
    model_file = st.file_uploader("Upload .h5 model (optional)", type=["h5"])
    model_path_text = st.text_input(
        "...or path to model on disk",
        value=default_path or "fog_pathology_model.h5",
    )
    norm_file = st.file_uploader("Normalization stats (.npz, optional)", type=["npz"])

    st.markdown("---")
    st.markdown("#### Data source")
    source = st.radio("Choose EMG input", ["Live Demo (synthetic)", "Upload CSV recording"])

    # Always define these so later code (outside the sidebar) never hits a
    # NameError regardless of which branch was chosen.
    scenario, inject_freeze, freeze_at, seed = CONDITION_NAMES[0], False, None, 42
    uploaded_csv, csv_fs = None, FS

    if source == "Live Demo (synthetic)":
        scenario = st.selectbox("Simulated condition", CONDITION_NAMES)
        if scenario == "Parkinson's / FOG risk":
            inject_freeze = st.checkbox("Simulate a freeze-of-gait event", value=True)
            if inject_freeze:
                freeze_at = st.slider("Freeze occurs at (s)", 10, 60, 25)
        seed = st.number_input("Random seed", value=42, step=1)
    else:
        uploaded_csv = st.file_uploader("EMG CSV (>=8 numeric channel columns)", type=["csv"])
        st.caption("First 8 numeric columns are used as channels 1-8, assumed sampled at "
                   f"{FS} Hz unless a 'fs' is specified below.")
        csv_fs = st.number_input("Sample rate of CSV (Hz)", value=FS, step=50)

    st.markdown("---")
    st.markdown("#### Playback")
    speed = st.slider("Playback speed multiplier", 0.5, 10.0, 3.0, step=0.5)
    chunk_ms = st.slider("Update interval (ms)", 50, 500, 150, step=50)

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
        st.title("WalkInPeace — Live EMG Pathology & Freeze-of-Gait Monitor")
else:
    st.title("🟢 WalkInPeace — Live EMG Pathology & Freeze-of-Gait Monitor")
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
    else:
        st.session_state.streamer = None  # CSV handled separately below

if stop_clicked:
    st.session_state.running = False
    st.session_state.example_active = False
    st.session_state.example_stop_at = None

if example_clicked:
    # Self-contained 15s synthetic run: Parkinson's/FOG scenario with a
    # freeze event, fully synthetic with light noise — same generator as
    # "Live Demo", just pre-configured and time-boxed regardless of
    # whatever the sidebar's data-source radio is currently set to.
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
        tmp_path = "/tmp/_uploaded_model.h5"
        with open(tmp_path, "wb") as f:
            f.write(model_file.getbuffer())
        model = load_model(tmp_path)
    elif model_path_text and os.path.exists(model_path_text):
        model = load_model(model_path_text)
    else:
        model_err = ("No model found yet. Upload a `.h5` file in the sidebar, or place "
                      "`fog_pathology_model.h5` next to app.py.")
except Exception as e:
    model_err = f"Could not load model: {e}"

if norm_file is not None:
    tmp_norm = "/tmp/_uploaded_norm.npz"
    with open(tmp_norm, "wb") as f:
        f.write(norm_file.getbuffer())
    nm, ns = load_norm_stats(tmp_norm)
    st.session_state.norm_mean, st.session_state.norm_std = nm, ns

if model_err:
    status_placeholder.warning(model_err)
else:
    status_placeholder.markdown(
        f'<span class="badge-live">MODEL READY</span> &nbsp; '
        f'{"🟢 LIVE" if st.session_state.running else "⏸ Paused"}',
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


if "_render_seq" not in st.session_state:
    st.session_state._render_seq = 0


def render_dashboard():
    # Within a single "Start" click, the live loop redraws these placeholders
    # many times in the SAME script run (not via rerun), so every widget
    # needs a unique key per redraw or Streamlit raises a duplicate-ID error.
    st.session_state._render_seq += 1
    seq = st.session_state._render_seq

    ext = st.session_state.extractor
    log = st.session_state.pred_log
    win_s = range_seconds(st.session_state.range_choice)

    # ── Metrics row ──
    with metrics_row.container():
        c1, c2, c3, c4 = st.columns(4)
        if log:
            last = log[-1]
            top_idx = int(np.argmax(last["path_probs"][1:])) + 1  # skip healthy slot
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

    # ── Raw/filtered EMG plot ──
    tail_n = FS * (win_s if win_s else 30)
    sig = ext.get_filtered_tail(tail_n) if ext else np.zeros((N_CHANNELS, 0))
    fig_emg = go.Figure()
    if sig.shape[1] > 0:
        tvec = (np.arange(sig.shape[1]) - sig.shape[1]) / FS + st.session_state.session_t
        for ch in range(N_CHANNELS):
            fig_emg.add_trace(go.Scatter(
                x=tvec, y=sig[ch] + ch * 3,  # vertical offset per channel
                mode="lines", name=f"Ch {ch+1}",
                line=dict(width=1.2, color=PLOTLY_GREENS[ch % len(PLOTLY_GREENS)]),
            ))
    fig_emg.update_layout(
        title="Live EMG — noise removed (10-200Hz band-pass + 50Hz notch)",
        height=380, plot_bgcolor="#f6fbf8", paper_bgcolor="white",
        xaxis_title="time (s)", yaxis_title="channel (offset)",
        legend=dict(orientation="h", y=-0.2),
        margin=dict(l=40, r=20, t=50, b=40),
    )
    emg_chart_ph.plotly_chart(fig_emg, use_container_width=True, key=f"emg_chart_{seq}")

    # ── Prediction-over-time chart ──
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
                line=dict(width=2, color=PLOTLY_GREENS[i % len(PLOTLY_GREENS)]),
            ))
        fig_pred.update_layout(
            title=f"Pathology-condition probability over time ({st.session_state.range_choice})",
            height=340, plot_bgcolor="#f6fbf8", paper_bgcolor="white",
            xaxis_title="time (s)", yaxis_title="probability", yaxis_range=[0, 1],
            legend=dict(orientation="h", y=-0.25),
            margin=dict(l=40, r=20, t=50, b=40),
        )
        pred_chart_ph.plotly_chart(fig_pred, use_container_width=True, key=f"pred_chart_{seq}")

        # ── FOG time-to-freeze gauge + trend ──
        fog_vals = [r["fog_seconds"] for r in log]
        latest_fog = fog_vals[-1]
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=float(np.clip(latest_fog, 0, TTF_MAX)),
            title={"text": "Estimated time-to-freeze (s)"},
            gauge={
                "axis": {"range": [0, TTF_MAX]},
                "bar": {"color": "#1b5e3a"},
                "steps": [
                    {"range": [0, 3], "color": "#d9534f"},
                    {"range": [3, 6], "color": "#f0ad4e"},
                    {"range": [6, TTF_MAX], "color": "#b7e4c7"},
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
            line=dict(width=2, color="#2e7d32"), fill="tozeroy",
            fillcolor="rgba(82,183,136,0.25)",
        ))
        fog_trend.update_layout(
            title="Time-to-freeze trend", height=280,
            plot_bgcolor="#f6fbf8", paper_bgcolor="white",
            xaxis_title="time (s)", yaxis_title="seconds", yaxis_range=[0, TTF_MAX],
            margin=dict(l=40, r=20, t=50, b=40),
        )

        with fog_chart_ph.container():
            gcol, tcol = st.columns([1, 1.4])
            gcol.plotly_chart(gauge, use_container_width=True, key=f"fog_gauge_{seq}")
            tcol.plotly_chart(fog_trend, use_container_width=True, key=f"fog_trend_{seq}")

        with table_ph.container():
            st.markdown("##### Latest prediction snapshot")
            snap = pd.DataFrame({
                "Condition": CONDITION_NAMES[1:],
                "Probability": [f"{p*100:0.1f}%" for p in log[-1]["path_probs"][1:]],
            })
            st.dataframe(snap, use_container_width=True, hide_index=True, key=f"snap_table_{seq}")
    else:
        pred_chart_ph.info("Waiting for enough signal (needs 7s of context) before predictions begin...")


# ─────────────────────────────────────────────────────────────────────────
# CSV PLAYBACK SETUP (non-live source)
# ─────────────────────────────────────────────────────────────────────────
csv_data = None
if source == "Upload CSV recording" and uploaded_csv is not None:
    raw_df = pd.read_csv(uploaded_csv)
    numeric_cols = raw_df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) >= N_CHANNELS:
        csv_data = raw_df[numeric_cols[:N_CHANNELS]].to_numpy().T  # (8, N)
        st.caption(f"Loaded {csv_data.shape[1]} samples across {N_CHANNELS} channels "
                   f"({csv_data.shape[1]/csv_fs:0.1f}s at {csv_fs} Hz).")
    else:
        st.error(f"CSV needs at least {N_CHANNELS} numeric columns; found {len(numeric_cols)}.")

if "csv_cursor" not in st.session_state:
    st.session_state.csv_cursor = 0

# ─────────────────────────────────────────────────────────────────────────
# MAIN LIVE LOOP
# One click of "Start" runs a bounded loop that keeps updating the same
# placeholders in-place (classic lightweight Streamlit live-dashboard
# pattern). Click "Stop" or let it exhaust the CSV to end the loop.
# ─────────────────────────────────────────────────────────────────────────
if model is not None and st.session_state.running:
    chunk_samples = max(1, int(FS * (chunk_ms / 1000.0) * speed))
    loop_iters = 0
    max_iters = 100000  # safety cap

    while st.session_state.running and loop_iters < max_iters:
        loop_iters += 1

        # 1) get next raw chunk
        if st.session_state.example_active or source == "Live Demo (synthetic)":
            chunk = st.session_state.streamer.generate_chunk(chunk_samples)
        else:
            if csv_data is None:
                status_placeholder.error("Upload a CSV recording to begin, or switch to Live Demo.")
                st.session_state.running = False
                break
            cur = st.session_state.csv_cursor
            end = cur + chunk_samples
            if cur >= csv_data.shape[1]:
                status_placeholder.info("End of recording reached.")
                st.session_state.running = False
                break
            chunk = csv_data[:, cur:end]
            st.session_state.csv_cursor = end

        # 2) stream through feature extractor
        new_slices = st.session_state.extractor.push_samples(chunk)
        st.session_state.session_t += chunk.shape[1] / FS

        # 3) run inference whenever we have a full 7s context and a new slice arrived
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
                break

            st.session_state.pred_log.append(dict(
                t=st.session_state.session_t,
                path_probs=path_probs,
                fog_seconds=fog_seconds,
            ))

        # 4) redraw
        render_dashboard()
        badge_label = "🧪 EXAMPLE" if st.session_state.example_active else "🟢 LIVE"
        status_placeholder.markdown(
            f'<span class="badge-live">{badge_label}</span> &nbsp; t = {st.session_state.session_t:0.1f}s',
            unsafe_allow_html=True,
        )

        # Example runs are time-boxed to 15s — stop after the final frame draws.
        if (st.session_state.example_active and st.session_state.example_stop_at is not None
                and st.session_state.session_t >= st.session_state.example_stop_at):
            st.session_state.running = False
            st.session_state.example_active = False
            status_placeholder.info("Example run complete — 15s synthetic FOG demo finished.")
            break

        time.sleep(chunk_ms / 1000.0 / max(speed, 0.1))

    st.rerun()
else:
    render_dashboard()
