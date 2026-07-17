"""
signal_utils.py — matches Working_Notebook.ipynb training pipeline exactly
(filters, WIN_LEN/STRIDE, 72-dim slice_features, condition order).
"""
import numpy as np
from scipy.signal import butter, filtfilt, sosfilt, sosfilt_zi, iirnotch

FS = 500
N_CHANNELS = 8
WIN_LEN_S, STRIDE_S = 0.25, 0.10
WIN_LEN = int(WIN_LEN_S * FS)      # 125
STRIDE = int(STRIDE_S * FS)        # 50
WINDOW_SLICES = 70                 # 7s context
TTF_MAX = 10.0
N_FEAT = 72                        # 48 base + 4 co-contraction + 4 asym + 16 delta

CONDITION_NAMES = [
    "Healthy", "Drop Foot", "Neuropathy", "Muscle Weakness",
    "Gait Problems", "Lower Back Pain", "Parkinson's / FOG risk",
]

_B_BP, _A_BP = butter(4, [10 / (FS / 2), min(200 / (FS / 2), 0.99)], btype="band")
_B_NF, _A_NF = iirnotch(50.0, 30.0, FS)
_TP_MASK = None  # set lazily once freq axis known


def preprocess_trial(trial):
    """Batch (offline) filtfilt version — matches notebook exactly. Use for
    CSV playback where the whole recording is available at once."""
    trial = np.atleast_2d(trial)
    bp = filtfilt(_B_BP, _A_BP, trial, axis=-1)
    return filtfilt(_B_NF, _A_NF, bp, axis=-1)


def _spectral_entropy(psd):
    p = psd / (psd.sum() + 1e-12)
    return float(-np.sum(p * np.log2(p + 1e-12)))


def slice_features(trial):
    """Exact port of notebook's slice_features: 72-dim per slice."""
    global _TP_MASK
    n_ch, n_samples = trial.shape
    freq = np.fft.rfftfreq(WIN_LEN, 1 / FS)
    if _TP_MASK is None:
        _TP_MASK = (freq >= 3.0) & (freq <= 8.0)
    tp_mask = _TP_MASK

    slices = []
    for start in range(0, n_samples - WIN_LEN + 1, STRIDE):
        win = trial[:, start:start + WIN_LEN]
        feat = []
        ch_rms = np.sqrt(np.mean(win ** 2, axis=1))
        for ch in range(n_ch):
            x = win[ch]
            rms = ch_rms[ch]
            mav = np.mean(np.abs(x))
            psd = np.abs(np.fft.rfft(x)) ** 2
            tp = float(np.sum(psd[tp_mask]))
            cum = np.cumsum(psd)
            mdf = float(freq[np.searchsorted(cum, 0.5 * cum[-1])]) if cum[-1] > 0 else 0.0
            var = np.var(x)
            se = _spectral_entropy(psd)
            feat.extend([rms, mav, tp, mdf, var, se])
        for p in [(0, 1), (2, 3), (4, 5), (6, 7)]:
            feat.append(float(np.mean(np.abs(win[p[0]]) * np.abs(win[p[1]]))))
        for i in range(4):
            feat.append(float(np.abs(np.mean(np.abs(win[i])) - np.mean(np.abs(win[i + 4])))))
        slices.append(feat)

    arr = np.array(slices, dtype=np.float32)
    rms_idx = [i * 6 for i in range(n_ch)]
    tp_idx = [i * 6 + 2 for i in range(n_ch)]
    d_rms = np.diff(arr[:, rms_idx], axis=0, prepend=arr[0:1, rms_idx])
    d_tp = np.diff(arr[:, tp_idx], axis=0, prepend=arr[0:1, tp_idx])
    return np.hstack([arr, d_rms, d_tp])  # (n_slices, 72)


class _StreamChannelFilter:
    """Streaming (causal, sosfilt-based) approx of the notebook's offline
    filtfilt band-pass+notch, so live mode doesn't need the whole signal."""
    def __init__(self):
        sos_bp = butter(4, [10 / (FS / 2), min(200 / (FS / 2), 0.99)], btype="band", output="sos")
        b_nf, a_nf = _B_NF, _A_NF
        from scipy.signal import tf2sos
        sos_nf = tf2sos(b_nf, a_nf)
        self.sos_bp, self.sos_nf = sos_bp, sos_nf
        self.zi_bp = sosfilt_zi(sos_bp)
        self.zi_nf = sosfilt_zi(sos_nf)

    def apply(self, x):
        y, self.zi_bp = sosfilt(self.sos_bp, x, zi=self.zi_bp)
        y, self.zi_nf = sosfilt(self.sos_nf, y, zi=self.zi_nf)
        return y


class StreamingFeatureExtractor:
    """Live version: causal filter -> rolling WIN_LEN/STRIDE slices ->
    slice_features (recomputed on filtered tail) -> WINDOW_SLICES context."""
    def __init__(self):
        self.filters = [_StreamChannelFilter() for _ in range(N_CHANNELS)]
        self.filtered_buf = np.zeros((N_CHANNELS, 0), dtype=np.float32)
        self.feat_slices = []  # rolling list of 72-dim feature vectors
        self._since_last = 0
        self._prev_rms_tp = None  # for delta continuity across pushes

    def push_samples(self, chunk):
        filt = np.vstack([self.filters[c].apply(chunk[c]) for c in range(N_CHANNELS)]).astype(np.float32)
        self.filtered_buf = np.hstack([self.filtered_buf, filt])
        max_len = FS * 60
        if self.filtered_buf.shape[1] > max_len:
            self.filtered_buf = self.filtered_buf[:, -max_len:]

        new_slice = False
        self._since_last += chunk.shape[1]
        while self._since_last >= STRIDE and self.filtered_buf.shape[1] >= WIN_LEN:
            win = self.filtered_buf[:, -WIN_LEN:]
            sf = slice_features(win)  # single-slice call -> shape (1, 72)
            row = sf[0]
            if self._prev_rms_tp is not None:
                rms_idx = list(range(0, 48, 6))
                tp_idx = [i + 2 for i in rms_idx]
                d_rms = row[rms_idx] - self._prev_rms_tp[0]
                d_tp = row[tp_idx] - self._prev_rms_tp[1]
                row = np.concatenate([row[:48 + 8], d_rms, d_tp])
            self._prev_rms_tp = (row[list(range(0, 48, 6))], row[[i + 2 for i in range(0, 48, 6)]])
            self.feat_slices.append(row)
            if len(self.feat_slices) > WINDOW_SLICES:
                self.feat_slices.pop(0)
            self._since_last -= STRIDE
            new_slice = True
        return new_slice

    def ready_for_inference(self):
        return len(self.feat_slices) >= WINDOW_SLICES

    def get_context_window(self):
        """(WINDOW_SLICES, N_FEAT) — matches model's Input(shape=(70,72))."""
        return np.stack(self.feat_slices[-WINDOW_SLICES:])

    def get_filtered_tail(self, n_samples):
        return self.filtered_buf[:, -n_samples:]


class RunningNormalizer:
    """Fallback only — prefer shipping the notebook's actual per-feature
    mean/std (computed via X[tri].mean/std(axis=(0,1))) as an .npz."""
    def __init__(self, n_features):
        self.n = 0
        self.mean = np.zeros(n_features)
        self.M2 = np.zeros(n_features)

    def update(self, row):
        self.n += 1
        delta = row - self.mean
        self.mean += delta / self.n
        self.M2 += delta * (row - self.mean)

    @property
    def std(self):
        return np.sqrt(self.M2 / max(self.n - 1, 1)) + 1e-8

    def normalize(self, ctx):
        return (ctx - self.mean) / self.std


class SyntheticEMGStreamer:
    """Demo-mode signal generator (not used for real inference accuracy)."""
    def __init__(self, scenario="Healthy", seed=42, freeze_at_s=None):
        self.scenario = scenario
        self.rng = np.random.default_rng(seed)
        self.freeze_at_s = freeze_at_s
        self.t = 0.0

    def generate_chunk(self, n_samples):
        t = self.t + np.arange(n_samples) / FS
        chunk = np.zeros((N_CHANNELS, n_samples))
        base_freq = 8 if "FOG" in self.scenario or "Parkinson" in self.scenario else 3
        for ch in range(N_CHANNELS):
            f = base_freq + ch * 0.5
            amp = 1.0 + 0.3 * ch
            sig = amp * np.sin(2 * np.pi * f * t + ch)
            sig += self.rng.normal(0, 0.3, n_samples)
            sig += 0.05 * np.sin(2 * np.pi * 50 * t)
            if self.freeze_at_s is not None and t[-1] >= self.freeze_at_s:
                sig *= 2.5
            chunk[ch] = sig
        self.t += n_samples / FS
        return chunk
