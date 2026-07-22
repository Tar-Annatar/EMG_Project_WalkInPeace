"""
arduino_cloud.py — bridges Arduino IoT Cloud to the EMG streaming pipeline.

WHY POLLING (not a webhook)
----------------------------
Streamlit Community Cloud apps don't expose a stable public endpoint that
Arduino Cloud could push a webhook to, and Arduino IoT Cloud doesn't offer
raw webhook delivery on variable change anyway — the supported integration
surface is its REST API. So the pattern here is: the Arduino device syncs
values *up* to Arduino Cloud as usual, and this app *polls* Arduino Cloud's
REST API on a timer to pull the latest value back down. That's the standard,
supported way to get Arduino Cloud data into any external app or backend.

WHY BATCHED JSON (not one sample per request)
------------------------------------------------
The EMG model needs 500 Hz per channel. Arduino Cloud is a state-sync
service (it stores each variable's *last value*), not a raw high-frequency
telemetry pipe, and polling the REST API 500x/sec would blow through rate
limits instantly. So the Arduino sketch buffers a short window of raw
samples locally and syncs ONE compact JSON-encoded batch (e.g. 50 samples x
8 channels every 100 ms) into a single Cloud "String" variable. This module
polls that variable at the same cadence and unpacks it back into a
(n_channels, n_samples) array — the exact shape push_samples() expects.
See ARDUINO_SKETCH.ino for the matching device-side code.

Get credentials: Arduino Cloud → account icon → "API Keys" → create a key
pair (CLIENT_ID / CLIENT_SECRET, shown once — store it in st.secrets).
Thing ID: open your Thing in Arduino Cloud, it's in the URL / Thing info panel.
"""
import json
import time

import numpy as np
import requests

TOKEN_URL = "https://api2.arduino.cc/iot/v1/clients/token"
API_BASE = "https://api2.arduino.cc/iot/v2"


class ArduinoCloudError(RuntimeError):
    pass


class ArduinoCloudClient:
    """Thin wrapper around the Arduino IoT Cloud REST API (client-credentials
    OAuth2 flow). One instance = one cached access token, refreshed as needed."""

    def __init__(self, client_id: str, client_secret: str):
        self.client_id = client_id
        self.client_secret = client_secret
        self._token = None
        self._token_expiry = 0.0

    def _get_token(self) -> str:
        if self._token and time.time() < self._token_expiry - 30:
            return self._token
        try:
            resp = requests.post(
                TOKEN_URL,
                data={
                    "grant_type": "client_credentials",
                    "client_id": self.client_id,
                    "client_secret": self.client_secret,
                    "audience": "https://api2.arduino.cc/iot",
                },
                timeout=10,
            )
            resp.raise_for_status()
        except requests.RequestException as e:
            raise ArduinoCloudError(f"Could not authenticate with Arduino Cloud: {e}") from e
        data = resp.json()
        self._token = data["access_token"]
        self._token_expiry = time.time() + float(data.get("expires_in", 3600))
        return self._token

    def _headers(self):
        return {"Authorization": f"Bearer {self._get_token()}"}

    def list_things(self):
        r = requests.get(f"{API_BASE}/things", headers=self._headers(), timeout=10)
        r.raise_for_status()
        return r.json()

    def get_properties(self, thing_id: str):
        r = requests.get(
            f"{API_BASE}/things/{thing_id}/properties",
            headers=self._headers(), timeout=10,
        )
        if r.status_code == 404:
            raise ArduinoCloudError(f"Thing '{thing_id}' not found (check the Thing ID).")
        r.raise_for_status()
        return r.json()

    def get_property_value(self, thing_id: str, property_name: str):
        """Latest value of one named Cloud Variable on a Thing."""
        props = self.get_properties(thing_id)
        for p in props:
            if p.get("name") == property_name:
                return p.get("last_value")
        raise ArduinoCloudError(
            f"No variable named '{property_name}' on this Thing. "
            f"Available: {[p.get('name') for p in props]}"
        )


def decode_emg_batch(raw_value, n_channels: int = 8):
    """
    Parse the JSON payload pushed by the Arduino sketch into a
    (n_channels, n_samples) float32 array.

    Expected shape from the sketch: {"n": 50, "ch": [[...ch1...], [...ch2...], ...]}
    Also accepts a flat list (interpreted channel-major) as a fallback.
    Returns None if the payload can't be decoded as a batch (e.g. it's the
    device's placeholder value on first boot).
    """
    if raw_value is None or isinstance(raw_value, (int, float, bool)):
        return None
    try:
        payload = json.loads(raw_value) if isinstance(raw_value, str) else raw_value
    except (json.JSONDecodeError, TypeError):
        return None

    if isinstance(payload, dict) and "ch" in payload:
        arr = np.array(payload["ch"], dtype=np.float32)
    elif isinstance(payload, list):
        arr = np.array(payload, dtype=np.float32)
        if arr.ndim == 1:
            if arr.size % n_channels != 0:
                return None
            arr = arr.reshape(n_channels, -1)
    else:
        return None

    if arr.ndim != 2 or arr.shape[0] != n_channels:
        return None
    return arr
