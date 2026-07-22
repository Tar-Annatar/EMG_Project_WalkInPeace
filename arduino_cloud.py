"""
Minimal, robust Arduino IoT Cloud REST client.

- OAuth2 client-credentials token flow (cached until near-expiry)
- Reads the latest value of a Thing property ("last_value")
- decode_emg_batch() never raises; it returns None on any malformed/missing
  data so the caller can safely treat it as "waiting for next batch".
"""
import base64
import json
import time

import numpy as np
import requests

TOKEN_URL = "https://api2.arduino.cc/iot/v1/clients/token"
API_BASE = "https://api2.arduino.cc/iot/v2"


class ArduinoCloudError(Exception):
    """Raised for any Arduino IoT Cloud auth/network/response failure."""
    pass


class ArduinoCloudClient:
    def __init__(self, client_id: str, client_secret: str, timeout: float = 8.0):
        self.client_id = client_id
        self.client_secret = client_secret
        self.timeout = timeout
        self._token = None
        self._token_exp = 0.0

    def _get_token(self) -> str:
        if self._token and time.time() < self._token_exp - 30:
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
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            self._token = data["access_token"]
            self._token_exp = time.time() + float(data.get("expires_in", 300))
            return self._token
        except requests.RequestException as e:
            raise ArduinoCloudError(f"Auth failed: {e}") from e
        except (KeyError, ValueError) as e:
            raise ArduinoCloudError(f"Auth response malformed: {e}") from e

    def get_property_value(self, thing_id: str, variable_name: str):
        if not (thing_id and variable_name):
            raise ArduinoCloudError("Thing ID and variable name are required.")
        token = self._get_token()
        try:
            resp = requests.get(
                f"{API_BASE}/things/{thing_id}/properties",
                headers={"Authorization": f"Bearer {token}"},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            props = resp.json()
        except requests.RequestException as e:
            raise ArduinoCloudError(f"Cloud request failed: {e}") from e
        except ValueError as e:
            raise ArduinoCloudError(f"Bad response from cloud: {e}") from e

        if not isinstance(props, list):
            raise ArduinoCloudError("Unexpected response format from Arduino Cloud.")

        for p in props:
            if isinstance(p, dict) and p.get("name") == variable_name:
                return p.get("last_value")

        raise ArduinoCloudError(f"Variable '{variable_name}' not found on this Thing.")


def decode_emg_batch(raw_value, n_channels: int):
    """
    Decode a raw property value into an (n_channels, n_samples) float32 array.
    Accepts: None, JSON array string, python list, or base64-encoded float32 bytes.
    Never raises — returns None for missing/malformed data so callers can treat
    that as "waiting for next batch" instead of crashing.
    """
    if raw_value is None:
        return None
    try:
        data = raw_value
        if isinstance(data, str):
            s = data.strip()
            if not s:
                return None
            try:
                data = json.loads(s)
            except ValueError:
                try:
                    buf = base64.b64decode(s, validate=True)
                    data = np.frombuffer(buf, dtype=np.float32)
                except Exception:
                    return None

        arr = np.asarray(data, dtype=np.float32)
        if arr.size == 0:
            return None

        if arr.ndim == 1:
            n_samples = arr.size // n_channels
            if n_samples < 1:
                return None
            arr = arr[: n_samples * n_channels].reshape(n_samples, n_channels).T
        elif arr.ndim == 2:
            if arr.shape[0] == n_channels:
                pass
            elif arr.shape[1] == n_channels:
                arr = arr.T
            else:
                return None
        else:
            return None

        return np.ascontiguousarray(arr, dtype=np.float32)
    except Exception:
        return None
