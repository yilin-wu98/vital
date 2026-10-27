"""Small client for the existing Robometer streaming protocol.

The model server is an external dependency. This module starts no service and
opens no connection until one of its functions is called.
"""
from io import BytesIO
import warnings

import numpy as np
import requests


def _url(base, suffix):
    return base.rstrip("/") + suffix


def _upload(base, session_id, operation, array, field, timeout_s):
    buffer = BytesIO()
    np.save(buffer, np.asarray(array), allow_pickle=False)
    response = requests.post(
        _url(base, f"/stream/{session_id}/{operation}"),
        files={field: (f"{field}.npy", buffer.getvalue(), "application/octet-stream")},
        timeout=timeout_s,
    )
    response.raise_for_status()
    return response.json()


def stream_create(eval_server_url, task, timeout_s=30.0, max_frames=None):
    response = requests.post(
        _url(eval_server_url, "/stream/create"),
        json={"task": task, "max_frames": max_frames},
        timeout=timeout_s,
    )
    response.raise_for_status()
    return response.json()["session_id"]


def stream_append_frame(eval_server_url, session_id, frame_np, timeout_s=120.0):
    data = _upload(eval_server_url, session_id, "append", frame_np, "frame", timeout_s)
    return float(data["reward"]), float(data["success_prob"])


def stream_evaluate_candidates_batch(eval_server_url, session_id, frames_np, timeout_s=120.0):
    data = _upload(
        eval_server_url, session_id, "evaluate_candidates", frames_np, "frames", timeout_s,
    )
    rewards = np.asarray(data["rewards"], dtype=np.float32)
    success = np.asarray(data["success_probs"], dtype=np.float32)
    if rewards.shape != (len(frames_np),) or success.shape != rewards.shape:
        raise ValueError("Robometer returned a different number of candidate scores")
    return rewards, success


def stream_delete(eval_server_url, session_id, timeout_s=10.0):
    try:
        response = requests.delete(
            _url(eval_server_url, f"/stream/{session_id}"), timeout=timeout_s,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        warnings.warn(f"Could not close Robometer session: {exc}", RuntimeWarning)
