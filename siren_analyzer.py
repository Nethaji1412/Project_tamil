# ─────────────────────────────────────────────────────────────────
# FILE 2: siren_analyzer.py
# Responsibility: Extract audio from a video file, slice it into
#                 per-frame windows, compute 40 MFCCs, run the
#                 .h5 model, return siren probability per timestamp.
# ─────────────────────────────────────────────────────────────────

import numpy as np
import librosa
import tensorflow as tf
from tensorflow import keras
from pathlib import Path

_audio_model = None   # module-level singleton
_audio_cache: dict = {}  # {video_path: (y, sr)}


# ── Model ─────────────────────────────────────────────────────────

def load_siren_model(model_path: str = "ambulance_siren_model.h5") -> keras.Model:
    """Load (or return cached) MFCC siren classifier."""
    global _audio_model
    if _audio_model is None:
        _audio_model = keras.models.load_model(model_path)
    return _audio_model


# ── Audio extraction ──────────────────────────────────────────────

def extract_audio(video_path: str,
                  sr: int = 22050,
                  use_cache: bool = True) -> tuple[np.ndarray, int]:
    """
    Extract the full mono audio track from a video file.

    Parameters
    ----------
    video_path : str   — path to the video file
    sr         : int   — target sample rate (default 22 050 Hz)
    use_cache  : bool  — cache audio in memory to avoid re-loading

    Returns
    -------
    (y, sr)  — float32 waveform array + sample rate
    """
    global _audio_cache
    key = str(Path(video_path).resolve())

    if use_cache and key in _audio_cache:
        return _audio_cache[key]

    y, sr_loaded = librosa.load(video_path, sr=sr, mono=True)
    result = (y, sr_loaded)

    if use_cache:
        _audio_cache[key] = result

    return result


def clear_audio_cache():
    """Free cached audio — call after processing is done."""
    global _audio_cache
    _audio_cache.clear()


# ── Feature extraction ────────────────────────────────────────────

def _extract_mfcc(segment: np.ndarray, sr: int,
                  n_mfcc: int = 40) -> np.ndarray:
    """
    Compute mean-pooled 40-dim MFCC from an audio segment.
    Returns shape (1, 40) ready for model inference.
    """
    if len(segment) < 512:
        segment = np.pad(segment, (0, 512 - len(segment)))
    mfccs = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfccs.T, axis=0).reshape(1, -1)   # (1, 40)


# ── Per-frame siren analysis ──────────────────────────────────────

def analyze_frame(y: np.ndarray,
                  sr: int,
                  timestamp: float,
                  window_sec: float = 0.5,
                  model_path: str = "ambulance_siren_model.h5") -> dict:
    """
    Analyse a ±window_sec audio slice around `timestamp`.

    Parameters
    ----------
    y           : np.ndarray  — full audio waveform (from extract_audio)
    sr          : int         — sample rate
    timestamp   : float       — frame timestamp in seconds
    window_sec  : float       — half-window size (default ±0.5 s)
    model_path  : str         — path to the .h5 siren model

    Returns
    -------
    {
        "siren_probability": float,   # 0.0 – 1.0
        "siren_detected":    bool,    # True if prob >= threshold
        "timestamp":         float,
        "mfcc_features":     np.ndarray  shape (40,)
    }
    Note: threshold is NOT applied here — caller decides the cutoff.
    """
    model   = load_siren_model(model_path)
    dur     = len(y) / sr
    t0      = max(0.0, timestamp - window_sec)
    t1      = min(dur,  timestamp + window_sec)
    s0, s1  = int(t0 * sr), int(t1 * sr)
    segment = y[s0:s1] if s1 > s0 else y[:1024]

    features = _extract_mfcc(segment, sr)                   # (1, 40)
    prob     = float(model.predict(features, verbose=0)[0][0])

    return {
        "siren_probability": prob,
        "siren_detected":    False,      # caller applies threshold
        "timestamp":         timestamp,
        "mfcc_features":     features[0],
    }


def analyze_full_video(video_path: str,
                       fps: float,
                       total_frames: int,
                       analyze_every: int = 1,
                       window_sec: float = 0.5,
                       threshold: float = 0.5,
                       model_path: str = "ambulance_siren_model.h5") -> list[dict]:
    """
    Pre-analyse ALL audio segments for a video upfront.
    Useful when you want fast per-frame lookup without re-running inference.

    Returns
    -------
    List of result dicts (one per analysed frame index), each containing:
        frame_index, timestamp, siren_probability, siren_detected, mfcc_features
    """
    y, sr = extract_audio(video_path)
    results = []

    for frame_idx in range(0, total_frames, analyze_every):
        timestamp = frame_idx / fps
        r = analyze_frame(y, sr, timestamp, window_sec, model_path)
        r["frame_index"]    = frame_idx
        r["siren_detected"] = r["siren_probability"] >= threshold
        results.append(r)

    return results
