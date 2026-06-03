# ─────────────────────────────────────────────────────────────────
# FILE 3: app.py  — Streamlit main app
# Responsibility: UI, video frame loop, orchestrates
#                 object_detector.py + siren_analyzer.py,
#                 renders live results.
# ─────────────────────────────────────────────────────────────────

import streamlit as st
import cv2
import numpy as np
import tempfile, os
from pathlib import Path

# ── Local modules ─────────────────────────────────────────────────
import object_detector as od
import siren_analyzer  as sa

# ─────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Ambulance Detector",
    page_icon="🚑",
    layout="wide",
)

st.markdown("""
<style>
  .stApp,.main{background:#0a0a14;color:#e0e0e0}
  h1{color:#ff4444!important;font-size:2.2rem!important}
  .mcard{background:#141428;border:1px solid #2a2a4a;border-radius:14px;
         padding:18px 22px;text-align:center;margin-bottom:10px}
  .mlabel{font-size:.78rem;color:#888;text-transform:uppercase;letter-spacing:1px}
  .mvalue{font-size:1.9rem;font-weight:700;margin-top:4px}
  .alert{border-radius:14px;padding:20px;text-align:center;
         font-size:1.4rem;font-weight:700;margin:14px 0}
  .alert-on {background:#ff000018;border:2px solid #ff4444;color:#ff4444}
  .alert-off{background:#00c85318;border:2px solid #00c853;color:#00c853}
  .alert-vis{background:#ffb30018;border:2px solid #ffb300;color:#ffb300}
  .flog{background:#0e0e20;border:1px solid #2a2a4a;border-radius:10px;
        padding:12px;font-size:.78rem;font-family:monospace;
        max-height:280px;overflow-y:auto;color:#aaa}
  .la{color:#ff4444;font-weight:bold}
  .lw{color:#ffb300}
  .ls{color:#00c853}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
# Sidebar — settings
# ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")

    audio_threshold = st.slider(
        "🔊 Siren threshold", 0.1, 0.9, 0.5, 0.05,
        help="Siren probability above this = siren detected"
    )
    visual_conf = st.slider(
        "👁️ YOLO confidence", 0.1, 0.9, 0.35, 0.05,
        help="Minimum confidence for vehicle detection"
    )
    window_sec = st.slider(
        "🎵 Audio window (±s)", 0.25, 2.0, 0.5, 0.25,
        help="Half-window of audio centred on each frame"
    )

    st.markdown("---")
    st.markdown("""
### 🧩 Module pipeline
```
app.py
 ├── object_detector.py
 │     └── YOLOv8n + color heuristic
 └── siren_analyzer.py
       ├── librosa (MFCC)
       └── ambulance_siren_model.h5
```
**Alert logic:** Visual **AND** Audio must both fire.
""")


# ─────────────────────────────────────────────────────────────────
# Overlay helper (status bar on frame)
# ─────────────────────────────────────────────────────────────────
def _overlay(frame: np.ndarray,
             visual: bool, audio: bool, alert: bool) -> np.ndarray:
    h, w = frame.shape[:2]
    if alert:
        label, color, bg = "🚨 AMBULANCE + SIREN", (0, 60, 255), (0, 0, 160)
    elif visual:
        label, color, bg = "🚑 Ambulance — no siren", (0,165,255), (0,80,160)
    elif audio:
        label, color, bg = "🔊 Siren — no ambulance", (0,200,255), (0,100,160)
    else:
        label, color, bg = "✅ Clear", (0,200,80), (0,80,40)

    cv2.rectangle(frame, (0, h - 36), (w, h), bg, -1)
    cv2.putText(frame, label, (12, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return frame


# ─────────────────────────────────────────────────────────────────
# Core processing loop
# ─────────────────────────────────────────────────────────────────
def run_pipeline(video_path: str,
                 frame_ph, metrics_ph, log_ph, progress_bar):
    """
    Iterate video frame-by-frame.
    For each analysed frame:
      1. object_detector.detect()      → visual result
      2. siren_analyzer.analyze_frame()→ audio result
      3. Combine → alert if both fire
    """

    # Pre-load audio once for the whole video
    y, sr = sa.extract_audio(video_path)
    audio_duration = len(y) / sr

    cap          = cv2.VideoCapture(video_path)
    fps          = cap.get(cv2.CAP_PROP_FPS) or 25
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    analyze_every = max(1, int(fps // 4))   # ~4 analysis calls/sec

    stats = dict(frames=0, alerts=0, visual_hits=0,
                 audio_hits=0, max_audio=0.0, log=[])

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        progress_bar.progress(min(frame_idx / max(total_frames, 1), 1.0))

        if frame_idx % analyze_every != 0:
            continue

        stats["frames"] += 1
        timestamp = frame_idx / fps

        # ── 1. Object detection ───────────────────────────────────
        vis_result = od.detect(frame,
                               conf_threshold=visual_conf,
                               weights="yolov8n.pt")
        visual_detected  = vis_result["ambulance_detected"]
        annotated_frame  = vis_result["annotated_frame"]

        # ── 2. Siren analysis ─────────────────────────────────────
        aud_result = sa.analyze_frame(y, sr, timestamp,
                                      window_sec=window_sec,
                                      model_path="ambulance_siren_model.h5")
        audio_prob     = aud_result["siren_probability"]
        audio_detected = audio_prob >= audio_threshold

        # ── 3. Combined alert ─────────────────────────────────────
        alert = visual_detected and audio_detected

        # ── Update stats ──────────────────────────────────────────
        if visual_detected: stats["visual_hits"] += 1
        if audio_detected:  stats["audio_hits"]  += 1
        if alert:           stats["alerts"]      += 1
        stats["max_audio"] = max(stats["max_audio"], audio_prob)

        # ── Annotate & display frame ──────────────────────────────
        annotated_frame = _overlay(annotated_frame,
                                   visual_detected, audio_detected, alert)
        rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        frame_ph.image(rgb, use_container_width=True)

        # ── Live metrics ──────────────────────────────────────────
        with metrics_ph.container():
            c1, c2, c3, c4 = st.columns(4)
            def mc(col, label, val, color="#e0e0e0"):
                col.markdown(
                    f'<div class="mcard"><div class="mlabel">{label}</div>'
                    f'<div class="mvalue" style="color:{color}">{val}</div></div>',
                    unsafe_allow_html=True
                )
            mc(c1, "🕐 Time",      f"{timestamp:.1f}s")
            mc(c2, "🚑 Ambulance", "YES" if visual_detected else "NO",
               "#ff4444" if visual_detected else "#00c853")
            mc(c3, "🔊 Siren",     f"{audio_prob*100:.1f}%",
               "#ff4444" if audio_detected else "#00c853")
            mc(c4, "🚨 Alerts",    str(stats["alerts"]),
               "#ff4444" if stats["alerts"] else "#aaa")

        # ── Log entry ─────────────────────────────────────────────
        ts = f"{timestamp:6.2f}s"
        if alert:
            entry = (f'<span class="la">[{ts}] 🚨 ALERT — '
                     f'Ambulance + Siren ({audio_prob:.1%})</span>')
        elif visual_detected:
            entry = f'<span class="lw">[{ts}] 🚑 Ambulance visible, siren {audio_prob:.1%}</span>'
        elif audio_detected:
            entry = f'<span class="lw">[{ts}] 🔊 Siren {audio_prob:.1%}, no ambulance</span>'
        else:
            entry = f'<span class="ls">[{ts}] ✅ Clear — siren {audio_prob:.1%}</span>'

        stats["log"].append(entry)
        log_ph.markdown(
            '<div class="flog">' + "<br>".join(stats["log"][-40:]) + "</div>",
            unsafe_allow_html=True,
        )

    cap.release()
    sa.clear_audio_cache()
    return stats


# ─────────────────────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────────────────────
st.markdown("# 🚑 Ambulance Detector — Vision + Audio")
st.markdown(
    "Upload a video. "
    "**`object_detector.py`** scans every frame with YOLOv8. "
    "**`siren_analyzer.py`** classifies the audio. "
    "A combined 🚨 alert fires only when **both** detect simultaneously."
)
st.markdown("---")

uploaded = st.file_uploader(
    "📁 Upload video",
    type=["mp4", "avi", "mov", "mkv", "webm"],
)

if uploaded:
    suffix = Path(uploaded.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

    st.audio(uploaded)
    st.markdown("---")
    st.markdown("### 🎬 Live Analysis")

    frame_col, _ = st.columns([3, 1])
    with frame_col:
        frame_ph = st.empty()

    metrics_ph  = st.empty()
    progress    = st.progress(0.0, text="Analysing…")
    st.markdown("### 📋 Frame Log")
    log_ph = st.empty()

    stats = run_pipeline(tmp_path, frame_ph, metrics_ph, log_ph, progress)
    os.unlink(tmp_path)
    progress.progress(1.0, text="✅ Done")

    # ── Summary ───────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📊 Summary")
    cols = st.columns(5)
    summary = [
        ("Frames", str(stats["frames"]),          "#e0e0e0"),
        ("🚑 Visual", str(stats["visual_hits"]),  "#ffb300"),
        ("🔊 Audio",  str(stats["audio_hits"]),   "#ffb300"),
        ("🚨 Alerts", str(stats["alerts"]),
         "#ff4444" if stats["alerts"] else "#aaa"),
        ("Peak siren", f"{stats['max_audio']*100:.1f}%", "#e0e0e0"),
    ]
    for col, (label, val, color) in zip(cols, summary):
        col.markdown(
            f'<div class="mcard"><div class="mlabel">{label}</div>'
            f'<div class="mvalue" style="color:{color}">{val}</div></div>',
            unsafe_allow_html=True
        )

    cls = "alert-on" if stats["alerts"] else "alert-off"
    msg = ("🚨 AMBULANCE WITH ACTIVE SIREN DETECTED"
           if stats["alerts"] else "✅ No combined alert in this video")
    st.markdown(f'<div class="alert {cls}">{msg}</div>', unsafe_allow_html=True)

else:
    st.info("👆 Upload a video file to begin.")

st.markdown("---")
st.caption("object_detector.py · siren_analyzer.py · YOLOv8n · ambulance_siren_model.h5 · Streamlit")
