# ─────────────────────────────────────────────────────────────────
# FILE 1: object_detector.py
# Responsibility: Load YOLOv8, detect vehicles frame-by-frame,
#                 apply color heuristic to flag ambulances,
#                 return annotated frame + detection result.
# ─────────────────────────────────────────────────────────────────

import cv2
import numpy as np
from ultralytics import YOLO

# COCO vehicle class IDs that could be an ambulance
VEHICLE_CLASSES = {2: "car", 5: "bus", 7: "truck"}

_model = None  # module-level singleton


def load_model(weights: str = "yolov8n.pt") -> YOLO:
    """Load (or return cached) YOLOv8 model."""
    global _model
    if _model is None:
        _model = YOLO(weights)
    return _model


def _is_ambulance_by_color(frame: np.ndarray, box) -> bool:
    """
    Heuristic: check if the bounding-box crop contains
    white / yellow / red ambulance color signatures.
    Returns True if the score exceeds threshold.
    """
    x1, y1, x2, y2 = map(int, box)
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return False

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # White (ambulance body)
    white  = cv2.inRange(hsv, np.array([0,   0, 200]), np.array([180,  40, 255]))
    # Yellow (ambulance markings)
    yellow = cv2.inRange(hsv, np.array([20, 100, 100]), np.array([35,  255, 255]))
    # Red/orange (emergency lights)
    red1   = cv2.inRange(hsv, np.array([0,  120, 100]), np.array([10,  255, 255]))
    red2   = cv2.inRange(hsv, np.array([160,120, 100]), np.array([180, 255, 255]))

    total  = roi.size // 3
    score  = (
        cv2.countNonZero(white)  / total +
        cv2.countNonZero(yellow) / total * 0.5 +
        (cv2.countNonZero(red1) + cv2.countNonZero(red2)) / total * 0.3
    )
    return score > 0.25


def _draw_box(frame: np.ndarray, box, label: str,
              color: tuple, conf: float) -> None:
    """Draw a labelled bounding box on the frame (in-place)."""
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    text = f"{label} {conf:.0%}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
    cv2.putText(frame, text, (x1 + 3, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)


def detect(frame: np.ndarray,
           conf_threshold: float = 0.35,
           weights: str = "yolov8n.pt") -> dict:
    """
    Run object detection on a single BGR frame.

    Parameters
    ----------
    frame           : np.ndarray  — BGR frame from OpenCV
    conf_threshold  : float       — YOLO confidence cutoff
    weights         : str         — path/name of YOLO weights

    Returns
    -------
    {
        "ambulance_detected": bool,
        "detections": [{"label", "confidence", "box", "is_ambulance"}, ...],
        "annotated_frame": np.ndarray  (BGR, boxes drawn)
    }
    """
    model = load_model(weights)
    results = model(frame, verbose=False, conf=conf_threshold)[0]

    annotated   = frame.copy()
    detections  = []
    ambulance_detected = False

    for box in results.boxes:
        cls_id = int(box.cls[0])
        if cls_id not in VEHICLE_CLASSES:
            continue

        conf   = float(box.conf[0])
        coords = box.xyxy[0].cpu().numpy()
        is_amb = _is_ambulance_by_color(frame, coords)
        label  = "Ambulance?" if is_amb else VEHICLE_CLASSES[cls_id]
        color  = (0, 60, 255) if is_amb else (0, 165, 255)

        _draw_box(annotated, coords, label, color, conf)

        detections.append({
            "label":        label,
            "confidence":   conf,
            "box":          coords.tolist(),
            "is_ambulance": is_amb,
        })

        if is_amb:
            ambulance_detected = True

    return {
        "ambulance_detected": ambulance_detected,
        "detections":         detections,
        "annotated_frame":    annotated,
    }
