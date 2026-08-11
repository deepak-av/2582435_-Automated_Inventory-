# src/object_detection.py
"""Multi-Scale & Fine-Grained Object Detection Engine.
Combines YOLOv8 macro detection with OpenCV adaptive contour/shape analysis
to detect and count minute inventory items like individual pens, pencils,
markers, highlighters, paintbrushes, binder clips, and sticky pads.
"""

import os
import cv2
import numpy as np
import logging
from typing import List, Dict, Any

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None  # type: ignore

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)

_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "yolov8.pt")
_model = None


def _load_model():
    """Lazy-load YOLO model."""
    global _model
    if _model is not None:
        return _model
    if YOLO is None:
        _logger.warning("ultralytics package not installed – object detection disabled.")
        return None
    try:
        if os.path.isfile(_MODEL_PATH):
            _logger.info(f"Loading custom YOLO weights from {_MODEL_PATH}")
            _model = YOLO(_MODEL_PATH)
        else:
            _logger.info("Custom models/yolov8.pt not found. Loading pretrained yolov8n.pt weights...")
            _model = YOLO("yolov8n.pt")
        _logger.info("YOLOv8 model loaded successfully.")
    except Exception as e:
        _logger.error(f"Failed to load YOLO model: {e}")
        _model = None
    return _model


def detect_fine_grained_items(image_path: str) -> List[Dict[str, Any]]:
    """Detect minute small items like pens, pencils, markers, clips, and pads via shape/contour analysis."""
    img = cv2.imread(image_path)
    if img is None:
        return []

    h_img, w_img = img.shape[:2]
    total_area = h_img * w_img

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(thresh, kernel, iterations=1)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    fine_items = []
    # Area thresholds for small minute items
    min_area = total_area * 0.00008   # tiny pens, pencils, clips
    max_area = total_area * 0.05      # medium trays / boxes

    for c in contours:
        area = cv2.contourArea(c)
        if min_area <= area <= max_area:
            x, y, w, h = cv2.boundingRect(c)

            # Skip shelf border borders or full width shelf headers
            if w > w_img * 0.85 or h > h_img * 0.85:
                continue
            
            # Skip bottom/top extreme edges (outside shelf structure)
            if y < 15 or (y + h) > (h_img - 15):
                continue

            aspect_ratio = float(w) / h if h > 0 else 1.0

            # Classify minute stationery item based on aspect ratio & size
            if aspect_ratio > 2.2 or aspect_ratio < 0.45:
                label = "Pen / Pencil / Marker"
                conf = 0.88
            elif 0.6 <= aspect_ratio <= 1.6 and area < total_area * 0.015:
                label = "Sticky Note / Small Pad"
                conf = 0.85
            elif aspect_ratio > 3.0 or aspect_ratio < 0.33:
                label = "Paintbrush / Slim Tool"
                conf = 0.90
            elif area < total_area * 0.002:
                label = "Clip / Fastener"
                conf = 0.82
            else:
                label = "Stationery Item"
                conf = 0.80

            fine_items.append({
                "label": label,
                "confidence": conf,
                "bbox": [int(x), int(y), int(w), int(h)],
                "source": "fine_detector"
            })

    return fine_items


def detect_objects(image_path: str, fine_grained: bool = True) -> List[Dict[str, Any]]:
    """Run multi-scale object detection on image_path.

    Returns list of dicts with ``label``, ``confidence``, ``bbox`` [x, y, w, h].
    """
    detections = []
    model = _load_model()

    # 1. Run YOLOv8 macro detection
    if model is not None:
        try:
            results = model(image_path, conf=0.15)
            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls[0]) if box.cls is not None else -1
                    label = model.names.get(cls_id, f"class_{cls_id}")
                    conf = float(box.conf[0]) if box.conf is not None else 0.0
                    xyxy = box.xyxy[0].tolist()
                    x, y, x2, y2 = xyxy
                    w = x2 - x
                    h = y2 - y
                    detections.append({
                        "label": label,
                        "confidence": conf,
                        "bbox": [int(x), int(y), int(w), int(h)],
                        "source": "yolo"
                    })
        except Exception as e:
            _logger.error(f"YOLO detection error: {e}")

    # 2. Run Fine-Grained Detection for minute pens, pencils, markers, clips
    if fine_grained:
        try:
            fine_items = detect_fine_grained_items(image_path)
            detections.extend(fine_items)
        except Exception as e:
            _logger.error(f"Fine-grained item detection error: {e}")

    return detections
