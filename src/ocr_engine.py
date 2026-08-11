# src/ocr_engine.py
"""EasyOCR wrapper for reading shelf tags.
Provides a simple `read_shelf_tags(image_path)` function returning a
list of dictionaries mapping detected text snippets and bounding boxes.
"""

import os
import logging
from typing import Dict, List, Any

try:
    import easyocr
except ImportError:
    easyocr = None  # type: ignore

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)

_reader = None


def _get_reader():
    global _reader
    if _reader is not None:
        return _reader
    if easyocr is None:
        _logger.warning("easyocr package not installed - OCR disabled.")
        return None
    try:
        _reader = easyocr.Reader(["en"], gpu=False)
        _logger.info("EasyOCR reader initialized.")
    except Exception as e:
        _logger.error(f"Failed to initialize EasyOCR reader: {str(e)}")
        _reader = None
    return _reader


def read_shelf_tags(image_path: str) -> List[Dict[str, Any]]:
    """Run EasyOCR on image_path.

    Returns
    -------
    List[Dict]
        Each dict contains text (string), confidence (float 0-1), and
        bbox ([x, y, w, h]) in pixel coordinates.
    """
    reader = _get_reader()
    if reader is None:
        return []
    try:
        import cv2
        img = cv2.imread(image_path)
        if img is None:
            return []
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        results = reader.readtext(gray, detail=1, paragraph=False)

        ocr_items = []
        for res in results:
            if not res or len(res) < 2:
                continue
            bbox = res[0]
            text = str(res[1])
            conf = float(res[2]) if len(res) > 2 else 1.0

            xs = [float(p[0]) for p in bbox]
            ys = [float(p[1]) for p in bbox]
            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
            w = max(1.0, x2 - x1)
            h = max(1.0, y2 - y1)
            ocr_items.append({
                "text": text,
                "confidence": float(conf),
                "bbox": [int(x1), int(y1), int(w), int(h)]
            })
        return ocr_items
    except Exception as e:
        import traceback
        _logger.error(f"Error during OCR processing: {e}\n{traceback.format_exc()}")
        return []



