# src/inventory_counter.py
"""Inventory Counting & Contextual Classification Engine.
Resolves mismatches (e.g. markers misclassified as books, notebooks as pens):
- Uses OCR shelf tag spatial context (MARKERS & PENS, ART SUPPLIES, STICKY NOTES).
- Enforces strict geometric dimension & area rules (thin items = pens/markers, large items = books).
- Overrides YOLO COCO misclassifications.
- Applies Non-Maximum Suppression (NMS).
- Assigns clean sequential item numbers (#1, #2, #3...).
"""

import re
from typing import List, Dict, Any


def compute_iou(box_a: List[int], box_b: List[int]) -> float:
    """Calculate Intersection over Union (IoU) for two [x, y, w, h] boxes."""
    ax1, ay1, aw, ah = box_a
    bx1, by1, bw, bh = box_b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = aw * ah
    area_b = bw * bh

    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def clean_ocr_tags(ocr_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Filter out single-digit noise or meaningless character fragments."""
    cleaned_tags = []
    for tag in ocr_results:
        text = str(tag.get("text", "")).strip()
        conf = float(tag.get("confidence", 1.0))
        
        if not text or conf < 0.2:
            continue

        # Skip isolated single digits (e.g. "9", "1", "4")
        if re.match(r'^\d{1,2}$', text) and len(text) <= 2:
            continue

        if len(text) <= 2 and not text.isalnum():
            continue

        cleaned_tags.append({
            "text": text.upper(),
            "confidence": conf,
            "bbox": tag["bbox"]
        })

    return cleaned_tags


def classify_item(bbox: List[int], raw_label: str, shelf_tags: List[Dict[str, Any]]) -> str:
    """Classify item accurately using spatial shelf tag context, physical dimensions, and shape geometry."""
    raw = raw_label.strip().lower()

    if "person" in raw:
        return "Shopper / Staff"

    x, y, w, h = bbox
    area = w * h
    aspect_ratio = float(w) / h if h > 0 else 1.0

    # 1. Spatial Context Alignment with Shelf Tags
    for tag in shelf_tags:
        txt = tag["text"].upper()
        tx, ty, tw, th = tag["bbox"]

        # If item is located under or near tag (within 200px horizontal and below tag Y)
        if abs(x - tx) < 200 and y >= (ty - 30) and y <= (ty + 380):
            if "MARKER" in txt or "PEN" in txt or "WRITING" in txt:
                return "Pens, Pencils & Markers"
            elif "ART" in txt or "BRUSH" in txt:
                return "Art Brushes & Supplies"
            elif "STICKY" in txt or "POST" in txt or "PAD" in txt or "NOTE 3X3" in txt:
                return "Sticky Notes & Pads"

    # 2. Thin Linear Items (Pens, Pencils, Markers)
    # Individual pens/pencils are thin in at least one dimension (w <= 38 or h <= 38)
    if (w <= 38 or h <= 38) and area < 5000:
        return "Pens, Pencils & Markers"

    # 3. Compact Square/Rectangular Pads (Sticky Notes)
    if 500 <= area <= 5500 and 0.6 <= aspect_ratio <= 1.6 and w <= 130 and h <= 130:
        return "Sticky Notes & Pads"

    # 4. Large Surface Items (Notebooks & Books)
    # Notebooks MUST have substantial area (> 5500) and both width > 45 and height > 45
    if area > 5500 and w >= 45 and h >= 45:
        return "Notebooks & Books"

    # 5. Very Small Blobs (Clips & Fasteners)
    if area < 650:
        return "Clips & Fasteners"

    # 6. Fallback based on raw label keyword
    if "pen" in raw or "marker" in raw or "pencil" in raw:
        return "Pens, Pencils & Markers"
    elif "sticky" in raw or "pad" in raw:
        return "Sticky Notes & Pads"
    elif "book" in raw or "notebook" in raw:
        if area > 3500:
            return "Notebooks & Books"
        else:
            return "Sticky Notes & Pads"
    elif "bag" in raw or "backpack" in raw:
        return "Bags & Cases"
    elif "cell phone" in raw or "calculator" in raw:
        return "Calculators & Electronics"
    else:
        return "Stationery Items"


def apply_nms(detections: List[Dict[str, Any]], iou_threshold: float = 0.45) -> List[Dict[str, Any]]:
    """Apply intelligent Non-Maximum Suppression (NMS)."""
    if not detections:
        return []

    def sort_key(d):
        is_fine = 1 if d.get("source") == "fine_detector" else 0
        conf = float(d.get("confidence", 0.0))
        return (is_fine, conf)

    sorted_dets = sorted(detections, key=sort_key, reverse=True)
    kept = []

    while sorted_dets:
        best = sorted_dets.pop(0)
        best_box = best["bbox"]
        best_area = best_box[2] * best_box[3]
        kept.append(best)

        remaining = []
        for det in sorted_dets:
            box = det["bbox"]
            area = box[2] * box[3]
            iou = compute_iou(best_box, box)

            # Keep both if one is much larger (>4x) than the other (e.g. shelf section vs individual pen)
            area_ratio = max(best_area, area) / max(min(best_area, area), 1.0)
            if area_ratio > 4.0:
                remaining.append(det)
                continue

            if iou > iou_threshold:
                continue

            remaining.append(det)

        sorted_dets = remaining

    return kept


def process_inventory(detections: List[Dict[str, Any]], ocr_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Process detections and OCR tags into clean, contextually corrected inventory counts."""
    shelf_tags = clean_ocr_tags(ocr_results)
    formatted_tags = []
    for idx, tag in enumerate(shelf_tags, start=1):
        formatted_tags.append({
            "tag_id": idx,
            "text": tag["text"],
            "confidence": tag["confidence"],
            "confidence_str": f"{int(tag['confidence'] * 100)}%",
            "bbox": tag["bbox"]
        })

    nms_detections = apply_nms(detections, iou_threshold=0.45)

    inventory_items = []
    item_counter = 1
    category_counts: Dict[str, int] = {}

    for det in nms_detections:
        raw_label = det.get("label", "item")
        category = classify_item(det["bbox"], raw_label, shelf_tags)

        if category == "Shopper / Staff":
            continue

        conf = float(det.get("confidence", 0.0))
        conf_pct = f"{int(conf * 100)}%"

        short_cat = category.split("&")[0].strip()
        item_label = f"#{item_counter} {short_cat}"

        item_entry = {
            "item_id": item_counter,
            "item_label": item_label,
            "category": category,
            "confidence": conf,
            "confidence_str": conf_pct,
            "bbox": det["bbox"],
            "source": det.get("source", "detector")
        }
        inventory_items.append(item_entry)

        category_counts[category] = category_counts.get(category, 0) + 1
        item_counter += 1

    return {
        "total_items": len(inventory_items),
        "category_counts": category_counts,
        "items": inventory_items,
        "shelf_tags": formatted_tags
    }
