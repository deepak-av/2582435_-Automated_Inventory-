# src/planogram_diff.py
"""Planogram comparison and inventory count variance engine.
Compares cleaned inventory detections + OCR tags against expected planogram items.
"""

from typing import List, Dict, Any
from src.inventory_counter import process_inventory, compute_iou


def compute_diff(detections: List[Dict[str, Any]],
                 ocr_results: List[Dict[str, Any]],
                 planogram: Dict[str, Any]) -> Dict[str, Any]:
    """Compare detections + OCR results against the expected planogram and include total counts.

    Returns
    -------
    Dict containing:
    - summary: total item count, category counts, item list, shelf tags
    - missing: items in planogram but not detected
    - extra: detected items not in planogram
    - misplaced: items with low IoU (<0.5)
    """
    summary = process_inventory(detections, ocr_results)
    items = summary["items"]
    shelf_tags = summary["shelf_tags"]

    plan_items = planogram.get("items", [])
    plan_by_sku = {item.get("sku", "").lower(): item for item in plan_items}

    # Combined candidate matches
    combined = []
    for item in items:
        combined.append({
            "item_id": item["item_id"],
            "key": item["category"].lower(),
            "bbox": item["bbox"],
            "source": "inventory_item"
        })
    for tag in shelf_tags:
        combined.append({
            "key": tag["text"].lower(),
            "bbox": tag["bbox"],
            "source": "shelf_tag"
        })

    missing = []
    extra = []
    misplaced = []

    matched_keys = set()
    for entry in combined:
        key = entry["key"]
        matched_sku = None
        for sku in plan_by_sku:
            if sku in key or key in sku:
                matched_sku = sku
                break

        if matched_sku:
            matched_keys.add(matched_sku)
            expected_bbox = plan_by_sku[matched_sku]["bbox"]
            iou_score = compute_iou(entry["bbox"], expected_bbox)
            if iou_score < 0.5:
                misplaced.append({
                    "sku": matched_sku,
                    "detected_bbox": entry["bbox"],
                    "expected_bbox": expected_bbox,
                    "iou": iou_score,
                    "source": entry["source"]
                })
        else:
            extra.append({
                "key": entry["key"],
                "bbox": entry["bbox"],
                "source": entry["source"]
            })

    for sku, plan_entry in plan_by_sku.items():
        if sku not in matched_keys:
            missing.append({"sku": sku, "expected_bbox": plan_entry["bbox"]})

    return {
        "summary": summary,
        "missing": missing,
        "extra": extra,
        "misplaced": misplaced
    }
