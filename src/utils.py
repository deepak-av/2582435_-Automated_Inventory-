# src/utils.py
"""Utility helpers for fine-grained inventory counting & overlay drawing.
- `draw_boxes(...)`: draws bounding boxes with dynamic color coding per category.
- `draw_summary_banner(...)`: renders a top banner showing total count + category breakdown.
- `overlay_boxes(...)`: creates annotated shelf image with numbered item boxes.
"""

import os
import cv2
import numpy as np
from typing import List, Dict, Any
from src.inventory_counter import process_inventory

_OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "outputs"))
os.makedirs(_OUTPUT_DIR, exist_ok=True)


def load_image(image_path: str) -> np.ndarray:
    """Load an image from *image_path* using OpenCV."""
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to read image at {image_path}")
    return img


def get_category_color(category: str) -> tuple:
    """Return distinct BGR color palette for fine inventory categories."""
    cat = category.lower()
    if "pen" in cat or "pencil" in cat or "marker" in cat:
        return (0, 255, 128)      # Bright Lime Green
    elif "brush" in cat or "paint" in cat or "tool" in cat:
        return (0, 140, 255)      # Vibrant Orange
    elif "clip" in cat or "fastener" in cat:
        return (255, 0, 200)      # Neon Magenta / Pink
    elif "sticky" in cat or "pad" in cat:
        return (0, 235, 255)      # Neon Yellow
    elif "book" in cat or "notebook" in cat:
        return (255, 200, 0)      # Cyan / Light Blue
    elif "calculator" in cat or "electronics" in cat:
        return (255, 100, 0)      # Deep Blue
    elif "bag" in cat:
        return (180, 0, 255)      # Purple
    else:
        return (0, 255, 0)        # Green


def draw_boxes(image: np.ndarray, boxes: List[List[int]], labels: List[str], colors: List[tuple] = None) -> np.ndarray:
    """Draw bounding *boxes* on *image* with high-visibility background labels."""
    if not boxes:
        return image

    h_img, w_img = image.shape[:2]
    line_thickness = max(1, int(min(h_img, w_img) / 450))
    font_scale = max(0.38, min(h_img, w_img) / 1400.0)

    if colors is None:
        colors = [(0, 255, 0)] * len(boxes)

    for (box, label, color) in zip(boxes, labels, colors):
        x, y, w, h = box
        pt1 = (int(x), int(y))
        pt2 = (int(x + w), int(y + h))
        cv2.rectangle(image, pt1, pt2, color, line_thickness)

        # Draw micro badge background
        (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        lbl_y1 = max(int(y) - text_h - baseline - 4, 0)
        lbl_y2 = max(int(y), text_h + baseline + 4)
        cv2.rectangle(image, (int(x), lbl_y1), (int(x) + text_w + 6, lbl_y2), color, cv2.FILLED)

        # Draw text label in contrasting color
        text_color = (0, 0, 0) if sum(color) > 380 else (255, 255, 255)
        cv2.putText(image, label, (int(x) + 3, lbl_y2 - baseline - 2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 1, cv2.LINE_AA)

    return image


def draw_summary_banner(image: np.ndarray, summary: Dict[str, Any]) -> np.ndarray:
    """Render a clean summary banner showing exact total count and minute item breakdown."""
    h_img, w_img = image.shape[:2]
    banner_h = max(45, int(h_img * 0.055))

    banner = np.zeros((banner_h, w_img, 3), dtype=np.uint8)
    cv2.rectangle(banner, (0, 0), (w_img, banner_h), (20, 20, 20), cv2.FILLED)

    total = summary.get("total_items", 0)
    cat_counts = summary.get("category_counts", {})
    
    cat_breakdown = []
    for cat, cnt in cat_counts.items():
        short_name = cat.split("&")[0].strip()
        cat_breakdown.append(f"{short_name}: {cnt}")
    
    breakdown_str = " | ".join(cat_breakdown[:4])
    text = f"TOTAL INVENTORY COUNT: {total} ITEMS  [{breakdown_str}]" if breakdown_str else f"TOTAL INVENTORY COUNT: {total} ITEMS"

    font_scale = max(0.55, banner_h / 70.0)
    thickness = max(2, int(banner_h / 28.0))
    (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    text_x = max(15, (w_img - text_w) // 2)
    text_y = (banner_h + text_h) // 2

    cv2.putText(banner, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 128), thickness, cv2.LINE_AA)
    image[0:banner_h, 0:w_img] = banner
    return image


def overlay_boxes(image_path: str, detections: List[Dict], ocr_results: List[Dict]) -> str:
    """Create an annotated version of *image_path*.
    Draws fine-grained numbered item boxes (#1, #2...) and shelf tags.
    Saves the annotated image to ``outputs/<original_name>_annotated.jpg`` and returns the new path.
    """
    img = load_image(image_path).copy()

    summary = process_inventory(detections, ocr_results)
    items = summary["items"]
    shelf_tags = summary["shelf_tags"]

    # 1. Draw Minute Item Boxes (#1, #2, #3...) with category colors
    item_boxes = [item["bbox"] for item in items]
    item_labels = [item["item_label"] for item in items]
    item_colors = [get_category_color(item["category"]) for item in items]

    if item_boxes:
        img = draw_boxes(img, item_boxes, item_labels, colors=item_colors)

    # 2. Draw Shelf Tags (Magenta / Purple)
    tag_boxes = [tag["bbox"] for tag in shelf_tags]
    tag_labels = [f"TAG: {tag['text']}" for tag in shelf_tags]
    if tag_boxes:
        img = draw_boxes(img, tag_boxes, tag_labels, colors=[(255, 0, 255)] * len(tag_boxes))

    # 3. Draw Summary Banner
    img = draw_summary_banner(img, summary)

    base_name = os.path.basename(image_path)
    name, ext = os.path.splitext(base_name)
    out_path = os.path.join(_OUTPUT_DIR, f"{name}_annotated{ext}")
    cv2.imwrite(out_path, img)
    return out_path
