# src/llm_reasoning.py
"""Interface to local DeepSeek-R1:7b LLM via Ollama.
Provides:
- `generate_report(...)`: Generates audit report & restocking checklist from CV context.
- `query_assistant(...)`: Answers natural-language staff queries using live inventory context.
- Fast non-blocking execution with graceful fallback.
"""

import json
import re
import logging
from typing import Tuple, Dict, Any

import requests

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)

_OLLAMA_URL = "http://localhost:11434/api/chat"
_MODEL_NAME = "deepseek-r1:7b"


def _strip_think_tags(text: str) -> Tuple[str, str]:
    """Separate `<think>` reasoning chain from the main response."""
    think_parts = re.findall(r"<think>(.*?)</think>", text, flags=re.DOTALL)
    clean = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    thoughts = "\n".join(part.strip() for part in think_parts)
    return clean, thoughts


def _build_context_str(diff: Dict[str, Any], operational_context: str = "Standard Audit") -> str:
    """Build structured CV context string for LLM reasoning."""
    summary = diff.get("summary", {})
    total_items = summary.get("total_items", 0)
    category_counts = summary.get("category_counts", {})
    tags = summary.get("shelf_tags", [])
    missing = diff.get("missing", [])
    misplaced = diff.get("misplaced", [])

    lines = [
        f"OPERATIONAL CONTEXT: {operational_context}",
        "--- CV ENGINE STRUCTURED OUTPUT ---",
        f"Total Physical Inventory Counted: {total_items} units",
        "Category Breakdown:"
    ]
    for cat, cnt in category_counts.items():
        lines.append(f"  - {cat}: {cnt} units")

    if tags:
        lines.append("Recognized Shelf Section Tags:")
        for t in tags[:6]:
            lines.append(f"  - Tag: '{t['text']}'")

    if missing:
        lines.append("Missing Planogram Items:")
        for m in missing:
            lines.append(f"  - SKU `{m['sku']}` missing at {m['expected_bbox']}")

    if misplaced:
        lines.append("Misplaced Items:")
        for mp in misplaced:
            lines.append(f"  - `{mp['sku']}` detected at {mp['detected_bbox']}, expected {mp['expected_bbox']} (IoU={mp['iou']:.2f})")

    lines.append("---")
    return "\n".join(lines)


def generate_report(diff: Dict[str, Any], operational_context: str = "Standard Audit") -> Tuple[str, str]:
    """Send CV context to local DeepSeek-R1:7b model with 15s quick timeout and fallback."""
    cv_context = _build_context_str(diff, operational_context)

    prompt = f"""You are an AI Store Operations Assistant powering an Automated Shelf Audit system.
{cv_context}

INSTRUCTIONS:
1. Evaluate restocking urgency & severity given context '{operational_context}'.
2. Generate a concise plain-language Audit Summary.
3. Provide a step-by-step shelf-correction checklist with priority tags ([CRITICAL], [HIGH], [MEDIUM]).

Use <think> tags for your step-by-step internal reasoning chain before your final response.
"""
    raw = ""
    try:
        payload = {
            "model": _MODEL_NAME,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {
                "num_predict": 250,
                "temperature": 0.2
            }
        }
        # 15s fast timeout to prevent UI hanging
        response = requests.post(_OLLAMA_URL, json=payload, timeout=15)
        if response.status_code == 200:
            data = response.json()
            raw = data.get("message", {}).get("content", "")
    except Exception as e:
        _logger.warning(f"Ollama local API call timed out / unavailable: {e}")
        raw = ""

    if raw:
        return _strip_think_tags(raw)
    else:
        return _generate_fallback_report(diff, operational_context)


def query_assistant(user_question: str, diff: Dict[str, Any], operational_context: str = "Standard Audit") -> Tuple[str, str]:
    """Answer natural-language staff queries using live CV context and local DeepSeek-R1."""
    cv_context = _build_context_str(diff, operational_context)

    prompt = f"""You are an AI Inventory Assistant helping store staff.
Current Shelf Audit Context:
{cv_context}

Store Staff Question: "{user_question}"

Answer the question directly, concisely, and accurately based on the shelf count data. Use <think> tags for internal reasoning.
"""
    try:
        payload = {
            "model": _MODEL_NAME,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {
                "num_predict": 200,
                "temperature": 0.2
            }
        }
        response = requests.post(_OLLAMA_URL, json=payload, timeout=15)
        if response.status_code == 200:
            data = response.json()
            raw = data.get("message", {}).get("content", "")
            if raw:
                return _strip_think_tags(raw)
    except Exception as e:
        _logger.warning(f"Ollama query timed out: {e}")

    summary = diff.get("summary", {})
    total = summary.get("total_items", 0)
    cat_counts = summary.get("category_counts", {})
    return f"Based on current shelf count context ({total} total items detected across {len(cat_counts)} categories), category counts are: {cat_counts}.", "Ollama fallback used."


def _generate_fallback_report(diff: Dict[str, Any], operational_context: str) -> Tuple[str, str]:
    """Rule-based report generator providing instant structured audit summaries."""
    summary = diff.get("summary", {})
    total_items = summary.get("total_items", 0)
    cat_counts = summary.get("category_counts", {})
    tags = summary.get("shelf_tags", [])

    cat_summary = ", ".join([f"**{cat}**: {cnt}" for cat, cnt in cat_counts.items()]) if cat_counts else "None"
    tag_summary = ", ".join([f"`{t['text']}`" for t in tags[:5]]) if tags else "None"

    missing_cnt = len(diff.get("missing", []))

    report = f"""### 📊 Inventory Audit & Stock Report ({operational_context})

- **Total Physical Count:** `{total_items}` units
- **Categories Identified:** {cat_summary}
- **Shelf Tags Recognized:** {tag_summary}

#### 🎯 Restocking Priority ({operational_context})
1. **Stock Alignment:** Verified **{total_items}** physical units on shelf layout.
2. **High Demand Items:** Refill writing tools (**Pens, Pencils & Markers: {cat_counts.get('Pens, Pencils & Markers', 0)}**) for {operational_context}.
3. **Planogram Discrepancies:** {missing_cnt} missing items identified.

#### 📋 Restocking & Store Staff Checklist
- [ ] Verify shelf section tags ({tag_summary}).
- [ ] Replenish depleted bins for high-turnover items ({cat_summary}).
- [ ] Ensure item count alignment with store POS system.
"""
    return report, "Automated instant summary generated."
