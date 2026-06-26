"""Phase L0.1 — false-checkbox rejection (additive, flag-gated, diagnostics-first).

Textract (and the token-checkbox fallback) sometimes emits a "checkbox" where
the page actually holds narrow OCR glyphs — ``11``, ``II``, ``|``, ``l``, ``1`` —
or a stray mark that is not a real control. This module scores every checkbox
mapping for *authenticity* using three deterministic signals:

* **aspect ratio** — a real checkbox is roughly square; vertical-stroke text is
  tall and narrow.
* **contour closure** — a real checkbox has a closed, ~rectangular border in the
  raster; OCR text does not.
* **OCR overlap** — a Textract WORD sitting on the box whose text is made of
  vertical strokes / ones is a strong "this is text, not a box" signal.

Gated behind ``FORM_PARSER_CHECKBOX_VALIDATION_ENABLED`` (default **OFF**). When
OFF the engine is never invoked and mappings are byte-for-byte unchanged. When
ON it drops checkboxes scoring below a threshold and records every decision in
diagnostics. Diagnostics are emitted whenever it runs; nothing else changes.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import cv2
import numpy as np

# Glyphs that, when they make up the entire overlapping token, indicate the
# "checkbox" is really vertical-stroke OCR text (11 / II / | / l / 1 / !).
_STROKE_CHARS = set("|Il1!ﬁ/\\¦ìíïî")
_SUSPICIOUS_TOKENS = {"11", "111", "ll", "ii", "i1", "1i", "iii", "|"}
# Tokens that AFFIRM a real control (a tick/cross/box drawn or OCR'd inside it).
_AFFIRMING_TOKENS = {"x", "✓", "✔", "☑", "☒", "☐", "[]", "[ ]", "[x]", "■", "□", "•"}

DEFAULT_REJECT_THRESHOLD = 0.35


def _bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _float_env(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def checkbox_validation_enabled() -> bool:
    return _bool_env("FORM_PARSER_CHECKBOX_VALIDATION_ENABLED", False)


def checkbox_validation_observe_enabled() -> bool:
    """Observe mode: score + diagnose every checkbox but NEVER drop one.

    Lets us measure rejection precision against real forms before any
    destructive filtering is switched on. Independent of the enforce flag.
    """
    return _bool_env("FORM_PARSER_CHECKBOX_VALIDATION_OBSERVE", False)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _box_right(box: dict[str, float]) -> float:
    return float(box["x"]) + float(box["width"])


def _box_bottom(box: dict[str, float]) -> float:
    return float(box["y"]) + float(box["height"])


def _center(box: dict[str, float]) -> tuple[float, float]:
    return float(box["x"]) + float(box["width"]) / 2.0, float(box["y"]) + float(box["height"]) / 2.0


def _overlaps(a: dict[str, float], b: dict[str, float]) -> float:
    left = max(float(a["x"]), float(b["x"]))
    top = max(float(a["y"]), float(b["y"]))
    right = min(_box_right(a), _box_right(b))
    bottom = min(_box_bottom(a), _box_bottom(b))
    if right <= left or bottom <= top:
        return 0.0
    inter = (right - left) * (bottom - top)
    denom = min(
        float(a["width"]) * float(a["height"]),
        float(b["width"]) * float(b["height"]),
    )
    return inter / denom if denom > 0 else 0.0


def _is_suspicious_text(text: str) -> bool:
    norm = re.sub(r"\s+", "", (text or "")).lower()
    if not norm or len(norm) > 4:
        return False
    if norm in _SUSPICIOUS_TOKENS:
        return True
    # Entirely made of vertical-stroke glyphs (|, I, l, 1, !) and short.
    return all(ch in _STROKE_CHARS for ch in norm)


def _is_affirming_text(text: str) -> bool:
    norm = re.sub(r"\s+", "", (text or "")).lower()
    return bool(norm) and norm in _AFFIRMING_TOKENS


def _pixel_aspect(box: dict[str, float], size: dict[str, int] | None) -> float:
    """Width/height aspect in *pixels* (corrects for page aspect)."""
    w_frac = max(float(box["width"]), 1e-6)
    h_frac = max(float(box["height"]), 1e-6)
    if size and size.get("width") and size.get("height"):
        ratio = float(size["width"]) / float(size["height"])
    else:
        ratio = 1700.0 / 2200.0  # assume portrait page when size is unknown
    return (w_frac * ratio) / h_frac


def _contour_closure(image: "np.ndarray | None", box: dict[str, float]) -> dict[str, Any]:
    """Look for a closed, ~rectangular border inside the checkbox crop."""
    if image is None:
        return {"checked": False, "closed": None, "fill_ratio": None}
    h, w = image.shape[:2]
    pad_x = float(box["width"]) * 0.25
    pad_y = float(box["height"]) * 0.25
    x0 = int(max(0.0, (float(box["x"]) - pad_x)) * w)
    y0 = int(max(0.0, (float(box["y"]) - pad_y)) * h)
    x1 = int(min(1.0, (_box_right(box) + pad_x)) * w)
    y1 = int(min(1.0, (_box_bottom(box) + pad_y)) * h)
    if x1 - x0 < 3 or y1 - y0 < 3:
        return {"checked": True, "closed": False, "fill_ratio": 0.0}
    crop = image[y0:y1, x0:x1]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    crop_area = float((x1 - x0) * (y1 - y0))
    best_fill = 0.0
    closed = False
    significant_components = 0
    for contour in contours:
        bx, by, bw, bh = cv2.boundingRect(contour)
        area = float(bw * bh)
        if area < crop_area * 0.18:
            continue
        significant_components += 1
        approx = cv2.approxPolyDP(contour, 0.04 * cv2.arcLength(contour, True), True)
        rect_aspect = bw / max(float(bh), 1.0)
        fill = cv2.contourArea(contour) / max(area, 1.0)
        if 3 <= len(approx) <= 6 and 0.45 <= rect_aspect <= 2.2:
            closed = True
            best_fill = max(best_fill, fill)
    # Connected-component count (Workstream C): a real box is ~1 closed
    # component; vertical-stroke OCR ("11"/"II") yields several thin ones.
    return {
        "checked": True,
        "closed": closed,
        "fill_ratio": round(best_fill, 4),
        "component_count": significant_components,
    }


def _score_checkbox(
    box: dict[str, float],
    page: int,
    text_boxes: list[dict[str, Any]],
    image: "np.ndarray | None",
    size: dict[str, int] | None,
) -> dict[str, Any]:
    reasons: list[str] = []
    score = 0.5

    aspect = _pixel_aspect(box, size)
    if 0.55 <= aspect <= 1.8:
        score += 0.25
        reasons.append("square_aspect")
    elif aspect < 0.45 or aspect > 2.4:
        score -= 0.30
        reasons.append("non_square_aspect")

    # OCR overlap: the strongest signal.
    overlap_text = ""
    for tb in text_boxes:
        if int(tb.get("page") or 1) != page:
            continue
        if _overlaps(box, tb["bbox"]) >= 0.55:
            overlap_text = str(tb.get("text") or "")
            break
    if overlap_text and _is_suspicious_text(overlap_text):
        score -= 0.45
        reasons.append(f"ocr_text_stroke_glyphs:{overlap_text!r}")
    elif overlap_text and _is_affirming_text(overlap_text):
        score += 0.10
        reasons.append(f"ocr_affirms_control:{overlap_text!r}")
    elif not overlap_text:
        score += 0.05
        reasons.append("no_ocr_overlap")

    closure = _contour_closure(image, box)
    if closure["closed"] is True:
        score += 0.30
        reasons.append("closed_contour")
    elif closure["closed"] is False:
        score -= 0.15
        reasons.append("no_closed_contour")

    return {
        "score": round(_clamp01(score), 4),
        "aspect": round(aspect, 4),
        "ocr_text": overlap_text,
        "contour": closure,
        "reasons": reasons,
    }


def validate_checkboxes(
    mappings: list[dict[str, Any]],
    *,
    page_images: dict[int, Any] | None = None,
    text_boxes: list[dict[str, Any]] | None = None,
    image_sizes: dict[int, dict[str, int]] | None = None,
    reject_threshold: float | None = None,
    observe: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Score checkbox mappings; drop the likely-OCR-text ones.

    Returns ``(kept_mappings, diagnostics)``. Non-checkbox mappings pass through
    untouched and in order. In ``observe`` mode every checkbox is scored and the
    would-reject decisions are recorded, but **no mapping is dropped** — the
    returned list is the input unchanged. Use observe mode to measure rejection
    precision before enabling destructive filtering.
    """
    text_boxes = text_boxes or []
    page_images = page_images or {}
    image_sizes = image_sizes or {}
    threshold = (
        reject_threshold
        if reject_threshold is not None
        else _float_env("FORM_PARSER_CHECKBOX_REJECT_THRESHOLD", DEFAULT_REJECT_THRESHOLD)
    )

    image_cache: dict[int, Any] = {}

    def _image_for(page: int) -> Any:
        if page in image_cache:
            return image_cache[page]
        path = page_images.get(page)
        img = cv2.imread(str(path)) if path else None
        image_cache[page] = img
        return img

    kept: list[dict[str, Any]] = []
    evaluations: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []

    for mapping in mappings:
        if not isinstance(mapping, dict) or mapping.get("field_type") != "checkbox":
            kept.append(mapping)
            continue
        box = mapping.get("bbox")
        if not isinstance(box, dict):
            kept.append(mapping)
            continue
        page = int(mapping.get("page") or 1)
        evaluation = _score_checkbox(box, page, text_boxes, _image_for(page), image_sizes.get(page))
        below_threshold = evaluation["score"] < threshold
        record = {
            "field_id": mapping.get("field_id"),
            "label": mapping.get("label"),
            "page": page,
            "authenticity_score": evaluation["score"],
            "aspect": evaluation["aspect"],
            "ocr_text": evaluation["ocr_text"],
            "closed_contour": evaluation["contour"]["closed"],
            "component_count": evaluation["contour"].get("component_count"),
            "reasons": evaluation["reasons"],
            # In observe mode nothing is dropped; record the hypothetical action.
            "decision": ("would_reject" if observe else "rejected") if below_threshold else "kept",
        }
        evaluations.append(record)
        if below_threshold:
            rejected.append(record)
            if not observe:
                continue  # drop only when enforcing
        kept.append(mapping)

    diagnostics = {
        "enabled": True,
        "mode": "observe" if observe else "enforce",
        "feature_flag": "FORM_PARSER_CHECKBOX_VALIDATION_ENABLED",
        "observe_flag": "FORM_PARSER_CHECKBOX_VALIDATION_OBSERVE",
        "reject_threshold": threshold,
        "checkbox_count": len(evaluations),
        "rejected_count": len(rejected),
        "kept_count": len(evaluations) - (0 if observe else len(rejected)),
        "would_reject_count": len(rejected) if observe else 0,
        "rejected": rejected,
        "evaluations": evaluations,
    }
    return kept, diagnostics
