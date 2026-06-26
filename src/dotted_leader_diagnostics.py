"""Phase L0.2 — dotted-leader source diagnostics (additive, diagnostics-ONLY).

A "dotted leader" (the ``......`` run before a fill-in blank) can reach us two
ways:

* **graphics** — a row of printed dots the CV detector picks up as a synthetic
  underline (``dotted_underline_detector``), or
* **OCR text** — Textract reads the dots as a WORD/LINE whose text is literally
  ``......`` / ``. . . .``.

The two need different downstream handling, but **this module changes nothing**.
It only classifies each detected dotted run by whether an OCR token of dots sits
on it, and separately surfaces dot-runs that exist only as OCR text (which CV
missed). Output is pure diagnostics to inform a future enforcement phase.

Gated behind ``FORM_PARSER_DOTTED_LEADER_DIAGNOSTICS_ENABLED`` (default OFF):
when OFF it is never invoked. There is no behavioural branch — even when ON it
returns diagnostics only.
"""
from __future__ import annotations

import os
import re
from typing import Any

# A token is "dot-like" when, ignoring spaces, it is ≥3 dot/leader glyphs.
_DOT_CHARS = ".·•‥…∙﹒｡"
_DOTRUN_RE = re.compile(rf"^[{re.escape(_DOT_CHARS)}\s]+$")


def _bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def dotted_leader_diagnostics_enabled() -> bool:
    return _bool_env("FORM_PARSER_DOTTED_LEADER_DIAGNOSTICS_ENABLED", False)


def _box_right(box: dict[str, float]) -> float:
    return float(box["x"]) + float(box["width"])


def _box_bottom(box: dict[str, float]) -> float:
    return float(box["y"]) + float(box["height"])


def _vertical_overlap_ratio(a: dict[str, float], b: dict[str, float]) -> float:
    top = max(float(a["y"]), float(b["y"]))
    bottom = min(_box_bottom(a), _box_bottom(b))
    if bottom <= top:
        return 0.0
    return (bottom - top) / max(min(float(a["height"]), float(b["height"])), 1e-9)


def _horizontal_overlap_ratio(a: dict[str, float], b: dict[str, float]) -> float:
    left = max(float(a["x"]), float(b["x"]))
    right = min(_box_right(a), _box_right(b))
    if right <= left:
        return 0.0
    return (right - left) / max(min(float(a["width"]), float(b["width"])), 1e-9)


def _is_dot_run(text: str) -> bool:
    norm = re.sub(r"\s+", "", text or "")
    dots = sum(1 for ch in norm if ch in _DOT_CHARS)
    return dots >= 3 and bool(_DOTRUN_RE.match(text or ""))


def _center(box: dict[str, float]) -> tuple[float, float]:
    return float(box["x"]) + float(box["width"]) / 2.0, float(box["y"]) + float(box["height"]) / 2.0


def _leader_associates_label(leader: dict[str, float], label: dict[str, float]) -> bool:
    """Approximate the field engine's association gate: a leader is usable by a
    field only if its label sits just left (same row) or just above it. Mirrors
    ``_synthetic_underline_candidates`` geometry (page-local fractions)."""
    lcx, lcy = _center(leader)
    bcx, bcy = _center(label)
    label_right = float(label["x"]) + float(label["width"])
    label_bottom = float(label["y"]) + float(label["height"])
    right_same_row = float(leader["x"]) >= label_right - 0.04 and abs(lcy - bcy) <= 0.025
    below = lcy > bcy and abs(lcx - bcx) <= max(float(label["width"]), 0.10) and 0 <= (float(leader["y"]) - label_bottom) <= 0.035
    return right_same_row or below


def _recall_accounting(
    dotted_debug_pages: list[dict[str, Any]],
    field_labels: list[dict[str, Any]],
    mappings: list[dict[str, Any]],
) -> dict[str, Any]:
    """Read-only accounting for the leaders-detected vs leaders-used gap.

    Approximate: leader->selection identity is not threaded through the engine,
    so ``leaders_selected`` is counted from final mappings and the per-leader
    association is recomputed geometrically. Intended for observability, not as
    a behavioural gate.
    """
    labels_by_page: dict[int, list[dict[str, float]]] = {}
    for rec in field_labels or []:
        box = rec.get("bbox")
        if isinstance(box, dict):
            labels_by_page.setdefault(int(rec.get("page") or 1), []).append(box)

    detected = 0
    associated = 0
    for page_debug in dotted_debug_pages or []:
        page = int(page_debug.get("page") or 1)
        page_labels = labels_by_page.get(page, [])
        for det in page_debug.get("detected", []) or []:
            box = det.get("bbox")
            if not isinstance(box, dict):
                continue
            detected += 1
            if any(_leader_associates_label(box, label) for label in page_labels):
                associated += 1

    selected = 0
    for m in mappings or []:
        anchoring = m.get("anchoring") if isinstance(m.get("anchoring"), dict) else {}
        if str(anchoring.get("anchor_type") or "") in {"dotted_underline", "broken_underline"}:
            selected += 1

    no_assoc = detected - associated
    assoc_not_selected = max(0, associated - selected)
    return {
        "leaders_detected": detected,
        "leaders_associated_to_a_label": associated,
        "leaders_selected": selected,
        "filter_reason_counts": {
            "no_field_label_association": no_assoc,
            "associated_but_lost_selection": assoc_not_selected,
            "selected_as_answer_region": selected,
        },
        "note": "approximate; leader->selection identity not threaded through the engine",
    }


def _dot_text_boxes(text_boxes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [tb for tb in text_boxes if _is_dot_run(str(tb.get("text") or ""))]


def analyze_dotted_leaders(
    dotted_debug_pages: list[dict[str, Any]] | None,
    *,
    text_boxes: list[dict[str, Any]] | None = None,
    field_labels: list[dict[str, Any]] | None = None,
    mappings: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Classify CV-detected dotted runs by origin and find OCR-only dot runs.

    When ``field_labels`` + ``mappings`` are supplied, also emit the
    leaders-detected vs leaders-used recall accounting (Workstream A).
    """
    dotted_debug_pages = dotted_debug_pages or []
    text_boxes = text_boxes or []
    dot_texts = _dot_text_boxes(text_boxes)

    classified: list[dict[str, Any]] = []
    text_origin = 0
    graphics_origin = 0
    matched_text_ids: set[int] = set()

    for page_debug in dotted_debug_pages:
        page = int(page_debug.get("page") or 1)
        for det in page_debug.get("detected", []) or []:
            box = det.get("bbox")
            if not isinstance(box, dict):
                continue
            # Find an OCR dot-run sitting on this detection (same row, overlapping x).
            origin = "graphics"
            matched_text = None
            for i, tb in enumerate(dot_texts):
                if int(tb.get("page") or 1) != page:
                    continue
                if _vertical_overlap_ratio(box, tb["bbox"]) >= 0.3 and _horizontal_overlap_ratio(box, tb["bbox"]) >= 0.2:
                    origin = "text"
                    matched_text = str(tb.get("text") or "")
                    matched_text_ids.add(i)
                    break
            if origin == "text":
                text_origin += 1
            else:
                graphics_origin += 1
            classified.append(
                {
                    "page": page,
                    "bbox": box,
                    "origin": origin,
                    "ocr_text": matched_text,
                    "source_image": page_debug.get("source_image"),
                }
            )

    # OCR dot-runs that no CV detection covered — leaders only Textract saw.
    ocr_only = [
        {"page": int(tb.get("page") or 1), "bbox": tb["bbox"], "text": str(tb.get("text") or "")}
        for i, tb in enumerate(dot_texts)
        if i not in matched_text_ids
    ]

    result = {
        "enabled": True,
        "feature_flag": "FORM_PARSER_DOTTED_LEADER_DIAGNOSTICS_ENABLED",
        "behavior_change": False,
        "cv_detected_count": len(classified),
        "text_origin_count": text_origin,
        "graphics_origin_count": graphics_origin,
        "ocr_only_dot_run_count": len(ocr_only),
        "classified": classified[:200],
        "ocr_only_dot_runs": ocr_only[:200],
    }
    if field_labels is not None or mappings is not None:
        result["recall"] = _recall_accounting(dotted_debug_pages, field_labels or [], mappings or [])
    return result
