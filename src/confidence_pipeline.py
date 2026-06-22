from __future__ import annotations

import os
from collections import Counter
from pathlib import Path
from typing import Any

import cv2


DEFAULT_HIGH_THRESHOLD = 0.82
DEFAULT_MEDIUM_THRESHOLD = 0.55
DEFAULT_LOW_THRESHOLD = 0.0

HIGH = "HIGH"
MEDIUM = "MEDIUM"
LOW = "LOW"


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
    except ValueError:
        return default


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _safe_score(value: Any, default: float | None = None) -> float | None:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return default
    if score != score:
        return default
    if score > 1.0 and score <= 100.0:
        score = score / 100.0
    return _clamp01(score)


def confidence_pipeline_enabled() -> bool:
    return _bool_env("FORM_PARSER_CONFIDENCE_PIPELINE_ENABLED", False)


def review_queue_enabled() -> bool:
    return _bool_env("FORM_PARSER_REVIEW_QUEUE_ENABLED", False)


def confidence_thresholds(
    *,
    high_threshold: float | None = None,
    medium_threshold: float | None = None,
    low_threshold: float | None = None,
) -> dict[str, float]:
    high = _clamp01(
        high_threshold
        if high_threshold is not None
        else _float_env("FORM_PARSER_CONFIDENCE_HIGH_THRESHOLD", DEFAULT_HIGH_THRESHOLD)
    )
    medium = _clamp01(
        medium_threshold
        if medium_threshold is not None
        else _float_env("FORM_PARSER_CONFIDENCE_MEDIUM_THRESHOLD", DEFAULT_MEDIUM_THRESHOLD)
    )
    low = _clamp01(
        low_threshold
        if low_threshold is not None
        else _float_env("FORM_PARSER_CONFIDENCE_LOW_THRESHOLD", DEFAULT_LOW_THRESHOLD)
    )
    if medium > high:
        medium = high
    if low > medium:
        low = medium
    return {"HIGH": high, "MEDIUM": medium, "LOW": low}


def confidence_level(score: float, thresholds: dict[str, float] | None = None) -> str:
    thresholds = thresholds or confidence_thresholds()
    score = _clamp01(float(score))
    if score >= thresholds["HIGH"]:
        return HIGH
    if score >= thresholds["MEDIUM"]:
        return MEDIUM
    return LOW


def _section_confidence(mapping: dict[str, Any]) -> float | None:
    section = mapping.get("section") if isinstance(mapping.get("section"), dict) else None
    if not section:
        return None
    return _safe_score(section.get("confidence"))


def _answer_confidence(mapping: dict[str, Any]) -> float | None:
    answer_region = mapping.get("answer_region") if isinstance(mapping.get("answer_region"), dict) else None
    if not answer_region:
        return None
    return _safe_score(answer_region.get("confidence"))


def _widget_confidence(mapping: dict[str, Any]) -> tuple[float | None, str | None]:
    widget_type = str(mapping.get("widget_type") or "").lower()
    if widget_type == "comb":
        return _safe_score(mapping.get("comb_confidence")), "comb_confidence"
    if widget_type == "radio":
        return _safe_score(mapping.get("radio_confidence")), "radio_confidence"
    return None, None


def _anchor_type(mapping: dict[str, Any]) -> str:
    anchoring = mapping.get("anchoring") if isinstance(mapping.get("anchoring"), dict) else {}
    answer_region = mapping.get("answer_region") if isinstance(mapping.get("answer_region"), dict) else {}
    return str(anchoring.get("anchor_type") or answer_region.get("type") or "unknown")


def _geometric_score(mapping: dict[str, Any], reasons: list[str]) -> float:
    anchor_type = _anchor_type(mapping)
    scores = {
        "value_block": 0.92,
        "checkbox_region": 0.90,
        "radio_region": 0.90,
        "comb_region": 0.88,
        "table_cell": 0.84,
        "empty_rectangle": 0.79,
        "underline": 0.76,
        "line_region": 0.72,
        "adjacent_estimate": 0.58,
        "label_projection": 0.52,
        "unresolved_label_region": 0.18,
        "photo_region": 0.78,
    }
    score = scores.get(anchor_type, 0.62)
    if anchor_type == "unresolved_label_region":
        reasons.append("weak_geometry_unresolved_label_region")
    elif anchor_type in {"adjacent_estimate", "label_projection"}:
        reasons.append(f"estimated_geometry_{anchor_type}")
    elif anchor_type == "table_cell":
        reasons.append("table_cell_geometry")
    elif anchor_type in {"comb_region", "radio_region"}:
        reasons.append(f"widget_geometry_{anchor_type}")
    else:
        reasons.append(f"geometry_{anchor_type}")
    return score


def _top_candidate_gap(mapping: dict[str, Any]) -> float | None:
    anchoring = mapping.get("anchoring") if isinstance(mapping.get("anchoring"), dict) else {}
    top_candidates = anchoring.get("top_candidates") if isinstance(anchoring.get("top_candidates"), list) else []
    scores: list[float] = []
    for candidate in top_candidates[:2]:
        if not isinstance(candidate, dict):
            continue
        score = _safe_score(candidate.get("score", candidate.get("candidate_score")))
        if score is not None:
            scores.append(score)
    if len(scores) < 2:
        return None
    return abs(scores[0] - scores[1])


def _penalties(mapping: dict[str, Any]) -> tuple[float, list[str], int]:
    penalty = 0.0
    reasons: list[str] = []
    ambiguity_count = 0

    confidence_class = str(mapping.get("confidence_class") or "").lower()
    if confidence_class in {"ambiguous", "weak_match"}:
        penalty += 0.08
        ambiguity_count += 1
        reasons.append(f"legacy_{confidence_class}")
    elif confidence_class == "low":
        penalty += 0.08
        reasons.append("legacy_low_confidence_class")
    elif confidence_class == "unresolved":
        penalty += 0.16
        reasons.append("legacy_unresolved_confidence_class")

    anchor_type = _anchor_type(mapping)
    if anchor_type == "unresolved_label_region":
        penalty += 0.12
    elif anchor_type in {"adjacent_estimate", "label_projection"}:
        penalty += 0.05

    anchoring = mapping.get("anchoring") if isinstance(mapping.get("anchoring"), dict) else {}
    candidate_count = int(anchoring.get("candidate_count") or 0) if str(anchoring.get("candidate_count") or "").isdigit() else 0
    if candidate_count >= 8:
        penalty += 0.05
        ambiguity_count += 1
        reasons.append("many_anchor_candidates")
    elif candidate_count >= 4:
        penalty += 0.03
        reasons.append("multiple_anchor_candidates")

    gap = _top_candidate_gap(mapping)
    if gap is not None and gap <= 0.06:
        penalty += 0.08
        ambiguity_count += 1
        reasons.append("close_top_anchor_candidates")

    label_overlap = _safe_score(anchoring.get("label_overlap_ratio"), 0.0) or 0.0
    if label_overlap >= 0.25:
        penalty += 0.08
        reasons.append("answer_region_overlaps_label")
    elif label_overlap >= 0.08:
        penalty += 0.03
        reasons.append("minor_answer_label_overlap")

    rejected = mapping.get("rejected_candidates")
    if isinstance(rejected, list) and rejected:
        penalty += 0.03
        reasons.append("has_rejected_candidates")

    widget_type = str(mapping.get("widget_type") or "").lower()
    if widget_type == "comb" and (_safe_score(mapping.get("comb_confidence"), 1.0) or 1.0) < 0.84:
        penalty += 0.04
        reasons.append("comb_confidence_below_default_threshold")
    if widget_type == "radio" and (_safe_score(mapping.get("radio_confidence"), 1.0) or 1.0) < 0.86:
        penalty += 0.04
        reasons.append("radio_confidence_below_default_threshold")

    return min(0.45, penalty), reasons, ambiguity_count


def score_mapping_confidence(mapping: dict[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    components: dict[str, dict[str, float]] = {}

    anchor_score = _safe_score(mapping.get("candidate_score"), _safe_score(mapping.get("confidence"), 0.55)) or 0.55
    components["anchor_score"] = {"score": anchor_score, "weight": 0.42}

    section_score = _section_confidence(mapping)
    if section_score is not None:
        components["section_confidence"] = {"score": section_score, "weight": 0.10}
    else:
        reasons.append("missing_section_confidence")

    answer_score = _answer_confidence(mapping)
    if answer_score is not None:
        components["answer_region_confidence"] = {"score": answer_score, "weight": 0.18}
    else:
        reasons.append("missing_answer_region_confidence")

    geometry_score = _geometric_score(mapping, reasons)
    components["geometric_evidence"] = {"score": geometry_score, "weight": 0.16}

    widget_score, widget_key = _widget_confidence(mapping)
    if widget_score is not None and widget_key is not None:
        components[widget_key] = {"score": widget_score, "weight": 0.10}

    answer_region = mapping.get("answer_region") if isinstance(mapping.get("answer_region"), dict) else {}
    if answer_region.get("type") == "table_cell":
        table_score = _safe_score(answer_region.get("confidence"), 0.75) or 0.75
        components["table_confidence"] = {"score": table_score, "weight": 0.08}

    weight_total = sum(item["weight"] for item in components.values()) or 1.0
    raw_score = sum(item["score"] * item["weight"] for item in components.values()) / weight_total
    penalty, penalty_reasons, ambiguity_count = _penalties(mapping)
    score = _clamp01(raw_score - penalty)
    reasons.extend(penalty_reasons)

    return {
        "score": round(score, 4),
        "raw_score": round(raw_score, 4),
        "penalty": round(penalty, 4),
        "components": {
            key: {"score": round(value["score"], 4), "weight": round(value["weight"], 4)}
            for key, value in components.items()
        },
        "reasons": reasons,
        "ambiguity_count": ambiguity_count,
    }


def _review_reason(level: str, reasons: list[str]) -> str:
    if level == LOW:
        return "low_confidence_auto_render_suppressed"
    if level == MEDIUM:
        if any("ambiguous" in reason or "candidate" in reason for reason in reasons):
            return "medium_confidence_ambiguous_candidates"
        if any("geometry" in reason for reason in reasons):
            return "medium_confidence_weak_geometry"
        return "medium_confidence_requires_review"
    return ""


def _review_evidence(mapping: dict[str, Any], confidence: dict[str, Any], level: str) -> dict[str, Any]:
    anchoring = mapping.get("anchoring") if isinstance(mapping.get("anchoring"), dict) else {}
    payload = {
        "field_id": mapping.get("field_id"),
        "label": mapping.get("label"),
        "page": mapping.get("page", 1),
        "bbox": mapping.get("bbox"),
        "confidence_score": confidence["score"],
        "confidence_level": level,
        "raw_score": confidence["raw_score"],
        "penalty": confidence["penalty"],
        "components": confidence["components"],
        "reasons": confidence["reasons"],
        "anchor_type": _anchor_type(mapping),
        "confidence_class": mapping.get("confidence_class"),
        "candidate_count": anchoring.get("candidate_count"),
        "top_candidates": (anchoring.get("top_candidates") or [])[:5] if isinstance(anchoring.get("top_candidates"), list) else [],
    }
    if mapping.get("widget_type"):
        payload["widget_type"] = mapping.get("widget_type")
    if mapping.get("comb_detection"):
        payload["comb_detection"] = mapping.get("comb_detection")
    if mapping.get("radio_detection"):
        payload["radio_detection"] = mapping.get("radio_detection")
    return payload


def _field_report(mapping: dict[str, Any]) -> dict[str, Any]:
    return {
        "field_id": mapping.get("field_id"),
        "label": mapping.get("label"),
        "page": mapping.get("page", 1),
        "field_type": mapping.get("field_type"),
        "widget_type": mapping.get("widget_type"),
        "confidence_score": mapping.get("confidence_score"),
        "confidence_level": mapping.get("confidence_level"),
        "needs_review": bool(mapping.get("needs_review")),
        "auto_render": mapping.get("auto_render", True),
        "review_reason": mapping.get("review_reason"),
    }


def _disabled_result(mappings: list[dict[str, Any]], thresholds: dict[str, float], review_enabled: bool) -> dict[str, Any]:
    diagnostics = {
        "enabled": False,
        "review_queue_enabled": review_enabled,
        "feature_flag": "FORM_PARSER_CONFIDENCE_PIPELINE_ENABLED",
        "review_feature_flag": "FORM_PARSER_REVIEW_QUEUE_ENABLED",
        "thresholds": thresholds,
        "fields_by_confidence": {},
        "review_count": 0,
        "low_confidence_count": 0,
        "ambiguity_count": 0,
        "render_skipped_count": 0,
        "reasoning": "Confidence pipeline is disabled; mappings and rendering policy are unchanged.",
    }
    report = {
        **diagnostics,
        "field_count": len(mappings),
        "rendered_count": len(mappings),
        "fields": [],
    }
    review_artifacts = {
        "enabled": False,
        "review_queue_enabled": review_enabled,
        "field_count": 0,
        "render_skipped_count": 0,
        "fields": [],
    }
    return {
        "mappings": mappings,
        "render_mappings": mappings,
        "confidence_report": report,
        "review_artifacts": review_artifacts,
        "diagnostics": diagnostics,
    }


def apply_confidence_pipeline(
    mappings: list[dict[str, Any]] | None,
    *,
    enabled: bool | None = None,
    review_enabled: bool | None = None,
    high_threshold: float | None = None,
    medium_threshold: float | None = None,
    low_threshold: float | None = None,
) -> dict[str, Any]:
    enabled = confidence_pipeline_enabled() if enabled is None else bool(enabled)
    review_enabled = review_queue_enabled() if review_enabled is None else bool(review_enabled)
    thresholds = confidence_thresholds(
        high_threshold=high_threshold,
        medium_threshold=medium_threshold,
        low_threshold=low_threshold,
    )
    input_mappings = [dict(mapping) if isinstance(mapping, dict) else mapping for mapping in (mappings or [])]
    if not enabled:
        return _disabled_result(input_mappings, thresholds, review_enabled)

    enriched: list[dict[str, Any]] = []
    review_fields: list[dict[str, Any]] = []
    ambiguity_count = 0
    level_counts: Counter[str] = Counter()
    render_skipped = 0

    for mapping in input_mappings:
        if not isinstance(mapping, dict):
            enriched.append(mapping)
            continue
        confidence = score_mapping_confidence(mapping)
        level = confidence_level(confidence["score"], thresholds)
        level_counts[level] += 1
        ambiguity_count += int(confidence.get("ambiguity_count") or 0)

        updated = dict(mapping)
        updated["confidence_score"] = confidence["score"]
        updated["confidence_level"] = level
        updated["confidence_evidence"] = {
            "raw_score": confidence["raw_score"],
            "penalty": confidence["penalty"],
            "components": confidence["components"],
            "reasons": confidence["reasons"],
        }

        if review_enabled:
            if level == LOW:
                updated["needs_review"] = True
                updated["review_reason"] = _review_reason(level, confidence["reasons"])
                updated["review_evidence"] = _review_evidence(updated, confidence, level)
                updated["auto_render"] = False
                render_skipped += 1
                review_fields.append(updated["review_evidence"])
            elif level == MEDIUM:
                updated["needs_review"] = True
                updated["review_reason"] = _review_reason(level, confidence["reasons"])
                updated["review_evidence"] = _review_evidence(updated, confidence, level)
                updated["auto_render"] = True
                review_fields.append(updated["review_evidence"])
            else:
                updated["needs_review"] = False
                updated["auto_render"] = True
        else:
            updated["auto_render"] = True

        enriched.append(updated)

    render_mappings = [
        mapping
        for mapping in enriched
        if not isinstance(mapping, dict) or bool(mapping.get("auto_render", True))
    ]
    fields = [_field_report(mapping) for mapping in enriched if isinstance(mapping, dict)]
    diagnostics = {
        "enabled": True,
        "review_queue_enabled": review_enabled,
        "feature_flag": "FORM_PARSER_CONFIDENCE_PIPELINE_ENABLED",
        "review_feature_flag": "FORM_PARSER_REVIEW_QUEUE_ENABLED",
        "thresholds": thresholds,
        "fields_by_confidence": dict(sorted(level_counts.items())),
        "review_count": len(review_fields),
        "medium_review_count": sum(1 for field in review_fields if field.get("confidence_level") == MEDIUM),
        "low_confidence_count": level_counts.get(LOW, 0),
        "ambiguity_count": ambiguity_count,
        "render_skipped_count": render_skipped,
        "reasoning": (
            "HIGH fields render normally; MEDIUM fields render with review metadata; "
            "LOW fields are withheld from auto-rendering only when the review queue flag is enabled."
        ),
    }
    confidence_report = {
        **diagnostics,
        "field_count": len(fields),
        "rendered_count": len(render_mappings),
        "fields": fields,
    }
    review_artifacts = {
        "enabled": review_enabled,
        "review_queue_enabled": review_enabled,
        "field_count": len(review_fields),
        "medium_count": diagnostics["medium_review_count"],
        "low_count": sum(1 for field in review_fields if field.get("confidence_level") == LOW),
        "render_skipped_count": render_skipped,
        "fields": review_fields,
    }
    return {
        "mappings": enriched,
        "render_mappings": render_mappings,
        "confidence_report": confidence_report,
        "review_artifacts": review_artifacts,
        "diagnostics": diagnostics,
    }


def _normalized_box_to_pixels(box: dict[str, Any], image_width: int, image_height: int) -> tuple[int, int, int, int] | None:
    try:
        x = float(box["x"])
        y = float(box["y"])
        width = float(box["width"])
        height = float(box["height"])
    except (KeyError, TypeError, ValueError):
        return None
    if width <= 0 or height <= 0:
        return None
    return (
        int(x * image_width),
        int(y * image_height),
        int((x + width) * image_width),
        int((y + height) * image_height),
    )


def _pixel_box(box: dict[str, Any]) -> tuple[int, int, int, int] | None:
    try:
        x = int(float(box["x"]))
        y = int(float(box["y"]))
        width = int(float(box["width"]))
        height = int(float(box["height"]))
    except (KeyError, TypeError, ValueError):
        return None
    if width <= 0 or height <= 0:
        return None
    return x, y, x + width, y + height


def draw_confidence_overlay(
    image_path: str | Path,
    mappings: list[dict[str, Any]] | None,
    output_path: str | Path,
    *,
    page: int = 1,
) -> bool:
    image = cv2.imread(str(image_path))
    if image is None:
        return False
    image_height, image_width = image.shape[:2]
    colors = {
        HIGH: (60, 170, 60),
        MEDIUM: (0, 180, 255),
        LOW: (40, 40, 220),
    }
    for index, mapping in enumerate(mappings or [], start=1):
        if not isinstance(mapping, dict):
            continue
        try:
            mapping_page = int(mapping.get("page") or 1)
        except (TypeError, ValueError):
            mapping_page = 1
        if mapping_page != page:
            continue
        level = str(mapping.get("confidence_level") or "").upper()
        if level not in colors:
            level = confidence_level(_safe_score(mapping.get("confidence_score"), _safe_score(mapping.get("candidate_score"), 0.0)) or 0.0)
        color = colors[level]
        boxes = mapping.get("field_bboxes") if isinstance(mapping.get("field_bboxes"), list) else []
        pixel_boxes = [_pixel_box(box) for box in boxes if isinstance(box, dict)]
        if not any(pixel_boxes):
            bbox = mapping.get("bbox") if isinstance(mapping.get("bbox"), dict) else None
            pixel_boxes = [_normalized_box_to_pixels(bbox, image_width, image_height)] if bbox else []
        for pixel in pixel_boxes:
            if pixel is None:
                continue
            x1, y1, x2, y2 = pixel
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            label = f"{level} {float(mapping.get('confidence_score') or 0.0):.2f}"
            cv2.putText(image, label, (x1, max(12, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
            cv2.putText(image, str(index), (x1 + 3, y1 + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(output), image))
