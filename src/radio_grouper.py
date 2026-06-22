from __future__ import annotations

import os
import re
from collections import Counter, defaultdict
from statistics import median
from typing import Any


DEFAULT_RADIO_CONFIDENCE_THRESHOLD = 0.86
GENERIC_CHECKBOX_RE = re.compile(r"^\s*(checkbox|option)\s*\d*\s*$", re.IGNORECASE)


VOCAB_GROUPS: tuple[dict[str, Any], ...] = (
    {
        "kind": "yes_no",
        "values": ("yes", "no"),
        "confidence": 0.92,
    },
    {
        "kind": "gender",
        "values": ("male", "female"),
        "confidence": 0.91,
    },
    {
        "kind": "holding_mode",
        "values": ("single", "joint", "anyone or survivor", "either or survivor", "former or survivor"),
        "confidence": 0.90,
    },
    {
        "kind": "account_type",
        "values": ("savings", "current"),
        "confidence": 0.89,
    },
    {
        "kind": "residency",
        "values": ("resident", "nri", "non resident", "non-resident"),
        "confidence": 0.89,
    },
    {
        "kind": "salutation",
        "values": ("mr", "mrs", "ms", "miss", "dr"),
        "confidence": 0.88,
    },
)


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


def radio_grouping_enabled() -> bool:
    return _bool_env("FORM_PARSER_RADIO_GROUPING_ENABLED", False)


def radio_confidence_threshold() -> float:
    return max(0.0, min(1.0, _float_env("FORM_PARSER_RADIO_MIN_CONFIDENCE", DEFAULT_RADIO_CONFIDENCE_THRESHOLD)))


def _clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


def _canonical_text(value: Any) -> str:
    text = _clean_text(value).lower()
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _slug(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", _canonical_text(value)).strip("_") or "radio"


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if parsed != parsed:
        return default
    return parsed


def _normalize_box(box: dict[str, Any] | None) -> dict[str, float] | None:
    if not isinstance(box, dict):
        return None
    x = _safe_float(box.get("x", box.get("Left")), -1.0)
    y = _safe_float(box.get("y", box.get("Top")), -1.0)
    width = _safe_float(box.get("width", box.get("Width")), 0.0)
    height = _safe_float(box.get("height", box.get("Height")), 0.0)
    if x < 0 or y < 0 or width <= 0 or height <= 0:
        return None
    width = max(0.0001, min(width, 1.0 - x))
    height = max(0.0001, min(height, 1.0 - y))
    return {"x": max(0.0, min(1.0, x)), "y": max(0.0, min(1.0, y)), "width": width, "height": height}


def _round_box(box: dict[str, float]) -> dict[str, float]:
    return {
        "x": round(float(box["x"]), 6),
        "y": round(float(box["y"]), 6),
        "width": round(float(box["width"]), 6),
        "height": round(float(box["height"]), 6),
    }


def _box_right(box: dict[str, float]) -> float:
    return float(box["x"]) + float(box["width"])


def _box_bottom(box: dict[str, float]) -> float:
    return float(box["y"]) + float(box["height"])


def _box_center(box: dict[str, float]) -> tuple[float, float]:
    return float(box["x"]) + float(box["width"]) / 2.0, float(box["y"]) + float(box["height"]) / 2.0


def _is_checked(mapping: dict[str, Any]) -> bool:
    for key in ("checked", "is_checked", "is_selected", "selected"):
        value = mapping.get(key)
        if isinstance(value, bool):
            return value
    return str(mapping.get("value") or "").strip().lower() in {"[x]", "x", "yes", "true", "1", "selected", "checked", "on"}


def _section_id(mapping: dict[str, Any]) -> str:
    section = mapping.get("section") if isinstance(mapping.get("section"), dict) else {}
    return str(section.get("section_id") or "document")


def _section_title(mapping: dict[str, Any]) -> str:
    section = mapping.get("section") if isinstance(mapping.get("section"), dict) else {}
    return _clean_text(section.get("title")) or "Document"


def _page(mapping: dict[str, Any]) -> int:
    try:
        return max(1, int(mapping.get("page") or 1))
    except (TypeError, ValueError):
        return 1


def _text_box_candidates(text_boxes: list[dict[str, Any]] | None, checkbox: dict[str, Any]) -> list[dict[str, Any]]:
    cb_box = _normalize_box(checkbox.get("bbox"))
    if cb_box is None:
        return []
    cb_cx, cb_cy = _box_center(cb_box)
    page = _page(checkbox)
    candidates: list[dict[str, Any]] = []
    for item in text_boxes or []:
        if int(item.get("page") or 1) != page:
            continue
        text = _clean_text(item.get("text"))
        if not text:
            continue
        text_box = _normalize_box(item.get("bbox"))
        if text_box is None:
            continue
        tx, ty = _box_center(text_box)
        row_tolerance = max(float(cb_box["height"]) * 1.7, float(text_box["height"]) * 1.2, 0.014)
        if abs(ty - cb_cy) > row_tolerance:
            continue
        dx = float(text_box["x"]) - _box_right(cb_box)
        left_dx = float(cb_box["x"]) - _box_right(text_box)
        if 0 <= dx <= max(float(cb_box["height"]) * 8.0, 0.13):
            candidates.append({"text": text, "bbox": text_box, "score": 1.0 - dx / 0.15, "side": "right"})
        elif 0 <= left_dx <= max(float(cb_box["height"]) * 5.0, 0.08):
            candidates.append({"text": text, "bbox": text_box, "score": 0.58 - left_dx / 0.16, "side": "left"})
        elif text_box["x"] <= cb_cx <= _box_right(text_box):
            candidates.append({"text": text, "bbox": text_box, "score": 0.42, "side": "overlap"})
    return sorted(candidates, key=lambda item: item["score"], reverse=True)


def _option_label(mapping: dict[str, Any], text_boxes: list[dict[str, Any]] | None = None) -> tuple[str, list[str]]:
    reasons: list[str] = []
    explicit = _clean_text(mapping.get("option_label") or mapping.get("export_value"))
    if explicit:
        reasons.append("explicit_option_label")
        return explicit, reasons

    label = _clean_text(mapping.get("label"))
    if label and not GENERIC_CHECKBOX_RE.match(label):
        reasons.append("mapping_label")
        return label, reasons

    candidates = _text_box_candidates(text_boxes, mapping)
    if candidates:
        reasons.append(f"nearby_text_{candidates[0]['side']}")
        return candidates[0]["text"], reasons

    reasons.append("missing_option_label")
    return label or "Checkbox", reasons


def _vocab_hit(label: str) -> tuple[str | None, float, str | None]:
    canonical = _canonical_text(label)
    if not canonical:
        return None, 0.0, None
    for group in VOCAB_GROUPS:
        for value in group["values"]:
            value_norm = _canonical_text(value)
            if canonical == value_norm:
                return str(group["kind"]), float(group["confidence"]), value_norm
    return None, 0.0, None


def _option_like_label(label: str) -> bool:
    canonical = _canonical_text(label)
    if not canonical:
        return False
    tokens = canonical.split()
    if len(tokens) > 4:
        return False
    sentence_words = {"agree", "acknowledge", "consent", "authorize", "accept", "terms", "conditions", "updates"}
    if any(token in sentence_words for token in tokens):
        return False
    return len(canonical) <= 32


def _spatial_quality(items: list[dict[str, Any]]) -> tuple[float, str, dict[str, Any]]:
    centers = [_box_center(item["bbox"]) for item in items]
    xs = [point[0] for point in centers]
    ys = [point[1] for point in centers]
    widths = [float(item["bbox"]["width"]) for item in items]
    heights = [float(item["bbox"]["height"]) for item in items]
    med_h = median(heights)
    med_w = median(widths)
    y_spread = max(ys) - min(ys)
    x_spread = max(xs) - min(xs)
    same_row = y_spread <= max(med_h * 1.35, 0.018)
    same_col = x_spread <= max(med_w * 1.55, 0.018)
    if same_row:
        ordered = sorted(items, key=lambda item: item["bbox"]["x"])
        gaps = [
            max(0.0, float(right["bbox"]["x"]) - _box_right(left["bbox"]))
            for left, right in zip(ordered, ordered[1:])
        ]
        gap_quality = _regularity(gaps)
        return (
            min(1.0, 0.80 + gap_quality * 0.18),
            "horizontal",
            {"same_row": True, "same_column": False, "gap_quality": round(gap_quality, 4), "y_spread": round(y_spread, 6)},
        )
    if same_col:
        ordered = sorted(items, key=lambda item: item["bbox"]["y"])
        gaps = [
            max(0.0, float(right["bbox"]["y"]) - _box_bottom(left["bbox"]))
            for left, right in zip(ordered, ordered[1:])
        ]
        gap_quality = _regularity(gaps)
        return (
            min(1.0, 0.74 + gap_quality * 0.18),
            "vertical",
            {"same_row": False, "same_column": True, "gap_quality": round(gap_quality, 4), "x_spread": round(x_spread, 6)},
        )
    return (
        0.35,
        "scattered",
        {"same_row": False, "same_column": False, "x_spread": round(x_spread, 6), "y_spread": round(y_spread, 6)},
    )


def _regularity(values: list[float]) -> float:
    if len(values) <= 1:
        return 1.0
    center = median(values)
    if center <= 0:
        return 0.85
    deviation = median(abs(value - center) for value in values)
    return max(0.0, min(1.0, 1.0 - deviation / center))


def _question_key(items: list[dict[str, Any]], kind: str) -> str:
    first = items[0]["mapping"]
    section = _slug(_section_title(first))
    page = _page(first)
    ys = [_box_center(item["bbox"])[1] for item in items]
    row_bucket = int(round(median(ys) * 100))
    return f"p{page}_{section}_{kind}_{row_bucket}"


def _candidate_groups(items: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    by_scope: dict[tuple[int, str, str], list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        if item.get("vocab_kind"):
            key = (_page(item["mapping"]), _section_id(item["mapping"]), str(item["vocab_kind"]))
            by_scope[key].append(item)

    groups: list[list[dict[str, Any]]] = []
    seen: set[tuple[str, ...]] = set()
    for scoped in by_scope.values():
        values = {item["canonical_value"] for item in scoped}
        if len(scoped) >= 2 and len(values) >= 2:
            group = sorted(scoped, key=lambda item: (item["bbox"]["y"], item["bbox"]["x"]))
            key = tuple(sorted(item["field_id"] for item in group))
            if key not in seen:
                groups.append(group)
                seen.add(key)

    # Geometry-only fallback: same page + same section + tight row/column, but
    # only when every option has a non-generic label. This catches 3/4-option
    # enterprise groups outside the built-in vocabulary without crossing rows.
    by_section: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        by_section[(_page(item["mapping"]), _section_id(item["mapping"]))].append(item)
    for scoped in by_section.values():
        ordered = sorted(scoped, key=lambda item: (item["bbox"]["y"], item["bbox"]["x"]))
        for row in _row_clusters(ordered):
            if len(row) < 2:
                continue
            if any(item["generic_label"] for item in row):
                continue
            if not all(_option_like_label(item["option_label"]) for item in row):
                continue
            spatial_score, _, _ = _spatial_quality(row)
            if spatial_score < 0.84:
                continue
            key = tuple(sorted(item["field_id"] for item in row))
            if key not in seen:
                groups.append(row)
                seen.add(key)
    return groups


def _row_clusters(items: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    rows: list[list[dict[str, Any]]] = []
    for item in items:
        _, cy = _box_center(item["bbox"])
        placed = False
        for row in rows:
            row_y = median(_box_center(existing["bbox"])[1] for existing in row)
            row_h = median(float(existing["bbox"]["height"]) for existing in row)
            if abs(cy - row_y) <= max(row_h * 1.8, 0.024):
                row.append(item)
                placed = True
                break
        if not placed:
            rows.append([item])
    return [sorted(row, key=lambda item: item["bbox"]["x"]) for row in rows]


def _score_group(group: list[dict[str, Any]]) -> dict[str, Any]:
    if len(group) < 2:
        return {"accepted": False, "confidence": 0.0, "reason": "too_few_options"}

    pages = {_page(item["mapping"]) for item in group}
    sections = {_section_id(item["mapping"]) for item in group}
    if len(pages) != 1:
        return {"accepted": False, "confidence": 0.0, "reason": "cross_page_candidate"}
    if len(sections) != 1:
        return {"accepted": False, "confidence": 0.0, "reason": "cross_section_candidate"}

    canonical_values = [item["canonical_value"] for item in group if item.get("canonical_value")]
    if len(canonical_values) != len(set(canonical_values)):
        return {"accepted": False, "confidence": 0.0, "reason": "duplicate_option_values"}

    vocab_kinds = [item.get("vocab_kind") for item in group if item.get("vocab_kind")]
    vocab_consistent = len(set(vocab_kinds)) == 1 and len(vocab_kinds) == len(group)
    vocab_score = median([item.get("vocab_score", 0.0) for item in group]) if vocab_consistent else 0.0
    spatial_score, layout, geometry_evidence = _spatial_quality(group)
    section_score = median(
        [
            float((item["mapping"].get("section") or {}).get("confidence") or 1.0)
            if isinstance(item["mapping"].get("section"), dict)
            else 1.0
            for item in group
        ]
    )
    label_score = 0.92 if all(not item.get("generic_label") for item in group) else 0.55

    if vocab_consistent:
        confidence = min(1.0, 0.42 * vocab_score + 0.34 * spatial_score + 0.16 * section_score + 0.08 * label_score)
        reason = f"vocabulary_{vocab_kinds[0]}_{layout}"
    else:
        confidence = min(1.0, 0.58 * spatial_score + 0.24 * section_score + 0.18 * label_score)
        reason = f"geometry_{layout}_same_section"

    if spatial_score < 0.72:
        return {"accepted": False, "confidence": round(confidence, 4), "reason": "weak_spatial_alignment"}
    if not vocab_consistent and len(group) > 4:
        return {"accepted": False, "confidence": round(confidence, 4), "reason": "geometry_group_too_large_without_vocab"}

    return {
        "accepted": True,
        "confidence": round(confidence, 4),
        "reason": reason,
        "layout": layout,
        "geometry_evidence": geometry_evidence,
        "vocabulary_kind": vocab_kinds[0] if vocab_consistent else None,
    }


def _prepared_items(mappings: list[dict[str, Any]], text_boxes: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for index, mapping in enumerate(mappings or []):
        if not isinstance(mapping, dict):
            continue
        if mapping.get("field_type") != "checkbox":
            continue
        if mapping.get("widget_type") and mapping.get("widget_type") != "radio":
            continue
        box = _normalize_box(mapping.get("bbox"))
        if box is None:
            continue
        label, label_reasons = _option_label(mapping, text_boxes)
        canonical = _canonical_text(label)
        kind, vocab_score, canonical_value = _vocab_hit(label)
        generic = bool(GENERIC_CHECKBOX_RE.match(_clean_text(mapping.get("label")))) and not canonical_value
        items.append(
            {
                "index": index,
                "field_id": str(mapping.get("field_id") or f"checkbox_{index}"),
                "mapping": mapping,
                "bbox": box,
                "option_label": label,
                "canonical_value": canonical_value or canonical,
                "vocab_kind": kind,
                "vocab_score": vocab_score,
                "generic_label": generic,
                "label_reasons": label_reasons,
                "checked": _is_checked(mapping),
            }
        )
    return items


def _group_name(group: list[dict[str, Any]], score: dict[str, Any]) -> str:
    kind = score.get("vocabulary_kind") or "radio"
    return _question_key(group, str(kind))


def _export_value(item: dict[str, Any]) -> str:
    value = _clean_text(item.get("canonical_value") or item.get("option_label"))
    if not value:
        value = item["field_id"]
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "Option"


def apply_radio_grouping(
    mappings: list[dict[str, Any]],
    *,
    text_boxes: list[dict[str, Any]] | None = None,
    enabled: bool | None = None,
    threshold: float | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    enabled = radio_grouping_enabled() if enabled is None else bool(enabled)
    threshold = radio_confidence_threshold() if threshold is None else max(0.0, min(1.0, float(threshold)))
    if not enabled:
        return list(mappings or []), {
            "enabled": False,
            "feature_flag": "FORM_PARSER_RADIO_GROUPING_ENABLED",
            "threshold": threshold,
            "group_count": 0,
            "groups": [],
            "rejected_candidates": [],
            "reasoning": "Radio grouping is disabled; checkbox mappings are unchanged.",
        }

    prepared = _prepared_items(mappings, text_boxes)
    enriched = [dict(mapping) if isinstance(mapping, dict) else mapping for mapping in (mappings or [])]
    accepted_groups: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    assigned_ids: set[str] = set()

    for group in _candidate_groups(prepared):
        group_ids = {item["field_id"] for item in group}
        if group_ids & assigned_ids:
            continue
        score = _score_group(group)
        if not score.get("accepted") or float(score.get("confidence") or 0.0) < threshold:
            rejected.append(
                {
                    "field_ids": sorted(group_ids),
                    "labels": [item["option_label"] for item in group],
                    "confidence": score.get("confidence", 0.0),
                    "reason": score.get("reason", "below_threshold"),
                    "threshold": threshold,
                }
            )
            continue

        radio_group = _group_name(group, score)
        selected_count = sum(1 for item in group if item.get("checked"))
        group_payload = {
            "radio_group": radio_group,
            "page": _page(group[0]["mapping"]),
            "section": group[0]["mapping"].get("section"),
            "option_count": len(group),
            "selected_count": selected_count,
            "confidence": score["confidence"],
            "reason": score["reason"],
            "layout": score["layout"],
            "labels": [item["option_label"] for item in group],
            "field_ids": [item["field_id"] for item in group],
            "geometry_evidence": score.get("geometry_evidence", {}),
        }
        accepted_groups.append(group_payload)
        assigned_ids.update(group_ids)
        selected_seen = False
        for item in group:
            index = int(item["index"])
            updated = dict(enriched[index])
            radio_selected = bool(item.get("checked")) and not selected_seen
            if radio_selected:
                selected_seen = True
            updated["widget_type"] = "radio"
            updated["radio_group"] = radio_group
            updated["export_value"] = _export_value(item)
            updated["radio_selected"] = radio_selected
            updated["option_label"] = item["option_label"]
            updated["radio_confidence"] = score["confidence"]
            updated["radio_reason"] = score["reason"]
            updated["radio_group_size"] = len(group)
            updated["radio_detection"] = {
                "confidence": score["confidence"],
                "reason": score["reason"],
                "layout": score["layout"],
                "vocabulary_kind": score.get("vocabulary_kind"),
                "option_label": item["option_label"],
                "label_reasons": item.get("label_reasons", []),
            }
            enriched[index] = updated

    reason_counts = Counter(group["reason"] for group in accepted_groups)
    return enriched, {
        "enabled": True,
        "feature_flag": "FORM_PARSER_RADIO_GROUPING_ENABLED",
        "threshold": threshold,
        "group_count": len(accepted_groups),
        "groups": accepted_groups,
        "rejected_count": len(rejected),
        "rejected_candidates": rejected[:50],
        "reason_counts": dict(sorted(reason_counts.items())),
        "candidate_checkbox_count": len(prepared),
        "reasoning": (
            "Radio groups are emitted only for same-page, same-section checkbox "
            "sets with strong vocabulary and/or spatial evidence."
        ),
    }
