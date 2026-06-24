"""Phase L0.3 — document-level confidence calibration (additive, flag-gated).

The existing confidence pipeline scores each mapping in isolation. Three of the
production failure modes are only visible *across* mappings:

* **duplicate mappings** — two widgets occupying the same physical region,
* **checkbox explosions** — an implausible cluster of checkboxes on one page,
* **ambiguous assignments** — a widget whose top anchor candidates were a near
  tie / a region contested by several fields.

This pass computes those document-level signals and attaches a calibration
penalty. To stay strictly backward-compatible it does **not** mutate the
existing ``confidence_score`` or any render decision — it adds two new keys:
``calibration`` (penalty + reasons) and ``calibrated_confidence_score``. Render
policy, output contract, and the original score are untouched.

Gated behind ``FORM_PARSER_CONFIDENCE_CALIBRATION_ENABLED`` (default OFF): when
OFF mappings are returned unchanged and only an ``{"enabled": False}``
diagnostic is produced.
"""
from __future__ import annotations

import os
from collections import defaultdict
from typing import Any

DEFAULT_DUPLICATE_IOU = 0.6
DEFAULT_CHECKBOX_EXPLOSION = 12  # checkboxes on one page above which we penalize
DEFAULT_AMBIGUITY_GAP = 0.06

DUPLICATE_PENALTY = 0.15
CHECKBOX_EXPLOSION_PENALTY = 0.10
AMBIGUITY_PENALTY = 0.10
MAX_PENALTY = 0.40


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


def _int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def confidence_calibration_enabled() -> bool:
    return _bool_env("FORM_PARSER_CONFIDENCE_CALIBRATION_ENABLED", False)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _box(mapping: dict[str, Any]) -> dict[str, float] | None:
    box = mapping.get("bbox")
    if not isinstance(box, dict):
        return None
    try:
        return {"x": float(box["x"]), "y": float(box["y"]), "width": float(box["width"]), "height": float(box["height"])}
    except (KeyError, TypeError, ValueError):
        return None


def _iou(a: dict[str, float], b: dict[str, float]) -> float:
    ax2, ay2 = a["x"] + a["width"], a["y"] + a["height"]
    bx2, by2 = b["x"] + b["width"], b["y"] + b["height"]
    left, top = max(a["x"], b["x"]), max(a["y"], b["y"])
    right, bottom = min(ax2, bx2), min(ay2, by2)
    if right <= left or bottom <= top:
        return 0.0
    inter = (right - left) * (bottom - top)
    union = a["width"] * a["height"] + b["width"] * b["height"] - inter
    return inter / union if union > 0 else 0.0


def _base_score(mapping: dict[str, Any]) -> float:
    for key in ("confidence_score", "candidate_score", "confidence"):
        value = mapping.get(key)
        try:
            if value is not None:
                return _clamp01(float(value))
        except (TypeError, ValueError):
            continue
    return 0.55


def _ambiguity_signal(mapping: dict[str, Any], gap_threshold: float) -> bool:
    anchoring = mapping.get("anchoring") if isinstance(mapping.get("anchoring"), dict) else {}
    candidate_count = anchoring.get("candidate_count")
    try:
        many = int(candidate_count) >= 6
    except (TypeError, ValueError):
        many = False
    top = anchoring.get("top_candidates") if isinstance(anchoring.get("top_candidates"), list) else []
    scores: list[float] = []
    for cand in top[:2]:
        if isinstance(cand, dict):
            val = cand.get("score", cand.get("candidate_score"))
            try:
                scores.append(float(val))
            except (TypeError, ValueError):
                pass
    close = len(scores) >= 2 and abs(scores[0] - scores[1]) <= gap_threshold
    return many or close


def apply_confidence_calibration(
    mappings: list[dict[str, Any]] | None,
    *,
    enabled: bool | None = None,
) -> dict[str, Any]:
    enabled = confidence_calibration_enabled() if enabled is None else bool(enabled)
    mappings = mappings or []
    if not enabled:
        return {"mappings": mappings, "diagnostics": {"enabled": False, "feature_flag": "FORM_PARSER_CONFIDENCE_CALIBRATION_ENABLED"}}

    duplicate_iou = _float_env("FORM_PARSER_CALIBRATION_DUP_IOU", DEFAULT_DUPLICATE_IOU)
    explosion_limit = _int_env("FORM_PARSER_CALIBRATION_CHECKBOX_EXPLOSION", DEFAULT_CHECKBOX_EXPLOSION)
    gap_threshold = _float_env("FORM_PARSER_CALIBRATION_AMBIGUITY_GAP", DEFAULT_AMBIGUITY_GAP)

    indexed = [(i, m) for i, m in enumerate(mappings) if isinstance(m, dict)]

    # 1) Duplicate clusters (overlapping boxes on the same page).
    duplicate_ids: set[int] = set()
    by_page: dict[int, list[tuple[int, dict[str, float]]]] = defaultdict(list)
    for i, m in indexed:
        box = _box(m)
        if box is not None:
            by_page[int(m.get("page") or 1)].append((i, box))
    duplicate_pairs = 0
    for page_boxes in by_page.values():
        for a in range(len(page_boxes)):
            for b in range(a + 1, len(page_boxes)):
                if _iou(page_boxes[a][1], page_boxes[b][1]) >= duplicate_iou:
                    duplicate_ids.add(page_boxes[a][0])
                    duplicate_ids.add(page_boxes[b][0])
                    duplicate_pairs += 1

    # 2) Checkbox explosions (per-page checkbox over-count).
    checkbox_by_page: dict[int, list[int]] = defaultdict(list)
    for i, m in indexed:
        if m.get("field_type") == "checkbox":
            checkbox_by_page[int(m.get("page") or 1)].append(i)
    exploded_ids: set[int] = set()
    exploded_pages: dict[int, int] = {}
    for page, ids in checkbox_by_page.items():
        if len(ids) > explosion_limit:
            exploded_pages[page] = len(ids)
            exploded_ids.update(ids)

    # 3) Ambiguous assignments.
    ambiguous_ids = {i for i, m in indexed if _ambiguity_signal(m, gap_threshold)}

    penalized = 0
    total_penalty = 0.0
    for i, m in indexed:
        reasons: list[str] = []
        penalty = 0.0
        if i in duplicate_ids:
            penalty += DUPLICATE_PENALTY
            reasons.append("duplicate_region")
        if i in exploded_ids:
            penalty += CHECKBOX_EXPLOSION_PENALTY
            reasons.append("checkbox_explosion")
        if i in ambiguous_ids:
            penalty += AMBIGUITY_PENALTY
            reasons.append("ambiguous_assignment")
        penalty = min(MAX_PENALTY, penalty)
        base = _base_score(m)
        m["calibration"] = {"penalty": round(penalty, 4), "reasons": reasons}
        m["calibrated_confidence_score"] = round(_clamp01(base - penalty), 4)
        if penalty > 0:
            penalized += 1
            total_penalty += penalty

    diagnostics = {
        "enabled": True,
        "feature_flag": "FORM_PARSER_CONFIDENCE_CALIBRATION_ENABLED",
        "mutates_original_score": False,
        "thresholds": {
            "duplicate_iou": duplicate_iou,
            "checkbox_explosion_limit": explosion_limit,
            "ambiguity_gap": gap_threshold,
        },
        "duplicate_pairs": duplicate_pairs,
        "duplicate_mappings": len(duplicate_ids),
        "checkbox_explosion_pages": exploded_pages,
        "checkbox_explosion_mappings": len(exploded_ids),
        "ambiguous_mappings": len(ambiguous_ids),
        "penalized_mappings": penalized,
        "mean_penalty": round(total_penalty / penalized, 4) if penalized else 0.0,
    }
    return {"mappings": mappings, "diagnostics": diagnostics}
