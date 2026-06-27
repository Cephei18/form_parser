"""Generalized field-hygiene pass for the Textract anchoring engine.

Textract's ``KEY_VALUE_SET`` output on real-world scanned forms produces three
recurring classes of bad widget. This module centralises the detection of all
three so the anchoring engine can drop / re-route them deterministically:

1. **Degenerate answer regions** — a KEY whose VALUE block collapses to a ~1px
   point. This is commonly triggered by stray printed glyphs such as the
   ``(   )`` parentheses around a phone area code: Textract emits ``(``, ``)``
   and ``( )`` as their own tokens / value blocks, so the "answer" for the
   neighbouring label becomes a zero-area point. Rendered, that point is an
   invisible / unusable widget — and because the value-block candidate scores
   ~0.94 it *out-ranks the real underline* sitting next to the label. Dropping
   the degenerate candidate lets the genuine underline / table cell win.

2. **Non-fillable page furniture** — watermarks, logos, source URLs and footer
   copyright lines (e.g. "SampleWords", a garbled "... downloaded ...") that
   Textract promotes to KEYs but which are not form fields. These are matched
   by boilerplate text patterns and by living in the extreme header/footer
   margin with no real value.

3. **Duplicate widgets** — two fields anchored onto (essentially) the same
   physical answer region (e.g. a label and its printed sub-caption both
   resolving to the same underline). The weaker duplicate is dropped.

Every rule only ever *removes clearly-bad widgets* or *re-routes to a better
candidate*, so the pass is ON by default. It can be disabled wholesale with
``FORM_PARSER_FIELD_HYGIENE_ENABLED=false`` for a deterministic rollback.
"""
from __future__ import annotations

import os
import re
from typing import Any

# --- Tunables (page fractions) ------------------------------------------------
# A writable answer region narrower/shorter than this cannot hold a glyph; any
# candidate below it is a collapsed Textract value point, not a real field.
DEGENERATE_MIN_WIDTH = 0.006
DEGENERATE_MIN_HEIGHT = 0.004

# Extreme top/bottom bands that hold page furniture (titles, page numbers,
# copyright / source-watermark lines) rather than fillable fields.
FOOTER_FURNITURE_Y = 0.955
HEADER_FURNITURE_Y = 0.035

# Two emitted fields whose answer boxes overlap by more than this (relative to
# the smaller box) are treated as the same physical widget.
DUPLICATE_OVERLAP = 0.8

# ...but only when the boxes are comparably sized. A large multiline answer
# region (e.g. a 5-line address block) can geometrically *contain* a small,
# distinct field (e.g. a "Pin" box on its last line): the smaller box overlaps
# the larger by ~100%, yet they are different widgets, not duplicates. Requiring
# the smaller box to be at least this fraction of the larger one's area keeps the
# genuine duplicate case (near-equal boxes) while sparing the container case.
DUPLICATE_MIN_AREA_RATIO = 0.35

# Anchor types that are real printed structure — never treated as degenerate
# even if the detected box is thin (a checkbox glyph is legitimately tiny).
_STRUCTURAL_SMALL_ANCHORS = {"checkbox_region"}

# Substrings / patterns that mark a KEY as boilerplate rather than a field.
# Matched case-insensitively against the normalised label text.
_BOILERPLATE_PATTERNS = (
    re.compile(r"samplewords"),
    re.compile(r"\ball rights reserved\b"),
    re.compile(r"copyright|©|\(c\)\s*\d"),
    re.compile(r"\bpage\s+\d+\s+of\s+\d+\b"),
    re.compile(r"https?://|www\.|\.com\b|\.org\b|\.net\b"),
    re.compile(r"\bthis form (is|was) (available|downloaded)"),
    # Printed form-control / version codes in the footer (e.g. "FO/Reg.",
    # "Form/Ver. 2.0/April'23"). These are document metadata, never fields.
    re.compile(r"\bfo\s*/\s*reg\b"),
    re.compile(r"\bform\s*/\s*ver\b"),
    re.compile(r"\bver\.?\s*\d"),
)


def _bool_env(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def field_hygiene_enabled() -> bool:
    """True when the generalized field-hygiene pass is active (default ON)."""
    return _bool_env("FORM_PARSER_FIELD_HYGIENE_ENABLED", True)


# --- Geometry (standalone so this module never imports the engine) ------------
def _box_area(box: dict[str, float]) -> float:
    return max(0.0, float(box.get("width", 0.0))) * max(0.0, float(box.get("height", 0.0)))


def _intersection_area(a: dict[str, float], b: dict[str, float]) -> float:
    ax2, ay2 = float(a["x"]) + float(a["width"]), float(a["y"]) + float(a["height"])
    bx2, by2 = float(b["x"]) + float(b["width"]), float(b["y"]) + float(b["height"])
    x1, y1 = max(float(a["x"]), float(b["x"])), max(float(a["y"]), float(b["y"]))
    x2, y2 = min(ax2, bx2), min(ay2, by2)
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _smaller_overlap_ratio(a: dict[str, float], b: dict[str, float]) -> float:
    inter = _intersection_area(a, b)
    return inter / max(min(_box_area(a), _box_area(b)), 1e-9)


def _area_ratio(a: dict[str, float], b: dict[str, float]) -> float:
    """Smaller-box area / larger-box area (1.0 == identical size)."""
    aa, ba = _box_area(a), _box_area(b)
    return min(aa, ba) / max(max(aa, ba), 1e-9)


def _center_y(box: dict[str, float]) -> float:
    return float(box["y"]) + float(box["height"]) / 2.0


# --- Rule 1: degenerate answer regions ---------------------------------------
def is_degenerate_box(
    box: dict[str, float] | None,
    *,
    min_width: float = DEGENERATE_MIN_WIDTH,
    min_height: float = DEGENERATE_MIN_HEIGHT,
) -> bool:
    """True when a box is too small to be a real writable answer region."""
    if not isinstance(box, dict):
        return True
    try:
        return float(box["width"]) < min_width or float(box["height"]) < min_height
    except (KeyError, TypeError, ValueError):
        return True


def drop_degenerate_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Filter collapsed (~1px) candidates so a real feature wins selection.

    Structural checkbox glyphs are exempt (they are legitimately tiny). When
    *every* candidate is degenerate the list is returned unchanged so the caller
    still has something to fall back on rather than losing the field outright.
    """
    if not field_hygiene_enabled():
        return candidates
    kept = [
        c
        for c in candidates
        if c.get("anchor_type") in _STRUCTURAL_SMALL_ANCHORS
        or not is_degenerate_box(c.get("bbox"))
    ]
    return kept if kept else candidates


# --- Rule 2: non-fillable page furniture -------------------------------------
def is_boilerplate_label(label: str) -> bool:
    """True when the label text is a watermark / copyright / source-URL line."""
    text = re.sub(r"\s+", " ", str(label or "")).strip().lower()
    if not text:
        return False
    return any(pattern.search(text) for pattern in _BOILERPLATE_PATTERNS)


def is_page_furniture(
    label: str,
    label_box: dict[str, float] | None,
    answer_box: dict[str, float] | None,
    value: str,
    *,
    field_type: str = "text",
) -> tuple[bool, str]:
    """Decide whether a resolved field is non-fillable page furniture.

    Returns ``(is_furniture, reason)``. Checkboxes / signatures / photos are
    never furniture — only text-like KEYs Textract invented from decorative or
    marginal print.
    """
    if not field_hygiene_enabled():
        return False, ""
    if field_type in {"checkbox", "signature", "photo", "comb", "radio"}:
        return False, ""

    if is_boilerplate_label(label):
        return True, "boilerplate_label"

    # Extreme header/footer band: page numbers, source URLs, copyright. Only
    # treat as furniture when there is no meaningful captured value, so a real
    # answer that happens to sit low on the page is preserved.
    boxes = [b for b in (label_box, answer_box) if isinstance(b, dict)]
    if boxes and not str(value or "").strip():
        in_margin = all(
            _center_y(b) >= FOOTER_FURNITURE_Y or _center_y(b) <= HEADER_FURNITURE_Y
            for b in boxes
        )
        if in_margin:
            return True, "margin_furniture"
    return False, ""


# --- Rule 3: duplicate widgets ------------------------------------------------
def dedupe_overlapping_fields(mappings: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    """Drop fields whose answer region duplicates a stronger field's region.

    Two text-like fields on the same page whose answer boxes overlap by more
    than :data:`DUPLICATE_OVERLAP` (relative to the smaller box) are the same
    physical widget — keep the higher-confidence one, drop the other. Checkboxes,
    photos and grouped widgets (comb/radio) are never deduped here. Returns the
    kept mappings (input order preserved) and the dropped ``field_id`` list.
    """
    if not field_hygiene_enabled():
        return mappings, []

    protected = {"checkbox", "photo", "signature"}
    eligible_idx = [
        i
        for i, m in enumerate(mappings)
        if m.get("field_type") not in protected
        and not m.get("widget_type")  # leave comb/radio groups intact
        and isinstance(m.get("bbox"), dict)
        and int(m.get("page") or 1)
    ]
    # Strongest first; ties keep earlier (lower field index) by stable sort.
    order = sorted(eligible_idx, key=lambda i: -float(mappings[i].get("confidence") or 0.0))
    kept_boxes: list[tuple[int, dict[str, float]]] = []
    dropped: set[int] = set()
    for i in order:
        box = mappings[i]["bbox"]
        page = int(mappings[i].get("page") or 1)
        if any(
            page == kp
            and _smaller_overlap_ratio(box, kb) >= DUPLICATE_OVERLAP
            and _area_ratio(box, kb) >= DUPLICATE_MIN_AREA_RATIO
            for kp, kb in kept_boxes
        ):
            dropped.add(i)
            continue
        kept_boxes.append((page, box))

    kept = [m for i, m in enumerate(mappings) if i not in dropped]
    dropped_ids = [str(mappings[i].get("field_id")) for i in sorted(dropped)]
    return kept, dropped_ids
