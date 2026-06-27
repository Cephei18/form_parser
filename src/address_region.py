"""Synthesize a multiline answer region for an address label that Textract gave
no KEY_VALUE_SET for.

Some forms print a "MAILING ADDRESS OF FIRST / SOLE APPLICANT" heading with two
blank lines beneath it for the applicant to write the address, then a CITY /
STATE / PIN row. Textract emits no key/value pair for that blank band, so the
anchoring engine — which only builds fields from KEYs — produces nothing.

This module fills that specific gap WITHOUT inventing fields elsewhere. It only
acts on a label whose text matches a qualified-address pattern
("mailing/permanent/correspondence/residential/present address"), only when no
existing field already covers that label (so addresses Textract DID key — e.g.
form_5's "Present Address" — are untouched), and only into the empty band
between the label and the next text row. Two consecutive empty lines under such
a label is the address writing area; if the band is occupied or absent, nothing
is emitted. ON by default; ``FORM_PARSER_ADDRESS_REGION_ENABLED=false`` reverts.
"""
from __future__ import annotations

import os
import re
from typing import Any, Callable

from src.section_detector import SectionIndex, qualify_label, section_summary

# A qualified-address heading. The qualifier must precede "address" so that a
# bare "Address Type:" checkbox prompt is NOT matched.
_ADDRESS_LABEL_RE = re.compile(
    r"\b(mailing|permanent|correspondence|residential|present|overseas|local)\s+address\b",
    re.IGNORECASE,
)

# Vertical search window (page fractions) below the label for the next text row.
_MAX_BAND_BELOW = 0.07
# Minimum writable band height to bother emitting.
_MIN_BAND_HEIGHT = 0.012


def _bool_env(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def address_region_enabled() -> bool:
    """True when key-less address regions are synthesized (default ON)."""
    return _bool_env("FORM_PARSER_ADDRESS_REGION_ENABLED", True)


def _right(b: dict[str, float]) -> float:
    return float(b["x"]) + float(b["width"])


def _bottom(b: dict[str, float]) -> float:
    return float(b["y"]) + float(b["height"])


def _cx(b: dict[str, float]) -> float:
    return float(b["x"]) + float(b["width"]) / 2.0


def _cy(b: dict[str, float]) -> float:
    return float(b["y"]) + float(b["height"]) / 2.0


def _x_overlaps(a: dict[str, float], b: dict[str, float]) -> bool:
    return max(float(a["x"]), float(b["x"])) < min(_right(a), _right(b))


def _overlap_ratio(a: dict[str, float], b: dict[str, float]) -> float:
    ix = max(0.0, min(_right(a), _right(b)) - max(float(a["x"]), float(b["x"])))
    iy = max(0.0, min(_bottom(a), _bottom(b)) - max(float(a["y"]), float(b["y"])))
    inter = ix * iy
    area_a = max(float(a["width"]) * float(a["height"]), 1e-9)
    area_b = max(float(b["width"]) * float(b["height"]), 1e-9)
    return inter / min(area_a, area_b)


def emit_address_regions(
    lines: list[dict[str, Any]],
    *,
    text_boxes: list[dict[str, Any]],
    mappings: list[dict[str, Any]],
    metrics_by_page: dict[int, dict[str, float]],
    section_index: SectionIndex,
    page_px: Callable[[int], tuple[int, int]],
    non_fillable_pages: set[int],
) -> list[dict[str, Any]]:
    """Return synthesized multiline address-field mappings (possibly empty)."""
    if not address_region_enabled():
        return []

    out: list[dict[str, Any]] = []
    for idx, line in enumerate(lines):
        page = int(line.get("page") or 1)
        if page in non_fillable_pages:
            continue
        text = str(line.get("text") or "")
        if not _ADDRESS_LABEL_RE.search(text):
            continue
        label = line.get("bbox")
        if not isinstance(label, dict):
            continue

        # Skip if a field already covers this address label (Textract keyed it):
        # a field whose label/answer box GEOMETRICALLY overlaps the address label
        # (not merely shares its row — the line above must not count). This keeps
        # addresses Textract DID key (e.g. form_5 "Present Address") untouched.
        if any(
            int(m.get("page") or 1) == page
            and isinstance(m.get(key), dict)
            and _overlap_ratio(m[key], label) >= 0.3
            for m in mappings
            for key in ("label_bbox", "bbox")
        ):
            continue

        metrics = metrics_by_page.get(page, metrics_by_page.get(1, {}))
        line_h = float(metrics.get("line_height", 0.014))
        page_right = float(metrics.get("page_right", 0.95))

        # The next text row below the label bounds the writing band.
        band_top = _bottom(label) + line_h * 0.4
        next_below = min(
            (
                float(t["bbox"]["y"])
                for t in text_boxes
                if int(t.get("page") or 1) == page
                and float(t["bbox"]["y"]) >= band_top
                and float(t["bbox"]["y"]) - _bottom(label) <= _MAX_BAND_BELOW
            ),
            default=_bottom(label) + 0.045,
        )
        band_bottom = next_below - line_h * 0.3
        if band_bottom - band_top < _MIN_BAND_HEIGHT:
            continue

        region = {
            "x": round(float(label["x"]), 6),
            "y": round(band_top, 6),
            "width": round(page_right - float(label["x"]), 6),
            "height": round(band_bottom - band_top, 6),
        }

        # The band must be empty: no existing field centre inside, and no text.
        if any(
            int(m.get("page") or 1) == page
            and isinstance(m.get("bbox"), dict)
            and region["x"] <= _cx(m["bbox"]) <= _right(region)
            and region["y"] <= _cy(m["bbox"]) <= _bottom(region)
            for m in mappings
        ):
            continue
        if any(
            int(t.get("page") or 1) == page
            and region["x"] <= _cx(t["bbox"]) <= _right(region)
            and region["y"] <= _cy(t["bbox"]) <= _bottom(region)
            for t in text_boxes
        ):
            continue

        owner = section_index.owner(page, float(region["y"]))
        px_w, px_h = page_px(page)
        clean_label = re.sub(r"\s+", " ", text).strip()
        out.append(
            {
                "field_id": f"address_region_{page}_{idx}",
                "label": clean_label,
                "qualified_label": qualify_label(owner, clean_label),
                "section": section_summary(owner),
                "value": "",
                "field_type": "multiline",
                "bbox": dict(region),
                "label_bbox": dict(label),
                "answer_region": {"bbox": dict(region), "type": "address_region", "confidence": 0.7},
                "page": page,
                "confidence": 0.7,
                "candidate_score": 0.7,
                "confidence_class": "medium",
                "multiline_group_size": 2,
                "field_bboxes": [
                    {
                        "x": float(region["x"]) * px_w,
                        "y": float(region["y"]) * px_h,
                        "width": float(region["width"]) * px_w,
                        "height": float(region["height"]) * px_h,
                    }
                ],
                "source": "textract_address_region",
                "render_border": True,
                "anchoring": {
                    "anchor_type": "address_region",
                    "type_reasons": ["keyless_address_label"],
                    "selection_reasons": ["address_label_empty_band"],
                    "label_overlap_ratio": 0.0,
                    "candidate_count": 1,
                    "top_candidates": [],
                    "key_block_id": None,
                    "value_block_ids": [],
                },
            }
        )
    return out
