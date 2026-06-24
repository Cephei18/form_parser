"""Box geometry for evaluation.

Boxes use the *same* convention as the pipeline ``mappings.json`` ``bbox`` field:
a dict ``{"x", "y", "width", "height"}`` in **page-local fractions** (0..1) with
a top-left origin. Keeping the convention identical means ground-truth boxes and
predicted boxes are directly comparable with no coordinate translation.
"""
from __future__ import annotations

from typing import Any

Box = dict[str, float]


def normalize_box(raw: Any) -> Box | None:
    """Coerce a loosely-typed box into the canonical fraction box.

    Accepts the pipeline shape ``{x,y,width,height}`` and also tolerates the
    Textract-style ``{Left,Top,Width,Height}`` so annotators can paste either.
    Returns ``None`` for anything unparseable (caller decides how to handle).
    """
    if not isinstance(raw, dict):
        return None
    try:
        x = float(raw.get("x", raw.get("Left", raw.get("left"))))
        y = float(raw.get("y", raw.get("Top", raw.get("top"))))
        w = float(raw.get("width", raw.get("Width")))
        h = float(raw.get("height", raw.get("Height")))
    except (TypeError, ValueError):
        return None
    if w < 0 or h < 0:
        return None
    return {"x": x, "y": y, "width": w, "height": h}


def area(box: Box) -> float:
    return max(0.0, float(box["width"])) * max(0.0, float(box["height"]))


def _right(box: Box) -> float:
    return float(box["x"]) + float(box["width"])


def _bottom(box: Box) -> float:
    return float(box["y"]) + float(box["height"])


def intersection_area(a: Box, b: Box) -> float:
    left = max(float(a["x"]), float(b["x"]))
    top = max(float(a["y"]), float(b["y"]))
    right = min(_right(a), _right(b))
    bottom = min(_bottom(a), _bottom(b))
    if right <= left or bottom <= top:
        return 0.0
    return (right - left) * (bottom - top)


def iou(a: Box, b: Box) -> float:
    """Intersection-over-union of two fraction boxes."""
    inter = intersection_area(a, b)
    if inter <= 0.0:
        return 0.0
    union = area(a) + area(b) - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def containment(inner: Box, outer: Box) -> float:
    """Fraction of ``inner`` covered by ``outer`` (asymmetric overlap).

    Used for fragmentation: how much of a predicted box lands inside a GT group.
    """
    a = area(inner)
    if a <= 0.0:
        return 0.0
    return intersection_area(inner, outer) / a
