"""Phase L0.4 — comb-group detection diagnostics (additive, diagnostics-ONLY).

A "comb" is a row of equal-pitch character cells for a single field (SSN, phone,
account number). Today each cell can leak out as its own widget. This module
*detects* likely comb groups from existing geometry and emits them as
diagnostics — it does **not** merge, suppress, or otherwise change any mapping.
That stays the job of the existing ``comb_detector`` (separately flagged); here
we only measure how often combs appear so a future enforcement decision is
data-driven.

A group is reported when ≥3 small, similar-width boxes sit on one row with a
near-constant horizontal pitch.

Gated behind ``FORM_PARSER_COMB_DIAGNOSTICS_ENABLED`` (default OFF): when OFF it
is never invoked. Even when ON it returns diagnostics only.
"""
from __future__ import annotations

import os
import re
from collections import defaultdict
from typing import Any

DEFAULT_MIN_CELLS = 3
ROW_BAND = 0.012  # fraction: cells within this y-center distance share a row
MAX_CELL_WIDTH = 0.07  # fraction: comb cells are small
MAX_CELL_HEIGHT = 0.06
PITCH_REGULARITY_MIN = 0.7  # 1 - coeff_of_variation(gaps) must clear this


def _bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def comb_diagnostics_enabled() -> bool:
    return _bool_env("FORM_PARSER_COMB_DIAGNOSTICS_ENABLED", False)


def _center(box: dict[str, float]) -> tuple[float, float]:
    return float(box["x"]) + float(box["width"]) / 2.0, float(box["y"]) + float(box["height"]) / 2.0


def _small_box(box: dict[str, float]) -> bool:
    try:
        return 0 < float(box["width"]) <= MAX_CELL_WIDTH and 0 < float(box["height"]) <= MAX_CELL_HEIGHT
    except (KeyError, TypeError, ValueError):
        return False


def _candidate_cells(
    text_boxes: list[dict[str, Any]],
    selection_regions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    for region in selection_regions or []:
        box = region.get("bbox")
        if isinstance(box, dict) and _small_box(box):
            cells.append({"page": int(region.get("page") or 1), "bbox": box, "source": "selection"})
    for tb in text_boxes or []:
        box = tb.get("bbox")
        text = re.sub(r"\s+", "", str(tb.get("text") or ""))
        if isinstance(box, dict) and _small_box(box) and len(text) <= 2:
            cells.append({"page": int(tb.get("page") or 1), "bbox": box, "source": "text", "text": text})
    return cells


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _stddev(values: list[float], mean: float) -> float:
    if len(values) < 2:
        return 0.0
    return (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5


def _union(boxes: list[dict[str, float]]) -> dict[str, float]:
    x0 = min(float(b["x"]) for b in boxes)
    y0 = min(float(b["y"]) for b in boxes)
    x1 = max(float(b["x"]) + float(b["width"]) for b in boxes)
    y1 = max(float(b["y"]) + float(b["height"]) for b in boxes)
    return {"x": round(x0, 6), "y": round(y0, 6), "width": round(x1 - x0, 6), "height": round(y1 - y0, 6)}


def analyze_comb_groups(
    mappings: list[dict[str, Any]] | None = None,
    *,
    text_boxes: list[dict[str, Any]] | None = None,
    selection_regions: list[dict[str, Any]] | None = None,
    min_cells: int | None = None,
) -> dict[str, Any]:
    """Detect equal-pitch cell runs and report them as candidate comb groups."""
    min_cells = min_cells if min_cells is not None else _int_env("FORM_PARSER_COMB_DIAGNOSTICS_MIN_CELLS", DEFAULT_MIN_CELLS)
    cells = _candidate_cells(text_boxes or [], selection_regions or [])

    # Bucket cells into rows: same page, similar y-center.
    rows: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for cell in cells:
        _, cy = _center(cell["bbox"])
        band = int(round(cy / ROW_BAND))
        rows[(cell["page"], band)].append(cell)

    groups: list[dict[str, Any]] = []
    for (page, _band), row_cells in rows.items():
        if len(row_cells) < min_cells:
            continue
        row_cells.sort(key=lambda c: float(c["bbox"]["x"]))
        centers = [_center(c["bbox"])[0] for c in row_cells]
        gaps = [b - a for a, b in zip(centers, centers[1:])]
        if not gaps:
            continue
        gap_mean = _mean(gaps)
        if gap_mean <= 0:
            continue
        regularity = 1.0 - (_stddev(gaps, gap_mean) / gap_mean)
        widths = [float(c["bbox"]["width"]) for c in row_cells]
        width_cv = _stddev(widths, _mean(widths)) / max(_mean(widths), 1e-9)
        if regularity < PITCH_REGULARITY_MIN or width_cv > 0.5:
            continue
        span = _union([c["bbox"] for c in row_cells])
        groups.append(
            {
                "page": page,
                "cell_count": len(row_cells),
                "bbox": span,
                "pitch": round(gap_mean, 6),
                "pitch_regularity": round(regularity, 4),
                "width_cv": round(width_cv, 4),
                "sources": sorted({c.get("source") for c in row_cells}),
            }
        )

    groups.sort(key=lambda g: (g["page"], g["bbox"]["y"], g["bbox"]["x"]))
    return {
        "enabled": True,
        "feature_flag": "FORM_PARSER_COMB_DIAGNOSTICS_ENABLED",
        "behavior_change": False,
        "candidate_cell_count": len(cells),
        "likely_comb_group_count": len(groups),
        "total_cells_in_groups": sum(g["cell_count"] for g in groups),
        "groups": groups[:120],
    }
