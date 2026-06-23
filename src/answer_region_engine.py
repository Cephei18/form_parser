"""Phase I — Answer Region Intelligence v2.

A generalized abstraction for every kind of place a person can write an answer.
Today answer-region detection is fragmented across the anchor engine
(``_underline_candidates``, ``_synthetic_underline_candidates``,
``_rectangle_candidates``, ``_adjacent_whitespace_candidate``, inline leaders).
Each new form shape needs another bespoke builder. This module unifies them
behind one ``AnswerRegion`` type and a single ``build_answer_regions`` pass that
also adds the region kinds the old code never modelled: stacked **multiline**
groups, merged **broken** underlines, **signature** zones, **freeform**
whitespace blocks and **table input cells**.

The engine is *detection only* — it consumes features that have already been
detected (CV underlines / dotted / broken, empty rectangles, table cells) plus
text boxes, and emits page-scoped ``AnswerRegion`` objects. It does not render,
classify fields, or pick winners; the anchor engine adapts these regions into
candidates for the existing selection / global-assignment stage.

Gated behind ``FORM_PARSER_ANSWER_REGION_V2_ENABLED`` (default OFF). When OFF
the anchor engine never calls this module and behaviour is unchanged.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from statistics import median
from typing import Any

# --- Region types ----------------------------------------------------------- #
UNDERLINE = "UNDERLINE"
DOTTED = "DOTTED"
BROKEN = "BROKEN"
MULTILINE = "MULTILINE"
GRID = "GRID"
COMB = "COMB"
WHITESPACE = "WHITESPACE"
TABLE_CELL = "TABLE_CELL"
SIGNATURE = "SIGNATURE"
FREEFORM = "FREEFORM"

REGION_TYPES = frozenset(
    {UNDERLINE, DOTTED, BROKEN, MULTILINE, GRID, COMB, WHITESPACE, TABLE_CELL, SIGNATURE, FREEFORM}
)

SIGNATURE_KEYWORDS = ("signature", "signed", "sign here", "authorised signatory", "authorized signatory")
FREEFORM_KEYWORDS = (
    "comment",
    "remark",
    "note",
    "description",
    "details",
    "particular",
    "observation",
    "address",
    "reason",
)


def _bool_env(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def answer_region_v2_enabled() -> bool:
    """True when the Phase I answer-region engine is active."""
    return _bool_env("FORM_PARSER_ANSWER_REGION_V2_ENABLED", False)


# --- Core model ------------------------------------------------------------- #
@dataclass
class AnswerRegion:
    region_id: str
    page: int
    bbox: tuple[float, float, float, float]  # (x, y, width, height), normalised
    region_type: str
    confidence: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def bbox_dict(self) -> dict[str, float]:
        x, y, w, h = self.bbox
        return {"x": x, "y": y, "width": w, "height": h}

    def to_dict(self) -> dict[str, Any]:
        return {
            "region_id": self.region_id,
            "page": self.page,
            "bbox": self.bbox_dict(),
            "region_type": self.region_type,
            "confidence": round(float(self.confidence), 4),
            "metadata": self.metadata,
        }


@dataclass
class AnswerRegionResult:
    regions: list[AnswerRegion]
    counts_by_type: dict[str, int]
    merged_regions: list[dict[str, Any]]
    rejected_regions: list[dict[str, Any]]

    def to_debug_dict(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "region_count": len(self.regions),
            "regions": [r.to_dict() for r in self.regions],
            "counts_by_type": self.counts_by_type,
            "merged_regions": self.merged_regions,
            "rejected_regions": self.rejected_regions,
        }


# --- Geometry helpers ------------------------------------------------------- #
def _tuple(box: dict[str, Any] | None) -> tuple[float, float, float, float] | None:
    if not isinstance(box, dict):
        return None
    try:
        x = float(box.get("x", box.get("Left", 0.0)))
        y = float(box.get("y", box.get("Top", 0.0)))
        w = float(box.get("width", box.get("Width", 0.0)))
        h = float(box.get("height", box.get("Height", 0.0)))
    except (TypeError, ValueError):
        return None
    if w <= 0 or h <= 0:
        return None
    return (round(x, 6), round(y, 6), round(w, 6), round(h, 6))


def _right(b: tuple[float, float, float, float]) -> float:
    return b[0] + b[2]


def _bottom(b: tuple[float, float, float, float]) -> float:
    return b[1] + b[3]


def _center(b: tuple[float, float, float, float]) -> tuple[float, float]:
    return (b[0] + b[2] / 2.0, b[1] + b[3] / 2.0)


def _union(boxes: list[tuple[float, float, float, float]]) -> tuple[float, float, float, float]:
    x1 = min(b[0] for b in boxes)
    y1 = min(b[1] for b in boxes)
    x2 = max(_right(b) for b in boxes)
    y2 = max(_bottom(b) for b in boxes)
    return (round(x1, 6), round(y1, 6), round(x2 - x1, 6), round(y2 - y1, 6))


def _intersection(a: tuple, b: tuple) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(_right(a), _right(b))
    y2 = min(_bottom(a), _bottom(b))
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _overlap_ratio(a: tuple, b: tuple) -> float:
    area = max(a[2] * a[3], 1e-9)
    return _intersection(a, b) / area


def _region_id(region_type: str, page: int, bbox: tuple[float, float, float, float]) -> str:
    return f"{region_type}|p{page}|{bbox[0]:.4f},{bbox[1]:.4f},{bbox[2]:.4f},{bbox[3]:.4f}"


def _line_height(metrics_by_page: dict[int, dict[str, float]] | None, page: int) -> float:
    if metrics_by_page and page in metrics_by_page:
        return float(metrics_by_page[page].get("line_height") or 0.014)
    if metrics_by_page and 1 in metrics_by_page:
        return float(metrics_by_page[1].get("line_height") or 0.014)
    return 0.014


def _page_bounds(metrics_by_page: dict[int, dict[str, float]] | None, page: int) -> tuple[float, float]:
    if metrics_by_page and page in metrics_by_page:
        m = metrics_by_page[page]
        return float(m.get("page_left") or 0.05), float(m.get("page_right") or 0.95)
    return 0.05, 0.95


# --- Builders (adapters over existing features) ----------------------------- #
def _line_features(visual_features: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalise every horizontal-line feature into a common shape with an
    origin region_type, preserving page + confidence + source metadata."""
    lines: list[dict[str, Any]] = []
    for feat in visual_features.get("underlines", []) or []:
        bbox = _tuple(feat.get("bbox"))
        if bbox:
            lines.append({
                "bbox": bbox, "page": int(feat.get("page") or 1), "type": UNDERLINE,
                "confidence": float(feat.get("confidence") or 0.76),
                "metadata": {"source": "cv_underline", "length": feat.get("length")},
            })
    for feat in visual_features.get("synthetic_underlines", []) or []:
        bbox = _tuple(feat.get("bbox"))
        if not bbox:
            continue
        source_type = str(feat.get("source_type") or "dotted")
        rtype = BROKEN if source_type == "broken" else DOTTED
        lines.append({
            "bbox": bbox, "page": int(feat.get("page") or 1), "type": rtype,
            "confidence": float(feat.get("confidence") or 0.62),
            "metadata": {"source": f"synthetic_{source_type}", "segment_count": feat.get("segment_count")},
        })
    return lines


def build_underlines(visual_features: dict[str, Any]) -> list[AnswerRegion]:
    """Adapter: CV solid underlines -> UNDERLINE regions."""
    out: list[AnswerRegion] = []
    for feat in visual_features.get("underlines", []) or []:
        bbox = _tuple(feat.get("bbox"))
        if not bbox:
            continue
        page = int(feat.get("page") or 1)
        out.append(AnswerRegion(_region_id(UNDERLINE, page, bbox), page, bbox, UNDERLINE,
                                float(feat.get("confidence") or 0.76),
                                {"source": "cv_underline", "length": feat.get("length")}))
    return out


def build_dotted_regions(visual_features: dict[str, Any]) -> list[AnswerRegion]:
    """Adapter: synthetic dotted/dashed leaders -> DOTTED regions."""
    out: list[AnswerRegion] = []
    for feat in visual_features.get("synthetic_underlines", []) or []:
        if str(feat.get("source_type") or "dotted") == "broken":
            continue
        bbox = _tuple(feat.get("bbox"))
        if not bbox:
            continue
        page = int(feat.get("page") or 1)
        out.append(AnswerRegion(_region_id(DOTTED, page, bbox), page, bbox, DOTTED,
                                float(feat.get("confidence") or 0.62),
                                {"source": "synthetic_dotted", "source_type": feat.get("source_type")}))
    return out


def build_broken_regions(visual_features: dict[str, Any]) -> list[AnswerRegion]:
    """Synthetic broken leaders PLUS solid-underline fragments on the same y
    band with small gaps merged into one BROKEN region."""
    out: list[AnswerRegion] = []
    for feat in visual_features.get("synthetic_underlines", []) or []:
        if str(feat.get("source_type") or "dotted") != "broken":
            continue
        bbox = _tuple(feat.get("bbox"))
        if not bbox:
            continue
        page = int(feat.get("page") or 1)
        out.append(AnswerRegion(_region_id(BROKEN, page, bbox), page, bbox, BROKEN,
                                float(feat.get("confidence") or 0.6),
                                {"source": "synthetic_broken"}))
    return out


def _merge_broken_fragments(
    lines: list[dict[str, Any]],
    metrics_by_page: dict[int, dict[str, float]] | None,
) -> tuple[list[AnswerRegion], set[int]]:
    """Merge UNDERLINE fragments that share a y band, have small horizontal gaps
    and similar thickness into one BROKEN region. Returns (regions, consumed)."""
    regions: list[AnswerRegion] = []
    consumed: set[int] = set()
    by_page: dict[int, list[int]] = {}
    for idx, ln in enumerate(lines):
        if ln["type"] != UNDERLINE:
            continue
        by_page.setdefault(ln["page"], []).append(idx)

    for page, idxs in by_page.items():
        lh = _line_height(metrics_by_page, page)
        y_tol = max(lh * 0.8, 0.008)
        gap_max = max(lh * 1.5, 0.03)
        ordered = sorted(idxs, key=lambda i: (round(_center(lines[i]["bbox"])[1] / max(y_tol, 1e-6)), lines[i]["bbox"][0]))
        run: list[int] = []

        def flush(r: list[int]) -> None:
            if len(r) >= 2:
                boxes = [lines[i]["bbox"] for i in r]
                merged = _union(boxes)
                regions.append(AnswerRegion(
                    _region_id(BROKEN, page, merged), page, merged, BROKEN,
                    0.66, {"source": "merged_fragments", "segment_count": len(r)}))
                consumed.update(r)

        for i in ordered:
            if not run:
                run = [i]
                continue
            prev = lines[run[-1]]["bbox"]
            cur = lines[i]["bbox"]
            same_band = abs(_center(cur)[1] - _center(prev)[1]) <= y_tol
            gap = cur[0] - _right(prev)
            similar_h = abs(cur[3] - prev[3]) <= max(prev[3], cur[3], 1e-6) * 0.8
            if same_band and -y_tol <= gap <= gap_max and similar_h:
                run.append(i)
            else:
                flush(run)
                run = [i]
        flush(run)
    return regions, consumed


def build_multiline_regions(
    lines: list[dict[str, Any]],
    metrics_by_page: dict[int, dict[str, float]] | None,
) -> tuple[list[AnswerRegion], set[int]]:
    """Cluster vertically stacked lines of equal width / consistent spacing into
    one MULTILINE region (so three stacked rules become one field, not three).
    Returns (regions, consumed line indices)."""
    regions: list[AnswerRegion] = []
    consumed: set[int] = set()

    by_page: dict[int, list[int]] = {}
    for idx, ln in enumerate(lines):
        by_page.setdefault(ln["page"], []).append(idx)

    for page, idxs in by_page.items():
        lh = _line_height(metrics_by_page, page)
        x_tol = max(lh * 1.5, 0.03)
        width_tol = 0.25  # relative
        gap_max = max(lh * 4.0, 0.06)

        # Cluster by similar x-start and width.
        clusters: list[list[int]] = []
        for i in sorted(idxs, key=lambda j: lines[j]["bbox"][1]):
            placed = False
            for cluster in clusters:
                ref = lines[cluster[0]]["bbox"]
                cur = lines[i]["bbox"]
                if abs(cur[0] - ref[0]) <= x_tol and abs(cur[2] - ref[2]) <= max(ref[2], cur[2]) * width_tol:
                    cluster.append(i)
                    placed = True
                    break
            if not placed:
                clusters.append([i])

        for cluster in clusters:
            if len(cluster) < 2:
                continue
            ordered = sorted(cluster, key=lambda j: lines[j]["bbox"][1])
            # Split into runs with consistent vertical spacing.
            run = [ordered[0]]
            for j in ordered[1:]:
                prev = lines[run[-1]]["bbox"]
                cur = lines[j]["bbox"]
                gap = cur[1] - prev[1]
                if 0 < gap <= gap_max:
                    run.append(j)
                else:
                    _emit_multiline(run, lines, page, regions, consumed)
                    run = [j]
            _emit_multiline(run, lines, page, regions, consumed)
    return regions, consumed


def _emit_multiline(run, lines, page, regions, consumed) -> None:
    if len(run) < 2:
        return
    boxes = [lines[i]["bbox"] for i in run]
    merged = _union(boxes)
    gaps = [lines[run[k + 1]]["bbox"][1] - lines[run[k]]["bbox"][1] for k in range(len(run) - 1)]
    regularity = 1.0
    if len(gaps) >= 2:
        med = median(gaps) or 1e-6
        spread = (max(gaps) - min(gaps)) / med
        regularity = max(0.0, 1.0 - min(spread, 1.0))
    confidence = round(min(0.85, 0.6 + 0.05 * len(run) + 0.1 * regularity), 4)
    regions.append(AnswerRegion(
        _region_id(MULTILINE, page, merged), page, merged, MULTILINE, confidence,
        {"line_count": len(run), "combined_bbox": list(merged), "spacing_regularity": round(regularity, 4),
         "member_types": sorted({lines[i]["type"] for i in run})}))
    consumed.update(run)


def build_comb_regions(visual_features: dict[str, Any], metrics_by_page=None) -> list[AnswerRegion]:
    """Detect a single horizontal run of >=5 equal small cells (character comb).
    Detection only — comb widget fitting still happens in comb_detector."""
    out: list[AnswerRegion] = []
    by_page: dict[int, list[tuple]] = {}
    for feat in visual_features.get("empty_boxes", []) or []:
        bbox = _tuple(feat.get("bbox"))
        if not bbox or int(feat.get("text_count") or 0) > 0:
            continue
        by_page.setdefault(int(feat.get("page") or 1), []).append(bbox)

    for page, boxes in by_page.items():
        lh = _line_height(metrics_by_page, page)
        y_tol = max(lh * 0.8, 0.01)
        rows: list[list[tuple]] = []
        for b in sorted(boxes, key=lambda bb: (_center(bb)[1], bb[0])):
            for row in rows:
                if abs(_center(b)[1] - _center(row[0])[1]) <= y_tol:
                    row.append(b)
                    break
            else:
                rows.append([b])
        for row in rows:
            if len(row) < 5:
                continue
            widths = [b[2] for b in row]
            if (max(widths) - min(widths)) / max(median(widths), 1e-6) > 0.5:
                continue
            merged = _union(row)
            out.append(AnswerRegion(_region_id(COMB, page, merged), page, merged, COMB,
                                    0.7, {"cell_count": len(row), "source": "empty_box_run"}))
    return out


def build_grid_regions(visual_features: dict[str, Any], metrics_by_page=None) -> list[AnswerRegion]:
    """Detect a 2-D grid (>=2 rows x >=2 aligned columns) of empty cells.
    Conservative detection only."""
    out: list[AnswerRegion] = []
    by_page: dict[int, list[tuple]] = {}
    for feat in visual_features.get("empty_boxes", []) or []:
        bbox = _tuple(feat.get("bbox"))
        if not bbox or int(feat.get("text_count") or 0) > 0:
            continue
        by_page.setdefault(int(feat.get("page") or 1), []).append(bbox)

    for page, boxes in by_page.items():
        if len(boxes) < 4:
            continue
        lh = _line_height(metrics_by_page, page)
        y_tol = max(lh * 0.8, 0.01)
        rows: list[list[tuple]] = []
        for b in sorted(boxes, key=lambda bb: (_center(bb)[1], bb[0])):
            for row in rows:
                if abs(_center(b)[1] - _center(row[0])[1]) <= y_tol:
                    row.append(b)
                    break
            else:
                rows.append([b])
        multi = [r for r in rows if len(r) >= 2]
        if len(multi) >= 2 and len({len(r) for r in multi}) == 1:
            merged = _union([b for r in multi for b in r])
            out.append(AnswerRegion(_region_id(GRID, page, merged), page, merged, GRID,
                                    0.68, {"rows": len(multi), "cols": len(multi[0]), "source": "empty_box_grid"}))
    return out


def build_table_cell_regions(table_cells: list[dict[str, Any]] | None) -> list[AnswerRegion]:
    """Empty table cells intended for input -> TABLE_CELL regions (detection
    only; full table semantics are a later phase)."""
    out: list[AnswerRegion] = []
    for cell in table_cells or []:
        text = str(cell.get("text") or "").strip()
        if text:
            continue
        bbox = _tuple(cell.get("bbox"))
        if not bbox:
            continue
        page = int(cell.get("page") or 1)
        out.append(AnswerRegion(_region_id(TABLE_CELL, page, bbox), page, bbox, TABLE_CELL,
                                0.7, {"table_id": cell.get("table_id"), "row_index": cell.get("row_index"),
                                      "column_index": cell.get("column_index"), "source": "empty_table_cell"}))
    return out


def build_signature_regions(
    text_boxes: list[dict[str, Any]],
    lines: list[dict[str, Any]],
    metrics_by_page: dict[int, dict[str, float]] | None,
) -> tuple[list[AnswerRegion], set[int]]:
    """A signature keyword + the nearest line / blank to its right or below.
    Returns (regions, consumed line indices)."""
    regions: list[AnswerRegion] = []
    consumed: set[int] = set()
    for tb in text_boxes or []:
        text = str(tb.get("text") or "").lower()
        if not any(kw in text for kw in SIGNATURE_KEYWORDS):
            continue
        lbox = _tuple(tb.get("bbox"))
        if not lbox:
            continue
        page = int(tb.get("page") or 1)
        lh = _line_height(metrics_by_page, page)
        lcx, lcy = _center(lbox)
        best_idx = None
        best_dist = None
        for idx, ln in enumerate(lines):
            if ln["page"] != page or idx in consumed:
                continue
            lncx, lncy = _center(ln["bbox"])
            right_same_row = ln["bbox"][0] >= _right(lbox) - lh and abs(lncy - lcy) <= max(lh * 2.0, 0.03)
            below = 0 <= ln["bbox"][1] - _bottom(lbox) <= lh * 4.0
            if not (right_same_row or below):
                continue
            dist = abs(lncy - lcy) + abs(ln["bbox"][0] - _right(lbox))
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_idx = idx
        if best_idx is not None:
            ln = lines[best_idx]
            bbox = ln["bbox"]
            consumed.add(best_idx)
        else:
            # Fallback: estimate a signing strip below the keyword.
            left, right = _page_bounds(metrics_by_page, page)
            bbox = (round(max(left, lbox[0]), 6), round(min(0.97, _bottom(lbox) + lh * 0.5), 6),
                    round(max(lbox[2] * 1.5, 0.2), 6), round(max(lh * 2.0, 0.03), 6))
            bbox = _tuple({"x": bbox[0], "y": bbox[1], "width": bbox[2], "height": bbox[3]}) or bbox
        regions.append(AnswerRegion(_region_id(SIGNATURE, page, bbox), page, bbox, SIGNATURE,
                                    0.78, {"keyword": text[:40], "source": "signature_keyword"}))
    return regions, consumed


def build_whitespace_regions(
    text_boxes: list[dict[str, Any]],
    metrics_by_page: dict[int, dict[str, float]] | None,
) -> list[AnswerRegion]:
    """Large blank area beneath a 'Comments:'-style label -> FREEFORM region.
    A blank strip to the right of a ':'-terminated label -> WHITESPACE region."""
    out: list[AnswerRegion] = []
    if not text_boxes:
        return out
    by_page: dict[int, list[dict[str, Any]]] = {}
    for tb in text_boxes:
        bbox = _tuple(tb.get("bbox"))
        if bbox:
            by_page.setdefault(int(tb.get("page") or 1), []).append({"bbox": bbox, "text": str(tb.get("text") or "")})

    for page, boxes in by_page.items():
        lh = _line_height(metrics_by_page, page)
        left, right = _page_bounds(metrics_by_page, page)
        for item in boxes:
            lbox = item["bbox"]
            text = item["text"].strip().lower()
            is_freeform = any(kw in text for kw in FREEFORM_KEYWORDS)
            is_label = text.endswith(":") or is_freeform
            if not is_label:
                continue
            lcx, lcy = _center(lbox)

            # FREEFORM: tall blank band below the label.
            band_top = _bottom(lbox) + lh * 0.5
            band_bottom = band_top + max(lh * 3.0, 0.05)
            blocked_below = any(
                other["bbox"] is not lbox
                and band_top <= _center(other["bbox"])[1] <= band_bottom
                and other["bbox"][0] < right
                and _right(other["bbox"]) > left
                for other in boxes
            )
            if is_freeform and not blocked_below and band_bottom <= 0.99:
                bbox = (round(max(left, lbox[0]), 6), round(band_top, 6),
                        round(right - max(left, lbox[0]), 6), round(band_bottom - band_top, 6))
                t = _tuple({"x": bbox[0], "y": bbox[1], "width": bbox[2], "height": bbox[3]})
                if t:
                    out.append(AnswerRegion(_region_id(FREEFORM, page, t), page, t, FREEFORM,
                                            0.55, {"source": "freeform_label", "label": text[:40]}))
                continue

            # WHITESPACE: blank slot to the right on the same row.
            row_lo = lcy - max(lh * 0.7, 0.012)
            row_hi = lcy + max(lh * 0.7, 0.012)
            obstacles = [
                other["bbox"][0]
                for other in boxes
                if other["bbox"] is not lbox
                and row_lo <= _center(other["bbox"])[1] <= row_hi
                and other["bbox"][0] > _right(lbox)
            ]
            start_x = min(right, _right(lbox) + max(lh * 0.75, 0.008))
            end_x = min(obstacles) - max(lh * 0.4, 0.006) if obstacles else right
            if end_x - start_x >= max(lh * 4.0, 0.08):
                t = _tuple({"x": start_x, "y": row_lo, "width": end_x - start_x, "height": row_hi - row_lo})
                if t:
                    out.append(AnswerRegion(_region_id(WHITESPACE, page, t), page, t, WHITESPACE,
                                            0.48, {"source": "right_of_label", "label": text[:40]}))
    return out


# --- Orchestrator ----------------------------------------------------------- #
def build_answer_regions(
    visual_features: dict[str, Any],
    text_boxes: list[dict[str, Any]] | None = None,
    table_cells: list[dict[str, Any]] | None = None,
    metrics_by_page: dict[int, dict[str, float]] | None = None,
) -> AnswerRegionResult:
    """Detect every answer region on every page and return them as one set.

    Coordination: stacked lines are clustered into MULTILINE regions first, then
    remaining fragments are merged into BROKEN regions; lines consumed by either
    step are recorded in ``merged_regions`` and not re-emitted as single lines.
    """
    text_boxes = text_boxes or []
    visual_features = visual_features or {}

    lines = _line_features(visual_features)

    merged_records: list[dict[str, Any]] = []
    regions: list[AnswerRegion] = []

    # 1. Multiline stacks (consume member lines).
    multiline, consumed_ml = build_multiline_regions(lines, metrics_by_page)
    regions.extend(multiline)

    # 2. Broken-fragment merges over the remaining solid underlines.
    remaining_lines = [ln if idx not in consumed_ml else None for idx, ln in enumerate(lines)]
    broken_merged, consumed_bm = _merge_broken_fragments(
        [ln for ln in remaining_lines if ln is not None], metrics_by_page
    )
    # Map consumed_bm (indices into the filtered list) back to original indices.
    filtered_index_map = [idx for idx, ln in enumerate(remaining_lines) if ln is not None]
    consumed_bm_original = {filtered_index_map[i] for i in consumed_bm}
    regions.extend(broken_merged)

    consumed = consumed_ml | consumed_bm_original
    for idx in sorted(consumed):
        merged_records.append({
            "line_index": idx,
            "type": lines[idx]["type"],
            "bbox": {"x": lines[idx]["bbox"][0], "y": lines[idx]["bbox"][1],
                     "width": lines[idx]["bbox"][2], "height": lines[idx]["bbox"][3]},
            "page": lines[idx]["page"],
        })

    # 3. Remaining single lines -> UNDERLINE / DOTTED / BROKEN.
    for idx, ln in enumerate(lines):
        if idx in consumed:
            continue
        regions.append(AnswerRegion(_region_id(ln["type"], ln["page"], ln["bbox"]),
                                    ln["page"], ln["bbox"], ln["type"],
                                    float(ln["confidence"]), dict(ln["metadata"])))

    # 4. Signatures (may consume a remaining line) — recompute over survivors.
    survivor_lines = [ln for idx, ln in enumerate(lines) if idx not in consumed]
    signatures, consumed_sig = build_signature_regions(text_boxes, survivor_lines, metrics_by_page)
    if consumed_sig:
        sig_boxes = {survivor_lines[i]["bbox"] for i in consumed_sig}
        regions = [r for r in regions if not (r.region_type in {UNDERLINE, DOTTED, BROKEN} and r.bbox in sig_boxes)]
    regions.extend(signatures)

    # 5. Combs, grids, table cells, freeform whitespace.
    regions.extend(build_comb_regions(visual_features, metrics_by_page))
    regions.extend(build_grid_regions(visual_features, metrics_by_page))
    regions.extend(build_table_cell_regions(table_cells))
    regions.extend(build_whitespace_regions(text_boxes, metrics_by_page))

    # 6. Deduplicate by region_id (deterministic order: page, y, x, type).
    seen: dict[str, AnswerRegion] = {}
    rejected: list[dict[str, Any]] = []
    for r in regions:
        if r.region_id in seen:
            rejected.append({"region_id": r.region_id, "reason": "duplicate_region_id"})
            continue
        seen[r.region_id] = r
    ordered = sorted(seen.values(), key=lambda r: (r.page, r.bbox[1], r.bbox[0], r.region_type))

    counts: dict[str, int] = {}
    for r in ordered:
        counts[r.region_type] = counts.get(r.region_type, 0) + 1

    return AnswerRegionResult(
        regions=ordered,
        counts_by_type=counts,
        merged_regions=merged_records,
        rejected_regions=rejected,
    )
