"""Emit fillable widgets for the empty data cells of an input-grid table.

Textract reports an education / experience grid as a TABLE whose first column(s)
hold row labels (``High School``, ``College or University`` …) and whose other
columns are the blank cells the applicant writes in. The anchoring engine only
ever creates widgets from ``KEY_VALUE_SET`` keys, checkboxes and photos, so an
empty table cell that Textract did *not* also expose as a KEY gets no widget —
typically only the one row that happens to be a KEY becomes fillable and the
rest of the grid is silently dropped.

This module closes that gap: for each table that looks like an *input grid*
(a label column plus mostly-empty data columns) it emits one ``table_cell``
widget per empty data cell, labelled from its row label and (when discoverable)
its column header. It is detection-driven and conservative — layout tables and
fully-populated data tables produce nothing — and is ON by default with a
``FORM_PARSER_TABLE_FILL_ENABLED=false`` kill-switch.
"""
from __future__ import annotations

import os
import re
from typing import Any, Callable

from src.section_detector import SectionIndex, qualify_label, section_summary

# A cell whose text carries a selection glyph is a checkbox/option cell, not a
# write-in label — used to keep the signature-table emitter off checkbox matrices.
_SELECTION_TOKEN_RE = re.compile(r"\[\s*[xX✓]?\s*\]|[☐□■●○✓]")

# A column counts as a label/header column when at least this fraction of its
# populated cells carry text; as a data (input) column when at most the
# complementary fraction do.
LABEL_COLUMN_FILL = 0.6
DATA_COLUMN_FILL = 0.4

# A cell is "already handled" when an existing widget covers this much of it.
COVERED_OVERLAP = 0.5


def _bool_env(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def table_fill_enabled() -> bool:
    """True when empty input-grid cells are emitted as widgets (default ON)."""
    return _bool_env("FORM_PARSER_TABLE_FILL_ENABLED", True)


# --- geometry (standalone; never imports the engine) -------------------------
def _norm_box(raw: dict[str, Any] | None) -> dict[str, float] | None:
    if not isinstance(raw, dict):
        return None
    try:
        x = float(raw.get("x", raw.get("Left", 0.0)))
        y = float(raw.get("y", raw.get("Top", 0.0)))
        w = float(raw.get("width", raw.get("Width", 0.0)))
        h = float(raw.get("height", raw.get("Height", 0.0)))
    except (TypeError, ValueError):
        return None
    if w <= 0 or h <= 0:
        return None
    return {"x": round(x, 6), "y": round(y, 6), "width": round(w, 6), "height": round(h, 6)}


def _right(b: dict[str, float]) -> float:
    return float(b["x"]) + float(b["width"])


def _bottom(b: dict[str, float]) -> float:
    return float(b["y"]) + float(b["height"])


def _area(b: dict[str, float]) -> float:
    return max(0.0, float(b["width"])) * max(0.0, float(b["height"]))


def _intersection(a: dict[str, float], b: dict[str, float]) -> float:
    x1, y1 = max(float(a["x"]), float(b["x"])), max(float(a["y"]), float(b["y"]))
    x2, y2 = min(_right(a), _right(b)), min(_bottom(a), _bottom(b))
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _covered_by_existing(box: dict[str, float], page: int, existing: list[tuple[int, dict[str, float]]]) -> bool:
    for ep, eb in existing:
        if ep == page and _intersection(box, eb) / max(_area(box), 1e-9) >= COVERED_OVERLAP:
            return True
    return False


def _confidence_class(score: float) -> str:
    if score >= 0.82:
        return "high"
    if score >= 0.62:
        return "medium"
    if score >= 0.42:
        return "low"
    return "unresolved"


def _cxy(b: dict[str, float]) -> tuple[float, float]:
    return float(b["x"]) + float(b["width"]) / 2.0, float(b["y"]) + float(b["height"]) / 2.0


def emit_table_input_cells(
    tables: list[dict[str, Any]],
    *,
    section_index: SectionIndex,
    page_px: Callable[[int], tuple[int, int]],
    existing_field_boxes: list[tuple[int, dict[str, float]]],
    start_index: int = 1,
) -> list[dict[str, Any]]:
    """Return ``table_cell`` widget mappings for every empty data cell of every
    input-grid table. Cells already covered by an existing widget are skipped."""
    if not table_fill_enabled():
        return []

    mappings: list[dict[str, Any]] = []
    next_index = start_index
    for t_idx, table in enumerate(tables or [], start=1):
        page = int(table.get("page") or 1)
        dims = table.get("dimensions") or {}
        nrows, ncols = int(dims.get("rows") or 0), int(dims.get("columns") or 0)
        if nrows < 2 or ncols < 2:
            continue
        cells = table.get("cells") or []

        grid: dict[tuple[int, int], dict[str, Any]] = {}
        for cell in cells:
            r, c = int(cell.get("row_index") or 0), int(cell.get("column_index") or 0)
            if r <= 0 or c <= 0:
                continue
            geom = (cell.get("geometry") or {}).get("bounding_box")
            grid[(r, c)] = {
                "text": str(cell.get("text") or "").strip(),
                "bbox": _norm_box(geom),
                "cell_id": cell.get("cell_id"),
                "confidence": cell.get("confidence"),
            }

        # Row fill ratios. A header row is a mostly-populated row that sits above
        # at least one mostly-empty body row (it labels the columns rather than
        # being a data row itself, e.g. "NAME | DATES | DEGREE").
        def _row_fill(r: int) -> float:
            present = [grid[(r, c)] for c in range(1, ncols + 1) if (r, c) in grid]
            return (sum(1 for g in present if g["text"]) / len(present)) if present else 0.0

        row_fill = {r: _row_fill(r) for r in range(1, nrows + 1)}
        header_rows = {
            r
            for r in range(1, nrows + 1)
            if row_fill[r] >= LABEL_COLUMN_FILL and any(row_fill[o] <= DATA_COLUMN_FILL for o in row_fill if o != r)
        }
        body_rows = [r for r in range(1, nrows + 1) if r not in header_rows]

        # A label column is mostly populated across the *body* rows (the row-label
        # column, e.g. "COLLEGE / HIGH SCHOOL"). Data columns are everything else.
        label_cols: list[int] = []
        for col in range(1, ncols + 1):
            present = [grid[(r, col)] for r in body_rows if (r, col) in grid]
            if present and sum(1 for g in present if g["text"]) / len(present) >= LABEL_COLUMN_FILL:
                label_cols.append(col)
        data_cols = [c for c in range(1, ncols + 1) if c not in label_cols]

        # Qualify as an input grid only with real structure: a header row that
        # labels columns, or a label column that names rows. This skips layout
        # tables and fully-populated data tables (which also have no empty cells).
        if not header_rows and not label_cols:
            continue
        if not data_cols:
            continue

        # Column headers come only from a clean *internal* header row. External
        # header text printed above the table is deliberately not used: it rarely
        # aligns to Textract's column boundaries and yields garbled labels.
        col_headers: dict[int, str] = {}
        if header_rows:
            head_r = min(header_rows)
            for col in data_cols:
                txt = (grid.get((head_r, col)) or {}).get("text") or ""
                if txt:
                    col_headers[col] = txt

        for r in body_rows:
            row_label = next(
                (grid[(r, c)]["text"] for c in sorted(label_cols) if grid.get((r, c), {}).get("text")),
                "",
            )
            # When the table has a row-label column, only emit for rows that
            # actually carry a label (real data rows) — a stray blank row is not
            # an input row. Header-only tables (no label column) emit every body
            # row, since their blank rows ARE the input rows.
            if label_cols and not row_label:
                continue
            for col in sorted(data_cols):
                cell = grid.get((r, col))
                if not cell or cell["text"] or not cell["bbox"]:
                    continue
                box = cell["bbox"]
                if _covered_by_existing(box, page, existing_field_boxes):
                    continue
                header = col_headers.get(col, "")
                if row_label and header:
                    label = f"{row_label} - {header}"
                elif row_label:
                    label = f"{row_label} (column {col})"
                elif header:
                    label = f"{header} (row {r})"
                else:
                    label = f"Cell (row {r}, column {col})"
                owner = section_index.owner(page, float(box["y"]))
                cell_conf = cell.get("confidence")
                try:
                    base = float(cell_conf) / 100.0 if cell_conf is not None else 0.8
                except (TypeError, ValueError):
                    base = 0.8
                confidence = round(min(0.95, 0.8 + base * 0.1), 4)
                px_w, px_h = page_px(page)
                mappings.append(
                    {
                        "field_id": f"table_cell_{t_idx}_{r}_{col}",
                        "label": label,
                        "qualified_label": qualify_label(owner, label),
                        "section": section_summary(owner),
                        "value": "",
                        "field_type": "text",
                        "bbox": dict(box),
                        "label_bbox": None,
                        "answer_region": {"bbox": dict(box), "type": "table_cell", "confidence": confidence},
                        "page": page,
                        "confidence": confidence,
                        "candidate_score": confidence,
                        "confidence_class": _confidence_class(confidence),
                        "multiline_group_size": 1,
                        "field_bboxes": [
                            {
                                "x": float(box["x"]) * px_w,
                                "y": float(box["y"]) * px_h,
                                "width": float(box["width"]) * px_w,
                                "height": float(box["height"]) * px_h,
                            }
                        ],
                        "source": "textract_table_fill",
                        "render_border": False,
                        "anchoring": {
                            "anchor_type": "table_cell",
                            "type_reasons": ["input_grid_empty_cell"],
                            "selection_reasons": ["table_fill_input_cell"],
                            "label_overlap_ratio": 0.0,
                            "candidate_count": 1,
                            "top_candidates": [],
                            "key_block_id": None,
                            "value_block_ids": [],
                            "table_id": table.get("table_block_id"),
                            "row_index": r,
                            "column_index": col,
                        },
                    }
                )
                next_index += 1

    return mappings


# Minimum trailing-space width (page fraction) for a cell to be considered as
# carrying an inline write-in answer after its label.
SIGNATURE_MIN_WIDTH = 0.04


def emit_signature_table_cells(
    tables: list[dict[str, Any]],
    *,
    text_boxes: list[dict[str, Any]],
    visual_features: dict[str, Any],
    section_index: SectionIndex,
    page_px: Callable[[int], tuple[int, int]],
) -> tuple[list[dict[str, Any]], list[tuple[int, dict[str, float]]]]:
    """Emit inline answer regions for a signature-style table and report the
    table regions whose Textract KEY fields should be suppressed.

    A signature table (e.g. ``Sign. Guardian ...... | Sign. Patient ......`` over
    ``Name ...... | Relation with Patient ......``) has *every* cell populated
    with a label followed by a dotted fill line — so it is neither an input grid
    (it has no empty cells) nor reliably handled by Textract's key/value linking
    (the value blocks land in the wrong column). For each such table we read the
    label text in each cell and emit the answer region as the gap after the
    label, extending to the next label on the row (the dotted leaders cross cell
    borders) or the table's right edge. The returned table boxes let the caller
    drop the unreliable KEY fields that fall inside them.

    Returns ``(mappings, suppression_boxes)``; both empty when the feature is off
    or no table qualifies.
    """
    if not table_fill_enabled():
        return [], []

    leaders = list(visual_features.get("synthetic_underlines") or [])
    mappings: list[dict[str, Any]] = []
    suppression: list[tuple[int, dict[str, float]]] = []

    for t_idx, table in enumerate(tables or [], start=1):
        page = int(table.get("page") or 1)
        cells = []
        for cell in table.get("cells", []) or []:
            box = _norm_box((cell.get("geometry") or {}).get("bounding_box"))
            if box:
                cells.append({"text": str(cell.get("text") or "").strip(), "bbox": box})
        if len(cells) < 2:
            continue
        # All-populated only: an input grid (with empty data cells) is handled
        # elsewhere; a signature table has a label in every cell.
        if any(not c["text"] for c in cells):
            continue
        # Skip checkbox / option matrices: their cells carry selection glyphs
        # ("[ ] PARK RANGER"), not write-in labels.
        if any(_SELECTION_TOKEN_RE.search(c["text"]) for c in cells):
            continue
        # Skip explicit "Sign Here" grids — handled by emit_signature_grid_cells.
        if any(_is_sign_here_cell(re.sub(r"[^a-z ]+", " ", c["text"].lower()).strip()) for c in cells):
            continue

        table_box = {
            "x": min(c["bbox"]["x"] for c in cells),
            "y": min(c["bbox"]["y"] for c in cells),
            "width": max(_right(c["bbox"]) for c in cells) - min(c["bbox"]["x"] for c in cells),
            "height": max(_bottom(c["bbox"]) for c in cells) - min(c["bbox"]["y"] for c in cells),
        }
        # Require a dotted/broken fill leader running through the table — the
        # signal that these cells are write-in lines, not a data table.
        row_leaders = [
            lf
            for lf in leaders
            if int(lf.get("page") or 1) == page and _intersection(lf["bbox"], table_box) > 0
        ]
        if not row_leaders:
            continue

        table_right = _right(table_box)
        page_words = [
            it
            for it in text_boxes
            if int(it.get("page") or 1) == page and isinstance(it.get("bbox"), dict)
        ]

        for cell in cells:
            cbox = cell["bbox"]
            ccx, ccy = _cxy(cbox)
            row_tol = max(cbox["height"], 0.012)
            # Words physically inside this cell give the label's right extent.
            inside = [
                w for w in page_words
                if cbox["x"] - 0.005 <= _cxy(w["bbox"])[0] <= _right(cbox) + 0.005
                and abs(_cxy(w["bbox"])[1] - ccy) <= row_tol
            ]
            if not inside:
                continue
            label_right = max(_right(w["bbox"]) for w in inside)
            gap = 0.008
            start_x = label_right + gap
            # Extend to the next label/text on the row (leaders cross cells), or
            # to the table's right edge.
            next_xs = [
                float(w["bbox"]["x"])
                for w in page_words
                if abs(_cxy(w["bbox"])[1] - ccy) <= row_tol and float(w["bbox"]["x"]) > start_x
            ]
            end_x = min([table_right] + next_xs) - gap
            width = end_x - start_x
            if width < SIGNATURE_MIN_WIDTH:
                continue
            # Align vertically to the nearest leader in this row, else the cell.
            row_lf = [lf for lf in row_leaders if abs(_cxy(lf["bbox"])[1] - ccy) <= row_tol]
            ly = _cxy(row_lf[0]["bbox"])[1] if row_lf else ccy
            height = max(cbox["height"] * 0.5, 0.014)
            y = max(0.0, ly - height * 0.7)
            box = _norm_box({"x": start_x, "y": y, "width": width, "height": height})
            if not box:
                continue
            label = cell["text"]
            wtype = "signature" if "sign" in label.lower() else "text"
            owner = section_index.owner(page, float(box["y"]))
            px_w, px_h = page_px(page)
            mappings.append(
                {
                    "field_id": f"sigtable_{t_idx}_{int(round(ccy * 1000))}_{int(round(ccx * 1000))}",
                    "label": label,
                    "qualified_label": qualify_label(owner, label),
                    "section": section_summary(owner),
                    "value": "",
                    "field_type": wtype,
                    "bbox": dict(box),
                    "label_bbox": None,
                    "answer_region": {"bbox": dict(box), "type": "table_inline", "confidence": 0.88},
                    "page": page,
                    "confidence": 0.88,
                    "candidate_score": 0.88,
                    "confidence_class": _confidence_class(0.88),
                    "multiline_group_size": 1,
                    "field_bboxes": [
                        {
                            "x": float(box["x"]) * px_w,
                            "y": float(box["y"]) * px_h,
                            "width": float(box["width"]) * px_w,
                            "height": float(box["height"]) * px_h,
                        }
                    ],
                    "source": "textract_table_fill",
                    "render_border": False,
                    "anchoring": {
                        "anchor_type": "table_cell",
                        "type_reasons": ["signature_table_inline_answer"],
                        "selection_reasons": ["table_fill_signature_cell"],
                        "label_overlap_ratio": 0.0,
                        "candidate_count": 1,
                        "top_candidates": [],
                        "key_block_id": None,
                        "value_block_ids": [],
                        "table_id": table.get("table_block_id"),
                    },
                }
            )
        suppression.append((page, table_box))

    return mappings, suppression


# Cell text (normalised) that marks an explicit signature box ("Sign Here").
# Matched tightly so a label like "Sign. Guardian" (a dotted-leader row handled
# by emit_signature_table_cells) is NOT treated as a signing box.
def _is_sign_here_cell(norm_text: str) -> bool:
    return "sign here" in norm_text or norm_text in {"sign", "signature"}


def emit_signature_grid_cells(
    tables: list[dict[str, Any]],
    *,
    section_index: SectionIndex,
    page_px: Callable[[int], tuple[int, int]],
) -> tuple[list[dict[str, Any]], list[tuple[int, dict[str, float]]]]:
    """Emit signature widgets for an explicit "Sign Here" grid + its suppression
    region.

    Some forms lay out signatures as a small table where one row of cells says
    "Sign Here" (the box to sign in) and an adjacent row carries the applicant
    label ("First / Sole Applicant", "Second Applicant", ...). Textract turns the
    labels into KEY fields and collapses the "Sign Here" values to points, so the
    real wide signing boxes are lost. Here we emit one signature widget per
    "Sign Here" cell (the full cell is the signing area) labelled from the
    vertically-adjacent label cell, and return the table region so the caller can
    drop the unreliable KEY fields inside it.
    """
    if not table_fill_enabled():
        return [], []

    mappings: list[dict[str, Any]] = []
    suppression: list[tuple[int, dict[str, float]]] = []

    for table in tables or []:
        page = int(table.get("page") or 1)
        cells = []
        for cell in table.get("cells", []) or []:
            box = _norm_box((cell.get("geometry") or {}).get("bounding_box"))
            if box:
                cells.append(
                    {
                        "text": str(cell.get("text") or "").strip(),
                        "norm": re.sub(r"[^a-z ]+", " ", str(cell.get("text") or "").lower()).strip(),
                        "row": int(cell.get("row_index") or 0),
                        "col": int(cell.get("column_index") or 0),
                        "bbox": box,
                    }
                )
        sign_cells = [c for c in cells if _is_sign_here_cell(c["norm"])]
        if not sign_cells:
            continue

        table_box = {
            "x": min(c["bbox"]["x"] for c in cells),
            "y": min(c["bbox"]["y"] for c in cells),
            "width": max(_right(c["bbox"]) for c in cells) - min(c["bbox"]["x"] for c in cells),
            "height": max(_bottom(c["bbox"]) for c in cells) - min(c["bbox"]["y"] for c in cells),
        }

        for sc in sign_cells:
            # Label cell: same column, a different row, with non-sign text.
            label_cell = next(
                (
                    c
                    for c in cells
                    if c["col"] == sc["col"] and c["row"] != sc["row"] and c["text"] and not _is_sign_here_cell(c["norm"])
                ),
                None,
            )
            label = label_cell["text"] if label_cell else "Signature"
            box = dict(sc["bbox"])
            owner = section_index.owner(page, float(box["y"]))
            px_w, px_h = page_px(page)
            mappings.append(
                {
                    "field_id": f"signgrid_{page}_{sc['row']}_{sc['col']}",
                    "label": label,
                    "qualified_label": qualify_label(owner, label),
                    "section": section_summary(owner),
                    "value": "",
                    "field_type": "signature",
                    "bbox": dict(box),
                    "label_bbox": None,
                    "answer_region": {"bbox": dict(box), "type": "signature_grid_cell", "confidence": 0.9},
                    "page": page,
                    "confidence": 0.9,
                    "candidate_score": 0.9,
                    "confidence_class": _confidence_class(0.9),
                    "multiline_group_size": 1,
                    "field_bboxes": [
                        {
                            "x": float(box["x"]) * px_w,
                            "y": float(box["y"]) * px_h,
                            "width": float(box["width"]) * px_w,
                            "height": float(box["height"]) * px_h,
                        }
                    ],
                    "source": "textract_table_fill",
                    "render_border": False,
                    "anchoring": {
                        "anchor_type": "signature_region",
                        "type_reasons": ["sign_here_grid_cell"],
                        "selection_reasons": ["table_fill_signature_grid"],
                        "label_overlap_ratio": 0.0,
                        "candidate_count": 1,
                        "top_candidates": [],
                        "key_block_id": None,
                        "value_block_ids": [],
                        "table_id": table.get("table_block_id"),
                    },
                }
            )
        suppression.append((page, table_box))

    return mappings, suppression
