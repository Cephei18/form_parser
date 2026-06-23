"""Phase J — Table Intelligence Engine.

The pipeline understands individual fields, underlines, rectangles and (since
Phase I) generalized answer regions — but it does not understand *tables*. A
"School | Degree | Year" grid with empty rows beneath it is seen as a handful of
independent cells, not as an input table whose blank cells are answer slots.

This module adds first-class table understanding from signals already present
(normalised Textract TABLE/CELL geometry + text, sections, Phase I answer
regions). For each Textract table it answers: is this a table, what *type*
(LAYOUT / INPUT / REPEATING_GROUP / MATRIX / LINE_ITEM), where are the headers,
which cells are intended for user input, and are the rows repeating.

It is detection + structuring only. It does not render, classify fields, or pick
winners. The anchor engine adapts the detected **input cells** into ordinary
``table_cell`` candidates so the existing selection / global-assignment stage is
unchanged — table intelligence simply produces *better* candidates.

Gated behind ``FORM_PARSER_TABLE_INTELLIGENCE_ENABLED`` (default OFF). When OFF
the engine is never invoked and behaviour is byte-identical.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any

# --- Table types ------------------------------------------------------------ #
LAYOUT_TABLE = "LAYOUT_TABLE"
INPUT_TABLE = "INPUT_TABLE"
REPEATING_GROUP = "REPEATING_GROUP"
MATRIX = "MATRIX"
LINE_ITEM = "LINE_ITEM"

TABLE_TYPES = frozenset({LAYOUT_TABLE, INPUT_TABLE, REPEATING_GROUP, MATRIX, LINE_ITEM})

# Selection glyphs / tokens that mark a matrix option cell.
SELECTION_TOKENS = {"[x]", "[ ]", "☐", "□", "■", "●", "○", "✓", "x", "selected", "not_selected"}

# Structural header cues for a line-item table (currency / quantity columns).
# These are structural form-vocabulary cues (same spirit as the existing section
# / comb / radio keyword tables), NOT semantic content understanding.
LINE_ITEM_HEADER_CUES = (
    "amount",
    "qty",
    "quantity",
    "price",
    "rate",
    "total",
    "units",
    "value",
    "sum",
    "cost",
)

_FILL_HEADER_THRESHOLD = 0.6   # row fill ratio to count as a header row
_EMPTY_BODY_THRESHOLD = 0.5    # body cells emptier than this => input table


def _bool_env(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def table_intelligence_enabled() -> bool:
    """True when the Phase J table intelligence engine is active."""
    return _bool_env("FORM_PARSER_TABLE_INTELLIGENCE_ENABLED", False)


# --- Core objects ----------------------------------------------------------- #
@dataclass
class TableCell:
    row: int
    column: int
    bbox: tuple[float, float, float, float]
    text: str | None
    is_input: bool
    confidence: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def bbox_dict(self) -> dict[str, float]:
        x, y, w, h = self.bbox
        return {"x": x, "y": y, "width": w, "height": h}

    def to_dict(self) -> dict[str, Any]:
        return {
            "row": self.row,
            "column": self.column,
            "bbox": self.bbox_dict(),
            "text": self.text,
            "is_input": self.is_input,
            "confidence": round(float(self.confidence), 4),
            "metadata": self.metadata,
        }


@dataclass
class Table:
    table_id: str
    page: int
    bbox: tuple[float, float, float, float]
    table_type: str
    headers: list[TableCell]
    rows: list[list[TableCell]]
    columns: list[list[TableCell]]
    cells: list[TableCell]
    metadata: dict[str, Any] = field(default_factory=dict)

    def bbox_dict(self) -> dict[str, float]:
        x, y, w, h = self.bbox
        return {"x": x, "y": y, "width": w, "height": h}

    def to_dict(self) -> dict[str, Any]:
        return {
            "table_id": self.table_id,
            "page": self.page,
            "bbox": self.bbox_dict(),
            "table_type": self.table_type,
            "row_count": self.metadata.get("row_count"),
            "column_count": self.metadata.get("column_count"),
            "header_rows": self.metadata.get("header_rows"),
            "input_cell_count": sum(1 for c in self.cells if c.is_input),
            "metadata": self.metadata,
        }


@dataclass
class TableIntelligenceResult:
    tables: list[Table]
    input_cells: list[TableCell]
    repeating_groups: list[dict[str, Any]]
    rejected_tables: list[dict[str, Any]]

    def to_debug_dict(self) -> dict[str, Any]:
        type_counts: dict[str, int] = {}
        for t in self.tables:
            type_counts[t.table_type] = type_counts.get(t.table_type, 0) + 1
        return {
            "enabled": True,
            "table_count": len(self.tables),
            "tables_detected": [t.to_dict() for t in self.tables],
            "table_types": type_counts,
            "input_cells": [c.to_dict() for c in self.input_cells],
            "repeating_groups": self.repeating_groups,
            "rejected_tables": self.rejected_tables,
        }


# --- Helpers ---------------------------------------------------------------- #
def _norm_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip()).lower()


def _bbox_tuple(box: dict[str, Any] | None) -> tuple[float, float, float, float] | None:
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


def _union(boxes: list[tuple[float, float, float, float]]) -> tuple[float, float, float, float]:
    x1 = min(b[0] for b in boxes)
    y1 = min(b[1] for b in boxes)
    x2 = max(b[0] + b[2] for b in boxes)
    y2 = max(b[1] + b[3] for b in boxes)
    return (round(x1, 6), round(y1, 6), round(x2 - x1, 6), round(y2 - y1, 6))


def _is_selection_text(text: str) -> bool:
    return _norm_text(text) in SELECTION_TOKENS


def _find_period(labels: list[str]) -> int | None:
    """Smallest period p (>=1, <=n/2) such that the label sequence repeats with
    at least two full cycles and the repeated block carries real text."""
    n = len(labels)
    if n < 2:
        return None
    for p in range(1, n // 2 + 1):
        if n % p != 0:
            continue
        if n // p < 2:
            continue
        if any(labels[i] != labels[i % p] for i in range(n)):
            continue
        if any(labels[i] for i in range(p)):  # repeated block is non-empty
            return p
    return None


# --- Per-table analysis ----------------------------------------------------- #
def _analyze_table(table_id: str, page: int, raw_cells: list[dict[str, Any]]) -> Table | None:
    cell_recs: list[dict[str, Any]] = []
    for c in raw_cells:
        bbox = _bbox_tuple(c.get("bbox"))
        if not bbox:
            continue
        cell_recs.append({
            "row": int(c.get("row_index") or 0),
            "col": int(c.get("column_index") or 0),
            "bbox": bbox,
            "text": str(c.get("text") or ""),
            "confidence": float(c.get("confidence") or 0.0),
            "cell_id": c.get("cell_id"),
        })
    if not cell_recs:
        return None

    n_rows = max(c["row"] for c in cell_recs) + 1
    n_cols = max(c["col"] for c in cell_recs) + 1

    by_row: dict[int, list[dict[str, Any]]] = {}
    by_col: dict[int, list[dict[str, Any]]] = {}
    for c in cell_recs:
        by_row.setdefault(c["row"], []).append(c)
        by_col.setdefault(c["col"], []).append(c)

    def fill_ratio(cells: list[dict[str, Any]]) -> float:
        if not cells:
            return 0.0
        return sum(1 for c in cells if c["text"].strip()) / len(cells)

    row_fill = {r: fill_ratio(by_row.get(r, [])) for r in range(n_rows)}
    col_fill = {c: fill_ratio(by_col.get(c, [])) for c in range(n_cols)}

    # Header rows: leading rows that are well-filled.
    header_rows: list[int] = []
    for r in range(n_rows):
        if row_fill.get(r, 0.0) >= _FILL_HEADER_THRESHOLD:
            header_rows.append(r)
        else:
            break
    # Header column (vertical key-value / label column on the left).
    header_cols: list[int] = []
    if not header_rows and n_cols >= 2 and col_fill.get(0, 0.0) >= _FILL_HEADER_THRESHOLD:
        header_cols = [0]

    body_rows = [r for r in range(n_rows) if r not in header_rows]
    body_cells = [c for c in cell_recs if c["row"] in body_rows]
    body_empty_ratio = (
        sum(1 for c in body_cells if not c["text"].strip()) / len(body_cells) if body_cells else 0.0
    )

    # Repeating detection on the first column (or whole-row signature).
    if n_cols == 1:
        seq = [_norm_text(by_row.get(r, [{}])[0].get("text", "")) for r in range(n_rows)]
    else:
        seq = []
        for r in range(n_rows):
            row_cells = sorted(by_row.get(r, []), key=lambda c: c["col"])
            seq.append("|".join(_norm_text(c["text"]) for c in row_cells))
    period = _find_period(seq)

    # Matrix signal: a column (or the table) carrying selection glyphs.
    selection_rows = sum(1 for c in cell_recs if _is_selection_text(c["text"]))

    header_texts = [_norm_text(c["text"]) for r in header_rows for c in by_row.get(r, [])]
    has_line_item_cue = any(any(cue in h for cue in LINE_ITEM_HEADER_CUES) for h in header_texts)

    # --- Classification (priority order) ---
    if selection_rows >= 2:
        table_type = MATRIX
    elif period is not None:
        table_type = REPEATING_GROUP
    elif header_rows and body_cells and body_empty_ratio >= _EMPTY_BODY_THRESHOLD:
        table_type = LINE_ITEM if has_line_item_cue else INPUT_TABLE
    elif header_cols and body_empty_ratio >= _EMPTY_BODY_THRESHOLD:
        table_type = INPUT_TABLE
    else:
        table_type = LAYOUT_TABLE

    # --- Cell intelligence ---
    cells: list[TableCell] = []
    for c in cell_recs:
        is_header = c["row"] in header_rows or c["col"] in header_cols
        is_empty = not c["text"].strip()
        is_label = (not is_header) and (not is_empty)
        is_repeating = table_type == REPEATING_GROUP
        if table_type in {INPUT_TABLE, LINE_ITEM, REPEATING_GROUP}:
            is_input = is_empty and not is_header
        elif table_type == MATRIX:
            # Matrix option cells: selection glyphs, or empty cells outside a label column.
            is_input = _is_selection_text(c["text"]) or (is_empty and c["col"] not in header_cols)
        else:  # LAYOUT_TABLE
            is_input = False
        confidence = 0.82 if is_input and table_type in {INPUT_TABLE, LINE_ITEM} else (0.7 if is_input else 0.6)
        cells.append(TableCell(
            row=c["row"], column=c["col"], bbox=c["bbox"], text=c["text"] or None,
            is_input=is_input, confidence=confidence,
            metadata={
                "cell_id": c["cell_id"], "table_id": table_id, "page": page,
                "is_header": is_header, "is_empty": is_empty, "is_label": is_label,
                "is_repeating": is_repeating,
            },
        ))

    rows = [sorted([c for c in cells if c.row == r], key=lambda c: c.column) for r in range(n_rows)]
    columns = [sorted([c for c in cells if c.column == col], key=lambda c: c.row) for col in range(n_cols)]
    headers = [c for c in cells if c.metadata.get("is_header")]
    table_bbox = _union([c["bbox"] for c in cell_recs])

    metadata: dict[str, Any] = {
        "row_count": n_rows,
        "column_count": n_cols,
        "header_rows": header_rows,
        "header_cols": header_cols,
        "body_empty_ratio": round(body_empty_ratio, 4),
        "row_fill": {str(r): round(v, 4) for r, v in row_fill.items()},
        "has_line_item_cue": has_line_item_cue,
        "selection_cell_count": selection_rows,
    }
    if period is not None:
        metadata["repeating_period"] = period
        metadata["group_count"] = n_rows // period

    return Table(
        table_id=table_id, page=page, bbox=table_bbox, table_type=table_type,
        headers=headers, rows=rows, columns=columns, cells=cells, metadata=metadata,
    )


def _repeating_groups_from_sections(sections: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Detect repeating section groups whose titles share a stem with a trailing
    index ("Nominee 1", "Nominee 2", ...). Section-derived, no semantics."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for sec in sections or []:
        title = _norm_text(sec.get("title"))
        stem = re.sub(r"\s*\d+\s*$", "", title).strip()
        if stem and stem != title:  # had a trailing index
            groups.setdefault(stem, []).append(sec)
    out: list[dict[str, Any]] = []
    for stem, members in groups.items():
        if len(members) >= 2:
            out.append({
                "stem": stem,
                "instance_count": len(members),
                "section_ids": [m.get("section_id") for m in members],
                "source": "section_index",
            })
    return out


# --- Orchestrator ----------------------------------------------------------- #
def build_tables(
    cells: list[dict[str, Any]] | None,
    sections: list[dict[str, Any]] | None = None,
    metrics_by_page: dict[int, dict[str, float]] | None = None,
) -> TableIntelligenceResult:
    """Detect and structure every table from the normalised cell list.

    ``cells`` is the flat list produced by ``field_anchor_engine._table_cells``
    (table_id, cell_id, page, row_index, column_index, text, bbox, confidence).
    """
    cells = cells or []
    by_table: dict[str, list[dict[str, Any]]] = {}
    table_page: dict[str, int] = {}
    for c in cells:
        tid = str(c.get("table_id") or "table")
        by_table.setdefault(tid, []).append(c)
        table_page.setdefault(tid, int(c.get("page") or 1))

    tables: list[Table] = []
    rejected: list[dict[str, Any]] = []
    for tid, raw_cells in by_table.items():
        table = _analyze_table(tid, table_page.get(tid, 1), raw_cells)
        if table is None:
            rejected.append({"table_id": tid, "reason": "no_cell_geometry"})
            continue
        tables.append(table)

    tables.sort(key=lambda t: (t.page, t.bbox[1], t.bbox[0]))

    input_cells = [c for t in tables for c in t.cells if c.is_input]

    repeating_groups: list[dict[str, Any]] = []
    for t in tables:
        if t.table_type == REPEATING_GROUP:
            repeating_groups.append({
                "table_id": t.table_id,
                "page": t.page,
                "group_size": t.metadata.get("repeating_period"),
                "group_count": t.metadata.get("group_count"),
                "source": "table_structure",
            })
    repeating_groups.extend(_repeating_groups_from_sections(sections))

    return TableIntelligenceResult(
        tables=tables,
        input_cells=input_cells,
        repeating_groups=repeating_groups,
        rejected_tables=rejected,
    )
