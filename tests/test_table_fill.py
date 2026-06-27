"""Tests for empty input-grid cell emission (src/table_fill.py)."""
from __future__ import annotations

import os

import pytest

from src.section_detector import SectionIndex
from src.table_fill import emit_table_input_cells, table_fill_enabled


def _cell(r, c, text, x, y, w=0.15, h=0.04):
    return {
        "cell_id": f"R{r}C{c}",
        "row_index": r,
        "column_index": c,
        "row_span": 1,
        "column_span": 1,
        "text": text,
        "confidence": 90.0,
        "geometry": {"bounding_box": {"Left": x, "Top": y, "Width": w, "Height": h}},
    }


def _table(cells, rows, cols, page=1, tid="t1"):
    return {
        "table_block_id": tid,
        "page": page,
        "dimensions": {"rows": rows, "columns": cols},
        "cells": cells,
    }


def _emit(tables, existing=None):
    return emit_table_input_cells(
        tables,
        section_index=SectionIndex([]),
        page_px=lambda p: (1000, 1300),
        existing_field_boxes=existing or [],
    )


@pytest.fixture(autouse=True)
def _clean_env():
    saved = os.environ.pop("FORM_PARSER_TABLE_FILL_ENABLED", None)
    yield
    if saved is not None:
        os.environ["FORM_PARSER_TABLE_FILL_ENABLED"] = saved
    else:
        os.environ.pop("FORM_PARSER_TABLE_FILL_ENABLED", None)


def test_enabled_by_default_and_kill_switch():
    assert table_fill_enabled() is True
    os.environ["FORM_PARSER_TABLE_FILL_ENABLED"] = "false"
    assert table_fill_enabled() is False


def test_label_column_grid_no_header_row():
    # form_3 shape: column 1 = row labels, columns 2-3 empty data cells, no
    # internal header row. Every empty data cell becomes a widget.
    cells = []
    labels = ["High School", "College", "Other"]
    for i, lab in enumerate(labels, start=1):
        y = 0.50 + i * 0.05
        cells.append(_cell(i, 1, lab, 0.10, y))
        cells.append(_cell(i, 2, "", 0.30, y))
        cells.append(_cell(i, 3, "", 0.50, y))
    out = _emit([_table(cells, 3, 3)])
    assert len(out) == 6  # 3 rows x 2 data columns
    assert all(m["field_type"] == "text" for m in out)
    assert all(m["anchoring"]["anchor_type"] == "table_cell" for m in out)
    assert all(m["render_border"] is False for m in out)
    labels_out = {m["label"] for m in out}
    assert "High School (column 2)" in labels_out
    assert "College (column 3)" in labels_out


def test_header_row_only_grid_emits_blank_rows():
    # references shape: header row + two blank input rows, no row-label column.
    cells = [
        _cell(1, 1, "Name", 0.10, 0.30),
        _cell(1, 2, "Phone", 0.40, 0.30),
        _cell(2, 1, "", 0.10, 0.35),
        _cell(2, 2, "", 0.40, 0.35),
        _cell(3, 1, "", 0.10, 0.40),
        _cell(3, 2, "", 0.40, 0.40),
    ]
    out = _emit([_table(cells, 3, 2)])
    assert len(out) == 4  # 2 blank rows x 2 columns
    labels_out = {m["label"] for m in out}
    assert "Name (row 2)" in labels_out
    assert "Phone (row 3)" in labels_out


def test_internal_header_row_used_for_column_labels():
    # Realistic internal header row: mostly filled (Degree + Year), labelling the
    # data columns of a single blank data row beneath it.
    cells = [
        _cell(1, 1, "", 0.10, 0.30),
        _cell(1, 2, "Degree", 0.40, 0.30),
        _cell(1, 3, "Year", 0.60, 0.30),
        _cell(2, 1, "College", 0.10, 0.35),
        _cell(2, 2, "", 0.40, 0.35),
        _cell(2, 3, "", 0.60, 0.35),
    ]
    out = _emit([_table(cells, 2, 3)])
    assert [m["label"] for m in out] == ["College - Degree", "College - Year"]


def test_fully_populated_table_emits_nothing():
    cells = [_cell(r, c, f"v{r}{c}", 0.1 + c * 0.2, 0.3 + r * 0.05) for r in (1, 2) for c in (1, 2)]
    assert _emit([_table(cells, 2, 2)]) == []


def test_layout_table_without_structure_skipped():
    # No header row, no label column (each column half-filled) -> not an input grid.
    cells = [
        _cell(1, 1, "a", 0.10, 0.30),
        _cell(1, 2, "", 0.40, 0.30),
        _cell(2, 1, "", 0.10, 0.35),
        _cell(2, 2, "b", 0.40, 0.35),
    ]
    assert _emit([_table(cells, 2, 2)]) == []


def test_covered_cells_skipped():
    cells = [
        _cell(1, 1, "College", 0.10, 0.35),
        _cell(1, 2, "", 0.40, 0.35, w=0.15, h=0.04),
    ]
    # An existing widget already covers the only data cell.
    existing = [(1, {"x": 0.40, "y": 0.35, "width": 0.15, "height": 0.04})]
    assert _emit([_table(cells, 2, 2)], existing=existing) == []


def test_disabled_emits_nothing():
    os.environ["FORM_PARSER_TABLE_FILL_ENABLED"] = "false"
    cells = [
        _cell(1, 1, "College", 0.10, 0.35),
        _cell(2, 1, "School", 0.10, 0.40),
        _cell(1, 2, "", 0.40, 0.35),
        _cell(2, 2, "", 0.40, 0.40),
    ]
    assert _emit([_table(cells, 2, 2)]) == []


def test_small_table_ignored():
    assert _emit([_table([_cell(1, 1, "x", 0.1, 0.3)], 1, 1)]) == []


# --- signature-style table emission ------------------------------------------
from src.table_fill import emit_signature_table_cells  # noqa: E402


def _tb_word(text, x, y, w, h=0.012, page=1):
    return {"bbox": {"x": x, "y": y, "width": w, "height": h}, "text": text, "page": page}


def _sig_emit(tables, text_boxes, leaders):
    return emit_signature_table_cells(
        tables,
        text_boxes=text_boxes,
        visual_features={"synthetic_underlines": leaders, "underlines": []},
        section_index=SectionIndex([]),
        page_px=lambda p: (1000, 1300),
    )


def test_signature_table_emits_inline_answers():
    # 2x2: every cell = label + dotted fill line (leaders cross cell borders).
    cells = [
        _cell(1, 1, "Sign. Guardian", 0.07, 0.85, w=0.23, h=0.033),
        _cell(1, 2, "Sign. Patient", 0.30, 0.85, w=0.60, h=0.033),
        _cell(2, 1, "Name", 0.07, 0.88, w=0.23, h=0.033),
        _cell(2, 2, "Relation with Patient", 0.30, 0.88, w=0.60, h=0.033),
    ]
    words = [
        _tb_word("Sign. Guardian", 0.087, 0.859, 0.126),
        _tb_word("Sign. Patient", 0.401, 0.861, 0.104),
        _tb_word("Name", 0.088, 0.893, 0.051),
        _tb_word("Relation with Patient", 0.376, 0.894, 0.170),
    ]
    leaders = [
        {"bbox": {"x": 0.088, "y": 0.860, "width": 0.632, "height": 0.014}, "page": 1, "source_type": "dotted"},
        {"bbox": {"x": 0.088, "y": 0.893, "width": 0.633, "height": 0.013}, "page": 1, "source_type": "dotted"},
    ]
    maps, suppression = _sig_emit([_table(cells, 2, 2)], words, leaders)
    labels = {m["label"] for m in maps}
    assert labels == {"Sign. Guardian", "Sign. Patient", "Name", "Relation with Patient"}
    assert len(suppression) == 1
    types = {m["label"]: m["field_type"] for m in maps}
    assert types["Sign. Guardian"] == "signature"
    assert types["Name"] == "text"
    # Guardian answer starts after its label and stops before "Sign. Patient".
    g = next(m for m in maps if m["label"] == "Sign. Guardian")
    assert g["bbox"]["x"] > 0.213
    assert g["bbox"]["x"] + g["bbox"]["width"] <= 0.401


def test_signature_table_skips_checkbox_matrix():
    cells = [
        _cell(1, 1, "[ ] PARK RANGER", 0.07, 0.78, w=0.23, h=0.03),
        _cell(1, 2, "[ ] CLERICAL", 0.30, 0.78, w=0.30, h=0.03),
        _cell(2, 1, "[ ] PARK MAINT", 0.07, 0.81, w=0.23, h=0.03),
        _cell(2, 2, "[ ] OTHER", 0.30, 0.81, w=0.30, h=0.03),
    ]
    leaders = [{"bbox": {"x": 0.07, "y": 0.79, "width": 0.5, "height": 0.02}, "page": 1, "source_type": "broken"}]
    maps, suppression = _sig_emit([_table(cells, 2, 2)], [], leaders)
    assert maps == []
    assert suppression == []


def test_signature_table_skips_without_leader():
    cells = [
        _cell(1, 1, "A", 0.07, 0.85, w=0.23, h=0.03),
        _cell(1, 2, "B", 0.30, 0.85, w=0.60, h=0.03),
    ]
    words = [_tb_word("A", 0.08, 0.86, 0.02), _tb_word("B", 0.31, 0.86, 0.02)]
    maps, suppression = _sig_emit([_table(cells, 1, 2)], words, leaders=[])
    assert maps == []


def test_signature_table_skips_input_grid_with_empty_cells():
    cells = [
        _cell(1, 1, "High School", 0.07, 0.50, w=0.20, h=0.04),
        _cell(1, 2, "", 0.30, 0.50, w=0.20, h=0.04),
    ]
    leaders = [{"bbox": {"x": 0.07, "y": 0.51, "width": 0.4, "height": 0.02}, "page": 1, "source_type": "dotted"}]
    maps, _ = _sig_emit([_table(cells, 1, 2)], [], leaders)
    assert maps == []


# --- signature "Sign Here" grid ----------------------------------------------
from src.table_fill import emit_signature_grid_cells  # noqa: E402


def _grid_emit(tables):
    return emit_signature_grid_cells(tables, section_index=SectionIndex([]), page_px=lambda p: (1000, 1400))


def test_sign_here_grid_emits_signature_widgets():
    # Row 1 = "Sign Here" boxes, row 2 = applicant labels (EUIN block shape).
    cells = [
        _cell(1, 1, "Sign Here", 0.063, 0.196, w=0.301, h=0.034),
        _cell(1, 2, "Sign Here", 0.363, 0.196, w=0.304, h=0.034),
        _cell(1, 3, "Sign Here", 0.668, 0.196, w=0.303, h=0.034),
        _cell(2, 1, "First / Sole Applicant", 0.063, 0.230, w=0.301, h=0.015),
        _cell(2, 2, "Second Applicant", 0.363, 0.230, w=0.304, h=0.015),
        _cell(2, 3, "Third Applicant", 0.668, 0.230, w=0.303, h=0.015),
    ]
    maps, suppression = _grid_emit([_table(cells, 2, 3)])
    assert len(maps) == 3
    assert all(m["field_type"] == "signature" for m in maps)
    labels = {m["label"] for m in maps}
    assert labels == {"First / Sole Applicant", "Second Applicant", "Third Applicant"}
    # The signature box is the wide "Sign Here" cell, not a thin line.
    assert all(m["bbox"]["width"] > 0.25 for m in maps)
    assert len(suppression) == 1


def test_sign_dot_guardian_not_treated_as_sign_here_grid():
    # form_5 bottom table cells ("Sign. Guardian") are dotted-leader labels,
    # NOT "Sign Here" boxes -> the grid emitter must ignore them.
    cells = [
        _cell(1, 1, "Sign. Guardian", 0.07, 0.85, w=0.23, h=0.03),
        _cell(1, 2, "Sign. Patient", 0.30, 0.85, w=0.60, h=0.03),
    ]
    maps, suppression = _grid_emit([_table(cells, 1, 2)])
    assert maps == []
    assert suppression == []
