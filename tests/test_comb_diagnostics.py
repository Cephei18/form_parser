"""Tests for comb-group detection diagnostics (src/comb_diagnostics.py)."""
from __future__ import annotations

from src.comb_diagnostics import analyze_comb_groups


def _cells(xs, y=0.5, w=0.02, h=0.02, page=1):
    return [{"page": page, "bbox": {"x": x, "y": y, "width": w, "height": h}} for x in xs]


def test_regular_pitch_detected_as_comb():
    cells = _cells([0.10, 0.13, 0.16, 0.19, 0.22])  # pitch 0.03, regular
    diag = analyze_comb_groups(selection_regions=cells)
    assert diag["likely_comb_group_count"] == 1
    g = diag["groups"][0]
    assert g["cell_count"] == 5
    assert g["pitch_regularity"] >= 0.9


def test_irregular_pitch_rejected():
    cells = _cells([0.10, 0.20, 0.25])  # gaps 0.10, 0.05 -> irregular
    diag = analyze_comb_groups(selection_regions=cells)
    assert diag["likely_comb_group_count"] == 0


def test_too_few_cells_rejected():
    cells = _cells([0.10, 0.13])  # only 2 cells (< min 3)
    diag = analyze_comb_groups(selection_regions=cells)
    assert diag["likely_comb_group_count"] == 0


def test_single_char_text_cells_count():
    text_boxes = [
        {"page": 1, "bbox": {"x": x, "y": 0.4, "width": 0.02, "height": 0.02}, "text": "1"}
        for x in (0.10, 0.14, 0.18, 0.22)
    ]
    diag = analyze_comb_groups(text_boxes=text_boxes)
    assert diag["likely_comb_group_count"] == 1
    assert diag["groups"][0]["sources"] == ["text"]


def test_diagnostics_only_flag():
    diag = analyze_comb_groups(selection_regions=_cells([0.1, 0.13, 0.16]))
    assert diag["behavior_change"] is False
    assert diag["enabled"] is True
