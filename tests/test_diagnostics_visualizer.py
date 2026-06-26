"""Tests for the diagnostics visualizer (Task 4)."""
from __future__ import annotations

import cv2
import numpy as np

from src.diagnostics_visualizer import render_diagnostics_pages


def test_renders_three_overlays_per_page(tmp_path):
    img = np.full((400, 400, 3), 255, dtype=np.uint8)
    page = tmp_path / "page_1.png"
    cv2.imwrite(str(page), img)

    mappings = [
        {"field_id": 1, "field_type": "checkbox", "page": 1, "bbox": {"x": 0.1, "y": 0.1, "width": 0.03, "height": 0.03},
         "confidence_level": "LOW", "calibration": {"reasons": ["duplicate_region"]}},
        {"field_id": 2, "field_type": "text", "page": 1, "bbox": {"x": 0.3, "y": 0.3, "width": 0.4, "height": 0.03},
         "confidence_level": "MEDIUM"},
    ]
    diagnostics = {
        "anchoring": {
            "comb_diagnostics": {"groups": [{"page": 1, "cell_count": 4, "bbox": {"x": 0.1, "y": 0.5, "width": 0.4, "height": 0.02}}]},
            "dotted_underlines": {"detected": [{"page": 1, "bbox": {"x": 0.1, "y": 0.6, "width": 0.3, "height": 0.005}}]},
            "checkbox_validation": {"rejected": [{"field_id": 1, "page": 1}]},
            "table_intelligence": {"input_cells": [{"page": 1, "bbox": {"x": 0.6, "y": 0.7, "width": 0.05, "height": 0.03}}]},
        }
    }
    written = render_diagnostics_pages({1: page}, mappings, diagnostics, tmp_path / "viz")
    assert (tmp_path / "viz" / "page_1_overlay.png").is_file()
    assert (tmp_path / "viz" / "page_1_structure.png").is_file()
    assert (tmp_path / "viz" / "page_1_diff.png").is_file()
    assert len(written["overlay"]) == 1 and len(written["diff"]) == 1


def test_missing_raster_skipped(tmp_path):
    written = render_diagnostics_pages({1: tmp_path / "nope.png"}, [], {}, tmp_path / "viz")
    assert written == {"overlay": [], "structure": [], "diff": []}
