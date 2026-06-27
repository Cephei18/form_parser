"""Tests for comb-run detection (src/comb_field_detector.py).

Synthetic rasters: a row of empty boxes is a comb run; a row of ink-filled
boxes (glyphs) is not; a forms-free / single-box page yields nothing.
"""
from __future__ import annotations

import os

import cv2
import numpy as np
import pytest

from src.comb_field_detector import comb_run_enabled, detect_comb_runs


@pytest.fixture(autouse=True)
def _clean_env():
    saved = os.environ.pop("FORM_PARSER_COMB_RUN_ENABLED", None)
    yield
    if saved is not None:
        os.environ["FORM_PARSER_COMB_RUN_ENABLED"] = saved
    else:
        os.environ.pop("FORM_PARSER_COMB_RUN_ENABLED", None)


def _canvas():
    return np.full((1000, 1400, 3), 255, np.uint8)


def _draw_empty_boxes(img, x0, y, n, cell=16, gap=5):
    x = x0
    for _ in range(n):
        cv2.rectangle(img, (x, y), (x + cell, y + cell), (0, 0, 0), 1)
        x += cell + gap


def test_enabled_by_default_and_kill_switch():
    assert comb_run_enabled() is True
    os.environ["FORM_PARSER_COMB_RUN_ENABLED"] = "false"
    assert comb_run_enabled() is False


def test_detects_row_of_empty_boxes(tmp_path):
    img = _canvas()
    _draw_empty_boxes(img, 200, 300, 10)
    p = tmp_path / "comb.png"
    cv2.imwrite(str(p), img)
    runs = detect_comb_runs(p, page=1)
    assert len(runs) == 1
    r = runs[0]
    assert r["cell_count"] >= 8
    assert r["width"] > 0.12  # spans the whole run
    assert r["page"] == 1


def test_ink_filled_boxes_not_a_comb(tmp_path):
    # Filled squares look like glyphs (interior inked) -> not a comb run.
    img = _canvas()
    x = 200
    for _ in range(10):
        cv2.rectangle(img, (x, 300), (x + 26, 326), (0, 0, 0), -1)  # filled
        x += 30
    p = tmp_path / "filled.png"
    cv2.imwrite(str(p), img)
    assert detect_comb_runs(p, page=1) == []


def test_short_row_below_min_cells(tmp_path):
    img = _canvas()
    _draw_empty_boxes(img, 200, 300, 3)  # fewer than _MIN_CELLS
    p = tmp_path / "short.png"
    cv2.imwrite(str(p), img)
    assert detect_comb_runs(p, page=1) == []


def test_blank_page_yields_nothing(tmp_path):
    p = tmp_path / "blank.png"
    cv2.imwrite(str(p), _canvas())
    assert detect_comb_runs(p, page=1) == []


def test_missing_image_returns_empty(tmp_path):
    assert detect_comb_runs(tmp_path / "nope.png", page=1) == []
