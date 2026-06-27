"""Tests for logical checkbox-matrix labelling (src/matrix_labeler.py)."""
from __future__ import annotations

import os

import pytest

from src.matrix_labeler import apply_matrix_labeling, matrix_labeling_enabled


def _cb(label, x, y):
    return {"field_type": "checkbox", "label": label, "page": 1,
            "bbox": {"x": x, "y": y, "width": 0.012, "height": 0.01}, "anchoring": {}}


def _etf_grid():
    # 3 schemes x (row-label + Cash + Portfolio).
    cbs = []
    for i, scheme in enumerate(["NIFTY Bank ETF", "NIFTY 50 ETF", "S&P BSE Sensex ETF"]):
        y = 0.10 + i * 0.03
        cbs.append(_cb(scheme, 0.096, y))
        cbs.append(_cb("Cash", 0.395, y))
        cbs.append(_cb("Portfolio", 0.481, y))
    return cbs


@pytest.fixture(autouse=True)
def _clean_env():
    saved = os.environ.pop("FORM_PARSER_MATRIX_LABELING_ENABLED", None)
    yield
    if saved is not None:
        os.environ["FORM_PARSER_MATRIX_LABELING_ENABLED"] = saved
    else:
        os.environ.pop("FORM_PARSER_MATRIX_LABELING_ENABLED", None)


def test_disabled_by_default_no_change():
    assert matrix_labeling_enabled() is False
    cbs = _etf_grid()
    out, diag = apply_matrix_labeling(cbs)
    assert diag == {"enabled": False}
    assert [c["label"] for c in out].count("Cash") == 3  # untouched


def test_enabled_qualifies_option_cells_with_row_label():
    os.environ["FORM_PARSER_MATRIX_LABELING_ENABLED"] = "true"
    cbs = _etf_grid()
    out, diag = apply_matrix_labeling(cbs)
    assert diag["enabled"] is True
    assert diag["cells_relabeled"] == 6  # 3 rows x 2 option cells
    labels = {c["label"] for c in out}
    assert "NIFTY Bank ETF - Cash" in labels
    assert "NIFTY 50 ETF - Portfolio" in labels
    # The scheme (row-label) cells are unchanged.
    assert "NIFTY Bank ETF" in labels


def test_no_matrix_when_too_few_rows():
    os.environ["FORM_PARSER_MATRIX_LABELING_ENABLED"] = "true"
    cbs = [_cb("Yes", 0.3, 0.1), _cb("No", 0.4, 0.1)]  # single row
    out, diag = apply_matrix_labeling(cbs)
    assert diag.get("cells_relabeled", 0) == 0


def test_idempotent_does_not_double_prefix():
    os.environ["FORM_PARSER_MATRIX_LABELING_ENABLED"] = "true"
    cbs = _etf_grid()
    apply_matrix_labeling(cbs)
    out, diag = apply_matrix_labeling(cbs)  # second pass
    assert "NIFTY Bank ETF - NIFTY Bank ETF" not in {c["label"] for c in out}
