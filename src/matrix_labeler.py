"""Logical labelling for checkbox matrices (Phase D / Bug 4).

A grid of checkboxes — rows = schemes/options, columns = a repeated choice such
as Cash / Portfolio or Yes / No — is detected by the engine as N independent
checkboxes whose labels are the bare column word ("Cash" x32, "Portfolio" x32).
That makes the widgets ambiguous: nothing ties a given "Cash" tick to its row.

This layer adds the row context to each matrix cell's *label only* — it never
moves a widget, changes a type, or adds/drops anything — so it is safe to leave
on. It finds the row-label column (the column whose cell labels are mostly
distinct down the rows, e.g. the scheme names) and rewrites each option cell's
label to ``"<row label> - <option>"``.

Flag-gated OFF by default (``FORM_PARSER_MATRIX_LABELING_ENABLED``): purely a
label-quality improvement to be enabled deliberately in rollout.
"""
from __future__ import annotations

import os
import re
from collections import defaultdict
from typing import Any

# Matrix qualification thresholds.
_MIN_ROWS = 3
_MIN_COLS = 2
_COL_X_TOL = 0.02          # checkboxes within this x are the same column
_ROW_Y_TOL = 0.008         # checkboxes within this y are the same row
_LABEL_COL_DISTINCT = 0.7  # fraction of distinct labels for a row-label column


def _bool_env(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def matrix_labeling_enabled() -> bool:
    """True when checkbox-matrix logical labelling is active (default OFF)."""
    return _bool_env("FORM_PARSER_MATRIX_LABELING_ENABLED", False)


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip())


def _cluster(values: list[float], tol: float) -> list[float]:
    """Return cluster centres for 1-D values within tol."""
    centres: list[float] = []
    for v in sorted(values):
        if centres and abs(v - centres[-1]) <= tol:
            continue
        centres.append(v)
    return centres


def apply_matrix_labeling(mappings: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Qualify checkbox-matrix cell labels with their row label. Returns the
    (mutated) mappings and a diagnostics dict."""
    if not matrix_labeling_enabled():
        return mappings, {"enabled": False}

    relabeled = 0
    matrices = 0
    by_page: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for m in mappings:
        if m.get("field_type") == "checkbox" and isinstance(m.get("bbox"), dict):
            by_page[int(m.get("page") or 1)].append(m)

    for checkboxes in by_page.values():
        if len(checkboxes) < _MIN_ROWS * _MIN_COLS:
            continue
        # Column centres by x.
        col_centres = _cluster([float(c["bbox"]["x"]) for c in checkboxes], _COL_X_TOL)
        if len(col_centres) < _MIN_COLS:
            continue

        def col_of(cb: dict[str, Any]) -> int:
            x = float(cb["bbox"]["x"])
            return min(range(len(col_centres)), key=lambda i: abs(col_centres[i] - x))

        # Rows by y.
        rows: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for cb in checkboxes:
            rows[round(float(cb["bbox"]["y"]) / _ROW_Y_TOL)].append(cb)
        grid_rows = [r for r in rows.values() if len(r) >= _MIN_COLS]
        if len(grid_rows) < _MIN_ROWS:
            continue

        # Distinctness per column across the grid rows: the row-label column has
        # mostly unique labels (scheme names); option columns repeat ("Cash").
        col_labels: dict[int, list[str]] = defaultdict(list)
        for row in grid_rows:
            for cb in row:
                col_labels[col_of(cb)].append(_norm(cb.get("label")).lower())
        label_col = None
        for ci in range(len(col_centres)):
            labels = [t for t in col_labels.get(ci, []) if t]
            if len(labels) >= _MIN_ROWS and len(set(labels)) / len(labels) >= _LABEL_COL_DISTINCT:
                label_col = ci
                break
        if label_col is None:
            continue

        matrices += 1
        for row in grid_rows:
            row_label_cb = next((cb for cb in row if col_of(cb) == label_col), None)
            if row_label_cb is None:
                continue
            row_label = _norm(row_label_cb.get("label"))
            if not row_label:
                continue
            for cb in row:
                if col_of(cb) == label_col:
                    continue
                option = _norm(cb.get("label"))
                if option and not option.lower().startswith(row_label.lower()):
                    cb["label"] = f"{row_label} - {option}"
                    cb.setdefault("anchoring", {}).setdefault("type_reasons", []).append("matrix_row_qualified")
                    relabeled += 1

    return mappings, {"enabled": True, "matrices": matrices, "cells_relabeled": relabeled}
