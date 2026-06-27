"""Detect comb runs (rows of empty character boxes) from a page raster.

Many forms lay out fixed-length fields (PAN, KYC, DP ID, Beneficiary A/c, PIN,
Folio No, …) as a row of small equal boxes — one character per cell. Textract
collapses such a field's value to the first box, so the anchoring engine renders
only a tiny widget over the leftmost cell. This module finds the full run so the
field can be widened into a proper comb.

The hard part is telling comb boxes apart from text and table borders. The
decisive signal used here is that a comb cell is an *empty box*: its interior is
almost all white, whereas a text glyph is ink-filled. Candidate cells are
size/aspect-filtered, kept only when their interior ink ratio is low, then
grouped into horizontal runs with consistent cell size and regular spacing. On
forms without comb fields this yields nothing (verified on the single-page
corpus forms), so enabling it does not perturb them.

ON by default; ``FORM_PARSER_COMB_RUN_ENABLED=false`` disables it.
"""
from __future__ import annotations

import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger("form_parser.comb_field")

# Cell geometry as page fractions.
_CELL_MIN_W = 0.004
_CELL_MAX_W = 0.032
_CELL_MIN_H = 0.007
_CELL_MAX_H = 0.026
_CELL_MIN_AR = 0.40
_CELL_MAX_AR = 2.8
# Interior ink ratio above which a candidate is a glyph, not an empty box.
_INTERIOR_INK_MAX = 0.20
# A run needs at least this many cells with regular spacing.
_MIN_CELLS = 5
_MAX_SPACING_CV = 0.45

# --- Continuous-comb (shared-wall) detection ------------------------------- #
# Some combs are drawn as one box with internal dividers (cells share walls), so
# the cells are not separate contours. They are also often printed in LIGHT GREY
# that OTSU thresholding drops, so we threshold at a light cut-off and look for a
# regular run of short vertical divider strokes. Regularity is the discriminator:
# >= _MIN_DIVIDERS strokes with spacing CV < _MAX_DIVIDER_CV, in an interior that
# is empty of dark ink. Verified to yield nothing on comb-free forms.
_DIVIDER_GREY_THRESHOLD = 205
_DARK_INK_THRESHOLD = 140
_MIN_DIVIDERS = 7
_MAX_DIVIDER_CV = 0.20
_DIVIDER_INTERIOR_INK_MAX = 0.10


def _bool_env(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def comb_run_enabled() -> bool:
    """True when comb-run detection + field widening is active (default ON)."""
    return _bool_env("FORM_PARSER_COMB_RUN_ENABLED", True)


def _overlaps(a: dict[str, Any], b: dict[str, Any]) -> bool:
    ax2, ay2 = a["x"] + a["width"], a["y"] + a["height"]
    bx2, by2 = b["x"] + b["width"], b["y"] + b["height"]
    ix = max(0.0, min(ax2, bx2) - max(a["x"], b["x"]))
    iy = max(0.0, min(ay2, by2) - max(a["y"], b["y"]))
    inter = ix * iy
    return inter / max(min(a["width"] * a["height"], b["width"] * b["height"]), 1e-9) > 0.5


def detect_comb_runs(image_path: str | Path, page: int = 1) -> list[dict[str, Any]]:
    """Return comb runs on a page raster as ``{x,y,width,height,cell_count,page}``
    in normalised page coordinates. Combines two methods: separate empty boxes
    (distinct contours) and continuous combs (regular light-grey dividers).
    Empty list when the image is unreadable or no comb run is found."""
    image = cv2.imread(str(image_path))
    if image is None:
        return []
    height, width = image.shape[:2]
    if height <= 0 or width <= 0:
        return []
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    runs = _detect_separate_box_runs(gray, width, height, page)
    for run in _detect_divider_runs(gray, width, height, page):
        if not any(_overlaps(run, existing) for existing in runs):
            runs.append(run)
    return runs


def _detect_separate_box_runs(gray: "np.ndarray", width: int, height: int, page: int) -> list[dict[str, Any]]:
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    integral = cv2.integral(binary // 255)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cells: list[tuple[int, int, int, int]] = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if not (_CELL_MIN_W * width <= w <= _CELL_MAX_W * width):
            continue
        if not (_CELL_MIN_H * height <= h <= _CELL_MAX_H * height):
            continue
        if not (_CELL_MIN_AR <= w / max(h, 1) <= _CELL_MAX_AR):
            continue
        ix0, iy0 = int(x + 0.22 * w), int(y + 0.22 * h)
        ix1, iy1 = int(x + 0.78 * w), int(y + 0.78 * h)
        if ix1 <= ix0 or iy1 <= iy0:
            continue
        ink_sum = integral[iy1, ix1] - integral[iy0, ix1] - integral[iy1, ix0] + integral[iy0, ix0]
        ink_ratio = ink_sum / max((ix1 - ix0) * (iy1 - iy0), 1)
        if ink_ratio > _INTERIOR_INK_MAX:
            continue  # interior is inked -> a glyph, not an empty box
        cells.append((x, y, w, h))

    # Bucket cells by row band, then carve maximal runs of consistent size +
    # regular spacing.
    rows: dict[int, list[tuple[int, int, int, int]]] = defaultdict(list)
    row_band = max(1.0, 0.010 * height)
    for cell in cells:
        rows[round((cell[1] + cell[3] / 2) / row_band)].append(cell)

    runs: list[dict[str, Any]] = []
    for row in rows.values():
        row.sort()
        i = 0
        while i < len(row):
            seq = [row[i]]
            j = i + 1
            while j < len(row):
                px, _, pw, _ = seq[-1]
                x, _, w, _ = row[j]
                gap = x - (px + pw)
                if abs(w - pw) <= max(4, 0.55 * pw) and -3 <= gap <= max(pw * 1.6, 0.012 * width):
                    seq.append(row[j])
                    j += 1
                else:
                    break
            if len(seq) >= _MIN_CELLS:
                xs = [s[0] for s in seq]
                gaps = [xs[k + 1] - xs[k] for k in range(len(xs) - 1)]
                spacing_cv = (float(np.std(gaps)) / max(float(np.mean(gaps)), 1.0)) if gaps else 9.0
                if spacing_cv < _MAX_SPACING_CV:
                    x0 = min(xs)
                    x1 = max(s[0] + s[2] for s in seq)
                    y0 = min(s[1] for s in seq)
                    y1 = max(s[1] + s[3] for s in seq)
                    runs.append(
                        {
                            "x": round(x0 / width, 6),
                            "y": round(y0 / height, 6),
                            "width": round((x1 - x0) / width, 6),
                            "height": round((y1 - y0) / height, 6),
                            "cell_count": len(seq),
                            "page": page,
                        }
                    )
                    i = j
                    continue
            i += 1
    return runs


def _detect_divider_runs(gray: "np.ndarray", width: int, height: int, page: int) -> list[dict[str, Any]]:
    """Detect continuous combs via a regular run of short light-grey vertical
    dividers over a dark-ink-empty interior."""
    light = (gray < _DIVIDER_GREY_THRESHOLD).astype(np.uint8) * 255
    dark_integral = cv2.integral((gray < _DARK_INK_THRESHOLD).astype(np.uint8))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(7, int(0.009 * height))))
    vertical = cv2.morphologyEx(light, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    segs: list[tuple[int, int, int, int]] = []
    max_w = max(3, int(0.0035 * width))
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w <= max_w and 0.009 * height <= h <= 0.028 * height:
            segs.append((x, y, w, h))

    rows: dict[int, list[tuple[int, int, int, int]]] = defaultdict(list)
    band = max(1.0, 0.011 * height)
    for seg in segs:
        rows[round((seg[1] + seg[3] / 2) / band)].append(seg)

    runs: list[dict[str, Any]] = []
    for row in rows.values():
        row.sort()
        i = 0
        while i < len(row):
            seq = [row[i]]
            j = i + 1
            while j < len(row):
                gap = row[j][0] - seq[-1][0]
                if 0.007 * width <= gap <= 0.040 * width:
                    seq.append(row[j])
                    j += 1
                else:
                    break
            if len(seq) >= _MIN_DIVIDERS:
                xs = [s[0] for s in seq]
                gaps = [xs[k + 1] - xs[k] for k in range(len(xs) - 1)]
                spacing_cv = (float(np.std(gaps)) / max(float(np.mean(gaps)), 1.0)) if gaps else 9.0
                if spacing_cv < _MAX_DIVIDER_CV:
                    x0 = min(xs)
                    x1 = max(s[0] + s[2] for s in seq)
                    y0 = min(s[1] for s in seq)
                    y1 = max(s[1] + s[3] for s in seq)
                    ix0, iy0 = int(x0 + (x1 - x0) * 0.02), int(y0 + (y1 - y0) * 0.2)
                    ix1, iy1 = int(x1 - (x1 - x0) * 0.02), int(y1 - (y1 - y0) * 0.2)
                    ink = 1.0
                    if ix1 > ix0 and iy1 > iy0:
                        s_ = (
                            dark_integral[iy1, ix1] - dark_integral[iy0, ix1]
                            - dark_integral[iy1, ix0] + dark_integral[iy0, ix0]
                        )
                        ink = s_ / max((ix1 - ix0) * (iy1 - iy0), 1)
                    if ink < _DIVIDER_INTERIOR_INK_MAX:
                        runs.append(
                            {
                                "x": round(x0 / width, 6),
                                "y": round(y0 / height, 6),
                                "width": round((x1 - x0) / width, 6),
                                "height": round((y1 - y0) / height, 6),
                                "cell_count": len(seq),
                                "page": page,
                            }
                        )
                        i = j
                        continue
            i += 1
    return runs
