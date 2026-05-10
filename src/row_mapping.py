"""
row_mapping.py  –  maps OCR rows to detected field lines.

Designed for forms where:
  - Labels are on the LEFT half of the page
  - Field lines are on the RIGHT half of the page
  - Labels and their field lines share approximately the SAME or SLIGHTLY LOWER Y
  - Multi-line labels: the field line aligns with the LAST line of the label, not the center
  - Duplicate lines (both edges of a printed underline) are de-duplicated first
"""


from __future__ import annotations
from typing import Any
import re


# ─────────────────────────────────────────────────────────────────
# Line geometry helpers
# ─────────────────────────────────────────────────────────────────

def _cy(line: dict) -> float:
    return (line["start"][1] + line["end"][1]) / 2.0

def _cx(line: dict) -> float:
    return (line["start"][0] + line["end"][0]) / 2.0

def _lx(line: dict) -> float:
    return float(min(line["start"][0], line["end"][0]))

def _width(line: dict) -> float:
    return abs(line["end"][0] - line["start"][0])


# ─────────────────────────────────────────────────────────────────
# Step 1 — De-duplicate lines
# Canny detects both the top and bottom edge of each printed underline.
# Keep only the first (topmost) of any pair within y_tol px vertically
# that share the same x-extent.
# ─────────────────────────────────────────────────────────────────

def deduplicate_lines(lines: list[dict], y_tol: float = 6.0, x_tol: float = 20.0) -> list[dict]:
    sorted_by_y = sorted(range(len(lines)), key=lambda i: _cy(lines[i]))
    used = [False] * len(lines)
    kept: list[dict] = []

    for pos, i in enumerate(sorted_by_y):
        if used[i]:
            continue
        line = lines[i]
        for j in sorted_by_y[pos + 1:]:
            if used[j]:
                continue
            other = lines[j]
            if abs(_cy(other) - _cy(line)) > y_tol:
                break
            if abs(_lx(other) - _lx(line)) < x_tol and abs(_width(other) - _width(line)) < x_tol:
                used[j] = True
        kept.append(line)

    return kept


# ─────────────────────────────────────────────────────────────────
# Step 2 — Remove table-rule clusters
# A group of 4+ lines at nearly the same Y is a table border, not a field.
# ─────────────────────────────────────────────────────────────────

def remove_table_lines(lines: list[dict], y_tol: float = 14.0, min_group: int = 4) -> list[dict]:
    if not lines:
        return lines
    sl = sorted(lines, key=_cy)
    bad: set[int] = set()
    group = [sl[0]]
    for prev, curr in zip(sl, sl[1:]):
        if abs(_cy(curr) - _cy(prev)) < y_tol:
            group.append(curr)
        else:
            if len(group) >= min_group:
                bad.update(id(ln) for ln in group)
            group = [curr]
    if len(group) >= min_group:
        bad.update(id(ln) for ln in group)
    return [ln for ln in lines if id(ln) not in bad]


# ─────────────────────────────────────────────────────────────────
# Step 3 — Row geometry
# bottom_y = Y of the lowest word (field line aligns with last label line).
# right_x  = capped at column_split so wide labels don't overshoot.
# ─────────────────────────────────────────────────────────────────

def row_label_text(row: list[dict]) -> str:
    return " ".join(item["text"] for item in row).strip()


def row_geometry(row: list[dict], column_split: float = 620.0) -> dict:
    """
    column_split: approximate x boundary between the label column and the
    field-line column.  Labels whose text extends past this (e.g. because
    of long parentheticals) are capped so right_x never exceeds it.
    Tune this to ~60-70 % of your form's page width.
    """
    ys, rights, lefts = [], [], []
    for item in row:
        if "bbox" in item:
            bbox = item["bbox"]
            if isinstance(bbox, (list, tuple)) and len(bbox) == 4 and all(isinstance(v, (int, float)) for v in bbox):
                bx, by, bw, bh = bbox
                ys.append(float(by + bh))
                rights.append(float(bx + bw))
                lefts.append(float(bx))
            else:
                # polygon or list-of-points
                try:
                    xs = [float(pt[0]) for pt in bbox]
                    ys_pts = [float(pt[1]) for pt in bbox]
                    ys.append(max(ys_pts))
                    rights.append(max(xs))
                    lefts.append(min(xs))
                except Exception:
                    cx, cy = item.get("center", (0.0, 0.0))
                    ys.append(float(cy))
                    rights.append(float(cx))
                    lefts.append(float(cx))
        else:
            cx, cy = item.get("center", (0.0, 0.0))
            ys.append(float(cy))
            rights.append(float(cx))
            lefts.append(float(cx))

    raw_right = max(rights)
    return {
        "bottom_y": max(ys),
        "top_y":    min(ys),
        "center_y": (min(ys) + max(ys)) / 2.0,
        "right_x":  min(raw_right, column_split),   # ← capped
        "raw_right_x": raw_right,
        "left_x":   min(lefts),
    }


# ─────────────────────────────────────────────────────────────────
# Step 4 — Match each label to its nearest UNASSIGNED field line
#
# "Unassigned" means not yet the primary match of any earlier label.
# This prevents two labels from sharing the same line as their anchor.
#
# Search window: [bottom_y - above_pad, bottom_y + below_pad]
# Must be to the right of the (capped) label right edge.
# Score: vertical distance only; tie-break by wider line.
# ─────────────────────────────────────────────────────────────────

def best_field_line(
    label_bottom_y: float,
    label_right_x: float,
    candidates: list[dict],
    assigned_ids: set[int],
    above_pad: float = 10.0,
    below_pad: float = 75.0,
) -> tuple[dict | None, float]:
    y_min = label_bottom_y - above_pad
    y_max = label_bottom_y + below_pad

    pool = [
        ln for ln in candidates
        if id(ln) not in assigned_ids
        and y_min <= _cy(ln) <= y_max
        and _lx(ln) >= label_right_x - 40
    ]

    if not pool:
        return None, float("inf")

    pool.sort(key=lambda ln: (abs(_cy(ln) - label_bottom_y), -_width(ln)))
    best = pool[0]
    return best, abs(_cy(best) - label_bottom_y)


# ─────────────────────────────────────────────────────────────────
# Step 5 — Expand multiline field blocks
#
# Only pull in additional lines that are:
#   • x-aligned with the seed (same field column)
#   • directly below with no label row between them
#   • within y_step px of the previous grouped line
#
# label_ys is the sorted list of all label bottom_y values — used to
# detect if a label exists between two candidate lines (which would mean
# the next line belongs to that label, not to the current continuation).
# ─────────────────────────────────────────────────────────────────

def expand_multiline(
    seed: dict,
    pool: list[dict],
    label_ys: list[float],
    x_tol: float = 120.0,
    y_step: float = 75.0,
) -> list[dict]:
    visited = {id(seed)}
    group = [seed]
    frontier = [seed]

    while frontier:
        current = frontier.pop()
        ccx, ccy = _cx(current), _cy(current)

        for ln in pool:
            if id(ln) in visited:
                continue
            lx, ly = _cx(ln), _cy(ln)
            if not (abs(lx - ccx) < x_tol and 0 < (ly - ccy) < y_step):
                continue
            # Check: is there a label between ccy and ly?
            # If yes, that line belongs to that label — don't steal it.
            label_between = any(ccy < label_y < ly for label_y in label_ys)
            if label_between:
                continue
            visited.add(id(ln))
            group.append(ln)
            frontier.append(ln)

    return sorted(group, key=_cy)


# ─────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────

_NOISE_KEYWORDS = {"snb", "jspcb", "affix", "contd"}


def map_rows_to_fields(
    rows: list[list[dict]],
    lines: list[dict],
    above_pad: float = 10.0,
    below_pad: float = 75.0,
    max_vert_dist: float = 85.0,
    column_split: float = 620.0,
) -> list[dict[str, Any]]:
    """
    Parameters
    ----------
    rows          : grouped OCR rows from row_grouping
    lines         : filtered horizontal lines from detect_fields
    above_pad     : px above label bottom_y included in search window
    below_pad     : px below label bottom_y included in search window
    max_vert_dist : reject match if vertical distance exceeds this (px)
    column_split  : x boundary separating label column from field column;
                    label right_x is capped here so wide labels don't
                    overshoot past the field lines
    """
    clean = deduplicate_lines(lines)
    clean = remove_table_lines(clean)
    print(f"[map] {len(lines)} raw -> {len(clean)} clean lines")

    # Pre-compute all label bottom_y values for multiline expansion guard
    all_geoms = []
    valid_rows = []
    for row in rows:
        if not row:
            continue
        label = row_label_text(row)
        if any(kw in label.lower() for kw in _NOISE_KEYWORDS):
            continue
        geom = row_geometry(row, column_split=column_split)
        all_geoms.append(geom)
        valid_rows.append((label, geom))

    label_ys = sorted(g["bottom_y"] for g in all_geoms)

    assigned_primary: set[int] = set()   # line ids used as primary anchors
    mappings: list[dict] = []

    for label, geom in valid_rows:
        bottom_y = geom["bottom_y"]
        right_x  = geom["right_x"]

        best, vert_dist = best_field_line(
            bottom_y, right_x, clean,
            assigned_primary,
            above_pad=above_pad,
            below_pad=below_pad,
        )

        if best is None:
            print(f"[map] NO match  '{label}'  bottom_y={bottom_y:.0f}  right_x={right_x:.0f}")
            continue

        if vert_dist > max_vert_dist:
            print(f"[map] TOO FAR ({vert_dist:.0f}px)  '{label}'")
            continue

        print(f"[map] OK  '{label}'  vert={vert_dist:.1f}  line_y={_cy(best):.0f}  bot={bottom_y:.0f}")

        grouped = expand_multiline(best, clean, label_ys, y_step=below_pad)

        # Mark the grouped field lines as used so a multiline field stays logical.
        assigned_primary.update(id(line) for line in grouped)

        mappings.append({
            "label":       label,
            "label_pos":   ((geom["left_x"] + geom["raw_right_x"]) / 2.0, geom["center_y"]),
            "field_lines": grouped,
        })

    return mappings
