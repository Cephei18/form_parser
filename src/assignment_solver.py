"""Phase H — Global Assignment Engine.

Replaces the per-field greedy ``max(candidates)`` selection in
``field_anchor_engine`` with one coordinated, document-wide optimisation.

Today every field independently picks its highest-scoring candidate region.
Nothing prevents two fields from claiming the *same* physical region (a shared
underline, table cell, rectangle, or Textract value block). That produces
"stolen boxes", duplicate mappings and nearest-neighbour cascades.

This module treats answer-region selection as a **maximum-weight bipartite
assignment**: fields on one side, distinct physical regions on the other, each
region usable by at most one field. It is deterministic, requires no semantics,
no ML and no LLM. The existing candidate builders are untouched — they become
the evidence providers that populate the cost matrix.

The engine is gated behind ``FORM_PARSER_GLOBAL_ASSIGNMENT_ENABLED`` (default
OFF). When OFF the caller never invokes this module and behaviour is identical
to the legacy argmax path.
"""

from __future__ import annotations

import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

# A field-private candidate (whitespace estimate, inline leader, unresolved
# fallback) is unique to the field that produced it and can never be "stolen".
# Only the anchor types below reference a *shared* physical feature that two
# different labels could both try to claim, so only these participate in the
# exclusivity constraint by their own identity.
SHARED_SOURCE_ANCHORS = {"value_block", "table_cell"}
SHARED_GEOMETRY_ANCHORS = {
    "underline",
    "broken_underline",
    "dotted_underline",
    "empty_rectangle",
    "photo_region",
    "signature_region",
}

# Cost used for a (field, region) pair that has no candidate edge. Far larger
# than any real edge cost (real costs are in [-1, 0]) so the solver only ever
# lands on it when a field has no other option.
_NO_EDGE_COST = 1_000_000.0


def _bool_env(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def global_assignment_enabled() -> bool:
    """True when the Phase H global assignment engine is active."""
    return _bool_env("FORM_PARSER_GLOBAL_ASSIGNMENT_ENABLED", False)


@dataclass
class CandidateRegion:
    """A distinct physical region fields may compete for."""

    region_id: str
    bbox: tuple[float, float, float, float]
    region_type: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CandidateEdge:
    """A field's claim on a region, weighted by the builder's score."""

    field_id: str
    region_id: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


def _bbox_tuple(candidate: dict[str, Any]) -> tuple[float, float, float, float]:
    box = candidate.get("bbox") or {}
    try:
        return (
            round(float(box.get("x", 0.0)), 4),
            round(float(box.get("y", 0.0)), 4),
            round(float(box.get("width", 0.0)), 4),
            round(float(box.get("height", 0.0)), 4),
        )
    except (TypeError, ValueError):
        return (0.0, 0.0, 0.0, 0.0)


def region_identity(candidate: dict[str, Any], page: int, field_id: str) -> str:
    """Stable id for the physical region a candidate references.

    Two candidates from *different* fields collapse to the same id only when
    they point at the same shared feature (same source block, or same anchor
    type + geometry on the same page). Field-private anchor types are namespaced
    by ``field_id`` so they can never be matched against another field.
    """
    anchor = str(candidate.get("anchor_type") or "unknown")
    source_id = candidate.get("source_id")
    if anchor in SHARED_SOURCE_ANCHORS and source_id:
        return f"{anchor}|p{page}|src={source_id}"
    if anchor in SHARED_GEOMETRY_ANCHORS:
        x, y, w, h = _bbox_tuple(candidate)
        return f"{anchor}|p{page}|{x},{y},{w},{h}"
    # Field-private region: unique to this field, never contended.
    x, y, w, h = _bbox_tuple(candidate)
    return f"{anchor}|f={field_id}|p{page}|{x},{y},{w},{h}"


def build_region_graph(
    fields: list[dict[str, Any]],
) -> tuple[list[CandidateRegion], list[CandidateEdge], dict[str, int]]:
    """Deduplicate candidate regions across all fields and build the edge list.

    ``fields`` items are dicts with ``field_id``, ``page`` and ``candidates``
    (the raw candidate dicts produced by the existing builders).

    Returns ``(regions, edges, region_col_index)`` where ``region_col_index``
    maps each ``region_id`` to its column position in the cost matrix.
    """
    region_col_index: dict[str, int] = {}
    regions: list[CandidateRegion] = []
    edges: list[CandidateEdge] = []
    for f in fields:
        field_id = str(f["field_id"])
        page = int(f.get("page") or 1)
        for cand in f.get("candidates") or []:
            rid = region_identity(cand, page, field_id)
            if rid not in region_col_index:
                region_col_index[rid] = len(regions)
                regions.append(
                    CandidateRegion(
                        region_id=rid,
                        bbox=_bbox_tuple(cand),
                        region_type=str(cand.get("anchor_type") or "unknown"),
                        metadata={"page": page},
                    )
                )
            edges.append(
                CandidateEdge(
                    field_id=field_id,
                    region_id=rid,
                    score=float(cand.get("score") or 0.0),
                    metadata={"anchor_type": cand.get("anchor_type")},
                )
            )
    return regions, edges, region_col_index


def _hungarian_square(cost: list[list[float]]) -> list[int]:
    """Pure-python Kuhn–Munkres (O(n^3)) for a square cost matrix.

    Minimises total cost. Returns ``assign`` where ``assign[row] == col``.
    Used only when scipy is unavailable.
    """
    n = len(cost)
    if n == 0:
        return []
    INF = float("inf")
    u = [0.0] * (n + 1)
    v = [0.0] * (n + 1)
    p = [0] * (n + 1)
    way = [0] * (n + 1)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [INF] * (n + 1)
        used = [False] * (n + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = INF
            j1 = -1
            for j in range(1, n + 1):
                if not used[j]:
                    cur = cost[i0 - 1][j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            for j in range(n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    assign = [-1] * n
    for j in range(1, n + 1):
        if p[j] > 0:
            assign[p[j] - 1] = j - 1
    return assign


def _solve(cost: list[list[float]], n_rows: int, n_cols: int) -> tuple[dict[int, int], str]:
    """Solve the (rectangular) min-cost assignment.

    Returns ``(row_to_col, solver_used)``. Only real rows that landed on a real
    edge (cost below ``_NO_EDGE_COST``) appear in ``row_to_col``.
    """
    if n_rows == 0 or n_cols == 0:
        return {}, "none"
    try:
        import numpy as np
        from scipy.optimize import linear_sum_assignment

        matrix = np.asarray(cost, dtype=float)
        row_ind, col_ind = linear_sum_assignment(matrix)
        result: dict[int, int] = {}
        for r, c in zip(row_ind.tolist(), col_ind.tolist()):
            if r < n_rows and c < n_cols and cost[r][c] < _NO_EDGE_COST:
                result[r] = c
        return result, "scipy"
    except Exception:  # noqa: BLE001 - fall back to the pure-python solver
        pass

    size = max(n_rows, n_cols)
    square = [[0.0] * size for _ in range(size)]
    for r in range(size):
        for c in range(size):
            if r < n_rows and c < n_cols:
                square[r][c] = cost[r][c]
            else:
                square[r][c] = 0.0  # dummy row/col absorbs unused capacity
    assign = _hungarian_square(square)
    result = {}
    for r in range(n_rows):
        c = assign[r]
        if 0 <= c < n_cols and cost[r][c] < _NO_EDGE_COST:
            result[r] = c
    return result, "hungarian"


def _empty_diagnostics() -> dict[str, Any]:
    return {
        "enabled": True,
        "solver_used": "none",
        "fields": 0,
        "regions": 0,
        "duplicate_regions_before": 0,
        "duplicate_regions_after": 0,
        "stolen_box_rate": 0.0,
        "legacy_score": 0.0,
        "assignment_score": 0.0,
        "total_score_delta": 0.0,
        "unassigned_fields": [],
        "changed_fields": [],
        "duplicate_fixes": [],
        "score_improvements": [],
    }


def solve_global_assignment(
    fields: list[dict[str, Any]],
) -> tuple[dict[str, str], dict[str, Any]]:
    """Run global assignment over all fields.

    ``fields`` items: ``{field_id, page, candidates, legacy_selected}`` where
    ``legacy_selected`` is the candidate the greedy argmax would have chosen.

    Returns ``(overrides, diagnostics)``:
      * ``overrides`` maps ``field_id -> region_id`` only for fields whose
        globally-assigned region differs from the legacy choice. The caller
        re-selects the matching candidate by ``region_identity``.
      * ``diagnostics`` is the assignment + diff report.
    """
    fields = [f for f in fields if f.get("candidates")]
    n_rows = len(fields)
    if n_rows == 0:
        return {}, _empty_diagnostics()

    regions, _edges, region_col_index = build_region_graph(fields)
    n_cols = len(regions)

    # best[(row, col)] = (score, region_id) keeping the strongest claim when a
    # field has several candidates resolving to the same region.
    best: dict[tuple[int, int], tuple[float, str]] = {}
    for r, f in enumerate(fields):
        field_id = str(f["field_id"])
        page = int(f.get("page") or 1)
        for cand in f.get("candidates") or []:
            rid = region_identity(cand, page, field_id)
            col = region_col_index[rid]
            score = float(cand.get("score") or 0.0)
            prev = best.get((r, col))
            if prev is None or score > prev[0]:
                best[(r, col)] = (score, rid)

    cost = [[_NO_EDGE_COST] * n_cols for _ in range(n_rows)]
    for (r, col), (score, _rid) in best.items():
        cost[r][col] = -score

    row_to_col, solver_used = _solve(cost, n_rows, n_cols)

    # Legacy choice per field (for collision stats + diff).
    legacy_region_by_field: dict[str, str] = {}
    legacy_score_by_field: dict[str, float] = {}
    legacy_anchor_by_field: dict[str, str] = {}
    legacy_region_fields: dict[str, list[str]] = defaultdict(list)
    for f in fields:
        field_id = str(f["field_id"])
        page = int(f.get("page") or 1)
        legacy = f.get("legacy_selected") or {}
        rid = region_identity(legacy, page, field_id) if legacy else ""
        legacy_region_by_field[field_id] = rid
        legacy_score_by_field[field_id] = float(legacy.get("score") or 0.0)
        legacy_anchor_by_field[field_id] = str(legacy.get("anchor_type") or "")
        if rid:
            legacy_region_fields[rid].append(field_id)

    # New assignment per field.
    overrides: dict[str, str] = {}
    new_region_by_field: dict[str, str] = {}
    new_score_by_field: dict[str, float] = {}
    new_anchor_by_field: dict[str, str] = {}
    unassigned_fields: list[str] = []
    col_to_rid = {region_col_index[r.region_id]: r.region_id for r in regions}

    for r, f in enumerate(fields):
        field_id = str(f["field_id"])
        col = row_to_col.get(r)
        legacy_rid = legacy_region_by_field[field_id]
        if col is None:
            # No real edge available — keep the legacy choice (graceful).
            unassigned_fields.append(field_id)
            new_region_by_field[field_id] = legacy_rid
            new_score_by_field[field_id] = legacy_score_by_field[field_id]
            new_anchor_by_field[field_id] = legacy_anchor_by_field[field_id]
            continue
        rid = col_to_rid[col]
        score, _ = best[(r, col)]
        new_region_by_field[field_id] = rid
        new_score_by_field[field_id] = score
        new_anchor_by_field[field_id] = rid.split("|", 1)[0]
        if rid != legacy_rid:
            overrides[field_id] = rid

    # --- Diagnostics ---------------------------------------------------------
    collisions = {rid: fs for rid, fs in legacy_region_fields.items() if rid and len(fs) > 1}
    duplicate_regions_before = len(collisions)
    stolen_fields: set[str] = set()
    for fs in collisions.values():
        stolen_fields.update(fs)
    stolen_box_rate = round(len(stolen_fields) / n_rows, 4) if n_rows else 0.0

    new_region_fields: dict[str, list[str]] = defaultdict(list)
    for field_id, rid in new_region_by_field.items():
        if rid:
            new_region_fields[rid].append(field_id)
    duplicate_regions_after = sum(1 for fs in new_region_fields.values() if len(fs) > 1)

    duplicate_fixes: list[dict[str, Any]] = []
    for rid, fs in collisions.items():
        awarded = [fid for fid in fs if new_region_by_field.get(fid) == rid]
        duplicate_fixes.append(
            {
                "region_id": rid,
                "contended_by": fs,
                "awarded_to": awarded[0] if awarded else None,
                "reassigned": [fid for fid in fs if fid not in awarded],
            }
        )

    changed_fields: list[dict[str, Any]] = []
    score_improvements: list[dict[str, Any]] = []
    for field_id in overrides:
        changed_fields.append(
            {
                "field_id": field_id,
                "from_region": legacy_region_by_field[field_id],
                "from_anchor": legacy_anchor_by_field[field_id],
                "from_score": round(legacy_score_by_field[field_id], 4),
                "to_region": new_region_by_field[field_id],
                "to_anchor": new_anchor_by_field[field_id],
                "to_score": round(new_score_by_field[field_id], 4),
            }
        )
        score_improvements.append(
            {
                "field_id": field_id,
                "delta": round(new_score_by_field[field_id] - legacy_score_by_field[field_id], 4),
            }
        )

    legacy_total = round(sum(legacy_score_by_field.values()), 4)
    assignment_total = round(sum(new_score_by_field.values()), 4)

    diagnostics = {
        "enabled": True,
        "solver_used": solver_used,
        "fields": n_rows,
        "regions": n_cols,
        "duplicate_regions_before": duplicate_regions_before,
        "duplicate_regions_after": duplicate_regions_after,
        "stolen_box_rate": stolen_box_rate,
        "legacy_score": legacy_total,
        "assignment_score": assignment_total,
        "total_score_delta": round(assignment_total - legacy_total, 4),
        "unassigned_fields": unassigned_fields,
        "changed_fields": changed_fields,
        "duplicate_fixes": duplicate_fixes,
        "score_improvements": score_improvements,
    }
    return overrides, diagnostics
