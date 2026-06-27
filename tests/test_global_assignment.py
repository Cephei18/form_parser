"""Phase H — global assignment engine validation.

Two layers of tests:

1. Solver-level (deterministic, no rasters): exercise the assignment logic
   directly with synthetic candidate sets — two-fields-one-underline,
   three-fields-two-boxes, duplicate table cell dedup, multipage scoping,
   independent columns, and the pure-python Hungarian fallback.

2. Integration (through ``build_anchored_mappings``): force a real contended
   region by pointing two Textract keys at the same VALUE block, and prove the
   flag is fully reversible (OFF == legacy; ON-uncontended == OFF).

It also emits validation artifacts to ``output/phase_h_assignment_validation/``.
"""
from __future__ import annotations

import itertools
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import logging

logging.disable(logging.CRITICAL)

from src import field_anchor_engine as fae
from src.assignment_solver import (
    _hungarian_square,
    _solve,
    build_region_graph,
    region_identity,
    solve_global_assignment,
)

FLAG = "FORM_PARSER_GLOBAL_ASSIGNMENT_ENABLED"


# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #
def _box(x, y, w, h):
    return {"x": x, "y": y, "width": w, "height": h}


def _cand(anchor, score, bbox, source_id=None):
    c = {"anchor_type": anchor, "score": score, "bbox": _box(*bbox), "reasons": []}
    if source_id is not None:
        c["source_id"] = source_id
    return c


def _field(field_id, page, cands):
    legacy = max(cands, key=lambda c: c["score"])
    return {"field_id": field_id, "page": page, "candidates": cands, "legacy_selected": legacy}


def _assigned_region(overrides, field, diag):
    """Region id a field ended up with: override if present, else its legacy."""
    if field["field_id"] in overrides:
        return overrides[field["field_id"]]
    return region_identity(field["legacy_selected"], field["page"], field["field_id"])


# --------------------------------------------------------------------------- #
# Solver-level scenarios
# --------------------------------------------------------------------------- #
def test_two_fields_one_underline():
    """Both fields' top candidate is the same underline. Only one may keep it."""
    u = (0.20, 0.50, 0.30, 0.02)
    f1 = _field("f1", 1, [_cand("underline", 0.80, u), _cand("adjacent_whitespace", 0.48, (0.20, 0.49, 0.30, 0.03))])
    f2 = _field("f2", 1, [_cand("underline", 0.70, u), _cand("adjacent_whitespace", 0.48, (0.20, 0.55, 0.30, 0.03))])

    overrides, diag = solve_global_assignment([f1, f2])

    assert diag["duplicate_regions_before"] == 1
    assert diag["duplicate_regions_after"] == 0
    # The stronger claim (f1, 0.80) keeps the underline; f2 is reassigned.
    assert "f1" not in overrides
    assert "f2" in overrides
    regions = {_assigned_region(overrides, f, diag) for f in (f1, f2)}
    assert len(regions) == 2  # no shared region
    assert diag["stolen_box_rate"] == 1.0


def test_three_fields_two_boxes():
    """Globally optimal beats greedy-local: the field that can cheaply move does."""
    r1 = (0.20, 0.30, 0.30, 0.05)
    r2 = (0.20, 0.40, 0.30, 0.05)
    f1 = _field("f1", 1, [_cand("empty_rectangle", 0.85, r1), _cand("adjacent_whitespace", 0.48, (0.2, 0.31, 0.3, 0.03))])
    f2 = _field("f2", 1, [
        _cand("empty_rectangle", 0.80, r1),
        _cand("empty_rectangle", 0.75, r2),
        _cand("adjacent_whitespace", 0.48, (0.2, 0.45, 0.3, 0.03)),
    ])
    f3 = _field("f3", 1, [_cand("empty_rectangle", 0.82, r2), _cand("adjacent_whitespace", 0.48, (0.2, 0.41, 0.3, 0.03))])

    overrides, diag = solve_global_assignment([f1, f2, f3])

    assert diag["duplicate_regions_after"] == 0
    # f1 keeps r1 (0.85), f3 keeps r2 (0.82); f2 yields to whitespace.
    assert "f1" not in overrides
    assert "f3" not in overrides
    assert "f2" in overrides
    regions = {_assigned_region(overrides, f, diag) for f in (f1, f2, f3)}
    assert len(regions) == 3


def test_duplicate_table_cell_deduplicated():
    """Two fields claim the same table cell (same source_id) -> one region."""
    cell = (0.5, 0.3, 0.2, 0.04)
    f1 = _field("f1", 1, [_cand("table_cell", 0.86, cell, source_id="cellA"), _cand("adjacent_whitespace", 0.48, (0.2, 0.3, 0.2, 0.03))])
    f2 = _field("f2", 1, [_cand("table_cell", 0.84, cell, source_id="cellA"), _cand("adjacent_whitespace", 0.48, (0.2, 0.5, 0.2, 0.03))])

    regions, edges, col_index = build_region_graph([f1, f2])
    table_regions = [r for r in regions if r.region_type == "table_cell"]
    assert len(table_regions) == 1  # both candidates collapsed to one region

    overrides, diag = solve_global_assignment([f1, f2])
    assert diag["duplicate_regions_before"] == 1
    assert diag["duplicate_regions_after"] == 0


def test_multipage_regions_are_page_scoped():
    """Identical underline geometry on different pages is NOT the same region."""
    u = (0.20, 0.50, 0.30, 0.02)
    f1 = _field("f1", 1, [_cand("underline", 0.80, u)])
    f2 = _field("f2", 2, [_cand("underline", 0.80, u)])

    assert region_identity(f1["candidates"][0], 1, "f1") != region_identity(f2["candidates"][0], 2, "f2")
    overrides, diag = solve_global_assignment([f1, f2])
    assert diag["duplicate_regions_before"] == 0
    assert overrides == {}  # neither field steals across the page boundary
    assert diag["unassigned_fields"] == []


def test_independent_columns_no_false_steal():
    """Fields whose candidates are distinct regions are left untouched."""
    f1 = _field("f1", 1, [_cand("underline", 0.80, (0.10, 0.50, 0.30, 0.02))])
    f2 = _field("f2", 1, [_cand("underline", 0.80, (0.55, 0.50, 0.30, 0.02))])
    overrides, diag = solve_global_assignment([f1, f2])
    assert overrides == {}
    assert diag["duplicate_regions_before"] == 0
    assert diag["duplicate_regions_after"] == 0


def test_empty_input():
    overrides, diag = solve_global_assignment([])
    assert overrides == {}
    assert diag["fields"] == 0
    assert diag["enabled"] is True


def test_diagnostics_shape():
    f1 = _field("f1", 1, [_cand("underline", 0.80, (0.2, 0.5, 0.3, 0.02)), _cand("adjacent_whitespace", 0.48, (0.2, 0.49, 0.3, 0.03))])
    f2 = _field("f2", 1, [_cand("underline", 0.70, (0.2, 0.5, 0.3, 0.02)), _cand("adjacent_whitespace", 0.48, (0.2, 0.55, 0.3, 0.03))])
    _, diag = solve_global_assignment([f1, f2])
    for key in (
        "enabled", "solver_used", "fields", "regions", "duplicate_regions_before",
        "duplicate_regions_after", "stolen_box_rate", "legacy_score", "assignment_score",
        "total_score_delta", "unassigned_fields", "changed_fields", "duplicate_fixes",
        "score_improvements",
    ):
        assert key in diag
    assert diag["solver_used"] in {"scipy", "hungarian"}
    assert len(diag["duplicate_fixes"]) == 1
    assert diag["duplicate_fixes"][0]["awarded_to"] == "f1"


# --------------------------------------------------------------------------- #
# Pure-python Hungarian fallback
# --------------------------------------------------------------------------- #
def _brute_force_min(cost):
    n = len(cost)
    best = None
    for perm in itertools.permutations(range(n)):
        total = sum(cost[i][perm[i]] for i in range(n))
        if best is None or total < best:
            best = total
    return best


def test_hungarian_matches_brute_force():
    matrices = [
        [[-0.80, -0.48], [-0.70, -0.48]],
        [[-0.85, -0.48, 1e6], [-0.80, -0.75, -0.48], [1e6, -0.82, -0.48]],
        [[4.0, 1.0, 3.0], [2.0, 0.0, 5.0], [3.0, 2.0, 2.0]],
    ]
    for cost in matrices:
        assign = _hungarian_square([row[:] for row in cost])
        total = sum(cost[i][assign[i]] for i in range(len(cost)))
        assert abs(total - _brute_force_min(cost)) < 1e-6
        assert sorted(assign) == list(range(len(cost)))  # valid permutation


def test_solve_filters_no_edge_cells():
    # Row 0 only really connects to col 0; row 1 only to col 1.
    cost = [[-0.8, 1e6], [1e6, -0.7]]
    row_to_col, solver = _solve(cost, 2, 2)
    assert row_to_col == {0: 0, 1: 1}
    assert solver in {"scipy", "hungarian"}


# --------------------------------------------------------------------------- #
# Integration through build_anchored_mappings
# --------------------------------------------------------------------------- #
def _blank_png(tmp_path) -> Path:
    img = np.full((1000, 800, 3), 255, dtype=np.uint8)
    path = tmp_path / "page1.png"
    cv2.imwrite(str(path), img)
    return path


def _word(bid, text, x, y, w=0.12, h=0.02, page=1):
    return {
        "Id": bid,
        "BlockType": "WORD",
        "Text": text,
        "Page": page,
        "Geometry": {"BoundingBox": {"Left": x, "Top": y, "Width": w, "Height": h}},
    }


def _response(value_blocks, key_words):
    child_ids = [b["Id"] for b in key_words] + [b["Id"] for b in value_blocks]
    page = {
        "Id": "PAGE1",
        "BlockType": "PAGE",
        "Page": 1,
        "Geometry": {"BoundingBox": {"Left": 0, "Top": 0, "Width": 1, "Height": 1}},
        "Relationships": [{"Type": "CHILD", "Ids": child_ids}],
    }
    return {"Blocks": [page, *key_words, *value_blocks]}


def _run(raw_response, parsed, image_path, enabled):
    if enabled:
        os.environ[FLAG] = "true"
    else:
        os.environ.pop(FLAG, None)
    # Isolate the global-assignment feature under test: the (default-ON) field
    # hygiene pass also collapses exact-overlap duplicates, which would mask the
    # contention these tests deliberately set up. Phase H does more than drop a
    # duplicate (it reassigns the loser to a different region), so we measure it
    # with hygiene neutralised.
    os.environ["FORM_PARSER_FIELD_HYGIENE_ENABLED"] = "false"
    try:
        return fae.build_anchored_mappings(raw_response, parsed, image_path)
    finally:
        os.environ.pop(FLAG, None)
        os.environ.pop("FORM_PARSER_FIELD_HYGIENE_ENABLED", None)


def _mapping_signature(result):
    sig = []
    for m in result["mappings"]:
        sig.append((m.get("label"), m.get("anchoring", {}).get("anchor_type"), tuple(sorted(m["bbox"].items()))))
    return sorted(sig)


def test_integration_contended_value_block(tmp_path):
    """Two keys -> one shared VALUE block. OFF duplicates it; ON resolves it."""
    image_path = _blank_png(tmp_path)
    k1 = _word("K1", "First Name", 0.10, 0.20)
    k2 = _word("K2", "Last Name", 0.10, 0.32)
    v = _word("V", "", 0.55, 0.25, w=0.25, h=0.02)
    raw = _response([v], [k1, k2])
    parsed = {
        "field_items": [
            {"key": "First Name", "key_block_id": "K1", "value_block_ids": ["V"], "value": ""},
            {"key": "Last Name", "key_block_id": "K2", "value_block_ids": ["V"], "value": ""},
        ],
        "tables": [],
        "checkboxes": [],
    }

    off = _run(raw, parsed, image_path, enabled=False)
    on = _run(raw, parsed, image_path, enabled=True)

    # OFF: both fields steal the same value block (duplicate mapping).
    off_value_blocks = [m for m in off["mappings"] if m["anchoring"]["anchor_type"] == "value_block"]
    assert len(off_value_blocks) == 2
    assert off["diagnostics"]["global_assignment"] == {"enabled": False}

    # ON: exactly one field keeps the value block; the other is reassigned.
    on_value_blocks = [m for m in on["mappings"] if m["anchoring"]["anchor_type"] == "value_block"]
    assert len(on_value_blocks) == 1
    diag = on["diagnostics"]["global_assignment"]
    assert diag["enabled"] is True
    assert diag["duplicate_regions_before"] >= 1
    assert diag["duplicate_regions_after"] == 0
    # No two mappings share the value block's geometry.
    boxes = [tuple(sorted(m["bbox"].items())) for m in on["mappings"]]
    assert len(boxes) == len(set(boxes))


def test_integration_backward_compatible_when_uncontended(tmp_path):
    """Distinct value blocks -> no contention -> ON output identical to OFF."""
    image_path = _blank_png(tmp_path)
    k1 = _word("K1", "First Name", 0.10, 0.20)
    k2 = _word("K2", "Last Name", 0.10, 0.32)
    v1 = _word("V1", "", 0.55, 0.20, w=0.25, h=0.02)
    v2 = _word("V2", "", 0.55, 0.32, w=0.25, h=0.02)
    raw = _response([v1, v2], [k1, k2])
    parsed = {
        "field_items": [
            {"key": "First Name", "key_block_id": "K1", "value_block_ids": ["V1"], "value": ""},
            {"key": "Last Name", "key_block_id": "K2", "value_block_ids": ["V2"], "value": ""},
        ],
        "tables": [],
        "checkboxes": [],
    }

    off = _run(raw, parsed, image_path, enabled=False)
    on = _run(raw, parsed, image_path, enabled=True)

    assert _mapping_signature(off) == _mapping_signature(on)
    assert on["diagnostics"]["global_assignment"]["duplicate_regions_before"] == 0
    assert on["diagnostics"]["global_assignment"]["changed_fields"] == []


def test_emit_validation_artifacts(tmp_path):
    """Generate before/after/diff/metrics artifacts for the contended case."""
    image_path = _blank_png(tmp_path)
    k1 = _word("K1", "First Name", 0.10, 0.20)
    k2 = _word("K2", "Last Name", 0.10, 0.32)
    v = _word("V", "", 0.55, 0.25, w=0.25, h=0.02)
    raw = _response([v], [k1, k2])
    parsed = {
        "field_items": [
            {"key": "First Name", "key_block_id": "K1", "value_block_ids": ["V"], "value": ""},
            {"key": "Last Name", "key_block_id": "K2", "value_block_ids": ["V"], "value": ""},
        ],
        "tables": [],
        "checkboxes": [],
    }

    off = _run(raw, parsed, image_path, enabled=False)
    on = _run(raw, parsed, image_path, enabled=True)
    diag = on["diagnostics"]["global_assignment"]

    out_dir = REPO_ROOT / "output" / "phase_h_assignment_validation"
    out_dir.mkdir(parents=True, exist_ok=True)

    def _dump(name, payload):
        (out_dir / name).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    _dump("before_assignment.json", {"mappings": off["mappings"]})
    _dump("after_assignment.json", {"mappings": on["mappings"]})
    _dump("assignment_diff.json", {
        "changed_fields": diag["changed_fields"],
        "duplicate_fixes": diag["duplicate_fixes"],
        "score_improvements": diag["score_improvements"],
    })
    _dump("validation_metrics.json", {
        "fields": diag["fields"],
        "regions": diag["regions"],
        "duplicate_regions_before": diag["duplicate_regions_before"],
        "duplicate_regions_after": diag["duplicate_regions_after"],
        "stolen_box_rate": diag["stolen_box_rate"],
        "solver_used": diag["solver_used"],
        "legacy_score": diag["legacy_score"],
        "assignment_score": diag["assignment_score"],
        "total_score_delta": diag["total_score_delta"],
        "unassigned_fields": diag["unassigned_fields"],
    })

    for name in ("before_assignment.json", "after_assignment.json", "assignment_diff.json", "validation_metrics.json"):
        assert (out_dir / name).exists()
    assert diag["duplicate_regions_after"] == 0
