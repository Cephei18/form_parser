"""
Advanced validation and normalization tools for Textract parsed output.
Provides modular functions for table normalization and checkbox validation.
"""
from __future__ import annotations

from typing import Any


def normalize_table_for_export(table: dict[str, Any]) -> dict[str, Any]:
    """
    Normalize a parsed table for export/display.
    - Ensure all cells have consistent type (None or str)
    - Fill sparse tables with None values
    - Calculate headers if present
    """
    rows = table.get("rows", [])
    cells = table.get("cells", [])
    dims = table.get("dimensions", {})

    max_row = dims.get("rows", len(rows))
    max_col = dims.get("columns", max(len(r) for r in rows) if rows else 0)

    # Rebuild normalized grid
    normalized_grid: list[list[str | None]] = [[None for _ in range(max_col)] for _ in range(max_row)]

    for cell in cells:
        row = cell.get("row_index", 0) - 1
        col = cell.get("column_index", 0) - 1
        text = cell.get("text", "").strip() if cell.get("text") else None
        row_span = cell.get("row_span", 1)
        col_span = cell.get("column_span", 1)

        if 0 <= row < max_row and 0 <= col < max_col:
            normalized_grid[row][col] = text

    # Detect header row (first row if all cells are non-empty and short)
    header_row_idx = None
    if normalized_grid:
        first_row = normalized_grid[0]
        if all(c is not None for c in first_row) and all(isinstance(c, str) and len(c) < 50 for c in first_row):
            header_row_idx = 0

    return {
        "table_id": table.get("table_block_id"),
        "page": table.get("page"),
        "confidence": table.get("confidence"),
        "geometry": table.get("geometry"),
        "dimensions": {"rows": max_row, "columns": max_col},
        "header_row_index": header_row_idx,
        "rows": normalized_grid,
        "cells": cells,
        "cell_count": len(cells),
        "empty_cells": sum(1 for c in cells if not c.get("text") or not c.get("text").strip()),
    }


def validate_checkbox_ownership(checkbox: dict[str, Any]) -> dict[str, Any]:
    """
    Validate checkbox ownership information.
    Returns validation report with confidence and issues.
    """
    ownership = checkbox.get("ownership", {})
    selection_status = checkbox.get("selection_status")
    confidence = checkbox.get("confidence")

    validation = {
        "checkbox_id": checkbox.get("selection_element_id"),
        "is_selected": checkbox.get("is_selected"),
        "selection_status": selection_status,
        "confidence": confidence,
        "ownership_type": ownership.get("ownership_type"),
        "has_owner": ownership.get("key_block_id") is not None,
        "owner_text": ownership.get("key_text"),
        "confidence_factors": ownership.get("confidence_factors", []),
        "issues": [],
    }

    # Validation checks
    if ownership.get("ownership_type") == "unassociated":
        validation["issues"].append("checkbox_unassociated")

    if not ownership.get("key_text"):
        validation["issues"].append("owner_text_empty")

    if confidence is not None and confidence < 0.7:
        validation["issues"].append("low_confidence")

    # Ownership confidence rating
    confidence_factors = ownership.get("confidence_factors", [])
    if "direct_value_key_relationship" in confidence_factors:
        validation["ownership_confidence"] = "high"
    elif "table_cell_context" in confidence_factors or "parent_key_value_set" in confidence_factors:
        validation["ownership_confidence"] = "medium"
    else:
        validation["ownership_confidence"] = "low"

    return validation


def summarize_table_extraction(tables: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Generate summary statistics for all extracted tables.
    """
    if not tables:
        return {
            "table_count": 0,
            "total_cells": 0,
            "avg_rows": 0,
            "avg_cols": 0,
            "avg_confidence": None,
        }

    total_cells = sum(len(t.get("cells", [])) for t in tables)
    total_rows = sum(t.get("dimensions", {}).get("rows", 0) for t in tables)
    total_cols = sum(t.get("dimensions", {}).get("columns", 0) for t in tables)

    confidences = [t.get("confidence") for t in tables if t.get("confidence") is not None]
    avg_conf = sum(confidences) / len(confidences) if confidences else None

    return {
        "table_count": len(tables),
        "total_cells": total_cells,
        "total_rows": total_rows,
        "total_cols": total_cols,
        "avg_rows": total_rows / len(tables) if tables else 0,
        "avg_cols": total_cols / len(tables) if tables else 0,
        "avg_confidence": avg_conf,
        "pages_with_tables": len(set(t.get("page") for t in tables if t.get("page") is not None)),
    }


def summarize_checkbox_extraction(checkboxes: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Generate summary statistics for all extracted checkboxes.
    """
    if not checkboxes:
        return {
            "checkbox_count": 0,
            "selected_count": 0,
            "unselected_count": 0,
            "avg_confidence": None,
            "ownership_breakdown": {},
        }

    selected = sum(1 for c in checkboxes if c.get("is_selected"))
    unselected = len(checkboxes) - selected

    ownership_types: dict[str, int] = {}
    for c in checkboxes:
        ot = c.get("ownership", {}).get("ownership_type", "unknown")
        ownership_types[ot] = ownership_types.get(ot, 0) + 1

    confidences = [c.get("confidence") for c in checkboxes if c.get("confidence") is not None]
    avg_conf = sum(confidences) / len(confidences) if confidences else None

    return {
        "checkbox_count": len(checkboxes),
        "selected_count": selected,
        "unselected_count": unselected,
        "selection_rate": selected / len(checkboxes) if checkboxes else 0,
        "avg_confidence": avg_conf,
        "ownership_breakdown": ownership_types,
        "pages_with_checkboxes": len(set(c.get("page") for c in checkboxes if c.get("page") is not None)),
    }


def compare_field_and_checkbox_consistency(fields: dict[str, Any], checkboxes: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Check consistency between extracted fields and checkboxes.
    Looks for checkboxes with field-like labels and vice versa.
    """
    field_keys = set(fields.keys())
    checkbox_owners = []

    for cb in checkboxes:
        owner = cb.get("ownership", {})
        owner_text = owner.get("key_text")
        if owner_text:
            checkbox_owners.append(owner_text)

    checkbox_owner_set = set(checkbox_owners)

    # Find overlaps
    overlap = field_keys & checkbox_owner_set
    fields_without_checkboxes = field_keys - checkbox_owner_set
    checkboxes_without_fields = checkbox_owner_set - field_keys

    return {
        "overlapping_keys": list(overlap),
        "fields_only": list(fields_without_checkboxes),
        "checkboxes_only": list(checkboxes_without_fields),
        "total_field_keys": len(field_keys),
        "total_checkbox_owners": len(checkbox_owner_set),
        "overlap_count": len(overlap),
        "consistency_score": len(overlap) / max(len(field_keys), len(checkbox_owner_set)) if max(len(field_keys), len(checkbox_owner_set)) > 0 else 0,
    }


def _is_nan(value: Any) -> bool:
    return isinstance(value, float) and value != value


def validate_mappings_sanity(mappings: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Pre-deployment sanity guard over the final answer-region mappings.
    Flags malformed geometry (NaN / out-of-range / zero-size) and surfaces
    whether the document yielded any fillable fields at all. Never raises.
    """
    issues: list[str] = []
    fillable = 0
    photo = 0
    out_of_range = 0
    zero_size = 0
    nan_boxes = 0
    low_confidence = 0

    for index, mapping in enumerate(mappings or []):
        if not isinstance(mapping, dict):
            issues.append(f"mapping_{index}_not_a_dict")
            continue

        field_type = mapping.get("field_type")
        if field_type == "photo":
            photo += 1
        else:
            fillable += 1

        bbox = mapping.get("bbox")
        if not isinstance(bbox, dict):
            nan_boxes += 1
            issues.append(f"mapping_{index}_missing_bbox")
            continue
        try:
            x = float(bbox["x"]); y = float(bbox["y"])
            w = float(bbox["width"]); h = float(bbox["height"])
        except (KeyError, TypeError, ValueError):
            nan_boxes += 1
            issues.append(f"mapping_{index}_non_numeric_bbox")
            continue
        if any(_is_nan(v) for v in (x, y, w, h)):
            nan_boxes += 1
            issues.append(f"mapping_{index}_nan_bbox")
            continue
        if w <= 0 or h <= 0:
            zero_size += 1
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0) or x + w > 1.001 or y + h > 1.001:
            out_of_range += 1

        confidence = mapping.get("confidence")
        if isinstance(confidence, (int, float)) and not _is_nan(float(confidence)) and confidence < 0.42:
            low_confidence += 1

    if fillable == 0:
        issues.append("no_fillable_fields_detected")

    return {
        "mapping_count": len(mappings or []),
        "fillable_count": fillable,
        "photo_count": photo,
        "out_of_range_boxes": out_of_range,
        "zero_size_boxes": zero_size,
        "nan_boxes": nan_boxes,
        "low_confidence_fields": low_confidence,
        "issues": issues,
        "passed": nan_boxes == 0 and zero_size == 0 and fillable > 0,
    }


def build_validation_report(parsed: dict[str, Any], mappings: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Single non-fatal entry point that aggregates every validator into one
    report for the pipeline diagnostics. Each section is isolated so a failure
    in one validator can never break document rendering.
    """
    report: dict[str, Any] = {}
    safe_parsed = parsed if isinstance(parsed, dict) else {}

    def _safe(name: str, fn: Any) -> None:
        try:
            report[name] = fn()
        except Exception as exc:  # pragma: no cover - defensive guard
            report[name] = {"error": f"{type(exc).__name__}: {exc}"}

    _safe("tables", lambda: summarize_table_extraction(safe_parsed.get("tables", []) or []))
    _safe("checkboxes", lambda: summarize_checkbox_extraction(safe_parsed.get("checkboxes", []) or []))
    _safe(
        "field_checkbox_consistency",
        lambda: compare_field_and_checkbox_consistency(
            safe_parsed.get("fields", {}) or {}, safe_parsed.get("checkboxes", []) or []
        ),
    )
    _safe("mapping_sanity", lambda: validate_mappings_sanity(mappings or []))

    sanity = report.get("mapping_sanity")
    report["passed"] = bool(isinstance(sanity, dict) and sanity.get("passed"))
    return report


if __name__ == "__main__":
    import json
    import sys

    if len(sys.argv) < 2:
        print("Usage: python textract_validators.py <parsed_json>")
        sys.exit(1)

    with open(sys.argv[1], "r", encoding="utf-8") as f:
        parsed = json.load(f)

    # Run all validators
    table_summary = summarize_table_extraction(parsed.get("tables", []))
    checkbox_summary = summarize_checkbox_extraction(parsed.get("checkboxes", []))
    consistency = compare_field_and_checkbox_consistency(parsed.get("fields", {}), parsed.get("checkboxes", []))

    report = {
        "tables": table_summary,
        "checkboxes": checkbox_summary,
        "field_checkbox_consistency": consistency,
    }

    print(json.dumps(report, indent=2, ensure_ascii=False))
