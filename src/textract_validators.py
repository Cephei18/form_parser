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
