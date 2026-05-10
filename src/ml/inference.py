"""Deterministic structural inference entrypoints.

This module intentionally avoids heavyweight semantic models. It provides a
thin façade over the page-structure engine so callers can ask for page-level
layout inference from a single import path.
"""

from __future__ import annotations

from typing import Any, Dict, List


def infer_page_structure(
    ocr_results: List[Dict[str, Any]],
    field_lines: List[Dict[str, Any]] | None = None,
    semantic_regions: List[Dict[str, Any]] | None = None,
) -> dict[str, Any]:
    from src.global_layout import infer_page_structure as _infer_page_structure

    return _infer_page_structure(ocr_results or [], field_lines or [], semantic_regions or [])


def infer_page_mappings(
    ocr_results: List[Dict[str, Any]],
    field_lines: List[Dict[str, Any]],
    semantic_regions: List[Dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    from src.mapping import map_labels_to_fields

    layout_structure = infer_page_structure(ocr_results, field_lines, semantic_regions)
    return map_labels_to_fields(
        ocr_results or [],
        field_lines or [],
        layout_structure=layout_structure,
        semantic_regions=semantic_regions or [],
    )


def ml_pipeline(ocr_results: List[Dict[str, Any]], field_lines: List[Dict[str, Any]] | None = None):
    """Backwards-compatible alias for the structural page inference path."""

    if field_lines is None:
        return infer_page_structure(ocr_results, [], [])
    return infer_page_mappings(ocr_results, field_lines, [])
