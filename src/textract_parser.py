from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

logger = logging.getLogger("form_parser.textract_parser")


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _relationship_ids(block: dict[str, Any], relationship_type: str | None = None) -> list[str]:
    ids: list[str] = []
    for relationship in block.get("Relationships", []) or []:
        if relationship_type is not None and relationship.get("Type") != relationship_type:
            continue
        ids.extend(relationship.get("Ids", []) or [])
    return ids


def _block_text(block_id: str, blocks_by_id: dict[str, dict[str, Any]], cache: dict[str, str]) -> str:
    if block_id in cache:
        return cache[block_id]

    block = blocks_by_id.get(block_id)
    if not block:
        cache[block_id] = ""
        return ""

    block_type = block.get("BlockType")
    if block_type == "WORD":
        text = str(block.get("Text") or "")
    elif block_type == "SELECTION_ELEMENT":
        text = "[X]" if block.get("SelectionStatus") == "SELECTED" else "[ ]"
    else:
        child_texts: list[str] = []
        child_types: set[str] = set()
        for child_id in _relationship_ids(block, "CHILD"):
            child_block = blocks_by_id.get(child_id, {})
            child_type = child_block.get("BlockType")
            if child_type:
                child_types.add(child_type)
            child_text = _block_text(child_id, blocks_by_id, cache)
            if child_text:
                child_texts.append(child_text)

        # If children contain LINE elements, preserve line breaks for multiline fields
        if "LINE" in child_types:
            text = "\n".join(child_texts).strip()
        else:
            text = " ".join(child_texts).strip()

    cache[block_id] = text
    return text


def _geometry_summary(block: dict[str, Any]) -> dict[str, Any] | None:
    geometry = block.get("Geometry")
    if not geometry:
        return None

    # Normalize geometry to ensure consistent numeric types and ordering
    bbox = geometry.get("BoundingBox") or {}
    polygon = geometry.get("Polygon") or []

    def _norm_bbox(b: dict[str, Any]) -> dict[str, float]:
        return {
            "Width": float(b.get("Width") or 0.0),
            "Height": float(b.get("Height") or 0.0),
            "Left": float(b.get("Left") or 0.0),
            "Top": float(b.get("Top") or 0.0),
        }

    def _norm_polygon(poly: list[dict[str, Any]]) -> list[dict[str, float]]:
        points: list[dict[str, float]] = []
        for p in poly:
            points.append({"X": float(p.get("X") or 0.0), "Y": float(p.get("Y") or 0.0)})
        return points

    return {
        "bounding_box": _norm_bbox(bbox),
        "polygon": _norm_polygon(polygon),
    }


def _append_field(fields: dict[str, Any], key: str, value: Any) -> None:
    if key in fields:
        existing = fields[key]
        if isinstance(existing, list):
            existing.append(value)
        else:
            fields[key] = [existing, value]
        return

    fields[key] = value


def _build_parent_map(blocks: list[dict[str, Any]]) -> dict[str, list[str]]:
    parent_map: dict[str, list[str]] = defaultdict(list)
    for block in blocks:
        block_id = block.get("Id")
        if not block_id:
            continue
        for child_id in _relationship_ids(block, "CHILD"):
            parent_map[child_id].append(block_id)
    return parent_map


def _extract_key_value_items(
    blocks_by_id: dict[str, dict[str, Any]],
    text_cache: dict[str, str],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, str]]:
    fields: dict[str, Any] = {}
    field_items: list[dict[str, Any]] = []
    value_block_to_key_block_id: dict[str, str] = {}

    for block in blocks_by_id.values():
        if block.get("BlockType") != "KEY_VALUE_SET":
            continue
        entity_types = block.get("EntityTypes", []) or []
        if "KEY" not in entity_types:
            continue

        key_text = _block_text(block.get("Id", ""), blocks_by_id, text_cache)
        value_ids = _relationship_ids(block, "VALUE")
        value_texts: list[str] = []

        for value_id in value_ids:
            value_block_to_key_block_id[value_id] = block.get("Id", "")
            value_texts.append(_block_text(value_id, blocks_by_id, text_cache))

        # Preserve line breaks for multiline VALUE blocks
        non_empty = [text for text in value_texts if text]
        if any("\n" in t for t in non_empty):
            value_text = "\n".join(non_empty).strip()
        else:
            value_text = " ".join(non_empty).strip()
        field_items.append(
            {
                "key_block_id": block.get("Id"),
                "key": key_text,
                "value": value_text,
                "confidence": block.get("Confidence"),
                "page": block.get("Page"),
                "geometry": _geometry_summary(block),
                "value_block_ids": value_ids,
            }
        )
        if key_text:
            _append_field(fields, key_text, value_text)

    return fields, field_items, value_block_to_key_block_id


def _extract_tables(blocks_by_id: dict[str, dict[str, Any]], text_cache: dict[str, str]) -> list[dict[str, Any]]:
    tables: list[dict[str, Any]] = []

    for block in blocks_by_id.values():
        if block.get("BlockType") != "TABLE":
            continue

        cells: list[dict[str, Any]] = []
        max_row = 0
        max_col = 0

        for cell_id in _relationship_ids(block, "CHILD"):
            cell_block = blocks_by_id.get(cell_id)
            if not cell_block or cell_block.get("BlockType") != "CELL":
                continue

            row_index = _safe_int(cell_block.get("RowIndex"), 0)
            column_index = _safe_int(cell_block.get("ColumnIndex"), 0)
            max_row = max(max_row, row_index)
            max_col = max(max_col, column_index)
            cells.append(
                {
                    "cell_id": cell_id,
                    "row_index": row_index,
                    "column_index": column_index,
                    "row_span": _safe_int(cell_block.get("RowSpan"), 1),
                    "column_span": _safe_int(cell_block.get("ColumnSpan"), 1),
                    "text": _block_text(cell_id, blocks_by_id, text_cache),
                    "confidence": cell_block.get("Confidence"),
                    "geometry": _geometry_summary(cell_block),
                }
            )

        rows = [["" for _ in range(max_col)] for _ in range(max_row)] if max_row and max_col else []
        for cell in cells:
            row_index = cell["row_index"]
            column_index = cell["column_index"]
            if row_index <= 0 or column_index <= 0:
                continue
            if row_index - 1 < len(rows) and column_index - 1 < len(rows[row_index - 1]):
                rows[row_index - 1][column_index - 1] = cell["text"]

        # Normalize rows: trim whitespace and replace empty strings with None for clarity
        normalized_rows: list[list[str | None]] = []
        for r in rows:
            normalized_rows.append([c.strip() if isinstance(c, str) and c.strip() else None for c in r])

        tables.append(
            {
                "table_block_id": block.get("Id"),
                "page": block.get("Page"),
                "confidence": block.get("Confidence"),
                "geometry": _geometry_summary(block),
                "dimensions": {"rows": max_row, "columns": max_col},
                "rows": normalized_rows,
                "cells": sorted(cells, key=lambda cell: (cell["row_index"], cell["column_index"])),
            }
        )

    return tables


def _find_owning_key_for_checkbox(
    checkbox_block_id: str,
    parent_ids: list[str],
    blocks_by_id: dict[str, dict[str, Any]],
    value_block_to_key_block_id: dict[str, str],
    text_cache: dict[str, str],
) -> dict[str, Any]:
    """
    Advanced checkbox ownership detection.
    Strategy: Prefer VALUE→KEY relationships, then look for cell context, then parent KEY_VALUE_SET.
    """
    ownership: dict[str, Any] = {
        "key_block_id": None,
        "key_text": None,
        "ownership_type": None,
        "confidence_factors": [],
    }

    # Strategy 1: Direct VALUE→KEY mapping (highest confidence)
    for parent_id in parent_ids:
        if parent_id in value_block_to_key_block_id:
            key_block_id = value_block_to_key_block_id[parent_id]
            ownership["key_block_id"] = key_block_id
            ownership["key_text"] = _block_text(key_block_id, blocks_by_id, text_cache)
            ownership["ownership_type"] = "value_to_key_mapping"
            ownership["confidence_factors"].append("direct_value_key_relationship")
            return ownership

    # Strategy 2: Parent is a CELL in a table (look at cell context)
    for parent_id in parent_ids:
        parent_block = blocks_by_id.get(parent_id)
        if parent_block and parent_block.get("BlockType") == "CELL":
            cell_row = parent_block.get("RowIndex")
            cell_col = parent_block.get("ColumnIndex")
            cell_text = _block_text(parent_id, blocks_by_id, text_cache)
            ownership["key_block_id"] = parent_id
            ownership["key_text"] = f"[Table Cell R{cell_row}C{cell_col}] {cell_text}" if cell_text else f"[Table Cell R{cell_row}C{cell_col}]"
            ownership["ownership_type"] = "table_cell"
            ownership["confidence_factors"].append("table_cell_context")
            return ownership

    # Strategy 3: Parent is a KEY_VALUE_SET KEY (look for nearest sibling key)
    for parent_id in parent_ids:
        parent_block = blocks_by_id.get(parent_id)
        if parent_block and parent_block.get("BlockType") == "KEY_VALUE_SET":
            entity_types = parent_block.get("EntityTypes", []) or []
            if "KEY" in entity_types:
                key_text = _block_text(parent_id, blocks_by_id, text_cache)
                ownership["key_block_id"] = parent_id
                ownership["key_text"] = key_text
                ownership["ownership_type"] = "parent_key_value_set"
                ownership["confidence_factors"].append("parent_is_key_value_set")
                return ownership

    # Strategy 4: Fallback - unassociated checkbox
    ownership["ownership_type"] = "unassociated"
    ownership["confidence_factors"].append("no_ownership_clues")
    return ownership


def _extract_checkboxes(
    blocks_by_id: dict[str, dict[str, Any]],
    parent_map: dict[str, list[str]],
    value_block_to_key_block_id: dict[str, str],
    text_cache: dict[str, str],
) -> list[dict[str, Any]]:
    checkboxes: list[dict[str, Any]] = []

    for block in blocks_by_id.values():
        if block.get("BlockType") != "SELECTION_ELEMENT":
            continue

        block_id = block.get("Id", "")
        parent_ids = parent_map.get(block_id, [])

        # Use advanced ownership detection
        ownership = _find_owning_key_for_checkbox(
            block_id, parent_ids, blocks_by_id, value_block_to_key_block_id, text_cache
        )

        checkbox_entry = {
            "selection_element_id": block_id,
            "selection_status": block.get("SelectionStatus"),
            "is_selected": block.get("SelectionStatus") == "SELECTED",
            "confidence": block.get("Confidence"),
            "page": block.get("Page"),
            "geometry": _geometry_summary(block),
            "parent_block_ids": parent_ids,
            "ownership": ownership,
        }

        checkboxes.append(checkbox_entry)

    return checkboxes


def parse_textract_response(response: dict[str, Any]) -> dict[str, Any]:
    # Production hardening: tolerate malformed / unexpected Textract payloads
    # instead of crashing the pipeline on a bad document.
    if not isinstance(response, dict):
        logger.warning("[parser] response is not a dict (got %s); treating as empty", type(response).__name__)
        response = {}
    raw_blocks = response.get("Blocks")
    if not isinstance(raw_blocks, list):
        if raw_blocks is not None:
            logger.warning("[parser] 'Blocks' is not a list (got %s); coercing to empty", type(raw_blocks).__name__)
        raw_blocks = []
    blocks = [block for block in raw_blocks if isinstance(block, dict)]
    if len(blocks) != len(raw_blocks):
        logger.warning("[parser] dropped %d non-dict block(s) from response", len(raw_blocks) - len(blocks))
    blocks_by_id = {block.get("Id"): block for block in blocks if block.get("Id")}
    text_cache: dict[str, str] = {}
    parent_map = _build_parent_map(blocks)

    block_type_counts: dict[str, int] = defaultdict(int)
    for block in blocks:
        block_type_counts[str(block.get("BlockType") or "UNKNOWN")] += 1

    fields, field_items, value_block_to_key_block_id = _extract_key_value_items(blocks_by_id, text_cache)
    tables = _extract_tables(blocks_by_id, text_cache)
    checkboxes = _extract_checkboxes(blocks_by_id, parent_map, value_block_to_key_block_id, text_cache)

    # Build page-aware grouping
    pages: dict[int, dict[str, Any]] = defaultdict(lambda: {"fields": [], "tables": [], "checkboxes": [], "block_type_counts": {}})

    def _safe_page(v: Any) -> int:
        try:
            return int(v or 0)
        except Exception:
            return 0

    for fi in field_items:
        p = _safe_page(fi.get("page"))
        pages[p]["fields"].append(fi)

    for t in tables:
        p = _safe_page(t.get("page"))
        pages[p]["tables"].append(t)

    for c in checkboxes:
        p = _safe_page(c.get("page"))
        pages[p]["checkboxes"].append(c)

    # Per-page block type counts
    for block in blocks:
        p = _safe_page(block.get("Page"))
        bt = str(block.get("BlockType") or "UNKNOWN")
        pages[p]["block_type_counts"][bt] = pages[p]["block_type_counts"].get(bt, 0) + 1

    pages_list: list[dict[str, Any]] = []
    for page_num in sorted(p for p in pages.keys() if p >= 0):
        pages_list.append({"page": page_num, **pages[page_num]})

    # Confidence summaries
    def _avg(values: list[float]) -> float | None:
        vals: list[float] = []
        for v in values:
            if v is None:
                continue
            try:
                fv = float(v)
            except (TypeError, ValueError):
                continue
            if fv != fv:  # skip NaN
                continue
            vals.append(fv)
        if not vals:
            return None
        return sum(vals) / len(vals)

    key_confidences = [fi.get("confidence") for fi in field_items if fi.get("confidence") is not None]
    table_confidences = [t.get("confidence") for t in tables if t.get("confidence") is not None]
    checkbox_confidences = [c.get("confidence") for c in checkboxes if c.get("confidence") is not None]
    word_confidences = [b.get("Confidence") for b in blocks if b.get("BlockType") in ("WORD", "LINE") and b.get("Confidence") is not None]

    confidence_summary = {
        "key_values_avg": _avg(key_confidences),
        "tables_avg": _avg(table_confidences),
        "checkboxes_avg": _avg(checkbox_confidences),
        "words_avg": _avg(word_confidences),
        "counts": {
            "keys": len(field_items),
            "tables": len(tables),
            "checkboxes": len(checkboxes),
            "words": len(word_confidences),
        },
    }

    return {
        "fields": fields,
        "field_items": field_items,
        "checkboxes": checkboxes,
        "tables": tables,
        "pages": pages_list,
        "metadata": {
            "document_metadata": response.get("DocumentMetadata"),
            "job_status": response.get("JobStatus"),
            "detect_document_text_model_version": response.get("DetectDocumentTextModelVersion"),
            "analyze_document_model_version": response.get("AnalyzeDocumentModelVersion"),
            "block_type_counts": dict(sorted(block_type_counts.items())),
            "page_count": response.get("DocumentMetadata", {}).get("Pages") if isinstance(response.get("DocumentMetadata"), dict) else None,
            "response_keys": sorted(response.keys()),
        },
        "confidence_summary": confidence_summary,
    }


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def _summarize_for_cli(parsed: dict[str, Any]) -> dict[str, Any]:
    return {
        "field_count": len(parsed.get("fields", {})),
        "checkbox_count": len(parsed.get("checkboxes", [])),
        "table_count": len(parsed.get("tables", [])),
        "block_type_counts": parsed.get("metadata", {}).get("block_type_counts", {}),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse a raw AWS Textract AnalyzeDocument JSON response.")
    parser.add_argument("input", type=Path, help="Path to a raw Textract response JSON file")
    parser.add_argument("--output", type=Path, help="Optional path for the parsed JSON output")
    args = parser.parse_args()

    response = _load_json(args.input)
    parsed = parse_textract_response(response)

    if args.output:
        _write_json(args.output, parsed)

    print(json.dumps(_summarize_for_cli(parsed), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()