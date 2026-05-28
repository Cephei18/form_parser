from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


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
        child_texts = []
        for child_id in _relationship_ids(block, "CHILD"):
            child_text = _block_text(child_id, blocks_by_id, cache)
            if child_text:
                child_texts.append(child_text)
        text = " ".join(child_texts).strip()

    cache[block_id] = text
    return text


def _geometry_summary(block: dict[str, Any]) -> dict[str, Any] | None:
    geometry = block.get("Geometry")
    if not geometry:
        return None

    return {
        "bounding_box": geometry.get("BoundingBox"),
        "polygon": geometry.get("Polygon"),
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
        value_texts = []

        for value_id in value_ids:
            value_block_to_key_block_id[value_id] = block.get("Id", "")
            value_texts.append(_block_text(value_id, blocks_by_id, text_cache))

        value_text = " ".join(text for text in value_texts if text).strip()
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

            row_index = int(cell_block.get("RowIndex") or 0)
            column_index = int(cell_block.get("ColumnIndex") or 0)
            max_row = max(max_row, row_index)
            max_col = max(max_col, column_index)
            cells.append(
                {
                    "cell_id": cell_id,
                    "row_index": row_index,
                    "column_index": column_index,
                    "row_span": int(cell_block.get("RowSpan") or 1),
                    "column_span": int(cell_block.get("ColumnSpan") or 1),
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

        tables.append(
            {
                "table_block_id": block.get("Id"),
                "page": block.get("Page"),
                "confidence": block.get("Confidence"),
                "geometry": _geometry_summary(block),
                "dimensions": {"rows": max_row, "columns": max_col},
                "rows": rows,
                "cells": sorted(cells, key=lambda cell: (cell["row_index"], cell["column_index"])),
            }
        )

    return tables


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
        associated_fields: list[str] = []
        associated_key_block_id: str | None = None

        for parent_id in parent_ids:
            key_block_id = value_block_to_key_block_id.get(parent_id)
            if key_block_id:
                associated_key_block_id = key_block_id
                key_block = blocks_by_id.get(key_block_id)
                if key_block:
                    key_text = _block_text(key_block_id, blocks_by_id, text_cache)
                    if key_text:
                        associated_fields.append(key_text)
            parent_block = blocks_by_id.get(parent_id)
            if parent_block and parent_block.get("BlockType") == "CELL":
                cell_text = _block_text(parent_id, blocks_by_id, text_cache)
                if cell_text:
                    associated_fields.append(cell_text)

        checkbox_entry = {
            "selection_element_id": block_id,
            "selection_status": block.get("SelectionStatus"),
            "is_selected": block.get("SelectionStatus") == "SELECTED",
            "confidence": block.get("Confidence"),
            "page": block.get("Page"),
            "geometry": _geometry_summary(block),
            "parent_block_ids": parent_ids,
            "associated_fields": list(dict.fromkeys(associated_fields)),
        }

        if associated_key_block_id:
            checkbox_entry["associated_key_block_id"] = associated_key_block_id
            checkbox_entry["associated_key"] = _block_text(associated_key_block_id, blocks_by_id, text_cache)

        checkboxes.append(checkbox_entry)

    return checkboxes


def parse_textract_response(response: dict[str, Any]) -> dict[str, Any]:
    blocks = response.get("Blocks", []) or []
    blocks_by_id = {block.get("Id"): block for block in blocks if block.get("Id")}
    text_cache: dict[str, str] = {}
    parent_map = _build_parent_map(blocks)

    block_type_counts: dict[str, int] = defaultdict(int)
    for block in blocks:
        block_type_counts[str(block.get("BlockType") or "UNKNOWN")] += 1

    fields, field_items, value_block_to_key_block_id = _extract_key_value_items(blocks_by_id, text_cache)
    tables = _extract_tables(blocks_by_id, text_cache)
    checkboxes = _extract_checkboxes(blocks_by_id, parent_map, value_block_to_key_block_id, text_cache)

    return {
        "fields": fields,
        "field_items": field_items,
        "checkboxes": checkboxes,
        "tables": tables,
        "metadata": {
            "document_metadata": response.get("DocumentMetadata"),
            "job_status": response.get("JobStatus"),
            "detect_document_text_model_version": response.get("DetectDocumentTextModelVersion"),
            "analyze_document_model_version": response.get("AnalyzeDocumentModelVersion"),
            "block_type_counts": dict(sorted(block_type_counts.items())),
            "page_count": response.get("DocumentMetadata", {}).get("Pages") if isinstance(response.get("DocumentMetadata"), dict) else None,
            "response_keys": sorted(response.keys()),
        },
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