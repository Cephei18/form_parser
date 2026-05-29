from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import cv2

from src.pdf_generator import create_pdf_with_fields
from src.textract_service import load_json, save_json
from src.textract_service import parse_response_file

logger = logging.getLogger("form_parser.pipeline.textract")


def _image_size(image_path: Path) -> tuple[int, int]:
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"Unable to read image for Textract pipeline: {image_path}")
    height, width = image.shape[:2]
    return width, height


def _geometry_to_bbox_pixels(geometry: dict[str, Any] | None, width: int, height: int) -> list[list[float]] | None:
    if not geometry:
        return None

    bbox = geometry.get("bounding_box") or geometry.get("BoundingBox") or {}
    if not isinstance(bbox, dict):
        return None

    try:
        left = float(bbox.get("Left", 0.0)) * width
        top = float(bbox.get("Top", 0.0)) * height
        box_width = float(bbox.get("Width", 0.0)) * width
        box_height = float(bbox.get("Height", 0.0)) * height
    except (TypeError, ValueError):
        return None

    if box_width <= 0 or box_height <= 0:
        return None

    x1 = left
    y1 = top
    x2 = left + box_width
    y2 = top + box_height
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


def _bbox_to_field_box(bbox: list[list[float]] | None) -> list[dict[str, float]]:
    if not bbox or len(bbox) != 4:
        return []

    x_values = [float(point[0]) for point in bbox]
    y_values = [float(point[1]) for point in bbox]
    x1 = min(x_values)
    y1 = min(y_values)
    x2 = max(x_values)
    y2 = max(y_values)
    return [{"x": x1, "y": y1, "width": max(1.0, x2 - x1), "height": max(1.0, y2 - y1)}]


def _confidence_class(confidence: float | None) -> str:
    if confidence is None:
        return "unknown"
    if confidence >= 0.9:
        return "high"
    if confidence >= 0.7:
        return "medium"
    return "low"


def _normalize_text(value: Any) -> str:
    text = str(value or "").strip()
    return text


def _field_items_to_mappings(parsed: dict[str, Any], image_path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    width, height = _image_size(image_path)
    mappings: list[dict[str, Any]] = []
    result_items: list[dict[str, Any]] = []

    for index, field_item in enumerate(parsed.get("field_items", []) or [], start=1):
        geometry = field_item.get("geometry") or {}
        bbox = _geometry_to_bbox_pixels(geometry, width, height)
        if bbox is None:
            continue

        label = _normalize_text(field_item.get("key")) or f"field_{index}"
        value = _normalize_text(field_item.get("value"))
        confidence = field_item.get("confidence") if isinstance(field_item.get("confidence"), (int, float)) else None
        mapping = {
            "label": label,
            "value": value,
            "field_type": "text",
            "field_bboxes": _bbox_to_field_box(bbox),
            "candidate_score": float(confidence or 0.0),
            "confidence_class": _confidence_class(confidence),
            "multiline_group_size": max(1, value.count("\n") + 1 if value else 1),
            "page": field_item.get("page"),
            "source": "textract",
        }
        mappings.append(mapping)
        result_items.append(
            {
                "text": value or label,
                "bbox": bbox,
                "center": [
                    (bbox[0][0] + bbox[2][0]) / 2.0,
                    (bbox[0][1] + bbox[2][1]) / 2.0,
                ],
                "confidence": confidence,
                "source_item_count": 1,
            }
        )

    for index, checkbox in enumerate(parsed.get("checkboxes", []) or [], start=1):
        geometry = checkbox.get("geometry") or {}
        bbox = _geometry_to_bbox_pixels(geometry, width, height)
        if bbox is None:
            continue

        ownership = checkbox.get("ownership") or {}
        label = _normalize_text(ownership.get("key_text")) or _normalize_text(ownership.get("key_block_id")) or f"checkbox_{index}"
        is_selected = bool(checkbox.get("is_selected"))
        confidence = checkbox.get("confidence") if isinstance(checkbox.get("confidence"), (int, float)) else None
        mappings.append(
            {
                "label": label,
                "value": "[X]" if is_selected else "[ ]",
                "field_type": "checkbox",
                "field_bboxes": _bbox_to_field_box(bbox),
                "candidate_score": float(confidence or 0.0),
                "confidence_class": _confidence_class(confidence),
                "multiline_group_size": 1,
                "page": checkbox.get("page"),
                "source": "textract",
            }
        )
        result_items.append(
            {
                "text": label,
                "bbox": bbox,
                "center": [
                    (bbox[0][0] + bbox[2][0]) / 2.0,
                    (bbox[0][1] + bbox[2][1]) / 2.0,
                ],
                "confidence": confidence,
                "source_item_count": 1,
            }
        )

    return mappings, result_items


def _draw_mapping_preview(image_path: Path, mappings: list[dict[str, Any]], output_path: Path) -> None:
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"Unable to read source image for preview: {image_path}")

    for index, mapping in enumerate(mappings, start=1):
        color = (0, 180, 255) if mapping.get("field_type") == "checkbox" else (80, 180, 80)
        for box in mapping.get("field_bboxes", []) or []:
            try:
                x = int(float(box.get("x", 0)))
                y = int(float(box.get("y", 0)))
                width = int(float(box.get("width", 0)))
                height = int(float(box.get("height", 0)))
            except (TypeError, ValueError):
                continue
            cv2.rectangle(image, (x, y), (x + width, y + height), color, 2)
            label = str(mapping.get("label", f"field_{index}"))[:60]
            cv2.putText(image, label, (x, max(12, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), image):
        raise RuntimeError(f"Failed to write mapping preview: {output_path}")


def _analyze_with_textract(image_path: Path) -> dict[str, Any]:
    try:
        import boto3
    except ModuleNotFoundError as exc:
        raise RuntimeError("boto3 is required for Textract pipeline mode") from exc

    logger.info("[pipeline] Calling AWS Textract AnalyzeDocument")
    textract = boto3.client("textract")
    image_bytes = image_path.read_bytes()
    return textract.analyze_document(Document={"Bytes": image_bytes}, FeatureTypes=["TABLES", "FORMS"])


def run_textract_pipeline(file_path: str | Path, output_dir: str | Path, reference_image_path: str | Path | None = None) -> dict[str, Any]:
    """Run the isolated Textract pipeline and emit API-compatible artifacts."""
    source_path = Path(file_path)
    destination_dir = Path(output_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)
    image_path = Path(reference_image_path) if reference_image_path else source_path

    logger.info("[pipeline] Running Textract pipeline")
    logger.info("[pipeline] ACTIVE PIPELINE: TEXTRACT")
    started = cv2.getTickCount()

    if source_path.suffix.lower() == ".json":
        raw_response = load_json(source_path)
    else:
        raw_response = _analyze_with_textract(source_path)

    raw_response_path = destination_dir / "textract_raw_response.json"
    save_json(raw_response_path, raw_response)

    parsed = parse_response_file(raw_response_path, destination_dir / "textract_parsed.json")
    mappings, result_items = _field_items_to_mappings(parsed, image_path)

    mappings_path = destination_dir / "mappings.json"
    result_path = destination_dir / "result.json"
    mapping_image_path = destination_dir / "mapping.png"
    pdf_output_path = destination_dir / "output.pdf"

    save_json(mappings_path, mappings)
    save_json(result_path, result_items)
    _draw_mapping_preview(image_path, mappings, mapping_image_path)
    create_pdf_with_fields(image_path, mappings, pdf_output_path)

    diagnostics = {
        "pipeline_mode": "textract",
        "raw_response_path": str(raw_response_path),
        "parsed_response_path": str(destination_dir / "textract_parsed.json"),
        "field_count": len(parsed.get("field_items", []) or []),
        "checkbox_count": len(parsed.get("checkboxes", []) or []),
        "table_count": len(parsed.get("tables", []) or []),
        "confidence_summary": parsed.get("confidence_summary", {}),
    }
    save_json(destination_dir / "mapping_diagnostics.json", diagnostics)
    save_json(destination_dir / "benchmark_summary.json", diagnostics)

    elapsed_ms = ((cv2.getTickCount() - started) / cv2.getTickFrequency()) * 1000.0
    logger.info(
        "[pipeline] Textract pipeline completed in %.1fms tables=%s checkboxes=%s",
        elapsed_ms,
        len(parsed.get("tables", []) or []),
        len(parsed.get("checkboxes", []) or []),
    )

    return {
        "pipeline_mode": "textract",
        "processing_time_ms": round(elapsed_ms, 2),
        "tables_detected": len(parsed.get("tables", []) or []),
        "checkboxes_detected": len(parsed.get("checkboxes", []) or []),
        "raw_response_path": raw_response_path,
        "parsed_response_path": destination_dir / "textract_parsed.json",
        "result_path": result_path,
        "mappings_path": mappings_path,
        "mapping_image_path": mapping_image_path,
        "pdf_output_path": pdf_output_path,
        "mappings": mappings,
        "lines_count": len(result_items),
        "filtered_lines_count": len(mappings),
        "confidence_summary": parsed.get("confidence_summary", {}),
        "metadata": parsed.get("metadata", {}),
        "pages": parsed.get("pages", []),
        "fields": parsed.get("fields", {}),
        "field_items": parsed.get("field_items", []),
        "checkboxes": parsed.get("checkboxes", []),
        "tables": parsed.get("tables", []),
        "parsed_output": parsed,
        "diagnostics_path": destination_dir / "mapping_diagnostics.json",
    }


run_pipeline = run_textract_pipeline
