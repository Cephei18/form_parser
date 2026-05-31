from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import cv2

from src.field_anchor_engine import build_anchored_mappings, draw_anchor_debug_overlay
from src.pdf_generator import create_pdf_with_fields
from src.textract_service import load_json, save_json
from src.textract_service import parse_response_file

logger = logging.getLogger("form_parser.pipeline.textract")


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

    anchor_output = build_anchored_mappings(raw_response, parsed, image_path)
    mappings = anchor_output["mappings"]
    field_objects = anchor_output["field_objects"]
    anchor_diagnostics = anchor_output["diagnostics"]
    visual_features = anchor_output["visual_features"]

    mappings_path = destination_dir / "mappings.json"
    result_path = destination_dir / "result.json"
    mapping_image_path = destination_dir / "mapping.png"
    debug_image_path = destination_dir / "textract_mapping_debug.png"
    anchoring_metadata_path = destination_dir / "anchoring_metadata.json"
    pdf_output_path = destination_dir / "output.pdf"

    save_json(mappings_path, mappings)
    save_json(result_path, mappings)
    save_json(anchoring_metadata_path, anchor_diagnostics)
    draw_anchor_debug_overlay(image_path, mappings, debug_image_path, visual_features=visual_features)
    _draw_mapping_preview(image_path, mappings, mapping_image_path)
    create_pdf_with_fields(image_path, mappings, pdf_output_path)

    elapsed_ms = ((cv2.getTickCount() - started) / cv2.getTickFrequency()) * 1000.0
    diagnostics = {
        "pipeline_mode": "textract",
        "processing_time_ms": round(elapsed_ms, 2),
        "raw_response_path": str(raw_response_path),
        "parsed_response_path": str(destination_dir / "textract_parsed.json"),
        "fields_detected": len([field for field in mappings if field.get("field_type") != "photo"]),
        "photos_detected": len([field for field in mappings if field.get("field_type") == "photo"]),
        "checkboxes_detected": len([field for field in mappings if field.get("field_type") == "checkbox"]),
        "tables_detected": len(parsed.get("tables", []) or []),
        "label_count": len(parsed.get("field_items", []) or []),
        "answer_region_count": len(mappings),
        "confidence_summary": parsed.get("confidence_summary", {}),
        "debug_image_path": str(debug_image_path),
        "anchoring_metadata_path": str(anchoring_metadata_path),
        "anchoring": anchor_diagnostics,
        "field_objects": field_objects,
    }
    save_json(destination_dir / "mapping_diagnostics.json", diagnostics)
    save_json(destination_dir / "benchmark_summary.json", diagnostics)

    logger.info(
        "[pipeline] Textract pipeline completed in %.1fms labels=%s anchored=%s fields=%s photos=%s checkboxes=%s",
        elapsed_ms,
        diagnostics["label_count"],
        diagnostics["answer_region_count"],
        diagnostics["fields_detected"],
        diagnostics["photos_detected"],
        diagnostics["checkboxes_detected"],
    )

    return {
        "pipeline_mode": "textract",
        "processing_time_ms": round(elapsed_ms, 2),
        "tables_detected": diagnostics["tables_detected"],
        "checkboxes_detected": diagnostics["checkboxes_detected"],
        "raw_response_path": raw_response_path,
        "parsed_response_path": destination_dir / "textract_parsed.json",
        "result_path": result_path,
        "mappings_path": mappings_path,
        "mapping_image_path": mapping_image_path,
        "pdf_output_path": pdf_output_path,
        "mappings": mappings,
        "lines_count": diagnostics["label_count"] + diagnostics["answer_region_count"],
        "filtered_lines_count": len(mappings),
        "confidence_summary": parsed.get("confidence_summary", {}),
        "metadata": parsed.get("metadata", {}),
        "pages": parsed.get("pages", []),
        "fields": {field["label"]: field["bbox"] for field in mappings},
        "field_items": field_objects,
        "checkboxes": [field for field in mappings if field.get("field_type") == "checkbox"],
        "tables": parsed.get("tables", []),
        "parsed_output": parsed,
        "diagnostics_path": destination_dir / "mapping_diagnostics.json",
        "debug_image_path": debug_image_path,
    }


run_pipeline = run_textract_pipeline
