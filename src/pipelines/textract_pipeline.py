from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import cv2

from src.artifact_store import get_artifact_store
from src.field_anchor_engine import build_anchored_mappings, draw_anchor_debug_overlay
from src.pdf_generator import create_pdf_with_fields
from src.section_detector import build_hierarchy
from src.pipeline_config import StorageConfig
from src.textract_service import load_json, save_json
from src.textract_service import parse_response_file
from src.textract_validators import build_validation_report

logger = logging.getLogger("form_parser.pipeline.textract")


def _preview_labels_enabled() -> bool:
    """Whether to draw per-field text labels on the mapping.png preview.

    Default True preserves the existing (validated) preview exactly. Set
    FORM_PARSER_PREVIEW_LABELS=false for a clean, label-free preview (e.g. demos)
    — boxes only. This affects ONLY the diagnostic preview image; the generated
    output.pdf (form widgets) and the debug overlay are untouched.
    """
    return os.getenv("FORM_PARSER_PREVIEW_LABELS", "true").strip().lower() not in {"0", "false", "no", "off"}


def _draw_mapping_preview(image_path: Path, mappings: list[dict[str, Any]], output_path: Path) -> None:
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"Unable to read source image for preview: {image_path}")

    draw_labels = _preview_labels_enabled()
    for index, mapping in enumerate(mappings, start=1):
        # The preview background is the page-1 raster; only page-1 field boxes are
        # in this coordinate space. Multi-page boxes are shown in the per-page PDF.
        if int(mapping.get("page") or 1) != 1:
            continue
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
            if draw_labels:
                label = str(mapping.get("label", f"field_{index}"))[:60]
                cv2.putText(image, label, (x, max(12, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), image):
        raise RuntimeError(f"Failed to write mapping preview: {output_path}")


def _validate_before_emit(parsed: dict[str, Any], mappings: list[dict[str, Any]]) -> dict[str, Any]:
    """Run the validation suite defensively and log any sanity warnings.

    Always returns a report dict; never raises, so a validator bug can never
    block PDF rendering in production.
    """
    try:
        report = build_validation_report(parsed, mappings)
    except Exception:  # pragma: no cover - last-resort guard
        logger.exception("[pipeline] validation report generation failed; continuing")
        return {"error": "validation_failed", "passed": False}

    sanity = report.get("mapping_sanity") if isinstance(report, dict) else None
    if isinstance(sanity, dict):
        if not sanity.get("passed"):
            logger.warning(
                "[pipeline] validation sanity check did not pass: fillable=%s nan_boxes=%s zero_size=%s issues=%s",
                sanity.get("fillable_count"),
                sanity.get("nan_boxes"),
                sanity.get("zero_size_boxes"),
                sanity.get("issues"),
            )
        if sanity.get("out_of_range_boxes"):
            logger.warning("[pipeline] %s mapping box(es) fall outside the page bounds", sanity.get("out_of_range_boxes"))
    return report


def _analyze_with_textract(image_path: Path) -> dict[str, Any]:
    try:
        import boto3
    except ModuleNotFoundError as exc:
        raise RuntimeError("boto3 is required for Textract pipeline mode") from exc

    logger.info("[pipeline] Calling AWS Textract AnalyzeDocument")
    textract = boto3.client("textract")
    image_bytes = image_path.read_bytes()
    return textract.analyze_document(Document={"Bytes": image_bytes}, FeatureTypes=["TABLES", "FORMS"])


def _multipage_async_enabled() -> bool:
    """Opt-in flag for the native async Textract path.

    Defaults OFF: the async StartDocumentAnalysis flow requires the worker role
    to grant ``textract:StartDocumentAnalysis``/``GetDocumentAnalysis`` (an
    admin-gated IAM change, see infra/worker_role_textract_async_policy.json).
    Until that policy is live, multi-page extraction uses the per-page sync
    merge, which needs no new permissions and is therefore rollback-safe.
    """
    return os.getenv("FORM_PARSER_TEXTRACT_ASYNC", "false").strip().lower() in {"1", "true", "yes", "on"}


def _merge_page_responses(page_responses: list[tuple[int, dict[str, Any]]]) -> dict[str, Any]:
    """Merge N single-page Textract responses into one multi-page-shaped response.

    Every block is stamped with its 1-based ``Page`` so the already-page-aware
    parser and anchor engine treat the merged result exactly like a native async
    multi-page job. Block ``Id``s are response-local UUIDs, so cross-page
    collisions cannot occur and intra-page Relationships remain valid.
    """
    blocks: list[dict[str, Any]] = []
    model_version: Any = None
    for page_number, response in page_responses:
        if not isinstance(response, dict):
            continue
        model_version = model_version or response.get("AnalyzeDocumentModelVersion")
        for block in response.get("Blocks", []) or []:
            if not isinstance(block, dict):
                continue
            stamped = dict(block)
            stamped["Page"] = page_number
            blocks.append(stamped)

    merged: dict[str, Any] = {
        "DocumentMetadata": {"Pages": len(page_responses)},
        "Blocks": blocks,
        "JobStatus": "SUCCEEDED",
    }
    if model_version is not None:
        merged["AnalyzeDocumentModelVersion"] = model_version
    return merged


def _analyze_pages(
    page_images: list[tuple[int, Path]],
    document_location: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Run multi-page Textract extraction and return a sync-compatible response.

    Default path (infra-stable, no new IAM): synchronous ``AnalyzeDocument``
    once per rasterised page, merged with ``_merge_page_responses``.

    Opt-in async path (``FORM_PARSER_TEXTRACT_ASYNC=1`` + an S3
    ``document_location``): the native ``StartDocumentAnalysis`` flow already
    built and unit-tested in ``src.textract_analysis``.
    """
    if document_location and _multipage_async_enabled():
        from src.textract_analysis import analyze_document_async

        logger.info("[pipeline] multi-page extraction via async StartDocumentAnalysis")
        return analyze_document_async(document_location["bucket"], document_location["key"])

    logger.info("[pipeline] multi-page extraction via per-page sync AnalyzeDocument pages=%s", len(page_images))
    page_responses = [(page_number, _analyze_with_textract(Path(image))) for page_number, image in page_images]
    return _merge_page_responses(page_responses)


def run_textract_pipeline(
    file_path: str | Path,
    output_dir: str | Path,
    reference_image_path: str | Path | None = None,
    *,
    page_images: list[tuple[int, str | Path]] | None = None,
    document_location: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Run the isolated Textract pipeline and emit API-compatible artifacts.

    ``page_images`` is an ordered ``[(page_no, image_path), ...]`` list of the
    rasterised pages of the source document. When supplied (multi-page mode) the
    pipeline analyses, maps and renders all pages; when omitted it behaves
    exactly as before (single page-1 image). ``document_location`` carries the
    S3 ``{bucket, key}`` for the optional native async Textract path.
    """
    source_path = Path(file_path)
    destination_dir = Path(output_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)
    image_path = Path(reference_image_path) if reference_image_path else source_path

    # Normalise the per-page raster set: [(page_no, path), ...] -> {page_no: Path}.
    page_image_pairs: list[tuple[int, Path]] = [(int(p), Path(img)) for p, img in (page_images or [])]
    page_image_map: dict[int, Path] = dict(page_image_pairs)
    multipage = len(page_image_map) > 0

    logger.info("[pipeline] Running Textract pipeline")
    logger.info("[pipeline] ACTIVE PIPELINE: TEXTRACT")
    if multipage:
        logger.info("[pipeline] multi-page mode enabled pages=%s", sorted(page_image_map))
    started = cv2.getTickCount()

    if source_path.suffix.lower() == ".json":
        raw_response = load_json(source_path)
    elif multipage:
        raw_response = _analyze_pages(page_image_pairs, document_location)
    else:
        raw_response = _analyze_with_textract(source_path)

    raw_response_path = destination_dir / "textract_raw_response.json"
    save_json(raw_response_path, raw_response)

    parsed = parse_response_file(raw_response_path, destination_dir / "textract_parsed.json")

    anchor_output = build_anchored_mappings(raw_response, parsed, image_path, page_images=page_image_map or None)
    mappings = anchor_output["mappings"]
    field_objects = anchor_output["field_objects"]
    anchor_diagnostics = anchor_output["diagnostics"]
    visual_features = anchor_output["visual_features"]
    sections = anchor_output.get("sections", [])

    mappings_path = destination_dir / "mappings.json"
    result_path = destination_dir / "result.json"
    mapping_image_path = destination_dir / "mapping.png"
    debug_image_path = destination_dir / "textract_mapping_debug.png"
    anchoring_metadata_path = destination_dir / "anchoring_metadata.json"
    validation_report_path = destination_dir / "validation_report.json"
    hierarchy_path = destination_dir / "hierarchy.json"
    pdf_output_path = destination_dir / "output.pdf"

    # Production hardening: validate parsed output and final mappings before
    # emitting the PDF. Non-fatal — a validator failure must never block
    # rendering — but surfaces malformed geometry / empty-result documents.
    validation_report = _validate_before_emit(parsed, mappings)

    # Document -> Section -> Field hierarchy artifact (Phase 2.2). Explainable,
    # debug-friendly tree of which fields fell under which detected section.
    hierarchy = build_hierarchy(sections, mappings)
    section_hierarchy_diag = anchor_diagnostics.get("hierarchy", {}) if isinstance(anchor_diagnostics, dict) else {}
    logger.info(
        "[pipeline] hierarchy sections=%s orphan_fields=%s types=%s",
        section_hierarchy_diag.get("section_count"),
        section_hierarchy_diag.get("orphan_field_count"),
        section_hierarchy_diag.get("section_types_detected"),
    )

    save_json(mappings_path, mappings)
    save_json(result_path, mappings)
    save_json(anchoring_metadata_path, anchor_diagnostics)
    save_json(validation_report_path, validation_report)
    save_json(hierarchy_path, hierarchy)
    draw_anchor_debug_overlay(image_path, mappings, debug_image_path, visual_features=visual_features)
    _draw_mapping_preview(image_path, mappings, mapping_image_path)
    create_pdf_with_fields(image_path, mappings, pdf_output_path, page_images=page_image_map or None)

    elapsed_ms = ((cv2.getTickCount() - started) / cv2.getTickFrequency()) * 1000.0

    # --- Multi-page observability ---------------------------------------------
    doc_metadata = parsed.get("metadata", {}) or {}
    pages_detected = doc_metadata.get("page_count")
    pages_analyzed = sorted(
        {int(block.get("Page") or 1) for block in (raw_response.get("Blocks") or []) if isinstance(block, dict)}
    ) or [1]
    pages_rendered = sorted(set(int(m.get("page") or 1) for m in mappings) | set(page_image_map))
    fields_per_page: dict[int, int] = {}
    for mapping in mappings:
        page = int(mapping.get("page") or 1)
        fields_per_page[page] = fields_per_page.get(page, 0) + 1

    missing_page_warnings: list[str] = []
    if multipage:
        analyzed_set = set(pages_analyzed)
        for page in sorted(page_image_map):
            if page not in analyzed_set:
                missing_page_warnings.append(f"page_{page}_rasterized_but_no_textract_blocks")
        if pages_detected and len(page_image_map) != int(pages_detected):
            missing_page_warnings.append(
                f"rasterized_{len(page_image_map)}_pages_but_textract_reported_{pages_detected}"
            )
        if pages_rendered and len(pages_rendered) != len(page_image_map):
            missing_page_warnings.append(
                f"rendered_{len(pages_rendered)}_pages_but_rasterized_{len(page_image_map)}"
            )
    if missing_page_warnings:
        logger.warning("[pipeline] multi-page warnings: %s", missing_page_warnings)

    page_observability = {
        "multipage_mode": multipage,
        "pages_detected": pages_detected,
        "pages_rasterized": sorted(page_image_map),
        "pages_analyzed": pages_analyzed,
        "pages_rendered": pages_rendered,
        "fields_per_page": {str(page): count for page, count in sorted(fields_per_page.items())},
        "missing_page_warnings": missing_page_warnings,
    }
    logger.info(
        "[pipeline] page summary multipage=%s detected=%s rasterized=%s analyzed=%s rendered=%s",
        multipage,
        pages_detected,
        len(page_image_map),
        len(pages_analyzed),
        len(pages_rendered),
    )

    diagnostics = {
        "pipeline_mode": "textract",
        "page_observability": page_observability,
        "hierarchy": section_hierarchy_diag,
        "hierarchy_path": str(hierarchy_path),
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
        "validation": validation_report,
        "validation_report_path": str(validation_report_path),
    }
    save_json(destination_dir / "mapping_diagnostics.json", diagnostics)
    save_json(destination_dir / "benchmark_summary.json", diagnostics)

    # Serverless-migration step 1: publish the finished artifacts through the
    # storage abstraction. Default backend is "local" (no-op); when configured
    # for S3 this mirrors every run artifact to the processed-documents bucket,
    # making the pipeline Lambda-compatible without changing local behavior.
    storage_config = StorageConfig.from_env()
    artifact_store = get_artifact_store(storage_config)
    job_id = destination_dir.name or "textract-job"
    artifact_manifest = artifact_store.publish(destination_dir, job_id=job_id)
    logger.info(
        "[pipeline] artifacts published backend=%s job_id=%s count=%s base=%s",
        artifact_manifest.get("backend"),
        job_id,
        artifact_manifest.get("artifact_count"),
        artifact_manifest.get("base_uri"),
    )

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
        "page_observability": page_observability,
        "hierarchy": hierarchy,
        "hierarchy_path": hierarchy_path,
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
        "artifact_store": artifact_manifest,
    }


run_pipeline = run_textract_pipeline
